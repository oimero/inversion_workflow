"""Training and evaluation for the band-limited evidence model."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import logging
import random
from typing import Any, Mapping

import numpy as np
import torch

from evidence.network import EvidenceModel, evidence_loss


CheckpointCallback = Callable[[int, EvidenceModel, Mapping[str, Any], bool], None]

_TENSOR_BATCH_KEYS = frozenset(
    {
        "seismic",
        "lfm_residual",
        "observed_valid",
        "lateral_m",
        "lateral_valid",
        "projected_log_ai_increment",
        "signed_reflectivity",
        "support",
    }
)


@dataclass(frozen=True)
class EvidenceLearningConfig:
    epochs: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    log_every_batches: int = 20
    gradient_clip_norm: float = 5.0
    increment_weight: float = 1.0
    reflectivity_weight: float = 0.5
    scale_weight: float = 0.1
    random_seed: int = 20260808

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.log_every_batches <= 0:
            raise ValueError("epochs and log_every_batches must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer controls are invalid.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        for name in (
            "increment_weight",
            "reflectivity_weight",
            "scale_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.random_seed, int) or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32).")


def seed_random_streams(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(
    batch: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    boolean = {"observed_valid", "lateral_valid", "support"}
    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key not in _TENSOR_BATCH_KEYS:
            continue
        result[key] = torch.as_tensor(
            value,
            dtype=torch.bool if key in boolean else torch.float32,
            device=device,
        )
    return result


def _forward_loss(
    model: EvidenceModel,
    batch: Mapping[str, torch.Tensor],
    config: EvidenceLearningConfig,
) -> dict[str, torch.Tensor]:
    output = model.network(
        batch["seismic"],
        batch["lfm_residual"],
        batch["observed_valid"],
        batch["lateral_m"],
        batch["lateral_valid"],
    )
    return evidence_loss(
        output,
        batch,
        config=model.network_config,
        increment_weight=config.increment_weight,
        reflectivity_weight=config.reflectivity_weight,
        scale_weight=config.scale_weight,
    )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _evaluate_losses(
    model: EvidenceModel,
    batches: Iterable[Mapping[str, object]],
    config: EvidenceLearningConfig,
) -> dict[str, float]:
    model.network.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for source in batches:
            loss = _forward_loss(model, _to_device(source, model.device), config)
            count += 1
            for key, value in loss.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    if count == 0:
        raise ValueError("evaluation received no batches.")
    return {key: value / count for key, value in totals.items()}


def train_evidence_model(
    model: EvidenceModel,
    training_batches: (
        Iterable[Mapping[str, object]]
        | Callable[[], Iterable[Mapping[str, object]]]
    ),
    tuning_batches: (
        Iterable[Mapping[str, object]]
        | Callable[[], Iterable[Mapping[str, object]]]
    ),
    *,
    config: EvidenceLearningConfig = EvidenceLearningConfig(),
    logger: logging.Logger | None = None,
    resume_state: Mapping[str, Any] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> dict[str, object]:
    """Train with durable epoch-boundary recovery hooks."""

    log = logger or logging.getLogger(__name__)
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resume = dict(resume_state or {})
    start_epoch = int(resume.get("epoch", 0)) + 1
    if start_epoch < 1:
        raise ValueError("resume epoch must be non-negative.")
    if resume_state is not None and "optimizer_state" not in resume:
        raise ValueError("resume checkpoint lacks optimizer_state.")
    if "optimizer_state" in resume:
        optimizer.load_state_dict(resume["optimizer_state"])
    if "rng_state" in resume:
        _restore_rng_state(resume["rng_state"])

    def batches(source):
        return source() if callable(source) else source

    history = [
        {str(key): float(value) for key, value in dict(row).items()}
        for row in resume.get("history", [])
    ]
    best_loss = float(resume.get("best_tuning_loss", float("inf")))
    best_epoch = int(resume.get("best_epoch", 0))
    if start_epoch > config.epochs:
        return {
            "best_tuning_loss": best_loss,
            "best_epoch": best_epoch,
            "history": history,
            "epochs_completed": start_epoch - 1,
            "start_epoch": start_epoch,
        }

    running_keys = (
        "loss",
        "increment_mean_huber",
        "increment_scale_huber",
        "reflectivity_mean_huber",
        "reflectivity_scale_huber",
    )
    for epoch in range(start_epoch, config.epochs + 1):
        model.network.train()
        running = {key: 0.0 for key in running_keys}
        peak_loss = float("-inf")
        peak_batch = 0
        batch_count = 0
        for batch_index, source in enumerate(batches(training_batches), start=1):
            batch_count = batch_index
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_loss(model, _to_device(source, model.device), config)
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.network.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            current = {key: float(loss[key].detach().cpu()) for key in running}
            for key, value in current.items():
                running[key] += value
            if current["loss"] > peak_loss:
                peak_loss = current["loss"]
                peak_batch = batch_index
            if batch_index % config.log_every_batches == 0:
                log.info(
                    "epoch %d/%d | batch %d | current_loss=%.6f | mean_loss=%.6f | "
                    "increment=%.6f | reflectivity=%.6f",
                    epoch,
                    config.epochs,
                    batch_index,
                    current["loss"],
                    running["loss"] / batch_index,
                    current["increment_mean_huber"],
                    current["reflectivity_mean_huber"],
                )
        if batch_count == 0:
            raise ValueError("training received no batches.")
        tuning = _evaluate_losses(model, batches(tuning_batches), config)
        row = {
            "epoch": float(epoch),
            **{f"train_{key}": value / batch_count for key, value in running.items()},
            "train_peak_loss": peak_loss,
            "train_peak_batch": float(peak_batch),
            **{f"tuning_{key}": value for key, value in tuning.items()},
        }
        history.append(row)
        log.info(
            "epoch %d/%d complete | train_loss=%.6f | tuning_loss=%.6f | "
            "peak_loss=%.6f@%d",
            epoch,
            config.epochs,
            row["train_loss"],
            row["tuning_loss"],
            peak_loss,
            peak_batch,
        )
        is_best = tuning["loss"] < best_loss
        if is_best:
            best_loss = tuning["loss"]
            best_epoch = epoch
        runtime_state = {
            "epoch": epoch,
            "best_tuning_loss": float(best_loss),
            "best_epoch": best_epoch,
            "history": [dict(item) for item in history],
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
        }
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, model, runtime_state, is_best)
    return {
        "best_tuning_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "epochs_completed": config.epochs,
        "start_epoch": start_epoch,
    }


def _new_statistics() -> dict[str, Any]:
    return {
        "count": 0,
        "increment_squared_error": 0.0,
        "increment_absolute_error": 0.0,
        "increment_scale": 0.0,
        "increment_coverage_50": 0,
        "increment_coverage_80": 0,
        "increment_coverage_95": 0,
        "increment_pair": np.zeros(5, dtype=np.float64),
        "reflectivity_squared_error": 0.0,
        "reflectivity_absolute_error": 0.0,
        "reflectivity_scale": 0.0,
        "reflectivity_coverage_50": 0,
        "reflectivity_coverage_80": 0,
        "reflectivity_coverage_95": 0,
        "reflectivity_pair": np.zeros(5, dtype=np.float64),
        "reflectivity_polarity_count": 0,
        "reflectivity_polarity_correct": 0,
    }


def _new_increment_statistics() -> dict[str, Any]:
    return {
        "count": 0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "pair": np.zeros(5, dtype=np.float64),
    }


def _update_pair(
    moments: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    moments += np.asarray(
        (
            np.sum(prediction),
            np.sum(target),
            np.sum(prediction * prediction),
            np.sum(target * target),
            np.sum(prediction * target),
        ),
        dtype=np.float64,
    )


def _correlation(moments: np.ndarray, count: int) -> float | None:
    if count < 2:
        return None
    sum_x, sum_y, sum_x2, sum_y2, sum_xy = moments
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_x2 - sum_x * sum_x / count
    variance_y = sum_y2 - sum_y * sum_y / count
    denominator = float(np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0)))
    return None if denominator <= 0.0 else float(covariance / denominator)


def _update_statistics(
    statistics: dict[str, Any],
    output: Mapping[str, np.ndarray],
    source: Mapping[str, object],
) -> None:
    support = np.asarray(source["support"], dtype=bool) & np.asarray(
        output["support"], dtype=bool
    )
    count = int(np.count_nonzero(support))
    if count == 0:
        raise ValueError("evidence evaluation has no supported samples.")
    target_increment = np.asarray(source["projected_log_ai_increment"], dtype=np.float64)[support]
    prediction_increment = np.asarray(output["projected_log_ai_increment_mean"], dtype=np.float64)[support]
    increment_scale = np.asarray(output["projected_log_ai_increment_scale"], dtype=np.float64)[support]
    residual = prediction_increment - target_increment
    statistics["count"] += count
    statistics["increment_squared_error"] += float(np.sum(residual**2))
    statistics["increment_absolute_error"] += float(np.sum(np.abs(residual)))
    statistics["increment_scale"] += float(np.sum(increment_scale))
    for name, quantile in (
        ("increment_coverage_50", 0.6744897501960817),
        ("increment_coverage_80", 1.2815515655446004),
        ("increment_coverage_95", 1.959963984540054),
    ):
        statistics[name] += int(np.count_nonzero(np.abs(residual) <= quantile * increment_scale))
    _update_pair(statistics["increment_pair"], prediction_increment, target_increment)

    target_reflectivity = np.asarray(source["signed_reflectivity"], dtype=np.float64)[support]
    prediction_reflectivity = np.asarray(output["signed_reflectivity_mean"], dtype=np.float64)[support]
    reflectivity_scale = np.asarray(output["signed_reflectivity_scale"], dtype=np.float64)[support]
    reflectivity_residual = prediction_reflectivity - target_reflectivity
    statistics["reflectivity_squared_error"] += float(np.sum(reflectivity_residual**2))
    statistics["reflectivity_absolute_error"] += float(np.sum(np.abs(reflectivity_residual)))
    statistics["reflectivity_scale"] += float(np.sum(reflectivity_scale))
    for name, quantile in (
        ("reflectivity_coverage_50", 0.6744897501960817),
        ("reflectivity_coverage_80", 1.2815515655446004),
        ("reflectivity_coverage_95", 1.959963984540054),
    ):
        statistics[name] += int(
            np.count_nonzero(np.abs(reflectivity_residual) <= quantile * reflectivity_scale)
        )
    _update_pair(
        statistics["reflectivity_pair"],
        prediction_reflectivity,
        target_reflectivity,
    )
    nonzero = np.abs(target_reflectivity) > 1.0e-8
    statistics["reflectivity_polarity_count"] += int(np.count_nonzero(nonzero))
    statistics["reflectivity_polarity_correct"] += int(
        np.count_nonzero(
            np.sign(prediction_reflectivity[nonzero])
            == np.sign(target_reflectivity[nonzero])
        )
    )


def _update_increment_statistics(
    statistics: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
) -> None:
    predicted = np.asarray(prediction, dtype=np.float64)[support]
    expected = np.asarray(target, dtype=np.float64)[support]
    residual = predicted - expected
    statistics["count"] += int(expected.size)
    statistics["squared_error"] += float(np.sum(residual**2))
    statistics["absolute_error"] += float(np.sum(np.abs(residual)))
    _update_pair(statistics["pair"], predicted, expected)


def _finalize_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    count = int(statistics["count"])
    if count <= 0:
        raise ValueError("cannot finalize empty evidence statistics.")
    polarity_count = int(statistics["reflectivity_polarity_count"])
    return {
        "supported_samples": count,
        "increment_rmse": float(np.sqrt(float(statistics["increment_squared_error"]) / count)),
        "increment_mae": float(statistics["increment_absolute_error"]) / count,
        "increment_correlation": _correlation(np.asarray(statistics["increment_pair"]), count),
        "increment_mean_scale": float(statistics["increment_scale"]) / count,
        "increment_coverage_50": float(statistics["increment_coverage_50"]) / count,
        "increment_coverage_80": float(statistics["increment_coverage_80"]) / count,
        "increment_coverage_95": float(statistics["increment_coverage_95"]) / count,
        "reflectivity_rmse": float(np.sqrt(float(statistics["reflectivity_squared_error"]) / count)),
        "reflectivity_mae": float(statistics["reflectivity_absolute_error"]) / count,
        "reflectivity_correlation": _correlation(np.asarray(statistics["reflectivity_pair"]), count),
        "reflectivity_mean_scale": float(statistics["reflectivity_scale"]) / count,
        "reflectivity_coverage_50": float(statistics["reflectivity_coverage_50"]) / count,
        "reflectivity_coverage_80": float(statistics["reflectivity_coverage_80"]) / count,
        "reflectivity_coverage_95": float(statistics["reflectivity_coverage_95"]) / count,
        "reflectivity_polarity_accuracy": (
            float(statistics["reflectivity_polarity_correct"]) / polarity_count
            if polarity_count > 0
            else None
        ),
    }


def _finalize_increment_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    count = int(statistics["count"])
    if count <= 0:
        raise ValueError("cannot finalize empty baseline statistics.")
    return {
        "supported_samples": count,
        "increment_rmse": float(np.sqrt(float(statistics["squared_error"]) / count)),
        "increment_mae": float(statistics["absolute_error"]) / count,
        "increment_correlation": _correlation(np.asarray(statistics["pair"]), count),
    }


def _network_output(
    model: EvidenceModel,
    source: Mapping[str, object],
) -> dict[str, np.ndarray]:
    batch = _to_device(source, model.device)
    output = model.network(
        batch["seismic"],
        batch["lfm_residual"],
        batch["observed_valid"],
        batch["lateral_m"],
        batch["lateral_valid"],
    )
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


def _matched_shuffle_batch(
    source: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, float]]:
    seismic = np.asarray(source["seismic"], dtype=np.float32)
    observed = np.asarray(source["observed_valid"], dtype=bool)
    lateral_valid = np.asarray(source["lateral_valid"], dtype=bool)
    if seismic.ndim != 3 or observed.shape != seismic.shape or seismic.shape[0] != 1:
        raise ValueError("matched shuffle expects one [lateral, sample] tile.")
    valid_indices = np.flatnonzero(lateral_valid[0]).tolist()
    if len(valid_indices) < 2:
        raise ValueError("matched shuffle requires at least two valid traces.")
    donors = valid_indices[1:] + valid_indices[:1]
    shuffled = seismic.copy()
    recipient_count = common_count = changed_count = 0
    squared_change = 0.0
    for recipient, donor in zip(valid_indices, donors, strict=True):
        common = observed[0, recipient] & observed[0, donor]
        recipient_count += int(np.count_nonzero(observed[0, recipient]))
        common_count += int(np.count_nonzero(common))
        difference = seismic[0, donor, common] - seismic[0, recipient, common]
        changed_count += int(np.count_nonzero(difference != 0.0))
        squared_change += float(np.sum(difference.astype(np.float64) ** 2))
        shuffled[0, recipient, common] = seismic[0, donor, common]
    if common_count == 0:
        raise ValueError("matched shuffle has no common observation support.")
    result = dict(source)
    result["seismic"] = shuffled
    return result, {
        "recipient_valid_samples": float(recipient_count),
        "common_valid_samples": float(common_count),
        "changed_valid_samples": float(changed_count),
        "squared_change": squared_change,
    }


def _bootstrap_delta(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    lower_is_better: bool,
) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("paired comparison has no parent values.")
    rng = np.random.default_rng(seed)
    if values.size == 1:
        low = high = float(values[0])
    else:
        draws = rng.integers(0, values.size, size=(replicates, values.size))
        low, high = (float(value) for value in np.quantile(np.mean(values[draws], axis=1), (0.025, 0.975)))
    return {
        "parent_count": int(values.size),
        "delta_definition": "full_minus_control",
        "lower_is_better": lower_is_better,
        "mean_delta": float(np.mean(values)),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def evaluate_evidence_model(
    model: EvidenceModel,
    batches: Iterable[Mapping[str, object]],
    *,
    controls: Mapping[str, EvidenceModel] | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    """Evaluate heads, LFM baselines, and matched seismic intervention."""

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    if model.network_config.input_mode != "full":
        raise ValueError("primary evaluation model must use full input.")
    models = {"full": model, **dict(controls or {})}
    for name, item in models.items():
        if item.network_config.input_mode != name:
            raise ValueError(f"control {name!r} has the wrong input_mode.")
        left = asdict(item.network_config)
        right = asdict(model.network_config)
        left.pop("input_mode")
        right.pop("input_mode")
        if left != right or item.target_contract != model.target_contract:
            raise ValueError("evaluation checkpoints do not share one model contract.")
        item.network.eval()

    evaluated_names = tuple(models) + ("matched_shuffle",)
    overall_models = {name: _new_statistics() for name in evaluated_names}
    overall_baselines = {
        "zone_linear_anchor_only": _new_increment_statistics(),
        "full_lfm_only": _new_increment_statistics(),
    }
    parent_models: dict[str, dict[str, dict[str, Any]]] = {}
    parent_baselines: dict[str, dict[str, dict[str, Any]]] = {}
    shuffle = {
        "recipient_valid_samples": 0.0,
        "common_valid_samples": 0.0,
        "changed_valid_samples": 0.0,
        "squared_change": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for source in batches:
            batch_count += 1
            parent_id = source.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError("evaluation batch requires parent_id.")
            parent_model = parent_models.setdefault(
                parent_id,
                {name: _new_statistics() for name in evaluated_names},
            )
            parent_baseline = parent_baselines.setdefault(
                parent_id,
                {name: _new_increment_statistics() for name in overall_baselines},
            )
            for name, item in models.items():
                output = _network_output(item, source)
                _update_statistics(overall_models[name], output, source)
                _update_statistics(parent_model[name], output, source)
            shuffled_source, diagnostics = _matched_shuffle_batch(source)
            for key, value in diagnostics.items():
                shuffle[key] += value
            shuffled_output = _network_output(model, shuffled_source)
            _update_statistics(overall_models["matched_shuffle"], shuffled_output, source)
            _update_statistics(parent_model["matched_shuffle"], shuffled_output, source)

            support = np.asarray(source["support"], dtype=bool)
            target = np.asarray(source["projected_log_ai_increment"], dtype=np.float64)
            baselines = {
                "zone_linear_anchor_only": np.zeros_like(target),
                "full_lfm_only": (
                    np.asarray(source["lfm"], dtype=np.float64)
                    - np.asarray(source["background_lfm_linear"], dtype=np.float64)
                ),
            }
            for name, prediction in baselines.items():
                _update_increment_statistics(overall_baselines[name], prediction, target, support)
                _update_increment_statistics(parent_baseline[name], prediction, target, support)
    if batch_count == 0:
        raise ValueError("evaluation received no batches.")

    parent_rows = [
        {
            "parent_id": parent_id,
            "models": {
                name: _finalize_statistics(statistics)
                for name, statistics in parent_models[parent_id].items()
            },
            "baselines": {
                name: _finalize_increment_statistics(statistics)
                for name, statistics in parent_baselines[parent_id].items()
            },
        }
        for parent_id in sorted(parent_models)
    ]
    comparisons: dict[str, Any] = {}
    index = 0
    for control in evaluated_names:
        if control == "full":
            continue
        for metric in (
            "increment_rmse",
            "reflectivity_rmse",
        ):
            values = np.asarray(
                [row["models"]["full"][metric] - row["models"][control][metric] for row in parent_rows],
                dtype=np.float64,
            )
            comparisons[f"full_vs_{control}:{metric}"] = _bootstrap_delta(
                values,
                seed=bootstrap_seed + index,
                replicates=bootstrap_replicates,
                lower_is_better=True,
            )
            index += 1
    for control in overall_baselines:
        values = np.asarray(
            [row["models"]["full"]["increment_rmse"] - row["baselines"][control]["increment_rmse"] for row in parent_rows],
            dtype=np.float64,
        )
        comparisons[f"full_vs_{control}:increment_rmse"] = _bootstrap_delta(
            values,
            seed=bootstrap_seed + index,
            replicates=bootstrap_replicates,
            lower_is_better=True,
        )
        index += 1
    return {
        "batch_count": batch_count,
        "parent_count": len(parent_rows),
        "models": {
            name: {
                "input_mode": (
                    models[name].network_config.input_mode
                    if name in models
                    else "matched_within_parent_seismic_shuffle"
                ),
                "metrics": _finalize_statistics(overall_models[name]),
            }
            for name in evaluated_names
        },
        "baselines": {
            name: _finalize_increment_statistics(statistics)
            for name, statistics in overall_baselines.items()
        },
        "paired_comparisons": comparisons,
        "matched_shuffle_diagnostics": {
            "common_valid_fraction": shuffle["common_valid_samples"]
            / max(shuffle["recipient_valid_samples"], 1.0),
            "changed_common_fraction": shuffle["changed_valid_samples"]
            / max(shuffle["common_valid_samples"], 1.0),
            "change_rms": float(
                np.sqrt(shuffle["squared_change"] / max(shuffle["common_valid_samples"], 1.0))
            ),
        },
        "parents": parent_rows,
    }


__all__ = [
    "CheckpointCallback",
    "EvidenceLearningConfig",
    "evaluate_evidence_model",
    "seed_random_streams",
    "train_evidence_model",
]
