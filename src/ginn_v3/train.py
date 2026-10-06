"""Simultaneous PIAI training for the independent ``ginn_v3`` workflow.

The trainer intentionally owns the PIAI loop instead of adapting the v2
trainer.  The three terms are the original independent, physics-informed, and
cross-learning terms, with a shared batch wavelet for each physical forward.
No v2 smoothing, low-frequency projection, anchor, TV, or waveform-shape
normalisation is introduced here.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ginn_v3.config import NetworkConfig, TrainingConfig
from ginn_v3.model import PIAINetwork
from ginn_v3.types import ForwardResult, ObservationBatch, Prediction, TrainingResult, WellBatch


CHECKPOINT_SCHEMA = "ginn_v3_checkpoint_v1"


def _as_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _normalization_dict(normalization: Any) -> dict[str, float]:
    names = (
        "seismic_mean",
        "seismic_std",
        "lfm_mean",
        "lfm_std",
        "impedance_mean",
        "impedance_std",
    )
    result: dict[str, float] = {}
    for name in names:
        if isinstance(normalization, Mapping):
            value = normalization[name]
        else:
            value = getattr(normalization, name)
        result[name] = _as_float(value, name=f"normalization.{name}")
    return result


def _sample_axis_metadata(axis: Any) -> dict[str, Any]:
    values = getattr(axis, "values", None)
    domain = getattr(axis, "domain", None)
    unit = getattr(axis, "unit", None)
    depth_basis = getattr(axis, "depth_basis", None)
    if values is None or domain is None or unit is None or not hasattr(axis, "depth_basis"):
        raise ValueError("sample_axis must expose explicit values, domain, unit, and depth_basis fields.")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("sample_axis must expose a finite one-dimensional values array.")
    return {
        "domain": str(domain),
        "unit": str(unit),
        "depth_basis": None if depth_basis is None else str(depth_basis),
        "values": values.tolist(),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive.")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _finite_scalar(value: Tensor, *, name: str) -> float:
    if value.numel() != 1 or not bool(torch.isfinite(value).item()):
        raise FloatingPointError(f"{name} is non-finite.")
    return float(value.detach().cpu().item())


def _masked_mse(predicted: Tensor, target: Tensor, mask: Tensor, *, name: str) -> Tensor:
    if predicted.shape != target.shape or predicted.shape != mask.shape:
        raise ValueError(
            f"{name} requires matching prediction, target, and mask shapes; "
            f"got {tuple(predicted.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}."
        )
    mask = mask.to(device=predicted.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        raise ValueError(f"{name} has no valid samples after masking.")
    selected = (predicted - target)[mask]
    if not bool(torch.isfinite(selected).all().item()):
        raise FloatingPointError(f"{name} contains a non-finite selected value.")
    return torch.mean(selected.square())


def _forward_result(result: Any) -> tuple[Tensor, Tensor]:
    if isinstance(result, ForwardResult):
        return result.seismic, result.valid_mask
    raise TypeError("physics.forward must return ForwardResult(seismic, valid_mask).")


class JointTrainer:
    """Train a :class:`PIAINetwork` using the original I/P/C mechanism.

    ``data`` and ``physics`` are intentionally accepted as dependencies.  The
    trainer only relies on their v3 contracts and therefore remains usable for
    both time-domain and fixed-velocity depth-domain adapters.
    """

    def __init__(self, data: Any, model: PIAINetwork, physics: Any, config: TrainingConfig, logger: Any = None) -> None:
        if not isinstance(model, PIAINetwork):
            raise TypeError("model must be a PIAINetwork instance.")
        if not isinstance(config, TrainingConfig):
            raise TypeError("config must be a TrainingConfig instance.")
        self.data = data
        self.model = model
        self.physics = physics
        self.config = config
        self.logger = logger
        self._normalization = _normalization_dict(data.reader.normalization)
        self._device = torch.device(config.device)
        self.model.to(self._device)
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._history: list[dict[str, Any]] = []

    def _log(self, message: str, *args: Any) -> None:
        if self.logger is None:
            return
        if hasattr(self.logger, "info"):
            self.logger.info(message, *args)
        elif callable(self.logger):
            self.logger(message, *args)

    def _standardized_observed(self, batch: ObservationBatch) -> Tensor:
        return (
            batch.observed_seismic - self._normalization["seismic_mean"]
        ) / self._normalization["seismic_std"]

    def _physics_mse(
        self,
        log_ai: Tensor,
        batch: ObservationBatch,
        wavelet: Tensor,
        *,
        support_mask: Tensor | None,
        name: str,
    ) -> Tensor:
        result = self.physics.forward(log_ai, batch, wavelet, support_mask=support_mask)
        synthetic, valid = _forward_result(result)
        if synthetic.shape != batch.observed_seismic.shape:
            raise ValueError(f"{name} physics output shape differs from observations.")
        valid = valid.to(device=synthetic.device, dtype=torch.bool)
        mask = batch.observed_mask.to(device=synthetic.device, dtype=torch.bool) & valid
        return _masked_mse(
            synthetic,
            self._standardized_observed(batch).to(device=synthetic.device, dtype=synthetic.dtype),
            mask,
            name=name,
        )

    def _well_ai_mse(self, prediction: Prediction, well: WellBatch, *, name: str) -> Tensor:
        mask = well.target_mask.to(device=prediction.log_ai.device, dtype=torch.bool)
        if mask.shape != prediction.log_ai.shape or mask.shape != well.target_log_ai.shape:
            raise ValueError(f"{name} requires a target mask matching both log-AI arrays.")
        if not bool(mask.any().item()):
            raise ValueError(f"{name} has no valid supervised samples.")
        # Select before exp: an unsupported sentinel must never overflow and
        # contaminate autograd before the mask can remove it.
        predicted_log = prediction.log_ai[mask]
        target_log = well.target_log_ai.to(device=prediction.log_ai.device, dtype=prediction.log_ai.dtype)[mask]
        if not bool(torch.isfinite(predicted_log).all().item()) or not bool(torch.isfinite(target_log).all().item()):
            raise FloatingPointError(f"{name} contains non-finite supported log-AI values.")
        predicted_ai = torch.exp(predicted_log)
        target_ai = torch.exp(target_log)
        if not bool(torch.isfinite(predicted_ai).all().item()) or not bool(torch.isfinite(target_ai).all().item()):
            raise FloatingPointError(f"{name} contains supported log-AI values whose exp is non-finite.")
        normalized = (predicted_ai - target_ai) / self._normalization["impedance_std"]
        if not bool(torch.isfinite(normalized).all().item()):
            raise FloatingPointError(f"{name} contains a non-finite normalized impedance error.")
        return torch.mean(normalized.square())

    def _step(self, labeled: WellBatch, unlabeled: ObservationBatch) -> dict[str, float]:
        labeled_prediction = self.model(
            labeled.observations.features,
            labeled.observations.initial_log_ai,
        )
        unlabeled_prediction = self.model(
            unlabeled.features,
            unlabeled.initial_log_ai,
        )
        wavelet_l = labeled_prediction.wavelets.mean(dim=0)
        wavelet_u = unlabeled_prediction.wavelets.mean(dim=0)

        independent_ai = self._well_ai_mse(
            labeled_prediction,
            labeled,
            name="independent impedance loss",
        )
        independent_physics = self._physics_mse(
            labeled.target_log_ai,
            labeled.observations,
            wavelet_l,
            support_mask=labeled.target_mask,
            name="independent target forward loss",
        )
        loss_i = independent_ai + independent_physics

        loss_p = self._physics_mse(
            labeled_prediction.log_ai,
            labeled.observations,
            wavelet_l,
            support_mask=None,
            name="physics-informed loss",
        )

        loss_c_unlabeled = self._physics_mse(
            unlabeled_prediction.log_ai,
            unlabeled,
            wavelet_l.detach(),
            support_mask=None,
            name="cross unlabeled loss",
        )
        loss_c_labeled = self._physics_mse(
            labeled.target_log_ai,
            labeled.observations,
            wavelet_u,
            support_mask=labeled.target_mask,
            name="cross labeled loss",
        )
        loss_c = 0.5 * (loss_c_unlabeled + loss_c_labeled)

        weights = self.config.loss_weights
        total = weights.independent * loss_i + weights.physics * loss_p + weights.cross * loss_c
        _finite_scalar(total, name="total loss")
        self._optimizer.zero_grad(set_to_none=True)
        total.backward()
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(f"non-finite gradient in {name}.")
        self._optimizer.step()
        if not all(bool(torch.isfinite(p).all().item()) for p in self.model.parameters()):
            raise FloatingPointError("optimizer produced non-finite model parameters.")
        return {
            "loss": _finite_scalar(total, name="total loss after backward"),
            "loss_independent": _finite_scalar(loss_i, name="independent loss"),
            "loss_physics": _finite_scalar(loss_p, name="physics loss"),
            "loss_cross": _finite_scalar(loss_c, name="cross loss"),
        }

    def _keys(self, name: str, *, limit: int | None = None) -> list[Any]:
        values = list(getattr(self.data, name))
        if limit is not None:
            values = values[:limit]
        if not values:
            raise ValueError(f"data.{name} is empty.")
        return values

    def _wavelet_time(self) -> list[float]:
        value = getattr(self.physics, "wavelet_time_s", None)
        if value is None:
            raise ValueError("physics.wavelet_time_s is required for v3 checkpoints.")
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if values.size != self.model.config.wavelet_samples:
            raise ValueError(
                "physics.wavelet_time_s length must equal model wavelet_samples."
            )
        if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
            raise ValueError("physics.wavelet_time_s must be finite and strictly increasing seconds.")
        return values.tolist()

    def _deterministic_mean_wavelet(self) -> Tensor:
        # The exported wavelet represents the complete deterministic training
        # cohort.  It is deliberately independent of the validation sampling
        # limit used for the selection metric.
        keys = self._keys("train_keys")
        self.model.eval()
        waves: list[Tensor] = []
        with torch.no_grad():
            for key_chunk in _chunks(keys, self.config.unlabeled_batch_size):
                batch = self.data.batch(key_chunk, self._device)
                prediction = self.model(batch.features, batch.initial_log_ai)
                waves.append(prediction.wavelets.detach())
        if not waves:
            raise ValueError("Cannot compute a deterministic wavelet cohort from empty data.")
        return torch.cat(waves, dim=0).mean(dim=0)

    def _validation(self) -> tuple[float, float, float]:
        keys = self._keys("validation_keys", limit=self.config.validation_traces)
        self.model.eval()
        seismic_sum = 0.0
        seismic_count = 0
        with torch.no_grad():
            for key_chunk in _chunks(keys, self.config.unlabeled_batch_size):
                batch = self.data.batch(key_chunk, self._device)
                prediction = self.model(batch.features, batch.initial_log_ai)
                forward = self.physics.forward(
                    prediction.log_ai,
                    batch,
                    prediction.wavelets.mean(dim=0),
                    support_mask=None,
                )
                synthetic, valid = _forward_result(forward)
                mask = batch.observed_mask.to(synthetic.device) & valid.to(synthetic.device, dtype=torch.bool)
                target = self._standardized_observed(batch).to(synthetic.device, synthetic.dtype)
                selected = (synthetic - target)[mask]
                if selected.numel() == 0:
                    raise ValueError("validation seismic metric has no valid samples.")
                if not bool(torch.isfinite(selected).all().item()):
                    raise FloatingPointError("validation seismic metric is non-finite.")
                seismic_sum += float(selected.square().sum().cpu())
                seismic_count += int(selected.numel())
        validation_seismic = seismic_sum / seismic_count

        well_sum = 0.0
        well_count = 0
        train_wells = list(getattr(self.data, "train_wells"))
        with torch.no_grad():
            for well_chunk in _chunks(train_wells, self.config.labeled_batch_size):
                well = self.data.well_batch(well_chunk, self._device)
                prediction = self.model(
                    well.observations.features,
                    well.observations.initial_log_ai,
                )
                mask = well.target_mask.to(prediction.log_ai.device, dtype=torch.bool)
                if not bool(mask.any().item()):
                    raise ValueError("training well validation metric has no valid target samples.")
                # Apply the support mask before exponentiating, for the same
                # reason as _well_ai_mse: unsupported sentinels are not data.
                predicted_log = prediction.log_ai[mask]
                target_log = well.target_log_ai.to(device=prediction.log_ai.device, dtype=prediction.log_ai.dtype)[mask]
                if not bool(torch.isfinite(predicted_log).all().item()) or not bool(torch.isfinite(target_log).all().item()):
                    raise FloatingPointError("training well validation log-AI metric is non-finite.")
                selected = (
                    torch.exp(predicted_log) - torch.exp(target_log)
                ) / self._normalization["impedance_std"]
                if not bool(torch.isfinite(selected).all().item()):
                    raise FloatingPointError("training well validation metric is non-finite.")
                well_sum += float(selected.square().sum().cpu())
                well_count += int(selected.numel())
        train_well_ai = well_sum / well_count
        return validation_seismic, train_well_ai, validation_seismic + train_well_ai

    def _checkpoint_payload(
        self,
        *,
        update: int,
        mean_wavelet: Tensor,
        checkpoint_context: Mapping[str, Any] | None,
        validation: Mapping[str, float] | None,
    ) -> dict[str, Any]:
        axis = self.data.reader.sample_axis
        return {
            "schema": CHECKPOINT_SCHEMA,
            "update": int(update),
            "model_config": asdict(self.model.config),
            "model_state": self.model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "training_config": asdict(self.config),
            "normalization": dict(self._normalization),
            "sample_axis": _sample_axis_metadata(axis),
            "wavelet_time_s": self._wavelet_time(),
            "mean_wavelet_normalized": mean_wavelet.detach().cpu().reshape(-1).tolist(),
            "mean_wavelet_cohort": {
                "kind": "all_train_keys",
                "count": len(list(getattr(self.data, "train_keys"))),
            },
            "context": _jsonable(dict(checkpoint_context or {})),
            "validation": dict(validation or {}),
        }

    def _save_checkpoint(
        self,
        path: Path,
        *,
        update: int,
        checkpoint_context: Mapping[str, Any] | None,
        validation: Mapping[str, float] | None,
    ) -> None:
        mean_wavelet = self._deterministic_mean_wavelet()
        payload = self._checkpoint_payload(
            update=update,
            mean_wavelet=mean_wavelet,
            checkpoint_context=checkpoint_context,
            validation=validation,
        )
        torch.save(payload, path)

    def fit(self, output_dir: str | Path, checkpoint_context: Mapping[str, Any] | None = None) -> TrainingResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed % (2**32 - 1))
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        train_keys = self._keys("train_keys", limit=self.config.max_train_traces)
        train_wells = self._keys("train_wells")
        if not getattr(self.data, "validation_keys", None):
            raise ValueError("data.validation_keys is empty.")

        rng = np.random.default_rng(self.config.seed)
        permutation = np.arange(len(train_keys), dtype=np.int64)
        rng.shuffle(permutation)
        well_permutation = np.arange(len(train_wells), dtype=np.int64)
        rng.shuffle(well_permutation)
        unlabeled_cursor = 0
        well_cursor = 0
        unlabeled_exposures = 0
        labeled_exposures = 0
        best_validation_metric = float("inf")
        best_validation_update = -1
        final_validation_metric = float("nan")
        selected_path = output / "selected_checkpoint.pt"
        last_path = output / "last_checkpoint.pt"
        best_path = output / "best_validation_checkpoint.pt"
        started_at = time.perf_counter()

        self.model.train()
        for update in range(1, self.config.updates + 1):
            if unlabeled_cursor >= len(train_keys):
                rng.shuffle(permutation)
                unlabeled_cursor = 0
            unlabeled_end = min(
                unlabeled_cursor + self.config.unlabeled_batch_size,
                len(train_keys),
            )
            indices = permutation[unlabeled_cursor:unlabeled_end]
            unlabeled_cursor = unlabeled_end
            unlabeled_exposures += int(indices.size)
            keys = [train_keys[int(index)] for index in indices]

            if well_cursor >= len(train_wells):
                rng.shuffle(well_permutation)
                well_cursor = 0
            well_end = min(
                well_cursor + self.config.labeled_batch_size,
                len(train_wells),
            )
            well_indices = well_permutation[well_cursor:well_end]
            wells = [train_wells[int(index)] for index in well_indices]
            well_cursor = well_end
            labeled_exposures += len(wells)
            labeled = self.data.well_batch(wells, self._device)
            unlabeled = self.data.batch(keys, self._device)
            metrics = self._step(labeled, unlabeled)
            metrics["update"] = update
            metrics["unlabeled_exposures"] = unlabeled_exposures
            metrics["labeled_exposures"] = labeled_exposures
            metrics["validation_seismic_mse"] = float("nan")
            metrics["train_well_ai_mse"] = float("nan")
            metrics["selection_metric"] = float("nan")

            validate = update % self.config.validate_every == 0 or update == self.config.updates
            validation_payload: dict[str, float] | None = None
            if validate:
                self.model.eval()
                val_seis, train_well, score = self._validation()
                validation_payload = {
                    "validation_seismic_mse": val_seis,
                    "train_well_ai_mse": train_well,
                    "selection_metric": score,
                }
                metrics.update(validation_payload)
                self._save_checkpoint(
                    last_path,
                    update=update,
                    checkpoint_context=checkpoint_context,
                    validation=validation_payload,
                )
                final_validation_metric = score if update == self.config.updates else final_validation_metric
                if score < best_validation_metric:
                    best_validation_metric = score
                    best_validation_update = update
                    self._save_checkpoint(
                        best_path,
                        update=update,
                        checkpoint_context=checkpoint_context,
                        validation=validation_payload,
                    )
                self.model.train()
            elif update == self.config.updates:
                self._save_checkpoint(
                    last_path,
                    update=update,
                    checkpoint_context=checkpoint_context,
                    validation=validation_payload,
                )

            metrics["elapsed_s"] = time.perf_counter() - started_at
            self._history.append(metrics)
            if update % self.config.log_every == 0 or update == 1 or update == self.config.updates:
                self._log(
                    "v3 update %d/%d loss=%.6g I=%.6g P=%.6g C=%.6g",
                    update,
                    self.config.updates,
                    metrics["loss"],
                    metrics["loss_independent"],
                    metrics["loss_physics"],
                    metrics["loss_cross"],
                )

        if not last_path.exists() or not math.isfinite(final_validation_metric):
            raise RuntimeError("The final update did not produce a validation checkpoint.")
        # The default selected artifact is deliberately the final optimizer
        # state.  Validation is diagnostic here, preserving the original
        # fixed-update PIAI procedure rather than introducing model selection.
        selected_path.write_bytes(last_path.read_bytes())

        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(self._history[0])
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._history)
        summary = {
            "schema": CHECKPOINT_SCHEMA,
            "updates_completed": self.config.updates,
            "selection_rule": "final_update",
            "selected_update": self.config.updates,
            "selected_metric": final_validation_metric,
            "best_validation_update": best_validation_update,
            "best_validation_metric": best_validation_metric,
            "unlabeled_exposures": unlabeled_exposures,
            "labeled_exposures": labeled_exposures,
            "elapsed_s": time.perf_counter() - started_at,
            "selected_checkpoint": str(selected_path),
            "last_checkpoint": str(last_path),
            "best_validation_checkpoint": str(best_path) if best_path.exists() else None,
            "training_config": _jsonable(asdict(self.config)),
            "model_config": _jsonable(asdict(self.model.config)),
        }
        (output / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return TrainingResult(
            output_dir=output,
            selected_checkpoint=selected_path,
            last_checkpoint=last_path,
            updates_completed=self.config.updates,
        )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and instantiate a v3 checkpoint without touching v2 artifacts."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported or missing v3 checkpoint schema: {payload.get('schema') if isinstance(payload, dict) else None!r}.")
    model = PIAINetwork(NetworkConfig.from_mapping(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    result = dict(payload)
    result["model"] = model
    result["device"] = str(device)
    return result


__all__ = ["CHECKPOINT_SCHEMA", "JointTrainer", "load_checkpoint"]
