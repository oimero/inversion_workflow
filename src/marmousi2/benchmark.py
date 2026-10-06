"""Small explicit benchmark runner built on the main GINN workflow.

This module owns candidate configuration generation and multi-seed bookkeeping;
the network, training loop, checkpoint selection, and volume inference remain
in :mod:`ginn_v2.workflow`.  No historical ablation registry or cached
pretraining branch is implied by this interface.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .evaluation import evaluate_marmousi2_prediction


DEFAULT_UNLABELED_STEPS = 256
DEFAULT_WELL_STEPS = 256


SMALL_COMPARISON_LABELS = ("baseline", "p0-random", "mw", "mvw-low-visible", "w-only")


def _repo_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def comparison_plan(
    *,
    candidates: Sequence[str] | None = None,
    visible_weight: float = 0.25,
    unlabeled_steps: int = DEFAULT_UNLABELED_STEPS,
    well_steps: int = DEFAULT_WELL_STEPS,
) -> list[dict[str, Any]]:
    """Return a small, explicit candidate matrix for review or execution."""
    selected = list(candidates) if candidates is not None else list(SMALL_COMPARISON_LABELS)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("candidates must be non-empty and unique.")
    unknown = sorted(set(selected) - set(SMALL_COMPARISON_LABELS))
    if unknown:
        raise ValueError(f"Unknown benchmark candidates: {unknown}; choose from {list(SMALL_COMPARISON_LABELS)}.")
    if not np.isfinite(float(visible_weight)) or float(visible_weight) < 0.0:
        raise ValueError("visible_weight must be finite and non-negative.")
    if isinstance(unlabeled_steps, bool) or int(unlabeled_steps) != unlabeled_steps or int(unlabeled_steps) < 0:
        raise ValueError("unlabeled_steps must be a non-negative integer.")
    if isinstance(well_steps, bool) or int(well_steps) != well_steps or int(well_steps) <= 0:
        raise ValueError("well_steps must be a positive integer.")
    common = {
        "finetune_unlabeled_steps": int(unlabeled_steps),
        "finetune_well_steps": int(well_steps),
    }
    candidate_changes = {
        "baseline": {"finetune_strategy": "mvw", "visible_seismic_weight": 1.0, **common},
        "p0-random": {"pretrain_epochs": 0, "finetune_strategy": "mvw", "visible_seismic_weight": 1.0, **common},
        "mw": {"finetune_strategy": "mw", "visible_seismic_weight": 1.0, **common},
        "mvw-low-visible": {
            "finetune_strategy": "mvw", "visible_seismic_weight": float(visible_weight), **common,
        },
        "w-only": {
            "finetune_strategy": "w_only", "finetune_unlabeled_steps": 0,
            "finetune_well_steps": int(well_steps), "visible_seismic_weight": 1.0,
        },
    }
    result: list[dict[str, Any]] = []
    for label in selected:
        result.append({"label": str(label), "changes": deepcopy(candidate_changes[str(label)])})
    return result


def _candidate_config(base_config: Mapping[str, Any], *, changes: Mapping[str, Any], seed: int) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    section = config.get("ginn_v2_body_inversion")
    if not isinstance(section, dict):
        raise ValueError("Base config lacks ginn_v2_body_inversion mapping.")
    training = section.get("training")
    if not isinstance(training, dict):
        raise ValueError("Base config lacks ginn_v2_body_inversion.training mapping.")
    training.update(dict(changes))
    training["seed"] = int(seed)
    return config


def write_comparison_plan(
    base_config_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: Path,
    candidates: Sequence[str] | None = None,
    seeds: Sequence[int] = (20261004,),
    visible_weight: float = 0.25,
    unlabeled_steps: int = DEFAULT_UNLABELED_STEPS,
    well_steps: int = DEFAULT_WELL_STEPS,
) -> dict[str, Any]:
    """Materialize candidate configs and a reviewable CSV/JSON manifest."""
    repo_root = Path(repo_root).resolve()
    base_path = _repo_path(base_config_path, repo_root=repo_root)
    output = _repo_path(output_dir, repo_root=repo_root)
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, Mapping):
        raise ValueError("Base benchmark config must contain a mapping.")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values) or any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must be unique non-negative integers.")
    plan = comparison_plan(
        candidates=candidates,
        visible_weight=visible_weight,
        unlabeled_steps=unlabeled_steps,
        well_steps=well_steps,
    )
    base_training = dict(dict(base.get("ginn_v2_body_inversion") or {}).get("training") or {})
    configs_dir = output / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate in plan:
        for seed in seed_values:
            label = str(candidate["label"])
            config_path = configs_dir / f"{label}__seed_{seed}.yaml"
            config = _candidate_config(base, changes=candidate["changes"], seed=seed)
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            changes = dict(candidate["changes"])
            strategy = str(changes.get("finetune_strategy", base_training.get("finetune_strategy", "mvw")))
            finetune_epochs = int(base_training.get("finetune_epochs", 1))
            requested_unlabeled = int(changes.get("finetune_unlabeled_steps", base_training.get("finetune_unlabeled_steps") or 0))
            masked_updates = (requested_unlabeled + 1) // 2 if strategy == "mvw" else requested_unlabeled if strategy == "mw" else 0
            visible_updates = requested_unlabeled // 2 if strategy == "mvw" else requested_unlabeled if strategy == "vw" else 0
            trusted_well_updates = int(changes.get("finetune_well_steps", base_training.get("finetune_well_steps") or 0))
            rows.append({
                "label": label,
                "seed": seed,
                "config": config_path.as_posix(),
                "finetune_strategy": strategy,
                "pretrain_epochs": int(changes.get("pretrain_epochs", base_training.get("pretrain_epochs", 1))),
                "finetune_epochs": finetune_epochs,
                "masked_updates_M": masked_updates,
                "visible_updates_V": visible_updates,
                "trusted_well_updates_W": trusted_well_updates,
                "budget_unit": "updates_per_finetune_epoch",
                "expected_total_M_updates": masked_updates * finetune_epochs,
                "expected_total_V_updates": visible_updates * finetune_epochs,
                "expected_total_W_updates": trusted_well_updates * finetune_epochs,
                "changes_json": json.dumps(changes, ensure_ascii=False, sort_keys=True),
                "status": "planned",
                "output_dir": (output / "runs" / f"{label}__seed_{seed}").as_posix(),
            })
    manifest = output / "comparison_plan.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "marmousi2_comparison_plan_v1",
        "status": "planned",
        "base_config": base_path.as_posix(),
        "manifest": manifest.as_posix(),
        "candidate_count": len(plan),
        "seed_count": len(seed_values),
        "rows": rows,
    }
    (output / "comparison_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _save_prediction(path: Path, result: Any) -> np.ndarray:
    values = np.asarray(result.body_log_ai, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        body_log_ai=values,
        direction_count=np.asarray(result.direction_count),
        fill_code=np.asarray(result.fill_code),
        inline_indices=np.asarray(result.inline_indices),
        xline_indices=np.asarray(result.xline_indices),
    )
    return values.astype(np.float64)


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _benchmark_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the compact metrics used in the comparison CSV."""
    metrics = {
        "well_log_ai_rmse": _nested(result, "impedance_metrics", "prediction", "overall", "log_ai", "rmse"),
        "well_linear_ai_rmse": _nested(result, "impedance_metrics", "prediction", "overall", "linear_ai", "rmse"),
        "forward_mean_corr": _nested(result, "physical_metrics", "prediction", "overall", "mean_waveform_corr"),
        "truth_forward_mean_corr": _nested(result, "truth_forward_correlation", "truth_vs_observed", "overall", "mean_waveform_corr"),
        "lf_drift_rmse": _nested(result, "low_frequency_drift", "against_lfm", "rmse"),
        "hf_rms": _nested(result, "highfrequency_energy", "prediction", "mean"),
        "hf_energy_fraction": _nested(result, "highfrequency_energy", "prediction", "relative_energy_fraction", "mean"),
        "shortwave_rms": _nested(result, "shortwave_energy", "prediction", "mean"),
        "shortwave_energy_fraction": _nested(result, "shortwave_energy", "prediction", "relative_energy_fraction", "mean"),
        "fixed_well_log_ai_rmse": _nested(result, "well_target_metrics", "fixed_original_well", "rmse"),
        "body_target_log_ai_rmse": _nested(result, "well_target_metrics", "body_target", "rmse"),
    }
    for label in ("supported_profile", "full_profile"):
        metrics[f"{label}_log_ai_rmse"] = _nested(result, label, "impedance_metrics", "prediction", "log_ai", "rmse")
        metrics[f"{label}_lfm_log_ai_rmse"] = _nested(result, label, "impedance_metrics", "lfm_baseline", "log_ai", "rmse")
        metrics[f"{label}_forward_mean_corr"] = _nested(result, label, "physical_metrics", "prediction", "mean_waveform_corr")
        metrics[f"{label}_shortwave_rms"] = _nested(result, label, "shortwave_energy", "prediction", "mean")
        metrics[f"{label}_shortwave_fraction"] = _nested(result, label, "shortwave_energy", "prediction", "relative_energy_fraction", "mean")
    prediction_rmse = metrics["supported_profile_log_ai_rmse"]
    lfm_rmse = metrics["supported_profile_lfm_log_ai_rmse"]
    metrics["supported_profile_rmse_vs_lfm"] = None if prediction_rmse is None or lfm_rmse is None else prediction_rmse - lfm_rmse
    return metrics


def summarize_seed_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the scientific metrics without pooling different candidates."""
    summaries = []
    for label in dict.fromkeys(str(row["label"]) for row in rows):
        selected = [row for row in rows if str(row["label"]) == label]
        summary = {"label": label, "seed_count": len(selected)}
        metric_names = tuple(_benchmark_metrics({}))
        for name in metric_names:
            values = [float(row[name]) for row in selected if row.get(name) is not None]
            if values:
                summary[f"{name}_mean"] = float(np.mean(values))
                summary[f"{name}_span"] = float(max(values) - min(values))
            else:
                summary[f"{name}_mean"] = None
                summary[f"{name}_span"] = None
        summary["seeds_better_than_lfm"] = sum(
            row.get("supported_profile_rmse_vs_lfm") is not None
            and float(row["supported_profile_rmse_vs_lfm"]) < 0 for row in selected
        )
        summaries.append(summary)
    return summaries


def run_comparison_plan(
    base_config_path: str | Path,
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    repo_root: Path,
    candidates: Sequence[str] | None = None,
    seeds: Sequence[int] = (20261004,),
    visible_weight: float = 0.25,
    unlabeled_steps: int = DEFAULT_UNLABELED_STEPS,
    well_steps: int = DEFAULT_WELL_STEPS,
    scope: str = "final",
    support_relative_threshold: float = 0.25,
) -> dict[str, Any]:
    """Train and evaluate the explicitly requested small candidate matrix."""
    repo_root = Path(repo_root).resolve()
    prepared = _repo_path(prepared_dir, repo_root=repo_root)
    output = _repo_path(output_dir, repo_root=repo_root)
    plan_payload = write_comparison_plan(
        base_config_path,
        output,
        repo_root=repo_root,
        candidates=candidates,
        seeds=seeds,
        visible_weight=visible_weight,
        unlabeled_steps=unlabeled_steps,
        well_steps=well_steps,
    )
    from ginn_v2.workflow import load_body, train_body

    results: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for row in plan_payload["rows"]:
        config_path = Path(row["config"])
        run_dir = Path(row["output_dir"])
        body_run = train_body(config_path, stage="all", output_dir=run_dir)
        checkpoint = body_run.selected_checkpoint or body_run.pretrain_checkpoint
        loaded = load_body(config_path, checkpoint=checkpoint)
        prediction_path = run_dir / "benchmark_prediction.npz"
        prediction = _save_prediction(prediction_path, loaded.predict_volume())
        lfm = np.asarray(loaded.lfm.log_ai, dtype=np.float64)
        metrics = evaluate_marmousi2_prediction(
            prepared,
            prediction,
            lfm,
            scope=scope,
            support_relative_threshold=support_relative_threshold,
        )
        updates_path = run_dir / "training_updates.json"
        if not updates_path.is_file():
            raise FileNotFoundError(f"Main workflow did not publish training updates: {updates_path}")
        updates = json.loads(updates_path.read_text(encoding="utf-8"))
        pretrain_updates = dict(dict(updates.get("stages") or {}).get("pretrain") or {})
        finetune_updates = dict(dict(updates.get("stages") or {}).get("finetune") or {})
        row["status"] = "completed"
        row["checkpoint"] = checkpoint.as_posix()
        row["prediction_npz"] = prediction_path.as_posix()
        row["actual_P_updates"] = int(pretrain_updates.get("optimizer_steps", 0))
        row["actual_M_updates"] = int(finetune_updates.get("n_M", 0))
        row["actual_V_updates"] = int(finetune_updates.get("n_V", 0))
        row["actual_W_updates"] = int(finetune_updates.get("n_W", 0))
        row["actual_optimizer_steps"] = int(finetune_updates.get("optimizer_steps", 0))
        results[f"{row['label']}__seed_{row['seed']}"] = metrics
        metric_rows.append({
            "label": row["label"],
            "seed": row["seed"],
            **{key: row.get(key) for key in ("finetune_strategy", "pretrain_epochs", "masked_updates_M", "visible_updates_V", "trusted_well_updates_W", "actual_P_updates", "actual_M_updates", "actual_V_updates", "actual_W_updates", "actual_optimizer_steps")},
            **_benchmark_metrics(metrics),
        })
    payload = {
        "schema_version": "marmousi2_benchmark_run_v1",
        "status": "completed",
        "plan": plan_payload,
        "results": results,
        "seed_summary": summarize_seed_metrics(metric_rows),
    }
    (output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    with (output / "comparison_plan.csv").open("w", encoding="utf-8", newline="") as handle:
        if plan_payload["rows"]:
            fields = list(plan_payload["rows"][0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(plan_payload["rows"])
    if metric_rows:
        with (output / "comparison_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
        with (output / "seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            summaries = payload["seed_summary"]
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return payload


__all__ = [
    "DEFAULT_UNLABELED_STEPS",
    "DEFAULT_WELL_STEPS",
    "SMALL_COMPARISON_LABELS",
    "comparison_plan",
    "run_comparison_plan",
    "write_comparison_plan",
    "summarize_seed_metrics",
]
