"""Validate sparse-well real-delta output-head adaptation after R1."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
repo_text = str(REPO_ROOT)
src_text = str(SRC_DIR)
sys.path = [path for path in sys.path if path not in {repo_text, src_text}]
sys.path.insert(0, repo_text)
sys.path.insert(0, src_text)

from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sha256_file, write_json  # noqa: E402
from cup.seismic.viz import plot_well_waveform_qc  # noqa: E402
from ginn_v2.real_delta_adapter import (  # noqa: E402
    HeadFit,
    HeadState,
    apply_head,
    build_band_metrics,
    build_role_decision,
    compare_synthetic_metrics,
    evaluate_synthetic_heads,
    extract_role_well_features,
    feature_reconstruction_summary,
    denormalize_delta_values,
    get_head_state,
    load_well_samples,
    run_role_validation,
)
from ginn_v2.training import load_checkpoint  # noqa: E402
from ginn_v2.real_field import forward_log_ai, load_selected_wavelet  # noqa: E402
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr  # noqa: E402
from wtie.processing import grid  # noqa: E402


DEFAULT_CONFIG = Path("experiments/common/common.yaml")
SYNTHETIC_SCOPE_DIRS = {
    "validation_base": "predict_validation_base",
    "validation_mismatch": "predict_validation_mismatch",
    "test_base": "predict_test_base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--forward-diagnostic-dir", type=Path, default=None)
    parser.add_argument("--zero-shot-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--plot-run-dir",
        type=Path,
        default=None,
        help="Only regenerate all-well R1-style QC figures for an existing R2 v2-input run.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _required_text(mapping: Mapping[str, Any], key: str, *, label: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"{label}.{key} must be explicit and non-empty.")
    return value


def _required_value(mapping: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{label}.{key} must be explicit.")
    return mapping[key]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _output_dir(explicit: Path | None, *, output_root: Path) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"r2_real_delta_adapter_{timestamp}"


def _validate_sources(
    *, forward_diagnostic_dir: Path, zero_shot_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    r1_summary_path = forward_diagnostic_dir / "real_field_forward_diagnostic_summary.json"
    r0_summary_path = zero_shot_dir / "real_field_zero_shot_summary.json"
    r1 = _load_json(r1_summary_path)
    r0 = _load_json(r0_summary_path)
    if r1.get("schema_version") != "real_field_forward_diagnostic_summary_v2" or r1.get("status") != "ok":
        raise ValueError(f"Unsupported R1 summary state: {r1_summary_path}")
    if r1.get("mode") != "volume":
        raise ValueError("R2 requires an R1 volume run.")
    if r0.get("schema_version") != "real_field_zero_shot_summary_v1":
        raise ValueError(f"Unsupported R0 summary schema: {r0_summary_path}")
    if r0.get("mode") != "volume":
        raise ValueError("R2 requires an R0 volume run.")
    recorded_r0 = resolve_relative_path(_required_text(r1, "zero_shot_dir", label="R1 summary"), root=REPO_ROOT)
    if recorded_r0.resolve() != zero_shot_dir.resolve():
        raise ValueError(f"R1 source R0 differs from configured zero_shot_dir: {recorded_r0} vs {zero_shot_dir}")
    expected_r0_hash = str(r1.get("zero_shot_summary_sha256") or "")
    actual_r0_hash = sha256_file(r0_summary_path)
    if expected_r0_hash != actual_r0_hash:
        raise ValueError(
            f"R1 zero_shot_summary_sha256 mismatch: expected={expected_r0_hash}, actual={actual_r0_hash}"
        )
    well_samples_text = str(dict(r1.get("outputs") or {}).get("well_ai_samples") or "").strip()
    if not well_samples_text:
        raise ValueError("R1 summary lacks outputs.well_ai_samples.")
    well_samples_path = resolve_relative_path(well_samples_text, root=REPO_ROOT)
    if well_samples_path.parent.resolve() != forward_diagnostic_dir.resolve():
        raise ValueError("R1 well_ai_samples path is outside configured forward_diagnostic_dir.")
    waveform_text = str(dict(r1.get("outputs") or {}).get("well_waveform_samples") or "").strip()
    if not waveform_text:
        raise ValueError("R1 summary lacks outputs.well_waveform_samples.")
    waveform_path = resolve_relative_path(waveform_text, root=REPO_ROOT)
    if waveform_path.parent.resolve() != forward_diagnostic_dir.resolve() or not waveform_path.is_file():
        raise ValueError("R1 well_waveform_samples path is invalid or outside forward_diagnostic_dir.")
    return r1, r0, well_samples_path, waveform_path


def _model_entries(r0_summary: Mapping[str, Any], *, roles: list[str]) -> dict[str, dict[str, Any]]:
    models = r0_summary.get("models")
    if not isinstance(models, list):
        raise ValueError("R0 summary.models must be a list.")
    by_role = {str(item.get("model_role")): dict(item) for item in models if isinstance(item, dict)}
    missing = [role for role in roles if role not in by_role]
    if missing:
        raise ValueError(f"R0 summary missing configured roles: {missing}")
    return {role: by_role[role] for role in roles}


def _validate_model_entry(
    *, role: str, entry: Mapping[str, Any], zero_shot_dir: Path
) -> dict[str, Path]:
    if entry.get("schema_version") != "real_field_zero_shot_model_v1" or entry.get("status") != "ok":
        raise ValueError(f"Invalid R0 model entry for role={role}")
    if entry.get("output_mode") != "volume":
        raise ValueError(f"R0 model role={role} is not a volume run.")
    checkpoint = resolve_relative_path(_required_text(entry, "checkpoint", label=role), root=REPO_ROOT)
    expected_checkpoint_hash = _required_text(entry, "checkpoint_sha256", label=role)
    if sha256_file(checkpoint) != expected_checkpoint_hash:
        raise ValueError(f"Checkpoint hash mismatch for role={role}: {checkpoint}")
    normalization = resolve_relative_path(
        _required_text(entry, "normalization_file", label=role), root=REPO_ROOT
    )
    expected_normalization_hash = _required_text(entry, "normalization_file_sha256", label=role)
    if sha256_file(normalization) != expected_normalization_hash:
        raise ValueError(f"Normalization hash mismatch for role={role}: {normalization}")
    outputs = dict(entry.get("outputs") or {})
    predictions = resolve_relative_path(_required_text(outputs, "predictions", label=f"{role}.outputs"), root=REPO_ROOT)
    prediction_index = resolve_relative_path(
        _required_text(outputs, "prediction_index", label=f"{role}.outputs"), root=REPO_ROOT
    )
    if predictions.parent.parent.resolve() != zero_shot_dir.resolve():
        raise ValueError(f"R0 predictions for role={role} are outside configured zero_shot_dir.")
    expected_prediction_hash = _required_text(entry, "output_prediction_sha256", label=role)
    if sha256_file(predictions) != expected_prediction_hash:
        raise ValueError(f"Prediction NPZ hash mismatch for role={role}: {predictions}")
    model_run_dir = resolve_relative_path(_required_text(entry, "model_run_dir", label=role), root=REPO_ROOT)
    manifest_path = model_run_dir / "model_run_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "ginn_v2_model_run_v1":
        raise ValueError(f"Unsupported model manifest for role={role}: {manifest_path}")
    if str(manifest.get("model_role") or "") != role:
        raise ValueError(f"Model manifest role mismatch: expected={role}, actual={manifest.get('model_role')}")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalization_payload = _load_json(normalization)
    if normalization_payload != dict(checkpoint_payload["normalization"]):
        raise ValueError(f"Normalization JSON differs from checkpoint for role={role}.")
    gate = dict(entry.get("synthetic_gate_evidence") or {})
    gate_card = resolve_relative_path(_required_text(gate, "report_card", label=f"{role}.synthetic_gate"), root=REPO_ROOT)
    if sha256_file(gate_card) != _required_text(gate, "report_card_sha256", label=f"{role}.synthetic_gate"):
        raise ValueError(f"Synthetic gate report-card hash mismatch for role={role}: {gate_card}")
    return {
        "checkpoint": checkpoint,
        "normalization": normalization,
        "predictions": predictions,
        "prediction_index": prediction_index,
        "model_run_dir": model_run_dir,
        "model_manifest": manifest_path,
        "synthetic_gate_report_card": gate_card,
    }


def _synthetic_prediction_dir(*, model_run_dir: Path, scope: str, checkpoint_hash: str) -> Path:
    if scope not in SYNTHETIC_SCOPE_DIRS:
        raise ValueError(f"Unsupported R2 synthetic scope: {scope}")
    path = model_run_dir / SYNTHETIC_SCOPE_DIRS[scope]
    required = ["prediction_manifest.json", "prediction_index.csv"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Synthetic scope {scope} missing {missing}: {path}")
    manifest = _load_json(path / "prediction_manifest.json")
    if str(manifest.get("checkpoint_sha256") or "") != checkpoint_hash:
        raise ValueError(f"Synthetic scope checkpoint mismatch for {scope}: {path}")
    return path


def _head_module_name(model: torch.nn.Module) -> str:
    output_head = getattr(model, "output_head", None)
    for name, module in model.named_modules():
        if module is output_head:
            if not name:
                raise ValueError("Output head cannot be the model root.")
            return name
    raise ValueError("Cannot resolve output head module name.")


def _save_head_package(
    *,
    path: Path,
    model: torch.nn.Module,
    anchor: HeadState,
    fit: HeadFit | None,
    model_role: str,
    head_id: str,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    eligible_for_r3: bool,
    selection_status: str,
) -> None:
    output_head = getattr(model, "output_head")
    module_name = _head_module_name(model)
    state = fit.head if fit is not None else anchor
    head_state_dict = {
        f"{module_name}.weight": torch.as_tensor(state.weight, dtype=output_head.weight.dtype).reshape_as(
            output_head.weight.detach().cpu()
        ),
        f"{module_name}.bias": torch.as_tensor([state.bias], dtype=output_head.bias.dtype),
    }
    package = {
        "schema_version": "r2_real_delta_head_v2",
        "model_role": model_role,
        "head_id": head_id,
        "source_checkpoint": repo_relative_path(source_checkpoint, root=REPO_ROOT),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "head_module_name": module_name,
        "adapted": fit is not None,
        "selection_status": selection_status,
        "eligible_for_r3": bool(eligible_for_r3),
        "validation_evidence": "not_unbiased" if head_id == "all_wells" else "held_out_fold",
        "lambda_anchor": fit.lambda_anchor if fit is not None else None,
        "training_clusters": list(fit.training_clusters) if fit is not None else [],
        "training_wells": list(fit.training_wells) if fit is not None else [],
        "feature_mean": fit.feature_mean if fit is not None else np.asarray([], dtype=np.float64),
        "feature_std": fit.feature_std if fit is not None else np.asarray([], dtype=np.float64),
        "active_mask": fit.active_mask if fit is not None else np.asarray([], dtype=bool),
        "beta": fit.beta if fit is not None else np.asarray([], dtype=np.float64),
        "beta_anchor": fit.beta_anchor if fit is not None else np.asarray([], dtype=np.float64),
        "head_state_dict": head_state_dict,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, path)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _head_from_package(package: Mapping[str, Any]) -> HeadState:
    if package.get("schema_version") != "r2_real_delta_head_v2":
        raise ValueError("R2 head package is not target-label schema v2.")
    state_dict = package.get("head_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("R2 head package lacks head_state_dict.")
    weight_values = [value for key, value in state_dict.items() if str(key).endswith(".weight")]
    bias_values = [value for key, value in state_dict.items() if str(key).endswith(".bias")]
    if len(weight_values) != 1 or len(bias_values) != 1:
        raise ValueError("R2 head package must contain exactly one head weight and bias.")
    weight = torch.as_tensor(weight_values[0]).detach().cpu().numpy().reshape(-1)
    bias = float(torch.as_tensor(bias_values[0]).detach().cpu().numpy().reshape(-1)[0])
    return HeadState(weight=weight, bias=bias)


def _plot_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    if int(np.count_nonzero(valid)) < 8:
        return float("nan"), float("nan")
    target_values = target[valid]
    prediction_values = prediction[valid]
    rmse = float(np.sqrt(np.mean((prediction_values - target_values) ** 2)))
    corr = (
        float(np.corrcoef(target_values, prediction_values)[0, 1])
        if np.std(target_values) > 0.0 and np.std(prediction_values) > 0.0
        else float("nan")
    )
    return corr, rmse


def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if starts.size == 0:
        return None
    index = int(np.argmax(stops - starts))
    return int(starts[index]), int(stops[index])


def _plot_r2_all_well_qc(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "r2_real_delta_adapter_summary.json"
    summary = _load_json(summary_path)
    if summary.get("schema_version") != "r2_real_delta_adapter_summary_v2":
        raise ValueError(f"Unsupported R2 summary: {summary_path}")
    roles = [str(value) for value in summary.get("model_roles") or []]
    source_runs = dict(summary.get("source_runs") or {})
    r1_dir = resolve_relative_path(
        _required_text(source_runs, "forward_diagnostic_dir", label="R2 source_runs"), root=REPO_ROOT
    )
    r0_dir = resolve_relative_path(
        _required_text(source_runs, "zero_shot_dir", label="R2 source_runs"), root=REPO_ROOT
    )
    r1_summary = _load_json(r1_dir / "real_field_forward_diagnostic_summary.json")
    if r1_summary.get("schema_version") != "real_field_forward_diagnostic_summary_v2":
        raise ValueError("R2 well QC requires R1 v2 waveform/target contracts")
    r1_outputs = dict(r1_summary.get("outputs") or {})
    well_samples_path = resolve_relative_path(
        _required_text(r1_outputs, "well_ai_samples", label="R1 outputs"), root=REPO_ROOT
    )
    waveform_path = resolve_relative_path(
        _required_text(r1_outputs, "well_waveform_samples", label="R1 outputs"), root=REPO_ROOT
    )
    samples = load_well_samples(well_samples_path, model_roles=roles)
    waveform = pd.read_csv(waveform_path)
    required_waveform = {
        "well_name", "forward_sample_index", "twt_s", "observed_seismic", "observed_valid",
        "tie_window_start_s", "tie_window_end_s",
    }
    missing = sorted(required_waveform - set(waveform.columns))
    if missing:
        raise ValueError(f"well_waveform_samples.csv missing columns: {missing}")
    waveform["forward_sample_index"] = pd.to_numeric(waveform["forward_sample_index"], errors="raise").astype(int)
    waveform["observed_valid"] = waveform["observed_valid"].astype(str).str.casefold().eq("true")
    r0_summary = _load_json(r0_dir / "real_field_zero_shot_summary.json")
    wavelet_dir_text = _required_text(
        dict(r0_summary.get("source_runs") or {}), "wavelet_generation_dir", label="R0 source_runs"
    )
    wavelet, wavelet_meta = load_selected_wavelet(resolve_relative_path(wavelet_dir_text, root=REPO_ROOT))
    if str(wavelet_meta.get("wavelet_sha256") or "") != str(r1_summary.get("wavelet_sha256") or ""):
        raise ValueError("R2 QC wavelet does not match the frozen R1 wavelet")

    figures_dir = run_dir / "figures" / "wells"
    figures_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    generated = 0
    for role in roles:
        role_summary = dict(dict(summary.get("roles") or {}).get(role) or {})
        checkpoint_path = resolve_relative_path(
            _required_text(role_summary, "checkpoint", label=f"R2 roles.{role}"), root=REPO_ROOT
        )
        model, checkpoint = load_checkpoint(checkpoint_path)
        normalization = dict(checkpoint["normalization"])
        role_samples = samples[samples["model_role"].astype(str).eq(role)].reset_index(drop=True)
        feature_path = run_dir / "features" / f"{role}_well_features.npz"
        with np.load(feature_path, allow_pickle=True) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
            feature_wells = np.asarray(arrays["well_name"]).astype(str)
            feature_samples = np.asarray(arrays["sample_index"], dtype=np.int64)
        if features.shape[0] != len(role_samples):
            raise ValueError(f"R2 feature/sample length mismatch for role={role}.")
        if not np.array_equal(feature_wells, role_samples["well_name"].astype(str).to_numpy()):
            raise ValueError(f"R2 feature well order mismatch for role={role}.")
        if not np.array_equal(feature_samples, role_samples["sample_index"].to_numpy(dtype=np.int64)):
            raise ValueError(f"R2 feature sample order mismatch for role={role}.")
        package_path = run_dir / "heads" / role / "all_wells_head.pt"
        package = torch.load(package_path, map_location="cpu", weights_only=False)
        if not bool(package.get("adapted", False)) or str(role_summary.get("all_wells_status")) != "ok":
            for well_name in sorted(role_samples["well_name"].astype(str).unique()):
                index_rows.append({
                    "well_name": well_name, "model_role": role, "status": "all_well_qc_unavailable",
                    "reason": str(role_summary.get("all_wells_status") or "all_well_head_not_adapted"), "figure": "",
                })
            continue
        all_head = _head_from_package(package)
        pred_delta = denormalize_delta_values(apply_head(features=features, head=all_head), normalization)
        pred_ai = role_samples["lfm_log_ai"].to_numpy(dtype=np.float64) + pred_delta

        for well_name, positions in role_samples.groupby("well_name", sort=True).indices.items():
            pos = np.asarray(positions, dtype=int)
            local = role_samples.iloc[pos].copy()
            order = np.argsort(local["sample_index"].to_numpy(dtype=int))
            pos = pos[order]
            local = role_samples.iloc[pos].reset_index(drop=True)
            local_pred = pred_ai[pos]
            sample_index = local["sample_index"].to_numpy(dtype=int)
            target = local["target_log_ai"].to_numpy(dtype=np.float64)
            twt = local["twt_s"].to_numpy(dtype=np.float64)
            well_waveform = waveform[waveform["well_name"].astype(str).eq(str(well_name))].set_index("forward_sample_index")
            transition_valid = np.diff(sample_index) == 1
            observed_values = np.full(max(len(local) - 1, 0), np.nan, dtype=np.float64)
            observed_ok = np.zeros(observed_values.shape, dtype=bool)
            for index, endpoint in enumerate(sample_index[1:]):
                forward_index = int(endpoint - 1)
                if forward_index not in well_waveform.index:
                    continue
                row = well_waveform.loc[forward_index]
                if isinstance(row, pd.DataFrame):
                    raise ValueError(f"Duplicate R1 waveform key for well={well_name}, index={forward_index}")
                observed_values[index] = float(pd.to_numeric(row["observed_seismic"], errors="coerce"))
                observed_ok[index] = bool(row["observed_valid"]) and np.isfinite(observed_values[index])
            transition_valid &= observed_ok
            transition_valid &= np.isfinite(target[:-1]) & np.isfinite(target[1:])
            transition_valid &= np.isfinite(local_pred[:-1]) & np.isfinite(local_pred[1:])
            if not well_waveform.empty:
                starts = pd.to_numeric(well_waveform["tie_window_start_s"], errors="coerce").dropna()
                ends = pd.to_numeric(well_waveform["tie_window_end_s"], errors="coerce").dropna()
                if not starts.empty and not ends.empty:
                    transition_valid &= (twt[1:] >= float(starts.iloc[0])) & (twt[1:] <= float(ends.iloc[0]))
            run = _largest_true_run(transition_valid)
            if run is None or run[1] - run[0] < 8:
                index_rows.append({
                    "well_name": str(well_name), "model_role": role, "status": "all_well_qc_unavailable",
                    "reason": "insufficient_contiguous_waveform_support", "figure": "",
                })
                continue
            start, stop = run
            ai_slice = slice(start, stop + 1)
            forward_slice = slice(start, stop)
            forward_twt = twt[1:][forward_slice]
            target_values = target[ai_slice]
            pred_values = local_pred[ai_slice]
            synthetic_values = forward_log_ai(pred_values[None, :], wavelet)[0]
            observed_trace = grid.Seismic(observed_values[forward_slice], forward_twt, "twt", name="Seismic")
            synthetic_trace = grid.Seismic(synthetic_values, forward_twt, "twt", name="R2 Synthetic")
            reflectivity = np.tanh(0.5 * np.diff(pred_values))
            reflectivity_trace = grid.Reflectivity(reflectivity, forward_twt, "twt", name="R2 Reflectivity")
            target_trace = grid.Log(np.exp(target_values[1:]), forward_twt, "twt", name="Target AI")
            r2_trace = grid.Log(np.exp(pred_values[1:]), forward_twt, "twt", name="R2 all-well AI")
            xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
            xcorr_basis = synthetic_trace.sampling_rate * np.arange(-(synthetic_trace.size - 1), synthetic_trace.size)
            xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
            dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
            corr, rmse = _plot_metrics(target_values, pred_values)
            fig, _axes = plot_well_waveform_qc(
                [target_trace, r2_trace], reflectivity_trace, synthetic_trace, observed_trace,
                xcorr, dxcorr, figsize=(12.0, 7.5), synthetic_ai=r2_trace,
                title=(
                    f"R2 well QC | {well_name} | {role} | all-well/trained-well view; "
                    f"not validation evidence | corr={corr:.3f}, rmse={rmse:.4f}"
                ),
            )
            figure_path = figures_dir / f"r2_well_qc_{well_name}_{role}.png"
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            generated += 1
            index_rows.append({
                "well_name": str(well_name), "model_role": role, "status": "ok",
                "n_ai_samples": int(len(target_values)), "target_corr": corr, "target_rmse": rmse,
                "figure": repo_relative_path(figure_path, root=REPO_ROOT),
            })
    index_path = run_dir / "r2_well_qc_index.csv"
    pd.DataFrame.from_records(index_rows).to_csv(index_path, index=False)
    figure_summary = {
        "schema_version": "r2_well_qc_v2", "status": "ok", "n_figures": generated,
        "prediction_semantics": "all-well/trained-well view; not validation evidence",
        "index": repo_relative_path(index_path, root=REPO_ROOT),
        "figures_dir": repo_relative_path(figures_dir, root=REPO_ROOT),
    }
    write_json(run_dir / "r2_well_qc_summary.json", figure_summary)
    summary_outputs = dict(summary.get("outputs") or {})
    summary_outputs["r2_well_qc_index"] = repo_relative_path(index_path, root=REPO_ROOT)
    summary_outputs["r2_well_qc_figures"] = repo_relative_path(figures_dir, root=REPO_ROOT)
    summary["outputs"] = summary_outputs
    write_json(summary_path, summary)
    return figure_summary


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    if args.plot_run_dir is not None:
        run_dir = resolve_relative_path(args.plot_run_dir, root=REPO_ROOT)
        result = _plot_r2_all_well_qc(run_dir)
        print("=== R2 Well QC Figures ===")
        print(f"Run: {run_dir}")
        print(f"Figures: {result['n_figures']}")
        return
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    config = load_yaml_config(config_path)
    run_cfg = dict(config.get("r2_real_delta_adapter") or {})
    output_root = resolve_relative_path(config.get("output_root", "scripts/output"), root=REPO_ROOT)
    forward_text = args.forward_diagnostic_dir or Path(
        _required_text(run_cfg, "forward_diagnostic_dir", label="r2_real_delta_adapter")
    )
    zero_text = args.zero_shot_dir or Path(_required_text(run_cfg, "zero_shot_dir", label="r2_real_delta_adapter"))
    forward_dir = resolve_relative_path(forward_text, root=REPO_ROOT)
    zero_dir = resolve_relative_path(zero_text, root=REPO_ROOT)
    output_dir = _output_dir(args.output_dir, output_root=output_root)
    if output_dir.exists():
        raise FileExistsError(f"R2 output directory already exists: {output_dir}")
    roles_value = run_cfg.get("model_roles")
    if not isinstance(roles_value, list):
        raise ValueError("r2_real_delta_adapter.model_roles must be an explicit list.")
    roles = [str(value) for value in roles_value]
    if set(roles) != {"lateral", "no_lateral"} or len(roles) != 2:
        raise ValueError("r2_real_delta_adapter.model_roles must contain lateral and no_lateral exactly once.")
    device_name = str(args.device or _required_text(run_cfg, "device", label="r2_real_delta_adapter"))
    lambda_value = run_cfg.get("lambda_candidates")
    if not isinstance(lambda_value, list):
        raise ValueError("r2_real_delta_adapter.lambda_candidates must be an explicit list.")
    lambda_candidates = [float(value) for value in lambda_value]
    if not lambda_candidates or any(not np.isfinite(value) or value <= 0.0 for value in lambda_candidates):
        raise ValueError("r2_real_delta_adapter.lambda_candidates must contain positive finite values.")
    thresholds_value = run_cfg.get("thresholds")
    if not isinstance(thresholds_value, dict):
        raise ValueError("r2_real_delta_adapter.thresholds must be an explicit mapping.")
    thresholds = dict(thresholds_value)
    minimum_clusters = int(_required_value(thresholds, "minimum_loco_clusters_for_decision", label="thresholds"))
    improvement_fraction = float(
        _required_value(thresholds, "minimum_loco_cluster_improvement_fraction", label="thresholds")
    )
    minimum_median_gain = float(_required_value(thresholds, "minimum_median_delta_corr_gain", label="thresholds"))
    maximum_synthetic_error = float(
        _required_value(thresholds, "maximum_synthetic_error_relative_increase", label="thresholds")
    )
    maximum_synthetic_corr = float(
        _required_value(thresholds, "maximum_synthetic_corr_decrease", label="thresholds")
    )
    reconstruction_tolerance = float(
        _required_value(run_cfg, "feature_reconstruction_tolerance_log_ai", label="r2_real_delta_adapter")
    )
    scopes_value = run_cfg.get("synthetic_scopes")
    if not isinstance(scopes_value, list):
        raise ValueError("r2_real_delta_adapter.synthetic_scopes must be an explicit list.")
    synthetic_scopes = [str(value) for value in scopes_value]
    synthetic_batch_size = int(_required_value(run_cfg, "synthetic_batch_size", label="r2_real_delta_adapter"))
    diagnostic_max_hz = float(_required_value(run_cfg, "diagnostic_max_hz", label="r2_real_delta_adapter"))
    if minimum_clusters < 2:
        raise ValueError("minimum_loco_clusters_for_decision must be >= 2.")
    if not 0.0 < improvement_fraction <= 1.0:
        raise ValueError("minimum_loco_cluster_improvement_fraction must be in (0, 1].")
    if not np.isfinite(minimum_median_gain):
        raise ValueError("minimum_median_delta_corr_gain must be finite.")
    if maximum_synthetic_error < 0.0 or maximum_synthetic_corr < 0.0:
        raise ValueError("Synthetic preservation tolerances must be non-negative.")
    if reconstruction_tolerance <= 0.0 or synthetic_batch_size <= 0 or diagnostic_max_hz <= 0.0:
        raise ValueError("R2 tolerance, synthetic batch size, and diagnostic max frequency must be positive.")
    if len(synthetic_scopes) != len(set(synthetic_scopes)) or set(synthetic_scopes) != set(SYNTHETIC_SCOPE_DIRS):
        raise ValueError(f"synthetic_scopes must contain exactly {sorted(SYNTHETIC_SCOPE_DIRS)}.")
    r1_summary, r0_summary, well_samples_path, waveform_samples_path = _validate_sources(
        forward_diagnostic_dir=forward_dir,
        zero_shot_dir=zero_dir,
    )
    entries = _model_entries(r0_summary, roles=roles)
    samples = load_well_samples(well_samples_path, model_roles=roles)
    output_dir.mkdir(parents=True, exist_ok=False)
    features_dir = output_dir / "features"
    heads_dir = output_dir / "heads"
    features_dir.mkdir()
    heads_dir.mkdir()

    fold_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    parameter_frames: list[pd.DataFrame] = []
    reconstruction_frames: list[pd.DataFrame] = []
    band_frames: list[pd.DataFrame] = []
    synthetic_frames: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    role_payloads: dict[str, dict[str, Any]] = {}

    for role in roles:
        paths = _validate_model_entry(role=role, entry=entries[role], zero_shot_dir=zero_dir)
        role_samples = samples[samples["model_role"].astype(str).eq(role)].reset_index(drop=True)
        features, reconstruction_detail, anchor, feature_meta = extract_role_well_features(
            samples=role_samples,
            predictions_path=paths["predictions"],
            prediction_index_path=paths["prediction_index"],
            checkpoint_path=paths["checkpoint"],
            device_name=device_name,
            reconstruction_tolerance_log_ai=reconstruction_tolerance,
        )
        feature_path = features_dir / f"{role}_well_features.npz"
        feature_detail_path = features_dir / f"{role}_feature_reconstruction_detail.csv"
        np.savez_compressed(
            feature_path,
            features=features.astype(np.float32),
            well_name=np.asarray(role_samples["well_name"].astype(str).tolist(), dtype=np.str_),
            sample_index=role_samples["sample_index"].to_numpy(dtype=np.int64),
        )
        reconstruction_detail.to_csv(feature_detail_path, index=False)
        write_json(
            features_dir / f"{role}_feature_manifest.json",
            _json_clean(
                {
                    "schema_version": "r2_well_feature_cache_v2",
                    "model_role": role,
                    "source_checkpoint": repo_relative_path(paths["checkpoint"], root=REPO_ROOT),
                    "source_checkpoint_sha256": sha256_file(paths["checkpoint"]),
                    "source_predictions": repo_relative_path(paths["predictions"], root=REPO_ROOT),
                    "source_predictions_sha256": sha256_file(paths["predictions"]),
                    "source_well_ai_samples": repo_relative_path(well_samples_path, root=REPO_ROOT),
                    "source_well_ai_samples_sha256": sha256_file(well_samples_path),
                    "features": repo_relative_path(feature_path, root=REPO_ROOT),
                    "features_sha256": sha256_file(feature_path),
                    "reconstruction_detail": repo_relative_path(feature_detail_path, root=REPO_ROOT),
                    "reconstruction_detail_sha256": sha256_file(feature_detail_path),
                    "metadata": feature_meta,
                }
            ),
        )
        reconstruction_frames.append(feature_reconstruction_summary(reconstruction_detail))
        _, checkpoint = load_checkpoint(paths["checkpoint"])
        normalization = dict(checkpoint["normalization"])
        validation = run_role_validation(
            frame=role_samples,
            features=features,
            anchor=anchor,
            normalization=normalization,
            lambda_candidates=lambda_candidates,
            model_role=role,
        )
        fold_frames.append(validation.fold_metrics)
        cluster_frames.append(validation.cluster_metrics)
        lambda_frames.append(validation.lambda_selection)
        parameter_frames.append(validation.head_parameters)
        with np.load(paths["predictions"], allow_pickle=False) as arrays:
            twt_s = np.asarray(arrays["twt_s"], dtype=np.float64)
        dt_s = float(np.median(np.diff(twt_s)))
        band_frames.append(
            build_band_metrics(
                fold_metrics=validation.fold_metrics,
                frame=role_samples,
                features=features,
                heads=validation.heads,
                anchor=anchor,
                normalization=normalization,
                dt_s=dt_s,
                diagnostic_max_hz=diagnostic_max_hz,
                model_role=role,
            )
        )
        checkpoint_hash = sha256_file(paths["checkpoint"])
        synthetic_dirs = {
            scope: _synthetic_prediction_dir(
                model_run_dir=paths["model_run_dir"],
                scope=scope,
                checkpoint_hash=checkpoint_hash,
            )
            for scope in synthetic_scopes
        }
        formal_heads = {
            head_id: fit
            for head_id, fit in validation.heads.items()
            if head_id.startswith("outer_loco_cluster_") or head_id == "all_wells"
        }
        synthetic_heads = {"r0": anchor} | {
            head_id: fit.head for head_id, fit in formal_heads.items()
        }
        for scope, directory in synthetic_dirs.items():
            scope_metrics = evaluate_synthetic_heads(
                    checkpoint_path=paths["checkpoint"],
                    prediction_dir=directory,
                    root=REPO_ROOT,
                    heads=synthetic_heads,
                    device_name=device_name,
                    batch_size=synthetic_batch_size,
            )
            for head_id in formal_heads:
                synthetic_frames.append(
                    compare_synthetic_metrics(
                        model_role=role,
                        head_id=head_id,
                        scope=scope,
                        r0_metrics=scope_metrics["r0"],
                        r2_metrics=scope_metrics[head_id],
                        maximum_error_relative_increase=maximum_synthetic_error,
                        maximum_corr_decrease=maximum_synthetic_corr,
                    )
                )
        role_synthetic = _concat([frame for frame in synthetic_frames if not frame.empty])
        if not role_synthetic.empty:
            role_synthetic = role_synthetic[role_synthetic["model_role"].astype(str).eq(role)].copy()
        if not validation.cluster_metrics.empty:
            synthetic_status_by_head = {
                head_id: (
                    "ok"
                    if not role_synthetic[role_synthetic["head_id"].astype(str).eq(head_id)].empty
                    and role_synthetic[role_synthetic["head_id"].astype(str).eq(head_id)]["status"]
                    .astype(str)
                    .eq("ok")
                    .all()
                    else "synthetic_preservation_failed"
                )
                for head_id in formal_heads
            }
            validation.cluster_metrics["synthetic_preservation_status"] = validation.cluster_metrics.apply(
                lambda row: synthetic_status_by_head.get(
                    str(row["fold_id"]),
                    "not_evaluated_diagnostic" if str(row["split_type"]) == "leave_one_well_out" else "missing",
                ),
                axis=1,
            )
        decision = build_role_decision(
            model_role=role,
            validation=validation,
            synthetic_preservation=role_synthetic,
            minimum_loco_clusters_for_decision=minimum_clusters,
            minimum_loco_cluster_improvement_fraction=improvement_fraction,
            minimum_median_delta_corr_gain=minimum_median_gain,
        )
        decisions.append(decision)
        model, _ = load_checkpoint(paths["checkpoint"])
        for head_id, fit in validation.heads.items():
            file_name = "all_wells_head.pt" if head_id == "all_wells" else f"{head_id}.pt"
            _save_head_package(
                path=heads_dir / role / file_name,
                model=model,
                anchor=anchor,
                fit=fit,
                model_role=role,
                head_id=head_id,
                source_checkpoint=paths["checkpoint"],
                source_checkpoint_sha256=checkpoint_hash,
                eligible_for_r3=bool(decision["eligible_for_r3"] and head_id == "all_wells"),
                selection_status=str(decision["decision"]) if head_id == "all_wells" else "ok",
            )
        if "all_wells" not in validation.heads:
            _save_head_package(
                path=heads_dir / role / "all_wells_head.pt",
                model=model,
                anchor=anchor,
                fit=None,
                model_role=role,
                head_id="all_wells",
                source_checkpoint=paths["checkpoint"],
                source_checkpoint_sha256=checkpoint_hash,
                eligible_for_r3=False,
                selection_status=validation.all_wells_status,
            )
        role_payloads[role] = {
            "feature_extraction": feature_meta,
            "outer_status": validation.outer_status,
            "all_wells_status": validation.all_wells_status,
            "decision": decision,
            "checkpoint": repo_relative_path(paths["checkpoint"], root=REPO_ROOT),
            "checkpoint_sha256": checkpoint_hash,
            "all_wells_head": repo_relative_path(heads_dir / role / "all_wells_head.pt", root=REPO_ROOT),
        }

    outputs = {
        "r2_fold_metrics": output_dir / "r2_fold_metrics.csv",
        "r2_cluster_metrics": output_dir / "r2_cluster_metrics.csv",
        "r2_lambda_selection": output_dir / "r2_lambda_selection.csv",
        "r2_head_parameters": output_dir / "r2_head_parameters.csv",
        "r2_synthetic_preservation": output_dir / "r2_synthetic_preservation.csv",
        "r2_band_metrics": output_dir / "r2_band_metrics.csv",
        "r2_feature_reconstruction": output_dir / "r2_feature_reconstruction.csv",
        "r2_decision_table": output_dir / "r2_decision_table.csv",
    }
    _concat(fold_frames).to_csv(outputs["r2_fold_metrics"], index=False)
    _concat(cluster_frames).to_csv(outputs["r2_cluster_metrics"], index=False)
    _concat(lambda_frames).to_csv(outputs["r2_lambda_selection"], index=False)
    _concat(parameter_frames).to_csv(outputs["r2_head_parameters"], index=False)
    _concat(synthetic_frames).to_csv(outputs["r2_synthetic_preservation"], index=False)
    _concat(band_frames).to_csv(outputs["r2_band_metrics"], index=False)
    _concat(reconstruction_frames).to_csv(outputs["r2_feature_reconstruction"], index=False)
    decision_frame = pd.DataFrame.from_records(decisions)
    decision_frame.to_csv(outputs["r2_decision_table"], index=False)
    eligible = decision_frame[decision_frame["eligible_for_r3"].astype(bool)].copy()
    if eligible.empty:
        recommended = (
            "r2_inconclusive"
            if decision_frame["decision"].astype(str).str.startswith("inconclusive").any()
            else "no_role_passed_r2"
        )
        selected_role = ""
    else:
        eligible = eligible.sort_values(
            [
                "median_delta_corr_gain",
                "median_full_ai_rmse_delta",
            ],
            ascending=[False, True],
        )
        recommended = "r3_candidate_available"
        selected_role = str(eligible.iloc[0]["model_role"])
    summary = {
        "schema_version": "r2_real_delta_adapter_summary_v2",
        "status": "ok",
        "config_file": repo_relative_path(config_path, root=REPO_ROOT),
        "source_runs": {
            "forward_diagnostic_dir": repo_relative_path(forward_dir, root=REPO_ROOT),
            "zero_shot_dir": repo_relative_path(zero_dir, root=REPO_ROOT),
        },
        "source_hashes": {
            "r1_summary": sha256_file(forward_dir / "real_field_forward_diagnostic_summary.json"),
            "r0_summary": sha256_file(zero_dir / "real_field_zero_shot_summary.json"),
            "well_ai_samples": sha256_file(well_samples_path),
            "well_waveform_samples": sha256_file(waveform_samples_path),
        },
        "conditional_validation_scope": (
            "adapter labels are held out; upstream LFM, R0 inputs, and synthetic calibration remain frozen"
        ),
        "model_roles": roles,
        "lambda_candidates": lambda_candidates,
        "thresholds": thresholds,
        "synthetic_scopes": synthetic_scopes,
        "synthetic_evidence_gap": ["probe", "paired_increment", "zero_x_false_energy"],
        "roles": role_payloads,
        "selected_role_for_r3": selected_role,
        "recommended_next_state": recommended,
        "outputs": {
            key: repo_relative_path(path, root=REPO_ROOT) for key, path in outputs.items()
        }
        | {
            "heads": repo_relative_path(heads_dir, root=REPO_ROOT),
            "features": repo_relative_path(features_dir, root=REPO_ROOT),
        },
        "code_version_or_git_commit": _git_commit(),
    }
    write_json(output_dir / "r2_real_delta_adapter_summary.json", _json_clean(summary))
    well_qc = _plot_r2_all_well_qc(output_dir)
    print("=== R2 Real-Delta Adapter Validation ===")
    print(f"Output: {output_dir}")
    print(f"Decision: {recommended}")
    if selected_role:
        print(f"R3 candidate role: {selected_role}")
    print(f"Well QC figures: {well_qc['n_figures']}")


if __name__ == "__main__":
    main()
