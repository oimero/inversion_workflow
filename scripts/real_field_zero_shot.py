"""Run R0 real-field zero-shot prediction for frozen GINN-v2 candidates.

Usage::

    python scripts/real_field_zero_shot.py
    python scripts/real_field_zero_shot.py --config experiments/common/common.yaml
    python scripts/real_field_zero_shot.py --device cuda
    python scripts/real_field_zero_shot.py --stitch-strategy center_crop
    python scripts/real_field_zero_shot.py --output-dir scripts/output/real_field_zero_shot_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
src_text = str(SRC_DIR)
sys.path = [path for path in sys.path if path != src_text]
sys.path.insert(0, src_text)

from cup.config.sources import resolve_source_run
from cup.seismic.lfm.artifacts import resolve_lfm_variant
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.well.real_field_controls import load_well_control_set
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    load_yaml_config,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    write_json,
)
from cup.seismic.volume_export import export_volume_like_source, log_ai_to_ai_volume
from ginn_v2.contracts import ZERO_SHOT_SUMMARY_SCHEMA_VERSION
from ginn_v2.real_field import (
    input_qc_frame,
    load_real_field_section,
    load_real_field_volume,
    run_zero_shot_model,
    run_zero_shot_volume_model,
)


DEFAULT_COMMON_CONFIG = Path("experiments/common/common.yaml")
DEFAULT_REAL_FIELD_SECTIONS = Path("experiments/common/real_field_sections.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Torch device. Use 'cuda' to fail if GPU is unavailable.")
    parser.add_argument("--stitch-strategy", choices=("uniform", "center_crop"), default=None)
    return parser.parse_args()


def _load_models(run_cfg: dict) -> list[dict]:
    if run_cfg.get("model_set_file") is not None:
        raise ValueError("real_field_zero_shot.model_set_file is retired; put models in common.yaml.")
    models = run_cfg.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("real_field_zero_shot.models must be a non-empty list.")
    clean = []
    for idx, item in enumerate(models):
        if not isinstance(item, dict):
            raise ValueError(f"real_field_zero_shot.models[{idx}] must be a mapping.")
        extra = sorted(set(item) - {"model_run_dir"})
        if extra:
            raise ValueError(f"real_field_zero_shot.models[{idx}] only accepts model_run_dir; got {extra}.")
        text = str(item.get("model_run_dir") or "").strip()
        if not text:
            raise ValueError(f"real_field_zero_shot.models[{idx}].model_run_dir must be non-empty.")
        clean.append({"model_run_dir": text})
    return clean


def _load_sections_file(run_cfg: dict) -> list[dict]:
    if run_cfg.get("section") is not None:
        raise ValueError("real_field_zero_shot.section is retired; use sections_file.")
    sections_file = str(run_cfg.get("sections_file") or DEFAULT_REAL_FIELD_SECTIONS).strip()
    payload = load_yaml_config(resolve_relative_path(sections_file, root=REPO_ROOT))
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError(f"{sections_file} must contain a non-empty sections list.")
    return [dict(item) for item in sections]


def _prepare_run_config(run_cfg: dict, cfg: dict) -> dict:
    prepared = dict(run_cfg)
    inputs = dict(prepared.get("real_field_inputs") or {})
    seismic = dict(cfg.get("seismic") or {})
    if any(key in inputs for key in ("seismic_file", "seismic_type", "type")):
        raise ValueError("real_field_zero_shot.real_field_inputs seismic fields are retired; use top-level seismic.")
    inputs["seismic_file"] = str(seismic.get("file") or "")
    inputs["seismic_type"] = str(seismic.get("type") or "")
    segy_options = {
        key: seismic[key]
        for key in ("iline", "xline", "istep", "xstep", "iline_byte", "xline_byte")
        if seismic.get(key) is not None
    }
    if segy_options:
        inputs["segy_options"] = segy_options
    prepared["real_field_inputs"] = inputs
    if str(prepared.get("mode") or "volume").casefold() == "section":
        sections = _load_sections_file(prepared)
        if len(sections) != 1:
            raise ValueError("R0 section mode requires sections_file with exactly one section.")
        prepared["section"] = sections[0]
    return prepared


def _prepare_real_field_inputs(run_cfg: dict, workflow_cfg: dict) -> dict:
    prepared = dict(run_cfg)
    inputs = dict(prepared.get("real_field_inputs") or {})
    output_root = resolve_relative_path(workflow_cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    if "lfm_file" in inputs:
        raise ValueError("real_field_inputs.lfm_file is retired; select lfm_run_dir + variant_id + well_control_run_dir.")
    selected = resolve_lfm_variant(inputs, repo_root=REPO_ROOT)
    seismic = dict(workflow_cfg.get("seismic") or {})
    selected_domain = str(selected.variant_metadata.get("sample_domain") or "")
    selected_basis = selected.variant_metadata.get("depth_basis")
    if str(seismic.get("domain") or "").casefold() != selected_domain:
        raise ValueError("Selected LFM variant sample_domain does not match top-level seismic.domain.")
    if seismic.get("depth_basis") != selected_basis:
        raise ValueError("Selected LFM variant depth_basis does not match top-level seismic.depth_basis.")
    data_root = resolve_relative_path(workflow_cfg.get("data_root", "data"), root=REPO_ROOT)
    current_seismic = resolve_relative_path(str(seismic.get("file") or ""), root=data_root)
    recorded_seismic_path = resolve_relative_path(
        str(selected.variant_metadata.get("seismic_path") or ""), root=REPO_ROOT
    )
    if current_seismic.resolve() != recorded_seismic_path.resolve():
        raise ValueError("Selected LFM variant seismic path does not match top-level seismic.file.")
    transform = str(inputs.get("lfm_value_transform") or "identity").casefold()
    if transform not in {"identity", "none"}:
        raise ValueError("Unified LFM v3 is already log(AI); lfm_value_transform must be identity.")
    if str(inputs.get("target_mask_file") or "").strip():
        raise ValueError("Unified LFM v3 owns valid_mask_model; external target_mask_file is not accepted.")
    inputs["lfm_file"] = repo_relative_path(selected.lfm_path, root=REPO_ROOT)
    inputs["lfm_value_transform"] = "identity"
    inputs["selected_lfm_contract_fingerprint_sha256"] = selected.contract_fingerprint_sha256
    inputs["selected_lfm_contract_path"] = repo_relative_path(
        selected.variant_dir / "variant_summary.json", root=REPO_ROOT
    )
    inputs["well_control_contract_fingerprint_sha256"] = (
        selected.well_control_contract_fingerprint_sha256
    )
    inputs["well_control_contract_path"] = repo_relative_path(
        selected.well_control_run_dir / "run_summary.json", root=REPO_ROOT
    )
    inputs["selected_lfm_metadata"] = dict(selected.variant_metadata)
    source_runs = dict(prepared.get("source_runs") or {})
    source_runs["lfm_run_dir"] = repo_relative_path(selected.run_dir, root=REPO_ROOT)
    source_runs["variant_id"] = selected.variant_id
    source_runs["well_control_run_dir"] = repo_relative_path(selected.well_control_run_dir, root=REPO_ROOT)
    wavelet_dir = resolve_source_run(
        source_runs.get("wavelet_generation_dir"),
        output_root=output_root,
        prefix="wavelet_generation",
        required_files=["selected_wavelet.csv", "selected_wavelet_summary.json"],
        root=REPO_ROOT,
        label="wavelet_generation",
    )
    source_runs["wavelet_generation_dir"] = repo_relative_path(wavelet_dir, root=REPO_ROOT)
    prepared["real_field_inputs"] = inputs
    prepared["source_runs"] = source_runs
    return prepared

def _resolve_output_dir(prefix: str, explicit: Path | None, *, output_root: Path) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{prefix}_{timestamp}"


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


def _load_seismic_reference_payload(*, run_cfg: dict, models: list[dict]) -> dict[str, object]:
    inputs = dict(run_cfg.get("real_field_inputs") or {})
    explicit = str(inputs.get("seismic_reference_stats_file") or "").strip()
    if explicit:
        stats_path = resolve_relative_path(explicit, root=REPO_ROOT)
    else:
        first_model = dict(models[0])
        model_run_dir = resolve_relative_path(str(first_model["model_run_dir"]), root=REPO_ROOT)
        manifest_path = model_run_dir / "model_run_manifest.json"
        stats_path = model_run_dir / "input_reference_stats.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest_stats = str(manifest.get("input_reference_stats") or "").strip()
            if manifest_stats:
                stats_path = resolve_relative_path(manifest_stats, root=REPO_ROOT)
    if not stats_path.is_file():
        raise FileNotFoundError(
            "seismic_value_transform requires frozen input reference stats. "
            f"Expected {stats_path}. Re-run the training command with the current "
            "scripts/ginn_v2.py so input_reference_stats.json is produced next to "
            "normalization.json and the best/final checkpoints."
        )
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"Input reference stats file lacks 'stats' object: {stats_path}")
    return {
        "stats": stats,
        "sampling": payload.get("sampling", {}),
        "file": repo_relative_path(stats_path, root=REPO_ROOT),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _write_well_prediction_qc(output_dir: Path, *, run_cfg: dict) -> dict[str, object]:
    cfg = dict(run_cfg.get("well_qc") or {})
    if not bool(cfg.get("enabled", True)):
        return {"csv": "", "figures": {}, "status": "disabled"}
    inputs = dict(run_cfg.get("real_field_inputs") or {})
    selected = resolve_lfm_variant(inputs, repo_root=REPO_ROOT)
    controls = load_well_control_set(selected.well_control_run_dir, repo_root=REPO_ROOT)
    model_arrays = {}
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and (child / "predictions.npz").is_file():
            model_arrays[child.name] = np.load(child / "predictions.npz", allow_pickle=True)
    if not model_arrays:
        return {"csv": "", "figures": {}, "status": "missing_predictions"}
    first = next(iter(model_arrays.values()))
    ilines = np.asarray(first["ilines"], dtype=np.float64)
    xlines = np.asarray(first["xlines"], dtype=np.float64)
    samples = np.asarray(first["twt_s"], dtype=np.float64)
    seismic_info = dict(selected.run_summary.get("seismic") or {})
    seismic_path = resolve_relative_path(str(seismic_info.get("path") or ""), root=REPO_ROOT)
    seismic_type = str(seismic_info.get("type") or "").casefold()
    survey = open_survey(
        seismic_path,
        seismic_type,
        segy_options=(segy_options_from_config(dict(inputs.get("segy_options") or {})) if seismic_type == "segy" else None),
    )
    section_xy = np.asarray(
        [survey.line_geometry.line_to_coord(float(il), float(xl)) for il, xl in zip(ilines, xlines)],
        dtype=np.float64,
    )
    if "max_line_distance" in cfg:
        raise ValueError("real_field_zero_shot.well_qc.max_line_distance is retired; use max_xy_distance_m.")
    if cfg.get("max_xy_distance_m") is None:
        raise ValueError("real_field_zero_shot.well_qc.max_xy_distance_m must be explicit.")
    max_distance = float(cfg.get("max_xy_distance_m"))
    rows: list[dict[str, object]] = []
    figure_outputs: dict[str, str] = {}
    figure_dir = output_dir / "figures" / "wells"
    figure_dir.mkdir(parents=True, exist_ok=True)
    include_deviated = bool(cfg.get("include_deviated_wells", True))
    for control in controls.controls:
        well = control.well_name
        if control.wellbore_class == "deviated" and not include_deviated:
            rows.append({"well_name": well, "model_role": "", "status": "skipped_deviated_well"})
            continue
        source_samples = control.sample_axis.values
        sample_indices = np.searchsorted(source_samples, samples)
        sample_indices = np.clip(sample_indices, 0, source_samples.size - 1)
        inside_axis = np.isclose(source_samples[sample_indices], samples, rtol=0.0, atol=1e-8)
        well_log_ai = np.where(inside_axis, control.log_ai.values[sample_indices], np.nan)
        well_x = np.where(inside_axis, control.x_m_by_sample[sample_indices], np.nan)
        well_y = np.where(inside_axis, control.y_m_by_sample[sample_indices], np.nan)
        distances = np.linalg.norm(np.column_stack([well_x, well_y])[:, None, :] - section_xy[None, :, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        trace_indices = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(samples.size), trace_indices]
        finite_distance = nearest_distance[np.isfinite(nearest_distance)]
        if finite_distance.size == 0:
            rows.append({"well_name": well, "model_role": "", "status": "skipped_missing_well_position"})
            continue
        base = {
            "well_name": well,
            "wellbore_class": control.wellbore_class,
            "well_position_source": control.sampling_mode,
            "nearest_section_distance_median_m": float(np.median(finite_distance)),
            "nearest_section_distance_max_m": float(np.max(finite_distance)),
        }
        representative_index = int(np.nanargmin(nearest_distance))
        trace_idx = int(trace_indices[representative_index])
        base.update(
            {
                "nearest_section_trace": trace_idx,
                "nearest_section_inline": float(ilines[trace_idx]),
                "nearest_section_xline": float(xlines[trace_idx]),
            }
        )
        if not np.any(nearest_distance <= max_distance):
            rows.append({**base, "model_role": "", "status": "skipped_outside_section_support", "max_xy_distance_m": max_distance})
            continue
        fig, ax = plt.subplots(figsize=(4.5, 6.0))
        ax.plot(well_log_ai, samples, label="canonical well log(AI)", color="black", linewidth=1.5)
        for role, arrays in model_arrays.items():
            predictions = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)
            pred = predictions[trace_indices, np.arange(samples.size)]
            prediction_mask = np.asarray(arrays["valid_mask_model"], dtype=bool)[trace_indices, np.arange(samples.size)]
            stitching = np.asarray(arrays["stitching_weight"], dtype=np.float64)[trace_indices, np.arange(samples.size)]
            valid = (
                inside_axis
                & (nearest_distance <= max_distance)
                & prediction_mask
                & (stitching > 0.0)
                & np.isfinite(well_log_ai)
                & np.isfinite(pred)
            )
            n_valid = int(np.count_nonzero(valid))
            residual = pred[valid] - well_log_ai[valid] if n_valid else np.asarray([])
            rows.append(
                {
                    **base,
                    "model_role": role,
                    "status": "ok" if n_valid >= 8 else "insufficient_valid_samples",
                    "n_valid": n_valid,
                    "rmse": float(np.sqrt(np.mean(residual * residual))) if n_valid else float("nan"),
                    "bias": float(np.mean(residual)) if n_valid else float("nan"),
                    "corr": _safe_corr(well_log_ai[valid], pred[valid]) if n_valid else float("nan"),
                    "pred_mean": float(np.mean(pred[valid])) if n_valid else float("nan"),
                    "well_log_ai_mean": float(np.mean(well_log_ai[valid])) if n_valid else float("nan"),
                }
            )
            ax.plot(pred, samples, label=role, linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xlabel("log(AI)")
        ax.set_ylabel(f"{controls.sample_domain} ({controls.sample_unit})")
        ax.set_title(f"R0 well prediction QC: {well}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = figure_dir / f"r0_well_prediction_qc_{well}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_outputs[well] = repo_relative_path(path, root=REPO_ROOT)
    csv_path = output_dir / "well_prediction_qc.csv"
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    return {
        "csv": repo_relative_path(csv_path, root=REPO_ROOT),
        "figures": figure_outputs,
        "status": "ok",
    }


def _plot_zero_shot_qc(output_dir: Path) -> dict[str, str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    model_arrays = {}
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir() or not (child / "predictions.npz").is_file():
            continue
        arrays = np.load(child / "predictions.npz", allow_pickle=True)
        model_arrays[child.name] = arrays
        lfm = arrays["lfm_input"]
        pred = arrays["stitched_pred_log_ai"]
        delta = arrays["pred_delta_vs_lfm"]
        if pred.ndim == 3:
            il_mid = pred.shape[0] // 2
            xl_mid = pred.shape[1] // 2
            twt_mid = pred.shape[2] // 2
            panel_sets = [
                ("central_inline", [lfm[il_mid], delta[il_mid], pred[il_mid]], "xline"),
                ("central_xline", [lfm[:, xl_mid, :], delta[:, xl_mid, :], pred[:, xl_mid, :]], "inline"),
                ("central_time_slice", [lfm[:, :, twt_mid], delta[:, :, twt_mid], pred[:, :, twt_mid]], "xline"),
            ]
            for suffix, panels, xlabel in panel_sets:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
                for ax, values, title in [
                    (axes[0], panels[0], "LFM log(AI)"),
                    (axes[1], panels[1], "Pred - LFM"),
                    (axes[2], panels[2], "Pred log(AI)"),
                ]:
                    image = ax.imshow(np.asarray(values).T, aspect="auto", origin="upper", cmap="viridis")
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                axes[0].set_ylabel("TWT sample" if suffix != "central_time_slice" else "inline")
                fig.suptitle(f"R0 volume QC {suffix}: {child.name}")
                fig.tight_layout()
                path = figures / f"{child.name}_{suffix}_lfm_delta_pred.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                outputs[f"{child.name}_{suffix}_triptych"] = repo_relative_path(path, root=REPO_ROOT)
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, values, title in [
            (axes[0], lfm, "LFM log(AI)"),
            (axes[1], delta, "Pred - LFM"),
            (axes[2], pred, "Pred log(AI)"),
        ]:
            image = ax.imshow(values.T, aspect="auto", origin="upper", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("lateral trace")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("TWT sample")
        fig.suptitle(f"R0 zero-shot QC: {child.name}")
        fig.tight_layout()
        path = figures / f"{child.name}_lfm_delta_pred.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[f"{child.name}_triptych"] = repo_relative_path(path, root=REPO_ROOT)
    if "lateral" in model_arrays and "no_lateral" in model_arrays:
        diff = model_arrays["lateral"]["stitched_pred_log_ai"] - model_arrays["no_lateral"]["stitched_pred_log_ai"]
        if diff.ndim == 3:
            il_mid = diff.shape[0] // 2
            xl_mid = diff.shape[1] // 2
            twt_mid = diff.shape[2] // 2
            panels = [
                ("central_inline", diff[il_mid], "xline", "TWT sample"),
                ("central_xline", diff[:, xl_mid, :], "inline", "TWT sample"),
                ("central_time_slice", diff[:, :, twt_mid], "xline", "inline"),
            ]
            for suffix, values, xlabel, ylabel in panels:
                fig, ax = plt.subplots(figsize=(6, 4))
                finite = values[np.isfinite(values)]
                vmax = float(np.nanquantile(np.abs(finite), 0.99)) if finite.size else 1.0
                image = ax.imshow(values.T, aspect="auto", origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_title(f"lateral - no_lateral {suffix}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                path = figures / f"lateral_minus_no_lateral_{suffix}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                outputs[f"lateral_minus_no_lateral_{suffix}"] = repo_relative_path(path, root=REPO_ROOT)
            return outputs
        fig, ax = plt.subplots(figsize=(6, 4))
        vmax = float(np.nanquantile(np.abs(diff), 0.99)) if np.isfinite(diff).any() else 1.0
        image = ax.imshow(diff.T, aspect="auto", origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title("lateral - no_lateral log(AI)")
        ax.set_xlabel("lateral trace")
        ax.set_ylabel("TWT sample")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = figures / "lateral_minus_no_lateral.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs["lateral_minus_no_lateral"] = repo_relative_path(path, root=REPO_ROOT)
    return outputs


def _configured_bands(run_cfg: dict, *, dt_s: float) -> list[dict[str, float | str | bool]]:
    spectral = dict(run_cfg.get("spectral_qc") or {})
    bands = spectral.get("bands")
    if bands is None:
        nyquist = 0.5 / float(dt_s)
        configured_max = float(run_cfg.get("diagnostic_max_hz", 80.0))
        high = min(configured_max, 0.45 * nyquist)
        if not high > 0.0:
            raise ValueError("Cannot build default spectral bands with non-positive diagnostic max frequency.")
        return [
            {"name": "lowfreq", "low_hz": 0.0, "high_hz": 0.2 * high, "manual_spectral_band_override": False},
            {"name": "observable_band", "low_hz": 0.2 * high, "high_hz": 0.4 * high, "manual_spectral_band_override": False},
            {"name": "highfreq_or_nullspace", "low_hz": 0.4 * high, "high_hz": high, "manual_spectral_band_override": False},
        ]
    reason = str(spectral.get("manual_override_reason") or "").strip()
    if not reason:
        raise ValueError("Manual spectral_qc.bands requires spectral_qc.manual_override_reason.")
    if not isinstance(bands, list) or not bands:
        raise ValueError("real_field_zero_shot.spectral_qc.bands must be a non-empty list.")
    out = []
    for item in bands:
        if not isinstance(item, dict):
            raise ValueError("Each spectral_qc band must be a mapping.")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError("Each spectral_qc band requires name.")
        out.append(
            {
                "name": name,
                "low_hz": float(item.get("low_hz", 0.0)),
                "high_hz": float(item["high_hz"]) if item.get("high_hz") is not None else float("inf"),
                "manual_spectral_band_override": True,
                "manual_override_reason": reason,
            }
        )
    return out


def _observability_evidence(run_cfg: dict, bands: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    source_runs = dict(run_cfg.get("source_runs") or {})
    obs_dir_text = str(run_cfg.get("observability_dir") or source_runs.get("forward_observability_dir") or "").strip()
    missing = {
        str(band["name"]): {
            "observability_evidence_status": "missing",
            "dominant_evidence_status": "",
            "operator_support_summary": "",
            "detectability_ratio_p25_range": "",
            "observability_source_run": "",
        }
        for band in bands
    }
    if not obs_dir_text:
        return missing
    obs_dir = resolve_relative_path(obs_dir_text, root=REPO_ROOT)
    path = obs_dir / "frequency_evidence_bands.csv"
    if not path.is_file():
        return missing
    frame = pd.read_csv(path)
    if "frequency_hz" not in frame:
        return missing
    out = {}
    for band in bands:
        name = str(band["name"])
        low = float(band["low_hz"])
        high = float(band["high_hz"])
        scoped = frame[
            pd.to_numeric(frame["frequency_hz"], errors="coerce").ge(low)
            & pd.to_numeric(frame["frequency_hz"], errors="coerce").lt(high)
        ].copy()
        if scoped.empty:
            out[name] = {
                **missing[name],
                "observability_evidence_status": "empty_band_overlap",
                "observability_source_run": repo_relative_path(obs_dir, root=REPO_ROOT),
            }
            continue
        status_col = "dominant_evidence_status" if "dominant_evidence_status" in scoped else "empirical_status"
        support_col = "operator_support_summary" if "operator_support_summary" in scoped else "operator_support_class"
        ratio_col = "detectability_ratio_p25" if "detectability_ratio_p25" in scoped else "cluster_p25_detectability_ratio"
        ratios = pd.to_numeric(scoped.get(ratio_col, pd.Series(dtype=float)), errors="coerce").dropna()
        out[name] = {
            "observability_evidence_status": "present",
            "dominant_evidence_status": str(scoped[status_col].mode().iloc[0]) if status_col in scoped and not scoped[status_col].dropna().empty else "",
            "operator_support_summary": str(scoped[support_col].mode().iloc[0]) if support_col in scoped and not scoped[support_col].dropna().empty else "",
            "detectability_ratio_p25_range": (
                f"{float(ratios.min()):.6g}..{float(ratios.max()):.6g}" if not ratios.empty else ""
            ),
            "observability_source_run": repo_relative_path(obs_dir, root=REPO_ROOT),
        }
    return out


def _band_rms(values: np.ndarray, valid: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if data.ndim == 3:
        data = data.reshape((-1, data.shape[-1]))
        mask = mask.reshape((-1, mask.shape[-1]))
    if data.ndim != 2 or not np.any(mask):
        return float("nan")
    prepared = np.zeros_like(data, dtype=np.float64)
    for idx in range(data.shape[0]):
        row_mask = mask[idx]
        if not np.any(row_mask):
            continue
        mean = float(np.mean(data[idx, row_mask]))
        prepared[idx] = np.where(row_mask, data[idx] - mean, 0.0)
    spectrum = np.fft.rfft(prepared, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(band):
        return 0.0
    filtered = np.zeros_like(spectrum)
    filtered[:, band] = spectrum[:, band]
    component = np.fft.irfft(filtered, n=data.shape[1], axis=1)
    return float(np.sqrt(np.mean(component[mask] ** 2)))


def _band_component_2d(values: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    prepared = np.where(np.isfinite(data), data - np.nanmean(data, axis=1, keepdims=True), 0.0)
    spectrum = np.fft.rfft(prepared, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    filtered = np.zeros_like(spectrum)
    filtered[:, band] = spectrum[:, band]
    return np.fft.irfft(filtered, n=data.shape[1], axis=1)


def _write_spectral_qc(output_dir: Path, *, run_cfg: dict, dt_s: float) -> dict[str, str]:
    bands = _configured_bands(run_cfg, dt_s=dt_s)
    evidence = _observability_evidence(run_cfg, bands)
    rows = []
    arrays_by_role = {}
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir() or not (child / "predictions.npz").is_file():
            continue
        arrays = np.load(child / "predictions.npz", allow_pickle=True)
        arrays_by_role[child.name] = arrays
        delta = arrays["pred_delta_vs_lfm"]
        valid = arrays["valid_mask_model"].astype(bool) & (arrays["stitching_weight"] > 0.0)
        full = _band_rms(delta, valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        summary_path = child / "real_field_zero_shot_model_summary.json"
        synthetic_delta_std = float("nan")
        if summary_path.is_file():
            import json

            summary = json.load(open(summary_path, encoding="utf-8"))
            synthetic_delta_std = float((summary.get("normalization") or {}).get("delta", {}).get("std", float("nan")))
        for band in bands:
            band_rms = _band_rms(
                delta,
                valid,
                dt_s=dt_s,
                low_hz=float(band["low_hz"]),
                high_hz=float(band["high_hz"]),
            )
            rows.append(
                {
                    "model_role": child.name,
                    "signal": "pred_delta_vs_lfm",
                    "band": band["name"],
                    "low_hz": band["low_hz"],
                    "high_hz": band["high_hz"],
                    "manual_spectral_band_override": bool(band.get("manual_spectral_band_override", False)),
                    "manual_override_reason": str(band.get("manual_override_reason", "")),
                    **evidence.get(str(band["name"]), {}),
                    "band_rms": band_rms,
                    "fullband_rms": full,
                    "energy_ratio": band_rms / full if np.isfinite(full) and full > 0 else float("nan"),
                    "synthetic_train_delta_std": synthetic_delta_std,
                    "pred_delta_spectrum_vs_synthetic_train": (
                        band_rms / synthetic_delta_std
                        if np.isfinite(synthetic_delta_std) and synthetic_delta_std > 0
                        else float("nan")
                    ),
                    "pred_delta_spectrum_vs_well": float("nan"),
                    "pred_delta_spectrum_vs_well_status": "reported_per_well_in_R1_well_forward_diagnostic_when_supported",
                }
            )
    spectral_path = output_dir / "real_field_spectral_qc.csv"
    pd.DataFrame.from_records(rows).to_csv(spectral_path, index=False)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    diff_rows = []
    if "lateral" in arrays_by_role and "no_lateral" in arrays_by_role:
        lateral = arrays_by_role["lateral"]
        no_lateral = arrays_by_role["no_lateral"]
        diff = lateral["stitched_pred_log_ai"] - no_lateral["stitched_pred_log_ai"]
        valid = (
            lateral["valid_mask_model"].astype(bool)
            & no_lateral["valid_mask_model"].astype(bool)
            & (lateral["stitching_weight"] > 0.0)
            & (no_lateral["stitching_weight"] > 0.0)
        )
        full = _band_rms(diff, valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        for band in bands:
            band_rms = _band_rms(
                diff,
                valid,
                dt_s=dt_s,
                low_hz=float(band["low_hz"]),
                high_hz=float(band["high_hz"]),
            )
            diff_rows.append(
                {
                    "comparison": "lateral_minus_no_lateral",
                    "band": band["name"],
                    "low_hz": band["low_hz"],
                    "high_hz": band["high_hz"],
                    "manual_spectral_band_override": bool(band.get("manual_spectral_band_override", False)),
                    "manual_override_reason": str(band.get("manual_override_reason", "")),
                    **evidence.get(str(band["name"]), {}),
                    "band_rms": band_rms,
                    "fullband_rms": full,
                    "energy_ratio": band_rms / full if np.isfinite(full) and full > 0 else float("nan"),
                    "lateral_minus_no_lateral_nullspace_energy": (
                        band_rms if str(band["name"]) == "highfreq_or_nullspace" else float("nan")
                    ),
                }
            )
        if diff.ndim == 2:
            band_fig_path = figures / "lateral_minus_no_lateral_band_split.png"
            panel_specs = [{"name": "fullband", "values": diff}]
            panel_specs.extend(
                {
                    "name": str(band["name"]),
                    "values": _band_component_2d(
                        diff,
                        dt_s=dt_s,
                        low_hz=float(band["low_hz"]),
                        high_hz=float(band["high_hz"]),
                    ),
                }
                for band in bands
            )
            fig, axes = plt.subplots(1, len(panel_specs), figsize=(4 * len(panel_specs), 4), sharey=True)
            if len(panel_specs) == 1:
                axes = [axes]
            for ax, panel in zip(axes, panel_specs):
                values = np.asarray(panel["values"], dtype=np.float64)
                finite = values[np.isfinite(values)]
                vmax = float(np.nanquantile(np.abs(finite), 0.99)) if finite.size else 1.0
                image = ax.imshow(values.T, aspect="auto", origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_title(str(panel["name"]))
                ax.set_xlabel("lateral trace")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            axes[0].set_ylabel("TWT sample")
            fig.suptitle("lateral - no_lateral band split")
            fig.tight_layout()
            fig.savefig(band_fig_path, dpi=160)
            plt.close(fig)
    diff_path = output_dir / "lateral_difference_band_qc.csv"
    pd.DataFrame.from_records(diff_rows).to_csv(diff_path, index=False)

    figure_path = figures / "spectral_band_energy_qc.png"
    if rows:
        frame = pd.DataFrame.from_records(rows)
        fig, ax = plt.subplots(figsize=(8, 4))
        for role, group in frame.groupby("model_role"):
            ax.plot(group["band"], group["band_rms"], marker="o", label=role)
        ax.set_ylabel("RMS log(AI)")
        ax.set_title("R0 pred_delta_vs_lfm band energy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
    return {
        "real_field_spectral_qc": repo_relative_path(spectral_path, root=REPO_ROOT),
        "lateral_difference_band_qc": repo_relative_path(diff_path, root=REPO_ROOT),
        "lateral_minus_no_lateral_band_split_figure": (
            repo_relative_path(band_fig_path, root=REPO_ROOT) if "band_fig_path" in locals() and band_fig_path.is_file() else ""
        ),
        "spectral_band_energy_figure": (
            repo_relative_path(figure_path, root=REPO_ROOT) if figure_path.is_file() else ""
        ),
    }


def _export_zero_shot_volumes(output_dir: Path, *, run_cfg: dict, data_root: Path) -> dict[str, dict[str, object]]:
    if str(run_cfg.get("mode") or "volume").casefold() != "volume":
        return {}
    export_cfg = dict(run_cfg.get("volume_export") or {})
    if not bool(export_cfg.get("enabled", True)):
        return {}
    inputs = dict(run_cfg.get("real_field_inputs") or {})
    source_file = resolve_relative_path(str(inputs.get("seismic_file") or ""), root=data_root)
    source_type = str(inputs.get("seismic_type", inputs.get("type", "zgy"))).casefold()
    inline_chunk_size = int(export_cfg.get("inline_chunk_size", inputs.get("zgy_inline_chunk_size", 16)))
    nan_fill = export_cfg.get("nan_fill")
    outputs: dict[str, dict[str, object]] = {}
    for child in sorted(output_dir.iterdir()):
        npz_path = child / "predictions.npz"
        summary_path = child / "real_field_zero_shot_model_summary.json"
        if not child.is_dir() or not npz_path.is_file():
            continue
        with np.load(npz_path, allow_pickle=False) as arrays:
            pred_log_ai = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float32)
            if pred_log_ai.ndim != 3:
                continue
            pred_ai = log_ai_to_ai_volume(pred_log_ai)
            payload = export_volume_like_source(
                output_base=child / f"{child.name}_pred_ai",
                volume=pred_ai,
                ilines=np.asarray(arrays["ilines"], dtype=np.float64),
                xlines=np.asarray(arrays["xlines"], dtype=np.float64),
                samples=np.asarray(arrays["twt_s"], dtype=np.float64),
                source_seismic_file=source_file,
                source_seismic_type=source_type,
                title=f"R0 zero-shot pred_ai: {child.name}",
                details=[
                    "schema=real_field_zero_shot_model_v1",
                    "field=pred_ai",
                    "domain=AI",
                    "source_field=stitched_pred_log_ai",
                    "transform=exp(stitched_pred_log_ai)",
                ],
                seismic_options=inputs,
                inline_chunk_size=inline_chunk_size,
                nan_fill=nan_fill,
            )
            payload.update(
                {
                    "field": "pred_ai",
                    "value_domain": "AI",
                    "source_field": "stitched_pred_log_ai",
                    "value_transform": "exp(stitched_pred_log_ai)",
                }
            )
        payload["path"] = repo_relative_path(Path(str(payload["path"])), root=REPO_ROOT)
        outputs[child.name] = payload
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            summary["volume_export"] = payload
            write_json(summary_path, summary)
    return outputs


def _input_distribution_warnings(qc_path: Path) -> list[dict[str, object]]:
    if not qc_path.is_file():
        return []
    frame = pd.read_csv(qc_path)
    warnings = []
    for _, row in frame.iterrows():
        name = str(row.get("input", ""))
        gt5 = pd.to_numeric(row.get("fraction_abs_normalized_gt_5"), errors="coerce")
        if np.isfinite(gt5) and float(gt5) > 0.05:
            warnings.append(
                {
                    "warning": "input_distribution_ood",
                    "input": name,
                    "fraction_abs_normalized_gt_5": float(gt5),
                    "threshold": 0.05,
                }
            )
    return warnings


def main() -> None:
    args = parse_args()
    cfg_path = resolve_relative_path(args.config, root=REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_cfg = dict(cfg.get("real_field_zero_shot") or {})
    if not run_cfg:
        raise ValueError("experiments/common/common.yaml lacks real_field_zero_shot section.")
    output_root = resolve_relative_path(cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    output_dir = _resolve_output_dir("real_field_zero_shot", args.output_dir, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    device = str(args.device or run_cfg.get("device") or "auto")
    stitch_strategy = str(args.stitch_strategy or run_cfg.get("stitch_strategy") or "uniform")
    run_cfg = _prepare_real_field_inputs(run_cfg, cfg)
    models = _load_models(run_cfg)
    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)
    run_cfg_for_load = _prepare_run_config(run_cfg, cfg)
    inputs_for_load = dict(run_cfg_for_load.get("real_field_inputs") or {})
    transform_name = str(inputs_for_load.get("seismic_value_transform") or inputs_for_load.get("seismic_transform") or "identity")
    if transform_name not in {"identity", "raw", "none"} and "seismic_reference_stats" not in inputs_for_load:
        reference_payload = _load_seismic_reference_payload(run_cfg=run_cfg_for_load, models=[dict(item) for item in models])
        inputs_for_load["seismic_reference_stats"] = reference_payload["stats"]
        inputs_for_load["seismic_reference_sampling"] = reference_payload["sampling"]
        inputs_for_load["seismic_reference_stats_file"] = reference_payload["file"]
        run_cfg_for_load["real_field_inputs"] = inputs_for_load
    output_mode = str(run_cfg.get("mode") or "volume").casefold()
    if output_mode == "section":
        field = load_real_field_section(config=run_cfg_for_load, root=REPO_ROOT, data_root=data_root)
    elif output_mode == "volume":
        field = load_real_field_volume(config=run_cfg_for_load, root=REPO_ROOT, data_root=data_root)
    else:
        raise ValueError(f"Unsupported real_field_zero_shot.mode: {output_mode}")

    model_summaries = []
    input_qc_written = False
    for model_cfg in models:
        if output_mode == "volume":
            summary = run_zero_shot_volume_model(
                volume=field,
                model_cfg=model_cfg,
                output_dir=output_dir,
                root=REPO_ROOT,
                device_name=device,
                stitch_strategy=stitch_strategy,
            )
        else:
            summary = run_zero_shot_model(
                section=field,
                model_cfg=model_cfg,
                output_dir=output_dir,
                root=REPO_ROOT,
                device_name=device,
                stitch_strategy=stitch_strategy,
            )
        model_summaries.append(summary)
        if not input_qc_written:
            qc = input_qc_frame(field, summary.get("normalization", {}) or {})
            qc.to_csv(output_dir / "model_input_qc.csv", index=False)
            input_qc_written = True
    figure_outputs = _plot_zero_shot_qc(output_dir)
    well_outputs = (
        {"csv": "", "figures": {}, "status": "disabled_for_volume_mode"}
        if output_mode == "volume"
        else _write_well_prediction_qc(output_dir, run_cfg=run_cfg_for_load)
    )
    volume_exports = _export_zero_shot_volumes(output_dir, run_cfg=run_cfg_for_load, data_root=data_root)
    dt_s = float(field.twt_s[1] - field.twt_s[0]) if field.twt_s.size > 1 else 0.002
    spectral_outputs = _write_spectral_qc(output_dir, run_cfg=run_cfg, dt_s=dt_s)

    source_runs = dict(run_cfg.get("source_runs") or {})
    boundary = dict(run_cfg.get("boundary") or {})
    dt_s = float(field.twt_s[1] - field.twt_s[0]) if field.twt_s.size > 1 else 0.002
    input_qc_path = output_dir / "model_input_qc.csv"
    if output_mode == "volume":
        axis_contract = {
            "n_inline": int(field.lfm.shape[0]),
            "n_xline": int(field.lfm.shape[1]),
            "n_twt": int(field.lfm.shape[2]),
            "inline_start": float(field.ilines[0]),
            "inline_stop": float(field.ilines[-1]),
            "xline_start": float(field.xlines[0]),
            "xline_stop": float(field.xlines[-1]),
            "twt_start_s": float(field.twt_s[0]),
            "twt_stop_s": float(field.twt_s[-1]),
            "dt_s": float(field.twt_s[1] - field.twt_s[0]) if field.twt_s.size > 1 else None,
        }
    else:
        axis_contract = {
            "n_lateral": int(field.lfm.shape[0]),
            "n_twt": int(field.lfm.shape[1]),
            "twt_start_s": float(field.twt_s[0]),
            "twt_stop_s": float(field.twt_s[-1]),
            "dt_s": float(field.twt_s[1] - field.twt_s[0]) if field.twt_s.size > 1 else None,
        }
    input_contracts = {
        "lfm_variant": {
            "path": str(inputs_for_load["selected_lfm_contract_path"]),
            "contract_fingerprint_sha256": str(
                inputs_for_load["selected_lfm_contract_fingerprint_sha256"]
            ),
        },
        "well_control_set": {
            "path": str(inputs_for_load["well_control_contract_path"]),
            "contract_fingerprint_sha256": str(
                inputs_for_load["well_control_contract_fingerprint_sha256"]
            ),
        },
    }
    for model_summary in model_summaries:
        model_role = str(model_summary["model_role"])
        model_input = dict(dict(model_summary.get("input_contracts") or {}).get("model_run") or {})
        if not model_input:
            raise ValueError(f"R0 model summary lacks model_run contract: {model_role}")
        input_contracts[f"model:{model_role}"] = model_input
    wavelet_dir_text = str(source_runs.get("wavelet_generation_dir") or "").strip()
    if wavelet_dir_text:
        wavelet_summary_path = resolve_relative_path(wavelet_dir_text, root=REPO_ROOT) / "run_summary.json"
        with wavelet_summary_path.open("r", encoding="utf-8") as handle:
            wavelet_summary = json.load(handle)
        input_contracts["wavelet"] = {
            "path": repo_relative_path(wavelet_summary_path, root=REPO_ROOT),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                wavelet_summary, label=f"wavelet run {wavelet_summary_path.parent}"
            ),
        }
    primary_artifacts = {
        f"prediction:{str(model_summary['model_role'])}": resolve_relative_path(
            str(dict(model_summary["outputs"])["predictions"]), root=REPO_ROOT
        )
        for model_summary in model_summaries
    }
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=ZERO_SHOT_SUMMARY_SCHEMA_VERSION,
        semantics={
            "mode": output_mode,
            "sample_domain": str(field.metadata.get("sample_domain") or "time"),
            "axis_contract": axis_contract,
        },
        business_config={
            "mode": output_mode,
            "stitch_strategy": stitch_strategy,
            "boundary": boundary,
            "volume": dict(run_cfg.get("volume") or {}),
        },
        input_contracts=input_contracts,
        primary_artifacts=primary_artifacts,
    )
    summary = {
        "schema_version": ZERO_SHOT_SUMMARY_SCHEMA_VERSION,
        "status": "needs_forward_diagnostic",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "mode": output_mode,
        "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
        "device_requested": device,
        "stitch_strategy": stitch_strategy,
        "source_runs": source_runs,
        "section": field.metadata if output_mode == "section" else {},
        "volume": field.metadata if output_mode == "volume" else {},
        "axis_contract": axis_contract,
        "mask_contract": {
            "valid_fraction": float(field.valid_mask.mean()),
            "valid_samples": int(field.valid_mask.sum()),
        },
        "boundary_contract": {
            "loss_or_eval_erosion_s": float(boundary.get("loss_or_eval_erosion_s", 0.0) or 0.0),
            "prediction_taper_halo_s": float(boundary.get("prediction_taper_halo_s", 0.0) or 0.0),
            "forward_diagnostic_erosion": "disabled",
            "loss_or_eval_erosion_samples": int(np.ceil(float(boundary.get("loss_or_eval_erosion_s", 0.0) or 0.0) / dt_s)),
            "prediction_taper_halo_samples": int(np.ceil(float(boundary.get("prediction_taper_halo_s", 0.0) or 0.0) / dt_s)),
            "dt_s": dt_s,
        },
        "input_distribution_qc": {
            "path": repo_relative_path(input_qc_path, root=REPO_ROOT),
            "warnings": _input_distribution_warnings(input_qc_path),
        },
        "outputs": {
            "model_input_qc": repo_relative_path(input_qc_path, root=REPO_ROOT),
            "figures": figure_outputs,
            "well_prediction_qc": well_outputs,
            "volume_exports": volume_exports,
            **spectral_outputs,
        },
        "models": model_summaries,
        "code_version_or_git_commit": str(run_cfg.get("code_version_or_git_commit") or _git_commit()),
    }
    write_json(output_dir / "real_field_zero_shot_summary.json", summary)
    print("=== Real Field Zero-Shot ===")
    print(f"Output: {output_dir}")
    print(f"Models: {len(model_summaries)}")
    print(f"Status: {summary['status']}")


if __name__ == "__main__":
    main()
