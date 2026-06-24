"""R2 low-frequency calibration-only helpers for real-field GINN-v2 outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cup.seismic.volume_export import export_volume_like_source
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sha256_file, write_json
from ginn_v2.real_field import forward_log_ai, load_selected_wavelet


SCHEMA_VERSION = "real_field_lowfreq_calibration_v1"
DEFAULT_MODEL_ROLES = ("no_lateral", "lateral")


@dataclass(frozen=True)
class CalibrationSource:
    section_id: str
    zero_shot_dir: Path
    forward_diagnostic_dir: Path
    lfm_dir: Path | None


@dataclass(frozen=True)
class CalibrationConfig:
    sources: tuple[CalibrationSource, ...]
    model_roles: tuple[str, ...]
    min_valid_samples: int
    output_arrays: bool
    wavelet_generation_dir: Path


def parse_lowfreq_calibration_config(raw: Mapping[str, Any], *, root: Path) -> CalibrationConfig:
    cfg = _mapping(raw.get("real_field_lowfreq_calibration"), "real_field_lowfreq_calibration")
    source_items = cfg.get("sources")
    volume_source = cfg.get("volume_source")
    if volume_source is not None:
        source_items = [volume_source]
    if not isinstance(source_items, Sequence) or isinstance(source_items, (str, bytes)) or not source_items:
        raise ValueError("real_field_lowfreq_calibration.sources or volume_source must be a non-empty list/mapping.")
    sources: list[CalibrationSource] = []
    for idx, item in enumerate(source_items):
        entry = _mapping(item, f"real_field_lowfreq_calibration.sources[{idx}]")
        section_id = str(entry.get("section_id") or entry.get("source_id") or "volume").strip()
        if not section_id:
            raise ValueError(f"real_field_lowfreq_calibration.sources[{idx}].section_id must be explicit.")
        lfm_text = str(entry.get("lfm_dir") or "").strip()
        sources.append(
            CalibrationSource(
                section_id=section_id,
                zero_shot_dir=resolve_relative_path(
                    _required_text(entry, "zero_shot_dir", path=f"real_field_lowfreq_calibration.sources[{idx}]"),
                    root=root,
                ),
                forward_diagnostic_dir=resolve_relative_path(
                    _required_text(
                        entry,
                        "forward_diagnostic_dir",
                        path=f"real_field_lowfreq_calibration.sources[{idx}]",
                    ),
                    root=root,
                ),
                lfm_dir=resolve_relative_path(lfm_text, root=root) if lfm_text else None,
            )
        )
    roles = tuple(str(x).strip() for x in cfg.get("model_roles", DEFAULT_MODEL_ROLES) if str(x).strip())
    if not roles:
        raise ValueError("real_field_lowfreq_calibration.model_roles must not be empty.")
    unsupported = sorted(set(roles) - set(DEFAULT_MODEL_ROLES))
    if unsupported:
        raise ValueError(f"R2 first pass only supports roles {DEFAULT_MODEL_ROLES}, got unsupported {unsupported}.")
    source_runs = _mapping(cfg.get("source_runs"), "real_field_lowfreq_calibration.source_runs")
    return CalibrationConfig(
        sources=tuple(sources),
        model_roles=roles,
        min_valid_samples=int(cfg.get("min_valid_samples", 16)),
        output_arrays=bool(cfg.get("output_arrays", True)),
        wavelet_generation_dir=resolve_relative_path(
            _required_text(source_runs, "wavelet_generation_dir", path="real_field_lowfreq_calibration.source_runs"),
            root=root,
        ),
    )


def run_lowfreq_calibration(
    config: CalibrationConfig,
    *,
    output_dir: Path,
    root: Path,
    git_commit: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    arrays_by_source_role: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for source in config.sources:
        source_rows.append(_source_status_row(source, root=root))
        r0_summary = _read_json(source.zero_shot_dir / "real_field_zero_shot_summary.json")
        r1_summary = _read_json(source.forward_diagnostic_dir / "real_field_forward_diagnostic_summary.json")
        predictions = _load_prediction_arrays(source.zero_shot_dir, roles=config.model_roles)
        for role, arrays in predictions.items():
            if config.output_arrays:
                arrays_by_source_role[(source.section_id, role)] = arrays
        well_table = _read_csv_required(source.forward_diagnostic_dir / "well_forward_diagnostic.csv")
        if "sampling_mode" in well_table.columns:
            tie_metrics: dict[str, Mapping[str, Any]] = {}
        else:
            r1_config = _load_r1_config(r1_summary, root=root)
            well_auto_tie_dir = _r1_well_auto_tie_dir(r1_config, root=root)
            tie_metrics = _load_tie_metrics(well_auto_tie_dir, root=root)
        for role, arrays in predictions.items():
            rows = _source_role_evidence(
                source=source,
                role=role,
                arrays=arrays,
                well_table=well_table,
                tie_metrics=tie_metrics,
                min_valid_samples=config.min_valid_samples,
                root=root,
            )
            evidence_rows.extend(rows)
        _write_source_metadata(
            output_dir=output_dir,
            source=source,
            r0_summary=r0_summary,
            r1_summary=r1_summary,
            root=root,
        )

    evidence = pd.DataFrame.from_records(evidence_rows)
    evidence_path = output_dir / "well_calibration_evidence.csv"
    evidence.to_csv(evidence_path, index=False)

    well_bias = _aggregate_well_bias(evidence, model_roles=config.model_roles)
    well_bias_path = output_dir / "calibration_bias_by_well.csv"
    well_bias.to_csv(well_bias_path, index=False)

    model_bias = _aggregate_model_bias(well_bias, model_roles=config.model_roles)
    if evidence.empty and well_bias.empty:
        model_bias = _noop_model_bias(config.model_roles, reason="no_well_evidence_for_volume_mode")
    model_bias_path = output_dir / "calibration_bias_by_model.csv"
    model_bias.to_csv(model_bias_path, index=False)

    well_comparison = _calibrated_well_comparison(evidence, model_bias)
    well_comparison_path = output_dir / "calibrated_well_ai_comparison.csv"
    well_comparison.to_csv(well_comparison_path, index=False)

    forward_metrics = _forward_invariance_metrics(
        config=config,
        model_bias=model_bias,
        arrays_by_source_role=arrays_by_source_role,
        output_dir=output_dir,
    )
    forward_path = output_dir / "calibrated_forward_metrics.csv"
    forward_metrics.to_csv(forward_path, index=False)

    figures = _write_figures(
        output_dir,
        root=root,
        well_bias=well_bias,
        model_bias=model_bias,
        well_comparison=well_comparison,
    )
    source_frame = pd.DataFrame.from_records(source_rows)
    source_frame.to_csv(output_dir / "source_runs.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": _summary_status(model_bias),
        "stage": "R2-lowfreq-calibration-only",
        "calibration_semantics": "pred_log_ai_calibrated = pred_log_ai + one_global_constant_bias_per_model_role",
        "forward_invariance_expected": True,
        "model_roles": list(config.model_roles),
        "min_valid_samples": int(config.min_valid_samples),
        "n_sources": int(len(config.sources)),
        "n_evidence_rows": int(evidence.shape[0]),
        "n_well_bias_rows": int(well_bias.shape[0]),
        "source_runs": source_rows,
        "outputs": {
            "well_calibration_evidence": repo_relative_path(evidence_path, root=root),
            "calibration_bias_by_well": repo_relative_path(well_bias_path, root=root),
            "calibration_bias_by_model": repo_relative_path(model_bias_path, root=root),
            "calibrated_well_ai_comparison": repo_relative_path(well_comparison_path, root=root),
            "calibrated_forward_metrics": repo_relative_path(forward_path, root=root),
            "source_runs": repo_relative_path(output_dir / "source_runs.csv", root=root),
            "figures": figures,
        },
        "code_version_or_git_commit": git_commit,
    }
    summary_path = output_dir / "lowfreq_calibration_summary.json"
    write_json(summary_path, summary)
    return summary


def _source_role_evidence(
    *,
    source: CalibrationSource,
    role: str,
    arrays: Mapping[str, np.ndarray],
    well_table: pd.DataFrame,
    tie_metrics: Mapping[str, Mapping[str, Any]],
    min_valid_samples: int,
    root: Path,
) -> list[dict[str, Any]]:
    if "sampling_mode" in well_table.columns:
        return _volume_source_role_evidence(
            source=source,
            role=role,
            well_table=well_table,
            min_valid_samples=min_valid_samples,
        )
    rows: list[dict[str, Any]] = []
    role_rows = well_table[
        well_table.get("model_role", pd.Series(dtype=str)).astype(str).eq(role)
        & well_table.get("wellbore_class", pd.Series(dtype=str)).astype(str).str.casefold().eq("vertical")
    ].copy()
    pred = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)
    valid_model = np.asarray(arrays["valid_mask_model"], dtype=bool)
    stitching_weight = np.asarray(arrays.get("stitching_weight", np.ones_like(pred)), dtype=np.float64)
    twt_s = np.asarray(arrays["twt_s"], dtype=np.float64)

    for _, row in role_rows.iterrows():
        well_name = str(row.get("well_name", "")).strip()
        base = _base_evidence_row(source=source, role=role, row=row)
        if str(row.get("status", "")) != "ok":
            rows.append({**base, "calibration_status": "skipped_r1_status", "skip_reason": str(row.get("status", ""))})
            continue
        if str(row.get("well_ai_status", "")) != "ok":
            rows.append(
                {**base, "calibration_status": "skipped_well_ai_status", "skip_reason": str(row.get("well_ai_status", ""))}
            )
            continue
        tie = tie_metrics.get(well_name)
        if tie is None:
            rows.append({**base, "calibration_status": "skipped_missing_tie_metrics", "skip_reason": "missing_tie_metrics"})
            continue
        trace_idx = _int_value(row.get("nearest_section_trace"), default=-1)
        if trace_idx < 0 or trace_idx >= pred.shape[0]:
            rows.append({**base, "calibration_status": "skipped_trace_out_of_range", "skip_reason": "trace_out_of_range"})
            continue
        try:
            well_log_ai = _log_ai_at_twt(
                las_path=resolve_relative_path(str(tie["filtered_las_file"]), root=root),
                tdt_path=resolve_relative_path(str(tie["optimized_tdt_file"]), root=root),
                twt_s=twt_s,
            )
        except Exception as exc:
            rows.append({**base, "calibration_status": "skipped_reference_load_failed", "skip_reason": str(exc)})
            continue
        pred_trace = pred[trace_idx]
        mask = valid_model[trace_idx] & (stitching_weight[trace_idx] > 0.0) & np.isfinite(well_log_ai) & np.isfinite(pred_trace)
        n_valid = int(np.count_nonzero(mask))
        if n_valid < min_valid_samples:
            rows.append(
                {
                    **base,
                    "calibration_status": "skipped_insufficient_valid_samples",
                    "skip_reason": f"n_valid={n_valid}",
                    "n_valid": n_valid,
                }
            )
            continue
        residual = well_log_ai[mask] - pred_trace[mask]
        bias = float(np.median(residual))
        before = _ai_metrics(reference=well_log_ai, prediction=pred_trace, mask=mask)
        after = _ai_metrics(reference=well_log_ai, prediction=pred_trace + bias, mask=mask)
        band_before = _band_metrics(reference=well_log_ai, prediction=pred_trace, mask=mask, dt_s=_dt_from_axis(twt_s))
        band_after = _band_metrics(reference=well_log_ai, prediction=pred_trace + bias, mask=mask, dt_s=_dt_from_axis(twt_s))
        rows.append(
            {
                **base,
                "calibration_status": "ok",
                "skip_reason": "",
                "n_valid": n_valid,
                "well_bias_filtered_minus_pred_median": bias,
                "well_residual_mean_filtered_minus_pred": float(np.mean(residual)),
                "well_residual_std_filtered_minus_pred": float(np.std(residual)),
                "rmse_before": before["rmse"],
                "rmse_after_well_bias": after["rmse"],
                "corr_before": before["corr"],
                "corr_after_well_bias": after["corr"],
                **_prefix_dict(band_before, "before_"),
                **_prefix_dict(band_after, "after_well_bias_"),
                "reference_quality_flag": _reference_quality_flag(row),
            }
        )
    return rows


def _volume_source_role_evidence(
    *,
    source: CalibrationSource,
    role: str,
    well_table: pd.DataFrame,
    min_valid_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_rows = well_table[well_table.get("model_role", pd.Series(dtype=str)).astype(str).eq(role)].copy()
    for _, row in role_rows.iterrows():
        base = _base_evidence_row(source=source, role=role, row=row)
        if str(row.get("status", "")) != "ok":
            rows.append({**base, "calibration_status": "skipped_r1_status", "skip_reason": str(row.get("status", ""))})
            continue
        if str(row.get("well_ai_status", "")) != "ok":
            rows.append(
                {**base, "calibration_status": "skipped_well_ai_status", "skip_reason": str(row.get("well_ai_status", ""))}
            )
            continue
        n_valid = _int_value(row.get("well_ai_n_valid"), default=0)
        if n_valid < min_valid_samples:
            rows.append(
                {
                    **base,
                    "calibration_status": "skipped_insufficient_valid_samples",
                    "skip_reason": f"n_valid={n_valid}",
                    "n_valid": n_valid,
                }
            )
            continue
        bias = _float_value(row.get("calibration_bias_filtered_minus_pred_median"))
        if not np.isfinite(bias):
            rows.append({**base, "calibration_status": "skipped_missing_bias", "skip_reason": "missing calibration bias"})
            continue
        rows.append(
            {
                **base,
                "calibration_status": "ok",
                "skip_reason": "",
                "n_valid": n_valid,
                "well_bias_filtered_minus_pred_median": bias,
                "well_residual_mean_filtered_minus_pred": -_float_value(row.get("well_ai_bias")),
                "well_residual_std_filtered_minus_pred": np.nan,
                "rmse_before": _float_value(row.get("well_ai_rmse")),
                "rmse_after_well_bias": np.nan,
                "corr_before": _float_value(row.get("well_ai_corr")),
                "corr_after_well_bias": _float_value(row.get("well_ai_corr")),
                "before_lowfreq_rmse": _float_value(row.get("well_ai_lowfreq_rmse")),
                "before_lowfreq_corr": _float_value(row.get("well_ai_lowfreq_corr")),
                "before_observable_band_rmse": _float_value(row.get("well_ai_observable_band_rmse")),
                "before_observable_band_corr": _float_value(row.get("well_ai_observable_band_corr")),
                "before_highfreq_or_nullspace_rmse": _float_value(row.get("well_ai_highfreq_or_nullspace_rmse")),
                "before_highfreq_or_nullspace_corr": _float_value(row.get("well_ai_highfreq_or_nullspace_corr")),
                "reference_quality_flag": _reference_quality_flag(row),
            }
        )
    return rows


def _aggregate_well_bias(evidence: pd.DataFrame, *, model_roles: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if evidence.empty:
        return pd.DataFrame()
    ok = evidence[evidence["calibration_status"].astype(str).eq("ok")].copy()
    for role in model_roles:
        role_frame = ok[ok["model_role"].astype(str).eq(str(role))]
        for well_name, group in role_frame.groupby("well_name", sort=True):
            biases = pd.to_numeric(group["well_bias_filtered_minus_pred_median"], errors="coerce").dropna().to_numpy()
            if biases.size == 0:
                continue
            rows.append(
                {
                    "model_role": role,
                    "well_name": well_name,
                    "well_bias": float(np.median(biases)),
                    "n_section_records": int(group.shape[0]),
                    "sections": ";".join(sorted(str(x) for x in group["section_id"].unique())),
                    "n_valid_total": int(pd.to_numeric(group["n_valid"], errors="coerce").fillna(0).sum()),
                    "reference_quality_flags": ";".join(sorted(set(str(x) for x in group["reference_quality_flag"].dropna()))),
                    "rmse_before_median": float(np.median(pd.to_numeric(group["rmse_before"], errors="coerce").dropna())),
                    "rmse_after_well_bias_median": float(
                        np.median(pd.to_numeric(group["rmse_after_well_bias"], errors="coerce").dropna())
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _aggregate_model_bias(well_bias: pd.DataFrame, *, model_roles: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role in model_roles:
        group = well_bias[well_bias.get("model_role", pd.Series(dtype=str)).astype(str).eq(str(role))]
        values = pd.to_numeric(group.get("well_bias", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        if values.size == 0:
            rows.append(
                {
                    "model_role": role,
                    "status": "insufficient_evidence",
                    "bias_model_role": np.nan,
                    "n_wells": 0,
                    "jackknife_bias_std": np.nan,
                    "jackknife_max_abs_shift": np.nan,
                    "jackknife_biases": "",
                }
            )
            continue
        bias = float(np.median(values))
        jackknife: list[float] = []
        for idx in range(values.size):
            subset = np.delete(values, idx)
            jackknife.append(float(np.median(subset)) if subset.size else bias)
        shifts = np.asarray(jackknife, dtype=np.float64) - bias
        rows.append(
            {
                "model_role": role,
                "status": "ok",
                "bias_model_role": bias,
                "n_wells": int(values.size),
                "well_bias_p10": float(np.quantile(values, 0.10)),
                "well_bias_p50": float(np.quantile(values, 0.50)),
                "well_bias_p90": float(np.quantile(values, 0.90)),
                "jackknife_bias_std": float(np.std(jackknife)),
                "jackknife_max_abs_shift": float(np.max(np.abs(shifts))) if shifts.size else 0.0,
                "jackknife_biases": ";".join(f"{x:.8g}" for x in jackknife),
            }
        )
    return pd.DataFrame.from_records(rows)


def _noop_model_bias(model_roles: Sequence[str], *, reason: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "model_role": role,
                "status": "no_op_no_well_evidence",
                "bias_model_role": 0.0,
                "n_wells": 0,
                "jackknife_bias_std": np.nan,
                "jackknife_max_abs_shift": np.nan,
                "jackknife_biases": "",
                "reason": reason,
            }
            for role in model_roles
        ]
    )


def _calibrated_well_comparison(evidence: pd.DataFrame, model_bias: pd.DataFrame) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame()
    bias_lookup = {
        str(row["model_role"]): float(row["bias_model_role"])
        for _, row in model_bias.iterrows()
        if str(row.get("status", "")) == "ok" and np.isfinite(float(row.get("bias_model_role", np.nan)))
    }
    rows: list[dict[str, Any]] = []
    for _, row in evidence[evidence["calibration_status"].astype(str).eq("ok")].iterrows():
        role = str(row["model_role"])
        bias = bias_lookup.get(role, np.nan)
        well_bias = float(row["well_bias_filtered_minus_pred_median"])
        residual_after_global = float(row["well_residual_mean_filtered_minus_pred"]) - bias
        rows.append(
            {
                "section_id": row["section_id"],
                "well_name": row["well_name"],
                "model_role": role,
                "global_bias_applied": bias,
                "well_bias": well_bias,
                "well_minus_global_bias": well_bias - bias if np.isfinite(bias) else np.nan,
                "n_valid": row["n_valid"],
                "rmse_before": row["rmse_before"],
                "rmse_after_well_bias": row["rmse_after_well_bias"],
                "rmse_after_global_bias": _rmse_after_constant_bias(
                    rmse_before=float(row["rmse_before"]),
                    mean_residual=float(row["well_residual_mean_filtered_minus_pred"]),
                    bias=bias,
                ),
                "mean_residual_after_global_bias": residual_after_global if np.isfinite(bias) else np.nan,
                "corr_before": row["corr_before"],
                "corr_after_global_bias": row["corr_before"],
                "reference_quality_flag": row["reference_quality_flag"],
            }
        )
    return pd.DataFrame.from_records(rows)


def _forward_invariance_metrics(
    *,
    config: CalibrationConfig,
    model_bias: pd.DataFrame,
    arrays_by_source_role: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    output_dir: Path,
) -> pd.DataFrame:
    bias_lookup = {
        str(row["model_role"]): float(row["bias_model_role"])
        for _, row in model_bias.iterrows()
        if str(row.get("status", "")) == "ok" and np.isfinite(float(row.get("bias_model_role", np.nan)))
    }
    wavelet, wavelet_meta = load_selected_wavelet(config.wavelet_generation_dir)
    rows: list[dict[str, Any]] = []
    prediction_root = output_dir / "calibrated_predictions"
    for source in config.sources:
        r1_forward = _read_csv_required(source.forward_diagnostic_dir / "forward_diagnostic_metrics.csv")
        for role in config.model_roles:
            arrays = arrays_by_source_role.get((source.section_id, role))
            if arrays is None:
                continue
            pred = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)
            valid = np.asarray(arrays["valid_mask_model"], dtype=bool)
            weight = np.asarray(arrays.get("stitching_weight", np.ones_like(pred)), dtype=np.float64)
            bias = bias_lookup.get(role, np.nan)
            calibrated = pred + bias if np.isfinite(bias) else np.full_like(pred, np.nan)
            before_syn = forward_log_ai(pred, wavelet)
            after_syn = forward_log_ai(calibrated, wavelet) if np.isfinite(bias) else np.full_like(before_syn, np.nan)
            valid_forward = valid[..., 1:] & np.isfinite(before_syn) & np.isfinite(after_syn) & (weight[..., 1:] > 0.0)
            diff = np.abs(after_syn - before_syn)
            max_diff = float(np.nanmax(diff[valid_forward])) if np.any(valid_forward) else np.nan
            rms_diff = float(np.sqrt(np.nanmean(diff[valid_forward] ** 2))) if np.any(valid_forward) else np.nan
            before_residual = _forward_metric_value(
                r1_forward,
                role=role,
                metric="residual_rms_scaled",
            )
            rows.append(
                {
                    "section_id": source.section_id,
                    "model_role": role,
                    "bias_model_role": bias,
                    "n_valid_forward": int(np.count_nonzero(valid_forward)),
                    "synthetic_max_abs_diff_before_after": max_diff,
                    "synthetic_rms_diff_before_after": rms_diff,
                    "residual_rms_before": before_residual,
                    "residual_rms_after": before_residual,
                    "residual_rms_delta": 0.0 if np.isfinite(before_residual) else np.nan,
                    "forward_invariance_status": "ok" if np.isfinite(max_diff) and max_diff <= 1.0e-10 else "failed",
                    "wavelet_sha256": wavelet_meta.get("wavelet_sha256", ""),
                }
            )
            if config.output_arrays and np.isfinite(bias):
                target = prediction_root / source.section_id / role / "calibrated_predictions.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target,
                    pred_log_ai_original=np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float32),
                    pred_log_ai_calibrated=np.asarray(calibrated, dtype=np.float32),
                    bias_applied=np.asarray(bias, dtype=np.float32),
                    valid_mask_model=np.asarray(arrays["valid_mask_model"], dtype=bool),
                    ilines=np.asarray(arrays["ilines"]),
                    xlines=np.asarray(arrays["xlines"]),
                    twt_s=np.asarray(arrays["twt_s"]),
                )
                if pred.ndim == 3:
                    r0_summary = _read_json(source.zero_shot_dir / "real_field_zero_shot_summary.json")
                    volume_meta = _mapping(r0_summary.get("volume"), "real_field_zero_shot_summary.volume")
                    seismic_file = Path(str(volume_meta.get("seismic_file") or ""))
                    seismic_type = str(volume_meta.get("seismic_type") or "").strip()
                    if seismic_file and seismic_type:
                        export = export_volume_like_source(
                            output_base=target.parent / f"{role}_pred_log_ai_calibrated",
                            volume=np.asarray(calibrated, dtype=np.float32),
                            ilines=np.asarray(arrays["ilines"], dtype=np.float64),
                            xlines=np.asarray(arrays["xlines"], dtype=np.float64),
                            samples=np.asarray(arrays["twt_s"], dtype=np.float64),
                            source_seismic_file=seismic_file,
                            source_seismic_type=seismic_type,
                            title=f"R2 calibrated {role}",
                            details=[
                                f"source zero-shot: {source.zero_shot_dir}",
                                f"bias_model_role: {bias}",
                            ],
                            inline_chunk_size=16,
                            nan_fill=None,
                        )
                        rows[-1]["calibrated_volume_export"] = export.get("path", "")
                        rows[-1]["calibrated_volume_export_sha256"] = export.get("sha256", "")
    return pd.DataFrame.from_records(rows)


def _write_figures(
    output_dir: Path,
    *,
    root: Path,
    well_bias: pd.DataFrame,
    model_bias: pd.DataFrame,
    well_comparison: pd.DataFrame,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    figures: dict[str, str] = {}
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not well_bias.empty:
        path = figure_dir / "calibration_bias_by_well.png"
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for role, group in well_bias.groupby("model_role"):
            ax.scatter(group["well_name"], group["well_bias"], label=str(role))
        for _, row in model_bias.iterrows():
            if str(row.get("status", "")) == "ok":
                ax.axhline(float(row["bias_model_role"]), linestyle="--", linewidth=1.0, label=f"{row['model_role']} global")
        ax.set_ylabel("filtered LAS - prediction median log(AI)")
        ax.set_title("R2 constant bias evidence by well")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures["calibration_bias_by_well"] = repo_relative_path(path, root=root)
    if not well_comparison.empty:
        path = figure_dir / "well_rmse_before_after_global_bias.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for role, group in well_comparison.groupby("model_role"):
            before = pd.to_numeric(group["rmse_before"], errors="coerce")
            after = pd.to_numeric(group["rmse_after_global_bias"], errors="coerce")
            ax.scatter(before, after, label=str(role))
        lim_values = pd.concat(
            [
                pd.to_numeric(well_comparison["rmse_before"], errors="coerce"),
                pd.to_numeric(well_comparison["rmse_after_global_bias"], errors="coerce"),
            ]
        ).dropna()
        if not lim_values.empty:
            high = float(lim_values.max()) * 1.05
            ax.plot([0, high], [0, high], color="black", linewidth=1.0, alpha=0.5)
            ax.set_xlim(0, high)
            ax.set_ylim(0, high)
        ax.set_xlabel("RMSE before")
        ax.set_ylabel("RMSE after global bias")
        ax.set_title("R2 well AI RMSE effect")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures["well_rmse_before_after_global_bias"] = repo_relative_path(path, root=root)
    return figures


def _source_status_row(source: CalibrationSource, *, root: Path) -> dict[str, Any]:
    lfm_status = "not_configured"
    lfm_summary = ""
    if source.lfm_dir is not None:
        if source.lfm_dir.is_dir():
            lfm_status = "ok"
            lfm_summary = str(source.lfm_dir / "real_field_lfm_summary.json")
            if not Path(lfm_summary).is_file():
                lfm_status = "missing_summary"
        else:
            lfm_status = "missing"
    return {
        "section_id": source.section_id,
        "zero_shot_dir": repo_relative_path(source.zero_shot_dir, root=root),
        "forward_diagnostic_dir": repo_relative_path(source.forward_diagnostic_dir, root=root),
        "lfm_dir": repo_relative_path(source.lfm_dir, root=root) if source.lfm_dir is not None else "",
        "lfm_source_status": lfm_status,
        "lfm_summary": repo_relative_path(lfm_summary, root=root) if lfm_summary else "",
    }


def _write_source_metadata(
    *,
    output_dir: Path,
    source: CalibrationSource,
    r0_summary: Mapping[str, Any],
    r1_summary: Mapping[str, Any],
    root: Path,
) -> None:
    target = output_dir / "sources" / source.section_id / "source_metadata.json"
    payload = {
        "section_id": source.section_id,
        "zero_shot_dir": repo_relative_path(source.zero_shot_dir, root=root),
        "forward_diagnostic_dir": repo_relative_path(source.forward_diagnostic_dir, root=root),
        "lfm_dir": repo_relative_path(source.lfm_dir, root=root) if source.lfm_dir is not None else "",
        "r0_summary_sha256": sha256_file(source.zero_shot_dir / "real_field_zero_shot_summary.json"),
        "r1_summary_sha256": sha256_file(source.forward_diagnostic_dir / "real_field_forward_diagnostic_summary.json"),
        "r0_status": r0_summary.get("status", ""),
        "r1_status": r1_summary.get("status", ""),
    }
    write_json(target, payload)


def _load_prediction_arrays(zero_shot_dir: Path, *, roles: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for role in roles:
        npz_path = zero_shot_dir / role / "predictions.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"R0 predictions not found for role={role}: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as data:
            out[role] = {name: np.asarray(data[name]) for name in data.files}
        required = {"stitched_pred_log_ai", "valid_mask_model", "twt_s", "stitching_weight", "ilines", "xlines"}
        missing = sorted(required - set(out[role]))
        if missing:
            raise ValueError(f"R0 predictions missing {missing}: {npz_path}")
    return out


def _load_r1_config(r1_summary: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    config_text = str(r1_summary.get("config_file") or "").strip()
    if not config_text:
        raise ValueError("R1 summary lacks config_file.")
    return load_yaml_config(resolve_relative_path(config_text, root=root))


def _r1_well_auto_tie_dir(r1_config: Mapping[str, Any], *, root: Path) -> Path:
    cfg = _mapping(r1_config.get("real_field_forward_diagnostic"), "real_field_forward_diagnostic")
    well_qc = _mapping(cfg.get("well_qc"), "real_field_forward_diagnostic.well_qc")
    return resolve_relative_path(
        _required_text(well_qc, "well_auto_tie_dir", path="real_field_forward_diagnostic.well_qc"),
        root=root,
    )


def _load_tie_metrics(well_auto_tie_dir: Path, *, root: Path) -> dict[str, Mapping[str, Any]]:
    path = well_auto_tie_dir / "well_tie_metrics.csv"
    frame = _read_csv_required(path)
    required = {"well_name", "filtered_las_file", "optimized_tdt_file"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"well_tie_metrics.csv missing {missing}: {path}")
    out: dict[str, Mapping[str, Any]] = {}
    for _, row in frame.iterrows():
        well = str(row.get("well_name", "")).strip()
        if not well:
            continue
        out[well] = row.to_dict()
    return out


def _log_ai_at_twt(*, las_path: Path, tdt_path: Path, twt_s: np.ndarray) -> np.ndarray:
    import lasio

    las = lasio.read(str(las_path))
    frame = las.df()
    if "AI" not in frame.columns:
        raise ValueError(f"Filtered LAS lacks AI curve: {las_path}")
    md_axis = frame.index.to_numpy(dtype=np.float64)
    ai = frame["AI"].to_numpy(dtype=np.float64)
    finite_ai = np.isfinite(md_axis) & np.isfinite(ai) & (ai > 0.0)
    if int(np.count_nonzero(finite_ai)) < 2:
        return np.full_like(twt_s, np.nan, dtype=np.float64)
    tdt = pd.read_csv(tdt_path)
    if not {"twt_s", "md_m"}.issubset(tdt.columns):
        raise ValueError(f"Optimized TDT must contain twt_s/md_m: {tdt_path}")
    tdt_twt = pd.to_numeric(tdt["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
    tdt_md = pd.to_numeric(tdt["md_m"], errors="coerce").to_numpy(dtype=np.float64)
    finite_tdt = np.isfinite(tdt_twt) & np.isfinite(tdt_md)
    if int(np.count_nonzero(finite_tdt)) < 2:
        return np.full_like(twt_s, np.nan, dtype=np.float64)
    order_tdt = np.argsort(tdt_twt[finite_tdt])
    tdt_twt_sorted = tdt_twt[finite_tdt][order_tdt]
    tdt_md_sorted = tdt_md[finite_tdt][order_tdt]
    md_at_twt = np.interp(twt_s, tdt_twt_sorted, tdt_md_sorted, left=np.nan, right=np.nan)
    order_ai = np.argsort(md_axis[finite_ai])
    md_sorted = md_axis[finite_ai][order_ai]
    ai_sorted = ai[finite_ai][order_ai]
    ai_at_twt = np.interp(md_at_twt, md_sorted, ai_sorted, left=np.nan, right=np.nan)
    return np.where(ai_at_twt > 0.0, np.log(ai_at_twt), np.nan)


def _band_metrics(*, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray, dt_s: float) -> dict[str, float | int]:
    bands = (
        ("lowfreq", 0.0, 10.0),
        ("observable_band", 10.0, 20.0),
        ("highfreq_or_nullspace", 20.0, 80.0),
    )
    rows: dict[str, float | int] = {}
    for name, low, high in bands:
        ref = _bandpass_1d(reference, dt_s=dt_s, low_hz=low, high_hz=high)
        pred = _bandpass_1d(prediction, dt_s=dt_s, low_hz=low, high_hz=high)
        metrics = _ai_metrics(reference=ref, prediction=pred, mask=mask)
        rows[f"{name}_rmse"] = metrics["rmse"]
        rows[f"{name}_corr"] = metrics["corr"]
    return rows


def _bandpass_1d(values: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> np.ndarray:
    series = _fill_nonfinite_1d(np.asarray(values, dtype=np.float64))
    spectrum = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(series.size, d=dt_s)
    keep = freqs >= float(low_hz)
    if np.isfinite(high_hz):
        keep &= freqs < float(high_hz)
    spectrum = np.where(keep, spectrum, 0.0)
    return np.fft.irfft(spectrum, n=series.size)


def _fill_nonfinite_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if np.all(finite):
        return out
    if not np.any(finite):
        return np.zeros_like(out)
    coords = np.arange(out.size, dtype=np.float64)
    out[~finite] = np.interp(coords[~finite], coords[finite], out[finite])
    return out


def _ai_metrics(*, reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    if int(np.count_nonzero(valid)) < 2:
        return {"rmse": np.nan, "corr": np.nan}
    ref = np.asarray(reference, dtype=np.float64)[valid]
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    residual = pred - ref
    corr = float(np.corrcoef(ref, pred)[0, 1]) if np.std(ref) > 0.0 and np.std(pred) > 0.0 else np.nan
    return {"rmse": float(np.sqrt(np.mean(residual * residual))), "corr": corr}


def _rmse_after_constant_bias(*, rmse_before: float, mean_residual: float, bias: float) -> float:
    if not (np.isfinite(rmse_before) and np.isfinite(mean_residual) and np.isfinite(bias)):
        return np.nan
    mean_square = rmse_before * rmse_before
    adjusted = mean_square - 2.0 * bias * mean_residual + bias * bias
    return float(np.sqrt(max(adjusted, 0.0)))


def _forward_metric_value(frame: pd.DataFrame, *, role: str, metric: str) -> float:
    subset = frame[frame.get("source_role", pd.Series(dtype=str)).astype(str).eq(str(role))]
    if subset.empty:
        subset = frame[frame.get("model_role", pd.Series(dtype=str)).astype(str).eq(f"zero_shot_{role}")]
    if subset.empty or metric not in subset:
        return np.nan
    return float(pd.to_numeric(subset.iloc[0].get(metric), errors="coerce"))


def _reference_quality_flag(row: pd.Series) -> str:
    flags: list[str] = []
    if str(row.get("waveform_status", "")) != "ok":
        flags.append(f"waveform_status={row.get('waveform_status', '')}")
    if str(row.get("scale_status", "")) not in {"", "ok"}:
        flags.append(f"scale_status={row.get('scale_status', '')}")
    return ";".join(flags) if flags else "ok"


def _base_evidence_row(*, source: CalibrationSource, role: str, row: pd.Series) -> dict[str, Any]:
    return {
        "section_id": source.section_id,
        "well_name": str(row.get("well_name", "")),
        "model_role": role,
        "wellbore_class": str(row.get("wellbore_class", "")),
        "section_xy_distance_m": _float_value(row.get("section_xy_distance_m")),
        "nearest_section_trace": _int_value(row.get("nearest_section_trace"), default=-1),
        "sampling_mode": str(row.get("sampling_mode", "")),
        "sample_method": str(row.get("sample_method", "")),
        "r1_status": str(row.get("status", "")),
        "r1_well_ai_status": str(row.get("well_ai_status", "")),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return value


def _required_text(mapping: Mapping[str, Any], key: str, *, path: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ValueError(f"{path}.{key} must be explicit.")
    return value


def _prefix_dict(values: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _float_value(value: Any) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def _int_value(value: Any, *, default: int) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return default
    return int(numeric)


def _dt_from_axis(twt_s: np.ndarray) -> float:
    if twt_s.size < 2:
        raise ValueError("twt_s axis must contain at least two samples.")
    return float(np.median(np.diff(np.asarray(twt_s, dtype=np.float64))))


def _summary_status(model_bias: pd.DataFrame) -> str:
    if model_bias.empty:
        return "insufficient_evidence"
    statuses = set(str(x) for x in model_bias.get("status", pd.Series(dtype=str)).dropna())
    if statuses == {"no_op_no_well_evidence"}:
        return "ok_noop_no_well_evidence"
    return "ok" if statuses == {"ok"} else "partial" if "ok" in statuses else "insufficient_evidence"
