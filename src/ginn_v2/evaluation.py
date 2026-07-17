"""Synthetic patch prediction and reporting for GINN-v2.

The composable runner owns training; this module owns benchmark patch evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cup.synthetic.benchmark import SynthoseisBenchmark
from cup.impedance import canonical_lowpass
from cup.synthetic.reporting.metrics import regression_metrics
from cup.utils.io import write_json
from ginn_v2.checkpoint import load_checkpoint
from ginn_v2.contracts import PATCH_SMOKE_REPORT_SCHEMA_VERSION, PREDICTION_SCHEMA_VERSION
from ginn_v2.data import PatchDataset, _aligned_arrays
from ginn_v2.runtime import resolve_device


def canonical_closure_arrays(
    *,
    target_log_ai: np.ndarray,
    target_increment_log_ai: np.ndarray,
    predicted_increment_log_ai: np.ndarray,
    input_lfm_log_ai: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the two canonical/deployment closures from physical arrays.

    The benchmark stores the complete target and its canonical increment.  The
    canonical background is therefore reconstructed as ``target - increment``;
    no second filter is applied at evaluation time.
    """
    target = np.asarray(target_log_ai)
    target_increment = np.asarray(target_increment_log_ai)
    predicted_increment = np.asarray(predicted_increment_log_ai)
    input_lfm = np.asarray(input_lfm_log_ai)
    mask = np.asarray(valid_mask, dtype=bool)
    shapes = {array.shape for array in (target, target_increment, predicted_increment, input_lfm, mask)}
    if len(shapes) != 1:
        raise ValueError(f"Canonical closure arrays must have one shape; got {sorted(shapes, key=str)}")
    canonical_background = target - target_increment
    finite = mask & np.isfinite(target) & np.isfinite(canonical_background) & np.isfinite(target_increment)
    if np.any(finite & (np.abs(target - canonical_background - target_increment) > 1e-5)):
        raise ValueError("Stored target and increment do not reconstruct canonical background.")
    canonical_prediction = canonical_background + predicted_increment
    deployment_prediction = input_lfm + predicted_increment
    return {
        "canonical_background_log_ai": canonical_background,
        "canonical_closure_log_ai": canonical_prediction,
        "deployment_closure_log_ai": deployment_prediction,
    }


def predict_patches(
    *,
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    checkpoint_path: Path,
    output_dir: Path,
    split: str | None,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_columns = {
        "patch_id",
        "sample_id",
        "sample_kind",
        "split",
        "lateral_start",
        "lateral_stop",
        "twt_start",
        "twt_stop",
        "patch_lateral_samples",
        "patch_twt_samples",
    }
    missing_columns = sorted(required_columns - set(patch_index.columns))
    if missing_columns:
        raise ValueError(
            "Synthetic patch prediction received an incompatible patch index; "
            f"missing={missing_columns}, available={sorted(patch_index.columns)}."
        )
    model, checkpoint = load_checkpoint(checkpoint_path)
    if str(checkpoint.get("output_semantics") or "") != "predicted_increment_log_ai":
        raise ValueError(
            "GINN-v2 checkpoint does not use predicted_increment_log_ai output semantics."
        )
    if list(checkpoint.get("input_channels") or []) != [
        "seismic", "input_lfm_log_ai", "valid_mask"
    ]:
        raise ValueError("GINN-v2 checkpoint input channel contract is not canonical.")
    device, device_metadata = resolve_device(device_name)
    model.to(device)
    model.eval()
    selected = patch_index.copy()
    if split and split != "all":
        selected = selected[selected["split"].astype(str).eq(split)].copy()
    if selected.empty:
        raise ValueError(f"No patches selected for prediction split={split}.")
    dataset = PatchDataset(
        benchmark,
        selected,
        split=sorted(set(selected["split"].astype(str))),
        normalization=checkpoint["normalization"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    npz_path = output_dir / "predictions.npz"
    patch_count = int(len(selected))
    lateral_sizes = selected["patch_lateral_samples"].astype(int).unique()
    vertical_sizes = selected["patch_twt_samples"].astype(int).unique()
    if len(lateral_sizes) != 1 or len(vertical_sizes) != 1:
        raise ValueError("Prediction patch index must use one fixed patch shape.")
    patch_shape = (int(lateral_sizes[0]), int(vertical_sizes[0]))
    buffer_dir = output_dir / ".predict_buffers"
    if buffer_dir.exists():
        shutil.rmtree(buffer_dir, ignore_errors=True)
    buffer_dir.mkdir()
    buffers = {
        "predicted_increment_log_ai": np.lib.format.open_memmap(
            buffer_dir / "predicted_increment_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "predicted_log_ai": np.lib.format.open_memmap(
            buffer_dir / "predicted_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "target_increment_log_ai": np.lib.format.open_memmap(
            buffer_dir / "target_increment_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "canonical_background_log_ai": np.lib.format.open_memmap(
            buffer_dir / "canonical_background_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "target_log_ai": np.lib.format.open_memmap(
            buffer_dir / "target_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "valid_mask": np.lib.format.open_memmap(
            buffer_dir / "valid_mask.npy", mode="w+", dtype=bool,
            shape=(patch_count, *patch_shape),
        ),
        "input_lfm_log_ai": np.lib.format.open_memmap(
            buffer_dir / "input_lfm_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
    }
    patch_ids: list[str] = []
    sample_ids: list[str] = []
    offset = 0
    try:
        with torch.no_grad():
            for batch in loader:
                x = batch["input"].to(device)
                predicted_increment = model(x).detach().cpu().numpy()[:, 0]
                input_lfm = batch["input_lfm_log_ai"].numpy()[:, 0]
                prediction = input_lfm + predicted_increment
                batch_count = int(prediction.shape[0])
                end = offset + batch_count
                if end > patch_count:
                    raise RuntimeError("Prediction loader returned more patches than its index.")
                buffers["predicted_increment_log_ai"][offset:end] = predicted_increment.astype(np.float32)
                buffers["predicted_log_ai"][offset:end] = prediction.astype(np.float32)
                buffers["target_increment_log_ai"][offset:end] = batch["target_increment_log_ai"].numpy()[:, 0].astype(np.float32)
                buffers["canonical_background_log_ai"][offset:end] = batch[
                    "canonical_background_log_ai"
                ].numpy()[:, 0].astype(np.float32)
                buffers["target_log_ai"][offset:end] = batch["target_log_ai"].numpy()[:, 0].astype(np.float32)
                buffers["valid_mask"][offset:end] = batch["valid_mask"].numpy()[:, 0].astype(bool)
                buffers["input_lfm_log_ai"][offset:end] = input_lfm.astype(np.float32)
                patch_ids.extend(str(value) for value in batch["patch_id"])
                sample_ids.extend(str(value) for value in batch["sample_id"])
                offset = end
        if offset != patch_count:
            raise RuntimeError(
                f"Prediction loader returned {offset} patches but its index contains {patch_count}."
            )
        for array in buffers.values():
            array.flush()
        np.savez_compressed(
            npz_path,
            predicted_increment_log_ai=buffers["predicted_increment_log_ai"],
            predicted_log_ai=buffers["predicted_log_ai"],
            target_increment_log_ai=buffers["target_increment_log_ai"],
            canonical_background_log_ai=buffers["canonical_background_log_ai"],
            target_log_ai=buffers["target_log_ai"],
            valid_mask=buffers["valid_mask"],
            input_lfm_log_ai=buffers["input_lfm_log_ai"],
            patch_id=np.asarray(patch_ids),
            sample_id=np.asarray(sample_ids),
        )
    finally:
        buffers.clear()
        shutil.rmtree(buffer_dir, ignore_errors=True)
    selected = selected.set_index("patch_id").loc[patch_ids].reset_index()
    selected["prediction_row"] = np.arange(len(selected), dtype=int)
    selected["architecture_id"] = str(checkpoint["architecture"]["id"])
    index_path = output_dir / "prediction_index.csv"
    selected.to_csv(index_path, index=False)
    return {
        "status": "ok",
        "prediction_npz": npz_path,
        "prediction_index": index_path,
        "n_predictions": patch_count,
        "architecture_id": str(checkpoint["architecture"]["id"]),
        "model_info": dict(checkpoint["model_info"]),
        "normalization": dict(checkpoint["normalization"]),
        "device_metadata": device_metadata,
    }


def report_predictions(*, prediction_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = prediction_dir / "prediction_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Canonical GINN-v2 prediction summary not found: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if str(summary.get("schema_version") or "") != PREDICTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GINN-v2 prediction schema {summary.get('schema_version')!r}; "
            f"expected {PREDICTION_SCHEMA_VERSION}."
        )
    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    arrays = np.load(prediction_dir / "predictions.npz", allow_pickle=True)
    pred = arrays["predicted_log_ai"]
    pred_increment = arrays["predicted_increment_log_ai"]
    target = arrays["target_log_ai"]
    target_increment = arrays["target_increment_log_ai"]
    mask = arrays["valid_mask"].astype(bool)
    lfm = arrays["input_lfm_log_ai"]
    closure = canonical_closure_arrays(
        target_log_ai=target,
        target_increment_log_ai=target_increment,
        predicted_increment_log_ai=pred_increment,
        input_lfm_log_ai=lfm,
        valid_mask=mask,
    )
    canonical_background = closure["canonical_background_log_ai"]
    rows = []
    lfm_rows = []
    oracle_rows = []
    increment_rows = []
    canonical_rows = []
    for _, row in index.iterrows():
        i = int(row["prediction_row"])
        metrics = regression_metrics(target[i], pred[i], valid_mask=mask[i])
        rows.append({**row.to_dict(), "series_id": row.get("series_id", ""), **metrics})
        increment_metrics = regression_metrics(
            target_increment[i], pred_increment[i], valid_mask=mask[i]
        )
        increment_rows.append(
            {**row.to_dict(), "series_id": "predicted_increment_log_ai", **increment_metrics}
        )
        canonical_prediction = closure["canonical_closure_log_ai"][i]
        canonical_metrics = regression_metrics(
            target[i], canonical_prediction, valid_mask=mask[i]
        )
        canonical_rows.append(
            {**row.to_dict(), "series_id": "canonical_closure", **canonical_metrics}
        )
        lfm_metrics = regression_metrics(target[i], lfm[i], valid_mask=mask[i])
        lfm_rows.append({**row.to_dict(), "series_id": "input_lfm_log_ai", **lfm_metrics})
        oracle_metrics = regression_metrics(target[i], target[i], valid_mask=mask[i])
        oracle_rows.append({**row.to_dict(), "series_id": "oracle_target", **oracle_metrics})
    metrics_frame = pd.DataFrame.from_records(rows)
    metrics_path = output_dir / "model_patch_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    increment_frame = pd.DataFrame.from_records(increment_rows)
    increment_path = output_dir / "increment_patch_metrics.csv"
    increment_frame.to_csv(increment_path, index=False)
    canonical_frame = pd.DataFrame.from_records(canonical_rows)
    canonical_path = output_dir / "canonical_closure_patch_metrics.csv"
    canonical_frame.to_csv(canonical_path, index=False)
    lowpass_frame = _increment_lowpass_qc(
        prediction_dir=prediction_dir,
        index=index,
        predicted_increment=pred_increment,
        target_increment=target_increment,
        mask=mask,
    )
    lowpass_path = output_dir / "increment_lowpass_qc.csv"
    lowpass_frame.to_csv(lowpass_path, index=False)
    sample_kind_path = output_dir / "model_patch_metrics_by_sample_kind.csv"
    _grouped_patch_metrics(metrics_frame, ["sample_kind"]).to_csv(
        sample_kind_path,
        index=False,
    )
    geometry_path = output_dir / "model_patch_metrics_by_geometry.csv"
    _grouped_patch_metrics(metrics_frame, ["geometry_family"]).to_csv(geometry_path, index=False)
    view_path = output_dir / "model_patch_metrics_by_seismic_view.csv"
    _grouped_patch_metrics(metrics_frame, ["view_id"]).to_csv(view_path, index=False)
    holdout_role = metrics_frame.get("evaluation_role", pd.Series("", index=metrics_frame.index)).astype(str).str.casefold()
    holdout_geometry = metrics_frame.get("geometry_family", pd.Series("", index=metrics_frame.index)).astype(str).str.casefold()
    holdout_frame = metrics_frame[holdout_role.eq("geometry_holdout") | holdout_geometry.eq("pinchout")].copy()
    holdout_path = output_dir / "model_patch_metrics_geometry_holdout.csv"
    holdout_frame.to_csv(holdout_path, index=False)
    holdout_geometry_path = output_dir / "model_patch_metrics_geometry_holdout_by_family.csv"
    _grouped_patch_metrics(holdout_frame, ["geometry_family"]).to_csv(holdout_geometry_path, index=False)
    geometry_detail_frame = _geometry_patch_metrics(metrics_frame, pred, target, mask, prediction_dir)
    geometry_detail_path = output_dir / "model_geometry_patch_metrics.csv"
    geometry_detail_frame.to_csv(geometry_detail_path, index=False)
    geometry_aggregate_path = output_dir / "model_geometry_metrics_by_family.csv"
    _grouped_geometry_metrics(geometry_detail_frame, ["geometry_family"]).to_csv(
        geometry_aggregate_path,
        index=False,
    )
    stitched = _stitch_realization_metrics(index, pred, target, mask, lfm)
    stitched_uniform_path = output_dir / "model_realization_metrics_uniform.csv"
    stitched["uniform"].to_csv(stitched_uniform_path, index=False)
    stitched_center_path = output_dir / "model_realization_metrics_center_crop.csv"
    stitched["center_crop"].to_csv(stitched_center_path, index=False)
    lfm_frame = pd.DataFrame.from_records(lfm_rows)
    lfm_path = output_dir / "lfm_patch_metrics.csv"
    lfm_frame.to_csv(lfm_path, index=False)
    oracle_frame = pd.DataFrame.from_records(oracle_rows)
    oracle_path = output_dir / "oracle_patch_metrics.csv"
    oracle_frame.to_csv(oracle_path, index=False)
    model_aggregate = _aggregate(metrics_frame)
    lfm_aggregate = _aggregate(lfm_frame)
    oracle_aggregate = _aggregate(oracle_frame)
    report = {
        "schema_version": PATCH_SMOKE_REPORT_SCHEMA_VERSION,
        "report_scope": "patch_smoke",
        "not_synthoseis_lite_report": True,
        "status": "ok",
        "n_patches": int(len(metrics_frame)),
        "aggregate": model_aggregate,
        "increment_aggregate": _aggregate(increment_frame),
        "canonical_closure_aggregate": _aggregate(canonical_frame),
        "increment_lowpass_qc": {
            "status": "ok" if not lowpass_frame.empty else "not_computed",
            "n_rows": int(len(lowpass_frame)),
        },
        "lfm_aggregate": lfm_aggregate,
        "oracle_aggregate": oracle_aggregate,
        "rmse_improvement_pct_vs_lfm": _rmse_improvement_pct(model_aggregate, lfm_aggregate),
        "geometry_aggregate": _aggregate_geometry(geometry_detail_frame),
        "geometry_holdout_aggregate": _aggregate(holdout_frame),
        "pinchout_holdout_n_patches": int(len(holdout_frame)),
        "realization_uniform_aggregate": _aggregate(
            stitched["uniform"][stitched["uniform"]["series_id"].astype(str).ne("input_lfm_log_ai")]
        ),
        "realization_center_crop_aggregate": _aggregate(
            stitched["center_crop"][
                stitched["center_crop"]["series_id"].astype(str).ne("input_lfm_log_ai")
            ]
        ),
    }
    report_path = output_dir / "model_report_card.json"
    write_json(report_path, report)
    return {
        "status": "ok",
        "model_patch_metrics": metrics_path,
        "increment_patch_metrics": increment_path,
        "canonical_closure_patch_metrics": canonical_path,
        "increment_lowpass_qc": lowpass_path,
        "model_patch_metrics_by_sample_kind": sample_kind_path,
        "model_patch_metrics_by_geometry": geometry_path,
        "model_patch_metrics_by_seismic_view": view_path,
        "model_patch_metrics_geometry_holdout": holdout_path,
        "model_patch_metrics_geometry_holdout_by_family": holdout_geometry_path,
        "model_geometry_patch_metrics": geometry_detail_path,
        "model_geometry_metrics_by_family": geometry_aggregate_path,
        "model_realization_metrics_uniform": stitched_uniform_path,
        "model_realization_metrics_center_crop": stitched_center_path,
        "lfm_patch_metrics": lfm_path,
        "oracle_patch_metrics": oracle_path,
        "model_report_card": report_path,
    }


def _clean_report_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "status" not in frame:
        return {"n_ok": 0}
    ok = frame[frame["status"].eq("ok")]
    return {
        "n_ok": int(len(ok)),
        "mean_rmse": float(ok["rmse"].mean()) if not ok.empty else float("nan"),
        "mean_nrmse": float(ok["nrmse"].mean()) if not ok.empty else float("nan"),
        "median_corr": float(ok["corr"].median()) if not ok.empty else float("nan"),
    }


def _grouped_patch_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty or not set(keys).issubset(frame.columns):
        return pd.DataFrame(columns=[*keys, "n_ok", "mean_rmse", "mean_nrmse", "median_corr"])
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(_aggregate(group))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _increment_lowpass_qc(
    *,
    prediction_dir: Path,
    index: pd.DataFrame,
    predicted_increment: np.ndarray,
    target_increment: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    """Report conservative low-pass response ratios on a bounded patch sample.

    Labels are filtered on the complete benchmark trace.  Predicted patches
    are filtered only when their local valid segment satisfies the canonical
    operator's length contract; otherwise the row is reported as
    ``not_computed`` rather than being used as a gate.
    """
    if len(index) > 2048:
        return pd.DataFrame(
            [{"status": "not_computed", "reason": "prediction_patch_count_over_2048"}]
        )
    benchmark_dir = _benchmark_dir_from_prediction(prediction_dir)
    if benchmark_dir is None:
        return pd.DataFrame([{"status": "not_computed", "reason": "benchmark_dir_missing"}])
    benchmark = SynthoseisBenchmark(benchmark_dir)
    rows: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    for _, row in index.iterrows():
        i = int(row["prediction_row"])
        sample_id = str(row["sample_id"])
        sample = cache.get(sample_id)
        if sample is None:
            sample = benchmark.load_sample(sample_id)
            cache[sample_id] = sample
        axis = np.asarray(sample.sample_axis, dtype=np.float64)
        full_mask = np.asarray(sample.valid_mask, dtype=bool)
        target_full = np.asarray(sample.target_increment_log_ai, dtype=np.float64)
        target_lp = canonical_lowpass(
            target_full,
            axis,
            benchmark.increment_contract,
            valid_mask=full_mask,
        )
        sl = _patch_slice(row)
        target_patch = target_full[sl]
        target_lp_patch = target_lp[sl]
        valid = mask[i] & np.isfinite(target_patch) & np.isfinite(target_lp_patch)
        row_out: dict[str, Any] = {
            "patch_id": row.get("patch_id", ""),
            "sample_id": sample_id,
            "status": "not_computed",
            "target_lowpass_output_power_ratio": float("nan"),
            "predicted_lowpass_output_power_ratio": float("nan"),
        }
        if np.count_nonzero(valid) >= 2:
            target_rms = float(np.sqrt(np.mean(target_patch[valid] ** 2)))
            target_lp_rms = float(np.sqrt(np.mean(target_lp_patch[valid] ** 2)))
            try:
                patch_axis = axis[int(row["twt_start"]):int(row["twt_stop"])]
                predicted_lp = canonical_lowpass(
                    np.asarray(predicted_increment[i], dtype=np.float64),
                    patch_axis,
                    benchmark.increment_contract,
                    valid_mask=mask[i],
                )
                predicted_values = np.asarray(predicted_increment[i], dtype=np.float64)
                predicted_rms = float(np.sqrt(np.mean(predicted_values[valid] ** 2)))
                predicted_lp_rms = float(np.sqrt(np.mean(predicted_lp[valid] ** 2)))
                row_out.update(
                    {
                        "status": "ok",
                        "target_lowpass_output_power_ratio": (
                            (target_lp_rms * target_lp_rms) / (target_rms * target_rms)
                            if target_rms > 0.0 else float("nan")
                        ),
                        "predicted_lowpass_output_power_ratio": (
                            (predicted_lp_rms * predicted_lp_rms) / (predicted_rms * predicted_rms)
                            if predicted_rms > 0.0 else float("nan")
                        ),
                    }
                )
            except ValueError as exc:
                row_out["reason"] = str(exc)
        rows.append(row_out)
    return pd.DataFrame.from_records(rows)


def _geometry_patch_metrics(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prediction_dir: Path,
) -> pd.DataFrame:
    """Compute geometry-focused patch metrics from frozen benchmark masks."""

    benchmark_dir = _benchmark_dir_from_prediction(prediction_dir)
    if benchmark_dir is None:
        return pd.DataFrame()
    h5_path = benchmark_dir / "synthetic_benchmark.h5"
    if not h5_path.is_file():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    geometry_cache: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as h5:
        for _, row in index.iterrows():
            prediction_row = int(row["prediction_row"])
            root = _root_from_group_path(str(row.get("hdf5_group", "")))
            if not root:
                continue
            geometry = geometry_cache.get(root)
            if geometry is None:
                geometry = _read_geometry_masks(h5, root)
                geometry_cache[root] = geometry
            sl = _patch_slice(row, offset=0)
            boundary = geometry["boundary_mask_model"][sl]
            event = geometry["geometry_event_mask_model"][sl]
            valid = (
                mask[prediction_row].astype(bool)
                & np.isfinite(pred[prediction_row])
                & np.isfinite(target[prediction_row])
            )
            residual = pred[prediction_row] - target[prediction_row]
            boundary_band = _dilate_mask(boundary, lateral_radius=1, twt_radius=2) & valid
            non_boundary = valid & ~boundary_band
            event_valid = event & valid
            non_event = valid & ~event
            lateral = _lateral_gradient_metrics(
                prediction=pred[prediction_row],
                target=target[prediction_row],
                valid=valid,
            )
            rows.append(
                {
                    "patch_id": row.get("patch_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "sample_kind": row.get("sample_kind", ""),
                    "split": row.get("split", ""),
                    "geometry_family": row.get("geometry_family", ""),
                    "scenario_id": row.get("scenario_id", ""),
                    "duration_mode": row.get("duration_mode", ""),
                    "geometry_metric_semantics": (
                        "boundary uses dilated truth/boundary_mask_model; "
                        "event uses model-grid projection of truth/geometry_event_mask_highres"
                    ),
                    **_region_metrics("all", residual, valid),
                    **_region_metrics("boundary", residual, boundary_band),
                    **_region_metrics("non_boundary", residual, non_boundary),
                    **_region_metrics("event", residual, event_valid),
                    **_region_metrics("non_event", residual, non_event),
                    **lateral,
                }
            )
    return pd.DataFrame.from_records(rows)


def _benchmark_dir_from_prediction(prediction_dir: Path) -> Path | None:
    import json

    manifest_path = prediction_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_dir = manifest.get("benchmark_dir")
    if not benchmark_dir:
        return None
    path = Path(benchmark_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _root_from_group_path(group_path: str) -> str:
    group_path = str(group_path).strip()
    if not group_path:
        return ""
    for marker in ("/seismic_views/",):
        if marker in group_path:
            return group_path.split(marker, maxsplit=1)[0]
    return group_path


def _read_geometry_masks(h5: h5py.File, root: str) -> dict[str, np.ndarray]:
    truth = h5[f"{root}/truth"]
    boundary = np.asarray(truth["boundary_mask_model"][()], dtype=bool)
    event_highres = np.asarray(truth["geometry_event_mask_highres"][()], dtype=bool)
    event_model = _highres_mask_to_model(event_highres, boundary.shape)
    return {
        "boundary_mask_model": boundary,
        "geometry_event_mask_model": event_model,
    }


def _highres_mask_to_model(mask_highres: np.ndarray, model_shape: tuple[int, int]) -> np.ndarray:
    mask_highres = np.asarray(mask_highres, dtype=bool)
    n_lateral, n_model = model_shape
    if mask_highres.shape[0] != n_lateral:
        return np.zeros(model_shape, dtype=bool)
    if mask_highres.shape[1] == n_model:
        return mask_highres.copy()
    factor = mask_highres.shape[1] / float(n_model)
    result = np.zeros(model_shape, dtype=bool)
    for model_index in range(n_model):
        start = int(np.floor(model_index * factor))
        stop = int(np.floor((model_index + 1) * factor))
        stop = max(stop, start + 1)
        start = min(start, mask_highres.shape[1])
        stop = min(stop, mask_highres.shape[1])
        if start < stop:
            result[:, model_index] = np.any(mask_highres[:, start:stop], axis=1)
    return result


def _patch_slice(row: Mapping[str, Any], *, offset: int = 0) -> tuple[slice, slice]:
    return (
        slice(int(row["lateral_start"]), int(row["lateral_stop"])),
        slice(int(row["twt_start"]) + offset, int(row["twt_stop"]) + offset),
    )


def _dilate_mask(mask: np.ndarray, *, lateral_radius: int, twt_radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros_like(mask, dtype=bool)
    for dl in range(-int(lateral_radius), int(lateral_radius) + 1):
        src_l0 = max(0, -dl)
        src_l1 = min(mask.shape[0], mask.shape[0] - dl)
        dst_l0 = max(0, dl)
        dst_l1 = min(mask.shape[0], mask.shape[0] + dl)
        for dt in range(-int(twt_radius), int(twt_radius) + 1):
            src_t0 = max(0, -dt)
            src_t1 = min(mask.shape[1], mask.shape[1] - dt)
            dst_t0 = max(0, dt)
            dst_t1 = min(mask.shape[1], mask.shape[1] + dt)
            result[dst_l0:dst_l1, dst_t0:dst_t1] |= mask[src_l0:src_l1, src_t0:src_t1]
    return result


def _region_metrics(prefix: str, residual: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return {
            f"{prefix}_n_valid": 0,
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_bias": float("nan"),
        }
    values = np.asarray(residual, dtype=np.float64)[valid]
    return {
        f"{prefix}_n_valid": n_valid,
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_mae": float(np.mean(np.abs(values))),
        f"{prefix}_bias": float(np.mean(values)),
    }


def _lateral_gradient_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    pair_valid = valid[1:, :] & valid[:-1, :]
    n_valid = int(np.count_nonzero(pair_valid))
    if n_valid == 0:
        return {
            "lateral_gradient_n_valid": 0,
            "lateral_gradient_rmse": float("nan"),
            "lateral_gradient_mae": float("nan"),
            "lateral_gradient_corr": float("nan"),
        }
    pred_grad = np.asarray(prediction[1:, :] - prediction[:-1, :], dtype=np.float64)
    target_grad = np.asarray(target[1:, :] - target[:-1, :], dtype=np.float64)
    residual = pred_grad - target_grad
    metrics = regression_metrics(target_grad, pred_grad, valid_mask=pair_valid)
    return {
        "lateral_gradient_n_valid": n_valid,
        "lateral_gradient_rmse": float(np.sqrt(np.mean(residual[pair_valid] ** 2))),
        "lateral_gradient_mae": float(np.mean(np.abs(residual[pair_valid]))),
        "lateral_gradient_corr": metrics.get("corr", float("nan")),
    }


def _aggregate_geometry(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_ok": 0}
    ok = frame[pd.to_numeric(frame.get("all_n_valid", 0), errors="coerce").fillna(0).gt(0)]
    return {
        "n_ok": int(len(ok)),
        "mean_boundary_rmse": _safe_mean(ok, "boundary_rmse"),
        "mean_event_rmse": _safe_mean(ok, "event_rmse"),
        "mean_lateral_gradient_rmse": _safe_mean(ok, "lateral_gradient_rmse"),
        "median_lateral_gradient_corr": _safe_median(ok, "lateral_gradient_corr"),
    }


def _grouped_geometry_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    columns = [
        *keys,
        "n_ok",
        "mean_all_rmse",
        "mean_boundary_rmse",
        "mean_non_boundary_rmse",
        "mean_event_rmse",
        "mean_non_event_rmse",
        "mean_lateral_gradient_rmse",
        "median_lateral_gradient_corr",
    ]
    if frame.empty or not set(keys).issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        ok = group[pd.to_numeric(group.get("all_n_valid", 0), errors="coerce").fillna(0).gt(0)]
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "n_ok": int(len(ok)),
                "mean_all_rmse": _safe_mean(ok, "all_rmse"),
                "mean_boundary_rmse": _safe_mean(ok, "boundary_rmse"),
                "mean_non_boundary_rmse": _safe_mean(ok, "non_boundary_rmse"),
                "mean_event_rmse": _safe_mean(ok, "event_rmse"),
                "mean_non_event_rmse": _safe_mean(ok, "non_event_rmse"),
                "mean_lateral_gradient_rmse": _safe_mean(ok, "lateral_gradient_rmse"),
                "median_lateral_gradient_corr": _safe_median(ok, "lateral_gradient_corr"),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=columns)


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float("nan")


def _stitch_realization_metrics(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    lfm: np.ndarray,
) -> dict[str, pd.DataFrame]:
    return {
        "uniform": _stitch_one_strategy(
            index,
            pred,
            target,
            mask,
            lfm,
            strategy="uniform",
        ),
        "center_crop": _stitch_one_strategy(
            index,
            pred,
            target,
            mask,
            lfm,
            strategy="center_crop",
        ),
    }


def _stitch_one_strategy(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    lfm: np.ndarray,
    *,
    strategy: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if index.empty:
        return pd.DataFrame()
    for sample_id, group in index.groupby("sample_id", dropna=False):
        n_lateral = int(pd.to_numeric(group["lateral_stop"], errors="coerce").max())
        n_twt = int(pd.to_numeric(group["twt_stop"], errors="coerce").max())
        if n_lateral <= 0 or n_twt <= 0:
            continue
        pred_sum = np.zeros((n_lateral, n_twt), dtype=np.float64)
        target_sum = np.zeros((n_lateral, n_twt), dtype=np.float64)
        lfm_sum = np.zeros((n_lateral, n_twt), dtype=np.float64)
        counts = np.zeros((n_lateral, n_twt), dtype=np.float64)
        first_row = group.iloc[0]
        for _, row in group.iterrows():
            patch_index = int(row["prediction_row"])
            local, dest = _stitch_slices(row, strategy=strategy)
            patch_valid = mask[patch_index][local]
            if not np.any(patch_valid):
                continue
            weights = patch_valid.astype(np.float64)
            pred_sum[dest] += np.where(patch_valid, pred[patch_index][local], 0.0)
            target_sum[dest] += np.where(patch_valid, target[patch_index][local], 0.0)
            lfm_sum[dest] += np.where(patch_valid, lfm[patch_index][local], 0.0)
            counts[dest] += weights
        valid = counts > 0.0
        if not np.any(valid):
            continue
        pred_full = np.divide(pred_sum, counts, out=np.zeros_like(pred_sum), where=valid)
        target_full = np.divide(target_sum, counts, out=np.zeros_like(target_sum), where=valid)
        lfm_full = np.divide(lfm_sum, counts, out=np.zeros_like(lfm_sum), where=valid)
        base = {
            "sample_id": sample_id,
            "sample_kind": first_row.get("sample_kind", ""),
            "split": first_row.get("split", ""),
            "geometry_family": first_row.get("geometry_family", ""),
            "scenario_id": first_row.get("scenario_id", ""),
            "duration_mode": first_row.get("duration_mode", ""),
            "stitch_strategy": strategy,
            "n_source_patches": int(len(group)),
            "stitched_lateral_samples": n_lateral,
            "stitched_twt_samples": n_twt,
            "coverage_fraction": float(np.mean(valid)),
            "mean_patch_overlap": float(np.mean(counts[valid])),
        }
        rows.append(
            {
                **base,
                "series_id": first_row.get("series_id", ""),
                **regression_metrics(target_full, pred_full, valid_mask=valid),
            }
        )
        rows.append(
            {
                **base,
                "series_id": "input_lfm_log_ai",
                **regression_metrics(target_full, lfm_full, valid_mask=valid),
            }
        )
    return pd.DataFrame.from_records(rows)


def _stitch_slices(row: Mapping[str, Any], *, strategy: str) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    lateral_start = int(row["lateral_start"])
    lateral_stop = int(row["lateral_stop"])
    twt_start = int(row["twt_start"])
    twt_stop = int(row["twt_stop"])
    patch_lateral = lateral_stop - lateral_start
    patch_twt = twt_stop - twt_start
    if strategy == "uniform":
        local_l0, local_l1 = 0, patch_lateral
        local_t0, local_t1 = 0, patch_twt
    elif strategy == "center_crop":
        local_l0, local_l1 = _center_crop_bounds(patch_lateral)
        local_t0, local_t1 = _center_crop_bounds(patch_twt)
    else:
        raise ValueError(f"Unsupported stitch strategy: {strategy}")
    return (
        (slice(local_l0, local_l1), slice(local_t0, local_t1)),
        (
            slice(lateral_start + local_l0, lateral_start + local_l1),
            slice(twt_start + local_t0, twt_start + local_t1),
        ),
    )


def _center_crop_bounds(size: int) -> tuple[int, int]:
    if size <= 4:
        return 0, size
    margin = max(1, size // 4)
    start = margin
    stop = size - margin
    if stop <= start:
        return 0, size
    return start, stop


def _rmse_improvement_pct(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    model_rmse = float(model.get("mean_rmse", float("nan")))
    baseline_rmse = float(baseline.get("mean_rmse", float("nan")))
    if not np.isfinite(model_rmse) or not np.isfinite(baseline_rmse) or baseline_rmse == 0.0:
        return float("nan")
    return float((baseline_rmse - model_rmse) / baseline_rmse * 100.0)


__all__ = ["canonical_closure_arrays", "predict_patches", "report_predictions"]
