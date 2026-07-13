"""Synthetic patch prediction and reporting for GINN-v2.

The composable runner owns training; this module owns benchmark patch evaluation.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cup.synthetic.benchmark import SynthoseisBenchmark
from cup.synthetic.reporting.metrics import regression_metrics
from cup.utils.io import write_json
from ginn_v2.checkpoint import load_checkpoint
from ginn_v2.contracts import PATCH_SMOKE_REPORT_SCHEMA_VERSION
from ginn_v2.data import PatchDataset, _aligned_arrays
from ginn_v2.runtime import resolve_device


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
    model, checkpoint = load_checkpoint(checkpoint_path)
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
        "pred_log_ai": np.lib.format.open_memmap(
            buffer_dir / "pred_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "target_log_ai": np.lib.format.open_memmap(
            buffer_dir / "target_log_ai.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "valid_mask_model": np.lib.format.open_memmap(
            buffer_dir / "valid_mask_model.npy", mode="w+", dtype=bool,
            shape=(patch_count, *patch_shape),
        ),
        "lfm_controlled_degraded": np.lib.format.open_memmap(
            buffer_dir / "lfm_controlled_degraded.npy", mode="w+", dtype=np.float32,
            shape=(patch_count, *patch_shape),
        ),
        "lfm_ideal": np.lib.format.open_memmap(
            buffer_dir / "lfm_ideal.npy", mode="w+", dtype=np.float32,
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
                pred_delta = model(x).detach().cpu().numpy()[:, 0]
                lfm = batch["lfm"].numpy()[:, 0]
                prediction = lfm + pred_delta
                batch_count = int(prediction.shape[0])
                end = offset + batch_count
                if end > patch_count:
                    raise RuntimeError("Prediction loader returned more patches than its index.")
                buffers["pred_log_ai"][offset:end] = prediction.astype(np.float32)
                buffers["target_log_ai"][offset:end] = batch["target_log_ai"].numpy()[:, 0].astype(np.float32)
                buffers["valid_mask_model"][offset:end] = batch["valid_mask"].numpy()[:, 0].astype(bool)
                buffers["lfm_controlled_degraded"][offset:end] = lfm.astype(np.float32)
                buffers["lfm_ideal"][offset:end] = batch["lfm_ideal"].numpy()[:, 0].astype(np.float32)
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
            pred_log_ai=buffers["pred_log_ai"],
            target_log_ai=buffers["target_log_ai"],
            valid_mask_model=buffers["valid_mask_model"],
            lfm_controlled_degraded=buffers["lfm_controlled_degraded"],
            lfm_ideal=buffers["lfm_ideal"],
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
    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    arrays = np.load(prediction_dir / "predictions.npz", allow_pickle=True)
    pred = arrays["pred_log_ai"]
    target = arrays["target_log_ai"]
    mask = arrays["valid_mask_model"].astype(bool)
    lfm = arrays["lfm_controlled_degraded"]
    lfm_ideal = (
        arrays["lfm_ideal"]
        if "lfm_ideal" in arrays
        else _load_lfm_ideal_patches(prediction_dir, index, target.shape)
    )
    rows = []
    lfm_rows = []
    lfm_ideal_rows = []
    oracle_rows = []
    for _, row in index.iterrows():
        i = int(row["prediction_row"])
        metrics = regression_metrics(target[i], pred[i], valid_mask=mask[i])
        rows.append({**row.to_dict(), "series_id": row.get("series_id", ""), **metrics})
        lfm_metrics = regression_metrics(target[i], lfm[i], valid_mask=mask[i])
        lfm_rows.append({**row.to_dict(), "series_id": "lfm_controlled_degraded", **lfm_metrics})
        lfm_ideal_metrics = regression_metrics(target[i], lfm_ideal[i], valid_mask=mask[i])
        lfm_ideal_rows.append({**row.to_dict(), "series_id": "lfm_ideal", **lfm_ideal_metrics})
        oracle_metrics = regression_metrics(target[i], target[i], valid_mask=mask[i])
        oracle_rows.append({**row.to_dict(), "series_id": "oracle_target", **oracle_metrics})
    metrics_frame = pd.DataFrame.from_records(rows)
    metrics_path = output_dir / "model_patch_metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    sample_kind_path = output_dir / "model_patch_metrics_by_sample_kind.csv"
    _grouped_patch_metrics(metrics_frame, ["sample_kind"]).to_csv(
        sample_kind_path,
        index=False,
    )
    geometry_path = output_dir / "model_patch_metrics_by_geometry.csv"
    _grouped_patch_metrics(metrics_frame, ["geometry_family"]).to_csv(geometry_path, index=False)
    mismatch_path = output_dir / "model_patch_metrics_by_mismatch_family.csv"
    _grouped_patch_metrics(metrics_frame, ["seismic_mismatch_family"]).to_csv(mismatch_path, index=False)
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
    lfm_ideal_frame = pd.DataFrame.from_records(lfm_ideal_rows)
    lfm_ideal_path = output_dir / "lfm_ideal_patch_metrics.csv"
    lfm_ideal_frame.to_csv(lfm_ideal_path, index=False)
    oracle_frame = pd.DataFrame.from_records(oracle_rows)
    oracle_path = output_dir / "oracle_patch_metrics.csv"
    oracle_frame.to_csv(oracle_path, index=False)
    model_aggregate = _aggregate(metrics_frame)
    lfm_aggregate = _aggregate(lfm_frame)
    lfm_ideal_aggregate = _aggregate(lfm_ideal_frame)
    oracle_aggregate = _aggregate(oracle_frame)
    report = {
        "schema_version": PATCH_SMOKE_REPORT_SCHEMA_VERSION,
        "report_scope": "patch_smoke",
        "not_synthoseis_lite_report": True,
        "status": "ok",
        "n_patches": int(len(metrics_frame)),
        "aggregate": model_aggregate,
        "lfm_aggregate": lfm_aggregate,
        "lfm_ideal_aggregate": lfm_ideal_aggregate,
        "oracle_aggregate": oracle_aggregate,
        "rmse_improvement_pct_vs_lfm": _rmse_improvement_pct(model_aggregate, lfm_aggregate),
        "geometry_aggregate": _aggregate_geometry(geometry_detail_frame),
        "geometry_holdout_aggregate": _aggregate(holdout_frame),
        "pinchout_holdout_n_patches": int(len(holdout_frame)),
        "realization_uniform_aggregate": _aggregate(
            stitched["uniform"][stitched["uniform"]["series_id"].astype(str).ne("lfm_controlled_degraded")]
        ),
        "realization_center_crop_aggregate": _aggregate(
            stitched["center_crop"][
                stitched["center_crop"]["series_id"].astype(str).ne("lfm_controlled_degraded")
            ]
        ),
    }
    report_path = output_dir / "model_report_card.json"
    write_json(report_path, report)
    return {
        "status": "ok",
        "model_patch_metrics": metrics_path,
        "model_patch_metrics_by_sample_kind": sample_kind_path,
        "model_patch_metrics_by_geometry": geometry_path,
        "model_patch_metrics_by_mismatch_family": mismatch_path,
        "model_patch_metrics_geometry_holdout": holdout_path,
        "model_patch_metrics_geometry_holdout_by_family": holdout_geometry_path,
        "model_geometry_patch_metrics": geometry_detail_path,
        "model_geometry_metrics_by_family": geometry_aggregate_path,
        "model_realization_metrics_uniform": stitched_uniform_path,
        "model_realization_metrics_center_crop": stitched_center_path,
        "lfm_patch_metrics": lfm_path,
        "lfm_ideal_patch_metrics": lfm_ideal_path,
        "oracle_patch_metrics": oracle_path,
        "model_report_card": report_path,
    }


def _paired_probe_rows(
    index: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prediction_dir: Path,
) -> list[dict[str, Any]]:
    by_patch = {str(row["patch_id"]): row for _, row in index.iterrows()}
    rows = []
    dt_s = _model_dt_s(prediction_dir)
    for _, row in index.iterrows():
        pair_id = _clean_report_text(row.get("paired_zero_patch_id"))
        if not pair_id or pair_id not in by_patch:
            continue
        amp = row.get("probe_amplitude_multiplier", "")
        try:
            if float(amp) == 0.0:
                continue
        except (TypeError, ValueError):
            pass
        i = int(row["prediction_row"])
        j = int(by_patch[pair_id]["prediction_row"])
        valid = mask[i] & mask[j]
        target_diff = target[i] - target[j]
        pred_diff = pred[i] - pred[j]
        metrics = regression_metrics(target_diff, pred_diff, valid_mask=valid)
        amp_phase = _amplitude_phase_metrics(
            target_diff=target_diff,
            pred_diff=pred_diff,
            valid=valid,
            frequency_hz=_safe_float(row.get("probe_frequency_hz")),
            dt_s=dt_s,
        )
        rows.append(
            {
                "patch_id": row["patch_id"],
                "paired_zero_patch_id": pair_id,
                "sample_id": row.get("sample_id", ""),
                "paired_zero_sample_id": row.get("paired_zero_sample_id", ""),
                "sample_kind": row.get("sample_kind", ""),
                "source_sample_id": row.get("source_sample_id", ""),
                "source_sample_kind": row.get("source_sample_kind", ""),
                "seismic_variant_id": row.get("seismic_variant_id", ""),
                "seismic_mismatch_family": row.get("seismic_mismatch_family", ""),
                "probe_frequency_hz": row.get("probe_frequency_hz", ""),
                "probe_phase": row.get("probe_phase", ""),
                "probe_lateral_shape": row.get("probe_lateral_shape", ""),
                "probe_amplitude_multiplier": amp,
                "probe_metric_semantics": (
                    "paired_probe_increment_error_under_seismic_mismatch"
                    if str(row.get("sample_kind", "")) == "frequency_probe_seismic_variant"
                    else "paired_probe_increment_error"
                ),
                **metrics,
                **amp_phase,
            }
        )
    return rows


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


def _load_lfm_ideal_patches(
    prediction_dir: Path,
    index: pd.DataFrame,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    benchmark_dir = _benchmark_dir_from_prediction(prediction_dir)
    if benchmark_dir is None:
        raise ValueError("Cannot load lfm_ideal: prediction manifest lacks benchmark_dir.")
    benchmark = SynthoseisBenchmark(benchmark_dir)
    patches: list[np.ndarray] = []
    for _, row in index.iterrows():
        sample = benchmark.load_sample(str(row["sample_id"]))
        target, _, _, _ = _aligned_arrays(sample)
        lfm_ideal = np.asarray(sample.priors["lfm_ideal"], dtype=np.float32)
        if lfm_ideal.shape != target.shape:
            raise ValueError(
                f"lfm_ideal/target shape mismatch for {sample.sample_id}: "
                f"{lfm_ideal.shape} vs {target.shape}"
            )
        patches.append(lfm_ideal[_patch_slice(row, offset=0)].astype(np.float32))
    result = np.stack(patches, axis=0)
    if result.shape != expected_shape:
        raise ValueError(f"lfm_ideal patch shape mismatch: {result.shape} vs {expected_shape}")
    return result


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
    for marker in ("/probes/", "/seismic_variants/"):
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
                "series_id": "lfm_controlled_degraded",
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


def _filter_unsupported_zero_x(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "operator_support" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["operator_support"].astype(str).eq("unsupported")].copy()


def _zero_x_false_frequency_energy(
    metrics_frame: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    prediction_dir: Path,
) -> pd.DataFrame:
    zero = _zero_x_false_prediction_error(metrics_frame, prediction_dir)
    if zero.empty:
        return zero
    dt_s = _model_dt_s(prediction_dir)
    rows: list[dict[str, Any]] = []
    for _, row in zero.iterrows():
        frequency = _safe_float(row.get("probe_frequency_hz"))
        prediction_row = int(row["prediction_row"])
        metrics = _frequency_projection_metrics(
            residual=pred[prediction_row] - target[prediction_row],
            target=target[prediction_row],
            valid=mask[prediction_row],
            frequency_hz=frequency,
            dt_s=dt_s,
        )
        rows.append(
            {
                **row.to_dict(),
                "false_energy_semantics": (
                    "weighted least-squares sin/cos projection of pred-target residual "
                    "at the 0x probe frequency"
                ),
                "model_dt_s": dt_s,
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def _frequency_projection_metrics(
    *,
    residual: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    frequency_hz: float,
    dt_s: float,
) -> dict[str, Any]:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return {"energy_status": "invalid_frequency"}
    residual = np.asarray(residual, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(residual) & np.isfinite(target)
    n_valid = int(np.count_nonzero(valid))
    if residual.ndim != 2 or n_valid < 8:
        return {"energy_status": "insufficient_valid_samples", "energy_n_valid": n_valid}
    nt = residual.shape[1]
    t = np.arange(nt, dtype=np.float64) * float(dt_s)
    taper = _tukey_window(nt, alpha=0.5)
    sin_basis = np.sin(2.0 * np.pi * frequency_hz * t)
    cos_basis = np.cos(2.0 * np.pi * frequency_hz * t)
    weights = valid.astype(np.float64) * taper[None, :]
    residual_amp = _weighted_sincos_amplitude(residual, weights, sin_basis, cos_basis)
    target_amp = _weighted_sincos_amplitude(target, weights, sin_basis, cos_basis)
    false_rms = residual_amp / np.sqrt(2.0) if np.isfinite(residual_amp) else float("nan")
    target_rms = target_amp / np.sqrt(2.0) if np.isfinite(target_amp) else float("nan")
    return {
        "energy_status": "ok" if np.isfinite(false_rms) else "invalid_projection",
        "energy_n_valid": n_valid,
        "false_frequency_amplitude": float(residual_amp),
        "false_frequency_rms": float(false_rms),
        "target_frequency_amplitude": float(target_amp),
        "target_frequency_rms": float(target_rms),
        "false_to_target_frequency_rms": (
            float(false_rms / target_rms)
            if np.isfinite(false_rms) and np.isfinite(target_rms) and target_rms > 0.0
            else float("nan")
        ),
    }


def _weighted_sincos_amplitude(
    values: np.ndarray,
    weights: np.ndarray,
    sin_basis: np.ndarray,
    cos_basis: np.ndarray,
) -> float:
    coeffs = _weighted_sincos_coefficients(values, weights, sin_basis, cos_basis)
    if coeffs is None:
        return float("nan")
    return float(np.sqrt(coeffs[0] ** 2 + coeffs[1] ** 2))


def _weighted_sincos_coefficients(
    values: np.ndarray,
    weights: np.ndarray,
    sin_basis: np.ndarray,
    cos_basis: np.ndarray,
) -> tuple[float, float] | None:
    mask = weights > 0.0
    if int(np.count_nonzero(mask)) < 8:
        return None
    y = np.asarray(values, dtype=np.float64)[mask]
    w = weights[mask]
    sin2d = np.broadcast_to(sin_basis[None, :], values.shape)[mask]
    cos2d = np.broadcast_to(cos_basis[None, :], values.shape)[mask]
    basis = np.column_stack([sin2d, cos2d, np.ones_like(sin2d)])
    sw = np.sqrt(w)
    try:
        beta, *_ = np.linalg.lstsq(basis * sw[:, None], y * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(beta[0]), float(beta[1])


def _amplitude_phase_metrics(
    *,
    target_diff: np.ndarray,
    pred_diff: np.ndarray,
    valid: np.ndarray,
    frequency_hz: float,
    dt_s: float,
) -> dict[str, Any]:
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return {"amplitude_phase_status": "invalid_frequency"}
    target_diff = np.asarray(target_diff, dtype=np.float64)
    pred_diff = np.asarray(pred_diff, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(target_diff) & np.isfinite(pred_diff)
    if target_diff.ndim != 2 or int(np.count_nonzero(valid)) < 8:
        return {
            "amplitude_phase_status": "insufficient_valid_samples",
            "amplitude_phase_n_valid": int(np.count_nonzero(valid)),
        }
    nt = target_diff.shape[1]
    t = np.arange(nt, dtype=np.float64) * float(dt_s)
    taper = _tukey_window(nt, alpha=0.5)
    sin_basis = np.sin(2.0 * np.pi * frequency_hz * t)
    cos_basis = np.cos(2.0 * np.pi * frequency_hz * t)
    weights = valid.astype(np.float64) * taper[None, :]
    target_coeff = _weighted_sincos_coefficients(target_diff, weights, sin_basis, cos_basis)
    pred_coeff = _weighted_sincos_coefficients(pred_diff, weights, sin_basis, cos_basis)
    if target_coeff is None or pred_coeff is None:
        return {"amplitude_phase_status": "invalid_projection"}
    target_amp = float(np.sqrt(target_coeff[0] ** 2 + target_coeff[1] ** 2))
    pred_amp = float(np.sqrt(pred_coeff[0] ** 2 + pred_coeff[1] ** 2))
    target_phase = float(np.arctan2(target_coeff[1], target_coeff[0]))
    pred_phase = float(np.arctan2(pred_coeff[1], pred_coeff[0]))
    phase_error = float(np.arctan2(np.sin(pred_phase - target_phase), np.cos(pred_phase - target_phase)))
    return {
        "amplitude_phase_status": "ok",
        "amplitude_phase_n_valid": int(np.count_nonzero(valid)),
        "target_probe_amplitude": target_amp,
        "pred_probe_amplitude": pred_amp,
        "probe_amplitude_error": float(pred_amp - target_amp),
        "probe_abs_amplitude_error": float(abs(pred_amp - target_amp)),
        "probe_amplitude_ratio": (
            float(pred_amp / target_amp) if np.isfinite(target_amp) and target_amp > 0.0 else float("nan")
        ),
        "target_probe_phase_rad": target_phase,
        "pred_probe_phase_rad": pred_phase,
        "probe_phase_error_rad": phase_error,
        "probe_abs_phase_error_deg": float(abs(np.rad2deg(phase_error))),
    }


def _tukey_window(size: int, *, alpha: float) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    if size == 1:
        return np.ones(1, dtype=np.float64)
    x = np.linspace(0.0, 1.0, size)
    window = np.ones(size, dtype=np.float64)
    edge = float(alpha) / 2.0
    if edge <= 0.0:
        return window
    left = x < edge
    right = x > 1.0 - edge
    window[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))
    window[right] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0)))
    return window


def _aggregate_false_energy(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "energy_status" not in frame:
        return {"n_ok": 0}
    ok = frame[frame["energy_status"].eq("ok")]
    return {
        "n_ok": int(len(ok)),
        "mean_false_frequency_rms": float(ok["false_frequency_rms"].mean()) if not ok.empty else float("nan"),
        "median_false_frequency_rms": float(ok["false_frequency_rms"].median()) if not ok.empty else float("nan"),
        "median_false_to_target_frequency_rms": (
            float(ok["false_to_target_frequency_rms"].median()) if not ok.empty else float("nan")
        ),
    }


def _aggregate_amplitude_phase(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "amplitude_phase_status" not in frame:
        return {"n_ok": 0}
    ok = frame[frame["amplitude_phase_status"].eq("ok")]
    return {
        "n_ok": int(len(ok)),
        "mean_abs_amplitude_error": (
            float(ok["probe_abs_amplitude_error"].mean()) if not ok.empty else float("nan")
        ),
        "median_amplitude_ratio": (
            float(ok["probe_amplitude_ratio"].median()) if not ok.empty else float("nan")
        ),
        "median_abs_phase_error_deg": (
            float(ok["probe_abs_phase_error_deg"].median()) if not ok.empty else float("nan")
        ),
    }


def _grouped_false_energy(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "n_ok",
                "mean_false_frequency_rms",
                "median_false_frequency_rms",
                "median_false_to_target_frequency_rms",
            ]
        )
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(_aggregate_false_energy(group))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _zero_x_false_prediction_error(metrics_frame: pd.DataFrame, prediction_dir: Path) -> pd.DataFrame:
    if metrics_frame.empty or "probe_amplitude_multiplier" not in metrics_frame:
        return pd.DataFrame()
    probe_amp = pd.to_numeric(metrics_frame["probe_amplitude_multiplier"], errors="coerce")
    zero = metrics_frame[
        metrics_frame["sample_kind"].astype(str).isin(PROBE_SAMPLE_KINDS)
        & probe_amp.eq(0.0)
    ].copy()
    if zero.empty:
        return zero
    zero["zero_x_false_error_semantics"] = (
        "absolute prediction error on 0x frequency-probe samples; "
        "not a full spectral false-energy decomposition"
    )
    catalog = _load_probe_frequency_catalog(prediction_dir)
    if catalog is not None and "probe_frequency_hz" in zero:
        zero["probe_frequency_hz"] = pd.to_numeric(zero["probe_frequency_hz"], errors="coerce")
        zero = zero.merge(
            catalog,
            left_on="probe_frequency_hz",
            right_on="frequency_hz",
            how="left",
            suffixes=("", "_catalog"),
        )
    return zero


def _load_probe_frequency_catalog(prediction_dir: Path) -> pd.DataFrame | None:
    import json

    manifest_path = prediction_dir / "prediction_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark_dir = manifest.get("benchmark_dir")
    if not benchmark_dir:
        return None
    catalog_path = Path(benchmark_dir) / "probe_frequency_catalog.csv"
    if not catalog_path.is_absolute():
        catalog_path = Path.cwd() / catalog_path
    if not catalog_path.is_file():
        return None
    catalog = pd.read_csv(catalog_path)
    if "frequency_hz" in catalog:
        catalog["frequency_hz"] = pd.to_numeric(catalog["frequency_hz"], errors="coerce")
    keep = [
        column
        for column in [
            "frequency_hz",
            "evidence_status",
            "operator_support",
            "experiment_class",
            "selection_reason",
            "calibration_status",
        ]
        if column in catalog
    ]
    return catalog[keep].drop_duplicates("frequency_hz")


def _model_dt_s(prediction_dir: Path) -> float:
    import json

    manifest_path = prediction_dir / "prediction_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        benchmark_dir = manifest.get("benchmark_dir")
        if benchmark_dir:
            summary_path = Path(benchmark_dir) / "run_summary.json"
            if not summary_path.is_absolute():
                summary_path = Path.cwd() / summary_path
            if summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                value = _safe_float(summary.get("output_dt_s"))
                if np.isfinite(value) and value > 0.0:
                    return float(value)
    return 0.002


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rmse_improvement_pct(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    model_rmse = float(model.get("mean_rmse", float("nan")))
    baseline_rmse = float(baseline.get("mean_rmse", float("nan")))
    if not np.isfinite(model_rmse) or not np.isfinite(baseline_rmse) or baseline_rmse == 0.0:
        return float("nan")
    return float((baseline_rmse - model_rmse) / baseline_rmse * 100.0)


def _grouped_probe_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        columns = [
            *keys,
            "n_ok",
            "mean_rmse",
            "mean_nrmse",
            "median_corr",
            "median_target_rms",
            "mean_abs_amplitude_error",
            "median_amplitude_ratio",
            "median_abs_phase_error_deg",
        ]
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        ok = group[group["status"].eq("ok")]
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(
            {
                "n_ok": int(len(ok)),
                "mean_rmse": float(ok["rmse"].mean()) if not ok.empty else float("nan"),
                "mean_nrmse": float(ok["nrmse"].mean()) if not ok.empty else float("nan"),
                "median_corr": float(ok["corr"].median()) if not ok.empty else float("nan"),
                "median_target_rms": float(ok["target_rms"].median()) if not ok.empty else float("nan"),
                "mean_abs_amplitude_error": (
                    float(ok["probe_abs_amplitude_error"].mean())
                    if not ok.empty and "probe_abs_amplitude_error" in ok
                    else float("nan")
                ),
                "median_amplitude_ratio": (
                    float(ok["probe_amplitude_ratio"].median())
                    if not ok.empty and "probe_amplitude_ratio" in ok
                    else float("nan")
                ),
                "median_abs_phase_error_deg": (
                    float(ok["probe_abs_phase_error_deg"].median())
                    if not ok.empty and "probe_abs_phase_error_deg" in ok
                    else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows)

__all__ = ["predict_patches", "report_predictions"]
