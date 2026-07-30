"""Producer-owned decoder, projection, and forward Oracle for canonical V2."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cup.physics.numpy_backend import forward_depth, forward_time, velocity_from_ai
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.synthetic.readers.structured import StructuredParent, StructuredSyntheticBenchmark


ORACLE_SCHEMA = "structured_synthetic_corpus_oracle_v2"


def _decode_effective(parent: StructuredParent) -> np.ndarray:
    axis = parent.highres_axis.coordinates
    output = np.full_like(parent.log_ai_highres, np.nan, dtype=np.float64)
    for trace in range(parent.lateral_m.size):
        zone_rows = [
            row for row in parent.zones if int(row["lateral_index"]) == trace
        ]
        segment_rows = [
            row
            for row in parent.segments
            if int(row["lateral_index"]) == trace
            and int(row["duration_samples"]) > 0
        ]
        for zone in zone_rows:
            zone_value = int(zone["zone_grid_value"])
            zone_mask = parent.zone_id_highres[trace] == zone_value
            top = float(zone["top"])
            bottom = float(zone["bottom"])
            zeta = (axis - top) / (bottom - top)
            background = float(zone["background_a"]) + float(
                zone["background_b"]
            ) * (2.0 * zeta - 1.0)
            output[trace, zone_mask] = background[zone_mask]
        for segment in segment_rows:
            object_id = int(segment["object_id"])
            mask = parent.object_id_highres[trace] == object_id
            if not np.any(mask):
                continue
            xi = np.asarray(parent.object_xi_highres[trace, mask], dtype=np.float64)
            profile = (
                float(segment["c0_effective"])
                + float(segment["c1_effective"]) * (2.0 * xi - 1.0)
                + float(segment["c2_effective"]) * np.sin(np.pi * xi)
            )
            output[trace, mask] += profile
    return output


def _project(parent: StructuredParent, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ratio = parent.model_axis.sample_interval / parent.highres_axis.sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise ValueError("Oracle axes are not integer nested.")
    return valid_filter_decimate(
        values,
        factor=factor,
        taps=finite_support_fir(factor),
    )


def _forward(parent: StructuredParent, model_log_ai: np.ndarray) -> np.ndarray:
    context = parent.forward_context
    wavelet_time = np.asarray(context["wavelet_time_s"], dtype=np.float64)
    wavelet = np.asarray(context["wavelet_amplitude"], dtype=np.float64)
    if parent.sample_domain == "time":
        return forward_time(model_log_ai, wavelet_time, wavelet)
    relation = dict(context["ai_velocity_relation"])
    velocity = velocity_from_ai(
        np.exp(model_log_ai),
        a=float(relation["a"]),
        b=float(relation["b"]),
    )
    return forward_depth(
        model_log_ai,
        velocity,
        parent.model_axis.coordinates,
        wavelet_time,
        wavelet,
        output_chunk_size=int(context["output_chunk_size"]),
    )


def _max_error(
    actual: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    left = np.asarray(actual, dtype=np.float64)[mask]
    right = np.asarray(expected, dtype=np.float64)[mask]
    if left.size == 0 or np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError(f"{label} has no finite comparison support.")
    difference = np.abs(left - right)
    maximum = float(np.max(difference))
    relative = float(
        np.max(difference / np.maximum(np.abs(right), np.finfo(np.float64).eps))
    )
    if not np.allclose(left, right, rtol=rtol, atol=atol):
        raise ValueError(
            f"{label} parity failed: max_abs={maximum:.6g}, "
            f"max_relative={relative:.6g}"
        )
    return maximum, relative


def run_canonical_oracle(
    root: str | Path,
    _calibration: Any = None,
    *,
    expected_parent_ids: Sequence[str] | None = None,
    selected_parent_ids: Sequence[str] | None = None,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-5,
) -> dict[str, Any]:
    benchmark = StructuredSyntheticBenchmark(root)
    available = {item.realization_id for item in benchmark.list_parents()}
    if expected_parent_ids is not None and available != {
        str(value) for value in expected_parent_ids
    }:
        raise ValueError("Oracle parent set differs from the published index.")
    selected = (
        sorted(available)
        if selected_parent_ids is None
        else sorted(str(value) for value in selected_parent_ids)
    )
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"Oracle selected unknown parents: {unknown[:5]}")
    metrics: dict[str, float] = {
        "decoder_max_abs_error": 0.0,
        "projection_max_abs_error": 0.0,
        "forward_max_abs_error": 0.0,
        "clipped_sample_count": 0.0,
    }
    failures: list[dict[str, str]] = []
    for parent_id in selected:
        try:
            parent = benchmark.read_parent(parent_id)
            decoded = _decode_effective(parent)
            decoder_mask = (
                parent.truth_valid_highres
                & ~parent.clipping_mask_highres
                & np.isfinite(parent.log_ai_highres)
                & np.isfinite(decoded)
            )
            decoder_error, _ = _max_error(
                decoded,
                parent.log_ai_highres,
                decoder_mask,
                label="effective decoder",
                rtol=rtol,
                atol=atol,
            )
            projected, support_1d = _project(parent, parent.log_ai_highres)
            projection_mask = np.broadcast_to(support_1d, projected.shape)
            projection_error, _ = _max_error(
                projected,
                parent.model_log_ai,
                projection_mask & np.isfinite(parent.model_log_ai),
                label="finite-support projection",
                rtol=rtol,
                atol=atol,
            )
            seismic = _forward(parent, parent.model_log_ai)
            forward_error, _ = _max_error(
                seismic,
                parent.model_consistent_seismic,
                parent.observed_valid,
                label=f"{parent.sample_domain} forward",
                rtol=rtol,
                atol=atol,
            )
            metrics["decoder_max_abs_error"] = max(
                metrics["decoder_max_abs_error"], decoder_error
            )
            metrics["projection_max_abs_error"] = max(
                metrics["projection_max_abs_error"], projection_error
            )
            metrics["forward_max_abs_error"] = max(
                metrics["forward_max_abs_error"], forward_error
            )
            metrics["clipped_sample_count"] += float(
                np.count_nonzero(parent.clipping_mask_highres)
            )
        except Exception as exc:
            failures.append(
                {
                    "parent_id": parent_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "schema": ORACLE_SCHEMA,
        "passed": not failures and bool(selected),
        "parent_count": len(selected),
        "failure_count": len(failures),
        "metrics": metrics,
        "failures": failures,
    }


__all__ = ["ORACLE_SCHEMA", "run_canonical_oracle"]
