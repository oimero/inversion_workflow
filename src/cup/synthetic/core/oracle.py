"""Producer-owned decoder, projection, and forward Oracle for the canonical corpus."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cup.physics.numpy_backend import forward_depth, forward_time, velocity_from_ai
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.synthetic.readers.structured import StructuredParent, StructuredSyntheticBenchmark


ORACLE_SCHEMA = "structured_synthetic_corpus_oracle_v2"

ORACLE_NUMERICAL_FAILURE_CLASS = "numerical_parity"
ORACLE_STRUCTURAL_FAILURE_CLASS = "structural"


class OracleParityError(ValueError):
    """A numerical replay mismatch on otherwise valid published data."""

    failure_class = ORACLE_NUMERICAL_FAILURE_CLASS


class OracleSupportError(ValueError):
    """The published artifact has no valid support for an Oracle comparison."""

    failure_class = ORACLE_STRUCTURAL_FAILURE_CLASS

# The published structured artifact stores the model-grid arrays as float32.
# A depth forward uses a finite-support wavelet, so a float32 round-trip can
# move one travel-time sample from just inside a non-zero wavelet endpoint to
# just outside it.  This is a representation boundary, not a decoder or
# projection failure.  Keep the margin explicit and report the affected
# samples instead of weakening parity for the rest of the trace.
FORWARD_ROUNDTRIP_BOUNDARY_MARGIN_S = 1.0e-7


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


def _depth_forward_boundary_sensitive_mask(
    parent: StructuredParent,
    *,
    margin_s: float = FORWARD_ROUNDTRIP_BOUNDARY_MARGIN_S,
) -> np.ndarray:
    """Return output samples that can change at a finite-support endpoint.

    The artifact round-trip stores ``model_log_ai`` as float32 while the
    producer forward is evaluated before that quantization.  For depth
    modeling, travel time depends on the model and the finite-support
    interpolation is discontinuous when an event crosses the first or last
    wavelet sample.  This mask identifies only that narrow representation
    seam.  It is intentionally not used for arbitrary forward mismatches.
    """
    output_shape = np.asarray(parent.observed_valid, dtype=bool).shape
    if parent.sample_domain != "depth":
        return np.zeros(output_shape, dtype=bool)
    if not np.isfinite(margin_s) or margin_s < 0.0:
        raise ValueError("forward round-trip boundary margin must be finite and non-negative")

    context = parent.forward_context
    wavelet_time = np.asarray(context["wavelet_time_s"], dtype=np.float64)
    wavelet_amplitude = np.asarray(
        context["wavelet_amplitude"], dtype=np.float64
    )
    if wavelet_time.ndim != 1 or wavelet_amplitude.shape != wavelet_time.shape:
        raise ValueError("forward context contains invalid wavelet arrays")
    # A zero-valued endpoint is continuous with the finite-support zero
    # extension and cannot create this particular round-trip discontinuity.
    if not (wavelet_amplitude[0] != 0.0 or wavelet_amplitude[-1] != 0.0):
        return np.zeros(output_shape, dtype=bool)

    relation = dict(context["ai_velocity_relation"])
    velocity = velocity_from_ai(
        np.exp(np.asarray(parent.model_log_ai, dtype=np.float64)),
        a=float(relation["a"]),
        b=float(relation["b"]),
    )
    depth = np.asarray(parent.model_axis.coordinates, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape != output_shape:
        raise ValueError(
            "depth forward boundary mask requires model arrays with shape "
            f"{output_shape}, got {velocity.shape}"
        )
    dz = np.diff(depth)
    interval_twt = 2.0 * dz[None, :] * 0.5 * (
        np.reciprocal(velocity[:, :-1]) + np.reciprocal(velocity[:, 1:])
    )
    sample_twt = np.empty_like(velocity, dtype=np.float64)
    sample_twt[:, 0] = 0.0
    sample_twt[:, 1:] = np.cumsum(interval_twt, axis=-1)
    interface_twt = 0.5 * (sample_twt[:, :-1] + sample_twt[:, 1:])

    sensitive = np.zeros(output_shape, dtype=bool)
    wavelet_min = float(wavelet_time[0])
    wavelet_max = float(wavelet_time[-1])
    chunk_size = max(1, int(context.get("output_chunk_size", 64)))
    for start in range(0, output_shape[-1], chunk_size):
        stop = min(start + chunk_size, output_shape[-1])
        tau = sample_twt[:, start:stop, None] - interface_twt[:, None, :]
        sensitive[:, start:stop] = np.any(
            (np.abs(tau - wavelet_min) <= margin_s)
            | (np.abs(tau - wavelet_max) <= margin_s),
            axis=-1,
        )
    return sensitive


def _error_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    left = np.asarray(actual, dtype=np.float64)[mask]
    right = np.asarray(expected, dtype=np.float64)[mask]
    if left.size == 0 or np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise OracleSupportError("comparison has no finite support")
    difference = np.abs(left - right)
    maximum = float(np.max(difference))
    relative = float(
        np.max(difference / np.maximum(np.abs(right), np.finfo(np.float64).eps))
    )
    return maximum, relative


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
    maximum, relative = _error_metrics(actual, expected, mask)
    if not np.allclose(left, right, rtol=rtol, atol=atol):
        raise OracleParityError(
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
        "forward_boundary_sensitive_sample_count": 0.0,
        "forward_boundary_sensitive_parent_count": 0.0,
        "forward_boundary_sensitive_max_abs_error": 0.0,
        "clipped_sample_count": 0.0,
    }
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
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
            boundary_mask = _depth_forward_boundary_sensitive_mask(parent)
            observed_mask = np.asarray(parent.observed_valid, dtype=bool)
            safe_forward_mask = observed_mask & ~boundary_mask
            forward_error, _ = _max_error(
                seismic,
                parent.model_consistent_seismic,
                safe_forward_mask,
                label=f"{parent.sample_domain} forward",
                rtol=rtol,
                atol=atol,
            )
            boundary_forward_mask = observed_mask & boundary_mask
            if np.any(boundary_forward_mask):
                metrics["forward_boundary_sensitive_sample_count"] += float(
                    np.count_nonzero(boundary_forward_mask)
                )
                metrics["forward_boundary_sensitive_parent_count"] += 1.0
                boundary_error, boundary_relative = _error_metrics(
                    seismic,
                    parent.model_consistent_seismic,
                    boundary_forward_mask,
                )
                metrics["forward_boundary_sensitive_max_abs_error"] = max(
                    metrics["forward_boundary_sensitive_max_abs_error"],
                    boundary_error,
                )
                boundary_actual = np.asarray(seismic, dtype=np.float64)[
                    boundary_forward_mask
                ]
                boundary_expected = np.asarray(
                    parent.model_consistent_seismic, dtype=np.float64
                )[boundary_forward_mask]
                if not np.allclose(
                    boundary_actual,
                    boundary_expected,
                    rtol=rtol,
                    atol=atol,
                ):
                    warnings.append(
                        {
                            "warning_type": "forward_roundtrip_boundary_sensitivity",
                            "parent_id": parent_id,
                            "sample_count": int(np.count_nonzero(boundary_forward_mask)),
                            "max_abs_error": boundary_error,
                            "max_relative_error": boundary_relative,
                            "margin_s": FORWARD_ROUNDTRIP_BOUNDARY_MARGIN_S,
                        }
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
                    "failure_class": getattr(
                        exc,
                        "failure_class",
                        ORACLE_STRUCTURAL_FAILURE_CLASS,
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {
        "schema": ORACLE_SCHEMA,
        "passed": not failures and bool(selected),
        "parent_count": len(selected),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "metrics": metrics,
        "failures": failures,
        "warnings": warnings,
        "forward_roundtrip_boundary_margin_s": FORWARD_ROUNDTRIP_BOUNDARY_MARGIN_S,
    }


__all__ = [
    "FORWARD_ROUNDTRIP_BOUNDARY_MARGIN_S",
    "ORACLE_NUMERICAL_FAILURE_CLASS",
    "ORACLE_STRUCTURAL_FAILURE_CLASS",
    "ORACLE_SCHEMA",
    "OracleParityError",
    "OracleSupportError",
    "run_canonical_oracle",
]
