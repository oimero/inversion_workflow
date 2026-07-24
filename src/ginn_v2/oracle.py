"""Stage 1 projection, forward, and Oracle contract checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cup.synthetic.core.projection import project_truth_to_model_grid
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.synthetic.core.records import SampleAxis
from cup.synthetic.core.truth import SyntheticTruth

from ginn_v2.decoder import decode_numpy, decode_torch
from ginn_v2.forward import ForwardContext, forward_numpy, forward_torch
from ginn_v2.truth import (
    RawSegmentParameters,
    StructuredSample,
    StructuredTruthArtifactReader,
)


class OracleContractError(ValueError):
    """Raised when the Stage 1 scientific contract does not close."""


@dataclass(frozen=True)
class ProjectionResult:
    """Finite-support projection of one high-resolution trace."""

    model_axis: SampleAxis
    model_log_ai: np.ndarray
    support_highres: np.ndarray
    support_model: np.ndarray
    factor: int


@dataclass(frozen=True)
class OracleReport:
    """Numerical evidence collected by one Oracle run."""

    decoded_highres_log_ai: np.ndarray
    projected_log_ai: np.ndarray
    forward_seismic: np.ndarray
    projection: ProjectionResult
    metrics: Mapping[str, float]


def _allclose_oracle(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    mask: np.ndarray,
    rtol: float,
    atol: float,
    label: str,
) -> tuple[float, float]:
    actual_values = np.asarray(actual, dtype=np.float64)[mask]
    expected_values = np.asarray(expected, dtype=np.float64)[mask]
    if actual_values.size == 0:
        raise OracleContractError(f"{label} has no valid comparison samples.")
    if not np.all(np.isfinite(actual_values)) or not np.all(np.isfinite(expected_values)):
        raise OracleContractError(f"{label} contains non-finite values on its comparison mask.")
    difference = np.abs(actual_values - expected_values)
    maximum = float(np.max(difference))
    scale = np.maximum(np.abs(expected_values), np.finfo(np.float64).eps)
    relative = float(np.max(difference / scale))
    if not np.allclose(actual_values, expected_values, rtol=rtol, atol=atol):
        raise OracleContractError(
            f"{label} parity failed: max_abs={maximum:.6g}, max_relative={relative:.6g}."
        )
    return maximum, relative


def project_log_ai_to_model_grid(
    log_ai_highres: Any,
    highres_axis: SampleAxis,
    model_axis: SampleAxis,
) -> ProjectionResult:
    """Apply the same finite-support projection used by cup.synthetic."""
    if not isinstance(highres_axis, SampleAxis) or not isinstance(model_axis, SampleAxis):
        raise TypeError("projection axes must be cup.synthetic.core.records.SampleAxis.")
    values = np.asarray(log_ai_highres)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("high-resolution log AI must have a floating dtype.")
    if values.ndim < 1 or values.shape[-1] != highres_axis.coordinates.size:
        raise ValueError("high-resolution log AI final dimension must match highres_axis.")
    ratio = model_axis.sample_interval / highres_axis.sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
        raise OracleContractError("projection axes are not integer nested.")
    nested = highres_axis.coordinates[::factor]
    if nested.shape != model_axis.coordinates.shape or not np.allclose(
        nested,
        model_axis.coordinates,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise OracleContractError("projection model axis is not nested in highres_axis.")
    taps = finite_support_fir(factor)
    projected, support_model = valid_filter_decimate(
        np.asarray(values, dtype=np.float64),
        factor=factor,
        taps=taps,
    )
    support_highres = np.zeros(highres_axis.coordinates.size, dtype=bool)
    half = taps.size // 2
    support_highres[half : highres_axis.coordinates.size - half] = True
    return ProjectionResult(
        model_axis=model_axis,
        model_log_ai=np.asarray(projected, dtype=np.float64),
        support_highres=np.broadcast_to(support_highres, values.shape).copy(),
        support_model=np.broadcast_to(support_model, projected.shape).copy(),
        factor=factor,
    )


def _correlation(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray) -> float:
    left = np.asarray(actual, dtype=np.float64)[mask]
    right = np.asarray(expected, dtype=np.float64)[mask]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _torch_decoder_gradient_check(
    sample: StructuredSample,
    calibration: Any,
) -> tuple[float, bool]:
    import torch

    axis = torch.as_tensor(sample.latent.latent_axis.coordinates, dtype=torch.float64)
    background_a = torch.tensor(
        sample.zone.background_a,
        dtype=torch.float64,
        requires_grad=True,
    )
    background_b = torch.tensor(
        sample.zone.background_b,
        dtype=torch.float64,
        requires_grad=True,
    )
    segments: list[RawSegmentParameters] = []
    trainable: list[torch.Tensor] = [background_a, background_b]
    for item in sample.segments:
        parameters = []
        for name in ("c0_raw", "c1_raw", "c2_raw"):
            tensor = torch.tensor(
                float(getattr(item, name)[0]),
                dtype=torch.float64,
                requires_grad=True,
            )
            parameters.append(tensor)
            trainable.append(tensor)
        segments.append(
            RawSegmentParameters(
                zone_id=item.zone_id,
                object_id=item.object_id,
                state=item.state,
                state_id=item.state_id,
                top=item.top,
                bottom=item.bottom,
                c0=parameters[0],
                c1=parameters[1],
                c2=parameters[2],
            )
        )
    decoded = decode_torch(
        sample.zone,
        segments,
        axis,
        calibration,
        background_a=background_a,
        background_b=background_b,
    )
    loss = torch.nan_to_num(decoded.log_ai, nan=0.0).sum()
    loss.backward()
    finite = all(
        value.grad is not None and bool(torch.all(torch.isfinite(value.grad)).item())
        for value in trainable
    )
    decoded_values = decoded.log_ai.detach().cpu().numpy()
    finite_values = decoded_values[np.isfinite(decoded_values)]
    if finite_values.size == 0:
        raise OracleContractError("Torch decoder produced no finite samples.")
    return float(np.max(np.abs(finite_values))), finite


def _torch_forward_gradient_check(
    context: ForwardContext,
    model_log_ai: np.ndarray,
) -> bool:
    import torch

    values = torch.tensor(
        np.asarray(model_log_ai, dtype=np.float64),
        dtype=torch.float64,
        requires_grad=True,
    )
    output = forward_torch(context, values)
    if not bool(torch.all(torch.isfinite(output)).item()):
        return False
    output.square().sum().backward()
    return values.grad is not None and bool(torch.all(torch.isfinite(values.grad)).item())


def run_oracle(
    sample: StructuredSample,
    calibration: Any,
    forward_context: ForwardContext,
    *,
    source_truth: SyntheticTruth | None = None,
    assembled_highres_log_ai: Any | None = None,
    decoder_mask: Any | None = None,
    decoder_rtol: float = 1e-4,
    decoder_atol: float = 1e-5,
    forward_rtol: float = 1e-4,
    forward_atol: float = 1e-5,
) -> OracleReport:
    """Close decoder, projection, forward, and Torch parity for one sample."""
    if not isinstance(sample, StructuredSample):
        raise TypeError("run_oracle.sample must be StructuredSample.")
    if not isinstance(forward_context, ForwardContext):
        raise TypeError("run_oracle.forward_context must be ForwardContext.")
    if not np.array_equal(
        sample.observed.sample_axis.coordinates,
        forward_context.sample_axis.coordinates,
    ):
        raise OracleContractError("ForwardContext axis differs from StructuredSample observed axis.")
    decoded = decode_numpy(
        sample.zone,
        sample.segments,
        sample.latent.latent_axis,
        calibration,
    )
    latent_mask = (
        sample.latent.latent_valid
        if decoder_mask is None
        else np.asarray(decoder_mask, dtype=bool)
    )
    if latent_mask.shape != sample.latent.latent_valid.shape:
        raise OracleContractError("decoder comparison mask differs from the latent axis.")
    decoder_max_abs, decoder_max_relative = _allclose_oracle(
        decoded.log_ai,
        sample.latent.log_ai_highres_truth,
        mask=latent_mask,
        rtol=decoder_rtol,
        atol=decoder_atol,
        label="decoder/high-resolution truth",
    )
    projected_expected = []
    effective_expected = []
    for item in sample.segments:
        projected_expected.append(
            np.asarray(
                [item.c0_projected[0], item.c1_projected[0], item.c2_projected[0]],
                dtype=np.float64,
            )
        )
        effective_expected.append(
            np.asarray(
                [item.c0_effective[0], item.c1_effective[0], item.c2_effective[0]],
                dtype=np.float64,
            )
        )
    projected_parameter_error = 0.0
    effective_parameter_error = 0.0
    for actual, expected in zip(
        decoded.projected_parameters,
        projected_expected,
        strict=True,
    ):
        projected_parameter_error = max(
            projected_parameter_error,
            float(np.max(np.abs(np.asarray(actual) - expected))),
        )
    for actual, expected in zip(
        decoded.effective_parameters,
        effective_expected,
        strict=True,
    ):
        effective_parameter_error = max(
            effective_parameter_error,
            float(np.max(np.abs(np.asarray(actual) - expected))),
        )
    if projected_parameter_error > decoder_atol + decoder_rtol * max(
        1.0,
        max(float(np.max(np.abs(value))) for value in projected_expected),
    ):
        raise OracleContractError("decoder projected parameters disagree with structured truth.")
    if effective_parameter_error > decoder_atol + decoder_rtol * max(
        1.0,
        max(float(np.max(np.abs(value))) for value in effective_expected),
    ):
        raise OracleContractError("decoder effective parameters disagree with structured truth.")
    projection_input = (
        np.asarray(decoded.log_ai, dtype=np.float64)
        if assembled_highres_log_ai is None
        else np.asarray(assembled_highres_log_ai, dtype=np.float64)
    )
    if projection_input.shape != sample.latent.log_ai_highres_truth.shape:
        raise OracleContractError("assembled decoder trace differs from the latent axis.")
    assembled_max_abs = float("nan")
    assembled_max_relative = float("nan")
    if assembled_highres_log_ai is not None:
        assembled_mask = np.isfinite(sample.latent.log_ai_highres_truth)
        assembled_max_abs, assembled_max_relative = _allclose_oracle(
            projection_input,
            sample.latent.log_ai_highres_truth,
            mask=assembled_mask,
            rtol=decoder_rtol,
            atol=decoder_atol,
            label="assembled decoder/high-resolution truth",
        )
    projection = project_log_ai_to_model_grid(
        projection_input,
        sample.latent.latent_axis,
        sample.observed.sample_axis,
    )
    truth_projection = project_log_ai_to_model_grid(
        sample.latent.log_ai_highres_truth,
        sample.latent.latent_axis,
        sample.observed.sample_axis,
    )
    projection_mask = sample.observed.observed_valid & projection.support_model
    projection_max_abs, projection_max_relative = _allclose_oracle(
        projection.model_log_ai,
        truth_projection.model_log_ai,
        mask=projection_mask,
        rtol=decoder_rtol,
        atol=decoder_atol,
        label="projection/high-resolution truth",
    )
    source_projection_max_abs = float("nan")
    if source_truth is not None:
        if not isinstance(source_truth, SyntheticTruth):
            raise TypeError("run_oracle.source_truth must be SyntheticTruth.")
        if sample.lateral_index >= source_truth.lateral_m.size:
            raise OracleContractError("source_truth lacks the StructuredSample lateral index.")
        expected_source = project_truth_to_model_grid(
            source_truth,
            sample.observed.sample_axis,
        ).model_target_log_ai[sample.lateral_index]
        source_projection_max_abs, _ = _allclose_oracle(
            truth_projection.model_log_ai,
            expected_source,
            mask=projection_mask,
            rtol=decoder_rtol,
            atol=decoder_atol,
            label="projection/cup.synthetic parity",
        )
    forward_seismic = forward_numpy(
        forward_context,
        projection.model_log_ai,
    )
    forward_mask = sample.observed.observed_valid & projection.support_model
    expected_forward = sample.observed.model_consistent_seismic
    forward_max_abs, forward_max_relative = _allclose_oracle(
        forward_seismic,
        expected_forward,
        mask=forward_mask,
        rtol=forward_rtol,
        atol=forward_atol,
        label="forward/model-consistent finite support",
    )
    observed = expected_forward[forward_mask]
    predicted = np.asarray(forward_seismic)[forward_mask]
    correlation = _correlation(forward_seismic, expected_forward, forward_mask)
    try:
        import torch

        torch_axis = torch.as_tensor(
            sample.latent.latent_axis.coordinates,
            dtype=torch.float64,
        )
        torch_decoded = decode_torch(
            sample.zone,
            sample.segments,
            torch_axis,
            calibration,
        )
        torch_decoded_values = torch_decoded.log_ai.detach().cpu().numpy()
        torch_decoder_max_abs, torch_decoder_max_relative = _allclose_oracle(
            torch_decoded_values,
            decoded.log_ai,
            mask=latent_mask,
            rtol=decoder_rtol,
            atol=decoder_atol,
            label="NumPy/Torch decoder",
        )
        _, torch_decoder_gradients_finite = _torch_decoder_gradient_check(
            sample,
            calibration,
        )
        torch_forward_gradients_finite = _torch_forward_gradient_check(
            forward_context,
            projection.model_log_ai,
        )
        if not torch_decoder_gradients_finite:
            raise OracleContractError("Torch decoder produced non-finite gradients.")
        if not torch_forward_gradients_finite:
            raise OracleContractError("Torch forward produced non-finite gradients.")
    except ImportError as exc:
        raise OracleContractError("Stage 1 Oracle requires PyTorch for parity and gradient checks.") from exc
    metrics = {
        "decoder_max_abs_error": decoder_max_abs,
        "decoder_max_relative_error": decoder_max_relative,
        "projected_parameter_max_abs_error": projected_parameter_error,
        "effective_parameter_max_abs_error": effective_parameter_error,
        "assembled_decoder_max_abs_error": assembled_max_abs,
        "assembled_decoder_max_relative_error": assembled_max_relative,
        "projection_max_abs_error": projection_max_abs,
        "projection_max_relative_error": projection_max_relative,
        "source_projection_max_abs_error": source_projection_max_abs,
        "forward_max_abs_error": forward_max_abs,
        "forward_max_relative_error": forward_max_relative,
        "forward_waveform_correlation": correlation,
        "forward_valid_sample_count": float(observed.size),
        "forward_predicted_finite_sample_count": float(np.count_nonzero(np.isfinite(predicted))),
        "torch_decoder_max_abs_error": torch_decoder_max_abs,
        "torch_decoder_max_relative_error": torch_decoder_max_relative,
        "torch_decoder_gradients_finite": 1.0,
        "torch_forward_gradients_finite": 1.0,
    }
    return OracleReport(
        decoded_highres_log_ai=projection_input,
        projected_log_ai=np.asarray(projection.model_log_ai, dtype=np.float64),
        forward_seismic=np.asarray(forward_seismic, dtype=np.float64),
        projection=projection,
        metrics=metrics,
    )


def _forward_context_from_sample(sample: StructuredSample) -> ForwardContext:
    contract = sample.forward_context
    required = (
        "wavelet_time_s",
        "wavelet_amplitude",
        "ai_velocity_relation",
        "output_chunk_size",
    )
    missing = sorted(set(required).difference(contract))
    if missing:
        raise OracleContractError(
            f"structured artifact forward_context is missing fields: {missing}"
        )
    return ForwardContext(
        sample_axis=sample.observed.sample_axis,
        wavelet_time_s=np.asarray(contract["wavelet_time_s"], dtype=np.float64),
        wavelet_amplitude=np.asarray(contract["wavelet_amplitude"], dtype=np.float64),
        ai_velocity_relation=contract["ai_velocity_relation"],
        output_chunk_size=int(contract["output_chunk_size"]),
        lateral_m=sample.lateral_m,
        inline=sample.inline,
        xline=sample.xline,
        xline_step=sample.xline_step,
    )


def run_artifact_oracle(
    root: str,
    calibration: Any,
    *,
    expected_parent_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Stage 1 Oracle after rereading every trace from disk."""
    from pathlib import Path

    artifact_root = Path(root)

    def report_path(path: Path) -> str:
        try:
            return path.relative_to(artifact_root).as_posix() or "."
        except ValueError:
            return str(path)

    trace_paths = sorted(
        (
            path
            for path in artifact_root.glob("realizations/*/traces/*/zone_*")
            if path.is_dir()
        ),
        key=lambda path: str(path),
    )
    reader = StructuredTruthArtifactReader()
    failures: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    parent_ids: set[str] = set()
    grouped: dict[
        tuple[str, int],
        list[tuple[Any, StructuredSample]],
    ] = {}
    for trace_path in trace_paths:
        try:
            sample = reader.read(trace_path)
            parent_ids.add(sample.realization_id)
            grouped.setdefault(
                (sample.realization_id, sample.lateral_index),
                [],
            ).append(
                (trace_path, sample)
            )
        except Exception as exc:
            failures.append(
                {
                    "artifact_path": report_path(trace_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    for (realization_id, lateral_index), items in sorted(grouped.items()):
        group_path = items[0][0].parent
        try:
            reference = items[0][1]
            truth = np.asarray(
                reference.latent.log_ai_highres_truth,
                dtype=np.float64,
            )
            assembled = np.full(truth.shape, np.nan, dtype=np.float64)
            coverage = np.zeros(truth.shape, dtype=bool)
            zone_ids: set[str] = set()
            for _, sample in items:
                if sample.zone.zone_id in zone_ids:
                    raise OracleContractError(
                        f"trace contains duplicate zone {sample.zone.zone_id!r}."
                    )
                zone_ids.add(sample.zone.zone_id)
                if not np.array_equal(
                    sample.latent.latent_axis.coordinates,
                    reference.latent.latent_axis.coordinates,
                ):
                    raise OracleContractError("trace zones use different latent axes.")
                if not np.array_equal(
                    sample.observed.sample_axis.coordinates,
                    reference.observed.sample_axis.coordinates,
                ):
                    raise OracleContractError("trace zones use different observed axes.")
                if not np.array_equal(
                    sample.latent.log_ai_highres_truth,
                    truth,
                    equal_nan=True,
                ):
                    raise OracleContractError(
                        "trace zones disagree on high-resolution truth."
                    )
                zone_mask = np.asarray(sample.zone.zone_valid, dtype=bool)
                if np.any(coverage & zone_mask):
                    raise OracleContractError("trace zone masks overlap.")
                decoded = decode_numpy(
                    sample.zone,
                    sample.segments,
                    sample.latent.latent_axis,
                    calibration,
                )
                assembled[zone_mask] = np.asarray(decoded.log_ai)[zone_mask]
                coverage |= zone_mask
            truth_mask = np.isfinite(truth)
            latent_zone_id = np.asarray(reference.latent.zone_id)
            context_mask = truth_mask & (latent_zone_id < 0)
            missing_structured = truth_mask & ~coverage & ~context_mask
            if np.any(missing_structured):
                raise OracleContractError(
                    "trace zones do not cover all finite structured-zone truth."
                )
            if np.any(coverage & ~truth_mask):
                raise OracleContractError(
                    "trace zone coverage extends beyond finite high-resolution truth."
                )
            assembled[context_mask] = truth[context_mask]
            for trace_path, sample in items:
                report = run_oracle(
                    sample,
                    calibration,
                    _forward_context_from_sample(sample),
                    assembled_highres_log_ai=assembled,
                    decoder_mask=sample.zone.zone_valid,
                )
                reports.append(
                    {
                        "artifact_path": report_path(trace_path),
                        "realization_id": realization_id,
                        "lateral_index": lateral_index,
                        "zone_id": sample.zone.zone_id,
                        "metrics": dict(report.metrics),
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "artifact_path": report_path(group_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    expected = None if expected_parent_ids is None else {str(value) for value in expected_parent_ids}
    if expected is not None and parent_ids != expected:
        failures.append(
            {
                "artifact_path": ".",
                "error_type": "OracleContractError",
                "error": (
                    "artifact parent set differs from expected parent set: "
                    f"expected={sorted(expected)}, actual={sorted(parent_ids)}"
                ),
            }
        )
    if not reports and not failures:
        failures.append(
            {
                "artifact_path": ".",
                "error_type": "OracleContractError",
                "error": "structured artifact root contains no trace artifacts",
            }
        )
    aggregated: dict[str, float] = {}
    metric_names = sorted(
        {name for report in reports for name in report["metrics"]}
    )
    for name in metric_names:
        values = [float(report["metrics"][name]) for report in reports]
        if name == "forward_waveform_correlation":
            aggregated[name] = float(np.min(values))
        elif name.endswith("sample_count"):
            aggregated[name] = float(np.sum(values))
        else:
            aggregated[name] = float(np.max(values))
    return {
        "schema": "structured_truth_v1_oracle_report",
        "passed": not failures,
        "trace_count": len(reports),
        "parent_count": len(parent_ids),
        "failure_count": len(failures),
        "metrics": aggregated,
        "failures": failures,
        "traces": reports,
    }


__all__ = [
    "OracleContractError",
    "OracleReport",
    "ProjectionResult",
    "project_log_ai_to_model_grid",
    "run_artifact_oracle",
    "run_oracle",
]
