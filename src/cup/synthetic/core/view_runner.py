"""Materialize configured seismic views for Synthoseis v5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cup.synthetic.core.random import ar1_irregular, operator_rng
from cup.synthetic.core.rejections import ForwardRejected
from cup.synthetic.core.views import SeismicViewSpec
from cup.utils.statistics import centered_rms


@dataclass(frozen=True)
class SeismicViewResult:
    view_id: str
    seismic_observed: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    metadata: Mapping[str, Any]
    qc: Mapping[str, Any]


def _normalize_noise(values: np.ndarray, mask: np.ndarray, rms: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(output)
    output[~valid] = 0.0
    output[valid] -= float(np.mean(output[valid]))
    current = float(np.sqrt(np.mean(output[valid] * output[valid])))
    if not np.isfinite(current) or current <= 0.0:
        return np.zeros_like(output)
    return output * (float(rms) / current)


def _static(values: np.ndarray, axis: np.ndarray, shift: float, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    coordinate = np.asarray(axis, dtype=np.float64).reshape(-1)
    source_support = np.asarray(support, dtype=bool)
    output = np.full_like(values, np.nan)
    output_support = np.zeros_like(source_support, dtype=bool)
    for trace in range(values.shape[0]):
        valid = source_support[trace] & np.isfinite(values[trace])
        if np.count_nonzero(valid) < 2:
            continue
        source_axis = coordinate[valid]
        sample_at = coordinate - float(shift)
        inside = (sample_at >= source_axis[0]) & (sample_at <= source_axis[-1])
        output[trace, inside] = np.interp(sample_at[inside], source_axis, values[trace, valid])
        output_support[trace, inside] = True
    return output, output_support


def _trace_gain(
    *,
    lateral: np.ndarray,
    shape: tuple[int, int],
    log_sigma: float,
    correlation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    lateral = np.asarray(lateral, dtype=np.float64).reshape(-1)
    if lateral.size < 2:
        return np.ones(shape, dtype=np.float64), {}
    requested = float(correlation_fraction) * max(float(lateral[-1] - lateral[0]), np.finfo(float).eps)
    effective = max(requested, 4.0 * float(np.median(np.diff(lateral))))
    field, field_qc = ar1_irregular(lateral, correlation_length_m=effective, rng=rng)
    return np.exp(float(log_sigma) * field)[:, None] * np.ones(shape), {
        "requested_correlation_length_m": requested,
        "effective_correlation_length_m": effective,
        "empirical_correlation_length_m": float(field_qc.get("empirical_correlation_length_m", np.nan)),
    }


def _regular_axis_field(size: int, correlation_fraction: float, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        return np.zeros(size, dtype=np.float64)
    correlation = max(float(correlation_fraction) * size, 1.0)
    rho = float(np.exp(-1.0 / correlation))
    raw = np.empty(size, dtype=np.float64)
    raw[0] = rng.normal()
    scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for index in range(1, size):
        raw[index] = rho * raw[index - 1] + scale * rng.normal()
    raw = np.clip(raw, -3.0, 3.0)
    raw -= float(np.mean(raw))
    rms = float(np.sqrt(np.mean(raw * raw)))
    return raw / rms if rms > 0.0 else np.zeros_like(raw)


def _coordinate_field(
    coordinates: np.ndarray,
    *,
    correlation_length: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample one normalized OU field and evaluate it at arbitrary coordinates."""

    values = np.asarray(coordinates, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.zeros_like(values, dtype=np.float64)
    if not np.any(finite):
        raise ValueError("coordinate field requires finite coordinates")
    lower = float(np.min(values[finite]))
    upper = float(np.max(values[finite]))
    span = upper - lower
    length = float(correlation_length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("coordinate field correlation length must be positive")
    if span <= np.finfo(np.float64).eps:
        return output, {
            "coordinate_min": lower,
            "coordinate_max": upper,
            "coordinate_span": span,
            "node_count": 1,
            "node_spacing": float("nan"),
            "requested_correlation_length": length,
            "requested_to_node_spacing_ratio": float("nan"),
            "empirical_correlation_length": float("nan"),
        }
    node_count = int(np.clip(np.ceil(8.0 * span / length) + 1, 65, 4097))
    nodes = np.linspace(lower, upper, node_count, dtype=np.float64)
    field, qc = ar1_irregular(nodes, correlation_length_m=length, rng=rng)
    output[finite] = np.interp(values[finite], nodes, field)
    return output, {
        "coordinate_min": lower,
        "coordinate_max": upper,
        "coordinate_span": span,
        "node_count": node_count,
        "node_spacing": float(nodes[1] - nodes[0]),
        "requested_correlation_length": length,
        "requested_to_node_spacing_ratio": float(length / (nodes[1] - nodes[0])),
        "empirical_correlation_length": float(
            qc.get("empirical_correlation_length_m", np.nan)
        ),
    }


def _calibrated_rgt_gain(
    *,
    operator_id: str,
    operator: Mapping[str, Any],
    spec: SeismicViewSpec,
    global_seed: int,
    generator_family: str,
    realization_id: str,
    lateral: np.ndarray,
    rgt_model: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample one real-minus-pilot calibrated RGT gain prior."""

    rgt = np.asarray(rgt_model, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    lateral = np.asarray(lateral, dtype=np.float64).reshape(-1)
    if rgt.ndim != 2 or rgt.shape != mask.shape or rgt.shape[0] != lateral.size:
        raise ValueError("calibrated_rgt_gain requires aligned lateral/RGT/mask arrays")
    if np.any(mask & ~np.isfinite(rgt)):
        raise ValueError("calibrated_rgt_gain requires finite RGT on the public mask")

    # The mean-only, residual-only, and full views are paired interventions.
    # Their residual draws therefore key off the calibrated prior, not the view
    # name or its two scale controls.
    operator_hash = str(operator["prior_sha256"])

    def rng(name: str) -> np.random.Generator:
        return operator_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            operator_id="calibrated_rgt_gain_prior",
            operator_spec_sha256=operator_hash,
            coefficient_name=name,
        )

    mean_model = dict(operator["mean_model"])
    residual_model = dict(operator["residual_model"])
    knots = np.asarray(mean_model["rgt_knots"], dtype=np.float64)
    template = np.asarray(mean_model["mean_log_gain_rgt"], dtype=np.float64)
    public_values = rgt[mask]
    if public_values.size == 0 or float(np.min(public_values)) < knots[0] - 1e-9 or float(np.max(public_values)) > knots[-1] + 1e-9:
        reason = "calibrated_rgt_gain_public_rgt_outside_prior_support"
        diagnostics = {
            "public_rgt_min": float(np.min(public_values)) if public_values.size else float("nan"),
            "public_rgt_max": float(np.max(public_values)) if public_values.size else float("nan"),
            "prior_rgt_min": float(knots[0]),
            "prior_rgt_max": float(knots[-1]),
        }
        raise ForwardRejected(
            [reason],
            diagnostics=diagnostics,
            details=[{"reason": reason, **diagnostics}],
        )
    public_rgt = np.where(mask, rgt, np.nan)
    active_lateral = np.any(mask, axis=1)
    residual = np.zeros_like(rgt, dtype=np.float64)
    rgt_qc: dict[str, Any] = {}
    rgt_component = dict(residual_model["rgt_component"])
    if bool(rgt_component["enabled"]):
        field, rgt_qc = _coordinate_field(
            public_rgt,
            correlation_length=float(rgt_component["correlation_length_rgt"]),
            rng=rng("calibrated_rgt_gain:rgt"),
        )
        residual += float(rgt_component["log_sigma"]) * field
    lateral_qc: dict[str, Any] = {}
    lateral_component = dict(residual_model["lateral_component"])
    if bool(lateral_component["enabled"]) and np.count_nonzero(active_lateral) >= 2:
        field, lateral_qc = ar1_irregular(
            lateral[active_lateral],
            correlation_length_m=float(lateral_component["correlation_length_m"]),
            rng=rng("calibrated_rgt_gain:lateral"),
        )
        lateral_field = np.zeros(lateral.size, dtype=np.float64)
        lateral_field[active_lateral] = field
        residual += float(lateral_component["log_sigma"]) * lateral_field[:, None]
    interaction_component = dict(residual_model["interaction_component"])
    interaction_empirical_lengths: list[float] = []
    if bool(interaction_component["enabled"]) and np.count_nonzero(active_lateral) >= 2:
        modes = np.asarray(interaction_component["rgt_modes"], dtype=np.float64)
        eigenvalues = np.asarray(interaction_component["eigenvalues"], dtype=np.float64)
        lengths = np.asarray(
            interaction_component["lateral_correlation_lengths_m"], dtype=np.float64
        )
        for index, (mode, eigenvalue, length) in enumerate(zip(modes, eigenvalues, lengths)):
            lateral_factor = np.zeros(lateral.size, dtype=np.float64)
            active_factor, factor_qc = ar1_irregular(
                lateral[active_lateral],
                correlation_length_m=float(length),
                rng=rng(f"calibrated_rgt_gain:interaction:{index}"),
            )
            lateral_factor[active_lateral] = active_factor
            mode_field = np.zeros_like(rgt)
            mode_field[mask] = np.interp(public_values, knots, mode)
            residual += np.sqrt(float(eigenvalue)) * lateral_factor[:, None] * mode_field
            interaction_empirical_lengths.append(float(
                factor_qc.get("empirical_correlation_length_m", np.nan)
            ))
    mean_log_gain = np.zeros_like(rgt)
    mean_log_gain[mask] = np.interp(public_values, knots, template)
    log_gain = (
        float(operator["mean_scale"]) * mean_log_gain
        + float(operator["residual_scale"]) * residual
    )
    center = float(np.median(log_gain[mask]))
    log_gain -= center
    maximum = float(operator["max_abs_log_gain"])
    clipped = mask & (np.abs(log_gain) > maximum)
    log_gain = np.clip(log_gain, -maximum, maximum)
    gain = np.ones_like(rgt, dtype=np.float64)
    gain[mask] = np.exp(log_gain[mask])
    return gain, {
        "prior_artifact_sha256": str(operator["prior_artifact_sha256"]),
        "prior_sha256": str(operator["prior_sha256"]),
        "mean_scale": float(operator["mean_scale"]),
        "residual_scale": float(operator["residual_scale"]),
        "rgt_empirical_correlation_length": float(rgt_qc.get("empirical_correlation_length", np.nan)),
        "lateral_empirical_correlation_length_m": float(lateral_qc.get("empirical_correlation_length_m", np.nan)),
        "interaction_rank": int(interaction_component["rank"]),
        "interaction_lateral_empirical_correlation_lengths_m": interaction_empirical_lengths,
        "log_gain_center_removed": center,
        "log_gain_rms": float(np.sqrt(np.mean(log_gain[mask] ** 2))),
        "log_gain_clipped_count": int(np.count_nonzero(clipped[mask])),
        "log_gain_clipped_fraction": float(np.mean(clipped[mask])),
    }


def _gain_operator(
    *,
    operator_id: str,
    operator: Mapping[str, Any],
    spec: SeismicViewSpec,
    global_seed: int,
    generator_family: str,
    realization_id: str,
    lateral: np.ndarray,
    shape: tuple[int, int],
    rgt_model: np.ndarray | None,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(operator["kind"])
    sigma = float(operator.get("log_sigma", 0.0))
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError(f"invalid log_sigma for seismic operator {operator_id!r}")
    if kind == "global_gain":
        rng = operator_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            operator_id=operator_id,
            operator_spec_sha256=spec.operator_spec_sha256(operator_id),
            coefficient_name="global_gain",
        )
        value = float(np.exp(sigma * rng.normal()))
        return np.full(shape, value, dtype=np.float64), {"global_gain": value}
    if kind == "tracewise_gain":
        gain, qc = _trace_gain(
            lateral=lateral,
            shape=shape,
            log_sigma=sigma,
            correlation_fraction=float(operator["lateral_correlation_fraction"]),
            rng=operator_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
                operator_id=operator_id,
                operator_spec_sha256=spec.operator_spec_sha256(operator_id),
                coefficient_name="tracewise_gain",
            ),
        )
        return gain, qc
    if kind == "axis_lateral_gain":
        lateral_gain, qc = _trace_gain(
            lateral=lateral,
            shape=shape,
            log_sigma=1.0,
            correlation_fraction=float(operator["lateral_correlation_fraction"]),
            rng=operator_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
                operator_id=operator_id,
                operator_spec_sha256=spec.operator_spec_sha256(operator_id),
                coefficient_name="axis_lateral_gain:lateral",
            ),
        )
        axis_field = _regular_axis_field(
            shape[1],
            float(operator["axis_correlation_fraction"]),
            operator_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
                operator_id=operator_id,
                operator_spec_sha256=spec.operator_spec_sha256(operator_id),
                coefficient_name="axis_lateral_gain:axis",
            ),
        )
        raw = np.log(lateral_gain[:, :1]) + axis_field[None, :]
        raw -= float(np.mean(raw))
        rms = float(np.sqrt(np.mean(raw * raw)))
        if rms > 0.0:
            raw /= rms
        return np.exp(sigma * raw), qc
    if kind == "calibrated_rgt_gain":
        if rgt_model is None:
            raise ValueError("calibrated_rgt_gain requires rgt_model")
        return _calibrated_rgt_gain(
            operator_id=operator_id,
            operator=operator,
            spec=spec,
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            lateral=lateral,
            rgt_model=rgt_model,
            valid_mask=valid_mask,
        )
    raise ValueError(f"operator {operator_id!r} is not a gain operator")


def generate_seismic_views(
    *,
    base_seismic: np.ndarray,
    valid_mask: np.ndarray,
    operator_source_support: np.ndarray,
    lateral_m: np.ndarray,
    sample_axis: np.ndarray,
    rgt_model: np.ndarray | None = None,
    view_specs: Sequence[SeismicViewSpec],
    global_seed: int,
    generator_family: str,
    realization_id: str,
    axis_unit: str,
    perturbed_forward: Callable[[float, float], tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[SeismicViewResult]:
    """Materialize only the configured views using the v5 operator grammar."""

    base = np.asarray(base_seismic, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    source_support = np.asarray(operator_source_support, dtype=bool)
    axis = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
    lateral = np.asarray(lateral_m, dtype=np.float64).reshape(-1)
    requires_rgt = any(
        "calibrated_rgt_gain" in spec.operator_kinds
        for spec in view_specs
    )
    rgt = None if rgt_model is None else np.asarray(rgt_model, dtype=np.float64)
    if (
        base.ndim != 2
        or mask.shape != base.shape
        or source_support.shape != base.shape
        or base.shape != (lateral.size, axis.size)
    ):
        raise ValueError(
            "base seismic, public mask, operator source support, lateral axis "
            "and sample axis are misaligned"
        )
    if np.any(mask & ~np.isfinite(base)):
        raise ValueError("base seismic has non-finite valid samples")
    if requires_rgt:
        if rgt is None or rgt.shape != base.shape:
            raise ValueError("configured RGT gain requires RGT aligned with base seismic")
        if np.any(mask & ~np.isfinite(rgt)):
            raise ValueError("configured RGT gain requires finite RGT on public samples")
    if np.any(mask & ~source_support):
        raise ValueError("public valid mask is outside operator source support")
    results: list[SeismicViewResult] = []
    for spec in view_specs:
        state = base.copy()
        state_mask = mask.copy()
        state_source_support = source_support.copy()
        cumulative_gain = np.ones_like(base)
        cumulative_noise = np.zeros_like(base)
        qc: dict[str, Any] = {}
        phase = 0.0
        shift = 0.0
        for operator_id in spec.forward_operator_ids:
            operator = spec.operators[operator_id]
            kind = str(operator["kind"])
            if kind == "wavelet_phase_rotation":
                phase += float(operator["degrees"])
            elif kind == "wavelet_time_shift":
                shift += float(operator["seconds"])
            qc[f"{operator_id}_parameter"] = dict(operator)
        if spec.forward_operator_ids:
            if perturbed_forward is None:
                raise ValueError(f"view {spec.view_id!r} requires a forward Adapter")
            state, forward_mask = perturbed_forward(phase, shift)
            state = np.asarray(state, dtype=np.float64)
            forward_mask = np.asarray(forward_mask, dtype=bool)
            if state.shape != base.shape or forward_mask.shape != base.shape:
                raise ValueError(
                    f"view {spec.view_id!r} forward output/support differs from base shape"
                )
            if np.any(mask & ~forward_mask):
                reason = "seismic_view_forward_support_incomplete"
                raise ForwardRejected(
                    [reason],
                    diagnostics={"view_id": spec.view_id},
                    details=[{"reason": reason, "view_id": spec.view_id}],
                )
            state_mask = mask.copy()
            state_source_support = forward_mask.copy()
        for operator_id in spec.sampled_operator_ids:
            operator = spec.operators[operator_id]
            kind = str(operator["kind"])
            if kind == "axis_static":
                shift_value = operator.get("shift")
                if not isinstance(shift_value, Mapping) or str(shift_value.get("unit")) != axis_unit:
                    raise ValueError(f"axis_static operator {operator_id!r} has wrong axis unit")
                shift_value_axis = float(shift_value["value"])
                state, support = _static(
                    state, axis, shift_value_axis, state_source_support
                )
                cumulative_gain, gain_support = _static(
                    cumulative_gain, axis, shift_value_axis, state_source_support
                )
                cumulative_noise, noise_support = _static(
                    cumulative_noise, axis, shift_value_axis, state_source_support
                )
                if not np.array_equal(mask & support, mask):
                    reason = "seismic_view_axis_static_support_incomplete"
                    raise ForwardRejected(
                        [reason],
                        diagnostics={"view_id": spec.view_id},
                        details=[{"reason": reason, "view_id": spec.view_id}],
                    )
                if not np.array_equal(support, gain_support) or not np.array_equal(
                    support, noise_support
                ):
                    raise ValueError(
                        f"view {spec.view_id!r} axis_static auxiliary support differs"
                    )
                state_source_support = support
            elif kind in {"global_gain", "tracewise_gain", "axis_lateral_gain", "calibrated_rgt_gain"}:
                gain, gain_qc = _gain_operator(
                    operator_id=operator_id,
                    operator=operator,
                    spec=spec,
                    global_seed=global_seed,
                    generator_family=generator_family,
                    realization_id=realization_id,
                    lateral=lateral,
                    shape=base.shape,
                    rgt_model=rgt,
                    valid_mask=state_mask,
                )
                state = gain * state
                cumulative_gain = gain * cumulative_gain
                cumulative_noise = gain * cumulative_noise
                qc.update({f"{operator_id}_{key}": value for key, value in gain_qc.items()})
            elif kind in {"additive_white_noise", "additive_colored_noise"}:
                signal_rms = centered_rms(state, state_mask)
                fraction = float(operator["rms_fraction"])
                if fraction < 0.0:
                    raise ValueError(f"view {spec.view_id!r} has invalid noise fraction")
                if not np.isfinite(signal_rms) or signal_rms <= 0.0:
                    reason = "seismic_view_signal_has_no_noise_reference_energy"
                    raise ForwardRejected(
                        [reason],
                        diagnostics={"view_id": spec.view_id, "signal_rms": signal_rms},
                        details=[{
                            "reason": reason,
                            "view_id": spec.view_id,
                            "signal_rms": signal_rms,
                        }],
                    )
                rng = operator_rng(
                    global_seed=global_seed,
                    generator_family=generator_family,
                    realization_id=realization_id,
                    operator_id=operator_id,
                    operator_spec_sha256=spec.operator_spec_sha256(operator_id),
                    coefficient_name="noise",
                )
                if kind == "additive_white_noise":
                    raw = rng.normal(size=base.shape)
                else:
                    correlation = operator.get("axis_correlation")
                    if not isinstance(correlation, Mapping) or str(correlation.get("unit")) != axis_unit:
                        raise ValueError(f"colored noise operator {operator_id!r} has wrong axis unit")
                    length = float(correlation["value"])
                    if length <= 0.0:
                        raise ValueError(f"colored noise operator {operator_id!r} requires positive correlation")
                    raw = np.empty_like(base)
                    raw[:, 0] = rng.normal(size=base.shape[0])
                    for index in range(1, base.shape[1]):
                        rho = float(np.exp(-(axis[index] - axis[index - 1]) / length))
                        raw[:, index] = rho * raw[:, index - 1] + np.sqrt(max(0.0, 1.0 - rho * rho)) * rng.normal(size=base.shape[0])
                noise = _normalize_noise(raw, state_mask, fraction * signal_rms)
                state = state + noise
                cumulative_noise = cumulative_noise + noise
                qc[f"{operator_id}_requested_noise_rms"] = fraction * signal_rms
            else:
                raise ValueError(f"unsupported sampled seismic operator kind: {kind!r}")
        if np.any(mask & ~np.isfinite(state)):
            raise ValueError(f"view {spec.view_id!r} has non-finite valid samples")
        results.append(
            SeismicViewResult(
                view_id=spec.view_id,
                seismic_observed=state,
                positive_gain=cumulative_gain,
                additive_noise=cumulative_noise,
                metadata={
                    **spec.metadata(),
                    "operator_trace_json": [
                        {
                            "operator_id": operator_id,
                            "kind": str(spec.operators[operator_id]["kind"]),
                            "parameters": dict(spec.operators[operator_id]),
                        }
                        for operator_id in spec.operator_ids
                    ],
                },
                qc={
                    **qc,
                    "seismic_view_status": "ok",
                    "seismic_view_id": spec.view_id,
                    "seismic_observed_rms": centered_rms(state, mask),
                    "seismic_base_rms": centered_rms(base, mask),
                    "positive_gain_min": float(np.min(cumulative_gain[mask])),
                    "positive_gain_max": float(np.max(cumulative_gain[mask])),
                    "positive_gain_mean": float(np.mean(cumulative_gain[mask])),
                    "positive_gain_median": float(np.median(cumulative_gain[mask])),
                    "positive_gain_p05": float(np.quantile(cumulative_gain[mask], 0.05)),
                    "positive_gain_p95": float(np.quantile(cumulative_gain[mask], 0.95)),
                    "additive_noise_rms": float(np.sqrt(np.mean(cumulative_noise[mask] ** 2))),
                },
            )
        )
    return results


__all__ = ["SeismicViewResult", "generate_seismic_views"]
