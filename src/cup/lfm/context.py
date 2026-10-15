"""Optional upper/lower calculation context for real-field LFM variants.

The primary LFM remains the model on the configured target interval.  This
module builds a complete calculation field containing that primary interval
and optional surrounding samples, so downstream forward operators can use
the surrounding samples without changing the primary artifact's support or
values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from cup.lfm.builders import BUILDERS
from cup.lfm.types import LfmContext, LfmVariantResult
from cup.well.controls import WellControlSet


@dataclass(frozen=True)
class ContextExtensionSettings:
    """Validated context-extension settings in both physical and sample units."""

    baseline_id: str
    top_axis_units: float
    bottom_axis_units: float
    transition_axis_units: float
    top_extension_samples: int
    bottom_extension_samples: int
    top_resolved_axis_units: float
    bottom_resolved_axis_units: float

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "baseline_id": self.baseline_id,
            "top_axis_units": self.top_axis_units,
            "bottom_axis_units": self.bottom_axis_units,
            "transition_axis_units": self.transition_axis_units,
            "top_extension_samples": self.top_extension_samples,
            "bottom_extension_samples": self.bottom_extension_samples,
            "top_resolved_axis_units": self.top_resolved_axis_units,
            "bottom_resolved_axis_units": self.bottom_resolved_axis_units,
            "seam_smoothing": {
                "enabled": True,
                "transition_axis_units": self.transition_axis_units,
                "weight": "1 - 3*u^2 + 2*u^3",
            },
        }


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"context_extension.{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"context_extension.{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"context_extension.{name} must be a finite number.")
    return result


def _extension_samples(value: float, *, axis_step: float, name: str) -> tuple[int, float]:
    if value < 0.0:
        raise ValueError(f"context_extension.{name} must be >= 0.")
    if not np.isfinite(axis_step) or axis_step <= 0.0:
        raise ValueError("Context extension requires a regular SampleAxis with a positive step.")
    count = int(np.floor(value / axis_step + 0.5))
    if value > 0.0 and count < 1:
        raise ValueError(
            f"context_extension.{name} is shorter than one sample step ({axis_step:g} {name.split('_')[0]} units)."
        )
    return count, float(count * axis_step)


def resolve_context_extension(
    config: Mapping[str, Any],
    *,
    baselines: Mapping[str, Mapping[str, Any]],
    sample_axis: Any,
) -> ContextExtensionSettings | None:
    """Validate and resolve the optional ``context_extension`` configuration."""

    raw = config.get("context_extension")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("real_field_lfm.context_extension must be a mapping.")
    allowed = {
        "enabled",
        "baseline_id",
        "top_axis_units",
        "bottom_axis_units",
        "transition_axis_units",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"context_extension contains unknown fields: {unknown}.")
    enabled = raw.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("context_extension.enabled must be a YAML boolean.")
    if not enabled:
        return None

    required = allowed
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Enabled context_extension is missing fields: {missing}.")
    baseline_id = str(raw.get("baseline_id") or "").strip()
    if not baseline_id or baseline_id not in baselines:
        raise ValueError(
            "context_extension.baseline_id must reference an existing real_field_lfm.baselines entry."
        )
    baseline = baselines[baseline_id]
    method = str(baseline.get("method") or "")
    if method not in BUILDERS:
        raise ValueError(f"context_extension baseline {baseline_id!r} has an unknown method {method!r}.")

    top = _finite_number(raw["top_axis_units"], name="top_axis_units")
    bottom = _finite_number(raw["bottom_axis_units"], name="bottom_axis_units")
    transition = _finite_number(raw["transition_axis_units"], name="transition_axis_units")
    if top < 0.0 or bottom < 0.0:
        raise ValueError("context_extension top_axis_units and bottom_axis_units must be >= 0.")
    if top == 0.0 and bottom == 0.0:
        raise ValueError("context_extension must extend at least one side.")
    if transition <= 0.0:
        raise ValueError("context_extension.transition_axis_units must be positive.")

    top_count, top_resolved = _extension_samples(
        top, axis_step=float(sample_axis.step), name="top_axis_units"
    )
    bottom_count, bottom_resolved = _extension_samples(
        bottom, axis_step=float(sample_axis.step), name="bottom_axis_units"
    )
    if top_count == 0 and bottom_count == 0:
        raise ValueError("context_extension distances resolve to zero samples on this SampleAxis.")
    if transition < float(sample_axis.step):
        raise ValueError("context_extension.transition_axis_units must be at least one sample step.")

    return ContextExtensionSettings(
        baseline_id=baseline_id,
        top_axis_units=top,
        bottom_axis_units=bottom,
        transition_axis_units=transition,
        top_extension_samples=top_count,
        bottom_extension_samples=bottom_count,
        top_resolved_axis_units=top_resolved,
        bottom_resolved_axis_units=bottom_resolved,
    )


def build_context_field(
    *,
    config: Mapping[str, Any],
    controls: WellControlSet,
    context: LfmContext,
) -> tuple[ContextExtensionSettings | None, LfmVariantResult | None]:
    """Build the selected baseline on the extended target interval."""

    baselines = config.get("baselines")
    if not isinstance(baselines, Mapping):
        raise ValueError("real_field_lfm.baselines must be a mapping.")
    settings = resolve_context_extension(
        config,
        baselines=baselines,
        sample_axis=context.sample_axis,
    )
    if settings is None:
        return None, None
    baseline_config = baselines[settings.baseline_id]
    method = str(baseline_config.get("method") or "")
    extended_zone = context.target_zone.with_boundary_extension(
        top_extension_samples=settings.top_extension_samples,
        bottom_extension_samples=settings.bottom_extension_samples,
        top_extension_name="context_top",
        bottom_extension_name="context_bottom",
    )
    extension_horizons = [
        name
        for name in ("context_top", "context_bottom")
        if name in extended_zone.horizon_names
    ]
    extended_context = replace(
        context,
        target_zone=extended_zone,
        common_sources={
            **dict(context.common_sources),
            "context_extension_horizons": extension_horizons,
        },
    )
    generated = BUILDERS[method].build(
        baseline_id=settings.baseline_id,
        config=dict(baseline_config),
        controls=controls,
        context=extended_context,
    )
    return settings, generated


def _smooth_external_seams(
    *,
    samples: np.ndarray,
    generated_values: np.ndarray,
    generated_mask: np.ndarray,
    primary_values: np.ndarray,
    primary_mask: np.ndarray,
    transition_axis_units: float,
) -> np.ndarray:
    """Smooth an external field into the unchanged primary support."""

    axis = np.asarray(samples, dtype=np.float64)
    values = np.asarray(generated_values, dtype=np.float64).copy()
    generated = np.asarray(generated_mask, dtype=bool)
    primary = np.asarray(primary_mask, dtype=bool)
    base = np.asarray(primary_values, dtype=np.float64)
    if axis.ndim != 1 or axis.size != values.shape[-1] or np.any(np.diff(axis) <= 0.0):
        raise ValueError("Context seam samples must match a strictly increasing result axis.")
    if values.shape != generated.shape or values.shape != primary.shape or values.shape != base.shape:
        raise ValueError("Context seam arrays must have matching shapes.")
    taper = float(transition_axis_units)
    if not np.isfinite(taper) or taper <= 0.0:
        raise ValueError("Context seam transition must be finite and positive.")

    flat_values = values.reshape(-1, axis.size)
    flat_generated = generated.reshape(-1, axis.size)
    flat_primary = primary.reshape(-1, axis.size)
    flat_base = base.reshape(-1, axis.size)
    sample_indices = np.arange(axis.size)
    for row in range(flat_values.shape[0]):
        inner = np.flatnonzero(flat_primary[row])
        if inner.size == 0:
            continue
        for direction in ("top", "bottom"):
            edge = int(inner[0] if direction == "top" else inner[-1])
            if direction == "top":
                external = np.flatnonzero(flat_generated[row] & (sample_indices < edge))
                nearest = int(external[-1]) if external.size else None
            else:
                external = np.flatnonzero(flat_generated[row] & (sample_indices > edge))
                nearest = int(external[0]) if external.size else None
            if nearest is None:
                continue
            offset = float(flat_base[row, edge] - flat_values[row, nearest])
            if not np.isfinite(offset):
                raise ValueError(f"Context seam anchor is non-finite on row {row} ({direction}).")
            for index in external:
                index = int(index)
                distance = (
                    float(axis[nearest] - axis[index])
                    if direction == "top"
                    else float(axis[index] - axis[nearest])
                )
                u = min(1.0, max(0.0, distance / taper))
                weight = 1.0 - 3.0 * u * u + 2.0 * u * u * u
                if weight > 0.0:
                    flat_values[row, index] += offset * weight
    return values.astype(np.float32)


def attach_context(
    result: LfmVariantResult,
    generated: LfmVariantResult,
    settings: ContextExtensionSettings,
    *,
    context: LfmContext,
) -> LfmVariantResult:
    """Attach a complete calculation field while preserving the primary field."""

    result.validate(context)
    generated.validate(context)
    primary_mask = np.asarray(result.valid_mask_model, dtype=bool)
    external = np.asarray(generated.valid_mask_model, dtype=bool) & ~primary_mask
    values = np.full(result.log_ai.shape, np.nan, dtype=np.float32)
    values[primary_mask] = np.asarray(result.log_ai, dtype=np.float32)[primary_mask]
    values[external] = np.asarray(generated.log_ai, dtype=np.float32)[external]
    values = _smooth_external_seams(
        samples=np.asarray(context.output_geometry.samples, dtype=np.float64),
        generated_values=values,
        generated_mask=external,
        primary_values=np.asarray(result.log_ai, dtype=np.float32),
        primary_mask=np.asarray(result.valid_mask_model, dtype=bool),
        transition_axis_units=settings.transition_axis_units,
    )
    context_mask = primary_mask | external
    values[~context_mask] = np.nan
    attached = replace(
        result,
        context_log_ai=values,
        context_valid_mask=context_mask,
        metadata={
            **dict(result.metadata),
            "context_extension": settings.describe(),
        },
    )
    attached.validate(context)
    return attached


__all__ = [
    "ContextExtensionSettings",
    "attach_context",
    "build_context_field",
    "resolve_context_extension",
]
