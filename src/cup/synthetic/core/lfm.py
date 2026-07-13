"""Domain-neutral metadata for low-frequency-model degradation."""

from __future__ import annotations

from typing import Any, Mapping


LFM_VARIANT_IDS = ("canonical", "controlled_default")
LFM_COMPONENT_ORDER = (
    "constant_bias",
    "axis_trend",
    "zonewise_bias",
    "lateral_smooth_bias",
    "amplitude_scale",
    "local_missing_control_bias",
    "over_smoothing",
)


def build_lfm_degradation_metadata(
    sample_domain: str,
    *,
    axis_unit: str,
    component_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the shared LFM degradation sequence without fixing its operator."""
    domain = str(sample_domain).strip().casefold()
    if domain not in {"time", "depth"}:
        raise ValueError(f"Unsupported LFM sample domain: {sample_domain!r}")
    unit = str(axis_unit).strip()
    if not unit:
        raise ValueError("LFM axis_unit must be non-empty.")
    values = dict(component_values or {})
    return {
        "sample_domain": domain,
        "axis_unit": unit,
        "variant_ids": list(LFM_VARIANT_IDS),
        "component_order": list(LFM_COMPONENT_ORDER),
        "component_values": values,
        "canonicalization": "canonical_lowpass_difference_once",
    }


def validate_lfm_degradation_metadata(
    value: Mapping[str, Any],
    *,
    sample_domain: str,
) -> dict[str, Any]:
    """Validate the shared LFM component/variant metadata in a manifest."""
    if not isinstance(value, Mapping):
        raise ValueError("lfm_degradation must be a mapping.")
    expected = build_lfm_degradation_metadata(
        sample_domain,
        axis_unit="s" if str(sample_domain).casefold() == "time" else "m",
    )
    for key in (
        "sample_domain",
        "axis_unit",
        "variant_ids",
        "component_order",
        "component_values",
        "canonicalization",
    ):
        if key not in value:
            raise ValueError(f"lfm_degradation lacks required field: {key}")
    if str(value["sample_domain"]).casefold() != expected["sample_domain"]:
        raise ValueError("lfm_degradation sample_domain does not match reader.")
    if str(value["axis_unit"]) != expected["axis_unit"]:
        raise ValueError("lfm_degradation axis_unit does not match reader.")
    if list(value["variant_ids"]) != list(LFM_VARIANT_IDS):
        raise ValueError("lfm_degradation variant_ids do not match v4.")
    if list(value["component_order"]) != list(LFM_COMPONENT_ORDER):
        raise ValueError("lfm_degradation component_order does not match v4.")
    if not isinstance(value["component_values"], Mapping):
        raise ValueError("lfm_degradation component_values must be a mapping.")
    if str(value["canonicalization"]) != "canonical_lowpass_difference_once":
        raise ValueError("lfm_degradation canonicalization is not supported.")
    return {str(key): item for key, item in value.items()}


__all__ = [
    "LFM_COMPONENT_ORDER",
    "LFM_VARIANT_IDS",
    "build_lfm_degradation_metadata",
    "validate_lfm_degradation_metadata",
]
