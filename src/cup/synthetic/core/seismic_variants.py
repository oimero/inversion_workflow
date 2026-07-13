"""Shared metadata contract for observed-seismic mismatch variants."""

from __future__ import annotations

import math
from typing import Any, Mapping


SEISMIC_VARIANT_REQUIRED_FIELDS = (
    "variant_id",
    "mismatch_family",
    "operator_source",
    "valid_sample_count",
)


def _valid_sample_count(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Variant valid_sample_count must be an integer.") from exc
    if not math.isfinite(number) or number != math.floor(number):
        raise ValueError("Variant valid_sample_count must be an integer.")
    count = int(number)
    if count < 1:
        raise ValueError("A seismic variant must contain a valid sample.")
    return count


def build_seismic_variant_metadata(
    *,
    variant_id: str,
    mismatch_family: str,
    operator_source: str,
    valid_sample_count: int,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the domain-neutral part of one variant result row."""
    variant = str(variant_id).strip()
    family = str(mismatch_family).strip()
    source = str(operator_source).strip()
    if not variant or not family or not source:
        raise ValueError("Variant id, family and operator_source must be non-empty.")
    count = _valid_sample_count(valid_sample_count)
    parameter_values = dict(parameters or {})
    overlap = set(parameter_values).intersection(SEISMIC_VARIANT_REQUIRED_FIELDS)
    if overlap:
        raise ValueError(
            "Variant parameters cannot overwrite contract fields: "
            + ", ".join(sorted(overlap))
        )
    return {
        "variant_id": variant,
        "mismatch_family": family,
        "operator_source": source,
        "valid_sample_count": count,
        **parameter_values,
    }


def validate_seismic_variant_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted variant row without inspecting domain arrays."""
    if not isinstance(value, Mapping):
        raise ValueError("seismic variant metadata must be a mapping.")
    missing = [key for key in SEISMIC_VARIANT_REQUIRED_FIELDS if key not in value]
    if missing:
        raise ValueError(f"Seismic variant metadata lacks fields: {missing}")
    return build_seismic_variant_metadata(
        variant_id=str(value["variant_id"]),
        mismatch_family=str(value["mismatch_family"]),
        operator_source=str(value["operator_source"]),
        valid_sample_count=_valid_sample_count(value["valid_sample_count"]),
        parameters={
            str(key): item
            for key, item in value.items()
            if key not in SEISMIC_VARIANT_REQUIRED_FIELDS
        },
    )


__all__ = [
    "SEISMIC_VARIANT_REQUIRED_FIELDS",
    "build_seismic_variant_metadata",
    "validate_seismic_variant_metadata",
]
