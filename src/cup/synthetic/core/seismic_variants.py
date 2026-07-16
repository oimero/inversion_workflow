"""Shared metadata contract for observed-seismic mismatch variants."""

from __future__ import annotations

from typing import Any, Mapping

from cup.synthetic.schemas import SEISMIC_VARIANT_CONTRACT_VERSION


SEISMIC_VARIANT_REQUIRED_FIELDS = (
    "variant_id",
    "mismatch_family",
    "operator_source",
)


def build_seismic_variant_metadata(
    *,
    variant_id: str,
    mismatch_family: str,
    operator_source: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the domain-neutral part of one variant result row."""
    variant = str(variant_id).strip()
    family = str(mismatch_family).strip()
    source = str(operator_source).strip()
    if not variant or not family or not source:
        raise ValueError("Variant id, family and operator_source must be non-empty.")
    parameter_values = dict(parameters or {})
    overlap = set(parameter_values).intersection(SEISMIC_VARIANT_REQUIRED_FIELDS)
    if overlap:
        raise ValueError(
            "Variant parameters cannot overwrite contract fields: "
            + ", ".join(sorted(overlap))
        )
    return {
        "contract_version": SEISMIC_VARIANT_CONTRACT_VERSION,
        "variant_id": variant,
        "mismatch_family": family,
        "operator_source": source,
        **parameter_values,
    }


def validate_seismic_variant_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted variant row without inspecting domain arrays."""
    if not isinstance(value, Mapping):
        raise ValueError("seismic variant metadata must be a mapping.")
    if "valid_sample_count" in value:
        raise ValueError(
            "Variant metadata uses the shared valid_mask; valid_sample_count is not variant-specific."
        )
    missing = [key for key in SEISMIC_VARIANT_REQUIRED_FIELDS if key not in value]
    if missing:
        raise ValueError(f"Seismic variant metadata lacks fields: {missing}")
    if value.get("contract_version") != SEISMIC_VARIANT_CONTRACT_VERSION:
        raise ValueError("seismic variant metadata does not use seismic_variants_v2")
    return build_seismic_variant_metadata(
        variant_id=str(value["variant_id"]),
        mismatch_family=str(value["mismatch_family"]),
        operator_source=str(value["operator_source"]),
        parameters={
            str(key): item
            for key, item in value.items()
            if key not in (*SEISMIC_VARIANT_REQUIRED_FIELDS, "contract_version")
        },
    )


__all__ = [
    "SEISMIC_VARIANT_REQUIRED_FIELDS",
    "build_seismic_variant_metadata",
    "validate_seismic_variant_metadata",
]
