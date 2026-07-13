"""Contracts shared by Synthoseis-lite domain adapters.

The generators deliberately keep their physical forward operators separate.
This module only fixes the meaning of the data consumed by a reader.
"""

from __future__ import annotations

from typing import Any, Mapping


SEISMIC_INPUT_POLICY = "observed_highres_forward"
SEISMIC_INPUT_SAMPLE_AXIS = "model"
SEISMIC_INPUT_VARIANT_FAMILY = "observed"
SEISMIC_MODEL_CONSISTENT_ROLE = "physics_and_closure"
SEISMIC_INPUT_OPERATORS = {
    "time": "time_forward_highres_wavelet_antialias",
    "depth": "depth_ai_vp_highres_forward_antialias",
}


def build_seismic_input_contract(
    sample_domain: str,
    *,
    operator: str,
) -> dict[str, Any]:
    """Build the domain-neutral seismic input contract for a v4 manifest."""
    domain = str(sample_domain).strip().casefold()
    if domain not in {"time", "depth"}:
        raise ValueError(f"Unsupported Synthoseis sample domain: {sample_domain!r}")
    operator_name = str(operator).strip()
    if operator_name != SEISMIC_INPUT_OPERATORS[domain]:
        raise ValueError(
            "Unsupported observed-input operator for "
            f"{domain}: {operator!r}."
        )
    return {
        "policy": SEISMIC_INPUT_POLICY,
        "sample_domain": domain,
        "sample_axis": SEISMIC_INPUT_SAMPLE_AXIS,
        "input_dataset": "seismic_observed",
        "base_source": "truth_highres_forward",
        "variant_input_family": SEISMIC_INPUT_VARIANT_FAMILY,
        "model_consistent_dataset": "seismic_model_consistent",
        "model_consistent_role": SEISMIC_MODEL_CONSISTENT_ROLE,
        "operator": operator_name,
    }


def validate_seismic_input_contract(
    value: Mapping[str, Any],
    *,
    sample_domain: str,
) -> dict[str, Any]:
    """Validate and return a materialized seismic input contract."""
    if not isinstance(value, Mapping):
        raise ValueError("seismic_input_contract must be a mapping.")
    expected_domain = str(sample_domain).strip().casefold()
    required = {
        "policy",
        "sample_domain",
        "sample_axis",
        "input_dataset",
        "base_source",
        "variant_input_family",
        "model_consistent_dataset",
        "model_consistent_role",
        "operator",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            "seismic_input_contract lacks required fields: " + ", ".join(missing)
        )
    if str(value["policy"]) != SEISMIC_INPUT_POLICY:
        raise ValueError(
            "Synthoseis v4 requires "
            f"seismic_input_contract.policy={SEISMIC_INPUT_POLICY!r}."
        )
    if str(value["sample_domain"]).casefold() != expected_domain:
        raise ValueError("seismic_input_contract sample_domain does not match reader.")
    if str(value["sample_axis"]) != SEISMIC_INPUT_SAMPLE_AXIS:
        raise ValueError("seismic_input_contract.sample_axis must be 'model'.")
    if str(value["input_dataset"]) != "seismic_observed":
        raise ValueError(
            "seismic_input_contract.input_dataset must be 'seismic_observed'."
        )
    if str(value["base_source"]) != "truth_highres_forward":
        raise ValueError(
            "seismic_input_contract.base_source must be truth_highres_forward."
        )
    if str(value["variant_input_family"]) != SEISMIC_INPUT_VARIANT_FAMILY:
        raise ValueError(
            "seismic_input_contract.variant_input_family must be observed."
        )
    if str(value["model_consistent_dataset"]) != "seismic_model_consistent":
        raise ValueError(
            "seismic_input_contract.model_consistent_dataset must be "
            "'seismic_model_consistent'."
        )
    if str(value["model_consistent_role"]) != SEISMIC_MODEL_CONSISTENT_ROLE:
        raise ValueError(
            "seismic_input_contract.model_consistent_role must be "
            "physics_and_closure."
        )
    operator = str(value["operator"]).strip()
    expected_operator = SEISMIC_INPUT_OPERATORS[expected_domain]
    if operator != expected_operator:
        raise ValueError(
            "seismic_input_contract.operator must be "
            f"{expected_operator!r} for {expected_domain}."
        )
    return {str(key): item for key, item in value.items()}


__all__ = [
    "SEISMIC_INPUT_POLICY",
    "SEISMIC_INPUT_SAMPLE_AXIS",
    "SEISMIC_INPUT_VARIANT_FAMILY",
    "SEISMIC_MODEL_CONSISTENT_ROLE",
    "SEISMIC_INPUT_OPERATORS",
    "build_seismic_input_contract",
    "validate_seismic_input_contract",
]
