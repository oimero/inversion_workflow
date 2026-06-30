"""Domain-neutral configuration contract for the shared impedance object core."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


LATERAL_KEYS = {
    "correlation_length_section_fractions",
    "coefficient_sigma_multipliers",
    "thickness_log_sigma_values",
}
QC_KEYS = {
    "max_global_reversal_fraction",
    "max_object_reversal_fraction",
    "max_global_clipping_fraction",
    "max_object_clipping_fraction",
}
ROBUST_SCALE_KEYS = {
    "huber_delta_parent_sigma_floor",
    "coefficient_sigma_parent_floor",
    "coefficient_sigma_parent_cap",
}


def _required_mapping(
    value: Mapping[str, Any], key: str, *, path: str, expected_keys: set[str]
) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"{path}.{key} must be a mapping.")
    item = dict(item)
    unknown = sorted(set(item) - expected_keys)
    if unknown:
        raise ValueError(f"{path}.{key} contains unknown keys: {unknown}.")
    missing = sorted(expected_keys - set(item))
    if missing:
        raise ValueError(f"{path}.{key} lacks required keys: {missing}.")
    return item


def _finite_list(value: Any, *, path: str, nonnegative: bool) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list.")
    parsed = [float(item) for item in value]
    if any(not np.isfinite(item) for item in parsed):
        raise ValueError(f"{path} must contain finite values.")
    if nonnegative and any(item < 0.0 for item in parsed):
        raise ValueError(f"{path} must contain nonnegative values.")
    if not nonnegative and any(item <= 0.0 for item in parsed):
        raise ValueError(f"{path} must contain positive values.")
    return parsed


def parse_object_core_controls(
    impedance: Mapping[str, Any],
    *,
    path: str = "synthoseis_lite.impedance_attribute_generator",
) -> dict[str, Any]:
    """Parse controls whose semantics are identical in time and depth."""
    lateral = _required_mapping(
        impedance, "lateral", path=path, expected_keys=LATERAL_KEYS
    )
    qc = _required_mapping(impedance, "qc", path=path, expected_keys=QC_KEYS)
    robust = _required_mapping(
        impedance, "robust_scale", path=path, expected_keys=ROBUST_SCALE_KEYS
    )

    correlations = _finite_list(
        lateral["correlation_length_section_fractions"],
        path=f"{path}.lateral.correlation_length_section_fractions",
        nonnegative=False,
    )
    coefficient_sigmas = _finite_list(
        lateral["coefficient_sigma_multipliers"],
        path=f"{path}.lateral.coefficient_sigma_multipliers",
        nonnegative=True,
    )
    thickness_sigmas = _finite_list(
        lateral["thickness_log_sigma_values"],
        path=f"{path}.lateral.thickness_log_sigma_values",
        nonnegative=True,
    )
    if len(coefficient_sigmas) != len(thickness_sigmas):
        raise ValueError(
            f"{path}.lateral coefficient_sigma_multipliers and "
            "thickness_log_sigma_values must have equal length."
        )

    parsed_qc: dict[str, float] = {}
    for key in sorted(QC_KEYS):
        value = float(qc[key])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}.qc.{key} must be within [0, 1].")
        parsed_qc[key] = value

    parsed_robust: dict[str, float] = {}
    for key in sorted(ROBUST_SCALE_KEYS):
        value = float(robust[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{path}.robust_scale.{key} must be nonnegative.")
        parsed_robust[key] = value
    if (
        parsed_robust["coefficient_sigma_parent_floor"]
        > parsed_robust["coefficient_sigma_parent_cap"]
    ):
        raise ValueError(
            f"{path}.robust_scale coefficient sigma floor must not exceed cap."
        )

    return {
        "correlation_length_section_fractions": correlations,
        "coefficient_sigma_multipliers": coefficient_sigmas,
        "thickness_log_sigma_values": thickness_sigmas,
        **parsed_qc,
        **parsed_robust,
    }


__all__ = [
    "LATERAL_KEYS",
    "QC_KEYS",
    "ROBUST_SCALE_KEYS",
    "parse_object_core_controls",
]
