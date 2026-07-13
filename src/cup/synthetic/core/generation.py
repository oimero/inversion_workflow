"""Domain-neutral generated-object records shared by time and depth adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class GenerationScenario:
    scenario_id: str
    duration_mode: str
    geometry_family: str
    geometry_direction: str
    correlation_length_fraction: float
    coefficient_sigma_multiplier: float
    thickness_log_sigma: float
    variant_id: str = ""


@dataclass(frozen=True)
class GeneratedSection:
    realization_id: str
    scenario: GenerationScenario
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    twt_highres_s: np.ndarray
    twt_model_s: np.ndarray
    truth_log_ai_highres: np.ndarray
    model_target_log_ai: np.ndarray
    reflectivity_highres: np.ndarray
    reflectivity_model: np.ndarray
    seismic_model_consistent: np.ndarray
    rgt_highres: np.ndarray
    rgt_model: np.ndarray
    state_id_highres: np.ndarray
    object_id_highres: np.ndarray
    object_xi_highres: np.ndarray
    zone_id_highres: np.ndarray
    geometry_event_mask_highres: np.ndarray
    boundary_mask_highres: np.ndarray
    boundary_fraction_model: np.ndarray
    boundary_mask_model: np.ndarray
    state_fraction_model: np.ndarray
    dominant_object_id_model: np.ndarray
    zone_id_model: np.ndarray
    valid_mask_model: np.ndarray
    forward_valid_mask_highres: np.ndarray
    forward_valid_mask_model: np.ndarray
    object_catalog: list[dict[str, Any]]
    object_lateral_coefficients: list[dict[str, Any]]
    qc: dict[str, Any]


class GenerationRejected(ValueError):
    """A complete realization that failed one or more frozen QC rules."""

    def __init__(
        self,
        reasons: Sequence[str],
        *,
        diagnostics: Mapping[str, Any],
        details: Sequence[Mapping[str, Any]],
    ) -> None:
        unique_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__(";".join(unique_reasons))
        self.reasons = unique_reasons
        self.diagnostics = dict(diagnostics)
        self.details = [dict(item) for item in details]


def generation_scenarios(script_cfg: Mapping[str, Any]) -> list[GenerationScenario]:
    """Build the shared HSMM/object scenario catalog from parsed controls."""
    generation = script_cfg["generation"]
    impedance = script_cfg["impedance"]
    scenarios: list[GenerationScenario] = []
    for duration_mode in generation["duration_modes"]:
        for correlation in impedance["correlation_length_section_fractions"]:
            pairs = zip(
                impedance["coefficient_sigma_multipliers"],
                impedance["thickness_log_sigma_values"],
            )
            for coefficient_sigma, thickness_sigma in pairs:
                for family in generation["geometry_families"]:
                    directions = (
                        generation["geometry_directions"] if family != "none" else ["none"]
                    )
                    variants = ["035", "065"] if family == "pinchout" else [""]
                    for direction in directions:
                        for variant in variants:
                            scenario_id = (
                                f"{duration_mode}__lx{correlation:g}__a{coefficient_sigma:g}"
                                f"__t{thickness_sigma:g}__{family}__{direction}"
                                + (f"__{variant}" if variant else "")
                            )
                            scenarios.append(
                                GenerationScenario(
                                    scenario_id=scenario_id,
                                    duration_mode=str(duration_mode),
                                    geometry_family=str(family),
                                    geometry_direction=str(direction),
                                    correlation_length_fraction=float(correlation),
                                    coefficient_sigma_multiplier=float(coefficient_sigma),
                                    thickness_log_sigma=float(thickness_sigma),
                                    variant_id=str(variant),
                                )
                            )
    return scenarios


__all__ = [
    "GeneratedSection",
    "GenerationRejected",
    "GenerationScenario",
    "generation_scenarios",
]
