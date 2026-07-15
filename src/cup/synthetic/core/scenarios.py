"""Field-conditioned scientific scenario records and catalog construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationScenario:
    """One scientific object-sequence and lateral-geometry scenario."""

    scenario_id: str
    duration_mode: str
    geometry_family: str
    geometry_direction: str
    correlation_length_fraction: float
    coefficient_sigma_multiplier: float
    thickness_log_sigma: float
    geometry_variant_id: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must be non-empty.")
        supported_families = {
            "none",
            "wedge",
            "pinchout",
            "horizontal_thin_beds",
            "dipping_layers",
            "lateral_impedance_change",
        }
        if self.geometry_family not in supported_families:
            raise ValueError(f"Unsupported geometry family: {self.geometry_family}")
        if self.duration_mode == "canonical":
            expected_directions = {"fixed"}
        else:
            expected_directions = {"none"} if self.geometry_family == "none" else {
                "left_to_right",
                "right_to_left",
            }
        if self.geometry_direction not in expected_directions:
            raise ValueError(
                f"Invalid geometry direction {self.geometry_direction!r} "
                f"for {self.geometry_family}."
            )
        if self.geometry_family == "pinchout" and self.duration_mode != "canonical":
            if self.geometry_variant_id not in {"035", "065"}:
                raise ValueError("pinchout geometry_variant_id must be '035' or '065'.")
        elif self.geometry_variant_id:
            raise ValueError("geometry_variant_id is only valid for pinchout scenarios.")
        for name in (
            "correlation_length_fraction",
            "coefficient_sigma_multiplier",
            "thickness_log_sigma",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or (self.duration_mode != "canonical" and value == 0.0):
                raise ValueError(f"{name} must be positive for field scenarios.")


def generation_scenarios(script_cfg: Mapping[str, Any]) -> list[GenerationScenario]:
    """Convert parsed controls into the frozen field-conditioned scenario catalog."""
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
                        for geometry_variant_id in variants:
                            scenario_id = (
                                f"{duration_mode}__lx{correlation:g}__a{coefficient_sigma:g}"
                                f"__t{thickness_sigma:g}__{family}__{direction}"
                                + (
                                    f"__{geometry_variant_id}"
                                    if geometry_variant_id
                                    else ""
                                )
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
                                    geometry_variant_id=geometry_variant_id,
                                )
                            )
    return scenarios


__all__ = ["GenerationScenario", "generation_scenarios"]
