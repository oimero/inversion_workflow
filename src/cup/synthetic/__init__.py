"""Truth-first synthetic impedance calibration and generation."""

from cup.synthetic.calibration import (
    ImpedanceCalibration,
    WellZoneCurves,
    calibrate_impedance,
)
from cup.synthetic.canonical import (
    CanonicalScenario,
    canonical_scenarios,
    generate_canonical_section,
)
from cup.synthetic.generation import (
    GeneratedSection,
    GenerationScenario,
    generate_field_section,
)

__all__ = [
    "GeneratedSection",
    "GenerationScenario",
    "CanonicalScenario",
    "ImpedanceCalibration",
    "WellZoneCurves",
    "calibrate_impedance",
    "canonical_scenarios",
    "generate_canonical_section",
    "generate_field_section",
]
