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
from cup.synthetic.forward import (
    HighresForwardResult,
    HighresWavelet,
    highres_forward_to_model_grid,
    resample_wavelet_to_highres,
)
from cup.synthetic.probes import (
    ProbeFrequency,
    ProbeResult,
    ProbeVariant,
    build_probe_frequency_catalog,
    generate_probe,
)

__all__ = [
    "GeneratedSection",
    "GenerationScenario",
    "HighresForwardResult",
    "HighresWavelet",
    "CanonicalScenario",
    "ImpedanceCalibration",
    "WellZoneCurves",
    "calibrate_impedance",
    "canonical_scenarios",
    "generate_canonical_section",
    "generate_field_section",
    "ProbeFrequency",
    "ProbeResult",
    "ProbeVariant",
    "build_probe_frequency_catalog",
    "generate_probe",
    "highres_forward_to_model_grid",
    "resample_wavelet_to_highres",
]
