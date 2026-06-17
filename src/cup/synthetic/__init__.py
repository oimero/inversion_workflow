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
from cup.synthetic.dataset import SyntheticSample, SynthoseisBenchmark
from cup.synthetic.dsp import antialias_taps, downsample_continuous
from cup.synthetic.generation import (
    GeneratedSection,
    GenerationScenario,
    generate_field_section,
)
from cup.synthetic.metrics import (
    aggregate_metric_rows,
    energy_rms,
    finite_mask,
    metric_row,
    regression_metrics,
)
from cup.synthetic.forward import (
    HighresForwardResult,
    HighresWavelet,
    highres_forward_to_model_grid,
    resample_wavelet_to_highres,
)
from cup.synthetic.lfm import (
    LfmResult,
    derive_lfm_priors,
    lowpass_model_grid,
)
from cup.synthetic.probes import (
    ProbeFrequency,
    ProbeResult,
    ProbeVariant,
    build_probe_frequency_catalog,
    generate_probe,
)
from cup.synthetic.seismic_variants import (
    SeismicVariantResult,
    generate_seismic_variants,
)

__all__ = [
    "GeneratedSection",
    "GenerationScenario",
    "HighresForwardResult",
    "HighresWavelet",
    "LfmResult",
    "CanonicalScenario",
    "ImpedanceCalibration",
    "SyntheticSample",
    "SynthoseisBenchmark",
    "WellZoneCurves",
    "aggregate_metric_rows",
    "antialias_taps",
    "calibrate_impedance",
    "canonical_scenarios",
    "downsample_continuous",
    "energy_rms",
    "finite_mask",
    "generate_canonical_section",
    "generate_field_section",
    "ProbeFrequency",
    "ProbeResult",
    "ProbeVariant",
    "SeismicVariantResult",
    "build_probe_frequency_catalog",
    "generate_probe",
    "generate_seismic_variants",
    "highres_forward_to_model_grid",
    "derive_lfm_priors",
    "lowpass_model_grid",
    "metric_row",
    "regression_metrics",
    "resample_wavelet_to_highres",
]
