"""Primary time-domain Synthoseis-lite facade.

The root package is the default time-domain product.  The depth-domain
extension lives under :mod:`cup.synthetic.depth`; readers and reporting stay
separate from generation so consumers depend on materialized contracts.
"""

from cup.synthetic.benchmark import SyntheticSample, SynthoseisBenchmark
from cup.synthetic.core.calibration import (
    ImpedanceCalibration,
    WellZoneCurves,
    calibrate_impedance,
)
from cup.synthetic.reporting.metrics import (
    aggregate_metric_rows,
    energy_rms,
    finite_mask,
    metric_row,
    regression_metrics,
)
from cup.synthetic.time.forward import (
    HighresForwardResult,
    HighresWavelet,
    antialias_taps,
    downsample_continuous,
    highres_forward_to_model_grid,
    resample_wavelet_to_highres,
)
from cup.synthetic.core.generation import (
    GeneratedSection,
    GenerationRejected,
    GenerationScenario,
)
from cup.synthetic.time.generation import generate_field_section
from cup.synthetic.time.lfm import LfmResult, derive_lfm_priors, lowpass_model_grid
from cup.synthetic.time.seismic_variants import (
    SeismicVariantResult,
    generate_seismic_variants,
)

__all__ = [
    "GeneratedSection",
    "GenerationRejected",
    "GenerationScenario",
    "HighresForwardResult",
    "HighresWavelet",
    "ImpedanceCalibration",
    "LfmResult",
    "SeismicVariantResult",
    "SyntheticSample",
    "SynthoseisBenchmark",
    "WellZoneCurves",
    "aggregate_metric_rows",
    "antialias_taps",
    "calibrate_impedance",
    "downsample_continuous",
    "energy_rms",
    "finite_mask",
    "generate_field_section",
    "generate_seismic_variants",
    "highres_forward_to_model_grid",
    "derive_lfm_priors",
    "lowpass_model_grid",
    "metric_row",
    "regression_metrics",
    "resample_wavelet_to_highres",
]
