"""Domain-neutral object model facade.

The object model is still implemented by the historical time-v1 modules.
Depth-v2 reaches it only through ``cup.synthetic.depth.object_core_adapter``;
new shared callers should use this facade while the implementation is extracted.
"""

from cup.synthetic.calibration import (  # noqa: F401
    ImpedanceCalibration,
    PROFILE_METRICS,
    WellZoneCurves,
    calibrate_impedance,
    object_profile_metrics,
)
from cup.synthetic.generation import (  # noqa: F401
    GeneratedSection,
    GenerationRejected,
    GenerationScenario,
    generate_field_section,
)
from cup.synthetic.random import named_rng, named_seed  # noqa: F401

__all__ = [
    "GeneratedSection",
    "GenerationRejected",
    "GenerationScenario",
    "ImpedanceCalibration",
    "PROFILE_METRICS",
    "WellZoneCurves",
    "calibrate_impedance",
    "generate_field_section",
    "named_rng",
    "named_seed",
    "object_profile_metrics",
]
