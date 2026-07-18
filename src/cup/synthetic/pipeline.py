"""Public shared Synthoseis-lite Pipeline and Adapter interfaces."""

from cup.synthetic.adapters import (
    DepthSyntheticDomainAdapter,
    TimeSyntheticDomainAdapter,
)
from cup.synthetic.core.pipeline import (
    GenerationAttempt,
    GenerationSession,
    SeismicViewContext,
    SeismicViewPipeline,
    SyntheticBenchmarkPipeline,
    SyntheticDomainAdapter,
)

__all__ = [
    "DepthSyntheticDomainAdapter",
    "GenerationAttempt",
    "GenerationSession",
    "SeismicViewContext",
    "SeismicViewPipeline",
    "SyntheticBenchmarkPipeline",
    "SyntheticDomainAdapter",
    "TimeSyntheticDomainAdapter",
]
