"""Public shared Synthoseis-lite Pipeline and Adapter interfaces."""

from cup.synthetic.adapters import (
    DepthSyntheticDomainAdapter,
    TimeSyntheticDomainAdapter,
)
from cup.synthetic.core.pipeline import (
    SeismicViewContext,
    SeismicViewPipeline,
    SyntheticBenchmarkPipeline,
    SyntheticDomainAdapter,
)

__all__ = [
    "DepthSyntheticDomainAdapter",
    "SeismicViewContext",
    "SeismicViewPipeline",
    "SyntheticBenchmarkPipeline",
    "SyntheticDomainAdapter",
    "TimeSyntheticDomainAdapter",
]
