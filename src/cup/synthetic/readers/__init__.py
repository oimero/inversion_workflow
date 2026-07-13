"""Domain-specific Synthoseis-lite artifact readers."""

from cup.synthetic.readers.depth import DepthBenchmark, DepthSyntheticSample
from cup.synthetic.readers.time import TimeBenchmark, TimeSyntheticSample
from cup.synthetic.core.protocols import SyntheticSampleProtocol

__all__ = [
    "DepthSyntheticSample",
    "DepthBenchmark",
    "TimeBenchmark",
    "TimeSyntheticSample",
    "SyntheticSampleProtocol",
]
