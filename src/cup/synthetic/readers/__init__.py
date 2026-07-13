"""Domain-specific Synthoseis-lite artifact readers."""

from cup.synthetic.readers.depth import DepthBenchmark, DepthSyntheticSample
from cup.synthetic.readers.time import TimeBenchmark, TimeSyntheticSample

__all__ = [
    "DepthSyntheticSample",
    "DepthBenchmark",
    "TimeBenchmark",
    "TimeSyntheticSample",
]
