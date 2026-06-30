"""Domain-specific Synthoseis-lite artifact readers."""

from cup.synthetic.readers.depth_v2 import DepthSyntheticSample, DepthV2Benchmark
from cup.synthetic.readers.time_v2 import TimeV2Benchmark, TimeV2SyntheticSample

__all__ = [
    "DepthSyntheticSample",
    "DepthV2Benchmark",
    "TimeV2Benchmark",
    "TimeV2SyntheticSample",
]
