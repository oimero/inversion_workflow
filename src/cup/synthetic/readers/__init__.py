"""Domain-specific Synthoseis-lite artifact readers."""

from cup.synthetic.readers.depth_v2 import DepthSyntheticSample, DepthV2Benchmark
from cup.synthetic.readers.time_v1 import TimeSyntheticSample, TimeV1Benchmark

__all__ = [
    "DepthSyntheticSample",
    "DepthV2Benchmark",
    "TimeSyntheticSample",
    "TimeV1Benchmark",
]
