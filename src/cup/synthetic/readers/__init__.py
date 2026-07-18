"""Domain-specific Synthoseis-lite artifact readers."""

from typing import Any

from cup.synthetic.readers.depth import DepthBenchmark, DepthSyntheticSample
from cup.synthetic.readers.time import TimeBenchmark, TimeSyntheticSample

__all__ = [
    "DepthSyntheticSample",
    "DepthBenchmark",
    "TimeBenchmark",
    "TimeSyntheticSample",
    "SyntheticSampleProtocol",
]


def __getattr__(name: str) -> Any:
    if name == "SyntheticSampleProtocol":
        from cup.synthetic.benchmark import SyntheticSampleProtocol

        return SyntheticSampleProtocol
    raise AttributeError(name)
