"""Stable facade for canonical structured synthetic benchmarks."""

from __future__ import annotations

from cup.synthetic.readers.structured import (
    ParentIdentity,
    StructuredParent,
    StructuredSyntheticBenchmark,
)

SynthoseisBenchmark = StructuredSyntheticBenchmark
SyntheticSample = StructuredParent

__all__ = [
    "ParentIdentity",
    "StructuredParent",
    "StructuredSyntheticBenchmark",
    "SynthoseisBenchmark",
    "SyntheticSample",
]
