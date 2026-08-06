"""Stable structured synthetic benchmark consumption interface."""

from cup.synthetic.benchmark import (
    ParentIdentity,
    StructuredParent,
    StructuredSyntheticBenchmark,
    SynthoseisBenchmark,
    SyntheticSample,
)
from cup.synthetic.core.prior import ProducerPrior, load_producer_prior

__all__ = [
    "ParentIdentity",
    "ProducerPrior",
    "StructuredParent",
    "StructuredSyntheticBenchmark",
    "SynthoseisBenchmark",
    "SyntheticSample",
    "load_producer_prior",
]
