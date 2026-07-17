"""Time adapter for the shared Synthoseis-lite v5 reader."""

from __future__ import annotations

from cup.synthetic.readers.v5 import V5Benchmark, V5SeismicView, V5SyntheticSample


TimeSyntheticSample = V5SyntheticSample


class TimeBenchmark(V5Benchmark):
    sample_domain = "time"

    def __init__(self, run_dir):
        super().__init__(run_dir, sample_domain="time")


__all__ = ["TimeBenchmark", "TimeSyntheticSample", "V5SeismicView"]
