"""Depth adapter for the shared Synthoseis-lite v5 reader."""

from __future__ import annotations

from cup.synthetic.readers.v5 import V5Benchmark, V5SeismicView, V5SyntheticSample


DepthSyntheticSample = V5SyntheticSample


class DepthBenchmark(V5Benchmark):
    sample_domain = "depth"

    def __init__(self, run_dir):
        super().__init__(run_dir, sample_domain="depth")


__all__ = ["DepthSyntheticSample", "DepthBenchmark", "V5SeismicView"]
