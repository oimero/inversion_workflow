"""Time-domain view of a materialized shared Benchmark sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.synthetic.core.records import BenchmarkSample


def _time_catalog(records: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    names = {
        "object_top_coordinate": "object_top_s",
        "object_bottom_coordinate": "object_bottom_s",
        "minimum_extent": "minimum_duration_s",
        "maximum_extent": "maximum_duration_s",
    }
    return [
        {names.get(key, key): value for key, value in row.items()}
        for row in records
    ]


@dataclass(frozen=True)
class TimeBenchmarkSample:
    sample: BenchmarkSample

    @property
    def realization_id(self) -> str:
        return self.sample.truth.realization_id

    @property
    def scenario(self):
        return self.sample.truth.scenario

    @property
    def lateral_m(self) -> np.ndarray:
        return self.sample.truth.lateral_m

    @property
    def valid_mask_model(self) -> np.ndarray:
        return self.sample.valid_mask

    @property
    def seismic_observed(self) -> np.ndarray:
        return self.sample.forward.seismic_observed

    @property
    def seismic_model_consistent(self) -> np.ndarray:
        return self.sample.forward.seismic_model_consistent

    @property
    def object_catalog(self) -> list[dict[str, Any]]:
        return _time_catalog(self.sample.truth.object_catalog)

    @property
    def object_lateral_coefficients(self) -> list[dict[str, Any]]:
        return _time_catalog(self.sample.truth.object_lateral_coefficients)

    @property
    def qc(self) -> dict[str, Any]:
        return dict(self.sample.qc)


__all__ = ["TimeBenchmarkSample"]
