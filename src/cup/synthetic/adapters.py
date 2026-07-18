"""Concrete domain adapters for the shared Synthoseis-lite seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cup.synthetic.core.pipeline import SyntheticDomainAdapter


@dataclass
class TimeSyntheticDomainAdapter:
    """Time-domain axis/unit adapter used by the shared benchmark pipeline."""

    forward_callback: Callable[[float, float], tuple[np.ndarray, np.ndarray]] | None = None
    sample_domain: str = "time"
    sample_unit: str = "s"
    depth_basis: str | None = None
    generator_family: str = ""
    runtime: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def validate_axis(self, sample_axis: np.ndarray) -> None:
        axis = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
        differences = np.diff(axis)
        if (
            axis.size < 2
            or not np.all(np.isfinite(axis))
            or not np.all(differences > 0.0)
            or not np.allclose(differences, differences[0], rtol=0.0, atol=1e-9)
        ):
            raise ValueError("time adapter requires a finite regular increasing seconds axis")

    def forward_with_parameters(self, phase_degrees: float, shift: float):
        if self.forward_callback is None:
            raise ValueError("time forward context is not configured")
        return self.forward_callback(float(phase_degrees), float(shift))

    def prepare_generation(self, config: Mapping[str, Any], calibration: Any, *, output_dir: Path, **runtime: Any):
        from cup.synthetic.time.pipeline import TimeGenerationSession

        values = {**dict(self.runtime), **runtime}
        return TimeGenerationSession.prepare(
            config,
            calibration,
            output_dir=output_dir,
            **values,
        )

    def prepare_calibration(self, config: Mapping[str, Any], *, output_dir: Path, **runtime: Any):
        from cup.synthetic.time.calibration_pipeline import TimeCalibrationResult

        values = {**dict(self.runtime), **runtime}
        return TimeCalibrationResult.prepare(config, output_dir=output_dir, **values)


@dataclass
class DepthSyntheticDomainAdapter:
    """Depth-domain axis/unit adapter used by the shared benchmark pipeline."""

    forward_callback: Callable[[float, float], tuple[np.ndarray, np.ndarray]] | None = None
    sample_domain: str = "depth"
    sample_unit: str = "m"
    depth_basis: str = "tvdss"
    generator_family: str = ""
    runtime: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def validate_axis(self, sample_axis: np.ndarray) -> None:
        axis = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
        differences = np.diff(axis)
        if (
            axis.size < 2
            or not np.all(np.isfinite(axis))
            or not np.all(differences > 0.0)
            or not np.allclose(differences, differences[0], rtol=0.0, atol=1e-9)
        ):
            raise ValueError("depth adapter requires a finite regular increasing TVDSS metres axis")

    def forward_with_parameters(self, phase_degrees: float, shift: float):
        if self.forward_callback is None:
            raise ValueError("depth forward context is not configured")
        return self.forward_callback(float(phase_degrees), float(shift))

    def prepare_generation(self, config: Mapping[str, Any], calibration: Any, *, output_dir: Path, **runtime: Any):
        from cup.synthetic.depth.generation import DepthGenerationSession

        values = {**dict(self.runtime), **runtime}
        return DepthGenerationSession.prepare(
            config,
            calibration,
            output_dir=output_dir,
            **values,
        )

    def prepare_calibration(self, config: Mapping[str, Any], *, output_dir: Path, **runtime: Any):
        from cup.synthetic.depth.calibration import DepthCalibrationResult

        values = {**dict(self.runtime), **runtime}
        return DepthCalibrationResult.prepare(config, output_dir=output_dir, **values)


__all__ = [
    "DepthSyntheticDomainAdapter",
    "SyntheticDomainAdapter",
    "TimeSyntheticDomainAdapter",
]
