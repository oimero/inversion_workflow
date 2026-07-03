"""Public domain-neutral types for unified real-field LFM construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry
from cup.seismic.target_zone import TargetZone
from cup.well.real_field_controls import WellControlSet


@dataclass(frozen=True)
class OutputGeometry:
    mode: str
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray

    def __post_init__(self) -> None:
        mode = str(self.mode).casefold()
        if mode not in {"volume", "window", "section"}:
            raise ValueError(f"Unsupported output geometry mode: {self.mode!r}.")
        ilines = np.asarray(self.ilines, dtype=np.float64)
        xlines = np.asarray(self.xlines, dtype=np.float64)
        samples = np.asarray(self.samples, dtype=np.float64)
        x_m = np.asarray(self.x_m, dtype=np.float64)
        y_m = np.asarray(self.y_m, dtype=np.float64)
        if samples.ndim != 1 or samples.size == 0 or np.any(np.diff(samples) <= 0.0):
            raise ValueError("Output samples must be a non-empty strictly increasing axis.")
        if mode == "section":
            if ilines.ndim != 1 or xlines.shape != ilines.shape or x_m.shape != ilines.shape or y_m.shape != ilines.shape:
                raise ValueError("Section line/XY arrays must be matching one-dimensional trace arrays.")
        else:
            if ilines.ndim != 1 or xlines.ndim != 1 or ilines.size == 0 or xlines.size == 0:
                raise ValueError("Volume/window line axes must be non-empty one-dimensional arrays.")
            if np.any(np.diff(ilines) <= 0.0) or np.any(np.diff(xlines) <= 0.0):
                raise ValueError("Volume/window line axes must be strictly increasing real line numbers.")
            if x_m.shape != (ilines.size, xlines.size) or y_m.shape != x_m.shape:
                raise ValueError("Volume/window XY grids do not match line axes.")
        for name, value in (("ilines", ilines), ("xlines", xlines), ("samples", samples), ("x_m", x_m), ("y_m", y_m)):
            if np.any(~np.isfinite(value)):
                raise ValueError(f"OutputGeometry {name} contains non-finite values.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "mode", mode)

    @property
    def is_section(self) -> bool:
        return self.mode == "section"

    @property
    def lateral_shape(self) -> tuple[int, ...]:
        return (self.ilines.size,) if self.is_section else (self.ilines.size, self.xlines.size)

    @property
    def volume_shape(self) -> tuple[int, ...]:
        return self.lateral_shape + (self.samples.size,)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "shape": list(self.volume_shape),
            "n_inline_or_trace": int(self.ilines.size),
            "n_xline": None if self.is_section else int(self.xlines.size),
            "n_sample": int(self.samples.size),
            "sample_min": float(self.samples[0]),
            "sample_max": float(self.samples[-1]),
        }


@dataclass(frozen=True)
class LfmContext:
    line_geometry: SurveyLineGeometry
    sample_axis: SampleAxis
    target_zone: TargetZone
    output_geometry: OutputGeometry
    depth_basis: str | None
    common_sources: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.allclose(self.target_zone.samples, self.sample_axis.values, rtol=0.0, atol=1e-9):
            raise ValueError("TargetZone and LfmContext SampleAxis differ.")
        selected = np.searchsorted(self.sample_axis.values, self.output_geometry.samples)
        selected = np.clip(selected, 0, self.sample_axis.values.size - 1)
        if not np.allclose(
            self.sample_axis.values[selected], self.output_geometry.samples, rtol=0.0, atol=1e-9
        ):
            raise ValueError("OutputGeometry samples must be an exact subset of the LfmContext SampleAxis.")
        if self.sample_axis.domain == "depth" and self.depth_basis != "tvdss":
            raise ValueError("Depth LFM context requires depth_basis='tvdss'.")
        if self.sample_axis.domain == "time" and self.depth_basis is not None:
            raise ValueError("Time LFM context cannot declare depth_basis.")


@dataclass
class LfmVariantResult:
    log_ai: np.ndarray
    valid_mask_model: np.ndarray
    baseline_id: str
    baseline_method: str
    method_fields: dict[str, np.ndarray] = field(default_factory=dict)
    modifier_fields: dict[str, np.ndarray] = field(default_factory=dict)
    qc_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, context: LfmContext) -> None:
        values = np.asarray(self.log_ai, dtype=np.float64)
        mask = np.asarray(self.valid_mask_model, dtype=bool)
        if values.shape != context.output_geometry.volume_shape or mask.shape != values.shape:
            raise ValueError("LFM result shape does not match output geometry.")
        if not np.any(mask):
            raise ValueError("LFM result valid_mask_model is empty in the requested output geometry.")
        if not np.all(np.isfinite(values[mask])):
            raise ValueError("LFM result contains non-finite values inside valid_mask_model.")
        if np.any(np.isfinite(values[~mask])):
            raise ValueError("LFM result must be NaN outside valid_mask_model.")
        self.log_ai = values.astype(np.float32)
        self.valid_mask_model = mask


class LfmBuilder(Protocol):
    method: str

    def build(
        self,
        *,
        baseline_id: str,
        config: Mapping[str, Any],
        controls: WellControlSet,
        context: LfmContext,
    ) -> LfmVariantResult: ...


class LfmModifier(Protocol):
    method: str

    def apply(
        self,
        *,
        modifier_id: str,
        config: Mapping[str, Any],
        parent: LfmVariantResult,
        context: LfmContext,
    ) -> LfmVariantResult: ...


__all__ = ["LfmBuilder", "LfmContext", "LfmModifier", "LfmVariantResult", "OutputGeometry"]
