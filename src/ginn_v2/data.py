"""Seismic patch data, spatial splits, and trusted-well targets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry
from cup.utils.masks import true_runs as _finite_runs
from cup.well.controls import WellControl, WellControlSet
from cup.well.scale import gaussian_smooth_finite_runs_numpy
from ginn_v2.model import BodySmoother


Orientation = Literal["inline", "xline"]
SeismicFeatureMode = Literal["global_trace_normalized", "local_amplitude_balanced"]


def _local_amplitude_balanced_trace(
    trace: np.ndarray,
    support: np.ndarray,
    *,
    window_samples: int,
    floor_fraction: float,
) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    kernel = np.ones(int(window_samples), dtype=np.float64)
    weights = valid.astype(np.float64)
    count = np.convolve(weights, kernel, mode="same")
    total = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    total_square = np.convolve(np.where(valid, np.square(values), 0.0), kernel, mode="same")
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    variance = np.maximum(
        np.divide(total_square, count, out=np.zeros_like(total_square), where=count > 0.0) - np.square(mean),
        0.0,
    )
    global_scale = float(np.sqrt(np.mean(np.square(values[valid] - np.mean(values[valid])))))
    denominator = np.maximum(np.sqrt(variance), float(floor_fraction) * global_scale)
    output = np.zeros(values.shape, dtype=np.float32)
    output[valid] = ((values[valid] - mean[valid]) / denominator[valid]).astype(np.float32)
    return output


def _finite_float_array(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True, order=True)
class PatchKey:
    """One center trace and one 2-D profile orientation.

    ``inline_index`` and ``xline_index`` are zero-based positions in the
    current survey arrays.  They are intentionally not line numbers.
    """

    inline_index: int
    xline_index: int
    orientation: Orientation = "inline"

    def __post_init__(self) -> None:
        if isinstance(self.inline_index, bool) or int(self.inline_index) != self.inline_index:
            raise TypeError("inline_index must be an integer array index.")
        if isinstance(self.xline_index, bool) or int(self.xline_index) != self.xline_index:
            raise TypeError("xline_index must be an integer array index.")
        if self.inline_index < 0 or self.xline_index < 0:
            raise ValueError("PatchKey array indices must be non-negative.")
        if self.orientation not in {"inline", "xline"}:
            raise ValueError("PatchKey orientation must be 'inline' or 'xline'.")


class TraceSource(Protocol):
    """Small seam for an in-memory or file-backed seismic trace source."""

    sample_axis: SampleAxis
    geometry: SurveyLineGeometry

    def read_traces(self, indices: Iterable[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]: ...


@dataclass(frozen=True)
class ArrayTraceSource:
    """TraceSource adapter for a ``(inline, xline, sample)`` NumPy volume."""

    volume: np.ndarray
    sample_axis: SampleAxis
    geometry: SurveyLineGeometry

    def __post_init__(self) -> None:
        volume = np.asarray(self.volume)
        if volume.ndim != 3 or not np.issubdtype(volume.dtype, np.floating):
            raise ValueError("ArrayTraceSource volume must be a floating 3-D array.")
        expected = (
            self.geometry.inline_axis.count,
            self.geometry.xline_axis.count,
            self.sample_axis.values.size,
        )
        if volume.shape != expected:
            raise ValueError(f"ArrayTraceSource volume shape {volume.shape} differs from {expected}.")
        object.__setattr__(self, "volume", volume)

    def read_traces(self, indices: Iterable[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
        result: dict[tuple[int, int], np.ndarray] = {}
        for inline_index, xline_index in sorted({(int(i), int(j)) for i, j in indices}):
            if not (0 <= inline_index < self.volume.shape[0] and 0 <= xline_index < self.volume.shape[1]):
                raise ValueError(f"Trace index is outside ArrayTraceSource: {(inline_index, xline_index)}")
            result[(inline_index, xline_index)] = np.asarray(
                self.volume[inline_index, xline_index, :], dtype=np.float64
            ).copy()
        return result


@dataclass(frozen=True)
class SurveyTraceSource:
    """TraceSource adapter for the existing SEG-Y/ZGY survey adapters."""

    survey: object
    sample_axis: SampleAxis
    geometry: SurveyLineGeometry

    def read_traces(self, indices: Iterable[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
        requested = sorted({(int(i), int(j)) for i, j in indices})
        traces = self.survey.read_traces_at_indices(
            requested,
            domain=self.sample_axis.domain,
        )
        result: dict[tuple[int, int], np.ndarray] = {}
        for key in requested:
            if key not in traces:
                raise ValueError(f"Survey trace source did not return requested trace: {key}")
            trace = traces[key]
            basis = np.asarray(trace.basis, dtype=np.float64)
            if not np.array_equal(basis, self.sample_axis.values):
                raise ValueError("Survey trace SampleAxis differs from the common training SampleAxis.")
            values = np.asarray(trace.values, dtype=np.float64)
            if values.shape != self.sample_axis.values.shape:
                raise ValueError(f"Survey trace {key} has an unexpected sample shape: {values.shape}")
            result[key] = values.copy()
        return result


@dataclass(frozen=True)
class InputNormalization:
    """Frozen feature normalization used by every batch and checkpoint."""

    lfm_mean: float
    lfm_scale: float
    geometry_scale_m: float

    def __post_init__(self) -> None:
        values = (self.lfm_mean, self.lfm_scale, self.geometry_scale_m)
        if any(not np.isfinite(float(value)) for value in values):
            raise ValueError("InputNormalization values must be finite.")
        if self.lfm_scale <= 0.0 or self.geometry_scale_m <= 0.0:
            raise ValueError("lfm_scale and geometry_scale_m must be positive.")


def fit_lfm_normalization(
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    *,
    geometry: SurveyLineGeometry,
) -> InputNormalization:
    """Fit the one frozen LFM statistic used by the inversion run.

    The statistic is computed only on finite LFM support.  Geometry scale is
    the median physical inline/xline spacing, never a line-number step.
    """

    values = np.asarray(lfm_log_ai)
    mask = np.asarray(lfm_valid_mask, dtype=bool)
    if values.ndim != 3 or values.shape != mask.shape:
        raise ValueError("lfm_log_ai and lfm_valid_mask must be matching 3-D arrays.")
    support = mask & np.isfinite(values)
    if not np.any(support):
        raise ValueError("LFM normalization has no finite valid support.")
    selected = np.asarray(values[support], dtype=np.float64)
    mean = float(np.mean(selected))
    scale = float(np.sqrt(np.mean(np.square(selected - mean))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("LFM normalization support must have positive variance.")
    spacing = geometry.bin_spacing_m()["nominal_bin_spacing_m"]
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Survey geometry must expose a positive metre bin spacing.")
    return InputNormalization(mean, scale, spacing)


@dataclass(frozen=True)
class PatchSample:
    """One normalized profile patch plus the center-trace supervision arrays."""

    key: PatchKey
    features: np.ndarray
    observed_seismic: np.ndarray
    observed_valid_mask: np.ndarray
    lfm_log_ai: np.ndarray
    lfm_valid_mask: np.ndarray
    xy_m: np.ndarray
    domain_extras: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        feature = np.asarray(self.features, dtype=np.float32)
        observed = np.asarray(self.observed_seismic, dtype=np.float32)
        observed_mask = np.asarray(self.observed_valid_mask, dtype=bool)
        lfm = np.asarray(self.lfm_log_ai, dtype=np.float32)
        lfm_mask = np.asarray(self.lfm_valid_mask, dtype=bool)
        if feature.ndim != 3 or observed.ndim != 1 or lfm.ndim != 1:
            raise ValueError("PatchSample features must be (channels, width, samples); traces must be 1-D.")
        if observed.shape != lfm.shape or observed_mask.shape != observed.shape or lfm_mask.shape != lfm.shape:
            raise ValueError("PatchSample center arrays must have matching shapes.")
        if feature.shape[1] < 1 or feature.shape[2] != observed.size:
            raise ValueError("PatchSample feature shape does not match the center sample count.")
        xy = np.asarray(self.xy_m, dtype=np.float64)
        if xy.shape != (2,) or np.any(~np.isfinite(xy)):
            raise ValueError("PatchSample xy_m must contain two finite metre coordinates.")
        if np.any(~np.isfinite(feature)) or np.any(~np.isfinite(observed)) or np.any(~np.isfinite(lfm)):
            raise ValueError("PatchSample arrays must be finite; masks represent support separately.")
        object.__setattr__(self, "features", feature)
        object.__setattr__(self, "observed_seismic", observed)
        object.__setattr__(self, "observed_valid_mask", observed_mask)
        object.__setattr__(self, "lfm_log_ai", lfm)
        object.__setattr__(self, "lfm_valid_mask", lfm_mask)
        object.__setattr__(self, "xy_m", xy)


@dataclass(frozen=True)
class PatchBatch:
    """Torch batch consumed by the shared network and domain adapter."""

    keys: tuple[PatchKey, ...]
    features: Tensor
    observed_seismic: Tensor
    observed_valid_mask: Tensor
    lfm_log_ai: Tensor
    lfm_valid_mask: Tensor
    xy_m: Tensor
    domain_extras: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError("PatchBatch.features must have shape (batch, channels, width, samples).")
        batch, _, _, samples = self.features.shape
        if len(self.keys) != batch:
            raise ValueError("PatchBatch key count differs from feature batch size.")
        for name, value in (
            ("observed_seismic", self.observed_seismic),
            ("lfm_log_ai", self.lfm_log_ai),
        ):
            if value.shape != (batch, samples):
                raise ValueError(f"PatchBatch {name} must have shape (batch, samples).")
            if not torch.is_floating_point(value):
                raise ValueError(f"PatchBatch {name} must be floating tensors.")
        for name, value in (("observed_valid_mask", self.observed_valid_mask), ("lfm_valid_mask", self.lfm_valid_mask)):
            if value.dtype != torch.bool or value.shape != (batch, samples):
                raise ValueError(f"PatchBatch {name} must be bool tensors matching center traces.")
        if self.xy_m.shape != (batch, 2) or not torch.is_floating_point(self.xy_m):
            raise ValueError("PatchBatch.xy_m must have shape (batch, 2) and floating dtype.")


class PatchReader:
    """Deep patch reader hiding line geometry, masking, and feature semantics."""

    input_channels = 6

    def __init__(
        self,
        source: TraceSource,
        *,
        lfm_log_ai: np.ndarray,
        lfm_valid_mask: np.ndarray,
        ilines: np.ndarray,
        xlines: np.ndarray,
        sample_axis: SampleAxis,
        normalization: InputNormalization,
        patch_radius: int,
        domain_extras: Mapping[str, np.ndarray] | None = None,
        cache_size: int = 256,
        seismic_feature_mode: SeismicFeatureMode = "global_trace_normalized",
        seismic_balance_window_samples: int = 61,
        seismic_balance_floor_fraction: float = 0.10,
    ) -> None:
        if isinstance(patch_radius, bool) or int(patch_radius) != patch_radius or patch_radius < 1:
            raise ValueError("patch_radius must be a positive integer.")
        if isinstance(cache_size, bool) or int(cache_size) < 0:
            raise ValueError("cache_size must be a non-negative integer.")
        if seismic_feature_mode not in {"global_trace_normalized", "local_amplitude_balanced"}:
            raise ValueError("Unsupported seismic_feature_mode.")
        if (
            isinstance(seismic_balance_window_samples, bool)
            or int(seismic_balance_window_samples) < 3
            or int(seismic_balance_window_samples) % 2 == 0
        ):
            raise ValueError("seismic_balance_window_samples must be an odd integer of at least three.")
        if not 0.0 < float(seismic_balance_floor_fraction) < 1.0:
            raise ValueError("seismic_balance_floor_fraction must be within (0, 1).")
        if int(seismic_balance_window_samples) > sample_axis.values.size:
            raise ValueError("seismic_balance_window_samples exceeds the SampleAxis length.")
        self.source = source
        self.geometry = source.geometry
        self.sample_axis = sample_axis
        self.patch_radius = int(patch_radius)
        self.width = 2 * self.patch_radius + 1
        self.normalization = normalization
        self.seismic_feature_mode = seismic_feature_mode
        self.seismic_balance_window_samples = int(seismic_balance_window_samples)
        self.seismic_balance_floor_fraction = float(seismic_balance_floor_fraction)
        self.ilines = _finite_float_array(ilines, name="ilines", ndim=1)
        self.xlines = _finite_float_array(xlines, name="xlines", ndim=1)
        self.lfm_log_ai = np.asarray(lfm_log_ai, dtype=np.float64)
        self.lfm_valid_mask = np.asarray(lfm_valid_mask, dtype=bool)
        expected = (self.ilines.size, self.xlines.size, self.sample_axis.values.size)
        if self.lfm_log_ai.shape != expected or self.lfm_valid_mask.shape != expected:
            raise ValueError(f"LFM arrays must have shape {expected}.")
        if not np.array_equal(self.ilines, self.geometry.inline_axis.values()):
            raise ValueError("LFM inline axis differs from SurveyLineGeometry.")
        if not np.array_equal(self.xlines, self.geometry.xline_axis.values()):
            raise ValueError("LFM xline axis differs from SurveyLineGeometry; line step is part of the contract.")
        if not np.array_equal(self.sample_axis.values, source.sample_axis.values):
            raise ValueError("PatchReader SampleAxis differs from TraceSource SampleAxis.")
        if np.any(self.lfm_valid_mask & ~np.isfinite(self.lfm_log_ai)):
            raise ValueError("LFM valid support contains non-finite values.")
        if np.any(np.isfinite(self.lfm_log_ai) & ~self.lfm_valid_mask):
            raise ValueError("LFM invalid support must be represented by non-finite values.")
        self.domain_extras = {
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in dict(domain_extras or {}).items()
        }
        for name, value in self.domain_extras.items():
            if value.shape != expected:
                raise ValueError(f"domain_extras[{name!r}] must have shape {expected}.")
            if np.any(np.isinf(value)):
                raise ValueError(f"domain_extras[{name!r}] must not contain infinite values.")
            if not np.any(np.isfinite(value)):
                raise ValueError(f"domain_extras[{name!r}] has no finite support.")
        self._cache_size = int(cache_size)
        self._trace_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._normalized_trace_cache: OrderedDict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()

    def _trace(self, index: tuple[int, int]) -> np.ndarray:
        if index in self._trace_cache:
            value = self._trace_cache.pop(index)
            self._trace_cache[index] = value
            return value.copy()
        values = self.source.read_traces([index])
        if index not in values:
            raise ValueError(f"TraceSource did not return requested trace {index}.")
        trace = np.asarray(values[index], dtype=np.float64)
        if trace.shape != self.sample_axis.values.shape:
            raise ValueError(f"Trace {index} does not match the SampleAxis shape.")
        if self._cache_size:
            self._trace_cache[index] = trace.copy()
            while len(self._trace_cache) > self._cache_size:
                self._trace_cache.popitem(last=False)
        return trace

    def _lateral_indices(self, key: PatchKey) -> list[tuple[int, int]]:
        i, j = key.inline_index, key.xline_index
        if not (0 <= i < self.ilines.size and 0 <= j < self.xlines.size):
            raise ValueError(f"PatchKey is outside the survey array: {key}")
        radius = self.patch_radius
        if key.orientation == "inline":
            if not radius <= j < self.xlines.size - radius:
                raise ValueError(f"Inline-oriented PatchKey is too close to an xline edge: {key}")
            return [(i, j + offset) for offset in range(-radius, radius + 1)]
        if not radius <= i < self.ilines.size - radius:
            raise ValueError(f"Xline-oriented PatchKey is too close to an inline edge: {key}")
        return [(i + offset, j) for offset in range(-radius, radius + 1)]

    def _normalized_trace(
        self,
        index: tuple[int, int],
        trace: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self._normalized_trace_cache.pop(index, None)
        if cached is not None:
            self._normalized_trace_cache[index] = cached
            return cached[0].copy(), cached[1].copy()
        support = np.isfinite(trace)
        if np.count_nonzero(support) < 2:
            raise ValueError(f"Trace {index} has fewer than two finite samples for normalization.")
        mean = float(np.mean(trace[support]))
        scale = float(np.sqrt(np.mean(np.square(trace[support] - mean))))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Trace {index} has no positive variance for normalization.")
        normalized = np.zeros(trace.shape, dtype=np.float32)
        normalized[support] = ((trace[support] - mean) / scale).astype(np.float32)
        if self.seismic_feature_mode == "local_amplitude_balanced":
            normalized = _local_amplitude_balanced_trace(
                trace,
                support,
                window_samples=self.seismic_balance_window_samples,
                floor_fraction=self.seismic_balance_floor_fraction,
            )
        if self._cache_size:
            self._normalized_trace_cache[index] = (normalized.copy(), support.copy())
            while len(self._normalized_trace_cache) > self._cache_size:
                self._normalized_trace_cache.popitem(last=False)
        return normalized, support

    def _xy(self, indices: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray(
            [
                self.geometry.line_to_coord(self.ilines[inline_index], self.xlines[xline_index])
                for inline_index, xline_index in indices
            ],
            dtype=np.float64,
        )

    def read(self, key: PatchKey, *, center_visible: bool) -> PatchSample:
        """Read one patch with an explicit center visibility semantic."""

        if not isinstance(center_visible, bool):
            raise TypeError("center_visible must be boolean.")
        indices = self._lateral_indices(key)
        missing_indices = [index for index in indices if index not in self._trace_cache]
        loaded: dict[tuple[int, int], np.ndarray] = {}
        if missing_indices:
            loaded = self.source.read_traces(missing_indices)
            for index in missing_indices:
                if index not in loaded:
                    raise ValueError(f"TraceSource did not return requested trace {index}.")
                trace = np.asarray(loaded[index], dtype=np.float64)
                if trace.shape != self.sample_axis.values.shape:
                    raise ValueError(f"Trace {index} does not match the SampleAxis shape.")
                if self._cache_size:
                    self._trace_cache[index] = trace.copy()
                    while len(self._trace_cache) > self._cache_size:
                        self._trace_cache.popitem(last=False)
        traces = np.stack(
            [
                np.asarray(loaded[index], dtype=np.float64)
                if index in loaded and index not in self._trace_cache
                else self._trace(index)
                for index in indices
            ],
            axis=0,
        )
        normalized_rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        for index, trace in zip(indices, traces):
            normalized_trace, trace_support = self._normalized_trace(index, trace)
            normalized_rows.append(normalized_trace)
            valid_rows.append(trace_support)
        normalized = np.stack(normalized_rows)
        trace_valid = np.stack(valid_rows)
        if not center_visible:
            normalized[self.patch_radius] = 0.0

        lfm_patch = np.asarray(self.lfm_log_ai[tuple(np.asarray(indices).T)], dtype=np.float64)
        lfm_valid_patch = np.asarray(self.lfm_valid_mask[tuple(np.asarray(indices).T)], dtype=bool)
        lfm_normalized = np.zeros_like(lfm_patch, dtype=np.float32)
        lfm_normalized[lfm_valid_patch] = (
            (lfm_patch[lfm_valid_patch] - self.normalization.lfm_mean) / self.normalization.lfm_scale
        ).astype(np.float32)
        missing = np.zeros_like(normalized, dtype=np.float32)
        if not center_visible:
            missing[self.patch_radius, :] = 1.0
        center_index = indices[self.patch_radius]
        center_seismic = traces[self.patch_radius].copy()
        # ``lfm_log_ai`` may be a read-only memmap.  The invalid-support
        # fallback below must therefore operate on a private writable copy.
        center_lfm = np.asarray(
            self.lfm_log_ai[center_index[0], center_index[1], :],
            dtype=np.float64,
        ).copy()
        center_lfm_valid = np.asarray(self.lfm_valid_mask[center_index[0], center_index[1], :], dtype=bool)
        center_valid = trace_valid[self.patch_radius].copy() & center_lfm_valid
        for value in self.domain_extras.values():
            center_valid &= np.isfinite(value[center_index[0], center_index[1], :])
        center_lfm[~center_lfm_valid] = self.normalization.lfm_mean
        xy = self._xy(indices)
        center_xy = xy[self.patch_radius]
        relative_xy = ((xy - center_xy[None, :]) / self.normalization.geometry_scale_m).astype(np.float32)
        lfm_valid_float = lfm_valid_patch.astype(np.float32)
        features = np.stack(
            (
                normalized,
                lfm_normalized,
                missing,
                lfm_valid_float,
                np.broadcast_to(relative_xy[:, 0, None], normalized.shape),
                np.broadcast_to(relative_xy[:, 1, None], normalized.shape),
            ),
            axis=0,
        )
        domain_extras = {
            name: np.asarray(value[center_index[0], center_index[1], :], dtype=np.float64)
            for name, value in self.domain_extras.items()
        }
        return PatchSample(
            key=key,
            features=features,
            observed_seismic=np.where(center_valid, center_seismic, 0.0).astype(np.float32),
            observed_valid_mask=center_valid,
            lfm_log_ai=np.where(center_lfm_valid, center_lfm, self.normalization.lfm_mean).astype(np.float32),
            lfm_valid_mask=center_lfm_valid,
            xy_m=center_xy,
            domain_extras=domain_extras,
        )

    def batch(self, keys: Iterable[PatchKey], *, center_visible: bool, device: torch.device | str) -> PatchBatch:
        samples = tuple(self.read(key, center_visible=center_visible) for key in keys)
        if not samples:
            raise ValueError("PatchReader.batch requires at least one PatchKey.")
        features = torch.from_numpy(np.stack([item.features for item in samples])).to(device)
        observed = torch.from_numpy(np.stack([item.observed_seismic for item in samples])).to(device)
        observed_mask = torch.from_numpy(np.stack([item.observed_valid_mask for item in samples])).to(device)
        lfm = torch.from_numpy(np.stack([item.lfm_log_ai for item in samples])).to(device)
        lfm_mask = torch.from_numpy(np.stack([item.lfm_valid_mask for item in samples])).to(device)
        xy = torch.from_numpy(np.stack([item.xy_m for item in samples])).to(device)
        extra_names = set().union(*(item.domain_extras.keys() for item in samples))
        extras: dict[str, Tensor] = {}
        for name in sorted(extra_names):
            values = []
            for item in samples:
                if name not in item.domain_extras:
                    raise ValueError(f"Patch batch domain extras are not uniform: missing {name!r}.")
                values.append(item.domain_extras[name])
            extras[name] = torch.from_numpy(np.stack(values)).to(device=device, dtype=torch.float32)
        return PatchBatch(
            keys=tuple(item.key for item in samples),
            features=features,
            observed_seismic=observed,
            observed_valid_mask=observed_mask,
            lfm_log_ai=lfm,
            lfm_valid_mask=lfm_mask,
            xy_m=xy,
            domain_extras=extras,
        )


def candidate_patch_keys(
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    *,
    patch_radius: int,
    orientations: Iterable[Orientation] = ("inline", "xline"),
    min_lfm_support: int = 8,
) -> tuple[PatchKey, ...]:
    """Return deterministic center identities with complete lateral patches."""

    values = np.asarray(lfm_log_ai)
    mask = np.asarray(lfm_valid_mask, dtype=bool)
    if values.ndim != 3 or values.shape != mask.shape:
        raise ValueError("candidate LFM arrays must be matching 3-D arrays.")
    if isinstance(patch_radius, bool) or int(patch_radius) != patch_radius or patch_radius < 1:
        raise ValueError("patch_radius must be a positive integer.")
    if isinstance(min_lfm_support, bool) or int(min_lfm_support) < 2:
        raise ValueError("min_lfm_support must be at least two.")
    selected = tuple(orientations)
    if not selected or any(item not in {"inline", "xline"} for item in selected):
        raise ValueError("orientations must contain inline and/or xline.")
    valid_center = np.count_nonzero(mask & np.isfinite(values), axis=-1) >= int(min_lfm_support)
    result: list[PatchKey] = []
    for i in range(patch_radius, values.shape[0] - patch_radius):
        for j in range(patch_radius, values.shape[1] - patch_radius):
            if not valid_center[i, j]:
                continue
            for orientation in selected:
                if orientation == "inline" and patch_radius <= j < values.shape[1] - patch_radius:
                    result.append(PatchKey(i, j, orientation))
                if orientation == "xline" and patch_radius <= i < values.shape[0] - patch_radius:
                    result.append(PatchKey(i, j, orientation))
    return tuple(result)


ValidationAnchor = Literal["maxmax", "maxmin", "minmax", "minmin", "center"]


@dataclass(frozen=True)
class SpatialSplit:
    train_keys: tuple[PatchKey, ...]
    validation_keys: tuple[PatchKey, ...]
    review_keys: tuple[PatchKey, ...]
    validation_centers: tuple[tuple[int, int], ...]
    block_xy_m: tuple[float, float, float, float]
    gap_m: float
    anchor: ValidationAnchor

    def __post_init__(self) -> None:
        train_centers = {(item.inline_index, item.xline_index) for item in self.train_keys}
        validation_centers = {(item.inline_index, item.xline_index) for item in self.validation_keys}
        review_centers = {(item.inline_index, item.xline_index) for item in self.review_keys}
        if train_centers & review_centers:
            raise ValueError("Spatial train and validation centers overlap.")
        if not self.train_keys or not self.validation_keys or not self.review_keys:
            raise ValueError("Spatial split must contain non-empty train and validation keys.")
        if not validation_centers <= review_centers:
            raise ValueError("Metric validation identities must be a subset of the review block.")
        if len(self.block_xy_m) != 4 or any(not np.isfinite(float(value)) for value in self.block_xy_m):
            raise ValueError("Spatial validation block must contain four finite metre bounds.")
        if self.gap_m < 0.0 or not np.isfinite(float(self.gap_m)):
            raise ValueError("Spatial split gap_m must be finite and non-negative.")


@dataclass(frozen=True)
class WellSampleSplit:
    well_name: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        train = set(self.train_indices)
        validation = set(self.validation_indices)
        if not train or train & validation:
            raise ValueError("Well split requires non-empty training indices and disjoint validation indices.")


@dataclass(frozen=True)
class WellTarget:
    well_name: str
    model_axis_target: np.ndarray
    valid_target_mask: np.ndarray
    native_body_target: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.model_axis_target, dtype=np.float64)
        mask = np.asarray(self.valid_target_mask, dtype=bool)
        native = np.asarray(self.native_body_target, dtype=np.float64)
        if target.ndim != 1 or mask.shape != target.shape or native.ndim != 1:
            raise ValueError("WellTarget arrays have invalid dimensions.")
        if np.any(mask & ~np.isfinite(target)) or np.any(np.isfinite(target) & ~mask):
            raise ValueError("WellTarget mask must exactly describe finite model-axis targets.")
        object.__setattr__(self, "model_axis_target", target)
        object.__setattr__(self, "valid_target_mask", mask)
        object.__setattr__(self, "native_body_target", native)


@dataclass(frozen=True)
class WellPatchTarget:
    well_name: str
    patch_key: PatchKey
    target_values: np.ndarray
    target_mask: np.ndarray
    target_scale: float

    def __post_init__(self) -> None:
        if not str(self.well_name).strip():
            raise ValueError("WellPatchTarget.well_name must be non-empty.")
        values = np.asarray(self.target_values, dtype=np.float64)
        mask = np.asarray(self.target_mask, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape or not np.any(mask):
            raise ValueError("WellPatchTarget requires a non-empty one-dimensional target mask.")
        if np.any(~np.isfinite(values)):
            raise ValueError("WellPatchTarget.target_values must be finite; target_mask carries support.")
        if not np.isfinite(float(self.target_scale)) or float(self.target_scale) <= 0.0:
            raise ValueError("WellPatchTarget.target_scale must be finite and positive.")
        object.__setattr__(self, "target_values", values)
        object.__setattr__(self, "target_mask", mask)


def _center_xy(key: PatchKey, geometry: SurveyLineGeometry) -> tuple[float, float]:
    inline = geometry.inline_axis.line_at_index(key.inline_index)
    xline = geometry.xline_axis.line_at_index(key.xline_index)
    return geometry.line_to_coord(inline, xline)


def make_spatial_split(
    keys: Iterable[PatchKey],
    *,
    geometry: SurveyLineGeometry,
    validation_fraction: float,
    gap_m: float,
    anchor: ValidationAnchor,
) -> SpatialSplit:
    """Select a fixed XY block using physical coordinates rather than line steps."""

    candidates = tuple(keys)
    if not candidates:
        raise ValueError("Cannot split an empty patch identity set.")
    fraction = float(validation_fraction)
    gap = float(gap_m)
    if not 0.0 < fraction < 1.0 or not np.isfinite(fraction):
        raise ValueError("validation_fraction must be finite and within (0, 1).")
    if gap < 0.0 or not np.isfinite(gap):
        raise ValueError("gap_m must be finite and non-negative.")
    if anchor not in {"maxmax", "maxmin", "minmax", "minmin", "center"}:
        raise ValueError("Unsupported validation block anchor.")
    centers = tuple(sorted({(item.inline_index, item.xline_index) for item in candidates}))
    xy = np.asarray([_center_xy(PatchKey(i, j), geometry) for i, j in centers], dtype=np.float64)
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    x_span = max(float(x_max - x_min), geometry.bin_spacing_m()["nominal_bin_spacing_m"])
    y_span = max(float(y_max - y_min), geometry.bin_spacing_m()["nominal_bin_spacing_m"])
    side = float(np.sqrt(fraction))
    block_x_span = x_span * side
    block_y_span = y_span * side
    if anchor == "center":
        x_start = float((x_min + x_max - block_x_span) / 2.0)
        y_start = float((y_min + y_max - block_y_span) / 2.0)
    else:
        x_start = float(x_max - block_x_span if anchor.startswith("max") else x_min)
        y_start = float(y_max - block_y_span if anchor.endswith("max") else y_min)
    x_stop = x_start + block_x_span
    y_stop = y_start + block_y_span
    center_validation: set[tuple[int, int]] = set()
    for (i, j), (x_value, y_value) in zip(centers, xy):
        if x_start <= x_value <= x_stop and y_start <= y_value <= y_stop:
            center_validation.add((i, j))
    if not center_validation or len(center_validation) == len(centers):
        raise ValueError("Spatial validation block does not produce both train and validation centers.")
    train_centers: set[tuple[int, int]] = set()
    for (i, j), (x_value, y_value) in zip(centers, xy):
        outside_gap = (
            x_value < x_start - gap
            or x_value > x_stop + gap
            or y_value < y_start - gap
            or y_value > y_stop + gap
        )
        if outside_gap:
            train_centers.add((i, j))
    if not train_centers:
        raise ValueError("Spatial gap removes every training center.")
    train_keys = tuple(item for item in candidates if (item.inline_index, item.xline_index) in train_centers)
    validation_keys = tuple(
        item for item in candidates if (item.inline_index, item.xline_index) in center_validation
    )
    return SpatialSplit(
        train_keys=train_keys,
        validation_keys=validation_keys,
        review_keys=validation_keys,
        validation_centers=tuple(sorted(center_validation)),
        block_xy_m=(x_start, x_stop, y_start, y_stop),
        gap_m=gap,
        anchor=anchor,
    )


def sample_lfm_trace(
    control: WellControl,
    *,
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    geometry: SurveyLineGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the volume LFM at the nearest physical survey center per well sample."""

    values = np.asarray(lfm_log_ai, dtype=np.float64)
    support = np.asarray(lfm_valid_mask, dtype=bool)
    expected = (
        geometry.inline_axis.count,
        geometry.xline_axis.count,
        control.sample_axis.values.size,
    )
    if values.shape != expected or support.shape != expected:
        raise ValueError(f"lfm_log_ai and lfm_valid_mask must have shape {expected}.")
    result = np.full(control.sample_axis.values.shape, np.nan, dtype=np.float64)
    result_mask = np.zeros(control.sample_axis.values.shape, dtype=bool)
    for sample_index, (x_m, y_m) in enumerate(zip(control.x_m_by_sample, control.y_m_by_sample)):
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            continue
        i_float, j_float = geometry.coord_to_index(float(x_m), float(y_m))
        i, j = int(round(i_float)), int(round(j_float))
        if (
            0 <= i < geometry.inline_axis.count
            and 0 <= j < geometry.xline_axis.count
            and abs(i_float - i) <= 0.5
            and abs(j_float - j) <= 0.5
            and support[i, j, sample_index]
            and np.isfinite(values[i, j, sample_index])
        ):
            result[sample_index] = values[i, j, sample_index]
            result_mask[sample_index] = True
    return result, result_mask


def well_target_zone_mask(
    control: WellControl,
    *,
    geometry: SurveyLineGeometry,
    target_zone_mask: np.ndarray,
) -> np.ndarray:
    """Sample the interpreted target-zone support along one well trajectory."""

    volume = np.asarray(target_zone_mask, dtype=bool)
    expected = (
        geometry.inline_axis.count,
        geometry.xline_axis.count,
        control.sample_axis.values.size,
    )
    if volume.shape != expected:
        raise ValueError(f"target_zone_mask must have shape {expected}.")
    result = np.zeros(control.sample_axis.values.shape, dtype=bool)
    for sample_index, (x_m, y_m) in enumerate(zip(control.x_m_by_sample, control.y_m_by_sample)):
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            continue
        i_float, j_float = geometry.coord_to_index(float(x_m), float(y_m))
        i, j = int(round(i_float)), int(round(j_float))
        if (
            0 <= i < geometry.inline_axis.count
            and 0 <= j < geometry.xline_axis.count
            and abs(i_float - i) <= 0.5
            and abs(j_float - j) <= 0.5
        ):
            result[sample_index] = volume[i, j, sample_index]
    return result


def build_well_body_target(
    control: WellControl,
    *,
    body_smoothing_fwhm_m: float,
    target_zone_support: np.ndarray,
    geometry: SurveyLineGeometry,
    smoother: BodySmoother,
    lfm_log_ai: np.ndarray | None = None,
    lfm_valid_mask: np.ndarray | None = None,
) -> WellTarget:
    """Build a trusted-well target by smoothing the complete model-axis curve once.

    The native filtered curve is first interpolated without smoothing onto the
    model sample axis.  The same physical Gaussian used on the network's
    complete initial-plus-correction curve is then applied once to that model
    axis.  The low-frequency model contributes only its support mask here; its
    values are deliberately not used to redefine the well target.
    """

    native_coordinates = np.asarray(control.native.coordinates, dtype=np.float64)
    native_values = np.asarray(control.native.native_filtered_log_ai, dtype=np.float64)
    # Keep this historical diagnostic for QC/reporting.  It is not used as
    # the training target; the actual target below starts from unsmoothed
    # native_filtered_log_ai and smooths once after model-axis interpolation.
    native_body = gaussian_smooth_finite_runs_numpy(
        native_values,
        native_coordinates,
        fwhm_m=body_smoothing_fwhm_m,
    )
    model_axis = np.asarray(control.sample_axis.values, dtype=np.float64)
    model_target_raw = np.full(model_axis.shape, np.nan, dtype=np.float64)
    for start, stop in _finite_runs(np.isfinite(native_values)):
        inside = (model_axis >= native_coordinates[start]) & (model_axis <= native_coordinates[stop - 1])
        model_target_raw[inside] = np.interp(
            model_axis[inside],
            native_coordinates[start:stop],
            native_values[start:stop],
        )
    native_model_support = np.isfinite(model_target_raw)
    model_target = smoother.smooth_numpy(
        model_target_raw,
        model_axis,
        native_model_support,
    )
    observed = np.asarray(control.observed_valid_mask, dtype=bool)
    zone_support = np.asarray(target_zone_support, dtype=bool)
    if zone_support.shape != observed.shape:
        raise ValueError("target_zone_support must match the well model axis.")
    if (lfm_log_ai is None) != (lfm_valid_mask is None):
        raise ValueError("lfm_log_ai and lfm_valid_mask must be provided together.")
    if lfm_log_ai is None:
        # The workflow's target-zone support already carries the LFM support
        # mask.  Keep the optional volume arguments for callers that want an
        # independent support cross-check, but never use their values.
        well_lfm_valid = np.ones_like(observed, dtype=bool)
    else:
        _well_lfm, well_lfm_valid = sample_lfm_trace(
            control,
            lfm_log_ai=lfm_log_ai,
            lfm_valid_mask=lfm_valid_mask,
            geometry=geometry,
        )
    valid_target = observed & zone_support & well_lfm_valid & np.isfinite(model_target)
    model_target[~valid_target] = np.nan
    if np.count_nonzero(valid_target) < 4:
        raise ValueError(f"{control.well_name}: native body target has fewer than four observed model samples.")
    return WellTarget(
        well_name=control.well_name,
        model_axis_target=model_target,
        valid_target_mask=valid_target,
        native_body_target=native_body,
    )


def build_well_splits(
    controls: WellControlSet,
    targets: Mapping[str, WellTarget],
) -> tuple[WellSampleSplit, ...]:
    """Use every trusted observed target sample as an anchor sample."""

    result: list[WellSampleSplit] = []
    for control in controls.controls:
        target = targets.get(control.well_name)
        if target is None:
            raise ValueError(f"Missing trusted well target: {control.well_name}")
        observed = target.valid_target_mask
        runs = _finite_runs(observed)
        if not runs:
            raise ValueError(f"{control.well_name}: no observed body-target run for the well split.")
        train_indices = tuple(int(index) for index in np.flatnonzero(observed))
        result.append(
            WellSampleSplit(
                well_name=control.well_name,
                train_indices=train_indices,
                validation_indices=(),
            )
        )
    return tuple(result)


def _nearest_patch_index(control: WellControl, sample_index: int, geometry: SurveyLineGeometry) -> tuple[int, int]:
    x_value = float(control.x_m_by_sample[sample_index])
    y_value = float(control.y_m_by_sample[sample_index])
    i_float, j_float = geometry.coord_to_index(x_value, y_value)
    i = int(round(i_float))
    j = int(round(j_float))
    if abs(i_float - i) > 0.5 or abs(j_float - j) > 0.5:
        raise ValueError(f"{control.well_name}: well sample is farther than half a trace from a seismic center.")
    return i, j


def build_well_patch_targets(
    controls: WellControlSet,
    targets: Mapping[str, WellTarget],
    splits: Iterable[WellSampleSplit],
    *,
    geometry: SurveyLineGeometry,
    subset: Literal["train", "validation"],
    orientations: Iterable[Orientation] = ("inline", "xline"),
) -> tuple[WellPatchTarget, ...]:
    """Group well targets by the actual center trace used by each sample."""

    control_by_name = {item.well_name: item for item in controls.controls}
    split_items = tuple(splits)
    selected_orientations = tuple(orientations)
    if not selected_orientations or any(item not in {"inline", "xline"} for item in selected_orientations):
        raise ValueError("well orientations must contain inline and/or xline.")
    result: list[WellPatchTarget] = []
    for split in split_items:
        control = control_by_name.get(split.well_name)
        target = targets.get(split.well_name)
        if control is None or target is None:
            raise ValueError(f"Well split references an unknown target: {split.well_name}")
        indices = split.train_indices if subset == "train" else split.validation_indices
        target_scale = float(np.std(target.model_axis_target[target.valid_target_mask]))
        if not np.isfinite(target_scale) or target_scale <= 0.0:
            raise ValueError(f"{split.well_name}: body target has no positive normalization scale.")
        grouped: dict[PatchKey, list[int]] = {}
        for sample_index in indices:
            i, j = _nearest_patch_index(control, sample_index, geometry)
            for orientation in selected_orientations:
                grouped.setdefault(PatchKey(i, j, orientation), []).append(int(sample_index))
        for patch_key, sample_indices in sorted(grouped.items()):
            values = np.zeros(target.model_axis_target.shape, dtype=np.float64)
            mask = np.zeros(target.model_axis_target.shape, dtype=bool)
            selected = np.asarray(sample_indices, dtype=np.int64)
            values[selected] = target.model_axis_target[selected]
            mask[selected] = True
            result.append(
                WellPatchTarget(
                    well_name=split.well_name,
                    patch_key=patch_key,
                    target_values=values,
                    target_mask=mask,
                    target_scale=target_scale,
                )
            )
    if not result:
        raise ValueError(f"No {subset} well patch targets were produced.")
    return tuple(result)


__all__ = [
    "ArrayTraceSource",
    "InputNormalization",
    "Orientation",
    "PatchBatch",
    "PatchKey",
    "PatchReader",
    "PatchSample",
    "SpatialSplit",
    "SurveyTraceSource",
    "TraceSource",
    "ValidationAnchor",
    "WellPatchTarget",
    "WellSampleSplit",
    "WellTarget",
    "build_well_body_target",
    "build_well_patch_targets",
    "build_well_splits",
    "candidate_patch_keys",
    "fit_lfm_normalization",
    "make_spatial_split",
    "sample_lfm_trace",
    "well_target_zone_mask",
]
