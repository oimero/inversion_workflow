"""Patch data seam for the GINN V2 body inverter.

The public interface in this module is deliberately expressed in array indices
and :class:`~cup.seismic.geometry.SurveyLineGeometry` coordinates separately.
An array index addresses a trace; a line number is only converted at the
geometry seam.  This keeps a survey whose xline step is four identical to one
whose xline step is one while all lateral features remain in physical metres.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry


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
            if not torch.is_floating_point(value) or not bool(torch.all(torch.isfinite(value)).item()):
                raise ValueError(f"PatchBatch {name} must be finite floating tensors.")
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
        center_lfm = np.asarray(self.lfm_log_ai[center_index[0], center_index[1], :], dtype=np.float64)
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


__all__ = [
    "ArrayTraceSource",
    "InputNormalization",
    "Orientation",
    "PatchBatch",
    "PatchKey",
    "PatchReader",
    "PatchSample",
    "SurveyTraceSource",
    "TraceSource",
    "candidate_patch_keys",
    "fit_lfm_normalization",
]
