"""Independent one-dimensional data contracts for the PIAI v3 workflow.

The v3 experiment intentionally has a small data seam.  A trace source knows
how to read one raw trace, while :class:`TraceReader` owns the fixed feature
normalization and the conversion to the tensors consumed by the network.  No
patch geometry, smoothing, low-frequency projection, or v2 data contracts
belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry
from cup.well.controls import WellControl, WellControlSet
from cup.well.evaluation_support import EvaluationSupport
from ginn_v3.config import TrainingConfig
from ginn_v3.types import (
    Normalization,
    ObservationBatch,
    TraceKey,
    WellBatch,
    WellTarget,
)


class TraceSource(Protocol):
    """Minimal raw trace source used by the independent v3 reader."""

    sample_axis: SampleAxis
    geometry: SurveyLineGeometry

    def read_traces(
        self, indices: Iterable[tuple[int, int]]
    ) -> dict[tuple[int, int], np.ndarray]: ...


def _as_key(value: TraceKey | tuple[int, int]) -> TraceKey:
    if isinstance(value, TraceKey):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("Trace keys must be TraceKey values or (inline_index, xline_index) tuples.")
    return TraceKey(int(value[0]), int(value[1]))


def _key_tuple(value: TraceKey | tuple[int, int]) -> tuple[int, int]:
    key = _as_key(value)
    return int(key.inline_index), int(key.xline_index)


def _ordered_keys(values: Iterable[TraceKey | tuple[int, int]]) -> tuple[TraceKey, ...]:
    result = tuple(_as_key(value) for value in values)
    if not result:
        raise ValueError("At least one trace key is required.")
    return result


@dataclass(frozen=True)
class ArrayTraceSource:
    """In-memory ``[inline_index, xline_index, sample]`` trace source."""

    volume: np.ndarray
    sample_axis: SampleAxis
    geometry: SurveyLineGeometry

    def __post_init__(self) -> None:
        volume = np.asarray(self.volume)
        if volume.ndim != 3 or not np.issubdtype(volume.dtype, np.floating):
            raise ValueError("ArrayTraceSource volume must be a floating three-dimensional array.")
        expected = (
            int(self.geometry.inline_axis.count),
            int(self.geometry.xline_axis.count),
            int(self.sample_axis.values.size),
        )
        if volume.shape != expected:
            raise ValueError(f"ArrayTraceSource volume shape {volume.shape} differs from {expected}.")
        object.__setattr__(self, "volume", volume)

    def read_traces(
        self, indices: Iterable[tuple[int, int]]
    ) -> dict[tuple[int, int], np.ndarray]:
        result: dict[tuple[int, int], np.ndarray] = {}
        for raw_key in indices:
            if not isinstance(raw_key, tuple) or len(raw_key) != 2:
                raise TypeError("ArrayTraceSource indices must be (inline_index, xline_index) tuples.")
            i, j = raw_key
            if isinstance(i, bool) or isinstance(j, bool) or int(i) != i or int(j) != j:
                raise TypeError("ArrayTraceSource indices must be integer array indices.")
            key = (int(i), int(j))
            if not (0 <= key[0] < self.volume.shape[0] and 0 <= key[1] < self.volume.shape[1]):
                raise ValueError(f"Trace index is outside ArrayTraceSource: {key}")
            result[key] = np.asarray(self.volume[key[0], key[1], :]).copy()
        return result


@dataclass(frozen=True)
class SurveyTraceSource:
    """Adapter around the existing SEG-Y/ZGY survey reader."""

    survey: object
    sample_axis: SampleAxis
    geometry: SurveyLineGeometry

    def read_traces(
        self, indices: Iterable[tuple[int, int]]
    ) -> dict[tuple[int, int], np.ndarray]:
        requested = tuple(dict.fromkeys(_key_tuple(value) for value in indices))
        if not requested:
            return {}
        reader = getattr(self.survey, "read_traces_at_indices", None)
        if reader is None:
            raise TypeError("SurveyTraceSource.survey must expose read_traces_at_indices().")
        traces = reader(list(requested), domain=self.sample_axis.domain)
        result: dict[tuple[int, int], np.ndarray] = {}
        for key in requested:
            if key not in traces:
                raise ValueError(f"Survey trace source did not return requested trace: {key}")
            item = traces[key]
            basis = getattr(item, "basis", None)
            if basis is not None and not np.array_equal(np.asarray(basis, dtype=np.float64), self.sample_axis.values):
                raise ValueError("Survey trace SampleAxis differs from the common v3 SampleAxis.")
            values = np.asarray(getattr(item, "values", item))
            if values.shape != self.sample_axis.values.shape:
                raise ValueError(f"Survey trace {key} has an unexpected sample shape: {values.shape}")
            if not np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32)
            result[key] = values.copy()
        return result


class TraceReader:
    """Read normalized two-channel traces for the independent v3 network."""

    def __init__(
        self,
        source: TraceSource,
        lfm_log_ai: np.ndarray,
        lfm_mask: np.ndarray,
        normalization: Normalization,
        velocity_mps: np.ndarray | None = None,
    ) -> None:
        self.source = source
        self.sample_axis = source.sample_axis
        self.geometry = source.geometry
        expected = (
            int(self.geometry.inline_axis.count),
            int(self.geometry.xline_axis.count),
            int(self.sample_axis.values.size),
        )
        lfm = np.asarray(lfm_log_ai)
        mask = np.asarray(lfm_mask, dtype=bool)
        if lfm.shape != expected or mask.shape != expected:
            raise ValueError(f"LFM arrays must both have shape {expected}.")
        if not np.issubdtype(lfm.dtype, np.floating):
            raise TypeError("lfm_log_ai must have a floating dtype.")
        finite = np.isfinite(lfm)
        if np.any(mask & ~finite) or np.any(finite & ~mask):
            raise ValueError("lfm_mask must exactly describe finite LFM support.")
        self.lfm_log_ai = lfm
        self.lfm_mask = mask
        if not isinstance(normalization, Normalization):
            raise TypeError("normalization must be ginn_v3.types.Normalization.")
        self.normalization = normalization
        if velocity_mps is None:
            self.velocity_mps = None
        else:
            velocity = np.asarray(velocity_mps)
            if velocity.shape != expected:
                raise ValueError(f"velocity_mps must have shape {expected}.")
            if not np.issubdtype(velocity.dtype, np.floating):
                raise TypeError("velocity_mps must have a floating dtype.")
            # A velocity volume derived from the LFM is allowed to carry NaN
            # outside LFM support.  The acoustic operator validates velocity
            # only where a supported trace is actually forwarded.
            if np.any(self.lfm_mask & (~np.isfinite(velocity) | (velocity <= 0.0))):
                raise ValueError("velocity_mps must be finite and positive on LFM support.")
            self.velocity_mps = velocity
        self.shape = expected

    def _validate_trace(self, key: TraceKey, raw: Mapping[tuple[int, int], np.ndarray]) -> np.ndarray:
        tuple_key = (key.inline_index, key.xline_index)
        if tuple_key not in raw:
            raise ValueError(f"TraceSource did not return requested trace {tuple_key}.")
        values = np.asarray(raw[tuple_key])
        if values.shape != self.sample_axis.values.shape:
            raise ValueError(f"Trace {tuple_key} does not match the common SampleAxis shape.")
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32)
        return values.astype(np.float32, copy=True)

    def _weighted_trace(
        self,
        rows: Sequence[tuple[TraceKey, float]],
        raw: Mapping[tuple[int, int], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Interpolate one raw trace, LFM trace, and optional velocity.

        Every quantity uses its own finite contributor denominator.  This is
        the same spatial interpolation rule used for the target support, so a
        missing nearest cell cannot erase a valid off-grid well sample.
        """

        n = int(self.sample_axis.values.size)
        seismic_sum = np.zeros(n, dtype=np.float64)
        seismic_weight = np.zeros(n, dtype=np.float64)
        lfm_sum = np.zeros(n, dtype=np.float64)
        lfm_weight = np.zeros(n, dtype=np.float64)
        velocity_sum = np.zeros(n, dtype=np.float64) if self.velocity_mps is not None else None
        velocity_weight = np.zeros(n, dtype=np.float64) if self.velocity_mps is not None else None
        for key, weight in rows:
            coefficient = float(weight)
            if coefficient <= 0.0 or not np.isfinite(coefficient):
                raise ValueError("Trace interpolation weights must be finite and positive.")
            observed = self._validate_trace(key, raw)
            finite_observed = np.isfinite(observed)
            seismic_sum[finite_observed] += coefficient * observed[finite_observed]
            seismic_weight[finite_observed] += coefficient
            lfm = np.asarray(self.lfm_log_ai[key.inline_index, key.xline_index, :], dtype=np.float64)
            lfm_valid = np.asarray(self.lfm_mask[key.inline_index, key.xline_index, :], dtype=bool) & np.isfinite(lfm)
            lfm_sum[lfm_valid] += coefficient * lfm[lfm_valid]
            lfm_weight[lfm_valid] += coefficient
            if velocity_sum is not None and velocity_weight is not None:
                velocity = np.asarray(self.velocity_mps[key.inline_index, key.xline_index, :], dtype=np.float64)
                finite_velocity = np.isfinite(velocity) & (velocity > 0.0)
                velocity_sum[finite_velocity] += coefficient * velocity[finite_velocity]
                velocity_weight[finite_velocity] += coefficient
        observed_mask = seismic_weight > 0.0
        observed = np.zeros(n, dtype=np.float32)
        observed[observed_mask] = (seismic_sum[observed_mask] / seismic_weight[observed_mask]).astype(np.float32)
        lfm_mask = lfm_weight > 0.0
        lfm = np.full(n, np.nan, dtype=np.float64)
        lfm[lfm_mask] = lfm_sum[lfm_mask] / lfm_weight[lfm_mask]
        velocity_result: np.ndarray | None = None
        if velocity_sum is not None and velocity_weight is not None:
            velocity_result = np.full(n, np.nan, dtype=np.float32)
            finite_velocity = velocity_weight > 0.0
            velocity_result[finite_velocity] = (velocity_sum[finite_velocity] / velocity_weight[finite_velocity]).astype(np.float32)
        return observed, observed_mask, lfm, velocity_result

    def weighted_batch(
        self,
        trace_weights: Iterable[Iterable[tuple[TraceKey | tuple[int, int], float]]],
        device: str | torch.device = "cpu",
        *,
        nominal_keys: Iterable[TraceKey | tuple[int, int]] | None = None,
    ) -> ObservationBatch:
        """Build one trace per weighted XY footprint with one source read.

        ``trace_weights`` contains one finite positive footprint per output
        trace.  ``nominal_keys`` labels those output traces for diagnostics;
        the numerical values come from the weighted footprint.
        """

        weight_rows: list[tuple[tuple[TraceKey, float], ...]] = []
        for row in trace_weights:
            normalized = tuple((_as_key(key), float(weight)) for key, weight in row)
            if not normalized:
                raise ValueError("Each weighted trace row must contain at least one source trace.")
            if len({key for key, _weight in normalized}) != len(normalized):
                raise ValueError("A weighted trace row must not repeat a source trace key.")
            total = float(sum(weight for _key, weight in normalized))
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError("Weighted trace rows must have positive finite total weight.")
            weight_rows.append(tuple((key, weight / total) for key, weight in normalized))
        if not weight_rows:
            raise ValueError("At least one weighted trace row is required.")
        if nominal_keys is None:
            selected = tuple(max(row, key=lambda item: item[1])[0] for row in weight_rows)
        else:
            selected = tuple(_as_key(key) for key in nominal_keys)
            if not selected:
                raise ValueError("nominal_keys must not be empty.")
            if len(selected) != len(weight_rows):
                raise ValueError("nominal_keys and weighted trace rows must have equal lengths.")
        source_keys = tuple(dict.fromkeys(key for row in weight_rows for key, _weight in row))
        raw = self.source.read_traces([(key.inline_index, key.xline_index) for key in source_keys])
        n = int(self.sample_axis.values.size)
        feature_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        initial_rows: list[np.ndarray] = []
        lfm_masks: list[np.ndarray] = []
        observed_masks: list[np.ndarray] = []
        velocity_rows: list[np.ndarray] = []
        for row in weight_rows:
            observed, observed_mask, lfm, velocity = self._weighted_trace(row, raw)
            lfm_mask = np.isfinite(lfm)
            lfm_feature = np.zeros(n, dtype=np.float32)
            lfm_feature[lfm_mask] = (
                (lfm[lfm_mask] - self.normalization.lfm_mean) / self.normalization.lfm_std
            ).astype(np.float32)
            observed_feature = np.zeros(n, dtype=np.float32)
            observed_feature[observed_mask] = (
                (observed[observed_mask].astype(np.float64) - self.normalization.seismic_mean)
                / self.normalization.seismic_std
            ).astype(np.float32)
            initial = np.zeros(n, dtype=np.float32)
            initial[lfm_mask] = lfm[lfm_mask].astype(np.float32)
            feature_rows.append(np.stack((observed_feature, lfm_feature), axis=0))
            raw_rows.append(observed)
            initial_rows.append(initial)
            lfm_masks.append(lfm_mask)
            observed_masks.append(observed_mask)
            if self.velocity_mps is not None:
                velocity_rows.append(
                    np.zeros(n, dtype=np.float32) if velocity is None else velocity
                )
        tensor = lambda rows, dtype=None: torch.as_tensor(np.stack(rows, axis=0), dtype=dtype, device=device)
        return ObservationBatch(
            keys=selected,
            sample_axis=self.sample_axis,
            features=tensor(feature_rows, torch.float32),
            initial_log_ai=tensor(initial_rows, torch.float32),
            lfm_mask=tensor(lfm_masks, torch.bool),
            observed_seismic=tensor(raw_rows, torch.float32),
            observed_mask=tensor(observed_masks, torch.bool),
            velocity_mps=(tensor(velocity_rows, torch.float32) if velocity_rows else None),
        )

    def batch(
        self,
        keys: Iterable[TraceKey | tuple[int, int]],
        device: str | torch.device = "cpu",
    ) -> ObservationBatch:
        selected = _ordered_keys(keys)
        return self.weighted_batch(
            (((key, 1.0),) for key in selected),
            device=device,
            nominal_keys=selected,
        )


@dataclass(frozen=True)
class TrainingData:
    """Prepared keys, trace reader, and raw well targets for one v3 run."""

    reader: TraceReader
    train_keys: tuple[TraceKey, ...]
    validation_keys: tuple[TraceKey, ...]
    train_wells: tuple[WellTarget, ...]
    evaluation_wells: tuple[WellTarget, ...]

    def batch(
        self,
        keys: Iterable[TraceKey | tuple[int, int]],
        device: str | torch.device = "cpu",
    ) -> ObservationBatch:
        """Forward a trace batch through the frozen reader contract."""

        return self.reader.batch(keys, device=device)

    def well_batch(
        self,
        wells: Iterable[WellTarget | str],
        device: str | torch.device = "cpu",
    ) -> WellBatch:
        by_name = {target.well_name.casefold(): target for target in self.evaluation_wells}
        selected: list[WellTarget] = []
        for item in wells:
            if isinstance(item, WellTarget):
                target = item
            else:
                target = by_name.get(str(item).casefold())
                if target is None:
                    raise ValueError(f"Unknown v3 well target: {item!r}")
            selected.append(target)
        if not selected:
            raise ValueError("well_batch requires at least one well target.")
        trace_rows = tuple(
            target.trace_weights if target.trace_weights else ((target.key, 1.0),)
            for target in selected
        )
        observations = self.reader.weighted_batch(
            trace_rows,
            device=device,
            nominal_keys=(target.key for target in selected),
        )
        values = np.stack(
            [np.where(target.valid_mask, np.asarray(target.log_ai, dtype=np.float32), 0.0) for target in selected],
            axis=0,
        )
        masks = np.stack([np.asarray(target.valid_mask, dtype=bool) for target in selected], axis=0)
        target_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=device)
        return WellBatch(
            observations=observations,
            target_log_ai=target_tensor,
            target_mask=mask_tensor,
            well_names=tuple(target.well_name for target in selected),
        )


def _finite_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.r_[False, values, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    return tuple((int(start), int(stop)) for start, stop in changes if stop > start)


def _expected_shape(source: TraceSource) -> tuple[int, int, int]:
    return (
        int(source.geometry.inline_axis.count),
        int(source.geometry.xline_axis.count),
        int(source.sample_axis.values.size),
    )


def _source_lfm(
    source_or_reader: TraceSource | TraceReader,
    lfm_log_ai: np.ndarray | None,
    lfm_mask: np.ndarray | None,
) -> tuple[TraceSource, np.ndarray, np.ndarray]:
    if isinstance(source_or_reader, TraceReader):
        source = source_or_reader.source
        values = source_or_reader.lfm_log_ai if lfm_log_ai is None else np.asarray(lfm_log_ai)
        mask = source_or_reader.lfm_mask if lfm_mask is None else np.asarray(lfm_mask, dtype=bool)
    else:
        source = source_or_reader
        if lfm_log_ai is None or lfm_mask is None:
            raise ValueError("lfm_log_ai and lfm_mask are required with a raw TraceSource.")
        values = np.asarray(lfm_log_ai)
        mask = np.asarray(lfm_mask, dtype=bool)
    expected = _expected_shape(source)
    if values.shape != expected or mask.shape != expected:
        raise ValueError(f"LFM arrays must both have shape {expected}.")
    if np.any(mask & ~np.isfinite(values)) or np.any(np.isfinite(values) & ~mask):
        raise ValueError("lfm_mask must exactly describe finite LFM support.")
    return source, values, mask


def _lfm_candidate_keys(
    source: TraceSource,
    lfm_log_ai: np.ndarray,
    lfm_mask: np.ndarray,
    *,
    min_support_samples: int,
) -> tuple[TraceKey, ...]:
    """Build candidate centres from the cheap volume-side LFM support scan."""

    minimum = int(min_support_samples)
    support_counts = np.count_nonzero(
        np.asarray(lfm_mask, dtype=bool) & np.isfinite(np.asarray(lfm_log_ai)), axis=2
    )
    selected = np.argwhere(support_counts >= minimum)
    return tuple(TraceKey(int(i), int(j)) for i, j in selected.tolist())


def candidate_trace_keys(
    source_or_reader: TraceSource | TraceReader,
    lfm_log_ai: np.ndarray | None = None,
    lfm_mask: np.ndarray | None = None,
    *,
    min_support_samples: int = 8,
) -> tuple[TraceKey, ...]:
    """Return deterministic usable traces with enough LFM and seismic support."""

    minimum = int(min_support_samples)
    if isinstance(min_support_samples, bool) or minimum < 2:
        raise ValueError("min_support_samples must be an integer of at least two.")
    source, lfm, mask = _source_lfm(source_or_reader, lfm_log_ai, lfm_mask)
    keys = [
        (key.inline_index, key.xline_index)
        for key in _lfm_candidate_keys(
            source, lfm, mask, min_support_samples=minimum
        )
    ]
    if not keys:
        raise ValueError("No trace has enough finite LFM support for v3 candidate selection.")
    result: list[TraceKey] = []
    for start in range(0, len(keys), 256):
        chunk = keys[start : start + 256]
        traces = source.read_traces(chunk)
        for key in chunk:
            values = np.asarray(traces[key])
            finite = np.isfinite(values)
            if np.count_nonzero(finite) < minimum:
                continue
            finite_values = values[finite].astype(np.float64)
            if not finite_values.size or not np.any(np.abs(finite_values) > 0.0):
                continue
            if not np.isfinite(np.std(finite_values)) or float(np.std(finite_values)) <= 0.0:
                continue
            result.append(TraceKey(*key))
    if not result:
        raise ValueError("No usable v3 trace has enough finite LFM and seismic support.")
    return tuple(result)


def _key_xy(key: TraceKey, geometry: SurveyLineGeometry) -> np.ndarray:
    inline = geometry.inline_axis.line_at_index(key.inline_index)
    xline = geometry.xline_axis.line_at_index(key.xline_index)
    return np.asarray(geometry.line_to_coord(inline, xline), dtype=np.float64)


def _bilinear_trace_weights(
    x_m: float,
    y_m: float,
    geometry: SurveyLineGeometry,
) -> tuple[TraceKey, tuple[tuple[TraceKey, float], ...]]:
    """Return the nearest nominal key and its fixed four-cell XY footprint."""

    i_float, j_float = geometry.coord_to_index(float(x_m), float(y_m))
    n_inline = int(geometry.inline_axis.count)
    n_xline = int(geometry.xline_axis.count)
    i0 = min(n_inline - 1, max(0, int(np.floor(i_float))))
    j0 = min(n_xline - 1, max(0, int(np.floor(j_float))))
    i1 = min(n_inline - 1, i0 + 1)
    j1 = min(n_xline - 1, j0 + 1)
    fi = 0.0 if i1 == i0 else float(i_float - i0)
    fj = 0.0 if j1 == j0 else float(j_float - j0)
    candidates = (
        (TraceKey(i0, j0), (1.0 - fi) * (1.0 - fj)),
        (TraceKey(i0, j1), (1.0 - fi) * fj),
        (TraceKey(i1, j0), fi * (1.0 - fj)),
        (TraceKey(i1, j1), fi * fj),
    )
    combined: dict[TraceKey, float] = {}
    for key, weight in candidates:
        if weight > 0.0:
            combined[key] = combined.get(key, 0.0) + float(weight)
    weights = tuple(sorted(combined.items()))
    if not weights or not np.isclose(sum(weight for _key, weight in weights), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("Bilinear trace footprint does not have unit finite weight.")
    nearest = TraceKey(
        min(n_inline - 1, max(0, int(round(i_float)))),
        min(n_xline - 1, max(0, int(round(j_float)))),
    )
    return nearest, weights


def _weighted_volume_trace(
    volume: np.ndarray,
    valid_mask: np.ndarray,
    weights: Sequence[tuple[TraceKey, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-aware weighted sampling of one volume trace."""

    n = int(volume.shape[2])
    numerator = np.zeros(n, dtype=np.float64)
    denominator = np.zeros(n, dtype=np.float64)
    for key, weight in weights:
        values = np.asarray(volume[key.inline_index, key.xline_index, :], dtype=np.float64)
        valid = np.asarray(valid_mask[key.inline_index, key.xline_index, :], dtype=bool) & np.isfinite(values)
        coefficient = float(weight)
        numerator[valid] += coefficient * values[valid]
        denominator[valid] += coefficient
    output = np.full(n, np.nan, dtype=np.float64)
    supported = denominator > 0.0
    output[supported] = numerator[supported] / denominator[supported]
    return output, supported


def _keys_xy(keys: Sequence[TraceKey], geometry: SurveyLineGeometry) -> np.ndarray:
    """Vectorized physical coordinates for array-index trace keys."""

    inline = np.asarray([key.inline_index for key in keys], dtype=np.float64)
    xline = np.asarray([key.xline_index for key in keys], dtype=np.float64)
    return np.column_stack(
        (
            float(geometry.x0) + inline * float(geometry.dx_inline) + xline * float(geometry.dx_xline),
            float(geometry.y0) + inline * float(geometry.dy_inline) + xline * float(geometry.dy_xline),
        )
    )


def _config_value(config: TrainingConfig | None, name: str, default: int | float) -> int | float:
    if config is None:
        return default
    return getattr(config, name)


def sample_trace_split(
    keys: Iterable[TraceKey | tuple[int, int]],
    *,
    geometry: SurveyLineGeometry,
    config: TrainingConfig | None = None,
    validation_traces: int | None = None,
    max_train_traces: int | None = None,
    validation_gap_m: float | None = None,
    seed: int | None = None,
) -> tuple[tuple[TraceKey, ...], tuple[TraceKey, ...]]:
    """Select deterministic train/validation traces with a metre gap.

    The validation block is selected from the high end of a deterministic
    physical-coordinate ordering.  If the requested block leaves no training
    trace after the configured metre gap, the block is reduced until a valid
    disjoint split exists.
    """

    candidates = tuple(sorted({_as_key(value) for value in keys}))
    if len(candidates) < 2:
        raise ValueError("At least two candidate traces are required for a train/validation split.")
    requested_validation = int(
        _config_value(config, "validation_traces", 128) if validation_traces is None else validation_traces
    )
    maximum_train = int(
        _config_value(config, "max_train_traces", 4096) if max_train_traces is None else max_train_traces
    )
    gap = float(
        _config_value(config, "validation_gap_m", 300.0) if validation_gap_m is None else validation_gap_m
    )
    if requested_validation <= 0 or maximum_train <= 0 or not np.isfinite(gap) or gap < 0.0:
        raise ValueError("Split counts must be positive and validation_gap_m must be finite and nonnegative.")
    # The seed is accepted as part of the public deterministic contract.  The
    # physical ordering is intentionally independent of it; this keeps runs
    # reproducible across NumPy versions and data-reader implementations.
    _ = int(_config_value(config, "seed", 0) if seed is None else seed)
    candidate_xy = _keys_xy(candidates, geometry)
    order = np.lexsort(
        (
            np.asarray([key.xline_index for key in candidates], dtype=np.int64),
            np.asarray([key.inline_index for key in candidates], dtype=np.int64),
            candidate_xy[:, 1],
            candidate_xy[:, 0],
        )
    )
    ordered = tuple(candidates[int(index)] for index in order)
    ordered_xy = candidate_xy[order]
    for validation_count in range(min(requested_validation, len(ordered) - 1), 0, -1):
        validation = ordered[-validation_count:]
        validation_xy = ordered_xy[-validation_count:]
        train_keys_without_gap = ordered[:-validation_count]
        train_xy = ordered_xy[:-validation_count]
        eligible: list[bool] = []
        for start in range(0, train_xy.shape[0], 65536):
            chunk_xy = train_xy[start : start + 65536]
            distances = np.linalg.norm(chunk_xy[:, None, :] - validation_xy[None, :, :], axis=2)
            eligible.extend((np.min(distances, axis=1) > gap).tolist())
        train_candidates = tuple(key for key, keep in zip(train_keys_without_gap, eligible) if keep)
        if not train_candidates:
            continue
        if len(train_candidates) > maximum_train:
            selection = np.linspace(0, len(train_candidates) - 1, maximum_train, dtype=np.int64)
            train_candidates = tuple(train_candidates[int(index)] for index in selection)
        return train_candidates, tuple(validation)
    raise ValueError("validation_gap_m leaves no disjoint training trace.")


def _support_mask(support: EvaluationSupport, sample_axis: SampleAxis) -> np.ndarray:
    validator = getattr(support, "validate_axis", None)
    if callable(validator):
        validator(sample_axis)
    direct = getattr(support, "mask", None)
    if direct is not None:
        mask = np.asarray(direct, dtype=bool)
    else:
        mask = np.zeros(sample_axis.values.shape, dtype=bool)
        indices = np.asarray(getattr(support, "indices"), dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= mask.size):
            raise ValueError(f"{support.well_name}: evaluation support indices are outside the SampleAxis.")
        mask[indices] = True
    if mask.shape != sample_axis.values.shape:
        raise ValueError(f"{support.well_name}: evaluation support mask differs from the SampleAxis.")
    return mask


def _control_trace_weights(
    control: WellControl,
    geometry: SurveyLineGeometry,
) -> tuple[TraceKey, tuple[tuple[TraceKey, float], ...]]:
    if str(control.wellbore_class).strip().casefold() != "vertical":
        raise ValueError(f"{control.well_name}: v3 currently accepts only vertical well controls.")
    valid = np.asarray(control.valid_mask, dtype=bool)
    if not np.any(valid):
        raise ValueError(f"{control.well_name}: well control has no valid positions.")
    x_all = np.asarray(control.x_m_by_sample, dtype=np.float64)
    y_all = np.asarray(control.y_m_by_sample, dtype=np.float64)
    finite_xy = np.isfinite(x_all) & np.isfinite(y_all)
    if np.any(valid & ~finite_xy):
        raise ValueError(f"{control.well_name}: a valid well position is not finite.")
    coordinates = np.flatnonzero(
        finite_xy
    )
    if not coordinates.size:
        raise ValueError(f"{control.well_name}: well control has no finite XY positions.")
    first = int(coordinates[0])
    x_reference = float(control.x_m_by_sample[first])
    y_reference = float(control.y_m_by_sample[first])
    x_values = x_all[coordinates]
    y_values = y_all[coordinates]
    if not np.allclose(x_values, x_reference, rtol=0.0, atol=1e-6) or not np.allclose(
        y_values, y_reference, rtol=0.0, atol=1e-6
    ):
        raise ValueError(f"{control.well_name}: v3 requires a straight vertical XY footprint.")
    nearest, weights = _bilinear_trace_weights(x_reference, y_reference, geometry)
    return nearest, weights


def _nearest_control_trace(control: WellControl, geometry: SurveyLineGeometry) -> TraceKey:
    """Return the nominal nearest cell retained in the well metadata."""

    nearest, _weights = _control_trace_weights(control, geometry)
    return nearest


def _resample_native_runs(control: WellControl) -> np.ndarray:
    axis = np.asarray(control.sample_axis.values, dtype=np.float64)
    native_axis = np.asarray(control.native.coordinates, dtype=np.float64)
    native_values = np.asarray(control.native.native_filtered_log_ai, dtype=np.float64)
    finite = np.isfinite(native_values)
    output = np.full(axis.shape, np.nan, dtype=np.float64)
    for start, stop in _finite_runs(finite):
        source_axis = native_axis[start:stop]
        source_values = native_values[start:stop]
        if source_axis.size == 1:
            exact = np.isclose(axis, source_axis[0], rtol=0.0, atol=1e-10)
            output[exact] = source_values[0]
            continue
        inside = (axis >= source_axis[0]) & (axis <= source_axis[-1])
        output[inside] = np.interp(axis[inside], source_axis, source_values)
    return output


def build_well_targets(
    controls: WellControlSet,
    *,
    geometry: SurveyLineGeometry,
    lfm_log_ai: np.ndarray,
    lfm_mask: np.ndarray,
    evaluation_supports: Mapping[str, EvaluationSupport],
) -> dict[str, WellTarget]:
    """Build raw native targets for every straight well and fixed QC support."""

    values = np.asarray(lfm_log_ai)
    masks = np.asarray(lfm_mask, dtype=bool)
    expected = (
        int(geometry.inline_axis.count),
        int(geometry.xline_axis.count),
        int(controls.sample_axis.values.size),
    )
    if values.shape != expected or masks.shape != expected:
        raise ValueError(f"lfm_log_ai and lfm_mask must both have shape {expected}.")
    if np.any(masks & ~np.isfinite(values)) or np.any(np.isfinite(values) & ~masks):
        raise ValueError("lfm_mask must exactly describe finite LFM support.")
    support_by_name = {str(name).casefold(): support for name, support in evaluation_supports.items()}
    targets: dict[str, WellTarget] = {}
    for control in controls.controls:
        if str(control.wellbore_class).strip().casefold() == "deviated":
            continue
        key, trace_weights = _control_trace_weights(control, geometry)
        support = support_by_name.get(control.well_name.casefold())
        if support is None:
            raise ValueError(f"Missing Step 6 evaluation support for {control.well_name}.")
        if str(getattr(support, "well_name", control.well_name)).casefold() != control.well_name.casefold():
            raise ValueError(f"{control.well_name}: evaluation support well name differs from the control.")
        evaluation_mask = _support_mask(support, controls.sample_axis)
        target = _resample_native_runs(control)
        model_valid = np.asarray(control.valid_mask, dtype=bool)
        observed_valid = np.asarray(control.observed_valid_mask, dtype=bool)
        _lfm_trace, lfm_valid = _weighted_volume_trace(values, masks, trace_weights)
        valid = np.isfinite(target) & model_valid & observed_valid & lfm_valid
        if np.any(evaluation_mask & ~valid):
            missing = int(np.count_nonzero(evaluation_mask & ~valid))
            raise ValueError(
                f"{control.well_name}: Step 6 evaluation support has {missing} unsupported samples; "
                "the fixed support cannot be shrunk for v3."
            )
        if np.count_nonzero(valid) < 2:
            raise ValueError(f"{control.well_name}: fewer than two usable v3 well target samples.")
        target[~valid] = np.nan
        targets[control.well_name] = WellTarget(
            well_name=control.well_name,
            key=key,
            log_ai=target,
            valid_mask=valid,
            evaluation_mask=evaluation_mask,
            trace_weights=trace_weights,
        )
    return targets


def _target_values(targets: Mapping[str, WellTarget] | Iterable[WellTarget]) -> tuple[WellTarget, ...]:
    if isinstance(targets, Mapping):
        return tuple(targets.values())
    return tuple(targets)


def fit_normalization(
    source_or_reader: TraceSource | TraceReader,
    train_keys: Iterable[TraceKey | tuple[int, int]],
    trusted_wells: Mapping[str, WellTarget] | Iterable[WellTarget],
    *,
    lfm_log_ai: np.ndarray | None = None,
    lfm_mask: np.ndarray | None = None,
) -> Normalization:
    """Fit one frozen global normalization from the specified training data."""

    source, lfm, mask = _source_lfm(source_or_reader, lfm_log_ai, lfm_mask)
    selected = tuple(_as_key(value) for value in train_keys)
    if not selected:
        raise ValueError("Cannot fit normalization from an empty training trace set.")
    seismic_parts: list[np.ndarray] = []
    lfm_parts: list[np.ndarray] = []
    for start in range(0, len(selected), 256):
        chunk = selected[start : start + 256]
        traces = source.read_traces([(key.inline_index, key.xline_index) for key in chunk])
        for key in chunk:
            seismic = np.asarray(traces[(key.inline_index, key.xline_index)], dtype=np.float64)
            seismic_parts.append(seismic[np.isfinite(seismic)])
            support = mask[key.inline_index, key.xline_index, :] & np.isfinite(lfm[key.inline_index, key.xline_index, :])
            lfm_parts.append(np.asarray(lfm[key.inline_index, key.xline_index, :], dtype=np.float64)[support])
    seismic_values = np.concatenate([part for part in seismic_parts if part.size])
    lfm_values = np.concatenate([part for part in lfm_parts if part.size])
    # The supervised loss is measured in linear AI after exponentiating the
    # log-AI curves.  Keep the normalization in that same physical quantity;
    # a log-domain standard deviation would rescale the trainer incorrectly.
    trusted_values = [
        np.exp(np.asarray(target.log_ai, dtype=np.float64)[np.asarray(target.valid_mask, dtype=bool)])
        for target in _target_values(trusted_wells)
    ]
    impedance_values = np.concatenate([part for part in trusted_values if part.size]) if trusted_values else np.empty(0)
    if not seismic_values.size or not lfm_values.size or not impedance_values.size:
        raise ValueError("Normalization requires finite seismic, LFM, and trusted impedance support.")

    def stats(values: np.ndarray, label: str) -> tuple[float, float]:
        mean = float(np.mean(values))
        std = float(np.sqrt(np.mean(np.square(values - mean))))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"{label} normalization support must have positive finite variance.")
        return mean, std

    seismic_mean, seismic_std = stats(seismic_values, "seismic")
    lfm_mean, lfm_std = stats(lfm_values, "LFM")
    impedance_mean, impedance_std = stats(impedance_values, "impedance")
    return Normalization(
        seismic_mean=seismic_mean,
        seismic_std=seismic_std,
        lfm_mean=lfm_mean,
        lfm_std=lfm_std,
        impedance_mean=impedance_mean,
        impedance_std=impedance_std,
    )


def prepare_training_data(
    source: TraceSource,
    lfm_log_ai: np.ndarray,
    lfm_mask: np.ndarray,
    controls: WellControlSet,
    evaluation_supports: Mapping[str, EvaluationSupport],
    trusted_well_names: Iterable[str],
    config: TrainingConfig,
    velocity_mps: np.ndarray | None = None,
    normalization: Normalization | None = None,
) -> TrainingData:
    """Prepare the complete independent v3 data contract."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be ginn_v3.config.TrainingConfig.")
    if not np.array_equal(source.sample_axis.values, controls.sample_axis.values):
        raise ValueError("TraceSource and WellControlSet SampleAxis values differ.")
    values = np.asarray(lfm_log_ai)
    masks = np.asarray(lfm_mask, dtype=bool)
    if values.shape != _expected_shape(source) or masks.shape != _expected_shape(source):
        raise ValueError(f"LFM arrays must both have shape {_expected_shape(source)}.")
    targets = build_well_targets(
        controls,
        geometry=source.geometry,
        lfm_log_ai=values,
        lfm_mask=masks,
        evaluation_supports=evaluation_supports,
    )
    trusted_names = tuple(str(name) for name in trusted_well_names)
    if not trusted_names:
        raise ValueError("At least one trusted well is required for v3 training.")
    trusted_keys = {name.casefold() for name in trusted_names}
    if len(trusted_keys) != len(trusted_names):
        raise ValueError("Trusted well names must be unique case-insensitively.")
    unknown = sorted(trusted_keys - {name.casefold() for name in targets})
    if unknown:
        raise ValueError(f"Trusted wells are unknown or invalid: {unknown}")
    # Select the physical split from the inexpensive LFM support volume first.
    # Only the selected traces are then read from SEG-Y/ZGY, which keeps data
    # preparation streaming for large surveys.  A selected trace with unusable
    # seismic support is an explicit input error; it is never silently replaced
    # by a nearby trace and the configured split is never re-fit around it.
    candidates = _lfm_candidate_keys(
        source,
        values,
        masks,
        min_support_samples=config.min_support_samples,
    )
    if not candidates:
        raise ValueError("No trace has enough finite LFM support for v3 candidate selection.")
    train_keys, validation_keys = sample_trace_split(candidates, geometry=source.geometry, config=config)
    selected_traces = source.read_traces(
        [(key.inline_index, key.xline_index) for key in (*train_keys, *validation_keys)]
    )
    for key in (*train_keys, *validation_keys):
        trace = np.asarray(selected_traces[(key.inline_index, key.xline_index)])
        finite = np.isfinite(trace)
        if np.count_nonzero(finite) < config.min_support_samples or float(np.std(trace[finite])) <= 0.0:
            raise ValueError(
                f"Selected trace {(key.inline_index, key.xline_index)} lacks usable finite seismic support."
            )
    trusted_targets = tuple(targets[name] for name in targets if name.casefold() in trusted_keys)
    frozen_normalization = normalization or fit_normalization(
        source,
        train_keys,
        trusted_targets,
        lfm_log_ai=values,
        lfm_mask=masks,
    )
    reader = TraceReader(
        source,
        values,
        masks,
        frozen_normalization,
        velocity_mps=velocity_mps,
    )
    return TrainingData(
        reader=reader,
        train_keys=tuple(train_keys),
        validation_keys=tuple(validation_keys),
        train_wells=trusted_targets,
        evaluation_wells=tuple(targets.values()),
    )


__all__ = [
    "ArrayTraceSource",
    "SurveyTraceSource",
    "TraceReader",
    "TraceSource",
    "TrainingData",
    "build_well_targets",
    "candidate_trace_keys",
    "fit_normalization",
    "prepare_training_data",
    "sample_trace_split",
]
