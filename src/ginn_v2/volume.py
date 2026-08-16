"""Full-volume scheduling and deterministic orientation fusion for GINN V2."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt

from ginn_v2.data import Orientation, PatchKey
from ginn_v2.inverter import BodyInverter


@dataclass(frozen=True)
class VolumeInferenceConfig:
    """Execution settings that do not alter the trained network contract."""

    batch_size: int = 32
    log_every_sections: int = 10
    min_lfm_support: int = 8
    orientations: tuple[Orientation, ...] = ("inline", "xline")

    def __post_init__(self) -> None:
        for name in ("batch_size", "log_every_sections", "min_lfm_support"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.orientations or len(set(self.orientations)) != len(self.orientations):
            raise ValueError("orientations must contain unique inline/xline values.")
        if any(value not in {"inline", "xline"} for value in self.orientations):
            raise ValueError("orientations must contain inline and/or xline.")


@dataclass(frozen=True)
class BodyVolumeResult:
    """Fused body volume plus direct-prediction and fill provenance.

    ``fill_code`` values are 0 outside the target zone, 1 for a direct network
    prediction, 2 for a nearest spatial increment fill, and 3 for an LFM-only
    fill used when an entire target depth slice has no direct prediction.
    """

    body_log_ai: np.ndarray
    direction_count: np.ndarray
    direction_disagreement_log_ai: np.ndarray
    fill_code: np.ndarray
    inline_indices: np.ndarray
    xline_indices: np.ndarray
    nearest_fill_mean_distance_m: float
    nearest_fill_max_distance_m: float

    def __post_init__(self) -> None:
        body = np.asarray(self.body_log_ai)
        count = np.asarray(self.direction_count)
        disagreement = np.asarray(self.direction_disagreement_log_ai)
        fill_code = np.asarray(self.fill_code)
        if (
            body.ndim != 3
            or count.shape != body.shape
            or disagreement.shape != body.shape
            or fill_code.shape != body.shape
        ):
            raise ValueError("BodyVolumeResult arrays must share [inline, xline, sample] shape.")
        if count.dtype != np.uint8 or np.any(count > 2):
            raise ValueError("direction_count must be uint8 with values in [0, 2].")
        if fill_code.dtype != np.uint8 or np.any(fill_code > 3):
            raise ValueError("fill_code must be uint8 with values in [0, 3].")
        if body.shape[:2] != (self.inline_indices.size, self.xline_indices.size):
            raise ValueError("BodyVolumeResult spatial axes differ from its array shape.")
        if np.any(np.isfinite(body) != (fill_code > 0)):
            raise ValueError("body_log_ai must be finite at every target-zone sample and only there.")
        if np.any((fill_code == 1) != (count > 0)):
            raise ValueError("fill_code=1 must identify exactly the direct network predictions.")
        if np.any(np.isfinite(disagreement) != (count == 2)):
            raise ValueError("direction disagreement finite support must equal direction_count == 2.")
        for name in ("nearest_fill_mean_distance_m", "nearest_fill_max_distance_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.nearest_fill_mean_distance_m > self.nearest_fill_max_distance_m:
            raise ValueError("nearest-fill mean distance cannot exceed its maximum distance.")

    def summary(self) -> dict[str, float | int | list[int]]:
        count = np.asarray(self.direction_count)
        fill_code = np.asarray(self.fill_code)
        target = fill_code > 0
        target_count = int(np.count_nonzero(target))
        denominator = max(target_count, 1)
        disagreement = np.asarray(self.direction_disagreement_log_ai)
        paired = disagreement[count == 2]
        return {
            "shape": [int(value) for value in self.body_log_ai.shape],
            "sample_count": int(count.size),
            "target_sample_count": target_count,
            "target_nan_count": int(np.count_nonzero(target & ~np.isfinite(self.body_log_ai))),
            "direct_prediction_fraction": float(np.count_nonzero(fill_code == 1) / denominator),
            "nearest_increment_fill_fraction": float(np.count_nonzero(fill_code == 2) / denominator),
            "lfm_only_fill_fraction": float(np.count_nonzero(fill_code == 3) / denominator),
            "single_direction_fraction": float(np.count_nonzero(count == 1) / denominator),
            "two_direction_fraction": float(np.count_nonzero(count == 2) / denominator),
            "nearest_fill_mean_distance_m": float(self.nearest_fill_mean_distance_m),
            "nearest_fill_max_distance_m": float(self.nearest_fill_max_distance_m),
            "direction_disagreement_rms_log_ai": (
                float(np.sqrt(np.mean(np.square(paired, dtype=np.float64)))) if paired.size else 0.0
            ),
        }


def _index_bounds(size: int, bounds: tuple[int, int] | None, *, name: str) -> tuple[int, int]:
    if bounds is None:
        return 0, int(size)
    start, stop = (int(bounds[0]), int(bounds[1]))
    if not 0 <= start < stop <= size:
        raise ValueError(f"{name} bounds must satisfy 0 <= start < stop <= {size}.")
    return start, stop


def _fill_target_zone(
    body_log_ai: np.ndarray,
    direction_count: np.ndarray,
    lfm_log_ai: np.ndarray,
    target_mask: np.ndarray,
    *,
    inline_spacing_m: float,
    xline_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fill target-zone gaps while preserving the local LFM background.

    A missing sample receives the body increment from the physically nearest
    direct prediction at the same vertical sample. If that entire depth slice
    lacks a direct prediction, the local LFM is used (zero body increment).
    """

    body = np.asarray(body_log_ai, dtype=np.float32)
    count = np.asarray(direction_count, dtype=np.uint8)
    lfm = np.asarray(lfm_log_ai, dtype=np.float32)
    target = np.asarray(target_mask, dtype=bool)
    if body.shape != count.shape or body.shape != lfm.shape or body.shape != target.shape:
        raise ValueError("body, direction_count, LFM, and target mask shapes differ.")
    if not np.isfinite(inline_spacing_m) or inline_spacing_m <= 0.0:
        raise ValueError("inline_spacing_m must be finite and positive.")
    if not np.isfinite(xline_spacing_m) or xline_spacing_m <= 0.0:
        raise ValueError("xline_spacing_m must be finite and positive.")
    if np.any(target & ~np.isfinite(lfm)):
        raise ValueError("target-zone LFM must be finite before volume filling.")

    direct = target & (count > 0)
    if np.any(direct & ~np.isfinite(body)):
        raise ValueError("direct network prediction contains non-finite target samples.")
    fill_code = np.zeros(body.shape, dtype=np.uint8)
    fill_code[direct] = np.uint8(1)
    distance_sum = 0.0
    distance_count = 0
    distance_max = 0.0

    for sample_index in range(body.shape[-1]):
        local_target = target[:, :, sample_index]
        local_direct = direct[:, :, sample_index]
        missing = local_target & ~local_direct
        if not np.any(missing):
            continue
        if not np.any(local_direct):
            body[:, :, sample_index][missing] = lfm[:, :, sample_index][missing]
            fill_code[:, :, sample_index][missing] = np.uint8(3)
            continue

        distances, nearest = distance_transform_edt(
            ~local_direct,
            sampling=(float(inline_spacing_m), float(xline_spacing_m)),
            return_distances=True,
            return_indices=True,
        )
        query_i, query_j = np.nonzero(missing)
        source_i = nearest[0, query_i, query_j]
        source_j = nearest[1, query_i, query_j]
        local_increment = body[:, :, sample_index] - lfm[:, :, sample_index]
        body[query_i, query_j, sample_index] = (
            lfm[query_i, query_j, sample_index] + local_increment[source_i, source_j]
        )
        fill_code[query_i, query_j, sample_index] = np.uint8(2)
        local_distances = distances[query_i, query_j]
        distance_sum += float(np.sum(local_distances, dtype=np.float64))
        distance_count += int(local_distances.size)
        distance_max = max(distance_max, float(np.max(local_distances)))

    body[~target] = np.nan
    mean_distance = distance_sum / distance_count if distance_count else 0.0
    return body, fill_code, float(mean_distance), float(distance_max)


class BodyVolumeInverter:
    """Predict a spatial tile or full survey and fuse available directions once."""

    def __init__(
        self,
        inverter: BodyInverter,
        config: VolumeInferenceConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.inverter = inverter
        self.config = config
        self.log = logger or logging.getLogger(__name__)
        if inverter.batch_size != config.batch_size:
            raise ValueError("BodyInverter and VolumeInferenceConfig batch sizes differ.")
        reader = inverter.reader
        self._lfm_support = np.count_nonzero(
            reader.lfm_valid_mask & np.isfinite(reader.lfm_log_ai),
            axis=-1,
        )

    def _section_keys(
        self,
        orientation: Orientation,
        fixed_index: int,
        *,
        inline_bounds: tuple[int, int],
        xline_bounds: tuple[int, int],
    ) -> tuple[PatchKey, ...]:
        reader = self.inverter.reader
        radius = reader.patch_radius
        keys: list[PatchKey] = []
        if orientation == "inline":
            start = max(xline_bounds[0], radius)
            stop = min(xline_bounds[1], reader.xlines.size - radius)
            for xline_index in range(start, stop):
                if self._lfm_support[fixed_index, xline_index] >= self.config.min_lfm_support:
                    keys.append(PatchKey(fixed_index, xline_index, orientation))
        else:
            start = max(inline_bounds[0], radius)
            stop = min(inline_bounds[1], reader.ilines.size - radius)
            for inline_index in range(start, stop):
                if self._lfm_support[inline_index, fixed_index] >= self.config.min_lfm_support:
                    keys.append(PatchKey(inline_index, fixed_index, orientation))
        return tuple(keys)

    def predict(
        self,
        *,
        inline_bounds: tuple[int, int] | None = None,
        xline_bounds: tuple[int, int] | None = None,
    ) -> BodyVolumeResult:
        """Predict and fuse one spatial tile; omitted bounds select the full survey."""

        reader = self.inverter.reader
        il_start, il_stop = _index_bounds(reader.ilines.size, inline_bounds, name="inline")
        xl_start, xl_stop = _index_bounds(reader.xlines.size, xline_bounds, name="xline")
        shape = (il_stop - il_start, xl_stop - xl_start, reader.sample_axis.values.size)
        body_sum = np.zeros(shape, dtype=np.float32)
        direction_count = np.zeros(shape, dtype=np.uint8)
        disagreement = np.full(shape, np.nan, dtype=np.float32)
        section_count = 0
        total_sections = sum(
            (il_stop - il_start) if orientation == "inline" else (xl_stop - xl_start)
            for orientation in self.config.orientations
        )
        started = time.perf_counter()

        for orientation in self.config.orientations:
            fixed_indices: Iterable[int]
            if orientation == "inline":
                fixed_indices = range(il_start, il_stop)
            else:
                fixed_indices = range(xl_start, xl_stop)
            for fixed_index in fixed_indices:
                keys = self._section_keys(
                    orientation,
                    fixed_index,
                    inline_bounds=(il_start, il_stop),
                    xline_bounds=(xl_start, xl_stop),
                )
                if keys:
                    prediction = self.inverter.predict_body(keys, center_visible=True)
                    bodies = prediction.body_log_ai.detach().cpu().numpy().astype(np.float32, copy=False)
                    supports = prediction.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
                    for row, key in enumerate(prediction.keys):
                        local_i = key.inline_index - il_start
                        local_j = key.xline_index - xl_start
                        support = supports[row]
                        previous_count = direction_count[local_i, local_j]
                        if np.any(previous_count[support] > 1):
                            raise ValueError("A volume sample received more than two orientation predictions.")
                        paired = support & (previous_count == 1)
                        disagreement[local_i, local_j, paired] = np.abs(
                            bodies[row, paired] - body_sum[local_i, local_j, paired]
                        )
                        body_sum[local_i, local_j, support] += bodies[row, support]
                        direction_count[local_i, local_j, support] += np.uint8(1)
                section_count += 1
                if section_count % self.config.log_every_sections == 0 or section_count == total_sections:
                    elapsed = time.perf_counter() - started
                    self.log.info(
                        "volume inference %d/%d sections | orientation=%s | fixed_index=%d | elapsed=%.1fs",
                        section_count,
                        total_sections,
                        orientation,
                        fixed_index,
                        elapsed,
                    )

        supported = direction_count > 0
        body_sum[supported] /= direction_count[supported].astype(np.float32)
        body_sum[~supported] = np.nan
        local_lfm = np.asarray(
            reader.lfm_log_ai[il_start:il_stop, xl_start:xl_stop],
            dtype=np.float32,
        )
        target_mask = np.asarray(
            reader.lfm_valid_mask[il_start:il_stop, xl_start:xl_stop],
            dtype=bool,
        ) & np.isfinite(local_lfm)
        spacing = reader.geometry.bin_spacing_m()
        body_sum, fill_code, fill_mean_m, fill_max_m = _fill_target_zone(
            body_sum,
            direction_count,
            local_lfm,
            target_mask,
            inline_spacing_m=float(spacing["inline_spacing_m"]),
            xline_spacing_m=float(spacing["xline_spacing_m"]),
        )
        return BodyVolumeResult(
            body_log_ai=body_sum,
            direction_count=direction_count,
            direction_disagreement_log_ai=disagreement,
            fill_code=fill_code,
            inline_indices=np.arange(il_start, il_stop, dtype=np.int64),
            xline_indices=np.arange(xl_start, xl_stop, dtype=np.int64),
            nearest_fill_mean_distance_m=fill_mean_m,
            nearest_fill_max_distance_m=fill_max_m,
        )


def centered_tile_bounds(size: int, tile_size: int) -> tuple[int, int]:
    """Return one deterministic centered half-open range for smoke inference."""

    if isinstance(tile_size, bool) or int(tile_size) != tile_size or not 1 <= int(tile_size) <= size:
        raise ValueError("tile_size must be a positive integer no larger than the axis.")
    start = (int(size) - int(tile_size)) // 2
    return start, start + int(tile_size)


__all__ = [
    "BodyVolumeInverter",
    "BodyVolumeResult",
    "VolumeInferenceConfig",
    "centered_tile_bounds",
]
