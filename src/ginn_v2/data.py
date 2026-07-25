"""Parent-atomic splits and streaming teacher-forcing batches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from ginn_v2.anchor import (
    LfmAnchoredStructuredSample,
    anchor_to_lfm,
    load_zone_ai_bounds,
)
from ginn_v2.oracle import project_log_ai_to_model_grid
from ginn_v2.truth import StructuredTruthAdapter


SPLIT_MANIFEST_SCHEMA = "structured_ginn_v2_parent_split_v1"
SPLIT_NAMES = ("training", "tuning_validation", "calibration", "geometry_holdout")


@dataclass(frozen=True)
class ParentSplitManifest:
    seed: int
    training: tuple[str, ...]
    tuning_validation: tuple[str, ...]
    calibration: tuple[str, ...]
    geometry_holdout: tuple[str, ...]
    strata_columns: tuple[str, ...] = (
        "section_id",
        "duration_mode",
        "geometry_family",
    )

    def __post_init__(self) -> None:
        sets = [set(getattr(self, name)) for name in SPLIT_NAMES]
        if any(not values for values in sets):
            raise ValueError("every parent split must be non-empty.")
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                if sets[left].intersection(sets[right]):
                    raise ValueError("parent split sets must be pairwise disjoint.")

    def parent_ids(self, split: str) -> tuple[str, ...]:
        if split not in SPLIT_NAMES:
            raise KeyError(f"unknown split {split!r}")
        return tuple(getattr(self, split))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SPLIT_MANIFEST_SCHEMA,
            "seed": int(self.seed),
            "strata_columns": list(self.strata_columns),
        }
        payload.update({name: list(getattr(self, name)) for name in SPLIT_NAMES})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["fingerprint_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ParentSplitManifest":
        if payload.get("schema") != SPLIT_MANIFEST_SCHEMA:
            raise ValueError("unsupported Structured GINN V2 split manifest.")
        expected = dict(payload)
        fingerprint = str(expected.pop("fingerprint_sha256", ""))
        canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if fingerprint != actual:
            raise ValueError("split manifest fingerprint mismatch.")
        return cls(
            seed=int(payload["seed"]),
            training=tuple(str(value) for value in payload["training"]),
            tuning_validation=tuple(
                str(value) for value in payload["tuning_validation"]
            ),
            calibration=tuple(str(value) for value in payload["calibration"]),
            geometry_holdout=tuple(
                str(value) for value in payload["geometry_holdout"]
            ),
            strata_columns=tuple(str(value) for value in payload["strata_columns"]),
        )


def build_parent_split_manifest(
    benchmark: StructuredSyntheticBenchmark,
    *,
    seed: int,
) -> ParentSplitManifest:
    """Create a deterministic 70/15/15 split and retain producer geometry holdout."""
    frame = benchmark.index.copy()
    required = {
        "realization_id",
        "evaluation_role",
        "section_id",
        "duration_mode",
        "geometry_family",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"benchmark index lacks split columns: {missing}")
    development = frame.loc[frame["evaluation_role"] == "development_pool"].copy()
    holdout = frame.loc[frame["evaluation_role"] == "geometry_holdout"].copy()
    if development.empty or holdout.empty:
        raise ValueError("benchmark must contain development_pool and geometry_holdout.")
    assignment: dict[str, list[str]] = {
        "training": [],
        "tuning_validation": [],
        "calibration": [],
    }
    strata = ("section_id", "duration_mode", "geometry_family")
    for key, group in development.groupby(list(strata), sort=True, dropna=False):
        parent_ids = sorted(str(value) for value in group["realization_id"])
        if len(parent_ids) < 3:
            raise ValueError(f"split stratum {key!r} has fewer than three parents.")
        digest = hashlib.sha256(
            f"{int(seed)}|{'|'.join(str(value) for value in key)}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        order = rng.permutation(len(parent_ids))
        shuffled = [parent_ids[int(index)] for index in order]
        n_training = max(1, int(np.floor(0.70 * len(shuffled))))
        n_tuning = max(1, int(np.floor(0.15 * len(shuffled))))
        if n_training + n_tuning >= len(shuffled):
            n_training = len(shuffled) - 2
            n_tuning = 1
        assignment["training"].extend(shuffled[:n_training])
        assignment["tuning_validation"].extend(
            shuffled[n_training : n_training + n_tuning]
        )
        assignment["calibration"].extend(shuffled[n_training + n_tuning :])
    return ParentSplitManifest(
        seed=int(seed),
        training=tuple(sorted(assignment["training"])),
        tuning_validation=tuple(sorted(assignment["tuning_validation"])),
        calibration=tuple(sorted(assignment["calibration"])),
        geometry_holdout=tuple(sorted(str(value) for value in holdout["realization_id"])),
        strata_columns=strata,
    )


def freeze_parent_split_manifest(
    benchmark: StructuredSyntheticBenchmark,
    path: str | Path,
    *,
    seed: int,
) -> ParentSplitManifest:
    """Publish one immutable split manifest, accepting only identical reruns."""
    target = Path(path)
    proposed = build_parent_split_manifest(benchmark, seed=seed)
    if target.exists():
        existing = ParentSplitManifest.from_dict(
            json.loads(target.read_text(encoding="utf-8"))
        )
        if existing.to_dict() != proposed.to_dict():
            raise FileExistsError(f"split manifest already exists with different content: {target}")
        return existing
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.staging")
    temporary.write_text(
        json.dumps(proposed.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(target)
    return proposed


@dataclass(frozen=True)
class TeacherForcingBatch:
    seismic: np.ndarray
    lfm_residual: np.ndarray
    observed_valid: np.ndarray
    background_highres: np.ndarray
    truth_highres: np.ndarray
    zone_valid: np.ndarray
    segment_basis: np.ndarray
    segment_mask: np.ndarray
    pooling_mask: np.ndarray
    segment_valid: np.ndarray
    state_id: np.ndarray
    duration_fraction: np.ndarray
    extent_fraction: np.ndarray
    target_parameters: np.ndarray
    parameter_supervision_valid: np.ndarray
    profile_supervision_valid: np.ndarray
    ai_bounds: np.ndarray
    projected_truth: np.ndarray
    projected_support: np.ndarray
    projection_factor: int
    sample_keys: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.seismic.shape[0])


def _jitter_pooling_masks(
    masks: np.ndarray,
    *,
    maximum_shift: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(masks, dtype=bool).copy()
    if maximum_shift <= 0 or output.shape[0] < 2:
        return output
    occupied = np.flatnonzero(np.any(output, axis=0))
    if occupied.size < output.shape[0]:
        return output
    start = int(occupied[0])
    stop = int(occupied[-1]) + 1
    original = [int(np.flatnonzero(output[index])[0]) for index in range(output.shape[0])]
    boundaries = [start]
    previous = start
    for index in range(1, output.shape[0]):
        remaining = output.shape[0] - index
        center = original[index]
        lower = max(previous + 1, center - maximum_shift)
        upper = min(stop - remaining, center + maximum_shift)
        chosen = int(rng.integers(lower, upper + 1)) if lower <= upper else center
        boundaries.append(chosen)
        previous = chosen
    boundaries.append(stop)
    output[:] = False
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        output[index, left:right] = True
    return output


def collate_teacher_forcing_samples(
    samples: Sequence[LfmAnchoredStructuredSample],
    *,
    boundary_jitter_samples: int = 0,
    random_seed: int = 0,
) -> TeacherForcingBatch:
    """Pad anchored samples into one dense model batch."""
    if not samples:
        raise ValueError("cannot collate an empty teacher-forcing batch.")
    model_size = samples[0].source.observed.sample_axis.coordinates.size
    highres_size = samples[0].source.latent.latent_axis.coordinates.size
    ratio = (
        samples[0].source.observed.sample_axis.sample_interval
        / samples[0].source.latent.latent_axis.sample_interval
    )
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
        raise ValueError("teacher-forcing axes are not integer nested.")
    maximum_segments = max(len(sample.segments) for sample in samples)
    batch_size = len(samples)
    seismic = np.zeros((batch_size, model_size), dtype=np.float32)
    lfm_residual = np.zeros_like(seismic)
    observed_valid = np.zeros_like(seismic, dtype=bool)
    background = np.zeros((batch_size, highres_size), dtype=np.float32)
    truth = np.zeros_like(background)
    zone_valid = np.zeros_like(background, dtype=bool)
    basis = np.zeros((batch_size, maximum_segments, highres_size, 3), dtype=np.float32)
    segment_mask = np.zeros(basis.shape[:-1], dtype=bool)
    pooling_mask = np.zeros_like(segment_mask)
    segment_valid = np.zeros((batch_size, maximum_segments), dtype=bool)
    state_id = np.zeros((batch_size, maximum_segments), dtype=np.int64)
    duration = np.zeros((batch_size, maximum_segments), dtype=np.float32)
    extent = np.zeros((batch_size, maximum_segments, 2), dtype=np.float32)
    parameters = np.zeros((batch_size, maximum_segments, 3), dtype=np.float32)
    parameter_valid = np.zeros((batch_size, maximum_segments), dtype=bool)
    profile_valid = np.zeros_like(parameter_valid)
    ai_bounds = np.zeros((batch_size, 2), dtype=np.float32)
    projected_truth = np.zeros((batch_size, model_size), dtype=np.float32)
    projected_support = np.zeros((batch_size, model_size), dtype=bool)
    keys: list[str] = []
    rng = np.random.default_rng(int(random_seed))
    for batch_index, sample in enumerate(samples):
        source = sample.source
        if (
            source.observed.sample_axis.coordinates.size != model_size
            or source.latent.latent_axis.coordinates.size != highres_size
        ):
            raise ValueError("all samples in a batch must have matching axis sizes.")
        seismic[batch_index] = np.nan_to_num(source.observed.seismic, nan=0.0)
        lfm_residual[batch_index] = np.nan_to_num(
            source.observed.lfm - sample.background_lfm_model,
            nan=0.0,
        )
        observed_valid[batch_index] = source.observed.observed_valid
        background[batch_index] = sample.background_lfm_highres
        truth[batch_index] = np.nan_to_num(
            source.latent.log_ai_highres_truth,
            nan=0.0,
        )
        zone_valid[batch_index] = source.zone.zone_valid
        ai_bounds[batch_index] = sample.ai_bounds
        for segment_index, segment in enumerate(sample.segments):
            indices = segment.sample_indices
            basis[batch_index, segment_index, indices] = segment.basis
            segment_mask[batch_index, segment_index, indices] = True
            segment_valid[batch_index, segment_index] = True
            state_id[batch_index, segment_index] = segment.source.state_id
            duration[batch_index, segment_index] = segment.source.duration_fraction
            extent[batch_index, segment_index] = (
                (segment.source.top - source.zone.top)
                / (source.zone.bottom - source.zone.top),
                (segment.source.bottom - source.zone.top)
                / (source.zone.bottom - source.zone.top),
            )
            parameters[batch_index, segment_index] = (
                segment.effective_parameters_lfm
            )
            parameter_valid[batch_index, segment_index] = (
                segment.parameter_supervision_valid
            )
            profile_valid[batch_index, segment_index] = (
                segment.profile_supervision_valid
            )
        segment_count = len(sample.segments)
        pooling_mask[batch_index, :segment_count] = _jitter_pooling_masks(
            segment_mask[batch_index, :segment_count],
            maximum_shift=int(boundary_jitter_samples),
            rng=rng,
        )
        projection = project_log_ai_to_model_grid(
            source.latent.log_ai_highres_truth,
            highres_axis=source.latent.latent_axis,
            model_axis=source.observed.sample_axis,
        )
        projected_truth[batch_index] = np.nan_to_num(projection.model_log_ai, nan=0.0)
        projected_support[batch_index] = (
            projection.support_model & source.observed.observed_valid
        )
        keys.append(
            f"{source.realization_id}|{source.lateral_index}|{source.zone.zone_id}"
        )
    return TeacherForcingBatch(
        seismic=seismic,
        lfm_residual=lfm_residual,
        observed_valid=observed_valid,
        background_highres=background,
        truth_highres=truth,
        zone_valid=zone_valid,
        segment_basis=basis,
        segment_mask=segment_mask,
        pooling_mask=pooling_mask,
        segment_valid=segment_valid,
        state_id=state_id,
        duration_fraction=duration,
        extent_fraction=extent,
        target_parameters=parameters,
        parameter_supervision_valid=parameter_valid,
        profile_supervision_valid=profile_valid,
        ai_bounds=ai_bounds,
        projected_truth=projected_truth,
        projected_support=projected_support,
        projection_factor=factor,
        sample_keys=tuple(keys),
    )


class TeacherForcingDataModule:
    """Stream parent-local trace batches without repeatedly loading the same HDF5 parent."""

    def __init__(
        self,
        benchmark_dir: str | Path,
        calibration_path: str | Path,
        split_manifest: ParentSplitManifest,
        *,
        condition_limit: float = 100.0,
    ) -> None:
        self.benchmark = StructuredSyntheticBenchmark(benchmark_dir)
        self.ai_bounds = load_zone_ai_bounds(calibration_path)
        self.split_manifest = split_manifest
        self.condition_limit = float(condition_limit)
        benchmark_ids = {
            item.realization_id for item in self.benchmark.list_parents()
        }
        manifest_ids = set().union(
            *(set(split_manifest.parent_ids(name)) for name in SPLIT_NAMES)
        )
        if benchmark_ids != manifest_ids:
            raise ValueError("split manifest parent set differs from benchmark.")

    def _parent_samples(
        self,
        parent_id: str,
        *,
        rng: np.random.Generator,
        samples_per_zone: int | None,
        maximum_samples: int | None,
    ) -> list[LfmAnchoredStructuredSample]:
        parent = self.benchmark.read_parent(parent_id)
        keys = sorted(
            {
                (int(row["lateral_index"]), str(row["zone_id"]))
                for row in parent.zones
                if bool(row.get("zone_valid", True))
            }
        )
        if samples_per_zone is not None:
            selected_by_zone: list[tuple[int, str]] = []
            zone_ids = sorted({zone_id for _, zone_id in keys})
            for zone_id in zone_ids:
                zone_keys = [
                    key for key in keys if key[1] == zone_id
                ]
                count = min(int(samples_per_zone), len(zone_keys))
                if count == len(zone_keys):
                    selected_by_zone.extend(zone_keys)
                    continue
                phase = float(rng.random())
                positions = np.floor(
                    (np.arange(count, dtype=np.float64) + phase)
                    * len(zone_keys)
                    / count
                ).astype(np.int64)
                if np.unique(positions).size != count:
                    raise RuntimeError("stratified lateral sampling produced duplicates.")
                selected_by_zone.extend(zone_keys[int(index)] for index in positions)
            keys = sorted(selected_by_zone)
        if maximum_samples is not None and len(keys) > maximum_samples:
            selected = rng.choice(len(keys), size=int(maximum_samples), replace=False)
            keys = [keys[int(index)] for index in sorted(selected)]
        samples: list[LfmAnchoredStructuredSample] = []
        for lateral_index, zone_id in keys:
            structured = StructuredTruthAdapter.from_structured_parent(
                parent,
                zone_id=zone_id,
                lateral_index=lateral_index,
            )
            samples.append(
                anchor_to_lfm(
                    structured,
                    ai_bounds=self.ai_bounds,
                    condition_limit=self.condition_limit,
                )
            )
        return samples

    def iter_batches(
        self,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        maximum_parents: int | None = None,
        samples_per_zone_per_parent: int | None = None,
        maximum_samples_per_parent: int | None = None,
        boundary_jitter_samples: int = 0,
    ) -> Iterator[TeacherForcingBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if samples_per_zone_per_parent is not None and samples_per_zone_per_parent <= 0:
            raise ValueError("samples_per_zone_per_parent must be positive when supplied.")
        parent_ids = list(self.split_manifest.parent_ids(split))
        rng = np.random.default_rng(int(seed))
        if maximum_parents is not None:
            parent_ids = parent_ids[: int(maximum_parents)]
        if shuffle:
            rng.shuffle(parent_ids)
        for parent_id in parent_ids:
            samples = self._parent_samples(
                parent_id,
                rng=rng,
                samples_per_zone=samples_per_zone_per_parent,
                maximum_samples=maximum_samples_per_parent,
            )
            if shuffle:
                rng.shuffle(samples)
            for start in range(0, len(samples), int(batch_size)):
                yield collate_teacher_forcing_samples(
                    samples[start : start + int(batch_size)],
                    boundary_jitter_samples=boundary_jitter_samples,
                    random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )


__all__ = [
    "ParentSplitManifest",
    "SPLIT_MANIFEST_SCHEMA",
    "SPLIT_NAMES",
    "TeacherForcingBatch",
    "TeacherForcingDataModule",
    "build_parent_split_manifest",
    "collate_teacher_forcing_samples",
    "freeze_parent_split_manifest",
]
