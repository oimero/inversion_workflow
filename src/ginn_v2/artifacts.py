"""Canonical corpus and checkpoint publication for Structured GINN V2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from cup.synthetic.core.prior import ProducerPrior, load_producer_prior
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION, STRUCTURED_ARTIFACT_VERSION
from cup.physics.numpy_backend import velocity_from_ai

from ginn_v2.contracts import EventTrackTruth, InputContractError, ObservationTile


CORPUS_MANIFEST_SCHEMA = BENCHMARK_SCHEMA_VERSION
CHECKPOINT_SCHEMA = "structured_ginn_v2_checkpoint_v7"
SPLIT_NAMES = ("training", "tuning", "calibration", "section_gate")


@dataclass(frozen=True)
class Corpus:
    root: Path
    benchmark: StructuredSyntheticBenchmark
    manifest: Mapping[str, Any]
    producer_prior: ProducerPrior
    splits: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class StructuredTrainingTile:
    observation: ObservationTile
    event_tracks: tuple[EventTrackTruth, ...]
    model_log_ai: np.ndarray
    log_ai_highres: np.ndarray
    truth_valid_highres: np.ndarray
    object_id_highres: np.ndarray
    state_fraction_model: np.ndarray
    boundary_fraction_model: np.ndarray
    categorical_valid_model: np.ndarray
    hidden_transition_count_model: np.ndarray
    projection_collapse_mask_model: np.ndarray
    model_zone_support: np.ndarray
    highres_zone_support: np.ndarray
    vp_model_mps: np.ndarray | None = None

    def __post_init__(self) -> None:
        model_shape = self.observation.seismic.shape
        highres_shape = (
            self.observation.width,
            self.observation.highres_axis.coordinates.size,
        )
        model_fields = {
            "model_log_ai": (self.model_log_ai, model_shape),
            "boundary_fraction_model": (self.boundary_fraction_model, model_shape),
            "categorical_valid_model": (self.categorical_valid_model, model_shape),
            "hidden_transition_count_model": (
                self.hidden_transition_count_model,
                model_shape,
            ),
            "projection_collapse_mask_model": (
                self.projection_collapse_mask_model,
                model_shape,
            ),
            "model_zone_support": (self.model_zone_support, model_shape),
        }
        for name, (value, expected) in model_fields.items():
            parsed = np.asarray(value)
            if parsed.shape != expected:
                raise InputContractError(f"{name} shape differs from the model grid.")
            object.__setattr__(self, name, parsed)
        state_fraction = np.asarray(self.state_fraction_model, dtype=np.float64)
        if state_fraction.shape != model_shape + (3,):
            raise InputContractError("state_fraction_model must be [lateral, sample, 3].")
        for name in (
            "log_ai_highres",
            "truth_valid_highres",
            "object_id_highres",
            "highres_zone_support",
        ):
            parsed = np.asarray(getattr(self, name))
            if parsed.shape != highres_shape:
                raise InputContractError(f"{name} shape differs from the high-resolution grid.")
            object.__setattr__(self, name, parsed)
        if not self.event_tracks:
            raise InputContractError("structured training tile requires event tracks.")
        if any(track.presence.size != self.observation.width for track in self.event_tracks):
            raise InputContractError("event-track width differs from the observation tile.")
        object.__setattr__(self, "state_fraction_model", state_fraction)
        if self.observation.sample_domain == "depth":
            if self.vp_model_mps is None:
                raise InputContractError("depth training tiles require model-grid Vp.")
            velocity = np.asarray(self.vp_model_mps, dtype=np.float64)
            if velocity.shape != model_shape or np.any(
                self.observation.observed_valid
                & (~np.isfinite(velocity) | (velocity <= 0.0))
            ):
                raise InputContractError("depth training-tile Vp is invalid.")
            object.__setattr__(self, "vp_model_mps", velocity)
        elif self.vp_model_mps is not None:
            raise InputContractError("time training tiles cannot carry model-grid Vp.")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise InputContractError(f"JSON root must be an object: {path}")
    return payload


def load_corpus(root: str | Path) -> Corpus:
    """Load and semantically validate the single canonical corpus publication."""

    directory = Path(root)
    manifest_path = directory / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA:
        raise InputContractError(
            f"Structured GINN V2 requires {CORPUS_MANIFEST_SCHEMA}."
        )
    if manifest.get("status") not in {"success", "completed_with_warnings"}:
        raise InputContractError("canonical corpus is not a completed publication.")
    if STRUCTURED_ARTIFACT_VERSION != 3:
        raise RuntimeError("runtime reader is not configured for canonical artifact V3.")
    benchmark = StructuredSyntheticBenchmark(directory)
    prior_value = manifest.get("producer_prior")
    if not isinstance(prior_value, Mapping):
        raise InputContractError("canonical V3 manifest lacks producer_prior.")
    producer_prior = load_producer_prior(prior_value)
    if (
        producer_prior.sample_domain != benchmark.sample_domain
        or producer_prior.sample_unit != benchmark.sample_unit
        or producer_prior.depth_basis != benchmark.depth_basis
    ):
        raise InputContractError("producer prior and canonical corpus domains differ.")
    index = benchmark.index
    required = {"realization_id", "corpus_role", "split_role"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise InputContractError(f"canonical V3 index lacks columns: {missing}")
    split_map: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for split in SPLIT_NAMES:
        values = tuple(
            sorted(
                index.loc[index["split_role"].eq(split), "realization_id"]
                .astype(str)
                .tolist()
            )
        )
        overlap = seen.intersection(values)
        if overlap:
            raise InputContractError(f"parent split overlap detected: {sorted(overlap)[:5]}")
        seen.update(values)
        split_map[split] = values
    indexed = set(index["realization_id"].astype(str))
    if seen != indexed:
        raise InputContractError("every canonical parent must belong to one split.")
    return Corpus(directory, benchmark, manifest, producer_prior, split_map)


def _observation_tiles(parent: Any) -> tuple[ObservationTile, ...]:
    zone_ids = sorted({str(row["zone_id"]) for row in parent.zones})
    tiles: list[ObservationTile] = []
    for zone_id in zone_ids:
        top = np.full(parent.lateral_m.size, np.nan, dtype=float)
        bottom = np.full_like(top, np.nan)
        lateral_valid = np.zeros(parent.lateral_m.size, dtype=bool)
        for row in parent.zones:
            if str(row["zone_id"]) != zone_id:
                continue
            trace = int(row["lateral_index"])
            top[trace] = float(row["top"])
            bottom[trace] = float(row["bottom"])
            lateral_valid[trace] = bottom[trace] > top[trace]
        tiles.append(
            ObservationTile(
                model_axis=parent.model_axis,
                highres_axis=parent.highres_axis,
                seismic=np.where(parent.observed_valid, parent.seismic, 0.0),
                lfm=np.where(parent.observed_valid, parent.lfm, 0.0),
                observed_valid=parent.observed_valid,
                lateral_m=parent.lateral_m,
                lateral_valid=lateral_valid,
                zone_top=top,
                zone_bottom=bottom,
                x_m=parent.x_m,
                y_m=parent.y_m,
                identity=f"{parent.identity.realization_id}:{zone_id}",
            )
        )
    return tuple(tiles)


def parent_observation_tiles(corpus: Corpus, parent_id: str) -> tuple[ObservationTile, ...]:
    """Adapt one canonical parent into zone tiles with explicit axes and masks."""

    parent = corpus.benchmark.read_parent(parent_id)
    return _observation_tiles(parent)


def _event_tracks(parent: Any, zone_id: str) -> tuple[EventTrackTruth, ...]:
    rows = [row for row in parent.segments if str(row["zone_id"]) == zone_id]
    zone_values = {int(row["zone_grid_value"]) for row in rows}
    if len(zone_values) != 1:
        raise InputContractError("producer zone rows have ambiguous grid identity.")
    zone_value = zone_values.pop()
    zone_support = (
        (np.asarray(parent.zone_id_highres) == zone_value)
        & np.asarray(parent.truth_valid_highres, dtype=bool)
    )
    object_id_highres = np.asarray(parent.object_id_highres)
    if object_id_highres.shape != zone_support.shape:
        raise InputContractError("producer object and zone high-resolution grids differ.")
    event_ids = sorted({int(row["object_id"]) for row in rows})
    if not event_ids:
        raise InputContractError(f"zone {zone_id!r} has no producer events.")
    width = parent.lateral_m.size
    tracks: list[EventTrackTruth] = []
    for event_id in event_ids:
        selected = [row for row in rows if int(row["object_id"]) == event_id]
        states = {int(row["state_id"]) for row in selected}
        if len(states) != 1:
            raise InputContractError("producer event state changes across lateral traces.")
        presence = np.zeros(width, dtype=bool)
        top = np.full(width, np.nan, dtype=np.float64)
        bottom = np.full(width, np.nan, dtype=np.float64)
        duration = np.zeros(width, dtype=np.float64)
        supervision = np.zeros(width, dtype=bool)
        coefficients = {
            name: np.full((width, 3), np.nan, dtype=np.float64)
            for name in ("raw", "projected", "effective")
        }
        seen_rows: set[int] = set()
        for row in selected:
            trace = int(row["lateral_index"])
            if trace < 0 or trace >= width or trace in seen_rows:
                raise InputContractError("producer event has duplicate or invalid lateral rows.")
            seen_rows.add(trace)
            event_support = zone_support[trace] & (object_id_highres[trace] == event_id)
            event_samples = int(np.count_nonzero(event_support))
            if event_samples == 0:
                if int(row.get("duration_samples", 0)) != 0:
                    raise InputContractError(
                        "zero-sample producer event disagrees with duration_samples."
                    )
                continue
            zone_samples = int(np.count_nonzero(zone_support[trace]))
            if zone_samples <= 0:
                raise InputContractError("producer event is present outside zone support.")
            presence[trace] = True
            top[trace] = float(row["top"])
            bottom[trace] = float(row["bottom"])
            duration[trace] = event_samples / zone_samples
            supervision[trace] = bool(row["segment_supervision_valid"])
            for name in coefficients:
                coefficients[name][trace] = [
                    float(row[f"c0_{name}"]),
                    float(row[f"c1_{name}"]),
                    float(row[f"c2_{name}"]),
                ]
        raster_presence = np.any(
            zone_support & (object_id_highres == event_id), axis=1
        )
        if not np.array_equal(presence, raster_presence):
            raise InputContractError(
                "producer catalog and high-resolution event presence differ."
            )
        tracks.append(
            EventTrackTruth(
                zone_id=zone_id,
                event_id=event_id,
                state_id=states.pop(),
                presence=presence,
                top=top,
                bottom=bottom,
                duration_fraction=duration,
                raw_coefficients=coefficients["raw"],
                projected_coefficients=coefficients["projected"],
                effective_coefficients=coefficients["effective"],
                segment_supervision_valid=supervision,
            )
        )
    states = [track.state_id for track in tracks]
    if any(left == right for left, right in zip(states, states[1:], strict=False)):
        raise InputContractError("producer event sequence contains adjacent equal states.")
    duration_stack = np.stack([track.duration_fraction for track in tracks], axis=0)
    active = np.any(np.stack([track.presence for track in tracks], axis=0), axis=0)
    if np.any(
        active
        & ~np.isclose(np.sum(duration_stack, axis=0), 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise InputContractError("producer event durations do not fill their zone.")
    for trace in range(width):
        grid_ids = object_id_highres[trace, zone_support[trace]].astype(np.int64)
        if grid_ids.size == 0:
            continue
        if np.any(grid_ids < 0):
            raise InputContractError("producer events do not cover high-resolution zone support.")
        event_order = {track.event_id: index for index, track in enumerate(tracks)}
        try:
            ordered = np.asarray([event_order[int(item)] for item in grid_ids])
        except KeyError as error:
            raise InputContractError(
                "high-resolution event identity is absent from the producer catalog."
            ) from error
        if np.any(np.diff(ordered) < 0):
            raise InputContractError("producer event order reverses on the high-resolution grid.")
    return tuple(tracks)


def parent_training_tiles(
    corpus: Corpus,
    parent_id: str,
) -> tuple[StructuredTrainingTile, ...]:
    """Read one parent through the complete Stage-0 training seam."""

    parent = corpus.benchmark.read_parent(parent_id)
    if parent.sample_domain == "depth":
        relation = parent.forward_context.get("ai_velocity_relation")
        if not isinstance(relation, Mapping):
            raise InputContractError("depth parent lacks its AI-Vp relation.")
        lfm = np.asarray(parent.lfm, dtype=np.float64)
        finite_lfm = np.isfinite(lfm)
        vp_model_mps = np.full(lfm.shape, np.nan, dtype=np.float64)
        vp_model_mps[finite_lfm] = velocity_from_ai(
            np.exp(lfm[finite_lfm]),
            a=float(relation["a"]),
            b=float(relation["b"]),
        )
    else:
        vp_model_mps = None
    observations = _observation_tiles(parent)
    zone_rows = {str(row["zone_id"]): int(row["zone_grid_value"]) for row in parent.zones}
    results: list[StructuredTrainingTile] = []
    for observation in observations:
        zone_id = observation.identity.rsplit(":", 1)[-1]
        zone_grid_value = zone_rows[zone_id]
        model_zone = parent.zone_id_model == zone_grid_value
        highres_zone = parent.zone_id_highres == zone_grid_value
        results.append(
            StructuredTrainingTile(
                observation=observation,
                event_tracks=_event_tracks(parent, zone_id),
                model_log_ai=parent.model_log_ai,
                log_ai_highres=parent.log_ai_highres,
                truth_valid_highres=parent.truth_valid_highres,
                object_id_highres=parent.object_id_highres,
                state_fraction_model=parent.state_fraction_model,
                boundary_fraction_model=parent.boundary_fraction_model,
                categorical_valid_model=parent.categorical_valid_model,
                hidden_transition_count_model=parent.hidden_transition_count_model,
                projection_collapse_mask_model=parent.projection_collapse_mask_model,
                model_zone_support=model_zone & parent.observed_valid,
                highres_zone_support=highres_zone & parent.truth_valid_highres,
                vp_model_mps=vp_model_mps,
            )
        )
    return tuple(results)


def save_checkpoint(
    path: str | Path,
    *,
    model_state: Mapping[str, Any],
    model_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Publish one current-format checkpoint without upstream fingerprint gates."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": dict(model_config),
        "model_state": dict(model_state),
        "metadata": dict(metadata),
    }
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    temporary = target.with_suffix(target.suffix + ".staging")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load only the current checkpoint contract."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise InputContractError("checkpoint does not use the current schema.")
    for name in ("model_config", "model_state", "metadata"):
        if name not in payload:
            raise InputContractError(f"checkpoint lacks {name}.")
    return payload


def public_checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise InputContractError("checkpoint metadata must be a mapping.")
    return dict(metadata)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CORPUS_MANIFEST_SCHEMA",
    "Corpus",
    "StructuredTrainingTile",
    "SPLIT_NAMES",
    "load_checkpoint",
    "load_corpus",
    "parent_observation_tiles",
    "parent_training_tiles",
    "public_checkpoint_metadata",
    "save_checkpoint",
]
