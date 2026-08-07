"""Corpus and checkpoint publication for Structured GINN V2.

This module validates semantic identity.  Recorded fingerprints are provenance
only and are never recomputed or compared as an admission condition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any, Mapping

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from cup.synthetic.schemas import STRUCTURED_ARTIFACT_VERSION

from ginn_v2.generator import ConditionalGenerator
from ginn_v2.augmentation import (
    ObservationAugmentationProfile,
    apply_observation_augmentation,
    stable_random_identity,
)
from ginn_v2.contracts import ObservableTargetContract, ObservationTile
from ginn_v2.evidence import build_observable_targets
from ginn_v2.representation import build_lfm_anchor, lfm_residual_from_anchor
from ginn_v2.semi_markov import SemiMarkovConditioning, SemiMarkovPrior


CORPUS_MANIFEST_SCHEMA = "structured_synthetic_corpus_v2"
CHECKPOINT_SCHEMA = "structured_ginn_v2_checkpoint_v3"
SPLIT_NAMES = ("training", "tuning", "calibration", "section_gate")


@dataclass(frozen=True)
class Corpus:
    root: Path
    benchmark: StructuredSyntheticBenchmark
    manifest: Mapping[str, Any]
    splits: Mapping[str, tuple[str, ...]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_corpus(root: str | Path) -> Corpus:
    directory = Path(root)
    manifest_path = directory / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError(
            f"Structured GINN V2 requires {CORPUS_MANIFEST_SCHEMA}; "
            "V1 artifacts are intentionally incompatible."
        )
    if manifest.get("status") not in {"success", "completed_with_warnings"}:
        raise ValueError("canonical corpus is not a completed publication.")
    benchmark = StructuredSyntheticBenchmark(directory)
    if STRUCTURED_ARTIFACT_VERSION != 2:
        raise RuntimeError("runtime reader is not configured for canonical artifact V2.")

    index = benchmark.index
    required = {"realization_id", "corpus_role", "split_role"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"canonical V2 index lacks columns: {missing}")
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
            raise ValueError(f"parent split overlap detected: {sorted(overlap)[:5]}")
        seen.update(values)
        split_map[split] = values
    indexed = set(index["realization_id"].astype(str))
    if seen != indexed:
        raise ValueError("every canonical parent must belong to exactly one split.")
    return Corpus(
        root=directory,
        benchmark=benchmark,
        manifest=manifest,
        splits=split_map,
    )


def load_observable_target_contract(
    path: str | Path,
    *,
    corpus: Corpus,
) -> ObservableTargetContract:
    contract = ObservableTargetContract.from_mapping(_read_json(Path(path)))
    benchmark = corpus.benchmark
    if (
        contract.sample_domain != benchmark.sample_domain
        or contract.sample_unit != benchmark.sample_unit
        or contract.depth_basis != benchmark.depth_basis
    ):
        raise ValueError(
            "observable target contract domain, unit, or depth basis differs "
            "from the canonical corpus."
        )
    return contract


def parent_observation_tiles(
    corpus: Corpus,
    parent_id: str,
) -> tuple[ObservationTile, ...]:
    parent = corpus.benchmark.read_parent(parent_id)
    zone_ids = sorted({str(row["zone_id"]) for row in parent.zones})
    tiles: list[ObservationTile] = []
    for zone_id in zone_ids:
        top = np.full(parent.lateral_m.size, np.nan, dtype=np.float64)
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
                identity=f"{parent_id}:{zone_id}",
            )
        )
    return tuple(tiles)


def iter_evidence_batches(
    corpus: Corpus,
    split: str,
    *,
    condition: str = "clean",
    augmentation_profile: ObservationAugmentationProfile | None = None,
    parent_limit: int | None = None,
    parent_ids: Sequence[str] | None = None,
) -> Iterator[Mapping[str, Any]]:
    if split not in {"training", "tuning", "calibration"}:
        raise ValueError("evidence batches require a development split.")
    if condition not in {"clean", "dirty"}:
        raise ValueError("condition must be clean or dirty.")
    if condition == "dirty" and augmentation_profile is None:
        raise ValueError("dirty evidence requires a frozen augmentation profile.")
    if parent_limit is not None and parent_limit <= 0:
        raise ValueError("parent_limit must be positive when provided.")
    if parent_limit is not None and parent_ids is not None:
        raise ValueError("parent_limit and parent_ids are mutually exclusive.")
    selected = corpus.splits[split]
    if parent_ids is not None:
        selected = tuple(str(value) for value in parent_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("parent_ids cannot contain duplicates.")
        invalid = sorted(set(selected).difference(corpus.splits[split]))
        if invalid:
            raise ValueError(
                f"parent_ids do not belong to split {split!r}: {invalid[:5]}"
            )
    elif parent_limit is not None:
        selected = selected[:parent_limit]
    index_by_parent = (
        corpus.benchmark.index.set_index("realization_id", drop=False)
    )
    for parent_id in selected:
        if parent_id not in index_by_parent.index:
            raise ValueError(f"canonical index cannot resolve parent {parent_id!r}.")
        index_row = index_by_parent.loc[parent_id]
        if getattr(index_row, "ndim", 1) != 1:
            raise ValueError(f"canonical index contains duplicate parent {parent_id!r}.")
        parent = corpus.benchmark.read_parent(parent_id)
        for tile in parent_observation_tiles(corpus, parent_id):
            anchor = build_lfm_anchor(tile)
            lfm_residual = lfm_residual_from_anchor(tile, anchor)
            targets = build_observable_targets(
                tile,
                model_log_ai=parent.model_log_ai,
                state_highres=parent.state_id_highres,
                background_lfm_linear=anchor.model,
                anchor_support=anchor.model_support,
            )
            ratio = tile.model_axis.sample_interval / tile.highres_axis.sample_interval
            factor = int(round(ratio))
            object_id_model = np.asarray(parent.object_id_highres[:, ::factor])
            if object_id_model.shape != targets.support.shape:
                raise ValueError("model-grid object identity changed shape.")
            object_id_model = np.where(
                targets.support,
                object_id_model,
                -1,
            ).astype(np.int64)
            training_tile = tile
            if condition == "dirty":
                random_identity = stable_random_identity(
                    parent_id,
                    tile.identity,
                    "dirty",
                )
                augmented = apply_observation_augmentation(
                    tile.seismic,
                    tile.observed_valid,
                    profile=augmentation_profile,
                    rng=np.random.default_rng(random_identity),
                    relative_lateral_m=tile.lateral_m
                    - float(np.median(tile.lateral_m)),
                )
                training_tile = replace(
                    tile,
                    seismic=augmented.seismic,
                    observed_valid=augmented.observed_valid,
                )
            yield {
                "parent_id": parent_id,
                "geometry_family": str(index_row["geometry_family"]),
                "tile_id": tile.identity,
                "condition": condition,
                "seismic": training_tile.seismic[None].astype(np.float32),
                "lfm": training_tile.lfm[None].astype(np.float32),
                "lfm_residual": lfm_residual[None].astype(np.float32),
                "background_lfm_linear": anchor.model[None].astype(np.float32),
                "observed_valid": training_tile.observed_valid[None],
                "lateral_m": training_tile.lateral_m[None].astype(np.float32),
                "lateral_valid": training_tile.lateral_valid[None],
                "projected_log_ai_increment": (
                    targets.projected_log_ai_increment[None].astype(np.float32)
                ),
                "signed_reflectivity": (
                    targets.signed_reflectivity[None].astype(np.float32)
                ),
                "state_emission": targets.state_id[None].astype(np.int64),
                "truth_object_id": object_id_model[None],
                "support": targets.support[None],
            }


def iter_paired_evidence_batches(
    corpus: Corpus,
    split: str,
    *,
    augmentation_profile: ObservationAugmentationProfile,
    parent_limit: int | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Yield each clean parent-zone batch immediately followed by its dirty pair."""
    clean = iter_evidence_batches(
        corpus,
        split,
        condition="clean",
        parent_limit=parent_limit,
    )
    dirty = iter_evidence_batches(
        corpus,
        split,
        condition="dirty",
        augmentation_profile=augmentation_profile,
        parent_limit=parent_limit,
    )
    for clean_batch, dirty_batch in zip(clean, dirty, strict=True):
        yield clean_batch
        yield dirty_batch


def iter_segment_profile_batches(
    corpus: Corpus,
    split: str,
    *,
    parent_limit: int | None = None,
    parent_ids: Sequence[str] | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Yield clean evidence batches with explicit high-resolution truth segments."""

    cached_parent_id: str | None = None
    parent = None
    tile_by_identity: dict[str, ObservationTile] = {}
    for source in iter_evidence_batches(
        corpus,
        split,
        parent_limit=parent_limit,
        parent_ids=parent_ids,
    ):
        parent_id = str(source["parent_id"])
        if parent_id != cached_parent_id:
            parent = corpus.benchmark.read_parent(parent_id)
            tile_by_identity = {
                tile.identity: tile
                for tile in parent_observation_tiles(corpus, parent_id)
            }
            cached_parent_id = parent_id
        if parent is None:
            raise RuntimeError("segment profile parent cache is empty.")
        tile_id = str(source["tile_id"])
        if tile_id not in tile_by_identity:
            raise ValueError("segment profile batch cannot resolve its observation tile.")
        tile = tile_by_identity[tile_id]
        zone_id = tile_id.rsplit(":", maxsplit=1)[-1]
        anchor = build_lfm_anchor(tile)
        truth_increment = parent.log_ai_highres - anchor.highres
        truth_support = (
            parent.truth_valid_highres
            & anchor.highres_support
            & np.isfinite(truth_increment)
        )

        trace_index: list[int] = []
        state_id: list[int] = []
        start_index: list[int] = []
        stop_index: list[int] = []
        duration_fraction: list[float] = []
        clipping_fraction: list[float] = []
        for row in parent.segments:
            if str(row["zone_id"]) != zone_id or not bool(
                row["segment_supervision_valid"]
            ):
                continue
            trace = int(row["lateral_index"])
            selected = (
                truth_support[trace]
                & (parent.zone_id_highres[trace] == int(row["zone_grid_value"]))
                & (parent.object_id_highres[trace] == int(row["object_id"]))
            )
            indices = np.flatnonzero(selected)
            if indices.size == 0:
                continue
            start = int(indices[0])
            stop = int(indices[-1]) + 1
            if indices.size != stop - start:
                raise ValueError("truth segment support is not contiguous.")
            trace_index.append(trace)
            state_id.append(int(row["state_id"]))
            start_index.append(start)
            stop_index.append(stop)
            duration_fraction.append(float(row["duration_fraction"]))
            clipping_fraction.append(
                float(np.mean(parent.clipping_mask_highres[trace, start:stop]))
            )
        if not trace_index:
            raise ValueError("segment profile batch contains no supervised segments.")
        result = dict(source)
        result.update(
            {
                "observation_tile": tile,
                "lfm_anchor_highres": anchor.highres,
                "highres_log_ai_increment": truth_increment,
                "highres_truth_support": truth_support,
                "segment_trace_index": np.asarray(trace_index, dtype=np.int64),
                "segment_state_id": np.asarray(state_id, dtype=np.int64),
                "segment_start_index": np.asarray(start_index, dtype=np.int64),
                "segment_stop_index": np.asarray(stop_index, dtype=np.int64),
                "segment_duration_fraction": np.asarray(
                    duration_fraction, dtype=np.float64
                ),
                "segment_clipping_fraction": np.asarray(
                    clipping_fraction, dtype=np.float64
                ),
                "ai_velocity_relation": dict(
                    parent.forward_context["ai_velocity_relation"] or {}
                ),
            }
        )
        yield result


def load_semi_markov_contract(
    directory: str | Path,
) -> tuple[SemiMarkovPrior, SemiMarkovConditioning, Mapping[str, Any]]:
    """Load the selected HSMM prior and conditioning from one publication."""

    root = Path(directory)
    prior_path = root / "semi_markov_prior.json"
    calibration_path = root / "hsmm_calibration.json"
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    prior_payload = _read_json(prior_path)
    calibration = _read_json(calibration_path)
    if calibration.get("schema") not in {
        "structured_ginn_v2_hsmm_calibration_v2",
        "structured_ginn_v2_hsmm_calibration_v3",
        "structured_ginn_v2_hsmm_calibration_v4",
    } or calibration.get("status") != "success":
        raise ValueError("HSMM calibration is not a successful V2/V3/V4 publication.")
    selected = calibration.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("conditioning"), Mapping
    ):
        raise ValueError("HSMM calibration lacks selected conditioning.")
    return (
        SemiMarkovPrior.from_mapping(prior_payload),
        SemiMarkovConditioning.from_mapping(selected["conditioning"]),
        {
            "schema": calibration["schema"],
            "status": calibration["status"],
            "split": calibration.get("split"),
            "mode": calibration.get("mode"),
            "selected_conditioning": dict(selected["conditioning"]),
            "prior_path": str(prior_path),
            "calibration_path": str(calibration_path),
        },
    )


def calibrate_semi_markov_prior(
    corpus: Corpus,
    *,
    parent_ids: Sequence[str],
) -> SemiMarkovPrior:
    """Calibrate the model-grid prior from trace-local training truth paths.

    Birth, death, and model-grid coarsening can remove an intermediate event
    on one trace and thereby create a same-state renewal.  The event catalog
    cannot represent that local adjacency, so the prior is calibrated through
    the same model-grid reader used by HSMM training and evaluation.
    """

    selected = tuple(str(value) for value in parent_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("semi-Markov prior parent_ids must be unique and non-empty.")
    invalid = sorted(set(selected).difference(corpus.splits["training"]))
    if invalid:
        raise ValueError(
            f"semi-Markov prior parents are outside training: {invalid[:5]}"
        )
    initial = np.ones(3, dtype=np.float64)
    transition = np.ones((3, 3), dtype=np.float64)
    durations: list[list[float]] = [[], [], []]
    trace_count = 0
    for source in iter_evidence_batches(
        corpus,
        "training",
        parent_ids=selected,
    ):
        support = np.asarray(source["support"], dtype=bool)
        state = np.asarray(source["state_emission"], dtype=np.int64)
        object_id = np.asarray(source["truth_object_id"], dtype=np.int64)
        if support.shape != state.shape or support.shape != object_id.shape:
            raise ValueError("model-grid prior truth arrays must share one shape.")
        if support.ndim != 3 or support.shape[0] != 1:
            raise ValueError("model-grid prior expects one parent-zone per batch.")
        for trace in range(support.shape[1]):
            selected_indices = np.flatnonzero(support[0, trace])
            if selected_indices.size == 0:
                continue
            if np.any(np.diff(selected_indices) != 1):
                raise ValueError("model-grid prior zone support must be contiguous.")
            local_state = state[0, trace, selected_indices]
            local_object = object_id[0, trace, selected_indices]
            if np.any((local_state < 0) | (local_state > 2)) or np.any(
                local_object < 0
            ):
                raise ValueError("model-grid prior truth path is invalid.")
            starts = np.r_[
                0,
                1 + np.flatnonzero(local_object[1:] != local_object[:-1]),
            ]
            stops = np.r_[starts[1:], local_state.size]
            states: list[int] = []
            for start, stop in zip(starts, stops, strict=True):
                local = local_state[start:stop]
                if np.any(local != local[0]):
                    raise ValueError("one model-grid truth object has multiple states.")
                state_id = int(local[0])
                states.append(state_id)
                durations[state_id].append(
                    float((stop - start) / local_state.size)
                )
            initial[states[0]] += 1.0
            for previous, current in zip(states[:-1], states[1:]):
                transition[previous, current] += 1.0
            trace_count += 1
    if trace_count == 0:
        raise ValueError("semi-Markov prior calibration has no supported traces.")
    if any(not values for values in durations):
        raise ValueError("training corpus does not cover every state duration.")
    return SemiMarkovPrior(
        initial_probability=initial,
        transition_probability=transition,
        duration_fraction_mean=np.asarray(
            [np.mean(values) for values in durations], dtype=np.float64
        ),
        duration_fraction_std=np.asarray(
            [max(np.std(values), 1.0e-3) for values in durations],
            dtype=np.float64,
        ),
    )


def save_checkpoint(
    path: str | Path,
    generator: ConditionalGenerator,
    *,
    training_state: Mapping[str, Any],
    corpus_provenance: Mapping[str, Any],
    runtime_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "generator": generator.state_dict_payload(),
        "training_state": dict(training_state),
        "corpus_provenance": dict(corpus_provenance),
        "runtime_state": dict(runtime_state or {}),
    }
    temporary = target.with_name(f".{target.name}.staging")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ConditionalGenerator, Mapping[str, Any]]:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Structured GINN V2 checkpoint.")
    generator = ConditionalGenerator.from_payload(payload["generator"], device=device)
    metadata = {
        "training_state": dict(payload.get("training_state") or {}),
        "corpus_provenance": dict(payload.get("corpus_provenance") or {}),
        "target_contract": generator.target_contract.to_mapping(),
        "runtime_state": dict(payload.get("runtime_state") or {}),
    }
    return generator, metadata


def public_checkpoint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON-facing subset of checkpoint metadata.

    Checkpoints also carry optimizer and RNG state so interrupted training can
    resume exactly.  Those fields contain tensors and are deliberately kept
    out of evaluation reports; they are operational state, not public model
    provenance.
    """
    training_state = dict(metadata.get("training_state") or {})
    corpus_provenance = dict(metadata.get("corpus_provenance") or {})
    runtime = dict(metadata.get("runtime_state") or {})
    public_runtime = {
        key: runtime[key]
        for key in ("epoch", "best_tuning_loss", "best_epoch", "history")
        if key in runtime
    }
    return {
        "training_state": training_state,
        "corpus_provenance": corpus_provenance,
        "target_contract": dict(metadata.get("target_contract") or {}),
        "runtime_state": public_runtime,
    }


def save_section_prediction(
    output_dir: str | Path,
    prediction: Any,
) -> Path:
    from dataclasses import asdict

    from ginn_v2.contracts import StructuredPrediction

    if not isinstance(prediction, StructuredPrediction):
        raise TypeError("save_section_prediction requires StructuredPrediction.")
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    realizations = prediction.realizations
    if realizations is None:
        raise ValueError("section publication requires retained realizations.")
    arrays = {
        "evidence_projected_increment_mean": (
            prediction.evidence.projected_log_ai_increment_mean
        ),
        "evidence_projected_increment_scale": (
            prediction.evidence.projected_log_ai_increment_scale
        ),
        "evidence_signed_reflectivity_mean": (
            prediction.evidence.signed_reflectivity_mean
        ),
        "evidence_signed_reflectivity_scale": (
            prediction.evidence.signed_reflectivity_scale
        ),
        "evidence_state_log_potential": prediction.evidence.state_log_potential,
        "evidence_support": prediction.evidence.support,
        "ensemble_highres_mean": prediction.summary.log_ai_highres_mean,
        "ensemble_highres_std": prediction.summary.log_ai_highres_std,
        "ensemble_projected_support": prediction.summary.projected_support,
        "representative_highres_log_ai": prediction.representative.log_ai_highres,
        "representative_projected_log_ai": (
            prediction.representative.projected_log_ai
        ),
        "realization_highres_log_ai": np.stack(
            [item.log_ai_highres for item in realizations]
        ),
        "realization_projected_log_ai": np.stack(
            [item.projected_log_ai for item in realizations]
        ),
    }
    np.savez_compressed(directory / "section_prediction.npz", **arrays)
    segment_rows = [
        {
            "realization_identity": item.identity,
            **asdict(segment),
        }
        for item in realizations
        for segment in item.segments
    ]
    with (directory / "segment_table.json").open("w", encoding="utf-8") as handle:
        json.dump(segment_rows, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema": "structured_ginn_v2_section_prediction_v1",
                "representative_identity": prediction.representative.identity,
                "realization_identities": list(prediction.realization_identities),
                "diagnostics": dict(prediction.diagnostics),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    return directory


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CORPUS_MANIFEST_SCHEMA",
    "Corpus",
    "load_checkpoint",
    "load_semi_markov_contract",
    "load_observable_target_contract",
    "public_checkpoint_metadata",
    "load_corpus",
    "calibrate_semi_markov_prior",
    "iter_evidence_batches",
    "iter_paired_evidence_batches",
    "iter_segment_profile_batches",
    "parent_observation_tiles",
    "save_section_prediction",
    "save_checkpoint",
]
