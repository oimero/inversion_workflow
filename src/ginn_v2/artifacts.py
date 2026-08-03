"""Corpus and checkpoint publication for Structured GINN V2.

This module validates semantic identity.  Recorded fingerprints are provenance
only and are never recomputed or compared as an admission condition.
"""

from __future__ import annotations

import csv
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
from ginn_v2.semi_markov import SemiMarkovPrior


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
    for parent_id in selected:
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


def calibrate_semi_markov_prior(corpus: Corpus) -> SemiMarkovPrior:
    """Calibrate an event-level zone-fraction prior from the canonical catalog.

    The catalog contains one ordered geological event per parent and zone, so
    this avoids loading the large HDF5 arrays for every training parent.  It is
    still truth from training parents; lateral endpoint durations provide the
    within-event duration variation.
    """

    catalog = corpus.root / "object_catalog.csv"
    if not catalog.is_file():
        raise FileNotFoundError(catalog)
    training = set(corpus.splits["training"])
    groups: dict[tuple[str, str], list[tuple[int, int, tuple[float, ...]]]] = {}
    with catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "realization_id",
            "zone_id",
            "object_id",
            "state_id",
            "base_duration_fraction",
            "duration_fraction_start",
            "duration_fraction_end",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"object catalog lacks semi-Markov fields: {missing}")
        for row in reader:
            parent_id = str(row["realization_id"])
            if parent_id not in training:
                continue
            values: list[float] = []
            for name in ("duration_fraction_start", "duration_fraction_end"):
                text = str(row[name]).strip()
                if text:
                    value = float(text)
                    if not np.isfinite(value) or value < 0.0:
                        raise ValueError(
                            "object catalog duration fractions must be finite "
                            "and non-negative."
                        )
                    if value > 0.0:
                        values.append(value)
            if not values:
                values.append(float(row["base_duration_fraction"]))
            if any(not np.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError("object catalog duration fractions must be positive.")
            key = (parent_id, str(row["zone_id"]))
            groups.setdefault(key, []).append(
                (
                    int(row["object_id"]),
                    int(row["state_id"]),
                    tuple(values),
                )
            )
    if not groups:
        raise ValueError("object catalog contains no training events.")

    initial = np.ones(3, dtype=np.float64)
    transition = np.ones((3, 3), dtype=np.float64)
    durations: list[list[float]] = [[], [], []]
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: row[0])
        states = [row[1] for row in ordered]
        if not states or any(state not in {0, 1, 2} for state in states):
            raise ValueError("training event states must be 0, 1, or 2.")
        initial[states[0]] += 1.0
        for previous, current in zip(states[:-1], states[1:]):
            transition[previous, current] += 1.0
        for (_, state, values) in ordered:
            durations[state].extend(values)
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
    "load_observable_target_contract",
    "public_checkpoint_metadata",
    "load_corpus",
    "calibrate_semi_markov_prior",
    "iter_evidence_batches",
    "iter_paired_evidence_batches",
    "parent_observation_tiles",
    "save_section_prediction",
    "save_checkpoint",
]
