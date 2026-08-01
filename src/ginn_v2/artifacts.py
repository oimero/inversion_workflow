"""Corpus and checkpoint publication for Structured GINN V2.

This module validates semantic identity.  Recorded fingerprints are provenance
only and are never recomputed or compared as an admission condition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Mapping

import numpy as np
import torch

from cup.physics.numpy_backend import velocity_from_ai
from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from cup.synthetic.schemas import STRUCTURED_ARTIFACT_VERSION

from ginn_v2.generator import ConditionalGenerator
from ginn_v2.augmentation import (
    ObservationAugmentationProfile,
    apply_observation_augmentation,
    stable_random_identity,
)
from ginn_v2.contracts import ObservationTile
from ginn_v2.evidence import build_tuning_targets
from ginn_v2.representation import build_lfm_anchor
from ginn_v2.semi_markov import SemiMarkovPrior


CORPUS_MANIFEST_SCHEMA = "structured_synthetic_corpus_v2"
CHECKPOINT_SCHEMA = "structured_ginn_v2_checkpoint_v1"
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
) -> Iterator[Mapping[str, np.ndarray]]:
    if split not in {"training", "tuning", "calibration"}:
        raise ValueError("evidence batches require a development split.")
    if condition not in {"clean", "dirty"}:
        raise ValueError("condition must be clean or dirty.")
    if condition == "dirty" and augmentation_profile is None:
        raise ValueError("dirty evidence requires a frozen augmentation profile.")
    for parent_id in corpus.splits[split]:
        parent = corpus.benchmark.read_parent(parent_id)
        wavelet_time = np.asarray(
            parent.forward_context["wavelet_time_s"], dtype=np.float64
        )
        wavelet = np.asarray(
            parent.forward_context["wavelet_amplitude"], dtype=np.float64
        )
        vp = None
        if parent.sample_domain == "depth":
            relation = dict(parent.forward_context["ai_velocity_relation"])
            vp = velocity_from_ai(
                np.exp(parent.model_log_ai),
                a=float(relation["a"]),
                b=float(relation["b"]),
            )
        for tile in parent_observation_tiles(corpus, parent_id):
            anchor = build_lfm_anchor(tile)
            targets = build_tuning_targets(
                tile,
                log_ai_highres=parent.log_ai_highres,
                state_highres=parent.state_id_highres,
                background_lfm_highres=anchor.highres,
                wavelet_time_s=wavelet_time,
                wavelet_amplitude=wavelet,
                vp_model_mps=vp,
            )
            supervised_trace = np.zeros(tile.width, dtype=bool)
            context_radius = 10
            if tile.width < 2 * context_radius + 1:
                raise ValueError(
                    "training parent must provide at least 21 lateral traces."
                )
            supervised_trace[
                context_radius : tile.width - context_radius
            ] = tile.lateral_valid[
                context_radius : tile.width - context_radius
            ]
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
                "seismic": training_tile.seismic[None].astype(np.float32),
                "lfm": training_tile.lfm[None].astype(np.float32),
                "observed_valid": training_tile.observed_valid[None],
                "lateral_m": training_tile.lateral_m[None].astype(np.float32),
                "lateral_valid": training_tile.lateral_valid[None],
                "increment": targets.increment[None].astype(np.float32),
                "state_occupancy": targets.state_occupancy[None].astype(
                    np.float32
                ),
                "interface_activity": targets.interface_activity[None].astype(
                    np.float32
                ),
                "support": (
                    targets.support & supervised_trace[:, None]
                )[None],
            }


def iter_paired_evidence_batches(
    corpus: Corpus,
    split: str,
    *,
    augmentation_profile: ObservationAugmentationProfile,
) -> Iterator[Mapping[str, np.ndarray]]:
    """Yield each clean parent-zone batch immediately followed by its dirty pair."""
    clean = iter_evidence_batches(corpus, split, condition="clean")
    dirty = iter_evidence_batches(
        corpus,
        split,
        condition="dirty",
        augmentation_profile=augmentation_profile,
    )
    for clean_batch, dirty_batch in zip(clean, dirty, strict=True):
        yield clean_batch
        yield dirty_batch


def calibrate_semi_markov_prior(corpus: Corpus) -> SemiMarkovPrior:
    initial = np.ones(3, dtype=np.float64)
    transition = np.ones((3, 3), dtype=np.float64)
    durations: list[list[float]] = [[], [], []]
    for parent_id in corpus.splits["training"]:
        parent = corpus.benchmark.read_parent(parent_id)
        groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
        for row in parent.segments:
            if int(row["duration_samples"]) <= 0:
                continue
            key = (int(row["lateral_index"]), str(row["zone_id"]))
            groups.setdefault(key, []).append(row)
        for rows in groups.values():
            ordered = sorted(rows, key=lambda row: (float(row["top"]), int(row["object_id"])))
            states = [int(row["state_id"]) for row in ordered]
            if not states or any(state not in {0, 1, 2} for state in states):
                raise ValueError("training segment states must be 0, 1, or 2.")
            initial[states[0]] += 1.0
            for previous, current in zip(states[:-1], states[1:]):
                transition[previous, current] += 1.0
            for row, state in zip(ordered, states):
                duration = float(row["duration_fraction"])
                if not np.isfinite(duration) or duration <= 0.0:
                    raise ValueError("duration_fraction must be finite and positive.")
                durations[state].append(duration)
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
        "evidence_increment_mean": prediction.evidence.bandlimited_increment_mean,
        "evidence_increment_scale": prediction.evidence.bandlimited_increment_scale,
        "evidence_state_occupancy": prediction.evidence.state_occupancy,
        "evidence_interface_activity": prediction.evidence.interface_activity,
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
    "public_checkpoint_metadata",
    "load_corpus",
    "calibrate_semi_markov_prior",
    "iter_evidence_batches",
    "iter_paired_evidence_batches",
    "parent_observation_tiles",
    "save_section_prediction",
    "save_checkpoint",
]
