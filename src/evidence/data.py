"""Canonical synthetic corpus adapter for evidence training and evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.physics.numpy_backend import velocity_from_ai
from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from cup.synthetic.schemas import STRUCTURED_ARTIFACT_VERSION

from evidence.augmentation import (
    ObservationAugmentationProfile,
    apply_observation_augmentation,
    stable_random_identity,
)
from evidence.contracts import (
    EvidenceInput,
    EvidenceTargetContract,
    zone_linear_lateral_support,
)
from evidence.features import (
    build_evidence_targets,
    build_lfm_anchor,
    lfm_residual_from_anchor,
)


CORPUS_MANIFEST_SCHEMA = "structured_synthetic_corpus_v2"
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
    schema = manifest.get("schema_version")
    if schema != CORPUS_MANIFEST_SCHEMA:
        raise ValueError(
            f"evidence training requires {CORPUS_MANIFEST_SCHEMA}; got {schema!r}."
        )
    if manifest.get("status") not in {"success", "completed_with_warnings"}:
        raise ValueError("canonical corpus is not a completed publication.")
    if STRUCTURED_ARTIFACT_VERSION != 2:
        raise RuntimeError("runtime reader is not configured for artifact V2.")
    benchmark = StructuredSyntheticBenchmark(directory)
    index = benchmark.index
    required = {"realization_id", "corpus_role", "split_role"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"canonical index lacks columns: {missing}")
    splits: dict[str, tuple[str, ...]] = {}
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
        splits[split] = values
    if seen != set(index["realization_id"].astype(str)):
        raise ValueError("every canonical parent must belong to exactly one split.")
    return Corpus(root=directory, benchmark=benchmark, manifest=manifest, splits=splits)


def load_target_contract(
    path: str | Path,
    *,
    corpus: Corpus,
) -> EvidenceTargetContract:
    contract = EvidenceTargetContract.from_mapping(_read_json(Path(path)))
    benchmark = corpus.benchmark
    if (
        contract.sample_domain != benchmark.sample_domain
        or contract.sample_unit != benchmark.sample_unit
        or contract.depth_basis != benchmark.depth_basis
    ):
        raise ValueError(
            "target contract domain, unit, or depth basis differs from the corpus."
        )
    return contract


def parent_evidence_inputs(
    corpus: Corpus,
    parent_id: str,
) -> tuple[EvidenceInput, ...]:
    parent = corpus.benchmark.read_parent(parent_id)
    vp_model_mps: np.ndarray | None = None
    if parent.sample_domain == "depth":
        relation = parent.forward_context.get("ai_velocity_relation")
        if not isinstance(relation, Mapping):
            raise ValueError("depth parent lacks its AI-Vp relation.")
        lfm = np.asarray(parent.lfm, dtype=np.float64)
        finite = np.isfinite(lfm)
        vp_model_mps = np.full(lfm.shape, np.nan, dtype=np.float64)
        vp_model_mps[finite] = velocity_from_ai(
            np.exp(lfm[finite]),
            a=float(relation["a"]),
            b=float(relation["b"]),
        )
    zone_ids = sorted({str(row["zone_id"]) for row in parent.zones})
    observations: list[EvidenceInput] = []
    for zone_id in zone_ids:
        top = np.full(parent.lateral_m.size, np.nan, dtype=np.float64)
        bottom = np.full_like(top, np.nan)
        for row in parent.zones:
            if str(row["zone_id"]) == zone_id:
                trace = int(row["lateral_index"])
                top[trace] = float(row["top"])
                bottom[trace] = float(row["bottom"])
        lateral_valid = zone_linear_lateral_support(
            parent.model_axis,
            parent.observed_valid,
            top,
            bottom,
        )
        observations.append(
            EvidenceInput(
                sample_axis=parent.model_axis,
                seismic=np.where(parent.observed_valid, parent.seismic, 0.0),
                lfm=np.where(parent.observed_valid, parent.lfm, 0.0),
                observed_valid=parent.observed_valid,
                lateral_m=parent.lateral_m,
                lateral_valid=lateral_valid,
                zone_top=top,
                zone_bottom=bottom,
                vp_model_mps=vp_model_mps,
                x_m=parent.x_m,
                y_m=parent.y_m,
                identity=f"{parent_id}:{zone_id}",
            )
        )
    return tuple(observations)


def _selected_parent_ids(
    corpus: Corpus,
    split: str,
    *,
    parent_limit: int | None,
    parent_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if split not in {"training", "tuning", "calibration"}:
        raise ValueError("evidence batches require a development split.")
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
            raise ValueError(f"parent_ids are outside split {split!r}: {invalid[:5]}")
    elif parent_limit is not None:
        selected = selected[:parent_limit]
    return selected


def iter_evidence_batches(
    corpus: Corpus,
    split: str,
    *,
    condition: str = "clean",
    augmentation_profile: ObservationAugmentationProfile | None = None,
    parent_limit: int | None = None,
    parent_ids: Sequence[str] | None = None,
) -> Iterator[Mapping[str, Any]]:
    if condition not in {"clean", "dirty"}:
        raise ValueError("condition must be clean or dirty.")
    if condition == "dirty" and augmentation_profile is None:
        raise ValueError("dirty evidence requires an augmentation profile.")
    selected = _selected_parent_ids(
        corpus,
        split,
        parent_limit=parent_limit,
        parent_ids=parent_ids,
    )
    index_by_parent = corpus.benchmark.index.set_index("realization_id", drop=False)
    for parent_id in selected:
        if parent_id not in index_by_parent.index:
            raise ValueError(f"canonical index cannot resolve parent {parent_id!r}.")
        index_row = index_by_parent.loc[parent_id]
        if getattr(index_row, "ndim", 1) != 1:
            raise ValueError(f"canonical index contains duplicate parent {parent_id!r}.")
        parent = corpus.benchmark.read_parent(parent_id)
        for observation in parent_evidence_inputs(corpus, parent_id):
            anchor = build_lfm_anchor(observation)
            lfm_residual = lfm_residual_from_anchor(observation, anchor)
            targets = build_evidence_targets(
                observation,
                model_log_ai=parent.model_log_ai,
                anchor=anchor,
            )
            training_input = observation
            if condition == "dirty":
                random_identity = stable_random_identity(
                    parent_id,
                    observation.identity,
                    "dirty",
                )
                augmented = apply_observation_augmentation(
                    observation.seismic,
                    observation.observed_valid,
                    profile=augmentation_profile,
                    rng=np.random.default_rng(random_identity),
                    relative_lateral_m=observation.lateral_m
                    - float(np.median(observation.lateral_m)),
                )
                training_input = replace(
                    observation,
                    seismic=augmented.seismic,
                    observed_valid=augmented.observed_valid,
                )
            yield {
                "parent_id": parent_id,
                "geometry_family": str(index_row["geometry_family"]),
                "input_id": observation.identity,
                "condition": condition,
                "seismic": training_input.seismic[None].astype(np.float32),
                "lfm": training_input.lfm[None].astype(np.float32),
                "lfm_residual": lfm_residual[None].astype(np.float32),
                "background_lfm_linear": anchor.values[None].astype(np.float32),
                "observed_valid": training_input.observed_valid[None],
                "lateral_m": training_input.lateral_m[None].astype(np.float32),
                "lateral_valid": training_input.lateral_valid[None],
                "projected_log_ai_increment": targets.projected_log_ai_increment[
                    None
                ].astype(np.float32),
                "signed_reflectivity": targets.signed_reflectivity[None].astype(
                    np.float32
                ),
                "support": targets.support[None],
            }


def iter_paired_evidence_batches(
    corpus: Corpus,
    split: str,
    *,
    augmentation_profile: ObservationAugmentationProfile,
    parent_limit: int | None = None,
) -> Iterator[Mapping[str, Any]]:
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


__all__ = [
    "Corpus",
    "iter_evidence_batches",
    "iter_paired_evidence_batches",
    "load_corpus",
    "load_target_contract",
    "parent_evidence_inputs",
]
