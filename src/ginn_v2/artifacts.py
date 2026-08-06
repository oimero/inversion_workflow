"""Canonical corpus and checkpoint publication for Structured GINN V2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from cup.synthetic.schemas import STRUCTURED_ARTIFACT_VERSION

from ginn_v2.contracts import InputContractError, ObservationTile


CORPUS_MANIFEST_SCHEMA = "structured_synthetic_corpus_v2"
CHECKPOINT_SCHEMA = "structured_ginn_v2_checkpoint_v7"
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
    if STRUCTURED_ARTIFACT_VERSION != 2:
        raise RuntimeError("runtime reader is not configured for canonical artifact V2.")
    benchmark = StructuredSyntheticBenchmark(directory)
    index = benchmark.index
    required = {"realization_id", "corpus_role", "split_role"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise InputContractError(f"canonical V2 index lacks columns: {missing}")
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
    return Corpus(directory, benchmark, manifest, split_map)


def parent_observation_tiles(corpus: Corpus, parent_id: str) -> tuple[ObservationTile, ...]:
    """Adapt one canonical parent into zone tiles with explicit axes and masks."""

    parent = corpus.benchmark.read_parent(parent_id)
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
                identity=f"{parent_id}:{zone_id}",
            )
        )
    return tuple(tiles)


def save_checkpoint(
    path: str | Path,
    *,
    model_state: Mapping[str, Any],
    model_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
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
    temporary = target.with_suffix(target.suffix + ".staging")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load only the current checkpoint contract."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise InputContractError("checkpoint does not use the current V2 schema.")
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
    "SPLIT_NAMES",
    "load_checkpoint",
    "load_corpus",
    "parent_observation_tiles",
    "public_checkpoint_metadata",
    "save_checkpoint",
]
