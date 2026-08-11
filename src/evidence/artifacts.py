"""Checkpoint publication for the evidence model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from evidence.network import EvidenceModel


CHECKPOINT_SCHEMA = "bandlimited_evidence_checkpoint_v1"


def save_checkpoint(
    path: str | Path,
    model: EvidenceModel,
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
        "model": model.state_dict_payload(),
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
) -> tuple[EvidenceModel, Mapping[str, Any]]:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported evidence checkpoint.")
    model = EvidenceModel.from_payload(payload["model"], device=device)
    metadata = {
        "training_state": dict(payload.get("training_state") or {}),
        "corpus_provenance": dict(payload.get("corpus_provenance") or {}),
        "target_contract": model.target_contract.to_mapping(),
        "runtime_state": dict(payload.get("runtime_state") or {}),
    }
    return model, metadata


def public_checkpoint_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(metadata.get("runtime_state") or {})
    return {
        "training_state": dict(metadata.get("training_state") or {}),
        "corpus_provenance": dict(metadata.get("corpus_provenance") or {}),
        "target_contract": dict(metadata.get("target_contract") or {}),
        "runtime_state": {
            key: runtime[key]
            for key in ("epoch", "best_tuning_loss", "best_epoch", "history")
            if key in runtime
        },
    }


__all__ = [
    "load_checkpoint",
    "public_checkpoint_metadata",
    "save_checkpoint",
]
