"""GINN-v2 canonical-increment checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ginn_v2.contracts import CHECKPOINT_SCHEMA_VERSION
from ginn_v2.models import build_model


def load_checkpoint(
    path: Path,
    *,
    hidden_channels: int | None = None,
    depth: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("schema_version") or "") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GINN-v2 checkpoint schema in {path}; expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    if str(checkpoint.get("output_semantics") or "") != "predicted_increment_log_ai":
        raise ValueError(
            f"GINN-v2 checkpoint in {path} does not use predicted_increment_log_ai semantics."
        )
    if list(checkpoint.get("input_channels") or []) != [
        "seismic", "input_lfm_log_ai", "valid_mask"
    ]:
        raise ValueError(
            f"GINN-v2 checkpoint in {path} has a non-canonical input channel contract."
        )
    architecture = dict(checkpoint.get("architecture") or {})
    architecture_id = str(architecture.get("id") or "")
    if not architecture_id:
        raise ValueError("GINN-v2 checkpoint lacks the canonical architecture contract.")
    model, _ = build_model(
        architecture_id,
        hidden_channels=int(hidden_channels or architecture.get("hidden_channels", 32)),
        depth=int(depth or architecture.get("depth", 5)),
        lateral_kernel=architecture.get("lateral_kernel"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


__all__ = ["load_checkpoint"]
