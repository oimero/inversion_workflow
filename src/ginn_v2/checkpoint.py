"""Per-epoch checkpoint persistence for GINN V2 body inversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ginn_v2.evaluation import EvaluationMetrics, GateReport
from ginn_v2.model import BodyNetworkConfig


CHECKPOINT_SCHEMA = "ginn_v2_body_inversion_checkpoint_v1"


def save_epoch_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    trial_id: int,
    network_config: BodyNetworkConfig,
    run_config: Mapping[str, Any],
    split_description: Mapping[str, Any],
    metrics: EvaluationMetrics,
    gate: GateReport,
) -> Path:
    """Write a self-contained recovery checkpoint, including optimizer state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(epoch, bool) or int(epoch) <= 0 or isinstance(trial_id, bool) or int(trial_id) <= 0:
        raise ValueError("epoch and trial_id must be positive integers.")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": int(epoch),
        "trial_id": int(trial_id),
        "network_config": dict(network_config.__dict__),
        "run_config": dict(run_config),
        "split": dict(split_description),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics.to_json_dict(),
        "gate": gate.to_json_dict(),
    }
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    expected_network_config: BodyNetworkConfig | None = None,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer state without changing contracts."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported GINN V2 checkpoint schema: {source}")
    required = {"epoch", "trial_id", "network_config", "run_config", "split", "model_state", "optimizer_state", "metrics", "gate"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"GINN V2 checkpoint is missing fields: {missing}")
    recorded_config = BodyNetworkConfig(**dict(payload["network_config"]))
    if expected_network_config is not None and recorded_config != expected_network_config:
        raise ValueError("Checkpoint network contract differs from the requested network configuration.")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return dict(payload)


__all__ = ["CHECKPOINT_SCHEMA", "load_checkpoint", "save_epoch_checkpoint"]
