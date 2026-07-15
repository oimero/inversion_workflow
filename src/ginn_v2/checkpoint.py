"""GINN-v2 canonical-increment checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cup.impedance import validate_increment_contract
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
    checkpoint["increment_contract"] = validate_increment_contract(
        checkpoint.get("increment_contract") or {}
    ).as_dict()
    if not isinstance(checkpoint.get("training_sources"), dict):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks training_sources provenance.")
    if not isinstance(checkpoint.get("stage_lineage"), list):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks stage_lineage provenance.")
    run_mode = str(checkpoint.get("run_mode") or "")
    if run_mode not in {"standard", "smoke"}:
        raise ValueError(
            f"GINN-v2 checkpoint in {path} lacks a valid run_mode (standard or smoke)."
        )
    development_limited = checkpoint.get("development_limited")
    if not isinstance(development_limited, bool):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks development_limited metadata.")
    deployment_eligible = checkpoint.get("deployment_eligible")
    if not isinstance(deployment_eligible, bool):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks deployment_eligible metadata.")
    stage_deployment_eligible = checkpoint.get("stage_deployment_eligible")
    if not isinstance(stage_deployment_eligible, bool):
        raise ValueError(
            f"GINN-v2 checkpoint in {path} lacks stage_deployment_eligible metadata."
        )
    if deployment_eligible != stage_deployment_eligible:
        raise ValueError(
            f"GINN-v2 checkpoint in {path} has inconsistent stage/deployment eligibility metadata."
        )
    if not isinstance(checkpoint.get("stage_deployment_eligibility_reason"), str):
        raise ValueError(
            f"GINN-v2 checkpoint in {path} lacks stage_deployment_eligibility_reason metadata."
        )
    if not isinstance(checkpoint.get("physics_closures"), list):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks physics_closures metadata.")
    stage_loss_blocks = checkpoint.get("stage_loss_blocks")
    if not isinstance(stage_loss_blocks, list):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks stage_loss_blocks metadata.")
    for item in stage_loss_blocks:
        if not isinstance(item, dict) or "weight" not in item or "update_interval" not in item:
            raise ValueError(
                f"GINN-v2 checkpoint in {path} has incomplete stage_loss_blocks metadata; "
                "weight and update_interval are required."
            )
    if not isinstance(checkpoint.get("stage_selection_metric"), str):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks stage_selection_metric metadata.")
    stage_kinds = {str(item.get("kind") or "") for item in stage_loss_blocks}
    if "physics" in stage_kinds:
        dense_ids = {
            str(item.get("block_id") or "")
            for item in stage_loss_blocks
            if str(item.get("kind") or "") == "synthetic_supervised"
            and float(item.get("weight", 0.0)) > 0.0
            and int(item.get("update_interval", 0)) == 1
        }
        safe_physics_selection = any(
            str(checkpoint["stage_selection_metric"]) == f"{block_id}.mse"
            for block_id in dense_ids
        ) and "real_well_supervised" not in stage_kinds
        expected_eligible = safe_physics_selection and str(checkpoint.get("checkpoint_kind")) == "best"
        if deployment_eligible != expected_eligible:
            raise ValueError(
                f"GINN-v2 checkpoint in {path} has invalid physics deployment eligibility; "
                "waveform-only and real-well physics checkpoints are diagnostic/experimental."
            )
    if run_mode == "smoke" and not development_limited:
        raise ValueError(
            f"GINN-v2 checkpoint in {path} marks a smoke run as not development_limited."
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
