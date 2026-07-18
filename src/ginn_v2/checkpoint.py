"""GINN-v2 canonical-increment checkpoint loading."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from cup.impedance import validate_increment_contract
from ginn_v2.contracts import CHECKPOINT_SCHEMA_VERSION
from ginn_v2.models import build_model


def _validate_probability_contract(
    value: object, *, label: str, allow_empty: bool = False,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"GINN-v2 checkpoint {label} must be a mapping.")
    result = {str(key): float(item) for key, item in value.items()}
    if not result and allow_empty:
        return result
    if not result or any(not math.isfinite(item) or item < 0.0 for item in result.values()):
        raise ValueError(f"GINN-v2 checkpoint {label} contains invalid weights.")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"GINN-v2 checkpoint {label} must sum to one.")
    return result


def _validate_parent_view_sampling(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"GINN-v2 checkpoint {label} must be a mapping.")
    parent = _validate_probability_contract(value.get("parent_weights"), label=f"{label}.parent_weights")
    if set(parent) != {"base", "variant"}:
        raise ValueError(f"GINN-v2 checkpoint {label} parent weights must contain base and variant.")
    raw_view = value.get("view_weights")
    if not isinstance(raw_view, dict) or "base" in raw_view:
        raise ValueError(f"GINN-v2 checkpoint {label} has invalid view weights.")
    view = {str(key): float(item) for key, item in raw_view.items()}
    if parent["variant"] > 0.0:
        if not view or any(not math.isfinite(item) or item <= 0.0 for item in view.values()):
            raise ValueError(f"GINN-v2 checkpoint {label} requires positive view weights.")
        if not math.isclose(sum(view.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"GINN-v2 checkpoint {label} view weights must sum to one.")
    elif view:
        raise ValueError(f"GINN-v2 checkpoint {label} must not contain views for a zero variant weight.")


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
    for key in (
        "benchmark_identity",
        "synthetic_sampling_contract",
        "split_assignment_contract",
        "normalization_identity",
        "sampling_statistics",
    ):
        if key not in checkpoint:
            raise ValueError(f"GINN-v2 checkpoint in {path} lacks {key} provenance.")
    if not isinstance(checkpoint.get("benchmark_identity"), dict):
        raise ValueError(f"GINN-v2 checkpoint in {path} has invalid benchmark identity provenance.")
    for source_id, identity in checkpoint["benchmark_identity"].items():
        if not isinstance(identity, dict):
            raise ValueError(
                f"GINN-v2 checkpoint in {path} has malformed benchmark identity for {source_id!r}."
            )
        required_identity = {
            "benchmark_dir", "schema_version", "sample_domain", "science_revision",
            "projection_contract_version", "seismic_view_contract_version",
            "seismic_operator_contract_version", "random_stream_contract_version",
            "contract_fingerprint_sha256", "materialized_view_ids",
            "materialized_view_spec_hashes", "split_assignment_path",
            "split_assignment_sha256",
        }
        if not required_identity.issubset(identity):
            raise ValueError(
                f"GINN-v2 checkpoint in {path} has incomplete benchmark identity for {source_id!r}."
            )
        if str(identity.get("schema_version")) != "synthoseis_lite_v5":
            raise ValueError(
                f"GINN-v2 checkpoint in {path} references a non-v5 benchmark source {source_id!r}."
            )
    sampling_contract = checkpoint.get("synthetic_sampling_contract")
    if not isinstance(sampling_contract, list):
        raise ValueError(f"GINN-v2 checkpoint in {path} has invalid synthetic sampling provenance.")
    for item in sampling_contract:
        if not isinstance(item, dict):
            raise ValueError(f"GINN-v2 checkpoint in {path} has a malformed synthetic sampling contract.")
        _validate_parent_view_sampling(
            {
                "parent_weights": item.get("parent_weights"),
                "view_weights": item.get("view_weights"),
            },
            label=f"synthetic_sampling_contract[{item.get('block_id', '')}]",
        )
        if not isinstance(item.get("validation"), dict):
            raise ValueError(f"GINN-v2 checkpoint in {path} lacks validation weight provenance.")
        validation_views = dict(item["validation"].get("seismic_views") or {})
        if not validation_views:
            raise ValueError(f"GINN-v2 checkpoint in {path} lacks validation seismic-view provenance.")
        _validate_parent_view_sampling(
            validation_views,
            label=f"synthetic_sampling_contract[{item.get('block_id', '')}].validation.seismic_views",
        )
    if not isinstance(checkpoint.get("stage_lineage"), list):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks stage_lineage provenance.")
    split_contract = checkpoint.get("split_assignment_contract")
    if not isinstance(split_contract, dict):
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks split assignment provenance.")
    required_split_fields = {
        "version", "owner", "seed", "hash_algorithm", "validation_fraction",
        "test_fraction", "geometry_holdout_role", "identity_sha256",
    }
    if not required_split_fields.issubset(split_contract):
        raise ValueError(
            f"GINN-v2 checkpoint in {path} has incomplete split assignment provenance."
        )
    if str(split_contract.get("version")) != "parent_hash_split_v1":
        raise ValueError(f"GINN-v2 checkpoint in {path} uses an unsupported split contract.")
    if not isinstance(checkpoint.get("normalization_identity"), str) or not checkpoint["normalization_identity"]:
        raise ValueError(f"GINN-v2 checkpoint in {path} lacks normalization identity provenance.")
    sampling_statistics = checkpoint.get("sampling_statistics")
    if not isinstance(sampling_statistics, dict):
        raise ValueError(f"GINN-v2 checkpoint in {path} has invalid sampling statistics provenance.")
    for key in ("kind_counts", "view_counts", "parent_counts"):
        value = sampling_statistics.get(key)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"GINN-v2 checkpoint in {path} has invalid {key} statistics.")
    if sampling_contract and not isinstance(sampling_statistics.get("history"), list):
        raise ValueError(
            f"GINN-v2 checkpoint in {path} lacks per-epoch sampling history."
        )
    if "history" in sampling_statistics and not isinstance(
        sampling_statistics["history"], list
    ):
        raise ValueError(
            f"GINN-v2 checkpoint in {path} has invalid per-epoch sampling history."
        )
    for epoch_index, epoch_stats in enumerate(sampling_statistics.get("history", [])):
        if not isinstance(epoch_stats, dict) or not {
            "epoch", "kind_counts", "view_counts", "parent_counts"
        }.issubset(epoch_stats):
            raise ValueError(
                f"GINN-v2 checkpoint in {path} has incomplete sampling history at index {epoch_index}."
            )
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
        if str(item.get("sampling", {}).get("kind") or "") == "parent_balanced_seismic_view":
            sampling = item["sampling"]
            _validate_parent_view_sampling(
                sampling,
                label=f"stage_loss_blocks[{item.get('block_id', '')}].sampling",
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
            str(checkpoint["stage_selection_metric"]) in {
                f"{block_id}.mse",
                f"{block_id}.weighted_mse",
            }
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
