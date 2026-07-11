"""Strict configuration contracts for composable GINN-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

from ginn_v2.contracts import EXPERIMENT_SCHEMA_VERSION
from ginn_v2.models import ARCHITECTURE_IDS


SOURCE_KINDS = {"synthoseis_lite", "real_field", "real_wells"}
LOSS_SOURCE_KINDS = {
    "synthetic_supervised": {"synthoseis_lite"},
    "physics": {"synthoseis_lite", "real_field"},
    "real_well_supervised": {"real_wells"},
}
LEGACY_KEYS = {
    "train", "model_id", "model_role", "min_valid_fraction",
    "lambda_physics", "lambda_real_delta", "real_delta",
}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ArchitectureConfig:
    id: str
    hidden_channels: int
    depth: int
    lateral_kernel: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {"id": self.id, "hidden_channels": self.hidden_channels, "depth": self.depth}
        if self.lateral_kernel is not None:
            result["lateral_kernel"] = self.lateral_kernel
        return result


@dataclass(frozen=True)
class PatchingConfig:
    lateral_samples: int
    vertical_samples: int
    lateral_stride: int
    vertical_stride: int

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int
    architecture: ArchitectureConfig
    sources: dict[str, dict[str, Any]]
    normalization_reference: str
    patching: PatchingConfig
    stages: tuple[dict[str, Any], ...]
    deployment_stage_id: str
    deployment_checkpoint_kind: str
    device: str
    raw: dict[str, Any]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def _finite(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        qualifier = "positive" if positive else "non-negative" if nonnegative else "finite"
        raise ValueError(f"{label} must be finite and {qualifier}.")
    return result


def _reject_extra(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ValueError(f"Unsupported keys in {label}: {extra}")


def parse_experiment_config(payload: Mapping[str, Any]) -> ExperimentConfig:
    root = dict(payload)
    if "ginn_v2" not in root:
        legacy = sorted(set(root).intersection(LEGACY_KEYS))
        detail = f" Legacy fields found: {legacy}." if legacy else ""
        raise ValueError(
            "GINN-v2 requires the ginn_v2_experiment_v1 configuration root 'ginn_v2'."
            + detail
            + " See docs/spec/GINN_V2_COMPOSABLE_TRAINING_DESIGN.md."
        )
    cfg = _mapping(root["ginn_v2"], "ginn_v2")
    legacy = sorted(set(cfg).intersection(LEGACY_KEYS))
    if legacy:
        raise ValueError(
            f"Legacy GINN-v2 fields are not accepted: {legacy}. "
            "Use architecture, sources, stages and experiment_id."
        )
    _reject_extra(cfg, {
        "schema_version", "experiment_id", "seed", "architecture", "sources",
        "normalization_reference", "patching", "stages", "deployment_checkpoint",
        "device",
    }, "ginn_v2")
    schema = str(cfg.get("schema_version") or EXPERIMENT_SCHEMA_VERSION)
    if schema != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported GINN-v2 experiment schema {schema!r}; expected {EXPERIMENT_SCHEMA_VERSION}.")
    experiment_id = str(cfg.get("experiment_id") or "").strip()
    if not _ID.fullmatch(experiment_id):
        raise ValueError("ginn_v2.experiment_id must be a non-empty directory-safe identifier.")

    arch_raw = _mapping(cfg.get("architecture"), "ginn_v2.architecture")
    _reject_extra(arch_raw, {"id", "hidden_channels", "depth", "lateral_kernel"}, "ginn_v2.architecture")
    architecture_id = str(arch_raw.get("id") or "")
    if architecture_id not in ARCHITECTURE_IDS:
        raise ValueError(f"architecture.id must be one of {list(ARCHITECTURE_IDS)}; legacy model IDs are rejected.")
    lateral_kernel = arch_raw.get("lateral_kernel")
    if architecture_id == "trace_lateral_mixer":
        lateral_kernel = 3 if lateral_kernel is None else _positive_int(lateral_kernel, "architecture.lateral_kernel")
        if lateral_kernel % 2 == 0:
            raise ValueError("architecture.lateral_kernel must be odd.")
    elif lateral_kernel is not None:
        raise ValueError(f"architecture {architecture_id} does not accept lateral_kernel.")
    architecture = ArchitectureConfig(
        architecture_id,
        _positive_int(arch_raw.get("hidden_channels", 32), "architecture.hidden_channels"),
        _positive_int(arch_raw.get("depth", 5), "architecture.depth"),
        lateral_kernel,
    )

    sources_raw = _mapping(cfg.get("sources"), "ginn_v2.sources")
    if not sources_raw:
        raise ValueError("ginn_v2.sources must not be empty.")
    sources: dict[str, dict[str, Any]] = {}
    for source_id, value in sources_raw.items():
        if not _ID.fullmatch(str(source_id)):
            raise ValueError(f"Invalid source id: {source_id!r}")
        source = _mapping(value, f"sources.{source_id}")
        kind = str(source.get("kind") or "")
        if kind not in SOURCE_KINDS:
            raise ValueError(f"sources.{source_id}.kind must be one of {sorted(SOURCE_KINDS)}.")
        required_by_kind = {
            "synthoseis_lite": {"benchmark_dir"},
            "real_field": {"lfm_run_dir", "variant_id", "well_control_run_dir"},
            "real_wells": {"field_source", "well_control_run_dir", "held_out_well"},
        }
        missing = sorted(required_by_kind[kind] - set(source))
        if missing:
            raise ValueError(f"sources.{source_id} is missing required fields: {missing}")
        if kind == "synthoseis_lite" and source.get("physics_target_variant", "model_consistent") != "model_consistent":
            raise ValueError("Synthoseis physics_target_variant must be model_consistent.")
        sources[str(source_id)] = source

    norm = _mapping(cfg.get("normalization_reference"), "ginn_v2.normalization_reference")
    _reject_extra(norm, {"source"}, "ginn_v2.normalization_reference")
    norm_source = str(norm.get("source") or "")
    if norm_source not in sources or sources[norm_source]["kind"] == "real_wells":
        raise ValueError("normalization_reference.source must reference a seismic/LFM source.")

    patch_raw = _mapping(cfg.get("patching"), "ginn_v2.patching")
    if "min_valid_fraction" in patch_raw:
        raise ValueError("patching.min_valid_fraction is obsolete; configure min_valid_samples per loss block.")
    _reject_extra(patch_raw, {"lateral_samples", "vertical_samples", "lateral_stride", "vertical_stride"}, "ginn_v2.patching")
    patching = PatchingConfig(**{
        key: _positive_int(patch_raw.get(key), f"patching.{key}")
        for key in ("lateral_samples", "vertical_samples", "lateral_stride", "vertical_stride")
    })

    stages_value = cfg.get("stages")
    if not isinstance(stages_value, list) or not stages_value:
        raise ValueError("ginn_v2.stages must be a non-empty list.")
    stages: list[dict[str, Any]] = []
    stage_ids: list[str] = []
    for stage_index, value in enumerate(stages_value):
        stage = _mapping(value, f"stages[{stage_index}]")
        _reject_extra(stage, {"stage_id", "epochs", "steps_per_epoch", "optimizer", "loss_blocks", "validation", "initialize_from"}, f"stages[{stage_index}]")
        stage_id = str(stage.get("stage_id") or "")
        if not _ID.fullmatch(stage_id) or stage_id in stage_ids:
            raise ValueError(f"Stage ID must be valid and unique: {stage_id!r}")
        initialize = str(stage.get("initialize_from") or ("zero" if stage_index == 0 else f"{stage_ids[-1]}.best"))
        if initialize != "zero":
            parts = initialize.rsplit(".", 1)
            if len(parts) != 2 or parts[0] not in stage_ids or parts[1] not in {"best", "final"}:
                raise ValueError(f"stages[{stage_index}].initialize_from must reference an earlier stage best/final.")
        optimizer = _mapping(stage.get("optimizer"), f"stages[{stage_index}].optimizer")
        _reject_extra(optimizer, {"kind", "learning_rate", "weight_decay"}, f"stages[{stage_index}].optimizer")
        if optimizer.get("kind") != "adamw":
            raise ValueError("GINN-v2 v1 only supports optimizer.kind=adamw.")
        optimizer["learning_rate"] = _finite(optimizer.get("learning_rate"), "optimizer.learning_rate", positive=True)
        optimizer["weight_decay"] = _finite(optimizer.get("weight_decay", 0.0), "optimizer.weight_decay", nonnegative=True)
        blocks_value = stage.get("loss_blocks")
        if not isinstance(blocks_value, list) or not blocks_value:
            raise ValueError(f"stages[{stage_index}].loss_blocks must be non-empty.")
        blocks: list[dict[str, Any]] = []
        block_ids: set[str] = set()
        for block_index, raw_block in enumerate(blocks_value):
            block = _mapping(raw_block, f"stages[{stage_index}].loss_blocks[{block_index}]")
            block_id = str(block.get("block_id") or "")
            kind = str(block.get("kind") or "")
            source_id = str(block.get("source") or "")
            if not _ID.fullmatch(block_id) or block_id in block_ids:
                raise ValueError(f"Loss block ID must be valid and unique in its stage: {block_id!r}")
            if kind not in LOSS_SOURCE_KINDS or source_id not in sources:
                raise ValueError(f"Invalid loss block kind/source: {kind!r}/{source_id!r}")
            if sources[source_id]["kind"] not in LOSS_SOURCE_KINDS[kind]:
                raise ValueError(f"Loss block {block_id} cannot consume source kind {sources[source_id]['kind']}.")
            block["weight"] = _finite(block.get("weight"), f"{block_id}.weight", nonnegative=True)
            for key in ("update_interval", "batch_size", "min_valid_samples"):
                block[key] = _positive_int(block.get(key), f"{block_id}.{key}")
            if kind == "physics":
                block["delta_l2_weight"] = _finite(block.get("delta_l2_weight"), f"{block_id}.delta_l2_weight", positive=True)
            block_ids.add(block_id)
            blocks.append(block)
        if not any(block["update_interval"] == 1 for block in blocks):
            raise ValueError(f"Stage {stage_id} requires at least one loss block with update_interval=1.")
        validation = _mapping(stage.get("validation"), f"stages[{stage_index}].validation")
        metric = str(validation.get("selection_metric") or "")
        if metric.split(".", 1)[0] not in block_ids:
            raise ValueError(f"Stage {stage_id} selection_metric must reference one of its loss blocks.")
        mode = str(validation.get("mode") or "")
        if mode not in {"full", "fixed_steps"}:
            raise ValueError(f"Stage {stage_id} validation.mode must be full or fixed_steps.")
        if mode == "full" and "steps" in validation:
            raise ValueError("validation.steps is not allowed for mode=full.")
        if mode == "fixed_steps":
            validation["steps"] = _positive_int(validation.get("steps"), "validation.steps")
        stage.update({
            "epochs": _positive_int(stage.get("epochs"), f"{stage_id}.epochs"),
            "steps_per_epoch": _positive_int(stage.get("steps_per_epoch"), f"{stage_id}.steps_per_epoch"),
            "optimizer": optimizer, "loss_blocks": blocks, "validation": validation,
            "initialize_from": initialize,
        })
        stages.append(stage)
        stage_ids.append(stage_id)

    deployment = str(cfg.get("deployment_checkpoint") or "last_stage.best")
    if deployment == "last_stage.best":
        deployment_stage, deployment_kind = stage_ids[-1], "best"
    else:
        parts = deployment.rsplit(".", 1)
        if len(parts) != 2 or parts[0] not in stage_ids or parts[1] not in {"best", "final"}:
            raise ValueError("deployment_checkpoint must reference a stage best/final.")
        deployment_stage, deployment_kind = parts
    return ExperimentConfig(
        experiment_id=experiment_id,
        seed=int(cfg.get("seed", 20260617)),
        architecture=architecture,
        sources=sources,
        normalization_reference=norm_source,
        patching=patching,
        stages=tuple(stages),
        deployment_stage_id=deployment_stage,
        deployment_checkpoint_kind=deployment_kind,
        device=str(cfg.get("device") or "auto"),
        raw=cfg,
    )


def resolve_source_path(value: Any, *, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


__all__ = [
    "ArchitectureConfig", "ExperimentConfig", "LOSS_SOURCE_KINDS", "PatchingConfig",
    "SOURCE_KINDS", "parse_experiment_config", "resolve_source_path",
]
