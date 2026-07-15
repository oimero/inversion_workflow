"""Strict configuration contracts for composable GINN-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping

from cup.impedance import validate_increment_contract
from ginn_v2.contracts import EXPERIMENT_SCHEMA_VERSION
from ginn_v2.models import ARCHITECTURE_IDS


SOURCE_KINDS = {"synthoseis_lite", "real_field", "real_wells"}
LOSS_SOURCE_KINDS = {
    "synthetic_supervised": {"synthoseis_lite"},
    "physics": {"synthoseis_lite", "real_field"},
    "real_well_supervised": {"real_wells"},
}
SOURCE_KEYS = {
    "synthoseis_lite": {
        "kind", "benchmark_dir", "input_seismic_variant", "physics_target_variant",
        "max_patches",
    },
    "real_field": {
        "kind", "lfm_run_dir", "variant_id", "well_control_run_dir",
        "model_input_seismic_transform", "physics_target_seismic_transform",
        "validation_split", "wavelet_generation_dir", "forward_model_inputs_path",
        "volume", "segy_options",
    },
    "real_wells": {
        "kind", "field_source", "well_control_run_dir", "held_out_well",
        "exclude_same_cluster", "cluster_radius_m", "diagnostic_max_hz",
        "reconstruction_tolerance_log_ai",
    },
}
LOSS_KEYS = {
    "synthetic_supervised": {
        "block_id", "kind", "source", "weight", "update_interval", "batch_size",
        "min_valid_samples", "sampling",
    },
    "physics": {
        "block_id", "kind", "source", "weight", "update_interval", "batch_size",
        "min_valid_samples", "increment_l2_weight", "waveform_standardization",
        "centered_rms_epsilon", "min_centered_rms", "sampling",
    },
    "real_well_supervised": {
        "block_id", "kind", "source", "weight", "update_interval", "batch_size",
        "min_valid_samples",
    },
}
VALIDATION_METRICS = {
    "synthetic_supervised": {"mse"},
    "physics": {"waveform_mse", "increment_l2", "total"},
    "real_well_supervised": {"mse"},
}
# Failure-only vocabulary: these names are rejected at the new-schema boundary;
# no compatibility or fallback execution path uses them.
LEGACY_KEYS = {
    "train", "model_id", "model_role", "min_valid_fraction",
    "lambda_physics", "lambda_real_delta", "real_delta", "delta_l2_weight",
    "physical_delta_log_ai", "pred_delta_log_ai", "target_delta",
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
    increment_contract: dict[str, Any]
    run_mode: str
    development_limited: bool
    validation_semantics: str
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


def _stage_deployment_eligibility(
    stage: Mapping[str, Any], *, development_limited: bool,
) -> tuple[bool, str]:
    """Return whether a stage checkpoint may be used by R0.

    Physics is deliberately conservative: a physics stage is deployable only
    when the same stage retains dense synthetic increment supervision and the
    selection metric is that supervised MSE.  A waveform-only physics stage is
    still useful as a diagnostic, but it must not silently become a model
    deployment choice.  Sparse real-well supervision follows the same rule and
    remains experimental when combined with physics.
    """
    if development_limited:
        return False, "development_limited_run"
    blocks = list(stage.get("loss_blocks") or [])
    kinds = {str(block.get("kind") or "") for block in blocks}
    if "physics" not in kinds:
        return True, "supervised_stage"
    synthetic_blocks = [
        block
        for block in blocks
        if str(block.get("kind") or "") == "synthetic_supervised"
        and float(block.get("weight", 0.0)) > 0.0
        and int(block.get("update_interval", 0)) == 1
    ]
    if not synthetic_blocks:
        if "real_well_supervised" in kinds:
            return False, "real_well_supervised_plus_physics_experimental"
        return False, "physics_diagnostic_only"
    metric = str(dict(stage.get("validation") or {}).get("selection_metric") or "")
    synthetic_ids = {str(block.get("block_id") or "") for block in synthetic_blocks}
    if metric.split(".", 1)[0] not in synthetic_ids or not metric.endswith(".mse"):
        return False, "physics_requires_dense_synthetic_mse_selection"
    if "real_well_supervised" in kinds:
        return False, "real_well_supervised_plus_physics_experimental"
    return True, "dense_synthetic_supervised_mse_with_physics"


def parse_experiment_config(payload: Mapping[str, Any]) -> ExperimentConfig:
    root = dict(payload)
    if "ginn_v2" not in root:
        legacy = sorted(set(root).intersection(LEGACY_KEYS))
        detail = f" Legacy fields found: {legacy}." if legacy else ""
        raise ValueError(
            "GINN-v2 requires the ginn_v2_experiment_v2 configuration root 'ginn_v2'."
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
        "device", "increment_contract", "run_mode", "development_limited",
        "validation_semantics",
    }, "ginn_v2")
    schema = str(cfg.get("schema_version") or EXPERIMENT_SCHEMA_VERSION)
    if schema != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported GINN-v2 experiment schema {schema!r}; expected {EXPERIMENT_SCHEMA_VERSION}.")
    increment_contract = validate_increment_contract(
        _mapping(cfg.get("increment_contract"), "ginn_v2.increment_contract")
    ).as_dict()
    run_mode = str(cfg.get("run_mode") or "standard").strip().lower()
    if run_mode not in {"standard", "smoke"}:
        raise ValueError("ginn_v2.run_mode must be standard or smoke.")
    development_limited = cfg.get("development_limited", run_mode == "smoke")
    if not isinstance(development_limited, bool):
        raise ValueError("ginn_v2.development_limited must be boolean.")
    if run_mode == "smoke" and not development_limited:
        raise ValueError("ginn_v2 smoke runs must set development_limited=true.")
    validation_semantics = str(
        cfg.get("validation_semantics")
        or ("duplicated_training_patch" if run_mode == "smoke" else "independent_parent")
    ).strip()
    if validation_semantics not in {"independent_parent", "duplicated_training_patch"}:
        raise ValueError(
            "ginn_v2.validation_semantics must be independent_parent or "
            "duplicated_training_patch."
        )
    if run_mode == "standard" and validation_semantics == "duplicated_training_patch":
        raise ValueError(
            "duplicated_training_patch validation is only allowed in run_mode=smoke."
        )
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
        _reject_extra(source, SOURCE_KEYS[kind], f"sources.{source_id}")
        required_by_kind = {
            "synthoseis_lite": {"benchmark_dir"},
            "real_field": {"lfm_run_dir", "variant_id"},
            "real_wells": {"field_source", "well_control_run_dir", "held_out_well"},
        }
        missing = sorted(required_by_kind[kind] - set(source))
        if missing:
            raise ValueError(f"sources.{source_id} is missing required fields: {missing}")
        if kind == "synthoseis_lite" and source.get("physics_target_variant", "model_consistent") != "model_consistent":
            raise ValueError("Synthoseis physics_target_variant must be model_consistent.")
        if kind == "synthoseis_lite" and source.get("max_patches") is not None:
            source["max_patches"] = _positive_int(
                source["max_patches"],
                f"sources.{source_id}.max_patches",
            )
        if kind == "real_field":
            split = _mapping(source.get("validation_split"), f"sources.{source_id}.validation_split")
            _reject_extra(split, {"kind", "fraction", "gap_m", "anchor"}, f"sources.{source_id}.validation_split")
            if split.get("kind") != "spatial_block" or split.get("anchor") != "high_inline_tail":
                raise ValueError(
                    f"sources.{source_id}.validation_split requires "
                    "kind=spatial_block and anchor=high_inline_tail."
                )
            split["fraction"] = _finite(split.get("fraction"), f"sources.{source_id}.validation_split.fraction", positive=True)
            if split["fraction"] >= 1.0:
                raise ValueError(f"sources.{source_id}.validation_split.fraction must be below one.")
            split["gap_m"] = _finite(split.get("gap_m", 0.0), f"sources.{source_id}.validation_split.gap_m", nonnegative=True)
            source["validation_split"] = split
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
    stage_has_supervised_ancestor: dict[str, bool] = {}
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
            raise ValueError("GINN-v2 v2 only supports optimizer.kind=adamw.")
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
            _reject_extra(block, LOSS_KEYS[kind], f"stages[{stage_index}].loss_blocks[{block_index}]")
            if sources[source_id]["kind"] not in LOSS_SOURCE_KINDS[kind]:
                raise ValueError(f"Loss block {block_id} cannot consume source kind {sources[source_id]['kind']}.")
            block["weight"] = _finite(block.get("weight"), f"{block_id}.weight", nonnegative=True)
            for key in ("update_interval", "batch_size", "min_valid_samples"):
                block[key] = _positive_int(block.get(key), f"{block_id}.{key}")
            if kind in {"synthetic_supervised", "physics"}:
                if sources[source_id]["kind"] == "synthoseis_lite":
                    sampling = _mapping(
                        block.get("sampling") or {"kind": "uniform_patch"},
                        f"{block_id}.sampling",
                    )
                    _reject_extra(sampling, {"kind"}, f"{block_id}.sampling")
                    sampling_kind = str(sampling.get("kind") or "")
                    if sampling_kind not in {"uniform_patch", "balanced_sample_kind"}:
                        raise ValueError(
                            f"{block_id}.sampling.kind must be uniform_patch or "
                            "balanced_sample_kind."
                        )
                    block["sampling"] = {"kind": sampling_kind}
                    if sampling_kind == "balanced_sample_kind":
                        block["sampling"]["groups"] = {
                            "base": 0.5,
                            "seismic_variant": 0.5,
                        }
                elif "sampling" in block:
                    raise ValueError(
                        f"{block_id}.sampling is only supported for synthoseis_lite blocks."
                    )
            if kind == "physics":
                block["increment_l2_weight"] = _finite(
                    block.get("increment_l2_weight"),
                    f"{block_id}.increment_l2_weight",
                    positive=True,
                )
                source_kind = str(sources[source_id]["kind"])
                expected_standardization = "masked_centered_rms" if source_kind == "real_field" else "raw"
                actual_standardization = str(block.get("waveform_standardization") or expected_standardization)
                if actual_standardization != expected_standardization:
                    raise ValueError(
                        f"{block_id}.waveform_standardization must be "
                        f"{expected_standardization!r} for {source_kind}."
                    )
                block["waveform_standardization"] = actual_standardization
                block["centered_rms_epsilon"] = _finite(
                    block.get("centered_rms_epsilon", 1e-12),
                    f"{block_id}.centered_rms_epsilon", positive=True,
                )
                block["min_centered_rms"] = _finite(
                    block.get("min_centered_rms", 1e-6),
                    f"{block_id}.min_centered_rms", positive=True,
                )
            block_ids.add(block_id)
            blocks.append(block)
        stage_has_physics = any(block["kind"] == "physics" for block in blocks)
        stage_has_current_supervised = any(
            block["kind"] in {"synthetic_supervised", "real_well_supervised"}
            and block["weight"] > 0.0
            for block in blocks
        )
        if stage_has_physics:
            if initialize == "zero":
                raise ValueError(
                    f"Stage {stage_id} contains physics but initialize_from=zero; "
                    "physics must start from an earlier supervised checkpoint."
                )
            parent_stage = initialize.rsplit(".", 1)[0]
            if not stage_has_supervised_ancestor.get(parent_stage, False):
                raise ValueError(
                    f"Stage {stage_id} contains physics but its checkpoint lineage has no "
                    "completed synthetic_supervised or real_well_supervised stage."
                )
        if not any(block["update_interval"] == 1 and block["weight"] > 0 for block in blocks):
            raise ValueError(f"Stage {stage_id} requires a positive-weight loss block with update_interval=1.")
        validation = _mapping(stage.get("validation"), f"stages[{stage_index}].validation")
        metric = str(validation.get("selection_metric") or "")
        metric_parts = metric.split(".", 1)
        if len(metric_parts) != 2 or metric_parts[0] not in block_ids:
            raise ValueError(f"Stage {stage_id} selection_metric must reference one of its loss blocks.")
        metric_block = next(block for block in blocks if block["block_id"] == metric_parts[0])
        if metric_parts[1] not in VALIDATION_METRICS[str(metric_block["kind"])]:
            raise ValueError(
                f"Stage {stage_id} selection_metric {metric!r} is not exported by "
                f"{metric_block['kind']}."
            )
        mode = str(validation.get("mode") or "full")
        if mode != "full":
            raise ValueError(
                f"Stage {stage_id} validation.mode must be full; v2 validation never resamples."
            )
        if "steps" in validation:
            raise ValueError("validation.steps is not supported; validation is a full deterministic pass.")
        validation["mode"] = "full"
        stage.update({
            "epochs": _positive_int(stage.get("epochs"), f"{stage_id}.epochs"),
            "steps_per_epoch": _positive_int(stage.get("steps_per_epoch"), f"{stage_id}.steps_per_epoch"),
            "optimizer": optimizer, "loss_blocks": blocks, "validation": validation,
            "initialize_from": initialize,
        })
        stage["physics_closures"] = []
        for block in blocks:
            if str(block["kind"]) != "physics":
                continue
            source_config = sources[str(block["source"])]
            source_kind = str(source_config["kind"])
            stage["physics_closures"].append({
                "block_id": str(block["block_id"]),
                "source": str(block["source"]),
                "source_kind": source_kind,
                "closure": (
                    "canonical_closure"
                    if source_kind == "synthoseis_lite"
                    else "deployment_closure"
                ),
            })
        parent_stage = initialize.rsplit(".", 1)[0] if initialize != "zero" else ""
        stage_has_supervised_ancestor[stage_id] = bool(
            stage_has_current_supervised
            or stage_has_supervised_ancestor.get(parent_stage, False)
        )
        eligible, reason = _stage_deployment_eligibility(
            stage, development_limited=development_limited
        )
        stage["deployment_eligible"] = bool(eligible)
        stage["deployment_eligibility_reason"] = reason
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
    selected_stage = next(stage for stage in stages if stage["stage_id"] == deployment_stage)
    if deployment_kind == "final" and any(
        str(block.get("kind") or "") == "physics"
        for block in selected_stage.get("loss_blocks", [])
    ):
        raise ValueError(
            f"deployment_checkpoint={deployment!r} is not deployment eligible: "
            "physics stages may deploy only their supervised-metric best checkpoint."
        )
    # Development-limited runs intentionally may select a diagnostic checkpoint;
    # the manifest and checkpoint carry deployment_eligible=false and R0 rejects
    # them.  Standard runs must select a stage whose selection contract is safe
    # for deployment.
    if not bool(selected_stage.get("deployment_eligible", False)) and not development_limited:
        raise ValueError(
            f"deployment_checkpoint={deployment!r} is not deployment eligible: "
            f"{selected_stage.get('deployment_eligibility_reason', 'unknown')}"
        )
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
        increment_contract=increment_contract,
        run_mode=run_mode,
        development_limited=development_limited,
        validation_semantics=validation_semantics,
        raw=cfg,
    )


def resolve_source_path(value: Any, *, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


__all__ = [
    "ArchitectureConfig", "ExperimentConfig", "LOSS_SOURCE_KINDS", "PatchingConfig",
    "SOURCE_KINDS", "parse_experiment_config", "resolve_source_path",
]
