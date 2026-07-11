"""Train, predict, and report GINN-v2 model-ablation baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
src_text = str(SRC_DIR)
sys.path = [path for path in sys.path if path != src_text]
sys.path.insert(0, src_text)

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.synthetic.contracts import REPORT_SCHEMA_VERSION as SYNTHOSEIS_REPORT_SCHEMA_VERSION
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    load_yaml_config,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    write_json,
)
from ginn_v2.contracts import (
    ABLATION_REPORT_CARD_SCHEMA_VERSION,
    ABLATION_SUMMARY_SCHEMA_VERSION,
    INPUT_REFERENCE_STATS_SCHEMA_VERSION,
    MODEL_RUN_SCHEMA_VERSION,
    PATCH_SMOKE_REPORT_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
)
from ginn_v2.data import (
    PatchSpec,
    build_patch_index,
    compute_input_reference_stats,
    compute_normalization,
    default_eval_kinds,
    default_train_kinds,
)
from ginn_v2.models import build_model
from ginn_v2.real_delta import evaluate_real_wells, prepare_real_delta_support
from ginn_v2.training import (
    configure_training_logger,
    load_checkpoint,
    predict_patches,
    report_predictions,
    resolve_device,
    train_model,
)
from ginn_v2.composable import run_experiment
from ginn_v2.experiment import parse_experiment_config


MODEL_IDS = (
    "trace_1d",
    "trace_1d_dilated_tcn",
    "trace_1d_dilated_tcn_mismatch_training",
    "trace_1d_tcn_lateral_mixer_k1_mismatch_training",
    "trace_1d_tcn_lateral_mixer_mismatch_training",
    "trace_1d_tcn_lateral_mixer_k5_mismatch_training",
    "trace_1d_mismatch_training",
    "patch_2d_supervised",
    "patch_2d_with_physics_loss",
    "patch_2d_mismatch_training",
)

TRAIN_DEFAULTS = {
    "benchmark_dir": None,
    "model_id": "patch_2d_supervised",
    "patch_lateral": 32,
    "patch_twt": 128,
    "lateral_stride": 16,
    "twt_stride": 64,
    "min_valid_fraction": 0.50,
    "split_policy": "derive",
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "max_patches": None,
    "patch_index": None,
    "normalization": None,
    "overfit_tiny": False,
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 1e-3,
    "hidden_channels": 32,
    "depth": 5,
    "lambda_physics": 0.0,
    "lambda_real_delta": 0.0,
    "log_interval_batches": 10,
    "device": "auto",
    "seed": 20260617,
    "model_role": None,
    "synthetic_gate_report_dir": None,
    "synthetic_gate_report_card": None,
    "synthetic_gate_frozen_candidate": False,
}

TRAIN_CONFIG_KEYS = {
    "benchmark_dir": "benchmark-dir",
    "model_id": "model-id",
    "patch_lateral": "patch-lateral",
    "patch_twt": "patch-twt",
    "lateral_stride": "lateral-stride",
    "twt_stride": "twt-stride",
    "min_valid_fraction": "min-valid-fraction",
    "split_policy": "split-policy",
    "validation_fraction": "validation-fraction",
    "test_fraction": "test-fraction",
    "max_patches": "max-patches",
    "patch_index": "patch-index",
    "normalization": "normalization",
    "overfit_tiny": "overfit-tiny",
    "epochs": "epochs",
    "batch_size": "batch-size",
    "learning_rate": "learning-rate",
    "hidden_channels": "hidden-channels",
    "depth": "depth",
    "lambda_physics": "lambda-physics",
    "lambda_real_delta": "lambda-real-delta",
    "log_interval_batches": "log-interval-batches",
    "device": "device",
    "seed": "seed",
    "model_role": "model-role",
    "synthetic_gate_report_dir": "synthetic-gate-report-dir",
    "synthetic_gate_report_card": "synthetic-gate-report-card",
    "synthetic_gate_frozen_candidate": "synthetic-gate-frozen-candidate",
}

TRAIN_CONFIG_GROUPS = {
    "sources": {"benchmark_dir", "patch_index", "normalization"},
    "model": {"model_id", "model_role", "hidden_channels", "depth"},
    "patching": {
        "patch_lateral",
        "patch_twt",
        "lateral_stride",
        "twt_stride",
        "min_valid_fraction",
    },
    "split": {"split_policy", "validation_fraction", "test_fraction"},
    "optimization": {"epochs", "batch_size", "learning_rate", "seed"},
    "losses": {"lambda_physics", "lambda_real_delta"},
    "runtime": {"device", "log_interval_batches"},
    "smoke_test": {"max_patches", "overfit_tiny"},
    "synthetic_gate": {
        "synthetic_gate_report_dir",
        "synthetic_gate_report_card",
        "synthetic_gate_frozen_candidate",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=None)
    train.add_argument("--benchmark-dir", type=Path, default=TRAIN_DEFAULTS["benchmark_dir"])
    train.add_argument("--model-id", choices=MODEL_IDS, default=TRAIN_DEFAULTS["model_id"])
    train.add_argument("--patch-lateral", type=int, default=TRAIN_DEFAULTS["patch_lateral"])
    train.add_argument("--patch-twt", type=int, default=TRAIN_DEFAULTS["patch_twt"])
    train.add_argument("--lateral-stride", type=int, default=TRAIN_DEFAULTS["lateral_stride"])
    train.add_argument("--twt-stride", type=int, default=TRAIN_DEFAULTS["twt_stride"])
    train.add_argument("--min-valid-fraction", type=float, default=TRAIN_DEFAULTS["min_valid_fraction"])
    train.add_argument("--split-policy", choices=("derive", "strict"), default=TRAIN_DEFAULTS["split_policy"])
    train.add_argument("--validation-fraction", type=float, default=TRAIN_DEFAULTS["validation_fraction"])
    train.add_argument("--test-fraction", type=float, default=TRAIN_DEFAULTS["test_fraction"])
    train.add_argument("--max-patches", type=int, default=TRAIN_DEFAULTS["max_patches"])
    train.add_argument("--patch-index", type=Path, default=TRAIN_DEFAULTS["patch_index"])
    train.add_argument("--normalization", type=Path, default=TRAIN_DEFAULTS["normalization"])
    train.add_argument("--overfit-tiny", action="store_true")
    train.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    train.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    train.add_argument("--learning-rate", type=float, default=TRAIN_DEFAULTS["learning_rate"])
    train.add_argument("--hidden-channels", type=int, default=TRAIN_DEFAULTS["hidden_channels"])
    train.add_argument("--depth", type=int, default=TRAIN_DEFAULTS["depth"])
    train.add_argument("--lambda-physics", type=float, default=TRAIN_DEFAULTS["lambda_physics"])
    train.add_argument(
        "--lambda-real-delta",
        type=float,
        default=TRAIN_DEFAULTS["lambda_real_delta"],
    )
    train.add_argument(
        "--log-interval-batches",
        type=int,
        default=TRAIN_DEFAULTS["log_interval_batches"],
    )
    train.add_argument("--device", default=TRAIN_DEFAULTS["device"])
    train.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    train.add_argument("--model-role", default=TRAIN_DEFAULTS["model_role"])
    train.add_argument("--synthetic-gate-report-dir", type=Path, default=TRAIN_DEFAULTS["synthetic_gate_report_dir"])
    train.add_argument("--synthetic-gate-report-card", type=Path, default=TRAIN_DEFAULTS["synthetic_gate_report_card"])
    train.add_argument(
        "--synthetic-gate-frozen-candidate",
        action="store_true",
        help="Mark this run as belonging to the current frozen synthetic-gate candidate set.",
    )

    predict = sub.add_parser("predict")
    predict.add_argument("--model-run-dir", type=Path, required=True)
    predict.add_argument("--benchmark-dir", type=Path, default=None)
    predict.add_argument("--index-source", choices=("train", "eval"), default="train")
    predict.add_argument("--eval-patch-index", type=Path, default=None)
    predict.add_argument("--sample-kind", action="append", choices=(
        "base",
        "frequency_probe",
        "seismic_variant",
        "frequency_probe_seismic_variant",
    ))
    predict.add_argument("--split", default="validation", choices=("train", "validation", "test", "benchmark", "all"))
    predict.add_argument("--batch-size", type=int, default=8)
    predict.add_argument("--device", default="auto")
    predict.add_argument(
        "--checkpoint",
        choices=("primary", "best", "final"),
        default="primary",
    )

    report = sub.add_parser("report")
    report.add_argument("--prediction-dir", type=Path, required=True)

    summarize = sub.add_parser("summarize")
    summarize.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="MODEL:SCOPE:REPORT_DIR",
        help="Report entry. Repeat for each model/scope report directory.",
    )

    stamp = sub.add_parser("stamp-gate")
    stamp.add_argument("--model-run-dir", type=Path, required=True)
    stamp.add_argument("--synthetic-gate-report-dir", type=Path, required=True)
    stamp.add_argument("--synthetic-gate-report-card", type=Path, required=True)
    stamp.add_argument(
        "--synthetic-gate-frozen-candidate",
        action="store_true",
        help="Mark this model run as belonging to the current frozen synthetic-gate candidate set.",
    )
    return parser.parse_args()


def _resolve_output_dir(prefix: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "scripts" / "output" / f"{prefix}_{timestamp}"


def _apply_train_config(args: argparse.Namespace) -> argparse.Namespace:
    args.real_delta_config = None
    if args.config is None:
        if args.benchmark_dir is None:
            raise ValueError("ginn_v2.py train requires --benchmark-dir or --config with train.benchmark_dir.")
        return args
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    payload = load_yaml_config(config_path)
    train_config = payload.get("train")
    if not isinstance(train_config, Mapping):
        raise ValueError("GINN-v2 config must contain a train mapping.")
    config, real_delta = _flatten_train_config(train_config)
    if real_delta is not None and not isinstance(real_delta, dict):
        raise ValueError("train.real_delta must be a mapping when configured.")
    args.real_delta_config = dict(real_delta) if real_delta is not None else None
    provided_flags = {token[2:] for token in sys.argv if token.startswith("--")}
    for key, flag in TRAIN_CONFIG_KEYS.items():
        if flag in provided_flags or key not in config:
            continue
        value = config[key]
        if key in {"benchmark_dir", "patch_index", "normalization", "synthetic_gate_report_dir", "synthetic_gate_report_card"}:
            value = None if value is None or str(value).strip() == "" else Path(str(value))
        setattr(args, key, value)
    if args.benchmark_dir is None:
        raise ValueError("ginn_v2.py train requires train.benchmark_dir in config or --benchmark-dir.")
    return args


def _flatten_train_config(
    train_config: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    allowed_groups = set(TRAIN_CONFIG_GROUPS) | {"real_delta"}
    unexpected_groups = sorted(set(train_config) - allowed_groups)
    if unexpected_groups:
        raise ValueError(f"Unsupported train config groups: {unexpected_groups}")
    flattened: dict[str, object] = {}
    for group_name, allowed_keys in TRAIN_CONFIG_GROUPS.items():
        group = train_config.get(group_name)
        if group is None:
            continue
        if not isinstance(group, Mapping):
            raise ValueError(f"train.{group_name} must be a mapping.")
        unexpected_keys = sorted(set(group) - allowed_keys)
        if unexpected_keys:
            raise ValueError(
                f"Unsupported keys in train.{group_name}: {unexpected_keys}"
            )
        for key, value in group.items():
            if key in flattened:
                raise ValueError(f"Duplicate train config key: {key}")
            flattened[str(key)] = value
    real_delta = train_config.get("real_delta")
    if real_delta is None:
        return flattened, None
    if not isinstance(real_delta, Mapping):
        raise ValueError("train.real_delta must be a mapping.")
    return flattened, dict(real_delta)


def _resolve_benchmark_dir(value: Path | str) -> Path:
    text = str(value).strip()
    if text.casefold() != "auto":
        return resolve_relative_path(value, root=REPO_ROOT)
    root = REPO_ROOT / "experiments" / "synthoseis_lite" / "results"
    required = [
        "generate_field_conditioned/synthetic_benchmark.h5",
        "generate_field_conditioned/sample_index.csv",
        "generate_field_conditioned/benchmark_manifest.json",
    ]
    candidates = [
        path
        for path in root.glob("*")
        if path.is_dir() and all((path / name).is_file() for name in required)
    ]
    if not candidates:
        raise FileNotFoundError(f"No synthoseis_lite result with generate_field_conditioned benchmark found under {root}.")
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1] / "generate_field_conditioned"


def _sample_kinds_for_training(model_id: str) -> set[str]:
    return default_train_kinds(model_id)


def _write_train_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    benchmark_dir: Path,
    patch_index_path: Path,
    normalization_path: Path,
    normalization: dict,
    input_reference_stats_path: Path,
    train_result: dict,
    patch_index_truncated: bool,
    max_patches: int | None,
    real_delta_manifest: Mapping[str, object] | None,
    real_well_outputs: Mapping[str, Path] | None,
) -> None:
    history_path = Path(train_result["history"])
    checkpoints = {"primary": str(train_result["checkpoints"]["primary"])}
    for checkpoint_name in ("best", "final"):
        record = dict(train_result["checkpoints"][checkpoint_name])
        checkpoint_path = Path(record["path"])
        checkpoints[checkpoint_name] = {
            "path": repo_relative_path(checkpoint_path, root=REPO_ROOT),
            "epoch": int(record["epoch"]),
            "validation_loss": float(record["validation_loss"]),
        }
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    with benchmark_manifest_path.open("r", encoding="utf-8") as handle:
        benchmark_manifest = json.load(handle)
    benchmark_contract_fingerprint = require_contract_fingerprint(
        benchmark_manifest, label=f"benchmark {benchmark_dir}"
    )
    sample_domain = str(benchmark_manifest.get("sample_domain") or "")
    if sample_domain == "time":
        sample_unit, depth_basis = "s", None
    elif sample_domain == "depth" and benchmark_manifest.get("depth_basis") == "tvdss":
        sample_unit, depth_basis = "m", "tvdss"
    else:
        raise ValueError("Benchmark has an invalid sample domain/depth basis contract.")
    input_contracts = {
        "benchmark": {
            "path": repo_relative_path(benchmark_manifest_path, root=REPO_ROOT),
            "contract_fingerprint_sha256": benchmark_contract_fingerprint,
        }
    }
    if sample_domain == "depth":
        rock_contract = dict(
            dict(benchmark_manifest.get("input_contracts") or {}).get(
                "rock_physics_analysis"
            )
            or {}
        )
        if not str(rock_contract.get("contract_fingerprint_sha256") or ""):
            raise ValueError(
                "Depth benchmark lacks input_contracts.rock_physics_analysis."
            )
        input_contracts["rock_physics_analysis"] = rock_contract
        forward_inputs_path = str(
            benchmark_manifest.get("forward_model_inputs_path") or ""
        ).strip()
        if not forward_inputs_path:
            raise ValueError("Depth benchmark lacks forward_model_inputs_path.")
    else:
        forward_inputs_path = ""
    if real_delta_manifest is not None:
        real_sources = dict(real_delta_manifest.get("sources") or {})
        for source_key, role in (
            ("lfm", "real_delta_lfm_variant"),
            ("well_control_summary", "real_delta_well_controls"),
        ):
            reference = dict(real_sources.get(source_key) or {})
            fingerprint = str(reference.get("contract_fingerprint_sha256") or "")
            if not fingerprint:
                raise ValueError(f"real_delta.sources.{source_key} lacks a contract fingerprint.")
            input_contracts[role] = {
                "path": str(reference["path"]),
                "contract_fingerprint_sha256": fingerprint,
            }
    manifest = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "status": "ok",
        "input_contracts": input_contracts,
        "model_id": args.model_id,
        "model_role": str(args.model_role or _infer_model_role(args.model_id)),
        "sample_domain": sample_domain,
        "sample_unit": sample_unit,
        "depth_basis": depth_basis,
        "forward_model_inputs_path": forward_inputs_path,
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "patch_index": repo_relative_path(patch_index_path, root=REPO_ROOT),
        "patch_index_truncated": bool(patch_index_truncated),
        "max_patches": max_patches,
        "patch_spec": {
            "lateral_samples": int(args.patch_lateral),
            "vertical_samples": int(args.patch_twt),
            "lateral_stride": int(args.lateral_stride),
            "vertical_stride": int(args.twt_stride),
            "min_valid_fraction": float(args.min_valid_fraction),
        },
        "split_policy": args.split_policy,
        "validation_fraction": float(args.validation_fraction),
        "test_fraction": float(args.test_fraction),
        "normalization": normalization,
        "normalization_path": repo_relative_path(normalization_path, root=REPO_ROOT),
        "input_reference_stats": repo_relative_path(input_reference_stats_path, root=REPO_ROOT),
        "input_channels": ["seismic", "lfm_controlled_degraded", "valid_mask_model"],
        "output_semantics": "pred_log_ai = lfm_controlled_degraded + pred_delta_log_ai",
        "train_sample_kinds": sorted(_sample_kinds_for_training(args.model_id)),
        "sample_kind_sampling_weights": {
            kind: 1.0 for kind in sorted(_sample_kinds_for_training(args.model_id))
        },
        "train_sampler": "balanced_by_sample_kind",
        "loss": {
            "lambda_ai": 1.0,
            "lambda_physics": float(args.lambda_physics),
            "lambda_real_delta": float(args.lambda_real_delta),
            "physics_loss_applied_sample_kinds": (
                ["base"] if float(args.lambda_physics) > 0.0 else []
            ),
            "physics_forward_operator_id": (
                (
                    "cup.physics.torch_backend.forward_depth"
                    if sample_domain == "depth"
                    else "cup.physics.torch_backend.forward_time"
                )
                if float(args.lambda_physics) > 0.0
                else ""
            ),
            "wavelet_scenario_id": "",
            "gain_scenario_id": "",
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "overfit_tiny": bool(args.overfit_tiny),
            "seed": int(args.seed),
            "log_interval_batches": int(args.log_interval_batches),
            "synthetic_sequence_sha256": train_result["synthetic_sequence_sha256"],
        },
        "device": train_result.get("device_metadata", {"resolved_device": train_result.get("device", "")}),
        "model_info": train_result["model_info"],
        "checkpoints": checkpoints,
        "training_history": repo_relative_path(history_path, root=REPO_ROOT),
        "training_log": repo_relative_path(output_dir / "training.log", root=REPO_ROOT),
        "best_validation_loss": train_result["best_validation_loss"],
        "real_delta": dict(real_delta_manifest) if real_delta_manifest is not None else None,
        "real_well_outputs": {
            key: {"path": repo_relative_path(path, root=REPO_ROOT)}
            for key, path in (real_well_outputs or {}).items()
        },
        "synthetic_gate_evidence_status": "pending",
    }
    if args.synthetic_gate_report_dir is not None or args.synthetic_gate_report_card is not None:
        if args.synthetic_gate_report_dir is None or args.synthetic_gate_report_card is None:
            raise ValueError(
                "--synthetic-gate-report-dir and --synthetic-gate-report-card must be provided together."
            )
        _stamp_gate_evidence(
            manifest,
            report_dir=resolve_relative_path(args.synthetic_gate_report_dir, root=REPO_ROOT),
            report_card=resolve_relative_path(args.synthetic_gate_report_card, root=REPO_ROOT),
            frozen_candidate=bool(args.synthetic_gate_frozen_candidate),
        )
    primary_name = str(checkpoints["primary"])
    primary_checkpoint = resolve_relative_path(
        str(dict(checkpoints[primary_name])["path"]), root=REPO_ROOT
    )
    manifest["contract_fingerprint_schema"] = CONTRACT_FINGERPRINT_SCHEMA
    manifest["contract_fingerprint_sha256"] = contract_fingerprint_sha256(
        contract_schema_version=MODEL_RUN_SCHEMA_VERSION,
        semantics={
            "model_id": manifest["model_id"],
            "model_role": manifest["model_role"],
            "input_channels": manifest["input_channels"],
            "output_semantics": manifest["output_semantics"],
            "patch_spec": manifest["patch_spec"],
            "sample_domain": sample_domain,
            "sample_unit": sample_unit,
            "depth_basis": depth_basis,
        },
        business_config={
            "split_policy": manifest["split_policy"],
            "validation_fraction": manifest["validation_fraction"],
            "test_fraction": manifest["test_fraction"],
            "loss": manifest["loss"],
            "training": manifest["training"],
            "model_info": manifest["model_info"],
            "synthetic_gate_evidence_status": manifest["synthetic_gate_evidence_status"],
            "synthetic_gate_evidence": manifest.get("synthetic_gate_evidence"),
        },
        input_contracts=input_contracts,
        primary_artifacts={
            "primary_checkpoint": primary_checkpoint,
            "normalization": normalization_path,
            "input_reference_stats": input_reference_stats_path,
            "patch_index": patch_index_path,
        },
    )
    write_json(output_dir / "model_run_manifest.json", manifest)


def _stamp_gate_evidence(
    manifest: dict,
    *,
    report_dir: Path,
    report_card: Path,
    frozen_candidate: bool,
) -> None:
    if not report_dir.is_dir():
        raise FileNotFoundError(f"synthetic gate report directory not found: {report_dir}")
    if not report_card.is_file():
        raise FileNotFoundError(f"synthetic gate report card not found: {report_card}")
    evaluation_summary_path = report_dir / "evaluation_summary.json"
    if not evaluation_summary_path.is_file():
        raise FileNotFoundError(
            f"synthetic gate evaluation summary not found: {evaluation_summary_path}"
        )
    with evaluation_summary_path.open("r", encoding="utf-8") as handle:
        evaluation_summary = json.load(handle)
    if evaluation_summary.get("schema_version") != SYNTHOSEIS_REPORT_SCHEMA_VERSION:
        raise ValueError(f"Synthetic gate evaluation must use {SYNTHOSEIS_REPORT_SCHEMA_VERSION}.")
    gate_contract_fingerprint = require_contract_fingerprint(
        evaluation_summary, label=f"synthetic gate evaluation {evaluation_summary_path}"
    )
    manifest["input_contracts"]["synthetic_gate_evaluation"] = {
        "path": repo_relative_path(evaluation_summary_path, root=REPO_ROOT),
        "contract_fingerprint_sha256": gate_contract_fingerprint,
    }
    manifest["synthetic_gate_evidence_status"] = "ok"
    manifest["synthetic_gate_evidence"] = {
        "report_dir": repo_relative_path(report_dir, root=REPO_ROOT),
        "report_card": repo_relative_path(report_card, root=REPO_ROOT),
        "is_current_frozen_candidate": bool(frozen_candidate),
    }


def _infer_model_role(model_id: str) -> str:
    text = str(model_id)
    if "lateral_mixer" in text:
        return "lateral"
    if "trace_1d" in text or "trace1d" in text:
        return "no_lateral"
    return text.replace("-", "_")


def _resolve_checkpoint_from_manifest(
    manifest: Mapping[str, object],
    *,
    selection: str,
) -> tuple[str, Path]:
    if str(manifest.get("schema_version") or "") != MODEL_RUN_SCHEMA_VERSION:
        raise ValueError(f"GINN-v2 model run must use schema {MODEL_RUN_SCHEMA_VERSION}.")
    deployment = manifest.get("deployment_checkpoint")
    if not isinstance(deployment, Mapping):
        raise ValueError("GINN-v2 experiment manifest lacks deployment_checkpoint.")
    resolved_name = str(deployment.get("kind")) if selection == "primary" else selection
    if resolved_name == str(deployment.get("kind")):
        record = deployment
    else:
        stage_id = str(deployment.get("stage_id"))
        stage = next((item for item in manifest.get("stages", []) if item.get("stage_id") == stage_id), None)
        if not isinstance(stage, Mapping):
            raise ValueError(f"Deployment stage is missing: {stage_id}")
        path_value = dict(stage.get("checkpoints") or {}).get(resolved_name)
        record = {"path": path_value}
    path = resolve_relative_path(str(record.get("path") or ""), root=REPO_ROOT)
    if not path.is_file():
        raise FileNotFoundError(path)
    return resolved_name, path


def run_train(args: argparse.Namespace) -> None:
    if args.config is None:
        raise ValueError(
            "GINN-v2 training now requires --config with the ginn_v2_experiment_v1 root. "
            "Legacy training flags are not accepted; see "
            "docs/spec/GINN_V2_COMPOSABLE_TRAINING_DESIGN.md."
        )
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    experiment = parse_experiment_config(load_yaml_config(config_path))
    output_dir = (
        resolve_relative_path(args.output_dir, root=REPO_ROOT)
        if args.output_dir is not None
        else REPO_ROOT / "experiments" / "ginn_v2" / "results" / experiment.experiment_id
    )
    logger = configure_training_logger(output_dir)
    manifest = run_experiment(
        config=experiment,
        root=REPO_ROOT,
        output_dir=output_dir,
        logger=logger,
    )
    print("=== GINN-v2 composable experiment ===")
    print(f"Output: {output_dir}")
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Deployment: {manifest['deployment_checkpoint']['path']}")
    return

    args = _apply_train_config(args)
    benchmark_dir = _resolve_benchmark_dir(args.benchmark_dir)
    output_dir = _resolve_output_dir("ginn_v2_train", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.output_dir is not None)
    logger = configure_training_logger(output_dir)
    logger.info("GINN-v2 train output: %s", output_dir)
    logger.info("benchmark: %s", benchmark_dir)
    benchmark = SynthoseisBenchmark(benchmark_dir)
    patch_spec = PatchSpec(
        lateral_samples=int(args.patch_lateral),
        twt_samples=int(args.patch_twt),
        lateral_stride=int(args.lateral_stride),
        twt_stride=int(args.twt_stride),
        min_valid_fraction=float(args.min_valid_fraction),
    )
    max_patches = 4 if args.overfit_tiny and args.max_patches is None else args.max_patches
    patch_index_truncated = max_patches is not None
    if args.patch_index is not None:
        if args.overfit_tiny or args.max_patches is not None:
            raise ValueError("--patch-index cannot be combined with --overfit-tiny or --max-patches.")
        patch_index_source = resolve_relative_path(args.patch_index, root=REPO_ROOT)
        patch_index = pd.read_csv(patch_index_source)
        allowed = _sample_kinds_for_training(args.model_id)
        unexpected = sorted(set(patch_index["sample_kind"].astype(str)) - allowed)
        if unexpected:
            raise ValueError(
                f"Provided patch index contains sample_kind values not allowed for {args.model_id}: "
                f"{unexpected}. Allowed: {sorted(allowed)}"
            )
    else:
        patch_index = build_patch_index(
            benchmark,
            patch_spec=patch_spec,
            sample_kinds=_sample_kinds_for_training(args.model_id),
            split_policy=args.split_policy,
            validation_fraction=float(args.validation_fraction),
            test_fraction=float(args.test_fraction),
            max_patches=max_patches,
        )
    if args.overfit_tiny:
        patch_index["split"] = "train"
        first = patch_index.head(min(4, len(patch_index))).copy()
        validation = first.copy()
        validation["patch_id"] = validation["patch_id"].astype(str) + "__overfit_validation"
        validation["split"] = "validation"
        patch_index = pd.concat([first, validation], ignore_index=True)
    patch_index_path = output_dir / "patch_index.csv"
    patch_index.to_csv(patch_index_path, index=False)
    logger.info("patch index ready: %d rows", len(patch_index))
    if args.normalization is not None:
        with resolve_relative_path(args.normalization, root=REPO_ROOT).open("r", encoding="utf-8") as handle:
            normalization = json.load(handle)
    else:
        normalization = compute_normalization(benchmark, patch_index)
    normalization_path = output_dir / "normalization.json"
    write_json(normalization_path, normalization)
    input_reference_stats = {
        "schema_version": INPUT_REFERENCE_STATS_SCHEMA_VERSION,
        "input_name": "seismic",
        "stats": compute_input_reference_stats(benchmark, patch_index, input_name="seismic"),
        "sampling": {
            "input_name": "seismic",
            "total_train_patches": int(patch_index[patch_index["split"].eq("train")].shape[0]),
            "sampled_train_patches": int(patch_index[patch_index["split"].eq("train")].shape[0]),
            "sampling_policy": "all_train_patches_from_this_model_run",
        },
        "sources": {
            "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
            "patch_index": repo_relative_path(patch_index_path, root=REPO_ROOT),
        },
    }
    input_reference_stats_path = output_dir / "input_reference_stats.json"
    write_json(input_reference_stats_path, input_reference_stats)
    real_delta_support = None
    if args.real_delta_config is not None:
        _probe_model, probe_info = build_model(
            args.model_id,
            hidden_channels=int(args.hidden_channels),
            depth=int(args.depth),
        )
        del _probe_model
        resolved_role = str(args.model_role or _infer_model_role(args.model_id))
        if float(args.lambda_real_delta) > 0.0 and (
            resolved_role != "no_lateral"
            or int(probe_info.receptive_field_lateral) != 1
        ):
            raise NotImplementedError(
                "Non-zero lambda_real_delta currently requires a no-lateral model "
                "role with receptive_field_lateral=1."
            )
        real_delta_support = prepare_real_delta_support(
            config=args.real_delta_config,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            normalization=normalization,
            patch_spec={
                "lateral_samples": int(args.patch_lateral),
                "vertical_samples": int(args.patch_twt),
                "lateral_stride": int(args.lateral_stride),
                "vertical_stride": int(args.twt_stride),
                "min_valid_fraction": float(args.min_valid_fraction),
            },
            input_reference_stats_path=input_reference_stats_path,
            lambda_real_delta=float(args.lambda_real_delta),
            seed=int(args.seed),
            logger=logger,
        )
    result = train_model(
        benchmark=benchmark,
        patch_index=patch_index,
        normalization=normalization,
        output_dir=output_dir,
        model_id=args.model_id,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        hidden_channels=int(args.hidden_channels),
        depth=int(args.depth),
        device_name=str(args.device),
        lambda_physics=float(args.lambda_physics),
        lambda_real_delta=float(args.lambda_real_delta),
        seed=int(args.seed),
        real_delta_support=real_delta_support,
        log_interval_batches=int(args.log_interval_batches),
        logger=logger,
    )
    real_well_outputs: dict[str, Path] = {}
    if real_delta_support is not None:
        device, _device_metadata = resolve_device(str(args.device))
        qc_models = {}
        for checkpoint_name in ("best", "final"):
            checkpoint_path = Path(result["checkpoints"][checkpoint_name]["path"])
            model, _checkpoint = load_checkpoint(checkpoint_path)
            model.to(device)
            qc_models[checkpoint_name] = model
        real_well_outputs = evaluate_real_wells(
            support=real_delta_support,
            models=qc_models,
            output_dir=output_dir,
            benchmark_dir=benchmark_dir,
            repo_root=REPO_ROOT,
            device=device,
            logger=logger,
        )
    _write_train_manifest(
        output_dir=output_dir,
        args=args,
        benchmark_dir=benchmark_dir,
        patch_index_path=patch_index_path,
        normalization_path=normalization_path,
        normalization=normalization,
        input_reference_stats_path=input_reference_stats_path,
        train_result=result,
        patch_index_truncated=patch_index_truncated,
        max_patches=max_patches,
        real_delta_manifest=(
            real_delta_support.manifest_payload(repo_root=REPO_ROOT)
            if real_delta_support is not None
            else None
        ),
        real_well_outputs=real_well_outputs,
    )
    print("=== GINN-v2 model ablation train ===")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_id}")
    print(f"Patches: {len(patch_index)}")
    print(f"Best validation loss: {result['best_validation_loss']:.6g}")


def run_predict(args: argparse.Namespace) -> None:
    model_run_dir = resolve_relative_path(args.model_run_dir, root=REPO_ROOT)
    with (model_run_dir / "model_run_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    checkpoint_name, checkpoint_path = _resolve_checkpoint_from_manifest(
        manifest,
        selection=str(args.checkpoint),
    )
    benchmark_dir = (
        resolve_relative_path(args.benchmark_dir, root=REPO_ROOT)
        if args.benchmark_dir is not None
        else resolve_relative_path(manifest["benchmark_dir"], root=REPO_ROOT)
    )
    output_dir = _resolve_output_dir("ginn_v2_predict", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    benchmark = SynthoseisBenchmark(benchmark_dir)
    if args.eval_patch_index is not None:
        patch_index_source_path = resolve_relative_path(args.eval_patch_index, root=REPO_ROOT)
        patch_index = pd.read_csv(patch_index_source_path)
        if args.sample_kind:
            sample_kinds = set(args.sample_kind)
            patch_index = patch_index[
                patch_index["sample_kind"].astype(str).isin(sample_kinds)
            ].copy()
            if patch_index.empty:
                raise ValueError(
                    f"No provided eval-index patches match requested sample kinds: {sorted(sample_kinds)}"
                )
        eval_index_path = output_dir / "eval_patch_index.csv"
        patch_index.to_csv(eval_index_path, index=False)
        patch_index_source = repo_relative_path(eval_index_path, root=REPO_ROOT)
    elif args.index_source == "train":
        patch_index = pd.read_csv(resolve_relative_path(manifest["patch_index"], root=REPO_ROOT))
        if args.sample_kind:
            sample_kinds = set(args.sample_kind)
            patch_index = patch_index[
                patch_index["sample_kind"].astype(str).isin(sample_kinds)
            ].copy()
            if patch_index.empty:
                raise ValueError(
                    f"No train-index patches match requested sample kinds: {sorted(sample_kinds)}"
                )
        patch_index_source = repo_relative_path(resolve_relative_path(manifest["patch_index"], root=REPO_ROOT), root=REPO_ROOT)
        eval_index_path = output_dir / "eval_patch_index.csv"
        patch_index.to_csv(eval_index_path, index=False)
        patch_index_source = repo_relative_path(eval_index_path, root=REPO_ROOT)
    else:
        sample_kinds = set(args.sample_kind) if args.sample_kind else default_eval_kinds()
        spec_cfg = dict(manifest["patch_spec"])
        patch_index = build_patch_index(
            benchmark,
            patch_spec=PatchSpec(
                lateral_samples=int(spec_cfg["lateral_samples"]),
                twt_samples=int(spec_cfg["vertical_samples"]),
                lateral_stride=int(spec_cfg["lateral_stride"]),
                twt_stride=int(spec_cfg["vertical_stride"]),
                min_valid_fraction=float(spec_cfg["min_valid_fraction"]),
            ),
            sample_kinds=sample_kinds,
            split_policy=str(manifest.get("split_policy", "derive")),
            validation_fraction=float(manifest.get("validation_fraction", 0.15)),
            test_fraction=float(manifest.get("test_fraction", 0.15)),
        )
        eval_index_path = output_dir / "eval_patch_index.csv"
        patch_index.to_csv(eval_index_path, index=False)
        patch_index_source = repo_relative_path(eval_index_path, root=REPO_ROOT)
    result = predict_patches(
        benchmark=benchmark,
        patch_index=patch_index,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split=args.split,
        batch_size=int(args.batch_size),
        device_name=str(args.device),
    )
    input_contracts = {
        "model_run": {
            "path": repo_relative_path(model_run_dir / "model_run_manifest.json", root=REPO_ROOT),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                manifest, label=f"model run {model_run_dir}"
            ),
        },
        "benchmark": dict(dict(manifest.get("input_contracts") or {}).get("benchmark") or {}),
    }
    prediction_contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=PREDICTION_SCHEMA_VERSION,
        semantics={
            "model_id": result["model_id"],
            "split": args.split,
            "sample_kinds": sorted(set(patch_index["sample_kind"].astype(str))),
        },
        business_config={
            "index_source": args.index_source,
            "checkpoint_selection": str(args.checkpoint),
            "batch_size": int(args.batch_size),
        },
        input_contracts=input_contracts,
        primary_artifacts={
            "predictions": result["prediction_npz"],
            "prediction_index": result["prediction_index"],
        },
    )
    summary = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "status": "ok",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": prediction_contract_fingerprint,
        "input_contracts": input_contracts,
        "model_run_dir": repo_relative_path(model_run_dir, root=REPO_ROOT),
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "split": args.split,
        "index_source": args.index_source,
        "patch_index": patch_index_source,
        "sample_kinds": sorted(set(patch_index["sample_kind"].astype(str))),
        "model_id": result["model_id"],
        "checkpoint_selection": str(args.checkpoint),
        "resolved_checkpoint": checkpoint_name,
        "checkpoint": repo_relative_path(checkpoint_path, root=REPO_ROOT),
        "model_info": result["model_info"],
        "normalization": result["normalization"],
        "input_channels": ["seismic", "lfm_controlled_degraded", "valid_mask_model"],
        "output_semantics": "pred_log_ai = lfm_controlled_degraded + pred_delta_log_ai",
        "outputs": {
            "predictions": repo_relative_path(result["prediction_npz"], root=REPO_ROOT),
            "prediction_index": repo_relative_path(result["prediction_index"], root=REPO_ROOT),
        },
        "device": result.get("device_metadata", {}),
    }
    write_json(output_dir / "prediction_summary.json", summary)
    write_json(output_dir / "prediction_manifest.json", summary)
    print("=== GINN-v2 model ablation predict ===")
    print(f"Output: {output_dir}")
    print(f"Predictions: {result['n_predictions']}")


def run_report(args: argparse.Namespace) -> None:
    prediction_dir = resolve_relative_path(args.prediction_dir, root=REPO_ROOT)
    output_dir = _resolve_output_dir("ginn_v2_report", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = report_predictions(prediction_dir=prediction_dir, output_dir=output_dir)
    summary = {
        "schema_version": PATCH_SMOKE_REPORT_SCHEMA_VERSION,
        "report_scope": "patch_smoke",
        "not_synthoseis_lite_report": True,
        "status": "ok",
        "prediction_dir": repo_relative_path(prediction_dir, root=REPO_ROOT),
        "outputs": {
            key: repo_relative_path(value, root=REPO_ROOT)
            for key, value in result.items()
            if key != "status"
        },
    }
    write_json(output_dir / "report_summary.json", summary)
    print("=== GINN-v2 model ablation report ===")
    print(f"Output: {output_dir}")


def run_summarize(args: argparse.Namespace) -> None:
    output_dir = _resolve_output_dir("ginn_v2_summary", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    frequency_frames: list[pd.DataFrame] = []
    for spec in args.report:
        model, scope, report_dir = _parse_report_spec(spec)
        report_path = resolve_relative_path(report_dir, root=REPO_ROOT)
        card_path = report_path / "model_report_card.json"
        with card_path.open("r", encoding="utf-8") as handle:
            card = json.load(handle)
        aggregate = dict(card.get("aggregate") or {})
        lfm_aggregate = dict(card.get("lfm_aggregate") or {})
        lfm_ideal_aggregate = dict(card.get("lfm_ideal_aggregate") or {})
        oracle_aggregate = dict(card.get("oracle_aggregate") or {})
        probe_aggregate = dict(card.get("probe_aggregate") or {})
        amplitude_phase_aggregate = dict(card.get("probe_amplitude_phase_aggregate") or {})
        zero_x_aggregate = dict(card.get("zero_x_false_prediction_aggregate") or {})
        unsupported_zero_x_aggregate = dict(
            card.get("unsupported_zero_x_false_prediction_aggregate") or {}
        )
        zero_x_energy_aggregate = dict(card.get("zero_x_false_energy_aggregate") or {})
        unsupported_energy_aggregate = dict(card.get("unsupported_false_energy_aggregate") or {})
        geometry_aggregate = dict(card.get("geometry_aggregate") or {})
        realization_uniform_aggregate = dict(card.get("realization_uniform_aggregate") or {})
        realization_center_aggregate = dict(card.get("realization_center_crop_aggregate") or {})
        rows.append(
            {
                "model": model,
                "scope": scope,
                "n_patches": card.get("n_patches"),
                "model_rmse": aggregate.get("mean_rmse"),
                "model_nrmse": aggregate.get("mean_nrmse"),
                "model_corr": aggregate.get("median_corr"),
                "lfm_rmse": lfm_aggregate.get("mean_rmse"),
                "lfm_nrmse": lfm_aggregate.get("mean_nrmse"),
                "lfm_corr": lfm_aggregate.get("median_corr"),
                "lfm_ideal_rmse": lfm_ideal_aggregate.get("mean_rmse"),
                "lfm_ideal_nrmse": lfm_ideal_aggregate.get("mean_nrmse"),
                "lfm_ideal_corr": lfm_ideal_aggregate.get("median_corr"),
                "oracle_rmse": oracle_aggregate.get("mean_rmse"),
                "oracle_nrmse": oracle_aggregate.get("mean_nrmse"),
                "oracle_corr": oracle_aggregate.get("median_corr"),
                "rmse_improvement_pct_vs_lfm": card.get("rmse_improvement_pct_vs_lfm"),
                "probe_n_ok": probe_aggregate.get("n_ok"),
                "probe_nrmse": probe_aggregate.get("mean_nrmse"),
                "probe_corr": probe_aggregate.get("median_corr"),
                "probe_amplitude_phase_n_ok": amplitude_phase_aggregate.get("n_ok"),
                "probe_mean_abs_amplitude_error": amplitude_phase_aggregate.get(
                    "mean_abs_amplitude_error"
                ),
                "probe_median_amplitude_ratio": amplitude_phase_aggregate.get(
                    "median_amplitude_ratio"
                ),
                "probe_median_abs_phase_error_deg": amplitude_phase_aggregate.get(
                    "median_abs_phase_error_deg"
                ),
                "zero_x_false_prediction_n_ok": zero_x_aggregate.get("n_ok"),
                "zero_x_false_prediction_rmse": zero_x_aggregate.get("mean_rmse"),
                "unsupported_zero_x_false_prediction_n_ok": unsupported_zero_x_aggregate.get("n_ok"),
                "unsupported_zero_x_false_prediction_rmse": unsupported_zero_x_aggregate.get("mean_rmse"),
                "zero_x_false_energy_n_ok": zero_x_energy_aggregate.get("n_ok"),
                "zero_x_false_frequency_rms": zero_x_energy_aggregate.get("mean_false_frequency_rms"),
                "unsupported_false_energy_n_ok": unsupported_energy_aggregate.get("n_ok"),
                "unsupported_false_frequency_rms": unsupported_energy_aggregate.get(
                    "mean_false_frequency_rms"
                ),
                "geometry_n_ok": geometry_aggregate.get("n_ok"),
                "geometry_boundary_rmse": geometry_aggregate.get("mean_boundary_rmse"),
                "geometry_event_rmse": geometry_aggregate.get("mean_event_rmse"),
                "geometry_lateral_gradient_rmse": geometry_aggregate.get(
                    "mean_lateral_gradient_rmse"
                ),
                "realization_uniform_n_ok": realization_uniform_aggregate.get("n_ok"),
                "realization_uniform_rmse": realization_uniform_aggregate.get("mean_rmse"),
                "realization_center_crop_n_ok": realization_center_aggregate.get("n_ok"),
                "realization_center_crop_rmse": realization_center_aggregate.get("mean_rmse"),
                "report_dir": repo_relative_path(report_path, root=REPO_ROOT),
            }
        )
        frequency_path = report_path / "model_probe_metrics_by_frequency.csv"
        if frequency_path.is_file():
            frame = pd.read_csv(frequency_path)
            if not frame.empty:
                frame.insert(0, "scope", scope)
                frame.insert(0, "model", model)
                frequency_frames.append(frame)
    summary = pd.DataFrame.from_records(rows)
    summary_path = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    frequency_path = output_dir / "probe_metrics_by_frequency.csv"
    if frequency_frames:
        pd.concat(frequency_frames, ignore_index=True).to_csv(frequency_path, index=False)
    else:
        pd.DataFrame().to_csv(frequency_path, index=False)
    report_card = _build_ablation_report_card(summary, frequency_path)
    report_card_path = output_dir / "ablation_report_card.json"
    write_json(report_card_path, report_card)
    markdown_path = output_dir / "ablation_report.md"
    markdown_path.write_text(_format_ablation_markdown(report_card), encoding="utf-8")
    run_summary = {
        "schema_version": ABLATION_SUMMARY_SCHEMA_VERSION,
        "status": "ok",
        "n_reports": int(len(rows)),
        "outputs": {
            "ablation_summary": repo_relative_path(summary_path, root=REPO_ROOT),
            "probe_metrics_by_frequency": repo_relative_path(frequency_path, root=REPO_ROOT),
            "ablation_report_card": repo_relative_path(report_card_path, root=REPO_ROOT),
            "ablation_report": repo_relative_path(markdown_path, root=REPO_ROOT),
        },
        "reports": rows,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print("=== GINN-v2 model ablation summarize ===")
    print(f"Output: {output_dir}")
    print(f"Reports: {len(rows)}")


def run_stamp_gate(args: argparse.Namespace) -> None:
    raise ValueError(
        "stamp-gate is retired for immutable model runs. Re-run training with "
        "--synthetic-gate-report-dir and --synthetic-gate-report-card so gate evidence "
        "is included before the v3 model contract is published."
    )


def _parse_report_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":", maxsplit=2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError("--report must have form MODEL:SCOPE:REPORT_DIR")
    return parts[0].strip(), parts[1].strip(), Path(parts[2].strip())


def _build_ablation_report_card(summary: pd.DataFrame, frequency_path: Path) -> dict[str, object]:
    coverage = {
        "base_fullband": bool((summary["scope"].astype(str) == "test_base").any()),
        "validation_base": bool((summary["scope"].astype(str) == "validation_base").any()),
        "probe_paired_increment": bool(summary["probe_n_ok"].fillna(0).astype(float).gt(0).any()),
        "probe_mismatch_paired_increment": bool(
            summary[summary["scope"].astype(str).eq("benchmark_probe_mismatch")]
            .get("probe_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "lfm_ideal_baseline": bool(
            summary.get("lfm_ideal_rmse", pd.Series(dtype=float))
            .fillna(float("nan"))
            .notna()
            .any()
        ),
        "oracle_target_self_check": bool(
            summary.get("oracle_rmse", pd.Series(dtype=float))
            .fillna(float("nan"))
            .astype(float)
            .le(1e-8)
            .any()
        ),
        "mismatch_degradation": bool((summary["scope"].astype(str) == "validation_mismatch").any()),
        "trace_1d": bool(summary["model"].astype(str).str.contains("trace1d").any()),
        "patch_2d_supervised": bool(summary["model"].astype(str).str.contains("patch2d").any()),
        "patch_2d_mismatch_training": bool(
            summary["model"].astype(str).str.contains(r"patch2d.*mismatch|patch_2d_mismatch").any()
        ),
        "trace_1d_mismatch_training": bool(
            summary["model"].astype(str).str.contains("trace1d_mismatch").any()
        ),
        "trace_1d_dilated_tcn_mismatch_training": bool(
            summary["model"].astype(str).str.contains("trace1d_tcn_mismatch").any()
        ),
        "zero_x_false_prediction_error": bool(
            summary.get("zero_x_false_prediction_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "unsupported_zero_x_false_prediction_error": bool(
            summary.get("unsupported_zero_x_false_prediction_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "physics_loss": bool(summary["model"].astype(str).str.contains("phys", case=False).any()),
        "geometry_metrics": False,
        "patch_geometry_metrics": bool(
            summary.get("geometry_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "band_amplitude_phase_metrics": bool(
            summary.get("probe_amplitude_phase_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "zero_x_false_energy": bool(
            summary.get("zero_x_false_energy_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "unsupported_false_energy": bool(
            summary.get("unsupported_false_energy_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
        "realization_level_stitching": bool(
            summary.get("realization_uniform_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
            and summary.get("realization_center_crop_n_ok", pd.Series(dtype=float))
            .fillna(0)
            .astype(float)
            .gt(0)
            .any()
        ),
    }
    coverage["geometry_metrics"] = bool(coverage["patch_geometry_metrics"])
    test_base = summary[summary["scope"].astype(str).eq("test_base")].copy()
    benchmark_probe = summary[summary["scope"].astype(str).eq("benchmark_probe")].copy()
    mismatch = summary[summary["scope"].astype(str).eq("validation_mismatch")].copy()
    best = {
        "test_base_by_rmse": _best_row(test_base, metric="model_rmse", ascending=True),
        "probe_by_nrmse": _best_row(benchmark_probe, metric="probe_nrmse", ascending=True),
        "probe_mismatch_by_nrmse": _best_row(
            summary[summary["scope"].astype(str).eq("benchmark_probe_mismatch")].copy(),
            metric="probe_nrmse",
            ascending=True,
        ),
        "mismatch_by_rmse": _best_row(mismatch, metric="model_rmse", ascending=True),
    }
    stability = _seed_stability(summary)
    status = "partial"
    if (
        coverage["base_fullband"]
        and coverage["probe_paired_increment"]
        and coverage["mismatch_degradation"]
        and stability.get("trace1d_test_model_rmse_std") is not None
    ):
        status = "baseline_evidence_ready"
    required_for_full_gate = [
        "base_fullband",
        "validation_base",
        "probe_paired_increment",
        "probe_mismatch_paired_increment",
        "mismatch_degradation",
        "lfm_ideal_baseline",
        "oracle_target_self_check",
        "trace_1d",
        "patch_2d_supervised",
        "patch_2d_mismatch_training",
        "trace_1d_mismatch_training",
        "trace_1d_dilated_tcn_mismatch_training",
        "physics_loss",
        "geometry_metrics",
        "patch_geometry_metrics",
        "band_amplitude_phase_metrics",
        "zero_x_false_energy",
        "unsupported_false_energy",
        "realization_level_stitching",
    ]
    missing = [key for key in required_for_full_gate if not coverage.get(key)]
    conclusion = _ablation_conclusion(best, stability)
    return {
        "schema_version": ABLATION_REPORT_CARD_SCHEMA_VERSION,
        "status": status,
        "coverage": coverage,
        "required_for_full_gate": required_for_full_gate,
        "metric_notes": _ablation_metric_notes(),
        "missing_for_full_gate": missing,
        "best": best,
        "stability": stability,
        "conclusion": conclusion,
        "source_frequency_table": repo_relative_path(frequency_path, root=REPO_ROOT),
    }


def _best_row(frame: pd.DataFrame, *, metric: str, ascending: bool) -> dict[str, object] | None:
    if frame.empty or metric not in frame:
        return None
    valid = frame[pd.to_numeric(frame[metric], errors="coerce").notna()].copy()
    if valid.empty:
        return None
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    row = valid.sort_values(metric, ascending=ascending).iloc[0].to_dict()
    return _jsonable_dict(row)


def _seed_stability(summary: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    trace = summary[summary["model"].astype(str).str.startswith("trace1d_s")].copy()
    if trace.empty:
        return result
    for scope, prefix in [("test_base", "trace1d_test"), ("benchmark_probe", "trace1d_probe")]:
        scoped = trace[trace["scope"].astype(str).eq(scope)].copy()
        if scoped.empty:
            continue
        for metric in ["model_rmse", "model_corr", "probe_nrmse", "probe_corr"]:
            if metric not in scoped:
                continue
            values = pd.to_numeric(scoped[metric], errors="coerce").dropna()
            if values.empty:
                continue
            result[f"{prefix}_{metric}_mean"] = float(values.mean())
            result[f"{prefix}_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result[f"{prefix}_{metric}_n"] = int(len(values))
    return result


def _ablation_conclusion(best: dict[str, object], stability: dict[str, object]) -> dict[str, object]:
    test_best = best.get("test_base_by_rmse") or {}
    probe_best = best.get("probe_by_nrmse") or {}
    mismatch_best = best.get("mismatch_by_rmse") or {}
    statements = []
    if test_best:
        statements.append(
            f"Best test-base RMSE is {test_best.get('model')} "
            f"({float(test_best.get('model_rmse')):.6g})."
        )
    if probe_best:
        statements.append(
            f"Best paired-probe NRMSE is {probe_best.get('model')} "
            f"({float(probe_best.get('probe_nrmse')):.6g})."
        )
    probe_mismatch_best = best.get("probe_mismatch_by_nrmse") or {}
    if probe_mismatch_best:
        statements.append(
            f"Best paired-probe+mismatch NRMSE is {probe_mismatch_best.get('model')} "
            f"({float(probe_mismatch_best.get('probe_nrmse')):.6g})."
        )
    if mismatch_best:
        statements.append(
            f"Best validation-mismatch RMSE is {mismatch_best.get('model')} "
            f"({float(mismatch_best.get('model_rmse')):.6g})."
        )
    if str(test_best.get("model", "")).startswith("trace1d") and str(probe_best.get("model", "")).startswith("trace1d"):
        recommendation = "treat_trace1d_as_strong_baseline"
        statements.append(
            "Trace-1D remains the strongest current baseline for base and paired-probe metrics."
        )
    else:
        recommendation = "continue_ablation"
    if stability:
        statements.append("Trace-1D seed stability has been measured and is low-variance for current metrics.")
    return {
        "recommendation": recommendation,
        "statements": statements,
    }


def _ablation_metric_notes() -> list[dict[str, str]]:
    return [
        {
            "metric": "zero_x_false_prediction_error",
            "role": "sanity_check",
            "note": (
                "Absolute prediction error on 0x frequency-probe samples. Because 0x samples "
                "share the same no-increment target across probe frequencies, this metric is "
                "not frequency selective and is not used as the primary unsupported-frequency "
                "false-energy gate."
            ),
        },
        {
            "metric": "unsupported_zero_x_false_prediction_error",
            "role": "sanity_check",
            "note": (
                "The unsupported subset verifies probe-frequency catalog filtering, but its "
                "absolute RMSE can match all 0x RMSE when the same 0x target is repeated across "
                "frequency labels. Use unsupported_false_energy for frequency-selective evidence."
            ),
        },
        {
            "metric": "zero_x_false_energy",
            "role": "gate_metric",
            "note": (
                "Weighted sin/cos projection of the 0x residual at the labeled probe frequency. "
                "This is the primary false-frequency-energy diagnostic."
            ),
        },
        {
            "metric": "unsupported_false_energy",
            "role": "gate_metric",
            "note": (
                "Frequency-projection false energy restricted to operator-unsupported 0x probes. "
                "This is the primary unsupported-frequency false-energy diagnostic."
            ),
        },
    ]


def _format_ablation_markdown(report_card: Mapping[str, object]) -> str:
    lines = [
        "# GINN-v2 Model Ablation Report",
        "",
        f"Status: `{report_card['status']}`",
        "",
        "## Conclusion",
    ]
    conclusion = report_card.get("conclusion") or {}
    for statement in conclusion.get("statements", []):
        lines.append(f"- {statement}")
    lines.extend(["", f"Recommendation: `{conclusion.get('recommendation', '')}`", "", "## Coverage"])
    coverage = report_card.get("coverage") or {}
    for key, value in coverage.items():
        mark = "yes" if value else "no"
        lines.append(f"- {key}: {mark}")
    lines.extend(["", "## Metric Notes"])
    for note in report_card.get("metric_notes") or []:
        lines.append(f"- {note.get('metric')}: `{note.get('role')}` - {note.get('note')}")
    missing = report_card.get("missing_for_full_gate") or []
    lines.extend(["", "## Missing For Full Gate"])
    for item in missing:
        lines.append(f"- {item}")
    lines.extend(["", "## Best Rows"])
    for key, row in (report_card.get("best") or {}).items():
        lines.append(f"- {key}: `{row}`")
    lines.extend(["", "## Stability"])
    for key, value in (report_card.get("stability") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _jsonable_dict(row: Mapping[str, object]) -> dict[str, object]:
    clean: dict[str, object] = {}
    for key, value in row.items():
        if pd.isna(value):
            clean[key] = None
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "report":
        run_report(args)
    elif args.command == "summarize":
        run_summarize(args)
    elif args.command == "stamp-gate":
        run_stamp_gate(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
