"""Train, predict, and report GINN-v2 composable experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Mapping

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
src_text = str(SRC_DIR)
sys.path = [path for path in sys.path if path != src_text]
sys.path.insert(0, src_text)

from cup.synthetic.benchmark import SynthoseisBenchmark
from cup.impedance import validate_increment_contract
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
    MODEL_RUN_SCHEMA_VERSION,
    PATCH_SMOKE_REPORT_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
)
from ginn_v2.data import PatchSpec, build_patch_index, default_eval_kinds
from ginn_v2.evaluation import predict_patches, report_predictions
from ginn_v2.composable import run_experiment
from ginn_v2.experiment import parse_experiment_config
from ginn_v2.runtime import configure_training_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=None)

    predict = sub.add_parser("predict")
    predict.add_argument("--model-run-dir", type=Path, required=True)
    predict.add_argument("--benchmark-dir", type=Path, default=None)
    predict.add_argument("--index-source", choices=("train", "eval"), default="train")
    predict.add_argument("--eval-patch-index", type=Path, default=None)
    predict.add_argument("--sample-kind", action="append", choices=(
        "base",
        "seismic_variant",
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

    return parser.parse_args()


def _resolve_output_dir(prefix: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "scripts" / "output" / f"{prefix}_{timestamp}"


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


def _benchmark_contract_payload(benchmark: SynthoseisBenchmark) -> dict[str, object]:
    contract = validate_increment_contract(
        benchmark.manifest.get("increment_contract") or {}
    ).as_dict()
    sample_axis_contract = {
        "sample_domain": contract["sample_domain"],
        "sample_unit": contract["sample_unit"],
        "depth_basis": contract.get("depth_basis"),
        "sample_step": contract["sample_interval"],
        "axis_direction": "increasing",
        "axis_regularity": "regular",
    }
    return {
        "schema_version": str(benchmark.schema),
        "sample_domain": str(benchmark.sample_domain),
        "increment_contract": contract,
        "sample_axis_contract": sample_axis_contract,
        "contract_fingerprint_sha256": require_contract_fingerprint(
            benchmark.manifest, label=f"benchmark {benchmark.run_dir}"
        ),
    }


def _validate_checkpoint_benchmark_compatibility(
    checkpoint: Mapping[str, object], benchmark: SynthoseisBenchmark,
) -> dict[str, object]:
    actual = _benchmark_contract_payload(benchmark)
    checkpoint_contract = validate_increment_contract(
        checkpoint.get("increment_contract") or {}
    ).as_dict()
    if checkpoint_contract != actual["increment_contract"]:
        raise ValueError(
            "Prediction benchmark increment_contract does not exactly match the checkpoint contract."
        )
    checkpoint_axis = dict(checkpoint.get("sample_axis_contract") or {})
    if checkpoint_axis != actual["sample_axis_contract"]:
        raise ValueError(
            "Prediction benchmark sample-axis contract does not exactly match the checkpoint contract."
        )
    return actual


def run_train(args: argparse.Namespace) -> None:
    if args.config is None:
        raise ValueError(
            "GINN-v2 training now requires --config with the ginn_v2_experiment_v2 root. "
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
    deployment = dict(manifest["deployment_checkpoint"])
    if deployment.get("eligible", True):
        print(f"Deployment: {deployment['path']}")
    else:
        print(
            "Checkpoint (smoke-only; deployment_eligible=false): "
            f"{deployment['path']}"
        )


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
    benchmark_contract = _validate_checkpoint_benchmark_compatibility(
        checkpoint=torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        benchmark=benchmark,
    )
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
        spec_cfg = dict(manifest.get("patching") or manifest.get("patch_spec") or {})
        if not spec_cfg:
            raise ValueError("Model run manifest lacks patching contract.")
        patch_index = build_patch_index(
            benchmark,
            patch_spec=PatchSpec(
                lateral_samples=int(spec_cfg["lateral_samples"]),
                twt_samples=int(spec_cfg["vertical_samples"]),
                lateral_stride=int(spec_cfg["lateral_stride"]),
                twt_stride=int(spec_cfg["vertical_stride"]),
            ),
            sample_kinds=sample_kinds,
            split_policy=str(manifest.get("split_policy", "derive")),
            validation_fraction=float(manifest.get("validation_fraction", 0.15)),
            test_fraction=float(manifest.get("test_fraction", 0.15)),
            min_valid_samples=1,
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
    model_contract = {
        "path": repo_relative_path(model_run_dir / "model_run_manifest.json", root=REPO_ROOT),
    }
    if str(manifest.get("contract_fingerprint_sha256") or ""):
        model_contract["contract_fingerprint_sha256"] = require_contract_fingerprint(
            manifest, label=f"model run {model_run_dir}"
        )
    input_contracts = {
        "model_run": model_contract,
        "benchmark": benchmark_contract,
    }
    prediction_contract_fingerprint = None
    if all(
        str(value.get("contract_fingerprint_sha256") or "")
        for value in input_contracts.values()
        if isinstance(value, Mapping)
    ):
        prediction_contract_fingerprint = contract_fingerprint_sha256(
            contract_schema_version=PREDICTION_SCHEMA_VERSION,
            semantics={
                "architecture_id": result["architecture_id"],
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
        "input_contracts": input_contracts,
        "experiment_id": str(manifest.get("experiment_id") or ""),
        "run_mode": str(manifest.get("run_mode") or "standard"),
        "development_limited": bool(manifest.get("development_limited", False)),
        "deployment_eligible": bool(manifest.get("deployment_eligible", False)),
        "model_run_dir": repo_relative_path(model_run_dir, root=REPO_ROOT),
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "benchmark_contract": benchmark_contract,
        "split": args.split,
        "index_source": args.index_source,
        "patch_index": patch_index_source,
        "sample_kinds": sorted(set(patch_index["sample_kind"].astype(str))),
        "architecture_id": result["architecture_id"],
        "checkpoint_selection": str(args.checkpoint),
        "resolved_checkpoint": checkpoint_name,
        "checkpoint": repo_relative_path(checkpoint_path, root=REPO_ROOT),
        "model_info": result["model_info"],
        "normalization": result["normalization"],
        "input_channels": ["seismic", "input_lfm_log_ai", "valid_mask"],
        "output_semantics": (
            "predicted_log_ai = input_lfm_log_ai + predicted_increment_log_ai"
        ),
        "outputs": {
            "predictions": repo_relative_path(result["prediction_npz"], root=REPO_ROOT),
            "prediction_index": repo_relative_path(result["prediction_index"], root=REPO_ROOT),
        },
        "device": result.get("device_metadata", {}),
    }
    if prediction_contract_fingerprint is not None:
        summary["contract_fingerprint_schema"] = CONTRACT_FINGERPRINT_SCHEMA
        summary["contract_fingerprint_sha256"] = prediction_contract_fingerprint
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
    for spec in args.report:
        model, scope, report_dir = _parse_report_spec(spec)
        report_path = resolve_relative_path(report_dir, root=REPO_ROOT)
        experiment_metadata = _load_report_experiment_metadata(report_path)
        card_path = report_path / "model_report_card.json"
        with card_path.open("r", encoding="utf-8") as handle:
            card = json.load(handle)
        aggregate = dict(card.get("aggregate") or {})
        increment_aggregate = dict(card.get("increment_aggregate") or {})
        canonical_closure_aggregate = dict(card.get("canonical_closure_aggregate") or {})
        lfm_aggregate = dict(card.get("lfm_aggregate") or {})
        lfm_ideal_aggregate = dict(card.get("lfm_ideal_aggregate") or {})
        oracle_aggregate = dict(card.get("oracle_aggregate") or {})
        geometry_aggregate = dict(card.get("geometry_aggregate") or {})
        geometry_holdout_aggregate = dict(card.get("geometry_holdout_aggregate") or {})
        realization_uniform_aggregate = dict(card.get("realization_uniform_aggregate") or {})
        realization_center_aggregate = dict(card.get("realization_center_crop_aggregate") or {})
        rows.append(
            {
                "model": model,
                "scope": scope,
                "experiment_id": experiment_metadata["experiment_id"],
                "architecture_id": experiment_metadata["architecture_id"],
                "architecture_family": experiment_metadata["architecture_family"],
                "loss_kinds": experiment_metadata["loss_kinds"],
                "has_physics_loss": experiment_metadata["has_physics_loss"],
                "has_mismatch_training": experiment_metadata["has_mismatch_training"],
                "metadata_source": experiment_metadata["metadata_source"],
                "n_patches": card.get("n_patches"),
                "model_rmse": aggregate.get("mean_rmse"),
                "model_nrmse": aggregate.get("mean_nrmse"),
                "model_corr": aggregate.get("median_corr"),
                "increment_rmse": increment_aggregate.get("mean_rmse"),
                "increment_nrmse": increment_aggregate.get("mean_nrmse"),
                "increment_corr": increment_aggregate.get("median_corr"),
                "canonical_closure_rmse": canonical_closure_aggregate.get("mean_rmse"),
                "canonical_closure_nrmse": canonical_closure_aggregate.get("mean_nrmse"),
                "canonical_closure_corr": canonical_closure_aggregate.get("median_corr"),
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
                "geometry_n_ok": geometry_aggregate.get("n_ok"),
                "geometry_boundary_rmse": geometry_aggregate.get("mean_boundary_rmse"),
                "geometry_event_rmse": geometry_aggregate.get("mean_event_rmse"),
                "geometry_lateral_gradient_rmse": geometry_aggregate.get(
                    "mean_lateral_gradient_rmse"
                ),
                "geometry_holdout_n_ok": geometry_holdout_aggregate.get("n_ok"),
                "geometry_holdout_rmse": geometry_holdout_aggregate.get("mean_rmse"),
                "realization_uniform_n_ok": realization_uniform_aggregate.get("n_ok"),
                "realization_uniform_rmse": realization_uniform_aggregate.get("mean_rmse"),
                "realization_center_crop_n_ok": realization_center_aggregate.get("n_ok"),
                "realization_center_crop_rmse": realization_center_aggregate.get("mean_rmse"),
                "report_dir": repo_relative_path(report_path, root=REPO_ROOT),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    summary_path = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    canonical_metrics_path = output_dir / "canonical_metrics_by_report.csv"
    summary.to_csv(canonical_metrics_path, index=False)
    report_card = _build_canonical_ablation_report_card(summary, canonical_metrics_path)
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
            "canonical_metrics_by_report": repo_relative_path(canonical_metrics_path, root=REPO_ROOT),
            "ablation_report_card": repo_relative_path(report_card_path, root=REPO_ROOT),
            "ablation_report": repo_relative_path(markdown_path, root=REPO_ROOT),
        },
        "reports": rows,
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print("=== GINN-v2 model ablation summarize ===")
    print(f"Output: {output_dir}")
    print(f"Reports: {len(rows)}")


def _load_report_experiment_metadata(report_path: Path) -> dict[str, object]:
    """Load architecture/training semantics from the report's model manifest.

    The user-provided report label remains a display label only. New coverage
    decisions use the manifest that produced the prediction, so arbitrary
    experiment IDs do not change the report card.
    """
    metadata: dict[str, object] = {
        "experiment_id": "",
        "architecture_id": "",
        "architecture_family": "",
        "loss_kinds": "",
        "has_physics_loss": False,
        "has_mismatch_training": False,
        "metadata_source": "unavailable",
    }
    prediction_summary: dict[str, object] = {}
    report_summary_path = report_path / "report_summary.json"
    if report_summary_path.is_file():
        with report_summary_path.open("r", encoding="utf-8") as handle:
            report_summary = json.load(handle)
        prediction_dir_value = str(report_summary.get("prediction_dir") or "").strip()
        if prediction_dir_value:
            prediction_dir = resolve_relative_path(prediction_dir_value, root=REPO_ROOT)
            for name in ("prediction_summary.json", "prediction_manifest.json"):
                candidate = prediction_dir / name
                if candidate.is_file():
                    with candidate.open("r", encoding="utf-8") as handle:
                        prediction_summary = json.load(handle)
                    break
    if not prediction_summary:
        for name in ("prediction_summary.json", "prediction_manifest.json"):
            candidate = report_path / name
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    prediction_summary = json.load(handle)
                break
    metadata["experiment_id"] = str(prediction_summary.get("experiment_id") or "")
    metadata["architecture_id"] = str(prediction_summary.get("architecture_id") or "")
    model_run_dir_value = str(prediction_summary.get("model_run_dir") or "").strip()
    manifest: dict[str, object] = {}
    if model_run_dir_value:
        model_run_dir = resolve_relative_path(model_run_dir_value, root=REPO_ROOT)
        for name in ("model_run_manifest.json", "experiment_manifest.json"):
            candidate = model_run_dir / name
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                metadata["metadata_source"] = "model_run_manifest"
                break
    if not manifest:
        if metadata["architecture_id"]:
            metadata["metadata_source"] = "prediction_summary"
        return metadata

    metadata["experiment_id"] = str(manifest.get("experiment_id") or metadata["experiment_id"])
    architecture = manifest.get("architecture")
    if isinstance(architecture, Mapping):
        metadata["architecture_id"] = str(architecture.get("id") or metadata["architecture_id"])
    architecture_id = str(metadata["architecture_id"])
    if architecture_id in {"trace_conv1d", "trace_dilated_tcn", "trace_lateral_mixer"}:
        metadata["architecture_family"] = "trace"
    elif architecture_id == "patch_conv2d":
        metadata["architecture_family"] = "patch"

    sources = manifest.get("sources")
    source_map = dict(sources) if isinstance(sources, Mapping) else {}
    loss_kinds: set[str] = set()
    has_mismatch_training = False
    stages = manifest.get("stages")
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            blocks = stage.get("loss_blocks")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, Mapping):
                    continue
                kind = str(block.get("kind") or "").strip()
                if kind:
                    loss_kinds.add(kind)
                source = source_map.get(str(block.get("source") or ""), {})
                if isinstance(source, Mapping) and str(
                    source.get("input_seismic_variant") or ""
                ).casefold() == "observed_mismatch":
                    has_mismatch_training = True
    if not has_mismatch_training:
        for source in source_map.values():
            if isinstance(source, Mapping) and str(
                source.get("input_seismic_variant") or ""
            ).casefold() == "observed_mismatch":
                has_mismatch_training = True
                break
    metadata["loss_kinds"] = ",".join(sorted(loss_kinds))
    metadata["has_physics_loss"] = "physics" in loss_kinds
    metadata["has_mismatch_training"] = has_mismatch_training
    return metadata


def _parse_report_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":", maxsplit=2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError("--report must have form MODEL:SCOPE:REPORT_DIR")
    return parts[0].strip(), parts[1].strip(), Path(parts[2].strip())


def _canonical_ablation_report_card(
    summary: pd.DataFrame, frequency_path: Path
) -> dict[str, object]:
    """Build the v4 coverage card from manifest/closure fields only.

    Frequency probes are not part of the v4 canonical benchmark.  Keeping the
    coverage gate on increment and closure metrics makes arbitrary experiment
    IDs harmless and prevents the historical probe report from becoming a
    deployment gate.
    """
    def _has_numeric(column: str) -> bool:
        return bool(
            column in summary
            and pd.to_numeric(summary[column], errors="coerce").notna().any()
        )

    architecture_ids = set(summary.get("architecture_id", pd.Series(dtype=str)).astype(str))
    coverage = {
        "increment_fidelity": _has_numeric("increment_rmse"),
        "canonical_closure": _has_numeric("canonical_closure_rmse"),
        "deployment_closure": _has_numeric("model_rmse"),
        "lfm_only_baseline": _has_numeric("lfm_rmse"),
        "geometry_holdout": bool(
            pd.to_numeric(summary.get("geometry_holdout_n_ok", pd.Series(dtype=float)), errors="coerce")
            .fillna(0).gt(0).any()
        ),
        "all_four_architectures": architecture_ids.issuperset(
            {"trace_conv1d", "trace_dilated_tcn", "trace_lateral_mixer", "patch_conv2d"}
        ),
    }
    required = ["increment_fidelity", "canonical_closure", "deployment_closure", "lfm_only_baseline"]
    missing = [key for key in required if not coverage[key]]
    status = "canonical_increment_ready" if not missing else "partial"
    best = {
        "best_increment_by_rmse": _best_row(summary, metric="increment_rmse", ascending=True),
        "best_canonical_closure_by_rmse": _best_row(summary, metric="canonical_closure_rmse", ascending=True),
        "best_deployment_closure_by_rmse": _best_row(summary, metric="model_rmse", ascending=True),
    }
    statements = [
        "Coverage is derived from checkpoint manifests and explicit closure metrics; experiment-id text is descriptive only.",
        "The v4 canonical benchmark has no frequency-probe coverage gate.",
    ]
    return {
        "schema_version": ABLATION_REPORT_CARD_SCHEMA_VERSION,
        "status": status,
        "coverage": coverage,
        "required_for_full_gate": required,
        "missing_for_full_gate": missing,
        "best": best,
        "stability": {},
        "conclusion": {"recommendation": "continue_canonical_increment_ablation", "statements": statements},
        "source_frequency_table": repo_relative_path(frequency_path, root=REPO_ROOT),
    }


_build_canonical_ablation_report_card = _canonical_ablation_report_card


def _best_row(frame: pd.DataFrame, *, metric: str, ascending: bool) -> dict[str, object] | None:
    if frame.empty or metric not in frame:
        return None
    valid = frame[pd.to_numeric(frame[metric], errors="coerce").notna()].copy()
    if valid.empty:
        return None
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    row = valid.sort_values(metric, ascending=ascending).iloc[0].to_dict()
    return _jsonable_dict(row)


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
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
