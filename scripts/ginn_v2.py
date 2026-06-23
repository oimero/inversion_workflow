"""Train, predict, and report GINN-v2 model-ablation baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.utils.io import repo_relative_path, resolve_relative_path, sha256_file, write_json
from ginn_v2.data import (
    PatchSpec,
    build_patch_index,
    compute_input_reference_stats,
    compute_normalization,
    default_eval_kinds,
    default_train_kinds,
)
from ginn_v2.training import predict_patches, report_predictions, train_model


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--benchmark-dir", type=Path, required=True)
    train.add_argument("--model-id", choices=MODEL_IDS, default="patch_2d_supervised")
    train.add_argument("--patch-lateral", type=int, default=32)
    train.add_argument("--patch-twt", type=int, default=128)
    train.add_argument("--lateral-stride", type=int, default=16)
    train.add_argument("--twt-stride", type=int, default=64)
    train.add_argument("--min-valid-fraction", type=float, default=0.50)
    train.add_argument("--split-policy", choices=("derive", "strict"), default="derive")
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--test-fraction", type=float, default=0.15)
    train.add_argument("--max-patches", type=int, default=None)
    train.add_argument("--patch-index", type=Path, default=None)
    train.add_argument("--normalization", type=Path, default=None)
    train.add_argument("--overfit-tiny", action="store_true")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-channels", type=int, default=32)
    train.add_argument("--depth", type=int, default=5)
    train.add_argument("--lambda-physics", type=float, default=0.0)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=20260617)

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


def _timestamped_output(prefix: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "scripts" / "output" / f"{prefix}_{timestamp}"


def _sample_kinds_for_training(model_id: str) -> set[str]:
    return default_train_kinds(model_id)


def _write_train_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    benchmark_dir: Path,
    patch_index_path: Path,
    normalization: dict,
    input_reference_stats_path: Path,
    train_result: dict,
    patch_index_truncated: bool,
    max_patches: int | None,
) -> None:
    manifest = {
        "schema_version": "ginn_v2_model_run_v1",
        "status": "ok",
        "model_id": args.model_id,
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "benchmark_hashes": {
            "synthetic_benchmark.h5": sha256_file(benchmark_dir / "synthetic_benchmark.h5"),
            "sample_index.csv": sha256_file(benchmark_dir / "sample_index.csv"),
            "benchmark_manifest.json": (
                sha256_file(benchmark_dir / "benchmark_manifest.json")
                if (benchmark_dir / "benchmark_manifest.json").is_file()
                else ""
            ),
        },
        "patch_index": repo_relative_path(patch_index_path, root=REPO_ROOT),
        "patch_index_sha256": sha256_file(patch_index_path),
        "patch_index_truncated": bool(patch_index_truncated),
        "max_patches": max_patches,
        "patch_spec": {
            "lateral_samples": int(args.patch_lateral),
            "twt_samples": int(args.patch_twt),
            "lateral_stride": int(args.lateral_stride),
            "twt_stride": int(args.twt_stride),
            "min_valid_fraction": float(args.min_valid_fraction),
        },
        "split_policy": args.split_policy,
        "validation_fraction": float(args.validation_fraction),
        "test_fraction": float(args.test_fraction),
        "normalization": normalization,
        "input_reference_stats": repo_relative_path(input_reference_stats_path, root=REPO_ROOT),
        "input_reference_stats_sha256": sha256_file(input_reference_stats_path),
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
            "physics_loss_applied_sample_kinds": (
                ["base"] if float(args.lambda_physics) > 0.0 else []
            ),
            "physics_forward_operator_id": (
                "nominal_selected_wavelet_forward_log_ai_interior_patch"
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
        },
        "device": train_result.get("device_metadata", {"resolved_device": train_result.get("device", "")}),
        "model_info": train_result["model_info"],
        "checkpoint": repo_relative_path(train_result["checkpoint"], root=REPO_ROOT),
        "training_history": repo_relative_path(train_result["history"], root=REPO_ROOT),
        "best_validation_loss": train_result["best_validation_loss"],
    }
    write_json(output_dir / "model_run_manifest.json", manifest)


def run_train(args: argparse.Namespace) -> None:
    benchmark_dir = resolve_relative_path(args.benchmark_dir, root=REPO_ROOT)
    output_dir = _timestamped_output("ginn_v2_train", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
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
    if args.normalization is not None:
        with resolve_relative_path(args.normalization, root=REPO_ROOT).open("r", encoding="utf-8") as handle:
            normalization = json.load(handle)
    else:
        normalization = compute_normalization(benchmark, patch_index)
    write_json(output_dir / "normalization.json", normalization)
    input_reference_stats = {
        "schema_version": "real_field_input_reference_stats_v1",
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
        "source_sha256": {
            "patch_index": sha256_file(patch_index_path),
            "synthetic_benchmark_h5": sha256_file(benchmark_dir / "synthetic_benchmark.h5"),
            "sample_index_csv": sha256_file(benchmark_dir / "sample_index.csv"),
            "benchmark_manifest_json": (
                sha256_file(benchmark_dir / "benchmark_manifest.json")
                if (benchmark_dir / "benchmark_manifest.json").is_file()
                else ""
            ),
        },
    }
    input_reference_stats_path = output_dir / "input_reference_stats.json"
    write_json(input_reference_stats_path, input_reference_stats)
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
        seed=int(args.seed),
    )
    _write_train_manifest(
        output_dir=output_dir,
        args=args,
        benchmark_dir=benchmark_dir,
        patch_index_path=patch_index_path,
        normalization=normalization,
        input_reference_stats_path=input_reference_stats_path,
        train_result=result,
        patch_index_truncated=patch_index_truncated,
        max_patches=max_patches,
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
    benchmark_dir = (
        resolve_relative_path(args.benchmark_dir, root=REPO_ROOT)
        if args.benchmark_dir is not None
        else resolve_relative_path(manifest["benchmark_dir"], root=REPO_ROOT)
    )
    output_dir = _timestamped_output("ginn_v2_predict", args.output_dir)
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
        patch_index_sha256 = sha256_file(eval_index_path)
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
        patch_index_sha256 = sha256_file(eval_index_path)
    else:
        sample_kinds = set(args.sample_kind) if args.sample_kind else default_eval_kinds()
        spec_cfg = dict(manifest["patch_spec"])
        patch_index = build_patch_index(
            benchmark,
            patch_spec=PatchSpec(
                lateral_samples=int(spec_cfg["lateral_samples"]),
                twt_samples=int(spec_cfg["twt_samples"]),
                lateral_stride=int(spec_cfg["lateral_stride"]),
                twt_stride=int(spec_cfg["twt_stride"]),
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
        patch_index_sha256 = sha256_file(eval_index_path)
    result = predict_patches(
        benchmark=benchmark,
        patch_index=patch_index,
        checkpoint_path=resolve_relative_path(manifest["checkpoint"], root=REPO_ROOT),
        output_dir=output_dir,
        split=args.split,
        batch_size=int(args.batch_size),
        device_name=str(args.device),
    )
    summary = {
        "schema_version": "ginn_v2_prediction_v1",
        "status": "ok",
        "model_run_dir": repo_relative_path(model_run_dir, root=REPO_ROOT),
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "split": args.split,
        "index_source": args.index_source,
        "patch_index": patch_index_source,
        "sample_kinds": sorted(set(patch_index["sample_kind"].astype(str))),
        "model_id": result["model_id"],
        "model_info": result["model_info"],
        "normalization": result["normalization"],
        "input_channels": ["seismic", "lfm_controlled_degraded", "valid_mask_model"],
        "output_semantics": "pred_log_ai = lfm_controlled_degraded + pred_delta_log_ai",
        "outputs": {
            "predictions": repo_relative_path(result["prediction_npz"], root=REPO_ROOT),
            "prediction_index": repo_relative_path(result["prediction_index"], root=REPO_ROOT),
        },
        "benchmark_hashes": manifest.get("benchmark_hashes", {}),
        "patch_index_sha256": patch_index_sha256,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "prediction_sha256": result["prediction_sha256"],
        "device": result.get("device_metadata", {}),
    }
    write_json(output_dir / "prediction_summary.json", summary)
    write_json(output_dir / "prediction_manifest.json", summary)
    print("=== GINN-v2 model ablation predict ===")
    print(f"Output: {output_dir}")
    print(f"Predictions: {result['n_predictions']}")


def run_report(args: argparse.Namespace) -> None:
    prediction_dir = resolve_relative_path(args.prediction_dir, root=REPO_ROOT)
    output_dir = _timestamped_output("ginn_v2_report", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    result = report_predictions(prediction_dir=prediction_dir, output_dir=output_dir)
    summary = {
        "schema_version": "ginn_v2_patch_smoke_report_v1",
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
    output_dir = _timestamped_output("ginn_v2_summary", args.output_dir)
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
                "report_card_sha256": sha256_file(card_path),
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
        "schema_version": "ginn_v2_ablation_summary_v1",
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
        "schema_version": "ginn_v2_ablation_report_card_v1",
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
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
