"""Evaluate simple baselines on a frozen synthoseis-lite benchmark.

This script is intentionally model-free. It validates the consumer contract before
GINN consumes the same benchmark interface.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.synthetic.benchmark import SynthoseisBenchmark
from cup.synthetic.schemas import REPORT_SCHEMA_VERSION
from cup.synthetic.reporting.metrics import aggregate_metric_rows, metric_row
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    write_json,
)


BASELINES = ("lfm_controlled_degraded", "lfm_ideal", "oracle_target")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Directory containing synthetic_benchmark.h5 and sample_index.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--sample-kind",
        action="append",
        choices=("base", "seismic_variant"),
        default=None,
        help="Restrict evaluation to one or more sample kinds.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=BASELINES,
        default=None,
        help="Restrict evaluated baseline ids.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Development smoke-test cap after sample-kind filtering.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace, benchmark_dir: Path) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return benchmark_dir.parent / f"synthoseis_lite_evaluate_{timestamp}"


def _prediction(sample: Any, baseline_id: str) -> np.ndarray:
    if baseline_id == "oracle_target":
        return sample.target_log_ai.copy()
    if baseline_id not in sample.priors:
        raise ValueError(f"Unsupported baseline: {baseline_id}")
    return sample.priors[baseline_id]


def _geometry_metrics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    if sample_metrics.empty:
        return pd.DataFrame(
            columns=[
                "baseline_id",
                "suite",
                "geometry_family",
                "n_samples",
                "mean_rmse",
                "mean_nrmse",
                "median_corr",
            ]
        )
    base = sample_metrics[
        sample_metrics["sample_kind"].eq("base") & sample_metrics["status"].eq("ok")
    ].copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "baseline_id",
                "suite",
                "geometry_family",
                "n_samples",
                "mean_rmse",
                "mean_nrmse",
                "median_corr",
            ]
        )
    grouped = (
        base.groupby(["baseline_id", "suite", "geometry_family"], dropna=False)
        .agg(
            n_samples=("sample_id", "count"),
            mean_rmse=("rmse", "mean"),
            mean_nrmse=("nrmse", "mean"),
            median_corr=("corr", "median"),
        )
        .reset_index()
    )
    grouped["metric_semantics"] = "base_sample_impedance_error_grouped_by_geometry_family"
    return grouped


def main() -> None:
    args = parse_args()
    benchmark_dir = resolve_relative_path(args.benchmark_dir, root=REPO_ROOT)
    benchmark = SynthoseisBenchmark(benchmark_dir)
    kinds = (
        set(args.sample_kind)
        if args.sample_kind
        else {
            "base",
            "seismic_variant",
        }
    )
    baselines = list(dict.fromkeys(args.baseline or BASELINES))
    sample_ids = benchmark.sample_ids(kinds=kinds, status="ok")
    if args.max_samples is not None:
        sample_ids = sample_ids[: int(args.max_samples)]
    if not sample_ids:
        raise ValueError("No benchmark samples selected for evaluation.")
    output_dir = _resolve_output_dir(args, benchmark_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    sample_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        sample = benchmark.load_sample(sample_id)
        for baseline_id in baselines:
            prediction = _prediction(sample, baseline_id)
            sample_rows.append(
                metric_row(
                    sample_row=sample.row,
                    baseline_id=baseline_id,
                    target=sample.target_log_ai,
                    prediction=prediction,
                    valid_mask=sample.valid_mask,
                )
            )
    sample_metrics = pd.DataFrame.from_records(sample_rows)
    sample_metrics_path = output_dir / "model_sample_metrics.csv"
    sample_metrics.to_csv(sample_metrics_path, index=False)

    geometry_metrics = _geometry_metrics(sample_metrics)
    geometry_metrics_path = output_dir / "model_geometry_metrics.csv"
    geometry_metrics.to_csv(geometry_metrics_path, index=False)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "baseline_ids": baselines,
        "sample_kinds": sorted(kinds),
        "sample_count": len(sample_ids),
        "baseline_aggregate": aggregate_metric_rows(sample_rows),
        "semantics": {
            "oracle_target": "pipeline self-check only, not a model baseline",
            "lfm_ideal": "ideal lowpass prior copied from benchmark",
            "lfm_controlled_degraded": "field-like degraded LFM prior copied from benchmark",
        },
    }
    report_path = output_dir / "model_report_card.json"
    write_json(report_path, report)

    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    with benchmark_manifest_path.open("r", encoding="utf-8") as handle:
        benchmark_manifest = json.load(handle)
    input_contracts = {
        "benchmark": {
            "path": repo_relative_path(benchmark_manifest_path, root=REPO_ROOT),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                benchmark_manifest, label=f"benchmark {benchmark_dir}"
            ),
        }
    }
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=REPORT_SCHEMA_VERSION,
        semantics={"baseline_ids": baselines, "sample_kinds": sorted(kinds)},
        business_config={"sample_count": len(sample_ids)},
        input_contracts=input_contracts,
        primary_artifacts={
            "model_sample_metrics": sample_metrics_path,
            "model_geometry_metrics": geometry_metrics_path,
            "model_report_card": report_path,
        },
    )
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "ok",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "outputs": {
            "model_sample_metrics": repo_relative_path(sample_metrics_path, root=REPO_ROOT),
            "model_geometry_metrics": repo_relative_path(geometry_metrics_path, root=REPO_ROOT),
            "model_report_card": repo_relative_path(report_path, root=REPO_ROOT),
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    write_json(summary_path, summary)
    print("=== synthoseis-lite baseline evaluation ===")
    print(f"Benchmark: {benchmark_dir}")
    print(f"Output: {output_dir}")
    print(f"Samples: {len(sample_ids)}")
    print(f"Baselines: {', '.join(baselines)}")


if __name__ == "__main__":
    main()
