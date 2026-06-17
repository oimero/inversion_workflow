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

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.synthetic.hashing import sha256_file
from cup.synthetic.metrics import aggregate_metric_rows, metric_row, regression_metrics
from cup.utils.io import repo_relative_path, resolve_relative_path, write_json


SCHEMA_VERSION = "synthoseis_lite_report_v1"
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
        choices=(
            "base",
            "frequency_probe",
            "seismic_variant",
            "frequency_probe_seismic_variant",
        ),
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


def _output_dir(args: argparse.Namespace, benchmark_dir: Path) -> Path:
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


def _probe_metrics(
    benchmark: SynthoseisBenchmark,
    sample_ids: list[str],
    baselines: list[str],
) -> list[dict[str, Any]]:
    probe_ids = [
        sample_id
        for sample_id in sample_ids
        if str(benchmark.row(sample_id).get("sample_kind", "")) == "frequency_probe"
    ]
    if not probe_ids:
        return []
    cache: dict[tuple[str, str], tuple[Any, np.ndarray]] = {}

    def get_prediction(sample_id: str, baseline_id: str) -> tuple[Any, np.ndarray]:
        key = (sample_id, baseline_id)
        if key not in cache:
            sample = benchmark.load_sample(sample_id)
            cache[key] = (sample, _prediction(sample, baseline_id))
        return cache[key]

    rows: list[dict[str, Any]] = []
    for sample_id in probe_ids:
        row = benchmark.row(sample_id)
        pair_id = str(row.get("paired_zero_sample_id") or "").strip()
        amplitude = row.get("probe_amplitude_multiplier", "")
        try:
            amplitude_value = float(amplitude)
        except (TypeError, ValueError):
            amplitude_value = float("nan")
        for baseline_id in baselines:
            sample, prediction = get_prediction(sample_id, baseline_id)
            base = {
                "sample_id": sample_id,
                "baseline_id": baseline_id,
                "probe_frequency_hz": row.get("probe_frequency_hz", ""),
                "probe_phase": row.get("probe_phase", ""),
                "probe_lateral_shape": row.get("probe_lateral_shape", ""),
                "probe_amplitude_multiplier": amplitude,
                "paired_zero_sample_id": pair_id,
            }
            if amplitude_value == 0.0 or not pair_id or pair_id not in benchmark._rows:
                metrics = regression_metrics(
                    sample.target_log_ai,
                    prediction,
                    valid_mask=sample.valid_mask,
                )
                rows.append(
                    {
                        **base,
                        "probe_metric_semantics": "absolute_zero_or_unpaired_probe_error",
                        **metrics,
                    }
                )
                continue
            pair, pair_prediction = get_prediction(pair_id, baseline_id)
            valid = sample.valid_mask & pair.valid_mask
            target_delta = sample.target_log_ai - pair.target_log_ai
            prediction_delta = prediction - pair_prediction
            metrics = regression_metrics(target_delta, prediction_delta, valid_mask=valid)
            rows.append(
                {
                    **base,
                    "probe_metric_semantics": "paired_probe_increment_error",
                    **metrics,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    benchmark_dir = resolve_relative_path(args.benchmark_dir, root=REPO_ROOT)
    benchmark = SynthoseisBenchmark(benchmark_dir)
    kinds = (
        set(args.sample_kind)
        if args.sample_kind
        else {
            "base",
            "frequency_probe",
            "seismic_variant",
            "frequency_probe_seismic_variant",
        }
    )
    baselines = list(dict.fromkeys(args.baseline or BASELINES))
    sample_ids = benchmark.sample_ids(kinds=kinds, status="ok")
    if args.max_samples is not None:
        sample_ids = sample_ids[: int(args.max_samples)]
    if not sample_ids:
        raise ValueError("No benchmark samples selected for evaluation.")
    output_dir = _output_dir(args, benchmark_dir)
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

    probe_rows = _probe_metrics(benchmark, sample_ids, baselines)
    probe_metrics = pd.DataFrame.from_records(probe_rows)
    probe_metrics_path = output_dir / "model_probe_metrics.csv"
    probe_metrics.to_csv(probe_metrics_path, index=False)

    geometry_metrics = _geometry_metrics(sample_metrics)
    geometry_metrics_path = output_dir / "model_geometry_metrics.csv"
    geometry_metrics.to_csv(geometry_metrics_path, index=False)

    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "baseline_ids": baselines,
        "sample_kinds": sorted(kinds),
        "sample_count": len(sample_ids),
        "baseline_aggregate": aggregate_metric_rows(sample_rows),
        "probe_aggregate": aggregate_metric_rows(probe_rows),
        "semantics": {
            "oracle_target": "pipeline self-check only, not a model baseline",
            "lfm_ideal": "ideal lowpass prior copied from benchmark",
            "lfm_controlled_degraded": "field-like degraded LFM prior copied from benchmark",
        },
    }
    report_path = output_dir / "model_report_card.json"
    write_json(report_path, report)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "benchmark_dir": repo_relative_path(benchmark_dir, root=REPO_ROOT),
        "benchmark_files": {
            "synthetic_benchmark.h5": sha256_file(benchmark_dir / "synthetic_benchmark.h5"),
            "sample_index.csv": sha256_file(benchmark_dir / "sample_index.csv"),
            "benchmark_manifest.json": (
                sha256_file(benchmark_dir / "benchmark_manifest.json")
                if (benchmark_dir / "benchmark_manifest.json").is_file()
                else ""
            ),
        },
        "outputs": {
            "model_sample_metrics": repo_relative_path(sample_metrics_path, root=REPO_ROOT),
            "model_probe_metrics": repo_relative_path(probe_metrics_path, root=REPO_ROOT),
            "model_geometry_metrics": repo_relative_path(geometry_metrics_path, root=REPO_ROOT),
            "model_report_card": repo_relative_path(report_path, root=REPO_ROOT),
        },
        "output_hashes": {
            "model_sample_metrics.csv": sha256_file(sample_metrics_path),
            "model_probe_metrics.csv": sha256_file(probe_metrics_path),
            "model_geometry_metrics.csv": sha256_file(geometry_metrics_path),
            "model_report_card.json": sha256_file(report_path),
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
