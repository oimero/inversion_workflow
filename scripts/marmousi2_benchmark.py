"""Run numerical Marmousi2 evaluation for one or more saved predictions.

The script intentionally keeps training orchestration outside the benchmark
adapter.  A normal GINN run can be evaluated by passing its saved volume, or
``--checkpoint`` can ask the existing ``load_body``/``predict_volume`` API to
produce that volume first.  A manifest makes multi-seed aggregation explicit:
each row has ``label`` and ``prediction_npz`` and may optionally provide a
``lfm_npz`` file containing ``log_ai``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marmousi2.evaluation import evaluate_marmousi2_prediction


def _resolve(path: str | Path, *, base: Path = ROOT) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _load_array(path: Path, *, preferred: tuple[str, ...] = ("body_log_ai", "log_ai", "prediction")) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in preferred:
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"{path} does not contain one of {preferred}.")


def _save_volume(path: Path, result: Any) -> np.ndarray:
    values = np.asarray(result.body_log_ai, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        body_log_ai=values,
        direction_count=np.asarray(result.direction_count),
        fill_code=np.asarray(result.fill_code),
        inline_indices=np.asarray(result.inline_indices),
        xline_indices=np.asarray(result.xline_indices),
    )
    return values.astype(np.float64)


def _flatten_metrics(label: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if "support_count" in value and any(key in value for key in ("rmse", "mean", "median", "waveform_corr")):
                rows.append({
                    "label": label,
                    "metric": path,
                    "rmse": value.get("rmse"),
                    "bias": value.get("bias"),
                    "corr": value.get("corr"),
                    "waveform_corr": value.get("waveform_corr"),
                    "mean_waveform_corr": value.get("mean_waveform_corr"),
                    "mean": value.get("mean"),
                    "median": value.get("median"),
                    "p95": value.get("p95"),
                    "units": value.get("units"),
                    "support_count": value.get("support_count"),
                })
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
    walk(result, "")
    return rows


def _evaluate_one(
    prepared_dir: Path,
    prediction_path: Path,
    *,
    lfm_path: Path | None,
    lfm_log_ai: np.ndarray | None,
    scope: str,
    support_threshold: float,
    cutoff_hz: float,
    shortwave_cutoff_hz: float,
) -> dict[str, Any]:
    prediction = _load_array(prediction_path)
    if lfm_path is not None:
        lfm = _load_array(lfm_path, preferred=("log_ai", "lfm_log_ai", "prediction"))
    elif lfm_log_ai is not None:
        lfm = lfm_log_ai
    else:
        raise ValueError("An LFM array is required; pass --lfm-npz or use --checkpoint.")
    result = evaluate_marmousi2_prediction(
        prepared_dir,
        prediction,
        lfm,
        scope=scope,
        cutoff_hz=cutoff_hz,
        shortwave_cutoff_hz=shortwave_cutoff_hz,
        support_relative_threshold=support_threshold,
    )
    result["prediction_path"] = str(prediction_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("opendata/prepared/marmousi2_postm_time"))
    parser.add_argument("--prediction-npz", type=Path, default=None,
                        help="NPZ containing body_log_ai/log_ai/prediction.")
    parser.add_argument("--lfm-npz", type=Path, default=None,
                        help="NPZ containing the LFM log_ai array.")
    parser.add_argument("--prediction-manifest", type=Path, default=None,
                        help="CSV with label,prediction_npz[,lfm_npz] for multi-seed aggregation.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--scope", choices=("validation", "final"), default="final")
    parser.add_argument("--support-relative-threshold", type=float, default=0.25)
    parser.add_argument("--cutoff-hz", type=float, default=5.0)
    parser.add_argument("--shortwave-cutoff-hz", type=float, default=20.0)
    parser.add_argument("--comparison-plan", action="store_true",
                        help="Materialize or execute the small explicit baseline/P0/MW/MVW/W-only plan.")
    parser.add_argument("--run-training", action="store_true",
                        help="Execute the selected comparison plan through the main GINN workflow.")
    parser.add_argument("--candidates", nargs="+",
                        choices=("baseline", "p0-random", "mw", "mvw-low-visible", "w-only"),
                        default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=(20261004,))
    parser.add_argument("--visible-weight", type=float, default=0.25)
    parser.add_argument("--unlabeled-steps", type=int, default=256,
                        help="Fixed U budget for the explicit M/V/MVW candidates.")
    parser.add_argument("--well-steps", type=int, default=256,
                        help="Fixed W budget shared by the explicit candidates.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional GINN config used with --checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional existing GINN checkpoint. Inference uses load_body/predict_volume.")
    parser.add_argument("--inference-output", type=Path, default=None)
    args = parser.parse_args()

    prepared = _resolve(args.prepared_dir)
    if args.comparison_plan:
        from marmousi2.benchmark import run_comparison_plan, write_comparison_plan

        if args.config is None:
            raise ValueError("--config is required with --comparison-plan.")
        plan_dir = _resolve(args.output_json).parent if args.output_json is not None else prepared / "benchmark"
        if args.run_training:
            payload = run_comparison_plan(
                _resolve(args.config),
                prepared,
                plan_dir,
                repo_root=ROOT,
                candidates=args.candidates,
                seeds=args.seeds,
                visible_weight=args.visible_weight,
                unlabeled_steps=args.unlabeled_steps,
                well_steps=args.well_steps,
                scope=args.scope,
                support_relative_threshold=args.support_relative_threshold,
            )
            print(f"Comparison results: {plan_dir / 'results.json'}")
        else:
            payload = write_comparison_plan(
                _resolve(args.config),
                plan_dir,
                repo_root=ROOT,
                candidates=args.candidates,
                seeds=args.seeds,
                visible_weight=args.visible_weight,
                unlabeled_steps=args.unlabeled_steps,
                well_steps=args.well_steps,
            )
            print(f"Comparison plan: {payload['manifest']}")
        return

    entries: list[tuple[str, Path, Path | None]] = []
    lfm_override: np.ndarray | None = None
    global_lfm_path = None if args.lfm_npz is None else _resolve(args.lfm_npz)
    if args.prediction_manifest is not None:
        manifest_path = _resolve(args.prediction_manifest)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("label") or row.get("seed") or "run").strip()
                prediction_text = str(row.get("prediction_npz") or "").strip()
                if not prediction_text:
                    raise ValueError("prediction manifest rows require prediction_npz.")
                lfm_text = str(row.get("lfm_npz") or "").strip()
                entries.append((
                    label,
                    _resolve(prediction_text, base=manifest_path.parent),
                    _resolve(lfm_text, base=manifest_path.parent) if lfm_text else None,
                ))
    elif args.prediction_npz is not None:
        entries.append(("run", _resolve(args.prediction_npz), None if args.lfm_npz is None else _resolve(args.lfm_npz)))
    elif args.checkpoint is not None:
        if args.config is None:
            raise ValueError("--config is required with --checkpoint.")
        from ginn_v2.workflow import load_body
        loaded = load_body(_resolve(args.config), checkpoint=_resolve(args.checkpoint))
        output = _resolve(args.inference_output or (prepared / "benchmark" / "prediction.npz"))
        values = _save_volume(output, loaded.predict_volume())
        # The loaded contract supplies the exact selected LFM without asking
        # callers to duplicate its path in a second argument.
        lfm = np.asarray(loaded.lfm.log_ai, dtype=np.float64)
        entries.append(("checkpoint", output, None))
        lfm_override = lfm
    else:
        raise ValueError("Pass --prediction-npz, --prediction-manifest, or --checkpoint.")

    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for label, prediction_path, lfm_path in entries:
        result = _evaluate_one(
            prepared,
            prediction_path,
            lfm_path=lfm_path or global_lfm_path,
            lfm_log_ai=lfm_override,
            scope=args.scope,
            support_threshold=args.support_relative_threshold,
            cutoff_hz=args.cutoff_hz,
            shortwave_cutoff_hz=args.shortwave_cutoff_hz,
        )
        result["label"] = label
        results[label] = result
        rows.extend(_flatten_metrics(label, result))
        lfm_override = None

    output_json = _resolve(args.output_json) if args.output_json else prepared / "benchmark" / "results.json"
    output_csv = _resolve(args.output_csv) if args.output_csv else prepared / "benchmark" / "results.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Results: {output_json}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
