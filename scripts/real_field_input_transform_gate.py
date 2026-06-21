"""Run R0/R1 for candidate real-field seismic input transforms."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _timestamped_output(prefix: str, explicit: Path | None, *, output_root: Path) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{prefix}_{timestamp}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _run_command(command: list[str], *, log_path: Path) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")


def _write_variant_config(base_cfg: dict, *, variant: str, path: Path) -> None:
    cfg = json.loads(json.dumps(base_cfg))
    r0 = dict(cfg.get("real_field_zero_shot") or {})
    inputs = dict(r0.get("real_field_inputs") or {})
    inputs["seismic_value_transform"] = "identity" if variant == "raw" else variant
    r0["real_field_inputs"] = inputs
    cfg["real_field_zero_shot"] = r0
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)


def _prediction_stats(r0_dir: Path, *, variant: str) -> list[dict[str, object]]:
    rows = []
    for model_dir in sorted(path for path in r0_dir.iterdir() if path.is_dir()):
        npz_path = model_dir / "predictions.npz"
        if not npz_path.is_file():
            continue
        arrays = np.load(npz_path, allow_pickle=True)
        valid = arrays["valid_mask_model"].astype(bool) & (arrays["stitching_weight"] > 0.0)
        for signal in ("stitched_pred_log_ai", "pred_delta_vs_lfm"):
            values = np.asarray(arrays[signal], dtype=np.float64)
            data = values[valid & np.isfinite(values)]
            if data.size:
                rows.append(
                    {
                        "variant": variant,
                        "model_role": model_dir.name,
                        "signal": signal,
                        "mean": float(np.mean(data)),
                        "std": float(np.std(data)),
                        "p01": float(np.quantile(data, 0.01)),
                        "p50": float(np.quantile(data, 0.50)),
                        "p99": float(np.quantile(data, 0.99)),
                        "max_abs": float(np.max(np.abs(data))),
                    }
                )
    return rows


def _first_numeric(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.iloc[0]) if not values.empty else float("nan")


def _summarize_variant(*, variant: str, r0_dir: Path, r1_dir: Path) -> dict[str, object]:
    input_qc = pd.read_csv(r0_dir / "model_input_qc.csv")
    spectral = pd.read_csv(r0_dir / "real_field_spectral_qc.csv")
    lateral_band = pd.read_csv(r0_dir / "lateral_difference_band_qc.csv")
    metrics = pd.read_csv(r1_dir / "forward_diagnostic_metrics.csv")
    wells = pd.read_csv(r1_dir / "well_forward_diagnostic.csv")
    with (r1_dir / "real_field_forward_diagnostic_summary.json").open("r", encoding="utf-8") as handle:
        r1_summary = json.load(handle)
    seismic_mask = input_qc["input"].astype(str).eq("seismic")
    highfreq_mask = lateral_band["band"].astype(str).eq("highfreq_or_nullspace")
    ph3 = wells[wells["well_name"].astype(str).eq("PH3")]
    no_lateral_ph3 = ph3[ph3["model_role"].astype(str).eq("no_lateral")]
    lateral_ph3 = ph3[ph3["model_role"].astype(str).eq("lateral")]
    return {
        "variant": variant,
        "r0_dir": repo_relative_path(r0_dir, root=REPO_ROOT),
        "r1_dir": repo_relative_path(r1_dir, root=REPO_ROOT),
        "seismic_fraction_abs_gt_5sigma": _first_numeric(input_qc, seismic_mask, "fraction_abs_normalized_gt_5"),
        "seismic_normalized_std": _first_numeric(input_qc, seismic_mask, "normalized_std"),
        "lateral_nullspace_energy_ratio": _first_numeric(lateral_band, highfreq_mask, "energy_ratio"),
        "zero_shot_lateral_synthetic_rms": _first_numeric(
            metrics, metrics["model_role"].astype(str).eq("zero_shot_lateral"), "synthetic_rms_before_scale"
        ),
        "zero_shot_no_lateral_synthetic_rms": _first_numeric(
            metrics, metrics["model_role"].astype(str).eq("zero_shot_no_lateral"), "synthetic_rms_before_scale"
        ),
        "zero_shot_lateral_residual_rms_scaled": _first_numeric(
            metrics, metrics["model_role"].astype(str).eq("zero_shot_lateral"), "residual_rms_scaled"
        ),
        "zero_shot_no_lateral_residual_rms_scaled": _first_numeric(
            metrics, metrics["model_role"].astype(str).eq("zero_shot_no_lateral"), "residual_rms_scaled"
        ),
        "ph3_lateral_well_ai_rmse": _first_numeric(lateral_ph3, lateral_ph3.index == lateral_ph3.index, "well_ai_rmse") if not lateral_ph3.empty else float("nan"),
        "ph3_no_lateral_well_ai_rmse": _first_numeric(no_lateral_ph3, no_lateral_ph3.index == no_lateral_ph3.index, "well_ai_rmse") if not no_lateral_ph3.empty else float("nan"),
        "red_flag_count": len(r1_summary.get("red_flags") or []),
        "recommended_next_state": r1_summary.get("recommended_next_state", ""),
    }


def _recommend(frame: pd.DataFrame) -> dict[str, object]:
    candidates = frame[frame["variant"].astype(str).ne("raw")].copy()
    if candidates.empty:
        return {"recommended_transform": "", "reason": "no non-raw candidates"}
    candidates["score"] = (
        candidates["seismic_fraction_abs_gt_5sigma"].fillna(1.0) * 100.0
        + candidates["lateral_nullspace_energy_ratio"].fillna(1.0) * 10.0
        + np.abs(candidates["seismic_normalized_std"].fillna(1.0) - 1.0)
    )
    chosen = candidates.sort_values(["score", "seismic_fraction_abs_gt_5sigma"]).iloc[0].to_dict()
    return {
        "recommended_transform": chosen["variant"],
        "reason": (
            "Lowest combined diagnostic score among non-raw candidates: "
            "normalized outlier fraction, lateral null-space ratio, and normalized std closeness."
        ),
        "score": float(chosen["score"]),
    }


def main() -> None:
    args = parse_args()
    cfg_path = resolve_relative_path(args.config, root=REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    gate_cfg = dict(cfg.get("real_field_input_transform_gate") or {})
    variants = gate_cfg.get("variants") or ["raw", "robust_rms_matched", "p95_abs_matched", "p99_abs_matched"]
    output_root = resolve_relative_path(cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    output_dir = _timestamped_output("real_field_input_transform_gate", args.output_dir, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_dir = output_dir / "variant_configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)
    r0_root = output_dir / "r0"
    r1_root = output_dir / "r1"
    r0_root.mkdir()
    r1_root.mkdir()
    summary_rows = []
    prediction_rows = []
    for variant in variants:
        variant = str(variant)
        variant_cfg = config_dir / f"{variant}.yaml"
        _write_variant_config(cfg, variant=variant, path=variant_cfg)
        r0_dir = r0_root / variant
        r1_dir = r1_root / variant
        r0_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "real_field_zero_shot.py"),
            "--config",
            str(variant_cfg),
            "--output-dir",
            str(r0_dir),
        ]
        if args.device:
            r0_cmd.extend(["--device", str(args.device)])
        _run_command(r0_cmd, log_path=logs_dir / f"{variant}_r0.log")
        r1_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "real_field_forward_diagnostic.py"),
            "--config",
            str(variant_cfg),
            "--zero-shot-dir",
            str(r0_dir),
            "--output-dir",
            str(r1_dir),
        ]
        _run_command(r1_cmd, log_path=logs_dir / f"{variant}_r1.log")
        summary_rows.append(_summarize_variant(variant=variant, r0_dir=r0_dir, r1_dir=r1_dir))
        prediction_rows.extend(_prediction_stats(r0_dir, variant=variant))
    summary = pd.DataFrame.from_records(summary_rows)
    summary_path = output_dir / "input_transform_gate_summary.csv"
    summary.to_csv(summary_path, index=False)
    prediction_path = output_dir / "input_transform_prediction_summary.csv"
    pd.DataFrame.from_records(prediction_rows).to_csv(prediction_path, index=False)
    recommendation = _recommend(summary)
    write_json(
        output_dir / "real_field_input_transform_gate_summary.json",
        {
            "schema_version": "real_field_input_transform_gate_v1",
            "status": "ok",
            "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
            "variants": list(map(str, variants)),
            "outputs": {
                "input_transform_gate_summary": repo_relative_path(summary_path, root=REPO_ROOT),
                "input_transform_prediction_summary": repo_relative_path(prediction_path, root=REPO_ROOT),
                "variant_configs": repo_relative_path(config_dir, root=REPO_ROOT),
                "r0_root": repo_relative_path(r0_root, root=REPO_ROOT),
                "r1_root": repo_relative_path(r1_root, root=REPO_ROOT),
                "logs": repo_relative_path(logs_dir, root=REPO_ROOT),
            },
            "recommendation": recommendation,
            "code_version_or_git_commit": _git_commit(),
        },
    )
    print("=== Real Field Input Transform Gate ===")
    print(f"Output: {output_dir}")
    print(f"Variants: {len(variants)}")
    print(f"Recommended: {recommendation.get('recommended_transform', '')}")


if __name__ == "__main__":
    main()
