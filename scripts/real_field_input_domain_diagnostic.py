"""Run R0.5 real-field input-domain diagnostics without training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.synthetic.dataset import SynthoseisBenchmark
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sha256_file, write_json
from ginn_v2.data import _aligned_arrays, _finite_values, _row_slice
from ginn_v2.real_field import (
    RealFieldSection,
    forward_log_ai,
    input_qc_frame,
    load_real_field_section,
    load_selected_wavelet,
    run_zero_shot_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-r0", action="store_true", help="Only write distribution and LFM diagnostics.")
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


def _stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "robust_rms": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "abs_p95": float("nan"),
            "abs_p99": float("nan"),
        }
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "rms": float(np.sqrt(np.mean(data * data))),
        "robust_rms": float(1.4826 * mad),
        "p01": float(np.quantile(data, 0.01)),
        "p05": float(np.quantile(data, 0.05)),
        "p50": median,
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "abs_p95": float(np.quantile(np.abs(data), 0.95)),
        "abs_p99": float(np.quantile(np.abs(data), 0.99)),
    }


def _normalized_outliers(values: np.ndarray, *, mean: float, std: float) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0 or std <= 0.0 or not np.isfinite(std):
        return {
            "normalized_mean": float("nan"),
            "normalized_std": float("nan"),
            "fraction_abs_gt_3sigma": float("nan"),
            "fraction_abs_gt_5sigma": float("nan"),
        }
    normalized = (data - float(mean)) / float(std)
    return {
        "normalized_mean": float(np.mean(normalized)),
        "normalized_std": float(np.std(normalized)),
        "fraction_abs_gt_3sigma": float(np.mean(np.abs(normalized) > 3.0)),
        "fraction_abs_gt_5sigma": float(np.mean(np.abs(normalized) > 5.0)),
    }


def _distribution_row(
    *,
    source: str,
    values: np.ndarray,
    normalization: dict,
    input_name: str = "seismic",
) -> dict[str, object]:
    stats = _stats(values)
    norm_stats = normalization[input_name]
    return {
        "source": source,
        "input": input_name,
        **stats,
        "synthetic_train_norm_mean": float(norm_stats["mean"]),
        "synthetic_train_norm_std": float(norm_stats["std"]),
        **_normalized_outliers(values, mean=float(norm_stats["mean"]), std=float(norm_stats["std"])),
    }


def _load_model_manifest(model_cfg: dict) -> dict:
    run_dir = resolve_relative_path(str(model_cfg["model_run_dir"]), root=REPO_ROOT)
    with (run_dir / "model_run_manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _synthetic_train_values(
    manifest: dict,
    *,
    input_name: str = "seismic",
    max_patches: int = 4096,
    seed: int = 20260620,
) -> tuple[np.ndarray, dict[str, object]]:
    benchmark_dir = resolve_relative_path(manifest["benchmark_dir"], root=REPO_ROOT)
    patch_index_path = resolve_relative_path(manifest["patch_index"], root=REPO_ROOT)
    benchmark = SynthoseisBenchmark(benchmark_dir)
    patch_index = pd.read_csv(patch_index_path)
    train = patch_index[patch_index["split"].astype(str).eq("train")]
    total_train = int(train.shape[0])
    if max_patches > 0 and total_train > max_patches:
        train = train.sample(n=int(max_patches), random_state=int(seed)).sort_index()
    chunks: list[np.ndarray] = []
    for _, row in train.iterrows():
        sample = benchmark.load_sample(str(row["sample_id"]))
        target, seismic, lfm, valid, _ = _aligned_arrays(sample)
        sl = _row_slice(row)
        patch_valid = valid[sl]
        if input_name == "seismic":
            chunks.append(_finite_values(seismic[sl], patch_valid))
        elif input_name == "lfm":
            chunks.append(_finite_values(lfm[sl], patch_valid))
        elif input_name == "target":
            chunks.append(_finite_values(target[sl], patch_valid))
        else:
            raise ValueError(f"Unsupported synthetic input: {input_name}")
    valid_chunks = [chunk for chunk in chunks if chunk.size]
    if not valid_chunks:
        raise ValueError(f"No finite synthetic train values for {input_name}.")
    metadata = {
        "input_name": input_name,
        "total_train_patches": total_train,
        "sampled_train_patches": int(train.shape[0]),
        "sampling_seed": int(seed),
        "sampling_policy": "deterministic_random_sample_without_replacement" if train.shape[0] < total_train else "all_train_patches",
    }
    return np.concatenate(valid_chunks), metadata


def _valid_values(section: RealFieldSection, values: np.ndarray) -> np.ndarray:
    valid = section.valid_mask & np.isfinite(values)
    return np.asarray(values, dtype=np.float64)[valid]


def _seismic_variants(
    section: RealFieldSection,
    *,
    synthetic_seismic: np.ndarray,
    cfg: dict,
) -> dict[str, np.ndarray]:
    real = np.asarray(section.seismic, dtype=np.float64)
    valid_real = _valid_values(section, real)
    synth_stats = _stats(synthetic_seismic)
    real_stats = _stats(valid_real)
    median_real = real_stats["p50"]
    variants = {"raw": real}
    robust_scale = synth_stats["robust_rms"] / real_stats["robust_rms"]
    p95_scale = synth_stats["abs_p95"] / real_stats["abs_p95"]
    p99_scale = synth_stats["abs_p99"] / real_stats["abs_p99"]
    variants["robust_rms_matched"] = (real - median_real) * robust_scale + synth_stats["mean"]
    variants["p95_abs_matched"] = (real - median_real) * p95_scale + synth_stats["mean"]
    variants["p99_abs_matched"] = (real - median_real) * p99_scale + synth_stats["mean"]
    if bool(cfg.get("include_polarity_flip", True)):
        variants["robust_rms_matched_polarity_flip"] = -1.0 * (real - median_real) * robust_scale + synth_stats["mean"]
    return {key: np.asarray(value, dtype=np.float32) for key, value in variants.items()}


def _prediction_stats(output_dir: Path, *, variant: str) -> list[dict[str, object]]:
    rows = []
    for child in sorted(output_dir.iterdir()):
        path = child / "predictions.npz"
        if not path.is_file():
            continue
        arrays = np.load(path, allow_pickle=True)
        valid = arrays["valid_mask_model"].astype(bool) & (arrays["stitching_weight"] > 0.0)
        for signal in ("stitched_pred_log_ai", "pred_delta_vs_lfm"):
            values = np.asarray(arrays[signal], dtype=np.float64)
            data = values[valid & np.isfinite(values)]
            rows.append(
                {
                    "variant": variant,
                    "model_role": child.name,
                    "signal": signal,
                    **_stats(data),
                    "finite_fraction": float(np.mean(np.isfinite(values) & valid)),
                    "max_abs": float(np.max(np.abs(data))) if data.size else float("nan"),
                }
            )
    return rows


def _run_variant_r0(
    *,
    variant: str,
    seismic: np.ndarray,
    base_section: RealFieldSection,
    run_cfg: dict,
    output_dir: Path,
    device: str,
    stitch_strategy: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    variant_dir = output_dir / "r0_variants" / variant
    variant_dir.mkdir(parents=True, exist_ok=False)
    section = replace(base_section, seismic=np.asarray(seismic, dtype=np.float32))
    models = run_cfg.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("real_field_zero_shot.models must be configured.")
    summaries = []
    for model_cfg in models:
        summaries.append(
            run_zero_shot_model(
                section=section,
                model_cfg=model_cfg,
                output_dir=variant_dir,
                root=REPO_ROOT,
                device_name=device,
                stitch_strategy=stitch_strategy,
            )
        )
    first_norm = summaries[0]["normalization"]
    input_qc = input_qc_frame(section, first_norm)
    input_qc.insert(0, "variant", variant)
    input_qc.to_csv(variant_dir / "model_input_qc.csv", index=False)
    write_json(
        variant_dir / "r0_variant_summary.json",
        {
            "schema_version": "real_field_input_domain_r0_variant_v1",
            "variant": variant,
            "status": "ok",
            "models": summaries,
            "input_qc": repo_relative_path(variant_dir / "model_input_qc.csv", root=REPO_ROOT),
        },
    )
    return input_qc.to_dict("records"), _prediction_stats(variant_dir, variant=variant)


def _lfm_zero_energy_diagnostic(section: RealFieldSection, *, wavelet: np.ndarray, crop_s: float) -> pd.DataFrame:
    lfm = np.asarray(section.lfm, dtype=np.float64)
    valid = np.asarray(section.valid_mask, dtype=bool)
    twt = np.asarray(section.twt_s, dtype=np.float64)
    dt_s = float(np.median(np.diff(twt))) if twt.size > 1 else 0.002
    reflectivity = np.tanh(0.5 * (lfm[:, 1:] - lfm[:, :-1]))
    synthetic = forward_log_ai(lfm, wavelet)
    valid_forward = valid[:, 1:] & np.isfinite(reflectivity) & np.isfinite(synthetic)
    crop_samples = int(np.ceil(float(crop_s) / dt_s)) if crop_s > 0.0 else 0
    if crop_samples > 0 and valid_forward.shape[1] > 2 * crop_samples:
        crop = np.zeros_like(valid_forward, dtype=bool)
        crop[:, crop_samples : valid_forward.shape[1] - crop_samples] = True
        valid_forward = valid_forward & crop
    rows = []
    for label, values, mask in [
        ("lfm_log_ai", lfm, valid),
        ("lfm_diff_log_ai", lfm[:, 1:] - lfm[:, :-1], valid[:, 1:]),
        ("lfm_reflectivity", reflectivity, valid[:, 1:]),
        ("lfm_synthetic_nominal_wavelet", synthetic, valid_forward),
    ]:
        data = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool) & np.isfinite(values)]
        row = {"signal": label, "n_valid": int(data.size), **_stats(data)}
        rows.append(row)
    explanation = "lfm_is_nearly_constant_on_time_axis" if rows[-1]["rms"] <= 0.0 else "lfm_forward_has_nonzero_energy"
    for row in rows:
        row["dt_s"] = dt_s
        row["crop_s"] = float(crop_s)
        row["crop_samples"] = crop_samples
        row["explanation"] = explanation
    return pd.DataFrame.from_records(rows)


def _plot_distribution(output_dir: Path, frame: pd.DataFrame) -> str:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    seismic = frame[frame["input"].astype(str).eq("seismic")].copy()
    path = figures / "real_vs_synthetic_input_distribution.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(seismic["source"], seismic["robust_rms"])
    axes[0].set_title("Robust RMS")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(seismic["source"], seismic["fraction_abs_gt_5sigma"])
    axes[1].set_title("|normalized| > 5")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return repo_relative_path(path, root=REPO_ROOT)


def main() -> None:
    args = parse_args()
    cfg_path = resolve_relative_path(args.config, root=REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_cfg = dict(cfg.get("real_field_zero_shot") or {})
    diag_cfg = dict(cfg.get("real_field_input_domain_diagnostic") or {})
    if not run_cfg:
        raise ValueError("experiments/common.yaml lacks real_field_zero_shot section.")
    output_root = resolve_relative_path(cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    output_dir = _timestamped_output("real_field_input_domain_diagnostic", args.output_dir, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)
    section = load_real_field_section(config=run_cfg, root=REPO_ROOT, data_root=data_root)
    models = run_cfg.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("real_field_zero_shot.models must be configured.")
    reference_manifest = _load_model_manifest(dict(models[0]))
    normalization = reference_manifest["normalization"]
    max_synth = int(diag_cfg.get("max_synthetic_train_patches", 4096))
    synth_seed = int(diag_cfg.get("synthetic_train_sampling_seed", 20260620))
    synthetic_seismic, synthetic_seismic_meta = _synthetic_train_values(
        reference_manifest,
        input_name="seismic",
        max_patches=max_synth,
        seed=synth_seed,
    )
    synthetic_lfm, synthetic_lfm_meta = _synthetic_train_values(
        reference_manifest,
        input_name="lfm",
        max_patches=max_synth,
        seed=synth_seed,
    )

    variants = _seismic_variants(section, synthetic_seismic=synthetic_seismic, cfg=diag_cfg)
    distribution_rows = [
        _distribution_row(source="synthetic_train", values=synthetic_seismic, normalization=normalization, input_name="seismic"),
        _distribution_row(source="real_raw", values=_valid_values(section, section.seismic), normalization=normalization, input_name="seismic"),
        _distribution_row(source="synthetic_train", values=synthetic_lfm, normalization=normalization, input_name="lfm"),
        _distribution_row(source="real_lfm", values=_valid_values(section, section.lfm), normalization=normalization, input_name="lfm"),
    ]
    for name, seismic in variants.items():
        if name == "raw":
            continue
        distribution_rows.append(
            _distribution_row(
                source=f"real_{name}",
                values=_valid_values(section, seismic),
                normalization=normalization,
                input_name="seismic",
            )
        )
    distribution = pd.DataFrame.from_records(distribution_rows)
    distribution_path = output_dir / "real_vs_synthetic_input_distribution.csv"
    distribution.to_csv(distribution_path, index=False)
    distribution_figure = _plot_distribution(output_dir, distribution)

    wavelet_dir = resolve_relative_path(str(run_cfg["source_runs"]["wavelet_generation_dir"]), root=REPO_ROOT)
    wavelet, wavelet_meta = load_selected_wavelet(wavelet_dir)
    boundary = dict(run_cfg.get("boundary") or {})
    crop_s = float(boundary.get("forward_diagnostic_crop_s", 0.0) or 0.0)
    lfm_diag = _lfm_zero_energy_diagnostic(section, wavelet=wavelet, crop_s=crop_s)
    lfm_diag_path = output_dir / "lfm_only_zero_energy_diagnostic.csv"
    lfm_diag.to_csv(lfm_diag_path, index=False)

    input_qc_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    run_r0 = bool(diag_cfg.get("run_diagnostic_r0", True)) and not args.skip_r0
    if run_r0:
        selected = diag_cfg.get("variants") or list(variants.keys())
        device = str(args.device or run_cfg.get("device") or "auto")
        stitch_strategy = str(run_cfg.get("stitch_strategy") or "uniform")
        for name in selected:
            if name not in variants:
                raise ValueError(f"Unknown input-domain variant: {name}")
            qc_records, pred_records = _run_variant_r0(
                variant=name,
                seismic=variants[name],
                base_section=section,
                run_cfg=run_cfg,
                output_dir=output_dir,
                device=device,
                stitch_strategy=stitch_strategy,
            )
            input_qc_rows.extend(qc_records)
            prediction_rows.extend(pred_records)
    input_qc_path = output_dir / "diagnostic_r0_input_qc.csv"
    pd.DataFrame.from_records(input_qc_rows).to_csv(input_qc_path, index=False)
    prediction_path = output_dir / "diagnostic_r0_prediction_summary.csv"
    pd.DataFrame.from_records(prediction_rows).to_csv(prediction_path, index=False)

    raw_gt5 = float(distribution.loc[distribution["source"].eq("real_raw") & distribution["input"].eq("seismic"), "fraction_abs_gt_5sigma"].iloc[0])
    best = distribution[distribution["input"].eq("seismic")].sort_values("fraction_abs_gt_5sigma").iloc[0].to_dict()
    summary = {
        "schema_version": "real_field_input_domain_diagnostic_v1",
        "status": "ok",
        "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
        "source_real_field_zero_shot_config": "real_field_zero_shot",
        "reference_model_run_dir": str(models[0]["model_run_dir"]),
        "reference_normalization": normalization,
        "synthetic_train_sampling": {
            "seismic": synthetic_seismic_meta,
            "lfm": synthetic_lfm_meta,
        },
        "outputs": {
            "real_vs_synthetic_input_distribution": repo_relative_path(distribution_path, root=REPO_ROOT),
            "distribution_figure": distribution_figure,
            "lfm_only_zero_energy_diagnostic": repo_relative_path(lfm_diag_path, root=REPO_ROOT),
            "diagnostic_r0_input_qc": repo_relative_path(input_qc_path, root=REPO_ROOT),
            "diagnostic_r0_prediction_summary": repo_relative_path(prediction_path, root=REPO_ROOT),
            "r0_variants": repo_relative_path(output_dir / "r0_variants", root=REPO_ROOT) if (output_dir / "r0_variants").is_dir() else "",
        },
        "decision": {
            "raw_real_seismic_fraction_abs_gt_5sigma": raw_gt5,
            "best_distribution_source_by_gt5": best.get("source", ""),
            "best_distribution_fraction_abs_gt_5sigma": best.get("fraction_abs_gt_5sigma", float("nan")),
            "lfm_only_explanation": str(lfm_diag["explanation"].iloc[0]) if not lfm_diag.empty else "",
            "allow_r2_calibration_only": False,
            "reason": "R0.5 is diagnostic-only; do not train, adapter-tune, or enter R2 from this run.",
        },
        "wavelet": wavelet_meta,
        "code_version_or_git_commit": _git_commit(),
    }
    write_json(output_dir / "real_field_input_domain_diagnostic_summary.json", summary)
    print("=== Real Field Input-Domain Diagnostic ===")
    print(f"Output: {output_dir}")
    print(f"R0 variants: {len(input_qc_rows)} input rows, {len(prediction_rows)} prediction rows")
    print("Status: ok")


if __name__ == "__main__":
    main()
