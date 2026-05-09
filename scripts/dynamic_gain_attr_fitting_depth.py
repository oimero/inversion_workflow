"""Fit seismic-attribute to amplitude-gain relationship in depth domain.

Reads batch synthetic QC outputs, segments each well's seismic trace into
local depth windows, compares three seismic attributes (RMS, abs-mean, abs-p90)
vs segment gain, selects the best attribute with tie-breaking toward RMS,
fits ln(gain) = intercept + slope * ln(attribute), then predicts depth-varying
gain curves.

Usage::

    python scripts/dynamic_gain_attr_fitting_depth.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ── Bootstrap: add src/ to sys.path before importing cup ──
def _find_repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "src").exists():
        root = Path.cwd().resolve()
    if not (root / "src").exists():
        raise RuntimeError("Could not locate repo root containing 'src'.")
    return root

def _ensure_import_path(src_root: Path) -> None:
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

_ensure_import_path(_find_repo_root() / "src")

from cup.utils.io import load_yaml_config, sanitize_filename, save_mpl_figure  # noqa: E402
from cup.utils.raw_trace import centered_moving_rms, meters_to_odd_samples  # noqa: E402
from cup.utils.statistics import normalized_mae, ols_fit, pearson_r, spearman_rho  # noqa: E402

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_columns", 120)

CANDIDATE_ATTRIBUTES = ["seismic_rms", "seismic_abs_mean", "seismic_abs_p90"]


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/common_depth.yaml"),
        help="Depth-domain common config YAML.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory. Defaults to <output_root>/dynamic_gain_attr_fitting_depth_<timestamp>.",
    )
    return parser.parse_args()


def _infer_depth_step(depth_values: np.ndarray) -> float:
    """Return median sample step from a depth axis."""
    v = depth_values[np.isfinite(depth_values)]
    if v.size < 3:
        raise ValueError("Need at least three finite depth samples.")
    step = float(np.nanmedian(np.abs(np.diff(v))))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"Invalid inferred depth step: {step}")
    return step


def candidate_metrics(
    seismic_norm: np.ndarray, synthetic: np.ndarray, mask: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(seismic_norm) & np.isfinite(synthetic)
    return {
        "corr": pearson_r(seismic_norm, synthetic, mask=valid),
        "nmae": normalized_mae(seismic_norm, synthetic, mask=valid),
        "n_eval_samples": int(valid.sum()),
    }


# =============================================================================
# Segmentation & gain
# =============================================================================


def split_valid_indices(
    valid_indices: np.ndarray, *, min_valid_samples: int, max_segments: int, min_segments: int,
) -> list[np.ndarray]:
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if valid_indices.size < int(min_valid_samples):
        return []
    segment_count = int(min(max_segments, valid_indices.size // int(min_valid_samples)))
    segment_count = max(int(min_segments), segment_count)
    return [s for s in np.array_split(valid_indices, segment_count) if s.size >= min_valid_samples]


def positive_ls_gain(
    seismic_values: np.ndarray, synthetic_raw_values: np.ndarray,
    *, eps: float, min_valid_samples: int,
) -> float:
    seismic_values = np.asarray(seismic_values, dtype=float)
    synthetic_raw_values = np.asarray(synthetic_raw_values, dtype=float)
    valid = np.isfinite(seismic_values) & np.isfinite(synthetic_raw_values)
    if int(valid.sum()) < int(min_valid_samples):
        return np.nan
    numerator = float(np.sum(seismic_values[valid] * synthetic_raw_values[valid]))
    denominator = float(np.sum(synthetic_raw_values[valid] ** 2))
    gain = numerator / (denominator + float(eps) * max(int(valid.sum()), 1))
    return float(gain) if np.isfinite(gain) and gain > 0.0 else np.nan


def segment_attribute_values(seismic_values: np.ndarray) -> dict[str, float]:
    seismic_values = np.asarray(seismic_values, dtype=float)
    finite = np.isfinite(seismic_values)
    if not np.any(finite):
        return {a: np.nan for a in CANDIDATE_ATTRIBUTES}
    values = seismic_values[finite]
    abs_values = np.abs(values)
    return {
        "seismic_rms": float(np.sqrt(np.nanmean(values ** 2))),
        "seismic_abs_mean": float(np.nanmean(abs_values)),
        "seismic_abs_p90": float(np.nanpercentile(abs_values, 90.0)),
    }


def set_log_column(df: pd.DataFrame, source_column: str, log_column: str) -> None:
    values = df[source_column].to_numpy(dtype=float)
    df[log_column] = np.nan
    positive = values > 0.0
    df.loc[positive, log_column] = np.log(values[positive])


def eval_mask_to_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def predict_gain_from_trace_rms(
    seismic_norm: np.ndarray, *, window_samples: int, intercept: float, slope: float,
    attribute_floor: float, log_gain_clip: tuple[float, float],
) -> pd.DataFrame:
    seismic_norm = np.asarray(seismic_norm, dtype=float)
    seismic_rms = centered_moving_rms(seismic_norm, window_samples)
    rms_safe = np.maximum(seismic_rms, float(attribute_floor))
    log_seismic_rms = np.where(np.isfinite(rms_safe) & (rms_safe > 0.0), np.log(rms_safe), np.nan)
    log_gain_pred = float(intercept) + float(slope) * log_seismic_rms
    log_gain_pred_clipped = np.clip(log_gain_pred, float(log_gain_clip[0]), float(log_gain_clip[1]))
    return pd.DataFrame({
        "seismic_rms": seismic_rms, "log_seismic_rms": log_seismic_rms,
        "log_gain_pred": log_gain_pred, "log_gain_pred_clipped": log_gain_pred_clipped,
        "gain_pred": np.exp(log_gain_pred), "gain_pred_clipped": np.exp(log_gain_pred_clipped),
    })


# =============================================================================
# Attribute comparison
# =============================================================================


def compare_attributes(
    segment_df: pd.DataFrame, attr_tie_threshold: float,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Compare log(attribute) vs log(gain) correlations and select best attribute.

    Tie-breaking: if the best attribute's |pearson| advantage over RMS is less
    than *attr_tie_threshold*, prefer RMS as the simpler default.
    """
    metric_rows = []
    for attr in CANDIDATE_ATTRIBUTES:
        x_col = f"log_{attr}"
        x = segment_df[x_col].to_numpy(dtype=float)
        y = segment_df["log_gain"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        metric_rows.append({
            "scope": "all_wells", "well_name": "__all__", "attribute": attr,
            "n_samples": int(mask.sum()),
            "pearson": pearson_r(x, y), "spearman": spearman_rho(x, y),
        })
        for well_name, well_df in segment_df.groupby("well_name"):
            xw = well_df[x_col].to_numpy(dtype=float)
            yw = well_df["log_gain"].to_numpy(dtype=float)
            mask_w = np.isfinite(xw) & np.isfinite(yw)
            metric_rows.append({
                "scope": "per_well", "well_name": well_name, "attribute": attr,
                "n_samples": int(mask_w.sum()),
                "pearson": pearson_r(xw, yw), "spearman": spearman_rho(xw, yw),
            })

    metrics_df = pd.DataFrame(metric_rows)

    # Pick best all-wells attribute
    all_df = metrics_df.loc[metrics_df["scope"] == "all_wells"].copy()
    all_df["abs_pearson"] = all_df["pearson"].abs()
    all_df["abs_spearman"] = all_df["spearman"].abs()
    all_df = all_df.sort_values(["abs_pearson", "abs_spearman"], ascending=False)
    best_row = all_df.iloc[0]
    best_attr = str(best_row["attribute"])

    # Tie-breaking: if best is not RMS, check how much better it is
    if best_attr != "seismic_rms":
        rms_row = all_df.loc[all_df["attribute"] == "seismic_rms"].iloc[0]
        advantage = abs(float(best_row["abs_pearson"])) - abs(float(rms_row["abs_pearson"]))
        if advantage < float(attr_tie_threshold):
            best_attr = "seismic_rms"
            best_row = rms_row

    best_df = pd.DataFrame([{**best_row.to_dict()}])
    return best_attr, metrics_df, best_df


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    repo_root = _find_repo_root()

    cfg = load_yaml_config(args.config, base_dir=repo_root)
    script_cfg = cfg.get("dynamic_gain_attr_fitting_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'dynamic_gain_attr_fitting_depth' section in config.")

    source_batch_dir = repo_root / str(script_cfg["source_batch_dir"])
    batch_metrics_file = source_batch_dir / "wavelet_batch_metrics.csv"
    if not batch_metrics_file.exists():
        raise FileNotFoundError(f"Batch metrics not found: {batch_metrics_file}")

    # ── Config params ──

    min_seg_samples = int(script_cfg["min_segment_valid_samples"])
    max_segments = int(script_cfg["max_segment_count"])
    min_segments = int(script_cfg["min_segments_per_well"])
    gain_eps = float(script_cfg["gain_eps"])
    attr_tie_threshold = float(script_cfg["attr_tie_threshold"])
    min_batch_corr = script_cfg.get("min_batch_corr")
    app_window_m = script_cfg.get("application_rms_window_m")
    pred_clip_pct = tuple(float(v) for v in script_cfg["prediction_clip_percentiles"])
    attr_floor_frac = float(script_cfg["attribute_floor_fraction"])

    # ── Output dir ──

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = repo_root / str(cfg.get("output_root", "scripts/output"))
        output_dir = out_root / f"dynamic_gain_attr_fitting_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    well_qc_dir = output_dir / "well_qc"
    figure_dir = output_dir / "figures"
    for d in [output_dir, well_qc_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)

    seg_file = output_dir / "gain_attr_segment_samples.csv"
    rel_file = output_dir / "gain_attr_relationship_metrics.csv"
    best_file = output_dir / "gain_attr_best_relationship.csv"
    fit_file = output_dir / "gain_attr_fit_metrics.csv"

    print("=== Dynamic Gain Attribute Fitting (Depth) ===")
    print(f"Batch metrics: {batch_metrics_file}")
    print(f"Output dir: {output_dir}")

    # ── Load batch metrics ──

    batch_df = pd.read_csv(batch_metrics_file)
    for col in ["well_name", "status", "scale", "synthetic_qc_path", "depth_shift_curve_path", "corr", "nmae"]:
        if col not in batch_df.columns:
            raise ValueError(f"Batch metrics missing column: {col}")

    ok_df = batch_df.loc[batch_df["status"] == "ok"].copy()
    if min_batch_corr is not None:
        ok_df = ok_df.loc[ok_df["corr"] >= float(min_batch_corr)].copy()
    if ok_df.empty:
        raise ValueError("No successful wells available.")

    # ── Build segments ──

    segment_rows = []
    for _, row in ok_df.iterrows():
        well_name = str(row["well_name"])
        scale = float(row["scale"])
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        qc_path = Path(row["synthetic_qc_path"])
        ds_path = Path(row["depth_shift_curve_path"])
        if not qc_path.is_absolute():
            qc_path = repo_root / qc_path
        if not ds_path.is_absolute():
            ds_path = repo_root / ds_path
        if not qc_path.exists() or not ds_path.exists():
            continue

        qc_df = pd.read_csv(qc_path)
        ds_df = pd.read_csv(ds_path)
        twt_s = qc_df["twt_s"].to_numpy(dtype=float)
        seismic_norm = qc_df["seismic_norm"].to_numpy(dtype=float)
        synthetic_raw = qc_df["synthetic_scaled"].to_numpy(dtype=float) / scale
        eval_mask = eval_mask_to_bool(qc_df["eval_mask"])

        ds_twt = ds_df["twt_s"].to_numpy(dtype=float)
        ds_z = ds_df["tvdss_m"].to_numpy(dtype=float)
        fd = np.isfinite(ds_twt) & np.isfinite(ds_z)
        if int(fd.sum()) < 2:
            continue
        order = np.argsort(ds_twt[fd])
        ds_twt, ds_z = ds_twt[fd][order], ds_z[fd][order]

        finite = eval_mask & np.isfinite(twt_s) & np.isfinite(seismic_norm) & np.isfinite(synthetic_raw)
        segments = split_valid_indices(
            np.flatnonzero(finite), min_valid_samples=min_seg_samples,
            max_segments=max_segments, min_segments=min_segments,
        )
        for seg_id, seg_idx in enumerate(segments):
            seg_twt = twt_s[seg_idx]
            gain = positive_ls_gain(
                seismic_norm[seg_idx], synthetic_raw[seg_idx],
                eps=gain_eps, min_valid_samples=min_seg_samples,
            )
            if not np.isfinite(gain):
                continue
            attrs = segment_attribute_values(seismic_norm[seg_idx])
            tvdss = np.interp(seg_twt, ds_twt, ds_z)
            segment_rows.append({
                "well_name": well_name, "segment_id": int(seg_id),
                "n_valid_samples": int(seg_idx.size),
                "twt_min_s": float(np.nanmin(seg_twt)), "twt_max_s": float(np.nanmax(seg_twt)),
                "tvdss_min_m": float(np.nanmin(tvdss)), "tvdss_max_m": float(np.nanmax(tvdss)),
                "segment_thickness_m": float(np.nanmax(tvdss) - np.nanmin(tvdss)),
                "gain": gain, "batch_scale": scale, "batch_corr": float(row["corr"]),
                **attrs,
            })

    seg_df = pd.DataFrame(segment_rows)
    if seg_df.empty:
        raise ValueError("No finite training segments were generated.")
    for col in ["gain", *CANDIDATE_ATTRIBUTES]:
        set_log_column(seg_df, col, f"log_{col}")
    seg_df.to_csv(seg_file, index=False)
    print(f"Segments: {len(seg_df)} from {seg_df['well_name'].nunique()} wells")

    # ── Compare attributes (QC only) ──
    # The prediction function predict_gain_from_trace_rms() always uses
    # sample-by-sample moving RMS, so the actual fit must use seismic_rms
    # regardless of which attribute wins the segment-level comparison.

    best_attr, rel_df, best_df = compare_attributes(seg_df, attr_tie_threshold)
    rel_df.to_csv(rel_file, index=False)
    best_df.to_csv(best_file, index=False)
    print(f"\nAttribute comparison (all wells, segment-level QC):")
    all_rel = rel_df.loc[rel_df["scope"] == "all_wells"].set_index("attribute")
    for attr in CANDIDATE_ATTRIBUTES:
        r = all_rel.loc[attr]
        print(f"  {attr}: pearson={r['pearson']:.3f}, spearman={r['spearman']:.3f}, n={int(r['n_samples'])}")
    print(f"  QC winner: {best_attr}  (tie_threshold={attr_tie_threshold})")
    if best_attr != "seismic_rms":
        print(f"  Note: fit uses seismic_rms because prediction uses moving-RMS input;"
              f" segment-level {best_attr} win is informational.")

    # ── Fit always against seismic_rms (matches predict_gain_from_trace_rms) ──

    fit_attr = "seismic_rms"
    fit_col = f"log_{fit_attr}"
    fit_df = seg_df.loc[np.isfinite(seg_df["log_gain"]) & np.isfinite(seg_df[fit_col])].copy()
    if fit_df.empty:
        raise ValueError(f"No finite ln(gain) / ln({fit_attr}) samples.")

    fit = ols_fit(fit_df[fit_col].to_numpy(dtype=float), fit_df["log_gain"].to_numpy(dtype=float))
    log_gain_clip = tuple(
        float(v) for v in np.nanpercentile(fit_df["log_gain"].to_numpy(dtype=float), pred_clip_pct)
    )
    attr_floor = float(np.nanpercentile(fit_df[fit_attr].to_numpy(dtype=float), 1.0) * attr_floor_frac)
    attr_floor = max(attr_floor, np.finfo(float).tiny)

    thick = fit_df["segment_thickness_m"].to_numpy(dtype=float)
    thick = thick[np.isfinite(thick) & (thick > 0.0)]
    app_win = float(np.nanmedian(thick)) if app_window_m is None else float(app_window_m)

    fit_row = {
        **fit, "target": "log_gain", "attribute": fit_col, "attribute_name": fit_attr,
        "n_training_segments": int(fit_df.shape[0]),
        "n_training_wells": int(fit_df["well_name"].nunique()),
        "log_gain_clip_p05": float(log_gain_clip[0]), "log_gain_clip_p95": float(log_gain_clip[1]),
        "gain_clip_p05": float(np.exp(log_gain_clip[0])), "gain_clip_p95": float(np.exp(log_gain_clip[1])),
        "attribute_floor": attr_floor, "application_window_m": app_win,
    }
    pd.DataFrame([fit_row]).to_csv(fit_file, index=False)

    print(f"\nFit: ln(gain) = {fit['intercept']:.3f} + {fit['slope']:.3f} * ln({fit_attr})")
    print(f"  R2={fit['r2']:.3f}, pearson={fit['pearson_r']:.3f}, n={fit['n_samples']}")
    print(f"  gain clip: [{fit_row['gain_clip_p05']:.3f}, {fit_row['gain_clip_p95']:.3f}]")
    print(f"  application window: {app_win:.1f} m")

    # ── Attribute comparison QC figures ──

    all_df = all_rel.reset_index()
    plot_df = all_df.copy()
    plot_df["label"] = plot_df["attribute"].str.replace("seismic_", "", regex=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    axes[0].bar(np.arange(len(plot_df)), plot_df["pearson"], color="tab:blue", alpha=0.75)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(plot_df)))
    axes[0].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title("log(attribute) vs log(gain)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(np.arange(len(plot_df)), plot_df["spearman"], color="tab:green", alpha=0.75)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(np.arange(len(plot_df)))
    axes[1].set_xticklabels(plot_df["label"], rotation=35, ha="right")
    axes[1].set_ylabel("Spearman rho")
    axes[1].set_title("Rank relationship")
    axes[1].grid(True, axis="y", alpha=0.25)
    save_mpl_figure(figure_dir / "qc_01_gain_attribute_relationship_bars.png")

    well_names = seg_df["well_name"].drop_duplicates().tolist()
    color_map = {w: plt.cm.tab10(i % 10) for i, w in enumerate(well_names)}
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    for w, wdf in seg_df.groupby("well_name"):
        ax.scatter(wdf[f"log_{fit_attr}"], wdf["log_gain"], s=28, alpha=0.75,
                   color=color_map[w], label=w)
    ax.set_xlabel(f"log({fit_attr})")
    ax.set_ylabel("log(gain)")
    ax.set_title(f"Fit attribute: {fit_attr} (pearson={all_rel.loc[fit_attr]['pearson']:.3f})")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)
    save_mpl_figure(figure_dir / "qc_02_best_gain_attribute_scatter.png")

    # ── Per-well gain QC ──

    summary_rows = []
    for _, row in ok_df.iterrows():
        well_name = str(row["well_name"])
        name = sanitize_filename(well_name)
        scale = float(row["scale"])
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        qc_path = Path(row["synthetic_qc_path"])
        ds_path = Path(row["depth_shift_curve_path"])
        if not qc_path.is_absolute():
            qc_path = repo_root / qc_path
        if not ds_path.is_absolute():
            ds_path = repo_root / ds_path
        if not qc_path.exists() or not ds_path.exists():
            continue

        qc_df = pd.read_csv(qc_path)
        ds_df = pd.read_csv(ds_path)
        twt_s = qc_df["twt_s"].to_numpy(dtype=float)
        seismic_norm = qc_df["seismic_norm"].to_numpy(dtype=float)
        synthetic_fixed = qc_df["synthetic_scaled"].to_numpy(dtype=float)
        synthetic_raw = synthetic_fixed / scale
        eval_mask = eval_mask_to_bool(qc_df["eval_mask"])

        ds_twt = ds_df["twt_s"].to_numpy(dtype=float)
        ds_z = ds_df["tvdss_m"].to_numpy(dtype=float)
        fd = np.isfinite(ds_twt) & np.isfinite(ds_z)
        if int(fd.sum()) < 2:
            continue
        order = np.argsort(ds_twt[fd])
        ds_twt, ds_z = ds_twt[fd][order], ds_z[fd][order]
        tvdss_m = np.interp(twt_s, ds_twt, ds_z, left=np.nan, right=np.nan)

        dz_m = _infer_depth_step(tvdss_m)
        win_smp = meters_to_odd_samples(app_win, dz_m)
        pred_df = predict_gain_from_trace_rms(
            seismic_norm, window_samples=win_smp, intercept=fit["intercept"],
            slope=fit["slope"], attribute_floor=attr_floor, log_gain_clip=log_gain_clip,
        )
        gain_curve = pred_df["gain_pred_clipped"].to_numpy(dtype=float)
        synthetic_gain = gain_curve * synthetic_raw

        fm = candidate_metrics(seismic_norm, synthetic_fixed, eval_mask)
        gm = candidate_metrics(seismic_norm, synthetic_gain, eval_mask)

        out_df = pd.DataFrame({
            "well_name": well_name, "twt_s": twt_s, "tvdss_m": tvdss_m,
            "seismic_norm": seismic_norm,
            "reflectivity_shifted": qc_df.get("reflectivity_shifted",
                pd.Series(np.full_like(twt_s, np.nan))).to_numpy(dtype=float),
            "synthetic_raw": synthetic_raw,
            "synthetic_fixed_scale": synthetic_fixed,
            "synthetic_gain_pred": synthetic_gain,
            "residual_fixed_scale": seismic_norm - synthetic_fixed,
            "residual_gain_pred": seismic_norm - synthetic_gain,
            "eval_mask": eval_mask, "batch_scale": scale,
            "rms_window_samples": win_smp, "rms_window_m": win_smp * dz_m,
        })
        out_df = pd.concat([out_df, pred_df], axis=1)
        out_path = well_qc_dir / f"wellside_gain_synthetic_qc_{name}.csv"
        out_df.to_csv(out_path, index=False)

        fg = gain_curve[np.isfinite(gain_curve)]
        summary_rows.append({
            "well_name": well_name, "n_samples": int(twt_s.size),
            "n_eval_samples": int(eval_mask.sum()),
            "batch_corr": float(row["corr"]), "batch_nmae": float(row["nmae"]),
            "fixed_corr_recomputed": fm["corr"], "fixed_nmae_recomputed": fm["nmae"],
            "gain_corr": gm["corr"], "gain_nmae": gm["nmae"],
            "batch_scale": scale,
            "gain_median": float(np.nanmedian(fg)) if fg.size else np.nan,
            "gain_p10": float(np.nanpercentile(fg, 10.0)) if fg.size else np.nan,
            "gain_p90": float(np.nanpercentile(fg, 90.0)) if fg.size else np.nan,
            "rms_window_samples": int(win_smp), "inferred_depth_step_m": dz_m,
            "rms_window_m": float(win_smp * dz_m), "well_qc_path": str(out_path),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "wellside_gain_synthetic_qc_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Per-well QC figures ──

    for _, srow in summary_df.iterrows():
        well_name = str(srow["well_name"])
        name = sanitize_filename(well_name)
        qc_df = pd.read_csv(srow["well_qc_path"])
        d = qc_df["tvdss_m"].to_numpy(dtype=float)
        s = qc_df["seismic_norm"].to_numpy(dtype=float)
        fsyn = qc_df["synthetic_fixed_scale"].to_numpy(dtype=float)
        gsyn = qc_df["synthetic_gain_pred"].to_numpy(dtype=float)
        g = qc_df["gain_pred_clipped"].to_numpy(dtype=float)
        r = qc_df["seismic_rms"].to_numpy(dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), sharey=True, constrained_layout=True)
        axes[0].plot(s, d, lw=0.9, color="black", label="Seismic")
        axes[0].plot(fsyn, d, lw=0.9, color="tab:red", alpha=0.85, label="Fixed-scale synthetic")
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Normalized amplitude"); axes[0].set_ylabel("TVDSS (m)")
        axes[0].set_title(f"Fixed: corr={srow['fixed_corr_recomputed']:.3f}, nmae={srow['fixed_nmae_recomputed']:.3f}")
        axes[0].grid(True, alpha=0.25); axes[0].legend(loc="best", fontsize=7)

        axes[1].plot(s, d, lw=0.9, color="black", label="Seismic")
        axes[1].plot(gsyn, d, lw=0.9, color="tab:blue", alpha=0.9, label="RMS-gain synthetic")
        axes[1].set_xlabel("Normalized amplitude")
        axes[1].set_title(f"Gain: corr={srow['gain_corr']:.3f}, nmae={srow['gain_nmae']:.3f}")
        axes[1].grid(True, alpha=0.25); axes[1].legend(loc="best", fontsize=7)

        fg = np.isfinite(g)
        gn = g / np.nanmedian(g[fg]) if np.any(fg) else g
        rn = r / np.nanmedian(r[np.isfinite(r)]) if np.any(np.isfinite(r)) else r
        axes[2].plot(gn, d, lw=1.0, color="tab:blue", label="gain / median")
        axes[2].plot(rn, d, lw=1.0, color="tab:green", label="RMS / median")
        axes[2].set_xlabel("Normalized attribute"); axes[2].set_title("Predicted gain curve")
        axes[2].grid(True, alpha=0.25); axes[2].legend(loc="best", fontsize=7)
        fig.suptitle(well_name)
        save_mpl_figure(figure_dir / f"qc_{name}_wellside_gain_synthetic.png")

    # ── Summary figure ──

    if not summary_df.empty:
        labels = summary_df["well_name"].tolist()
        x = np.arange(len(summary_df))
        w = 0.34
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
        axes[0].bar(x - w / 2, summary_df["fixed_corr_recomputed"], w, color="tab:red", alpha=0.8, label="fixed")
        axes[0].bar(x + w / 2, summary_df["gain_corr"], w, color="tab:blue", alpha=0.8, label="gain")
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("Correlation"); axes[0].set_ylim(-1, 1)
        axes[0].set_title("Synthetic correlation"); axes[0].grid(True, axis="y", alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)

        axes[1].bar(x - w / 2, summary_df["fixed_nmae_recomputed"], w, color="tab:red", alpha=0.8, label="fixed")
        axes[1].bar(x + w / 2, summary_df["gain_nmae"], w, color="tab:blue", alpha=0.8, label="gain")
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("NMAE"); axes[1].set_title("Synthetic NMAE")
        axes[1].grid(True, axis="y", alpha=0.25); axes[1].legend(loc="best", fontsize=8)

        axes[2].plot(x, summary_df["batch_scale"], marker="o", lw=1.1, color="tab:red", label="batch fixed scale")
        axes[2].plot(x, summary_df["gain_median"], marker="o", lw=1.1, color="tab:blue", label="median predicted gain")
        axes[2].fill_between(x, summary_df["gain_p10"].to_numpy(dtype=float),
                             summary_df["gain_p90"].to_numpy(dtype=float),
                             color="tab:blue", alpha=0.15, label="gain P10-P90")
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=45, ha="right")
        axes[2].set_ylabel("Gain"); axes[2].set_title("Fixed scale vs predicted gain")
        axes[2].grid(True, axis="y", alpha=0.25); axes[2].legend(loc="best", fontsize=8)
        save_mpl_figure(figure_dir / "qc_00_wellside_gain_synthetic_summary.png")

    # ── Manifest ──

    print(f"\n=== Outputs ===")
    for p in [seg_file, rel_file, best_file, fit_file, summary_path]:
        print(f"  {p}")
    print(f"  {well_qc_dir}")
    print(f"  {figure_dir}")


if __name__ == "__main__":
    main()
