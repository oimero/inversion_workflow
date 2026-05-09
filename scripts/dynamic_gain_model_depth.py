"""Build a dynamic gain volume from fitted attribute-gain relationship.

Reads the fit parameters produced by ``dynamic_gain_attr_fitting_depth``,
loads the full depth-domain seismic volume, and computes a sample-by-sample
dynamic gain via ``ln(gain) = intercept + slope * ln(moving_rms)``.

Outputs an NPZ + SEG-Y gain volume for use as ``dynamic_gain_model_file`` in
GINN training.

Usage::

    python scripts/dynamic_gain_model_depth.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ── Bootstrap ──
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

from cup.utils.io import build_segy_textual_header, load_yaml_config  # noqa: E402
from cup.utils.raw_trace import (  # noqa: E402
    centered_moving_rms_axis,
    centered_moving_sum_axis,
    meters_to_odd_samples,
    zscore_traces_axis,
)

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common_depth.yaml"),
        help="Depth-domain common config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/dynamic_gain_model_depth_<timestamp>.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    repo_root = _find_repo_root()

    cfg = load_yaml_config(args.config, base_dir=repo_root)
    script_cfg = cfg.get("dynamic_gain_model_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'dynamic_gain_model_depth' section in config.")

    # ── Inputs ──

    source_fit_dir = repo_root / str(script_cfg["source_fit_dir"])
    fit_file = source_fit_dir / "gain_attr_fit_metrics.csv"
    if not fit_file.exists():
        raise FileNotFoundError(f"Fit metrics not found: {fit_file}")

    seg_file = source_fit_dir / "gain_attr_segment_samples.csv"
    seismic_file = repo_root / str(cfg["data_root"]) / str(cfg["seismic_depth"]["file"])
    if not seismic_file.exists():
        raise FileNotFoundError(f"Seismic not found: {seismic_file}")

    volume_batch_traces = int(script_cfg.get("volume_batch_traces", 2048))

    # ── Output dir ──

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = repo_root / str(cfg.get("output_root", "scripts/output"))
        output_dir = out_root / f"dynamic_gain_model_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir

    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    gain_npz = output_dir / "dynamic_gain_depth.npz"
    gain_segy = output_dir / "dynamic_gain_depth.segy"

    # ── Read fit parameters ──

    fit_df = pd.read_csv(fit_file)
    fit_row = fit_df.iloc[0]
    intercept = float(fit_row["intercept"])
    slope = float(fit_row["slope"])
    attribute_name = str(fit_row["attribute_name"])
    app_window_m = float(fit_row["application_window_m"])
    gain_clip_p05 = float(fit_row["gain_clip_p05"])
    gain_clip_p95 = float(fit_row["gain_clip_p95"])
    attr_floor = float(fit_row["attribute_floor"])
    r2 = float(fit_row["r2"])

    print("=== Dynamic Gain Model (Depth) ===")
    print(f"Fit source: {fit_file}")
    print(f"Seismic: {seismic_file}")
    print(f"Model: ln(gain) = {intercept:.3f} + {slope:.3f} * ln({attribute_name})")
    print(f"  R2={r2:.3f}, gain clip=[{gain_clip_p05:.3f}, {gain_clip_p95:.3f}]")
    print(f"  application window: {app_window_m:.1f} m")

    # ── Survey geometry ──

    from cup.seismic.survey import open_survey

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }
    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)
    geometry = survey.query_geometry(domain="depth")
    print(json.dumps(geometry, indent=2, ensure_ascii=False))
    assert geometry["sample_domain"] == "depth"
    assert geometry["sample_unit"] == "m"

    sample_step_m = float(geometry["sample_step"])
    window_samples = meters_to_odd_samples(app_window_m, sample_step_m)
    print(f"RMS window: {window_samples} samples = {window_samples * sample_step_m:.1f} m")

    # ── Load seismic volume ──

    from cup.petrel.load import import_seismic

    print("Loading seismic volume...")
    seismic_volume = import_seismic(
        seismic_file,
        seismic_type="segy",
        iline=segy_options["iline"],
        xline=segy_options["xline"],
        istep=segy_options["istep"],
        xstep=segy_options["xstep"],
    )
    expected_shape = (
        int(geometry["n_il"]),
        int(geometry["n_xl"]),
        int(geometry["n_sample"]),
    )
    if tuple(seismic_volume.shape) != expected_shape:
        raise ValueError(f"Seismic volume shape {seismic_volume.shape} != geometry {expected_shape}")

    # ── Compute gain volume ──

    flat = seismic_volume.reshape(-1, seismic_volume.shape[-1])
    gain_flat = np.empty_like(flat, dtype=np.float32)
    for start in range(0, flat.shape[0], volume_batch_traces):
        end = min(start + volume_batch_traces, flat.shape[0])
        seismic_zscore = zscore_traces_axis(flat[start:end])
        seismic_rms = centered_moving_rms_axis(seismic_zscore, window_samples)
        rms_safe = np.maximum(seismic_rms, float(attr_floor))
        log_seismic_rms = np.where(
            np.isfinite(rms_safe) & (rms_safe > 0.0),
            np.log(rms_safe),
            np.nan,
        )
        log_gain = float(intercept) + float(slope) * log_seismic_rms
        log_gain = np.clip(log_gain, np.log(gain_clip_p05), np.log(gain_clip_p95))
        gain_flat[start:end] = np.exp(log_gain).astype(np.float32)
        if start == 0 or end == flat.shape[0] or (start // volume_batch_traces) % 20 == 0:
            print(f"  Processed traces {end}/{flat.shape[0]}")

    gain_volume = gain_flat.reshape(seismic_volume.shape)
    print(f"Gain volume shape: {gain_volume.shape}")
    print(
        f"Gain min/median/max: {float(np.nanmin(gain_volume)):.3f} / "
        f"{float(np.nanmedian(gain_volume)):.3f} / {float(np.nanmax(gain_volume)):.3f}"
    )

    # ── QC figures ──

    # Figure 01: fit scatter (re-plot from segments for QC)
    if seg_file.exists():
        seg_df = pd.read_csv(seg_file)
        fit_col = f"log_{attribute_name}"
        if fit_col in seg_df.columns and "log_gain" in seg_df.columns:
            fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
            for w, wdf in seg_df.groupby("well_name"):
                ax.scatter(wdf[fit_col], wdf["log_gain"], s=24, alpha=0.75, label=w)
            valid = np.isfinite(seg_df[fit_col]) & np.isfinite(seg_df["log_gain"])
            x_min, x_max = np.nanpercentile(seg_df.loc[valid, fit_col], [1, 99])
            x_line = np.linspace(float(x_min), float(x_max), 100)
            ax.plot(x_line, intercept + slope * x_line, color="black", lw=1.3, label="OLS fit")
            ax.set_xlabel(f"ln({attribute_name})")
            ax.set_ylabel("ln(gain)")
            ax.set_title("Dynamic gain training segments")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7)
            _save_fig(figure_dir / "qc_01_dynamic_gain_fit.png")
        else:
            print("Segment samples missing fit columns, skipping qc_01.")
    else:
        print("Segment samples file not found, skipping qc_01.")

    # Figure 02: gain volume sections
    mid_il = gain_volume.shape[0] // 2
    mid_xl = gain_volume.shape[1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    axes[0].hist(
        gain_volume[np.isfinite(gain_volume)].ravel(),
        bins=80,
        color="tab:blue",
        alpha=0.85,
    )
    axes[0].set_xlabel("Gain")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dynamic gain distribution")
    axes[0].grid(True, alpha=0.25)

    im1 = axes[1].imshow(
        gain_volume[mid_il].T,
        aspect="auto",
        origin="upper",
        cmap="viridis",
    )
    axes[1].set_title(f"Inline index {mid_il}")
    axes[1].set_xlabel("Xline index")
    axes[1].set_ylabel("Depth sample")
    fig.colorbar(im1, ax=axes[1], shrink=0.82)

    im2 = axes[2].imshow(
        gain_volume[:, mid_xl, :].T,
        aspect="auto",
        origin="upper",
        cmap="viridis",
    )
    axes[2].set_title(f"Xline index {mid_xl}")
    axes[2].set_xlabel("Inline index")
    axes[2].set_ylabel("Depth sample")
    fig.colorbar(im2, ax=axes[2], shrink=0.82)
    _save_fig(figure_dir / "qc_02_dynamic_gain_volume.png")

    # ── Export NPZ ──

    ilines = np.arange(
        float(geometry["inline_min"]),
        float(geometry["inline_max"]) + 0.5 * float(geometry["inline_step"]),
        float(geometry["inline_step"]),
    ).astype(np.float32)
    xlines = np.arange(
        float(geometry["xline_min"]),
        float(geometry["xline_max"]) + 0.5 * float(geometry["xline_step"]),
        float(geometry["xline_step"]),
    ).astype(np.float32)
    samples = np.arange(
        float(geometry["sample_min"]),
        float(geometry["sample_max"]) + 0.5 * float(geometry["sample_step"]),
        float(geometry["sample_step"]),
    ).astype(np.float32)

    metadata = {
        "artifact": gain_npz.name,
        "gain_model_type": "dynamic_rms_log_linear_depth",
        "gain_model_is_relative_to_fixed_gain": False,
        "intended_usage": "Set fixed_gain=None and dynamic_gain_model_file to this NPZ/SEG-Y model.",
        "selected_attribute": attribute_name,
        "intercept": intercept,
        "slope": slope,
        "log_gain_clip": [np.log(gain_clip_p05), np.log(gain_clip_p95)],
        "gain_clip": [gain_clip_p05, gain_clip_p95],
        "attribute_floor": attr_floor,
        "application_window_m": app_window_m,
        "application_window_samples": int(window_samples),
        "sample_step_m": sample_step_m,
        "source_seismic_file": str(seismic_file),
        "fit_metrics_file": str(fit_file),
    }
    np.savez_compressed(
        gain_npz,
        volume=gain_volume.astype(np.float32),
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        geometry_json=json.dumps(geometry, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"Saved NPZ: {gain_npz}")

    # ── Export SEG-Y ──

    import cigsegy

    keylocs = [
        segy_options["iline"],
        segy_options["xline"],
        segy_options["istep"],
        segy_options["xstep"],
    ]
    textual = build_segy_textual_header(
        "Depth-domain dynamic gain model from RMS attribute",
        [
            f"artifact={gain_npz.name}",
            f"model=ln(gain)={intercept:.6g}+{slope:.6g}*ln({attribute_name})",
            f"rms_window_m={app_window_m:.3f} samples={window_samples}",
            f"gain_clip={gain_clip_p05:.6g}-{gain_clip_p95:.6g}",
            "usage=fixed_gain None; dynamic_gain_model_file this artifact",
        ],
    )
    cigsegy.create_by_sharing_header(
        str(gain_segy),
        str(seismic_file),
        np.ascontiguousarray(gain_volume.astype(np.float32)),
        keylocs=keylocs,
        textual=textual,
    )
    print(f"Saved SEG-Y: {gain_segy}")

    # ── Manifest ──

    print(f"\n=== Outputs ===")
    for p in [gain_npz, gain_segy]:
        print(f"  {p}")
    print(f"  {figure_dir}")


if __name__ == "__main__":
    main()
