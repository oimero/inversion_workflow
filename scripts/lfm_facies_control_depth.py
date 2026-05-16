"""Apply reef-core facies controls to a depth-domain AI low-frequency model.

This post-processing script reads an ``ai_lfm_depth.npz`` produced by
``lfm_precomputed_depth.py`` and writes a facies-controlled AI LFM NPZ/SEG-Y.
It only modifies AI; Vp LFM and GINN anchor constraints are intentionally left
to separate workflow steps.

Usage::

    python scripts/lfm_facies_control_depth.py
    python scripts/lfm_facies_control_depth.py --config experiments/common_depth.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.facies_control_depth import (
    SCHEMA_VERSION,
    apply_depth_facies_controls,
    build_target_layer_from_lfm_metadata,
    build_trace_xy_grids,
    load_depth_facies_control_points_csv,
)
from cup.seismic.survey import open_survey
from cup.utils.io import (
    build_segy_textual_header,
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    to_json_compatible,
    write_json,
)
from ginn_depth.data import load_lfm_depth_npz


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
        help="Output directory. Defaults to <output_root>/lfm_facies_control_depth_<timestamp>.",
    )
    parser.add_argument(
        "--skip-segy",
        action="store_true",
        help="Do not export SEG-Y even when export_segy is true in config.",
    )
    return parser.parse_args()


def _json_from_npz(path: Path, key: str) -> dict[str, Any] | None:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        value = str(np.asarray(data[key]).item())
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else None


def _save_controlled_lfm_npz(
    path: Path,
    *,
    volume: np.ndarray,
    variance_volume: np.ndarray | None,
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    geometry: dict[str, Any],
    metadata: dict[str, Any],
    coverage_stats: dict[str, Any] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "volume": np.asarray(volume, dtype=np.float32),
        "ilines": np.asarray(ilines),
        "xlines": np.asarray(xlines),
        "samples": np.asarray(samples),
        "geometry_json": np.asarray(json.dumps(to_json_compatible(geometry), ensure_ascii=False)),
        "metadata_json": np.asarray(json.dumps(to_json_compatible(metadata), ensure_ascii=False)),
    }
    if variance_volume is not None:
        payload["variance_volume"] = np.asarray(variance_volume, dtype=np.float32)
    if coverage_stats is not None:
        payload["coverage_stats_json"] = np.asarray(json.dumps(to_json_compatible(coverage_stats), ensure_ascii=False))
    np.savez_compressed(path, **payload)


def _export_controlled_segy(
    path: Path,
    *,
    source_seismic_file: Path,
    volume: np.ndarray,
    segy_options: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    import cigsegy

    textual = build_segy_textual_header(
        "Depth-domain AI LFM with facies control",
        [
            f"artifact={path.name}",
            "source=lfm_facies_control_depth.py",
            f"schema={SCHEMA_VERSION}",
            f"n_controls={metadata['facies_control']['n_controls']}",
        ],
    )
    keylocs = [
        segy_options["iline"],
        segy_options["xline"],
        segy_options["istep"],
        segy_options["xstep"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    cigsegy.create_by_sharing_header(
        str(path),
        str(source_seismic_file),
        np.ascontiguousarray(volume.astype(np.float32)),
        keylocs=keylocs,
        textual=textual,
    )


def _stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _plot_qc(
    path: Path,
    *,
    source: np.ndarray,
    controlled: np.ndarray,
    diff: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    qc_df: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if qc_df.empty:
        return
    row = qc_df.sort_values("affected_samples", ascending=False).iloc[0]
    i_il = int(np.argmin(np.abs(ilines.astype(float) - float(row["inline"]))))
    i_xl = int(np.argmin(np.abs(xlines.astype(float) - float(row["xline"]))))
    max_abs_map = np.max(np.abs(diff), axis=2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    extent_inline = [xlines[0], xlines[-1], samples[-1], samples[0]]
    vmin = float(np.nanpercentile(source[i_il], 1.0))
    vmax = float(np.nanpercentile(source[i_il], 99.0))

    im0 = axes[0, 0].imshow(
        source[i_il].T, aspect="auto", origin="upper", extent=extent_inline, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0, 0].set_title(f"Before | inline={ilines[i_il]:.0f}")
    axes[0, 0].set_xlabel("Xline")
    axes[0, 0].set_ylabel("Depth (m)")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.85)

    im1 = axes[0, 1].imshow(
        controlled[i_il].T, aspect="auto", origin="upper", extent=extent_inline, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0, 1].set_title("After")
    axes[0, 1].set_xlabel("Xline")
    axes[0, 1].set_ylabel("Depth (m)")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.85)

    diff_clip = float(np.nanpercentile(np.abs(diff[i_il]), 99.0))
    diff_clip = max(diff_clip, 1.0)
    im2 = axes[1, 0].imshow(
        diff[i_il].T,
        aspect="auto",
        origin="upper",
        extent=extent_inline,
        cmap="coolwarm",
        vmin=-diff_clip,
        vmax=diff_clip,
    )
    axes[1, 0].set_title("After - Before")
    axes[1, 0].set_xlabel("Xline")
    axes[1, 0].set_ylabel("Depth (m)")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.85)

    im3 = axes[1, 1].imshow(
        max_abs_map.T,
        aspect="auto",
        origin="lower",
        extent=[ilines[0], ilines[-1], xlines[0], xlines[-1]],
        cmap="magma",
    )
    axes[1, 1].scatter([float(row["inline"])], [float(row["xline"])], c="cyan", s=40, edgecolors="black")
    axes[1, 1].set_title("Max |AI delta| map")
    axes[1, 1].set_xlabel("Inline")
    axes[1, 1].set_ylabel("Xline")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.85)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("lfm_facies_control_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'lfm_facies_control_depth' section in config.")

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    source_ai_lfm_file = resolve_relative_path(str(script_cfg["source_ai_lfm_file"]), root=REPO_ROOT)
    control_points_file = resolve_relative_path(str(script_cfg["control_points_file"]), root=REPO_ROOT)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)

    for input_path in (source_ai_lfm_file, control_points_file, seismic_file):
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input: {input_path}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"lfm_facies_control_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    qc_dir = output_dir / "target_layer_qc"
    figures_dir = output_dir / "figures"
    for directory in (output_dir, qc_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_npz = output_dir / "ai_lfm_depth_facies_controlled.npz"
    output_segy = output_dir / "ai_lfm_depth_facies_controlled.segy"
    qc_csv = output_dir / "facies_control_qc.csv"
    qc_png = figures_dir / "qc_ai_lfm_facies_control.png"
    run_summary_path = output_dir / "run_summary.json"

    if str(script_cfg.get("influence_kernel", "raised_cosine")) != "raised_cosine":
        raise ValueError("Only influence_kernel='raised_cosine' is supported.")
    if str(script_cfg.get("blend_domain", "log_ai")) != "log_ai":
        raise ValueError("Only blend_domain='log_ai' is supported.")

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": int(segy_cfg["iline_byte"]),
        "xline": int(segy_cfg["xline_byte"]),
        "istep": int(segy_cfg["istep"]),
        "xstep": int(segy_cfg["xstep"]),
    }

    print("=== LFM Facies Control Depth ===")
    print(f"Source AI LFM: {source_ai_lfm_file}")
    print(f"Control points: {control_points_file}")
    print(f"Seismic: {seismic_file}")
    print(f"Output dir: {output_dir}")

    ai_lfm = load_lfm_depth_npz(source_ai_lfm_file)
    coverage_stats = _json_from_npz(source_ai_lfm_file, "coverage_stats_json")
    control_points = load_depth_facies_control_points_csv(control_points_file)
    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)
    target_layer = build_target_layer_from_lfm_metadata(ai_lfm.metadata, ai_lfm.geometry, qc_output_dir=qc_dir)
    x_grid, y_grid = build_trace_xy_grids(survey, ai_lfm.ilines, ai_lfm.xlines)

    controlled, qc_df = apply_depth_facies_controls(
        ai_lfm.volume,
        ilines=ai_lfm.ilines,
        xlines=ai_lfm.xlines,
        samples=ai_lfm.samples,
        target_layer=target_layer,
        survey=survey,
        control_points=control_points,
        x_grid=x_grid,
        y_grid=y_grid,
    )
    diff = controlled.astype(np.float64) - np.asarray(ai_lfm.volume, dtype=np.float64)

    qc_df_for_csv = qc_df.copy()
    if "horizon_values" in qc_df_for_csv.columns:
        qc_df_for_csv["horizon_values"] = qc_df_for_csv["horizon_values"].map(
            lambda value: json.dumps(to_json_compatible(value), ensure_ascii=False)
        )
    qc_df_for_csv.to_csv(qc_csv, index=False, encoding="utf-8-sig")

    metadata = dict(ai_lfm.metadata)
    metadata["facies_control"] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": Path(__file__).name,
        "path_style": "repo_relative",
        "source_ai_lfm_file": repo_relative_path(source_ai_lfm_file, root=REPO_ROOT),
        "control_points_file": repo_relative_path(control_points_file, root=REPO_ROOT),
        "n_controls": int(len(control_points)),
        "influence_kernel": "raised_cosine",
        "blend_domain": "log_ai",
        "modifies": "AI",
        "does_not_modify": ["Vp", "log_ai_anchor"],
        "qc_csv": repo_relative_path(qc_csv, root=REPO_ROOT),
        "qc_figure": repo_relative_path(qc_png, root=REPO_ROOT),
        "controls": qc_df.to_dict("records"),
        "source_ai_stats": _stats(ai_lfm.volume),
        "controlled_ai_stats": _stats(controlled),
        "delta_ai_stats": _stats(diff),
        "changed_samples": int(np.count_nonzero(np.abs(diff) > 0.0)),
    }

    _save_controlled_lfm_npz(
        output_npz,
        volume=controlled,
        variance_volume=ai_lfm.variance_volume,
        ilines=ai_lfm.ilines,
        xlines=ai_lfm.xlines,
        samples=ai_lfm.samples,
        geometry=ai_lfm.geometry,
        metadata=metadata,
        coverage_stats=coverage_stats,
    )

    export_segy = bool(script_cfg.get("export_segy", True)) and not args.skip_segy
    if export_segy:
        _export_controlled_segy(
            output_segy,
            source_seismic_file=seismic_file,
            volume=controlled,
            segy_options=segy_options,
            metadata=metadata,
        )

    _plot_qc(
        qc_png,
        source=np.asarray(ai_lfm.volume),
        controlled=controlled,
        diff=diff,
        ilines=ai_lfm.ilines,
        xlines=ai_lfm.xlines,
        samples=ai_lfm.samples,
        qc_df=qc_df,
    )

    run_summary = {
        "created_at_utc": metadata["facies_control"]["created_at_utc"],
        "source_script": Path(__file__).name,
        "path_style": "repo_relative",
        "source_ai_lfm_file": repo_relative_path(source_ai_lfm_file, root=REPO_ROOT),
        "control_points_file": repo_relative_path(control_points_file, root=REPO_ROOT),
        "output_npz": repo_relative_path(output_npz, root=REPO_ROOT),
        "output_segy": repo_relative_path(output_segy, root=REPO_ROOT) if export_segy else None,
        "qc_csv": repo_relative_path(qc_csv, root=REPO_ROOT),
        "qc_figure": repo_relative_path(qc_png, root=REPO_ROOT),
        "schema_version": SCHEMA_VERSION,
        "n_controls": int(len(control_points)),
        "changed_samples": metadata["facies_control"]["changed_samples"],
        "delta_ai_stats": metadata["facies_control"]["delta_ai_stats"],
        "controls": qc_df.to_dict("records"),
    }
    write_json(run_summary_path, run_summary)

    print(qc_df_for_csv.to_string(index=False))
    print(f"Saved controlled AI LFM NPZ: {output_npz}")
    if export_segy:
        print(f"Saved controlled AI LFM SEG-Y: {output_segy}")
    print(f"Saved QC CSV: {qc_csv}")
    print(f"Saved QC figure: {qc_png}")
    print(f"Saved run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
