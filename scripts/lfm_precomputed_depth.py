"""Build depth-domain low-frequency models from shifted LAS curves.

Reads depth-shifted LAS files, constructs a target layer from interpreted
horizons, then builds Vp and AI low-frequency models via kriging and
low-pass filtering. Outputs NPZ and SEG-Y volumes.

Usage::

    python scripts/lfm_precomputed_depth.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import (  # noqa: E402
    build_segy_textual_header,
    load_yaml_config,
    resolve_relative_path,
)

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120


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
        help="Output directory. Defaults to scripts/output/lfm_precomputed_depth_<timestamp>.",
    )
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


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
# LFM model building
# =============================================================================


def build_target_layer(
    horizon_files: dict[str, Path],
    geometry: dict[str, Any],
    qc_output_dir: Path,
    train_config: dict[str, Any],
) -> Any:
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.target_layer import TargetLayer

    raw_horizons = {name: import_interpretation_petrel(hf) for name, hf in horizon_files.items()}
    return TargetLayer(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=list(horizon_files.keys()),
        qc_output_dir=qc_output_dir,
        min_thickness=train_config.get("target_layer_min_thickness"),
        nearest_distance_limit=train_config.get("target_layer_nearest_distance_limit"),
        outlier_threshold=train_config.get("target_layer_outlier_threshold"),
        outlier_min_neighbor_count=train_config.get("target_layer_outlier_min_neighbor_count"),  # type: ignore
    )


def collect_well_records(
    las_files: list[Path],
    well_heads_df: pd.DataFrame,
    survey: Any,
    target_layer: Any,
    vp_mnemonic: str,
    rho_mnemonic: str,
    vp_unit: str,
    rho_unit: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    from cup.petrel.load import load_vp_rho_logset_from_las

    head_lookup = well_heads_df.copy()
    head_lookup["_name_norm"] = head_lookup["Name"].astype(str).str.strip()
    head_lookup = head_lookup.set_index("_name_norm", drop=False)

    records = []
    summary_rows = []
    for las_file in las_files:
        well_name = las_file.stem
        if well_name not in head_lookup.index:
            raise KeyError(f"Well '{well_name}' not found in well heads file.")

        head = head_lookup.loc[well_name]
        well_x = float(head["Surface X"])
        well_y = float(head["Surface Y"])
        kb_m = float(head["Well datum value"])
        inline, xline = survey.coord_to_line(well_x, well_y)
        horizon_depths = target_layer.get_interpretation_values_at_location(inline, xline)

        logset_md = load_vp_rho_logset_from_las(
            las_file,
            vp_mnemonic=vp_mnemonic,
            rho_mnemonic=rho_mnemonic,
            vp_unit=vp_unit,
            rho_unit=rho_unit,
        )
        md_m = logset_md.basis.astype(float)
        tvdss_m = md_m - kb_m
        vp_values = np.asarray(logset_md.Vp.values, dtype=float)
        rho_values = np.asarray(logset_md.Rho.values, dtype=float)
        ai_values = np.asarray(logset_md.AI.values, dtype=float)

        records.append(
            {
                "well_name": well_name,
                "las_file": las_file,
                "logset_md": logset_md,
                "x": well_x,
                "y": well_y,
                "inline": float(inline),
                "xline": float(xline),
                "kb": kb_m,
                "horizon_depths": {name: float(value) for name, value in horizon_depths.items()},
            }
        )

        summary_rows.append(
            {
                "well_name": well_name,
                "inline": float(inline),
                "xline": float(xline),
                "kb_m": kb_m,
                "md_min_m": float(np.nanmin(md_m)),
                "md_max_m": float(np.nanmax(md_m)),
                "tvdss_min_m": float(np.nanmin(tvdss_m)),
                "tvdss_max_m": float(np.nanmax(tvdss_m)),
                "vp_min_mps": float(np.nanmin(vp_values)),
                "vp_max_mps": float(np.nanmax(vp_values)),
                "rho_min_gcc": float(np.nanmin(rho_values)),
                "rho_max_gcc": float(np.nanmax(rho_values)),
                "ai_min": float(np.nanmin(ai_values)),
                "ai_max": float(np.nanmax(ai_values)),
                **{f"horizon_{name}_m": float(value) for name, value in horizon_depths.items()},
            }
        )

    return records, pd.DataFrame.from_records(summary_rows)


def build_tvdss_log_from_md(log_md: Any, *, kb_m: float, dz_m: float) -> Any:
    from wtie.processing.grid import Log

    md_m = np.asarray(log_md.basis, dtype=float)
    values = np.asarray(log_md.values, dtype=float)
    tvdss_m = md_m - float(kb_m)

    finite = np.isfinite(tvdss_m) & np.isfinite(values)
    if int(finite.sum()) < 2:
        raise ValueError(f"Log {log_md.name!r} has too few finite samples.")
    tvdss_m = tvdss_m[finite]
    values = values[finite]
    order = np.argsort(tvdss_m)
    tvdss_m = tvdss_m[order]
    values = values[order]
    unique_depth, unique_depth_idx = np.unique(tvdss_m, return_index=True)
    unique_values = values[unique_depth_idx]
    if unique_depth.size < 2:
        raise ValueError(f"Log {log_md.name!r} has too few unique TVDSS samples.")
    regular_depth = np.arange(float(unique_depth[0]), float(unique_depth[-1]) + 0.5 * dz_m, float(dz_m))
    regular_values = np.interp(regular_depth, unique_depth, unique_values)
    return Log(
        regular_values.astype(np.float64),
        regular_depth.astype(np.float64),
        "tvdss",
        name=log_md.name,
        unit=getattr(log_md, "unit", None),
        allow_nan=False,
    )


def build_lfm_wells(
    well_records: list[dict[str, Any]],
    property_name: str,
    geometry_depth: dict[str, Any],
) -> list[Any]:
    from cup.seismic.lfm_depth import LfmDepthWell

    if property_name not in {"Vp", "AI"}:
        raise ValueError(f"Unsupported property_name={property_name!r}")

    dz_m = float(geometry_depth["sample_step"])
    wells = []
    for record in well_records:
        logset_md = record["logset_md"]
        property_log_md = logset_md.Vp if property_name == "Vp" else logset_md.AI
        property_log = build_tvdss_log_from_md(property_log_md, kb_m=float(record["kb"]), dz_m=dz_m)
        wells.append(
            LfmDepthWell(
                well_name=record["well_name"],
                property_name=property_name,
                property_log=property_log,
                x=record["x"],
                y=record["y"],
                kb=record["kb"],
                metadata={
                    "las_file": str(record["las_file"]),
                    "kb_m": float(record["kb"]),
                    "depth_conversion_assumption": "wavelet_batch_shifted_MD_minus_KB",
                    "input_basis_type_before_tvdss": property_log_md.basis_type,
                    "input_basis_type": property_log.basis_type,
                    "input_unit": getattr(property_log, "unit", ""),
                },
            )
        )
    return wells


# =============================================================================
# Output
# =============================================================================


def save_lfm_result(
    result: Any,
    npz_file: Path,
    segy_file: Path,
    *,
    title: str,
    header_lines: list[str],
    seismic_file: Path,
    segy_options: dict[str, int],
) -> None:
    import cigsegy

    npz_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_file,
        volume=result.volume.astype(np.float32),
        variance_volume=result.variance_volume.astype(np.float32),
        ilines=result.ilines,
        xlines=result.xlines,
        samples=result.samples,
        geometry_json=json.dumps(result.geometry, ensure_ascii=False),
        metadata_json=json.dumps(result.metadata, ensure_ascii=False),
        coverage_stats_json=json.dumps(result.coverage_stats, ensure_ascii=False),
    )

    textual = build_segy_textual_header(title, header_lines)
    volume_export = np.ascontiguousarray(result.volume.astype(np.float32))
    keylocs = [
        segy_options["iline"],
        segy_options["xline"],
        segy_options["istep"],
        segy_options["xstep"],
    ]
    cigsegy.create_by_sharing_header(
        str(segy_file),
        str(seismic_file),
        volume_export,
        keylocs=keylocs,
        textual=textual,
    )
    print(f"Saved NPZ: {npz_file}")
    print(f"Saved SEG-Y: {segy_file}")


# =============================================================================
# QC plotting
# =============================================================================


def plot_lfm_result(result: Any, title: str, output_path: Path, cmap: str = "viridis") -> None:
    ilines = result.ilines
    xlines = result.xlines
    samples = result.samples
    volume = result.volume

    i_il = len(ilines) // 2
    i_xl = len(xlines) // 2
    i_z = len(samples) // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(
        volume[i_il, :, :].T,
        aspect="auto",
        origin="upper",
        extent=[xlines[0], xlines[-1], samples[-1], samples[0]],
        cmap=cmap,
    )
    axes[0].set_title(f"{title} inline @ {ilines[i_il]:.0f}")
    axes[0].set_xlabel("Xline")
    axes[0].set_ylabel("TVDSS depth (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(
        volume[:, i_xl, :].T,
        aspect="auto",
        origin="upper",
        extent=[ilines[0], ilines[-1], samples[-1], samples[0]],
        cmap=cmap,
    )
    axes[1].set_title(f"{title} xline @ {xlines[i_xl]:.0f}")
    axes[1].set_xlabel("Inline")
    axes[1].set_ylabel("TVDSS depth (m)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(
        volume[:, :, i_z].T,
        aspect="auto",
        origin="lower",
        extent=[ilines[0], ilines[-1], xlines[0], xlines[-1]],
        cmap=cmap,
    )
    axes[2].set_title(f"{title} depth slice @ {samples[i_z]:.1f} m")
    axes[2].set_xlabel("Inline")
    axes[2].set_ylabel("Xline")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    _save_fig(output_path)


def summarize_coverage(result: Any) -> pd.DataFrame:
    rows = []
    for zone_name, zone_stats in result.coverage_stats["zones"].items():
        counts = np.asarray(zone_stats["slice_control_counts"], dtype=int)
        rows.append(
            {
                "property": result.metadata["property_name"],
                "zone": zone_name,
                "n_slices": int(counts.size),
                "min_controls": int(np.min(counts)),
                "max_controls": int(np.max(counts)),
                "mean_controls": float(np.mean(counts)),
                "modes": ";".join(sorted(set(zone_stats["slice_modes"]))),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_target_layer_mask_qc(qc_dir: Path, output_path: Path) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    trace_qc_file = qc_dir / "target_layer_trace_qc.csv"
    summary_qc_file = qc_dir / "target_layer_qc_summary.csv"
    if not trace_qc_file.exists() or not summary_qc_file.exists():
        print("Target layer QC files not found, skipping mask QC plot.")
        return

    trace_qc = pd.read_csv(trace_qc_file, low_memory=False)
    summary_qc = pd.read_csv(summary_qc_file, low_memory=False)
    required = {"inline", "xline", "masked_trace", "no_support", "crossing", "thin"}
    missing = required - set(trace_qc.columns)
    if missing:
        print(f"Target layer QC missing required columns: {sorted(missing)}, skipping mask plot.")
        return

    ilines_qc = np.asarray(sorted(trace_qc["inline"].unique()), dtype=float)
    xlines_qc = np.asarray(sorted(trace_qc["xline"].unique()), dtype=float)

    def trace_grid(column: str) -> np.ndarray:
        grid_df = trace_qc.pivot(index="inline", columns="xline", values=column)
        grid_df = grid_df.reindex(index=ilines_qc, columns=xlines_qc)
        return grid_df.to_numpy(dtype=bool)

    masked = trace_grid("masked_trace")
    no_support = trace_grid("no_support")
    crossing = trace_grid("crossing")
    thin = trace_grid("thin")

    mask_reason = np.zeros(masked.shape, dtype=np.uint8)
    mask_reason[masked] = 4
    mask_reason[no_support] = 1
    mask_reason[crossing] = 2
    mask_reason[thin] = 3

    reason_labels = {0: "valid control", 1: "no support", 2: "crossing", 3: "thin", 4: "other masked"}
    reason_colors = ["#f7f7f2", "#d95f02", "#1b9e77", "#7570b3", "#444444"]
    cmap = ListedColormap(reason_colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

    counts = pd.Series(mask_reason.ravel()).value_counts().reindex(range(5), fill_value=0)
    qc_stats = pd.DataFrame(
        {
            "reason": [reason_labels[i] for i in range(5)],
            "trace_count": [int(counts.loc[i]) for i in range(5)],
            "ratio": [float(counts.loc[i] / mask_reason.size) for i in range(5)],
        }
    )
    print("\nTarget layer mask QC stats:")
    print(qc_stats.to_string(index=False))
    print()
    print(summary_qc.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.5, 7.2), constrained_layout=True)
    im = ax.imshow(
        mask_reason,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[xlines_qc[0], xlines_qc[-1], ilines_qc[0], ilines_qc[-1]],  # type: ignore
        cmap=cmap,
        norm=norm,
    )
    ax.set_title("TargetLayer trace mask QC")
    ax.set_xlabel("Xline")
    ax.set_ylabel("Inline")
    handles = [
        Patch(facecolor=reason_colors[i], edgecolor="0.35", label=f"{reason_labels[i]} ({counts.loc[i]:,})")
        for i in range(5)
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)
    ax.grid(False)
    _save_fig(output_path)
    print(f"masked_trace count: {int(masked.sum())}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    data_root_str = cfg.get("data_root", "data")

    script_cfg = cfg.get("lfm_precomputed_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'lfm_precomputed_depth' section in config.")

    # ── Resolve inputs ──

    data_root = REPO_ROOT / data_root_str
    seismic_file = resolve_relative_path(str(cfg["seismic_depth"]["file"]), root=data_root)
    well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)
    train_config_file = resolve_relative_path(str(script_cfg["train_config"]), root=REPO_ROOT)
    las_dir = resolve_relative_path(str(script_cfg["las_dir"]), root=REPO_ROOT)

    horizon_files = {
        name: resolve_relative_path(str(cfg["horizons"][name]), root=data_root) for name in cfg["horizons"]
    }

    for p in [seismic_file, well_heads_file, train_config_file, las_dir, *horizon_files.values()]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    with train_config_file.open("r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    las_vp_unit = str(script_cfg.get("las_vp_unit", "m/s"))
    las_rho_unit = str(script_cfg.get("las_rho_unit", "g/cm3"))

    # ── Output dir ──

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / cfg.get("output_root", "scripts/output")
        output_dir = output_root / f"lfm_precomputed_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    qc_dir = output_dir / "target_layer_qc"
    figures_dir = output_dir / "figures"
    for d in [output_dir, qc_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    vp_npz = output_dir / "vp_lfm_depth.npz"
    vp_segy = output_dir / "vp_lfm_depth.segy"
    ai_npz = output_dir / "ai_lfm_depth.npz"
    ai_segy = output_dir / "ai_lfm_depth.segy"

    print("=== LFM Precomputed Depth ===")
    print(f"LAS dir: {las_dir}")
    print(f"Seismic: {seismic_file}")
    print(f"Train config: {train_config_file}")
    print(f"Output dir: {output_dir}")

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }

    # ── Open survey and build target layer ──

    from cup.petrel.load import import_well_heads_petrel
    from cup.seismic.lfm_depth import build_lfm_depth_model
    from cup.seismic.survey import open_survey

    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)
    geometry_depth = survey.query_geometry(domain="depth")
    print(json.dumps(geometry_depth, indent=2, ensure_ascii=False))
    assert geometry_depth["sample_domain"] == "depth"
    assert geometry_depth["sample_unit"] == "m"

    target_layer = build_target_layer(
        horizon_files=horizon_files,
        geometry=geometry_depth,
        qc_output_dir=qc_dir,
        train_config=train_config,
    )
    print(f"horizons: {target_layer.horizon_names}")
    print(f"zones: {target_layer.iter_zones()}")
    print(target_layer.qc_summary_df.to_string(index=False))

    # ── Load well records ──

    well_heads_df = import_well_heads_petrel(well_heads_file)
    las_files = sorted(las_dir.glob("*.las"))
    if not las_files:
        raise ValueError(f"No LAS files found in {las_dir}")

    print(f"\nLAS count: {len(las_files)}")

    well_records, well_summary = collect_well_records(
        las_files=las_files,
        well_heads_df=well_heads_df,
        survey=survey,
        target_layer=target_layer,
        vp_mnemonic="VP_MPS",
        rho_mnemonic="RHO_GCC",
        vp_unit=las_vp_unit,
        rho_unit=las_rho_unit,
    )
    print(well_summary.to_string(index=False))

    # ── Build Vp and AI LFM models ──

    vp_wells = build_lfm_wells(well_records, "Vp", geometry_depth)
    ai_wells = build_lfm_wells(well_records, "AI", geometry_depth)
    print(f"\nVp wells: {[w.well_name for w in vp_wells]}")
    print(f"AI wells: {[w.well_name for w in ai_wells]}")

    build_params = train_config.get("lfm_precomputed")
    if not build_params:
        raise ValueError("Missing 'lfm_precomputed' section in train config.")

    vp_result = build_lfm_depth_model(
        target_layer=target_layer,
        wells=vp_wells,
        survey=survey,
        **build_params,
    )
    print(f"\n[Vp] volume shape: {vp_result.volume.shape}")
    print(f"[Vp] variance shape: {vp_result.variance_volume.shape}")
    print(f"[Vp] sample range: {float(vp_result.samples[0]):.1f} to {float(vp_result.samples[-1]):.1f}")
    print(f"[Vp] property: {vp_result.metadata['property_name']}")
    print(f"[Vp] zones: {vp_result.metadata['zone_names']}")
    print(f"[Vp] wells: {vp_result.metadata['well_names']}")

    ai_result = build_lfm_depth_model(
        target_layer=target_layer,
        wells=ai_wells,
        survey=survey,
        **build_params,
    )
    print(f"\n[AI] volume shape: {ai_result.volume.shape}")
    print(f"[AI] variance shape: {ai_result.variance_volume.shape}")
    print(f"[AI] sample range: {float(ai_result.samples[0]):.1f} to {float(ai_result.samples[-1]):.1f}")
    print(f"[AI] property: {ai_result.metadata['property_name']}")
    print(f"[AI] zones: {ai_result.metadata['zone_names']}")
    print(f"[AI] wells: {ai_result.metadata['well_names']}")

    # ── Coverage stats ──

    coverage_df = pd.concat([summarize_coverage(vp_result), summarize_coverage(ai_result)], ignore_index=True)
    print(f"\nCoverage stats:\n{coverage_df.to_string(index=False)}")

    # ── QC plots ──

    plot_lfm_result(vp_result, "Vp LFM", figures_dir / "qc_vp_lfm.png")
    plot_lfm_result(ai_result, "AI LFM", figures_dir / "qc_ai_lfm.png")
    plot_target_layer_mask_qc(qc_dir, figures_dir / "qc_target_layer_mask.png")

    # ── Export ──

    build_params_desc = ", ".join(f"{k}={v}" for k, v in build_params.items())

    save_lfm_result(
        vp_result,
        vp_npz,
        vp_segy,
        title="Depth-domain Vp low-frequency model from shifted LAS",
        header_lines=[
            f"artifact={vp_npz.name}",
            f"n_wells={len(vp_wells)}",
            "source=wavelet_batch_shifted_las",
            f"shifted_las_dir={las_dir.name}",
            build_params_desc,
        ],
        seismic_file=seismic_file,
        segy_options=segy_options,
    )

    save_lfm_result(
        ai_result,
        ai_npz,
        ai_segy,
        title="Depth-domain AI low-frequency model from shifted LAS",
        header_lines=[
            f"artifact={ai_npz.name}",
            f"n_wells={len(ai_wells)}",
            "source=wavelet_batch_shifted_las",
            "AI=Vp_mps*Rho_gcc",
            f"shifted_las_dir={las_dir.name}",
            build_params_desc,
        ],
        seismic_file=seismic_file,
        segy_options=segy_options,
    )

    # ── Manifest ──

    print("\n=== Outputs ===")
    for p in [
        vp_npz,
        vp_segy,
        ai_npz,
        ai_segy,
        figures_dir / "qc_vp_lfm.png",
        figures_dir / "qc_ai_lfm.png",
        figures_dir / "qc_target_layer_mask.png",
    ]:
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
