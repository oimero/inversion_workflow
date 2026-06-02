"""Inventory well assets for the time-domain workflow.

The script is deliberately non-destructive. It scans asset presence, classifies
coarse spatial status, reports platform clusters and non-platform same-trace
conflicts, but does not read LAS curves or parse deviated-well trajectories.

Usage::

    python scripts/well_inventory.py
    python scripts/well_inventory.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_well_heads_petrel, import_well_tops_petrel
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json
from cup.well.assets import (
    WellHead,
    WellInventory,
    WellInventoryRecord,
    build_cluster_rows,
    build_file_lookup,
    build_name_lookup,
    build_neighbor_pairs,
    classify_wellbore,
    determine_inventory_status,
    is_finite_number,
    normalize_well_name,
    value_counts,
)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common.yaml"),
        help="Time-domain common config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/well_inventory_<timestamp>.",
    )
    return parser.parse_args()


# =============================================================================
# Config and paths
# =============================================================================


def _resolve_data_path(value: str | Path, *, data_root: Path) -> Path:
    return resolve_relative_path(value, root=data_root)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"well_inventory_{timestamp}"


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("well_inventory") or {})
    source_data = dict(script_cfg.get("source_data") or {})
    source_data.setdefault("well_heads_file", "raw/well_heads")
    source_data.setdefault("las_dir", "all_well_las")
    source_data.setdefault("well_trace_dir", "all_well_trace")
    source_data.setdefault("well_tops_file", "raw/well_tops")
    source_data.setdefault("time_depth_dir", "time_depth_table")
    script_cfg["source_data"] = source_data
    script_cfg.setdefault(
        "seismic",
        {
            "file": "raw/obn-clipped-240-912-872-1544.zgy",
            "type": "zgy",
        },
    )
    spatial_qc = dict(script_cfg.get("spatial_qc") or {})
    spatial_qc.setdefault("near_survey_threshold_m", 500.0)
    spatial_qc.setdefault("vertical_bottom_offset_threshold_m", 30.0)
    spatial_qc.setdefault("platform_cluster_threshold_m", 12.5)
    spatial_qc.setdefault("dense_well_neighbor_threshold_m", 150.0)
    script_cfg["spatial_qc"] = spatial_qc
    return script_cfg


# =============================================================================
# Inventory
# =============================================================================


def _load_asset_lookups(
    *,
    well_heads_file: Path,
    las_dir: Path,
    well_trace_dir: Path,
    well_tops_file: Path,
    time_depth_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Path], dict[str, Path], dict[str, str]]:
    well_heads_df = import_well_heads_petrel(well_heads_file)
    import_well_heads = build_name_lookup(well_heads_df["Name"].tolist(), asset_label=str(well_heads_file))
    well_heads_df = well_heads_df.copy()
    well_heads_df["_well_key"] = well_heads_df["Name"].map(normalize_well_name)
    if len(import_well_heads) != int(well_heads_df["_well_key"].nunique()):
        raise ValueError("Unexpected duplicate well-head keys after case-insensitive normalization.")

    las_lookup = build_file_lookup(las_dir.glob("*.las"), asset_label=str(las_dir))
    # Petrel trajectory export suffixes are user-defined; match trajectory assets by stem only.
    trace_lookup = build_file_lookup(well_trace_dir.iterdir() if well_trace_dir.exists() else [], asset_label=str(well_trace_dir))
    time_depth_lookup = build_file_lookup(
        time_depth_dir.iterdir() if time_depth_dir.exists() else [],
        asset_label=str(time_depth_dir),
    )

    if well_tops_file.exists():
        well_tops_df = import_well_tops_petrel(well_tops_file)
        tops_lookup = build_name_lookup(well_tops_df["Well"].tolist(), asset_label=str(well_tops_file))
    else:
        tops_lookup = {}

    return well_heads_df, las_lookup, trace_lookup, time_depth_lookup, tops_lookup


def _head_lookup(well_heads_df: pd.DataFrame) -> dict[str, WellHead]:
    lookup: dict[str, WellHead] = {}
    for _, row in well_heads_df.iterrows():
        head = WellHead.from_petrel_row(row)
        lookup[normalize_well_name(head.well_name)] = head
    return lookup


def _record_reasons_for_assets(
    record: WellInventoryRecord,
    *,
    has_well_trace: bool,
    has_time_depth: bool,
    has_well_tops: bool,
) -> None:
    if not has_well_trace:
        record.reasons.append("no_well_trace")
    if not has_time_depth:
        record.reasons.append("no_time_depth")
    if not has_well_tops:
        record.reasons.append("no_well_tops")


def _classify_survey_position(
    record: WellInventoryRecord,
    *,
    survey: Any,
    near_survey_threshold_m: float,
) -> None:
    if record.surface_x is None or record.surface_y is None:
        record.survey_position = "invalid_xy"
        record.reasons.append("invalid_surface_xy")
        return
    if not is_finite_number(record.surface_x) or not is_finite_number(record.surface_y):
        record.survey_position = "invalid_xy"
        record.reasons.append("invalid_surface_xy")
        return

    x = float(record.surface_x)
    y = float(record.surface_y)
    try:
        record.distance_to_survey_m = float(survey.line_geometry.distance_to_footprint_m(x, y))
    except ValueError:
        record.distance_to_survey_m = None

    try:
        inline_float, xline_float = survey.line_geometry.coord_to_line(x, y)
        record.inline_float = float(inline_float)
        record.xline_float = float(xline_float)
        record.nearest_inline = survey.line_geometry.snap_inline(float(inline_float))
        record.nearest_xline = survey.line_geometry.snap_xline(float(xline_float))
        record.survey_position = "inside"
    except ValueError:
        if record.distance_to_survey_m is not None and record.distance_to_survey_m <= float(near_survey_threshold_m):
            record.survey_position = "near_outside"
            record.reasons.append("near_outside_survey")
        else:
            record.survey_position = "outside"
            record.reasons.append("outside_survey")


def build_inventory(
    *,
    well_heads_df: pd.DataFrame,
    las_lookup: dict[str, Path],
    trace_lookup: dict[str, Path],
    time_depth_lookup: dict[str, Path],
    tops_lookup: dict[str, str],
    survey: Any,
    config: dict[str, Any],
) -> WellInventory:
    heads = _head_lookup(well_heads_df)
    master_keys = set(heads) | set(las_lookup)

    records: list[WellInventoryRecord] = []
    for key in sorted(master_keys, key=lambda item: (heads.get(item).well_name if item in heads else las_lookup[item].stem).casefold()):
        head = heads.get(key)
        has_head = head is not None
        has_las = key in las_lookup
        has_trace = key in trace_lookup
        has_time_depth = key in time_depth_lookup
        has_tops = key in tops_lookup
        well_name = head.well_name if head is not None else las_lookup[key].stem

        inventory_status, reasons = determine_inventory_status(has_las=has_las, has_well_head=has_head)
        record = WellInventoryRecord(
            well_name=well_name,
            has_well_head=has_head,
            has_las=has_las,
            has_well_trace=has_trace,
            has_time_depth=has_time_depth,
            has_well_tops=has_tops,
            surface_x=head.surface_x if head is not None else None,
            surface_y=head.surface_y if head is not None else None,
            bottom_x=head.bottom_x if head is not None else None,
            bottom_y=head.bottom_y if head is not None else None,
            kb_m=head.kb_m if head is not None else None,
            inventory_status=inventory_status,
            reasons=reasons,
        )

        if has_head:
            record.wellbore_class, record.bottom_offset_m = classify_wellbore(
                record.surface_x,
                record.surface_y,
                record.bottom_x,
                record.bottom_y,
                vertical_bottom_offset_threshold_m=float(config["spatial_qc"]["vertical_bottom_offset_threshold_m"]),
            )
            if record.wellbore_class == "unknown":
                record.reasons.append("invalid_bottom_xy")
            _classify_survey_position(
                record,
                survey=survey,
                near_survey_threshold_m=float(config["spatial_qc"]["near_survey_threshold_m"]),
            )
        else:
            record.survey_position = "invalid_xy"
            record.wellbore_class = "unknown"

        _record_reasons_for_assets(
            record,
            has_well_trace=has_trace,
            has_time_depth=has_time_depth,
            has_well_tops=has_tops,
        )
        records.append(record)

    neighbor_pairs, neighbor_summary = build_neighbor_pairs(
        records,
        dense_well_neighbor_threshold_m=float(config["spatial_qc"]["dense_well_neighbor_threshold_m"]),
        platform_cluster_threshold_m=float(config["spatial_qc"]["platform_cluster_threshold_m"]),
    )
    cluster_rows = build_cluster_rows(
        records,
        platform_cluster_threshold_m=float(config["spatial_qc"]["platform_cluster_threshold_m"]),
    )
    cluster_ids = {row.cluster_id for row in cluster_rows}

    summary = {
        "well_count": len(records),
        "asset_counts": {
            "well_heads": len(heads),
            "las": len(las_lookup),
            "well_trace": len(trace_lookup),
            "time_depth": len(time_depth_lookup),
            "well_tops_wells": len(tops_lookup),
            "head_only": sum(1 for record in records if record.has_well_head and not record.has_las),
            "las_only": sum(1 for record in records if record.has_las and not record.has_well_head),
            "head_and_las": sum(1 for record in records if record.has_well_head and record.has_las),
        },
        "survey_position_counts": value_counts(records, "survey_position"),
        "inventory_status_counts": value_counts(records, "inventory_status"),
        "wellbore_class_counts": value_counts(records, "wellbore_class"),
        "neighbor_summary": {
            **neighbor_summary,
            "platform_cluster_count": len(cluster_ids),
            "platform_cluster_well_count": len(cluster_rows),
        },
        "head_only_wells": sorted([record.well_name for record in records if record.has_well_head and not record.has_las]),
        "las_only_wells": sorted([record.well_name for record in records if record.has_las and not record.has_well_head]),
    }
    return WellInventory(records=records, neighbor_pairs=neighbor_pairs, cluster_rows=cluster_rows, summary=summary)


def _write_outputs(inventory: WellInventory, output_dir: Path, run_summary: dict[str, Any]) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "well_inventory_csv": output_dir / "well_inventory.csv",
        "well_neighbor_pairs_csv": output_dir / "well_neighbor_pairs.csv",
        "well_clusters_csv": output_dir / "well_clusters.csv",
        "run_summary_json": output_dir / "run_summary.json",
    }
    inventory.records_dataframe().to_csv(paths["well_inventory_csv"], index=False, encoding="utf-8")
    inventory.neighbor_pairs_dataframe().to_csv(paths["well_neighbor_pairs_csv"], index=False, encoding="utf-8")
    inventory.clusters_dataframe().to_csv(paths["well_clusters_csv"], index=False, encoding="utf-8")
    write_json(paths["run_summary_json"], run_summary)
    return paths


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _script_config(cfg)

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_dir = _resolve_output_dir(args, cfg)

    source_data = dict(script_cfg["source_data"])
    well_heads_file = _resolve_data_path(source_data["well_heads_file"], data_root=data_root)
    las_dir = _resolve_data_path(source_data["las_dir"], data_root=data_root)
    well_trace_dir = _resolve_data_path(source_data["well_trace_dir"], data_root=data_root)
    well_tops_file = _resolve_data_path(source_data["well_tops_file"], data_root=data_root)
    time_depth_dir = _resolve_data_path(source_data["time_depth_dir"], data_root=data_root)
    seismic_cfg = dict(script_cfg["seismic"])
    seismic_file = _resolve_data_path(seismic_cfg["file"], data_root=data_root)
    seismic_type = str(seismic_cfg.get("type", "zgy")).lower()

    well_heads_df, las_lookup, trace_lookup, time_depth_lookup, tops_lookup = _load_asset_lookups(
        well_heads_file=well_heads_file,
        las_dir=las_dir,
        well_trace_dir=well_trace_dir,
        well_tops_file=well_tops_file,
        time_depth_dir=time_depth_dir,
    )

    survey = open_survey(
        seismic_file,
        seismic_type=seismic_type,
        segy_options=segy_options_from_config(seismic_cfg) if seismic_type == "segy" else None,
    )
    geometry = survey.describe_geometry(domain="time")
    bin_spacing = survey.line_geometry.bin_spacing_m()
    footprint = survey.line_geometry.footprint_xy()

    inventory = build_inventory(
        well_heads_df=well_heads_df,
        las_lookup=las_lookup,
        trace_lookup=trace_lookup,
        time_depth_lookup=time_depth_lookup,
        tops_lookup=tops_lookup,
        survey=survey,
        config=script_cfg,
    )

    run_summary = {
        "script": "well_inventory.py",
        "config_file": repo_relative_path(args.config, root=REPO_ROOT),
        "inputs": {
            "data_root": repo_relative_path(data_root, root=REPO_ROOT),
            "well_heads_file": repo_relative_path(well_heads_file, root=REPO_ROOT),
            "las_dir": repo_relative_path(las_dir, root=REPO_ROOT),
            "well_trace_dir": repo_relative_path(well_trace_dir, root=REPO_ROOT),
            "well_tops_file": repo_relative_path(well_tops_file, root=REPO_ROOT),
            "time_depth_dir": repo_relative_path(time_depth_dir, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
            "seismic_type": seismic_type,
        },
        "thresholds": {
            "near_survey_threshold_m": float(script_cfg["spatial_qc"]["near_survey_threshold_m"]),
            "vertical_bottom_offset_threshold_m": float(script_cfg["spatial_qc"]["vertical_bottom_offset_threshold_m"]),
            "platform_cluster_threshold_m": float(script_cfg["spatial_qc"]["platform_cluster_threshold_m"]),
            "dense_well_neighbor_threshold_m": float(script_cfg["spatial_qc"]["dense_well_neighbor_threshold_m"]),
        },
        "geometry": geometry,
        "bin_spacing_m": bin_spacing,
        "footprint_xy": footprint,
        **inventory.summary,
    }
    paths = _write_outputs(inventory, output_dir, run_summary)

    print(f"Saved well inventory: {paths['well_inventory_csv']}")
    print(f"Saved neighbor pairs: {paths['well_neighbor_pairs_csv']}")
    print(f"Saved well clusters: {paths['well_clusters_csv']}")
    print(f"Saved run summary: {paths['run_summary_json']}")
    print(
        "Inventory summary: "
        f"{inventory.summary['well_count']} wells, "
        f"{inventory.summary['asset_counts']['head_only']} head-only, "
        f"{inventory.summary['asset_counts']['las_only']} LAS-only, "
        f"{inventory.summary['neighbor_summary']['exported_neighbor_pair_count']} exported same-trace conflicts, "
        f"{inventory.summary['neighbor_summary']['platform_cluster_count']} platform clusters."
    )


if __name__ == "__main__":
    main()
