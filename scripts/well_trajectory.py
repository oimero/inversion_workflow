"""QC Petrel well trajectories for the time-domain workflow.

This side-car script consumes the first workflow inventory output and parses
Petrel well trace exports.  It produces trajectory geometry facts for later
deviated-well routing without changing the LAS screening/preprocess chain.

Usage::

    python scripts/well_trajectory.py
    python scripts/well_trajectory.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.coerce import as_bool, optional_float
from cup.utils.config import merge_dict_defaults
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.well.assets import build_file_lookup, normalize_well_name
from cup.well.trajectory import WellTrajectory, trajectory_summary, z_tvd_residual_m


# =============================================================================
# CLI and config
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
        help="Output directory. Defaults to <output_root>/well_trajectory_<timestamp>.",
    )
    return parser.parse_args()


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("well_trajectory") or {})
    merge_dict_defaults(script_cfg, "source_runs", {"mode": "latest", "well_inventory_dir": None})
    script_cfg.setdefault("well_trace_dir", "all_well_trace")
    merge_dict_defaults(
        script_cfg,
        "seismic",
        {
            "file": "raw/obn-clipped-240-912-872-1544.zgy",
            "type": "zgy",
        },
    )
    merge_dict_defaults(
        script_cfg,
        "classification",
        {
            "vertical_max_offset_m": 30.0,
            "min_deviated_max_offset_m": 30.0,
            "surface_xy_tolerance_m": 2.0,
            "kb_tolerance_m": 0.5,
            "z_tvd_tolerance_m": 0.1,
        },
    )
    merge_dict_defaults(script_cfg, "survey_qc", {"enabled": True, "allow_partial_outside": True})
    merge_dict_defaults(script_cfg, "output", {"write_trajectory_points": True, "sampled_trajectory_dir": "trajectory_points"})
    return script_cfg


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_data_path(value: str | Path, *, data_root: Path) -> Path:
    return resolve_relative_path(value, root=data_root)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"well_trajectory_{timestamp}"


def _discover_latest_inventory_file(cfg: dict[str, Any], script_cfg: dict[str, Any]) -> Path:
    source_runs = dict(script_cfg.get("source_runs") or {})
    mode = str(source_runs.get("mode", "latest")).strip().casefold()
    if mode != "latest":
        raise ValueError(f"well_trajectory.source_runs.mode only supports 'latest' for now, got {mode!r}.")
    inventory_dir = source_runs.get("well_inventory_dir")
    if inventory_dir is not None:
        return _resolve_repo_path(inventory_dir) / "well_inventory.csv"

    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    all_candidates = [path for path in output_root.glob("well_inventory_*/well_inventory.csv") if path.is_file()]
    timestamped = [
        path
        for path in all_candidates
        if re.fullmatch(r"well_inventory_\d{8}_\d{6}", path.parent.name)
    ]
    candidates = sorted(timestamped or all_candidates, key=lambda path: path.parent.name)
    if not candidates:
        raise FileNotFoundError("No well_inventory_*/well_inventory.csv file found under output_root.")
    return candidates[-1]


# =============================================================================
# Survey helpers
# =============================================================================


@dataclass(frozen=True)
class SurveyPointQc:
    inline_float: float | None
    xline_float: float | None
    nearest_inline: float | None
    nearest_xline: float | None
    survey_position: str


def _classify_point_survey_position(x: float, y: float, *, survey: Any, geometry: dict[str, Any]) -> SurveyPointQc:
    try:
        inline_float, xline_float = survey.line_geometry.coord_to_line(float(x), float(y))
    except ValueError:
        return SurveyPointQc(None, None, None, None, "outside")

    return SurveyPointQc(
        inline_float=float(inline_float),
        xline_float=float(xline_float),
        nearest_inline=survey.line_geometry.snap_inline(float(inline_float)),
        nearest_xline=survey.line_geometry.snap_xline(float(xline_float)),
        survey_position="inside",
    )


def _trajectory_point_qc(trajectory: WellTrajectory, *, survey: Any | None, geometry: dict[str, Any] | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(trajectory.point_count):
        point_qc = (
            _classify_point_survey_position(
                float(trajectory.x_m[index]),
                float(trajectory.y_m[index]),
                survey=survey,
                geometry=geometry,
            )
            if survey is not None and geometry is not None
            else SurveyPointQc(None, None, None, None, "not_checked")
        )
        rows.append(
            {
                "well_name": trajectory.well_name,
                "sample_index": index,
                "md_m": float(trajectory.md_m[index]),
                "tvd_kb_m": float(trajectory.tvd_kb_m[index]),
                "tvdss_m": float(trajectory.tvdss_m[index]),
                "z_m": float(trajectory.z_m[index]),
                "x_m": float(trajectory.x_m[index]),
                "y_m": float(trajectory.y_m[index]),
                "dx_m": float(trajectory.dx_m[index]) if np.isfinite(trajectory.dx_m[index]) else None,
                "dy_m": float(trajectory.dy_m[index]) if np.isfinite(trajectory.dy_m[index]) else None,
                "azim_deg": float(trajectory.azim_deg[index]) if np.isfinite(trajectory.azim_deg[index]) else None,
                "incl_deg": float(trajectory.incl_deg[index]) if np.isfinite(trajectory.incl_deg[index]) else None,
                "dls": float(trajectory.dls[index]) if np.isfinite(trajectory.dls[index]) else None,
                "inline_float": point_qc.inline_float,
                "xline_float": point_qc.xline_float,
                "nearest_inline": point_qc.nearest_inline,
                "nearest_xline": point_qc.nearest_xline,
                "survey_position": point_qc.survey_position,
            }
        )
    return pd.DataFrame.from_records(rows, columns=TRAJECTORY_POINT_COLUMNS)


# =============================================================================
# QC logic
# =============================================================================


MAIN_COLUMNS = [
    "well_name",
    "trajectory_file",
    "trajectory_status",
    "wellbore_class_initial",
    "wellbore_class_qc",
    "class_changed",
    "point_count",
    "md_min_m",
    "md_max_m",
    "tvd_kb_min_m",
    "tvd_kb_max_m",
    "tvdss_min_m",
    "tvdss_max_m",
    "surface_x_m",
    "surface_y_m",
    "bottom_x_m",
    "bottom_y_m",
    "surface_to_bottom_offset_m",
    "max_horizontal_offset_m",
    "max_incl_deg",
    "max_dls",
    "surface_survey_position",
    "bottom_survey_position",
    "trajectory_inside_fraction",
    "trajectory_inside_sample_count",
    "trajectory_outside_sample_count",
    "qc_flags",
    "reasons",
]

FAILED_COLUMNS = ["well_name", "trajectory_file", "trajectory_status", "qc_flags", "reasons"]

TRAJECTORY_POINT_COLUMNS = [
    "well_name",
    "sample_index",
    "md_m",
    "tvd_kb_m",
    "tvdss_m",
    "z_m",
    "x_m",
    "y_m",
    "dx_m",
    "dy_m",
    "azim_deg",
    "incl_deg",
    "dls",
    "inline_float",
    "xline_float",
    "nearest_inline",
    "nearest_xline",
    "survey_position",
]


def _classify_wellbore_from_trajectory(
    max_horizontal_offset_m: float | None,
    *,
    vertical_max_offset_m: float,
    min_deviated_max_offset_m: float,
) -> str:
    if max_horizontal_offset_m is None or not np.isfinite(max_horizontal_offset_m):
        return "unknown"
    if float(max_horizontal_offset_m) <= float(vertical_max_offset_m):
        return "vertical"
    if float(max_horizontal_offset_m) > float(min_deviated_max_offset_m):
        return "deviated"
    return "unknown"


def _status_from_flags(*, hard_reasons: Sequence[str], qc_flags: Sequence[str]) -> str:
    if hard_reasons:
        return "failed"
    if qc_flags:
        return "warning"
    return "passed"


def _joined(values: Sequence[str]) -> str:
    return ";".join(values)


def _base_missing_row(well_name: str, wellbore_class_initial: str, trace_path: Path | None, status: str, reasons: Sequence[str]) -> dict[str, Any]:
    row = {column: None for column in MAIN_COLUMNS}
    row.update(
        {
            "well_name": well_name,
            "trajectory_file": repo_relative_path(trace_path, root=REPO_ROOT) if trace_path is not None else None,
            "trajectory_status": status,
            "wellbore_class_initial": wellbore_class_initial,
            "wellbore_class_qc": "unknown",
            "class_changed": wellbore_class_initial != "unknown",
            "qc_flags": "",
            "reasons": _joined(reasons),
        }
    )
    return row


def _qc_inventory_consistency(
    trajectory: WellTrajectory,
    inventory_row: pd.Series,
    *,
    classification_cfg: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    inv_x = optional_float(inventory_row.get("surface_x"))
    inv_y = optional_float(inventory_row.get("surface_y"))
    if inv_x is not None and inv_y is not None:
        surface_distance = float(np.hypot(trajectory.x_m[0] - inv_x, trajectory.y_m[0] - inv_y))
        if surface_distance > float(classification_cfg["surface_xy_tolerance_m"]):
            flags.append("surface_xy_mismatch")

    inv_kb = optional_float(inventory_row.get("kb_m"))
    if inv_kb is not None and abs(float(trajectory.kb_m) - inv_kb) > float(classification_cfg["kb_tolerance_m"]):
        flags.append("kb_mismatch")

    residual = z_tvd_residual_m(trajectory)
    max_abs_residual = float(np.nanmax(np.abs(residual)))
    if np.isfinite(max_abs_residual) and max_abs_residual > float(classification_cfg["z_tvd_tolerance_m"]):
        flags.append("z_tvd_inconsistent")

    if np.any(trajectory.md_m < 0.0) or np.any(trajectory.tvd_kb_m < 0.0):
        flags.append("invalid_depth_values")

    invalid_row_count = int(trajectory.metadata.get("invalid_required_row_count") or 0)
    if invalid_row_count > 0:
        flags.append("invalid_required_rows_dropped")

    return flags


def _build_success_row(
    *,
    trajectory: WellTrajectory,
    trace_path: Path,
    inventory_row: pd.Series,
    classification_cfg: dict[str, Any],
    survey_cfg: dict[str, Any],
    point_df: pd.DataFrame,
    hard_reasons: list[str],
    qc_flags: list[str],
) -> dict[str, Any]:
    summary = dict(trajectory_summary(trajectory))
    wellbore_class_initial = str(inventory_row.get("wellbore_class", "unknown"))

    inside_count = int((point_df["survey_position"] == "inside").sum()) if not point_df.empty else 0
    outside_count = int((point_df["survey_position"] == "outside").sum()) if not point_df.empty else 0
    checked_count = inside_count + outside_count
    inside_fraction = float(inside_count / checked_count) if checked_count > 0 else None
    surface_position = str(point_df.iloc[0]["survey_position"]) if not point_df.empty else None
    bottom_position = str(point_df.iloc[-1]["survey_position"]) if not point_df.empty else None
    is_partial_outside = inside_count > 0 and outside_count > 0
    if is_partial_outside:
        if as_bool(survey_cfg.get("allow_partial_outside", True)):
            qc_flags.append("partial_outside_survey")
        else:
            hard_reasons.append("partial_outside_survey")

    status = _status_from_flags(hard_reasons=hard_reasons, qc_flags=qc_flags)
    wellbore_class_qc = (
        "unknown"
        if status == "failed"
        else _classify_wellbore_from_trajectory(
            summary.get("max_horizontal_offset_m"),
            vertical_max_offset_m=float(classification_cfg["vertical_max_offset_m"]),
            min_deviated_max_offset_m=float(classification_cfg["min_deviated_max_offset_m"]),
        )
    )

    row = {column: None for column in MAIN_COLUMNS}
    row.update(summary)
    row.update(
        {
            "well_name": str(inventory_row["well_name"]),
            "trajectory_file": repo_relative_path(trace_path, root=REPO_ROOT),
            "trajectory_status": status,
            "wellbore_class_initial": wellbore_class_initial,
            "wellbore_class_qc": wellbore_class_qc,
            "class_changed": wellbore_class_initial != wellbore_class_qc,
            "surface_survey_position": surface_position,
            "bottom_survey_position": bottom_position,
            "trajectory_inside_fraction": inside_fraction,
            "trajectory_inside_sample_count": inside_count,
            "trajectory_outside_sample_count": outside_count,
            "qc_flags": _joined(qc_flags),
            "reasons": _joined(hard_reasons),
        }
    )
    return row


def _failed_report_rows(main_rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "well_name": row.get("well_name"),
            "trajectory_file": row.get("trajectory_file"),
            "trajectory_status": row.get("trajectory_status"),
            "qc_flags": row.get("qc_flags"),
            "reasons": row.get("reasons"),
        }
        for row in main_rows
        if row.get("trajectory_status") in {"failed", "missing"}
    ]
    return pd.DataFrame.from_records(rows, columns=FAILED_COLUMNS)


def _value_counts(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) if row.get(key) is not None else "null")
        counts[value] = counts.get(value, 0) + 1
    return counts


# =============================================================================
# Main workflow
# =============================================================================


def run_trajectory(
    *,
    inventory_df: pd.DataFrame,
    trace_lookup: dict[str, Path],
    survey: Any | None,
    geometry: dict[str, Any] | None,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    classification_cfg = dict(config["classification"])
    survey_cfg = dict(config.get("survey_qc") or {})
    output_cfg = dict(config.get("output") or {})
    write_points = as_bool(output_cfg.get("write_trajectory_points", True))
    point_dir = output_dir / str(output_cfg.get("sampled_trajectory_dir", "trajectory_points"))
    if write_points:
        point_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, Any]] = []
    inventory_keys = {normalize_well_name(name) for name in inventory_df["well_name"].tolist()}

    for _, inventory_row in inventory_df.iterrows():
        well_name = str(inventory_row["well_name"])
        well_key = normalize_well_name(well_name)
        wellbore_class_initial = str(inventory_row.get("wellbore_class", "unknown"))
        trace_path = trace_lookup.get(well_key)
        if trace_path is None:
            main_rows.append(_base_missing_row(well_name, wellbore_class_initial, None, "missing", ["no_well_trace"]))
            continue

        hard_reasons: list[str] = []
        qc_flags: list[str] = []
        try:
            trajectory = WellTrajectory.from_petrel_trace(trace_path)
        except ValueError as exc:
            main_rows.append(_base_missing_row(well_name, wellbore_class_initial, trace_path, "failed", [str(exc)]))
            continue

        if trajectory.header_well_name is None:
            qc_flags.append("missing_header_well_name")
        elif normalize_well_name(trajectory.header_well_name) != well_key:
            hard_reasons.append("name_mismatch")

        qc_flags.extend(
            _qc_inventory_consistency(
                trajectory,
                inventory_row,
                classification_cfg=classification_cfg,
            )
        )

        point_df = _trajectory_point_qc(trajectory, survey=survey, geometry=geometry)
        point_df["well_name"] = well_name
        trajectory = trajectory.with_well_name(well_name)

        if write_points:
            point_path = point_dir / f"{sanitize_filename(well_name)}.csv"
            point_df.to_csv(point_path, index=False)

        main_rows.append(
            _build_success_row(
                trajectory=trajectory,
                trace_path=trace_path,
                inventory_row=inventory_row,
                classification_cfg=classification_cfg,
                survey_cfg=survey_cfg,
                point_df=point_df,
                hard_reasons=hard_reasons,
                qc_flags=qc_flags,
            )
        )

    orphan_trace_files = [
        repo_relative_path(path, root=REPO_ROOT)
        for key, path in sorted(trace_lookup.items(), key=lambda item: item[1].name.casefold())
        if key not in inventory_keys
    ]
    main_df = pd.DataFrame.from_records(main_rows, columns=MAIN_COLUMNS)
    failed_df = _failed_report_rows(main_rows)
    summary = {
        "well_count": int(len(main_rows)),
        "trace_file_count": int(len(trace_lookup)),
        "orphan_trace_file_count": int(len(orphan_trace_files)),
        "orphan_trace_files": orphan_trace_files,
        "trajectory_status_counts": _value_counts(main_rows, "trajectory_status"),
        "wellbore_class_initial_counts": _value_counts(main_rows, "wellbore_class_initial"),
        "wellbore_class_qc_counts": _value_counts(main_rows, "wellbore_class_qc"),
        "class_changed_count": int(sum(1 for row in main_rows if bool(row.get("class_changed")))),
        "failed_or_missing_count": int(len(failed_df)),
        "trajectory_points_written": bool(write_points),
        "trajectory_point_dir": repo_relative_path(point_dir, root=REPO_ROOT) if write_points else None,
    }
    return main_df, failed_df, summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _script_config(cfg)
    data_root = _resolve_repo_path(str(cfg.get("data_root", "data")))
    output_dir = _resolve_output_dir(args, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_file = _discover_latest_inventory_file(cfg, script_cfg)
    if not inventory_file.exists():
        raise FileNotFoundError(f"well_inventory.csv does not exist: {inventory_file}")
    inventory_df = pd.read_csv(inventory_file)
    if "well_name" not in inventory_df.columns:
        raise ValueError(f"Inventory file is missing required column 'well_name': {inventory_file}")

    well_trace_dir = _resolve_data_path(script_cfg["well_trace_dir"], data_root=data_root)
    trace_lookup = build_file_lookup(well_trace_dir.iterdir() if well_trace_dir.exists() else [], asset_label=str(well_trace_dir))

    survey_cfg = dict(script_cfg.get("survey_qc") or {})
    survey = None
    geometry = None
    seismic_file = None
    seismic_cfg = dict(script_cfg.get("seismic") or {})
    if as_bool(survey_cfg.get("enabled", True)):
        seismic_file = _resolve_data_path(seismic_cfg["file"], data_root=data_root)
        survey = open_survey(
            seismic_file,
            seismic_type=str(seismic_cfg.get("type", "segy")),
            segy_options=segy_options_from_config(seismic_cfg) or None,
        )
        geometry = survey.describe_geometry(domain="time")

    main_df, failed_df, summary = run_trajectory(
        inventory_df=inventory_df,
        trace_lookup=trace_lookup,
        survey=survey,
        geometry=geometry,
        config=script_cfg,
        output_dir=output_dir,
    )

    paths = {
        "well_trajectory": output_dir / "well_trajectory.csv",
        "failed_trajectories": output_dir / "failed_trajectories.csv",
        "run_summary": output_dir / "run_summary.json",
    }
    main_df.to_csv(paths["well_trajectory"], index=False)
    failed_df.to_csv(paths["failed_trajectories"], index=False)

    run_summary = {
        "script": "well_trajectory.py",
        "config_file": repo_relative_path(args.config, root=REPO_ROOT),
        "inputs": {
            "data_root": repo_relative_path(data_root, root=REPO_ROOT),
            "inventory_file": repo_relative_path(inventory_file, root=REPO_ROOT),
            "well_trace_dir": repo_relative_path(well_trace_dir, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT) if seismic_file is not None else None,
            "seismic_type": seismic_cfg.get("type"),
        },
        "thresholds": dict(script_cfg["classification"]),
        "survey_qc": survey_cfg,
        "geometry": geometry,
        "summary": summary,
        "paths": {key: repo_relative_path(path, root=REPO_ROOT) for key, path in paths.items()},
    }
    write_json(paths["run_summary"], run_summary)

    print(
        f"Wrote trajectory QC for {len(main_df)} wells to {repo_relative_path(output_dir, root=REPO_ROOT)} "
        f"({summary['trajectory_status_counts']})."
    )


if __name__ == "__main__":
    main()
