"""Prepare canonical real-field WellControlSet artifacts for every LFM variant.

Usage::

    python scripts/real_field_well_controls.py
    python scripts/real_field_well_controls.py --config experiments/my_project.yaml
    python scripts/real_field_well_controls.py --output-dir scripts/output/well_controls_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.sources import resolve_source_file_from_run
from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import latest_checked_run, load_yaml_config, resolve_relative_path
from cup.well.real_field_controls import (
    DEPTH_SOURCE_SCHEMA,
    TIME_SOURCE_SCHEMA,
    build_well_control_set,
    write_well_control_set,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_composed_config(path: Path) -> dict[str, Any]:
    experiment = load_yaml_config(path)
    workflow_config = str(experiment.get("workflow_config") or "").strip()
    if not workflow_config:
        return experiment
    common = load_yaml_config(resolve_relative_path(workflow_config, root=REPO_ROOT))
    overlay = {key: value for key, value in experiment.items() if key != "workflow_config"}
    return deep_merge_dict(common, overlay)


def _source_validator(*, source_run_type: str, domain: str, depth_basis: str | None):
    expected_schema = TIME_SOURCE_SCHEMA if source_run_type == "well_auto_tie" else DEPTH_SOURCE_SCHEMA

    def validate(path: Path) -> None:
        with (path / "run_summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("schema_version") != expected_schema:
            raise ValueError(f"schema_version is not {expected_schema}")
        if summary.get("sample_domain") != domain or summary.get("depth_basis") != depth_basis:
            raise ValueError("source domain/depth_basis does not match workflow seismic")
        if summary.get("status") not in {"ok", "success"}:
            raise ValueError("source run is not successful")

    return validate


def _resolve_source_run(config: dict[str, Any], workflow: WorkflowConfig) -> Path:
    source_run_type = str(config.get("source_run_type") or "").strip()
    if source_run_type not in {"well_auto_tie", "wavelet_batch_synthetic_depth"}:
        raise ValueError("real_field_well_controls.source_run_type must be explicit.")
    explicit = str(config.get("source_run_dir") or "").strip()
    if explicit:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    prefix = source_run_type
    required = (
        ["run_summary.json", "well_tie_metrics.csv"]
        if source_run_type == "well_auto_tie"
        else ["run_summary.json", "wavelet_batch_metrics.csv"]
    )
    return latest_checked_run(
        resolve_relative_path(workflow.output_root, root=REPO_ROOT),
        prefix,
        required_files=required,
        validator=_source_validator(
            source_run_type=source_run_type,
            domain=workflow.seismic.domain,
            depth_basis=workflow.seismic.depth_basis,
        ),
    )


def _resolved_config(raw: dict[str, Any], workflow: WorkflowConfig) -> dict[str, Any]:
    config = dict(raw.get("real_field_well_controls") or {})
    if not config:
        raise ValueError("Config lacks real_field_well_controls section.")
    config["source_run_dir"] = str(_resolve_source_run(config, workflow))
    inventory_value = str(config.get("well_inventory_file") or "").strip()
    if inventory_value:
        inventory = resolve_relative_path(inventory_value, root=REPO_ROOT)
    else:
        inventory = resolve_source_file_from_run(
            None,
            output_root=resolve_relative_path(workflow.output_root, root=REPO_ROOT),
            prefix="well_inventory",
            file_name="well_inventory.csv",
            root=REPO_ROOT,
            label="well_inventory_file",
        )
    config["well_inventory_file"] = str(inventory)
    config["well_trace_dir"] = workflow.assets.well_trace_dir
    return config


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = _load_composed_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    config = _resolved_config(raw, workflow)
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options=segy_options_from_config(workflow.seismic.as_dict()),
    )
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    controls, manifest = build_well_control_set(
        config=config,
        sample_axis=sample_axis,
        line_geometry=survey.line_geometry,
        domain=workflow.seismic.domain,
        depth_basis=workflow.seismic.depth_basis,
        repo_root=REPO_ROOT,
        data_root=data_root,
        seismic_path=seismic_path,
    )
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = resolve_relative_path(workflow.output_root, root=REPO_ROOT) / f"real_field_well_controls_{timestamp}"
    else:
        output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    summary = write_well_control_set(
        controls,
        manifest,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        resolved_config=config,
    )
    print("=== Real-field Well Controls ===")
    print(f"Output: {output_dir}")
    print(f"Successful wells: {summary['counts']['successful_wells']}")


if __name__ == "__main__":
    main()
