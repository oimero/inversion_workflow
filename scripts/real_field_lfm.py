"""Build explicit unified real-field LFM v3 variants from canonical well controls.

Usage::

    python scripts/real_field_lfm.py
    python scripts/real_field_lfm.py --config experiments/my_project.yaml
    python scripts/real_field_lfm.py --output-dir scripts/output/lfm_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.seismic.lfm.pipeline import build_lfm_context, run_lfm_pipeline
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import latest_checked_run, load_yaml_config, repo_relative_path, resolve_relative_path
from cup.well.real_field_controls import SCHEMA_VERSION as WELL_CONTROL_SCHEMA, load_well_control_set


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


def _validate_control_run(path: Path, *, domain: str, depth_basis: str | None) -> None:
    with (path / "run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != WELL_CONTROL_SCHEMA or summary.get("status") != "ok":
        raise ValueError(f"not a successful {WELL_CONTROL_SCHEMA} run")
    axis = dict(summary.get("sample_axis") or {})
    if axis.get("sample_domain") != domain or summary.get("depth_basis") != depth_basis:
        raise ValueError("well controls domain/depth_basis does not match seismic")


def _control_run(config: dict[str, Any], workflow: WorkflowConfig) -> Path:
    source_runs = config.get("source_runs")
    if not isinstance(source_runs, dict):
        raise ValueError("real_field_lfm.source_runs must be a mapping.")
    text = str(source_runs.get("well_control_run_dir") or "").strip()
    if text:
        path = resolve_relative_path(text, root=REPO_ROOT)
        _validate_control_run(path, domain=workflow.seismic.domain, depth_basis=workflow.seismic.depth_basis)
        return path
    return latest_checked_run(
        resolve_relative_path(workflow.output_root, root=REPO_ROOT),
        "real_field_well_controls",
        required_files=["run_summary.json", "well_control_manifest.csv"],
        validator=lambda path: _validate_control_run(
            path, domain=workflow.seismic.domain, depth_basis=workflow.seismic.depth_basis
        ),
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = _load_composed_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    lfm_config = raw.get("real_field_lfm")
    if not isinstance(lfm_config, dict):
        raise ValueError("Config lacks real_field_lfm section.")
    controls_run = _control_run(lfm_config, workflow)
    controls = load_well_control_set(controls_run, repo_root=REPO_ROOT)
    if controls.sample_domain != workflow.seismic.domain or controls.depth_basis != workflow.seismic.depth_basis:
        raise ValueError("Canonical WellControlSet domain/depth_basis does not match seismic.")

    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    recorded_seismic_path = resolve_relative_path(
        str(controls.provenance.get("target_seismic_path") or ""), root=REPO_ROOT
    )
    if recorded_seismic_path.resolve() != seismic_path.resolve():
        raise ValueError("WellControlSet target seismic path does not match the current workflow seismic.")
    seismic_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=seismic_options or None)
    if not np.array_equal(controls.sample_axis.values, survey.sample_axis(workflow.seismic.domain).values):
        raise ValueError("WellControlSet SampleAxis differs from the current source seismic.")
    context, horizon_sources = build_lfm_context(
        raw_config=raw,
        workflow=workflow,
        survey=survey,
        data_root=data_root,
        repo_root=REPO_ROOT,
        common_sources={
            "seismic": {
                "path": repo_relative_path(seismic_path, root=REPO_ROOT),
                "type": workflow.seismic.type,
            }
        },
    )
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = resolve_relative_path(workflow.output_root, root=REPO_ROOT) / f"real_field_lfm_{timestamp}"
    else:
        output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    summary = run_lfm_pipeline(
        config=lfm_config,
        controls=controls,
        context=context,
        controls_run=controls_run,
        horizon_sources=horizon_sources,
        source_seismic_file=seismic_path,
        source_seismic_type=workflow.seismic.type,
        seismic_options=workflow.seismic.as_dict(),
        output_dir=output_dir,
        repo_root=REPO_ROOT,
    )
    print("=== Unified Real-field LFM v3 ===")
    print(f"Output: {output_dir}")
    print(f"Variants: {len(summary['requested_variant_ids'])}")
    print(f"Status: {summary['status']}")


if __name__ == "__main__":
    main()
