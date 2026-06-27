"""Build the Step 7 real-field model input package for R0/R1/R2.

The package contains the real-field LFM and full-preprocessed model-grid well
targets.  It is a hard v2 contract; historical ``real_field_lfm`` configuration
and artifacts are intentionally rejected.

Usage::

    python scripts/real_field_model_inputs.py
    python scripts/real_field_model_inputs.py --config experiments/common/common.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.real_field_model_inputs import (
    parse_real_field_model_inputs_config,
    run_real_field_model_inputs,
)
from cup.config.workflow import TimeWorkflowConfig
from cup.config.sources import resolve_source_file_from_run, resolve_source_run
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path


DEFAULT_COMMON_CONFIG = Path("experiments/common/common.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_COMMON_CONFIG,
        help="YAML config containing a real_field_model_inputs section.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _output_dir(args: argparse.Namespace, workflow: TimeWorkflowConfig) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"real_field_model_inputs_{timestamp}"


def _prepare_real_field_model_inputs_config(raw: dict, workflow: TimeWorkflowConfig) -> dict:
    prepared = dict(raw)
    if "real_field_lfm" in prepared:
        raise ValueError("real_field_lfm is retired; use real_field_model_inputs and rerun Step 7.")
    root = dict(prepared.get("real_field_model_inputs") or {})
    source_runs = dict(root.get("source_runs") or {})
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    source_runs["well_auto_tie_dir"] = repo_relative_path(
        resolve_source_run(
            source_runs.get("well_auto_tie_dir"),
            output_root=output_root,
            prefix="well_auto_tie",
            required_files=["well_tie_metrics.csv", "well_tie_plan.csv", "wavelet_inventory.csv"],
            root=REPO_ROOT,
            label="well_auto_tie",
        ),
        root=REPO_ROOT,
    )
    source_runs["well_preprocess_dir"] = repo_relative_path(
        resolve_source_run(
            source_runs.get("well_preprocess_dir"),
            output_root=output_root,
            prefix="well_preprocess",
            required_files=["well_preprocess_status.csv", "run_summary.json"],
            root=REPO_ROOT,
            label="well_preprocess",
        ),
        root=REPO_ROOT,
    )
    root["well_inventory_file"] = repo_relative_path(
        resolve_source_file_from_run(
            root.get("well_inventory_file"),
            output_root=output_root,
            prefix="well_inventory",
            file_name="well_inventory.csv",
            root=REPO_ROOT,
            label="well_inventory_file",
        ),
        root=REPO_ROOT,
    )
    root["source_runs"] = source_runs
    prepared["real_field_model_inputs"] = root
    return prepared


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = load_yaml_config(config_path)
    workflow = TimeWorkflowConfig.from_mapping(raw)
    prepared = _prepare_real_field_model_inputs_config(raw, workflow)
    script_cfg = parse_real_field_model_inputs_config(prepared)
    output_dir = _output_dir(args, workflow)
    summary = run_real_field_model_inputs(
        config=script_cfg,
        repo_root=REPO_ROOT,
        data_root=resolve_relative_path(workflow.data_root, root=REPO_ROOT),
        output_dir=output_dir,
    )
    print("=== Step 7 Real-field Model Inputs ===")
    print(f"Output: {output_dir}")
    print(f"Status: {summary.get('status', 'unknown')}")
    controls = summary.get("control_wells", {})
    print(f"Accepted controls: {controls.get('n_accepted', 0)} / {controls.get('n_total_metrics', 0)}")


if __name__ == "__main__":
    main()
