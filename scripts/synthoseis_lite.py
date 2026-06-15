"""Calibrate and generate the truth-first synthoseis-lite benchmark slice.

Usage::

    python scripts/synthoseis_lite.py calibrate
    python scripts/synthoseis_lite.py generate --impedance-calibration <file>
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

from cup.synthetic.workflow import (
    parse_synthoseis_config,
    resolve_sources,
    run_calibration,
    run_generation,
)
from cup.time_config import TimeWorkflowConfig
from cup.utils.io import load_yaml_config, resolve_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common.yaml"),
        help="YAML configuration containing synthoseis_lite.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("calibrate", help="Freeze impedance calibration from third/fourth-step LAS.")
    generate = subparsers.add_parser("generate", help="Generate field-conditioned truth and nominal seismic.")
    generate.add_argument("--impedance-calibration", type=Path, required=True)
    generate.add_argument(
        "--debug-attempt-limit",
        type=int,
        default=None,
        help="Development-only attempt cap; acceptance thresholds are not evaluated.",
    )
    generate.add_argument(
        "--geometry-family",
        action="append",
        choices=("none", "wedge", "pinchout"),
        default=None,
        help="Restrict generation to one or more geometry families.",
    )
    generate.add_argument(
        "--qc-only",
        action="store_true",
        help="Run full generation and acceptance QC without persisting realization arrays.",
    )
    return parser.parse_args()


def _output_dir(args: argparse.Namespace, workflow: TimeWorkflowConfig) -> Path:
    if args.output_dir is not None:
        return resolve_relative_path(args.output_dir, root=REPO_ROOT)
    root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"synthoseis_lite_{args.command}_{timestamp}"


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = load_yaml_config(config_path)
    workflow = TimeWorkflowConfig.from_mapping(raw)
    script_cfg = parse_synthoseis_config(raw)
    sources = resolve_sources(script_cfg, repo_root=REPO_ROOT)
    output_dir = _output_dir(args, workflow)
    if args.command == "calibrate":
        summary = run_calibration(
            workflow=workflow,
            script_cfg=script_cfg,
            sources=sources,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
        )
    else:
        calibration = resolve_relative_path(args.impedance_calibration, root=REPO_ROOT)
        summary = run_generation(
            workflow=workflow,
            script_cfg=script_cfg,
            sources=sources,
            calibration_path=calibration,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            debug_attempt_limit=args.debug_attempt_limit,
            geometry_families=args.geometry_family,
            qc_only=args.qc_only,
        )
    print("=== synthoseis-lite ===")
    print(f"Command: {args.command}")
    print(f"Output: {output_dir}")
    print(f"Status: {summary.get('status', 'ok')}")


if __name__ == "__main__":
    main()
