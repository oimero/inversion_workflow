"""Run L0 synthetic + configured real-well delta-anchor holdout validation.

Usage::

    python scripts/l0_real_delta_anchor.py
    python scripts/l0_real_delta_anchor.py --config experiments/common/common.yaml
    python scripts/l0_real_delta_anchor.py --output-dir scripts/output/l0_real_delta_anchor_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
src_text = str(SRC_DIR)
sys.path = [path for path in sys.path if path != src_text]
sys.path.insert(0, src_text)

from cup.config.workflow import TimeWorkflowConfig
from cup.utils.io import load_yaml_config, resolve_relative_path, write_json
from ginn_v2.l0 import run_l0


DEFAULT_CONFIG = Path("experiments/common/common.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    raw = load_yaml_config(config_path)
    workflow = TimeWorkflowConfig.from_mapping(raw)
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    output_dir = (
        resolve_relative_path(args.output_dir, root=REPO_ROOT)
        if args.output_dir is not None
        else output_root / f"l0_real_delta_anchor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print("=== L0 Real-Delta Anchor Validation ===")
    print(f"Output: {output_dir}")
    try:
        summary = run_l0(
            raw_config=raw,
            repo_root=REPO_ROOT,
            data_root=resolve_relative_path(workflow.data_root, root=REPO_ROOT),
            output_dir=output_dir,
        )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "l0_failure.json",
            {
                "schema_version": "l0_failure_v1",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(f"Decision: {summary['status']}")
    print(f"Eligible for L1: {summary['eligible_for_l1']}")


if __name__ == "__main__":
    main()
