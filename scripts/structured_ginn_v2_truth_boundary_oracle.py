"""Run the Stage 1 truth-boundary three-parameter observability diagnostic.

Usage::

    python scripts/structured_ginn_v2_truth_boundary_oracle.py \
        --config experiments/ginn_v2/stage1_truth_boundary_oracle.yaml \
        --output-dir experiments/ginn_v2/results/<run>
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, resolve_relative_path  # noqa: E402
from ginn_v2.observability import (  # noqa: E402
    TruthBoundaryOracleConfig,
    run_truth_boundary_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_truth_boundary_oracle.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one calibration sample, two starts, and four CPU optimizer steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    payload = load_yaml_config(config_path)
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(
        section.get("truth_boundary_oracle"), dict
    ):
        raise ValueError(
            "config requires structured_ginn_v2.truth_boundary_oracle."
        )
    config = TruthBoundaryOracleConfig.from_mapping(
        section["truth_boundary_oracle"],
        root=REPO_ROOT,
    )
    if args.smoke:
        config = replace(
            config,
            device="cpu",
            maximum_parents=1,
            samples_per_parent=1,
            random_starts=2,
            optimization_steps=4,
            early_stopping_patience=4,
            progress_log_every_samples=1,
            progress_log_every_steps=2,
        )
    output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    run_truth_boundary_oracle(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
