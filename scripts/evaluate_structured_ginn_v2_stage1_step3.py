"""Evaluate a frozen Stage 1 Step 3 checkpoint without retraining.

Usage::

    python scripts/evaluate_structured_ginn_v2_stage1_step3.py \
        --config experiments/ginn_v2/stage1_step3.yaml \
        --checkpoint experiments/ginn_v2/results/<run>/stage1_step3_checkpoint.pt \
        --split-manifest experiments/ginn_v2/results/<run>/parent_split_manifest.json \
        --output-dir experiments/ginn_v2/results/<run>_eval_fixed \
        --input-mode full
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
from ginn_v2.lateral_training import (  # noqa: E402
    Stage1Step3Config,
    evaluate_stage1_step3_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_step3.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=("full", "no-seismic"),
        required=True,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate one parent per split on CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    payload = load_yaml_config(config_path)
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(
        section.get("stage1_step3"), dict
    ):
        raise ValueError("config requires structured_ginn_v2.stage1_step3.")
    config = Stage1Step3Config.from_mapping(
        section["stage1_step3"],
        root=REPO_ROOT,
    )
    if args.smoke:
        base = replace(
            config.training.base,
            device="cpu",
            maximum_validation_parents=1,
            maximum_samples_per_parent=1,
            maximum_validation_batches=1,
            progress_log_every_parents=1,
        )
        config = replace(
            config,
            training=replace(
                config.training,
                base=base,
                final_exact_validation_batches=1,
                final_dirty_validation_batches=1,
            ),
        )
    evaluate_stage1_step3_checkpoint(
        config,
        checkpoint_path=resolve_relative_path(args.checkpoint, root=REPO_ROOT),
        split_manifest_path=resolve_relative_path(
            args.split_manifest,
            root=REPO_ROOT,
        ),
        output_dir=resolve_relative_path(args.output_dir, root=REPO_ROOT),
        input_mode=args.input_mode,
    )


if __name__ == "__main__":
    main()
