"""Train Stage 1 Step 2: single-trace evidence and exact HSMM.

Usage::

    python scripts/structured_ginn_v2_stage1_step2.py \
        --config experiments/ginn_v2/stage1_step2.yaml \
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
from ginn_v2.structure_training import (  # noqa: E402
    Stage1Step2Config,
    run_stage1_step2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_step2.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=("full", "no-seismic"),
        default="full",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny CPU epoch over one training and validation parent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    payload = load_yaml_config(config_path)
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(
        section.get("stage1_step2"), dict
    ):
        raise ValueError("config requires structured_ginn_v2.stage1_step2.")
    config = Stage1Step2Config.from_mapping(
        section["stage1_step2"],
        root=REPO_ROOT,
    )
    if args.smoke:
        base = replace(
            config.training.base,
            epochs=1,
            minimum_epochs=1,
            early_stopping_patience=1,
            batch_size=2,
            boundary_jitter_samples=1,
            progress_log_every_parents=1,
            device="cpu",
            maximum_training_parents=1,
            maximum_validation_parents=1,
            maximum_samples_per_parent=2,
            maximum_training_batches=1,
            maximum_validation_batches=1,
        )
        config = replace(
            config,
            training=replace(
                config.training,
                base=base,
                exact_validation_batches_per_epoch=1,
                final_exact_validation_batches=1,
            ),
        )
    output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    run_stage1_step2(
        config,
        output_dir=output_dir,
        input_mode=args.input_mode,
    )


if __name__ == "__main__":
    main()
