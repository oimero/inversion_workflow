"""Train the Stage 1 step-1 teacher-forcing Structured GINN V2 model.

Usage::

    python scripts/structured_ginn_v2_stage1_step1.py \
        --config experiments/ginn_v2/stage1_step1.yaml \
        --output-dir experiments/ginn_v2/results/<run>

    python scripts/structured_ginn_v2_stage1_step1.py \
        --config experiments/ginn_v2/stage1_step1.yaml \
        --output-dir experiments/ginn_v2/results/smoke \
        --smoke
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
from ginn_v2.training import Stage1Step1Config, run_stage1_step1  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_step1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two tiny CPU epochs over one parent and four samples.",
    )
    return parser.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    return resolve_relative_path(args.output_dir, root=REPO_ROOT)


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    payload = load_yaml_config(config_path)
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(
        section.get("stage1_step1"), dict
    ):
        raise ValueError("config requires structured_ginn_v2.stage1_step1.")
    config = Stage1Step1Config.from_mapping(
        section["stage1_step1"],
        root=REPO_ROOT,
    )
    if args.smoke:
        config = replace(
            config,
            training=replace(
                config.training,
                epochs=2,
                minimum_epochs=2,
                early_stopping_patience=1,
                batch_size=2,
                boundary_jitter_samples=1,
                progress_log_every_parents=1,
                device="cpu",
                maximum_training_parents=1,
                maximum_validation_parents=1,
                maximum_samples_per_parent=4,
                maximum_training_batches=2,
                maximum_validation_batches=2,
            ),
        )
    report = run_stage1_step1(config, output_dir=_resolve_output_dir(args))
    if args.smoke:
        first = float(report["history"][0]["training"]["loss"])
        last = float(report["history"][-1]["training"]["loss"])
        if not last < first:
            raise RuntimeError(
                f"teacher-forcing smoke did not reduce training loss: {first} -> {last}"
            )


if __name__ == "__main__":
    main()
