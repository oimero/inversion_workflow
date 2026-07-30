"""Audit Stage 1 boundary ranking at matched truth segment counts."""

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
from ginn_v2.boundary_localization import (  # noqa: E402
    BoundaryLocalizationAuditOptions,
    run_boundary_localization_audit,
)
from ginn_v2.lateral_training import Stage1Step3Config  # noqa: E402


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
        "--condition",
        choices=("clean", "dirty"),
        default="clean",
    )
    parser.add_argument("--maximum-parents", type=int, default=16)
    parser.add_argument(
        "--boundary-scales",
        type=float,
        nargs="+",
        default=(-1.0, 0.0, 0.5, 1.0, 2.0),
    )
    parser.add_argument(
        "--minimum-f1-improvement",
        type=float,
        default=0.02,
    )
    parser.add_argument("--progress-every-parents", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tuning parent with scales -1, 0, and 1 on CPU.",
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
        config = replace(
            config,
            training=replace(
                config.training,
                base=replace(
                    config.training.base,
                    device="cpu",
                    maximum_samples_per_parent=1,
                    progress_log_every_parents=1,
                ),
            ),
        )
    options = BoundaryLocalizationAuditOptions(
        maximum_parents=1 if args.smoke else int(args.maximum_parents),
        condition=str(args.condition),
        boundary_scales=(
            (-1.0, 0.0, 1.0)
            if args.smoke
            else tuple(float(value) for value in args.boundary_scales)
        ),
        minimum_f1_improvement=float(args.minimum_f1_improvement),
        progress_every_parents=(
            1 if args.smoke else int(args.progress_every_parents)
        ),
        resume=bool(args.resume),
    )
    run_boundary_localization_audit(
        config,
        checkpoint_path=resolve_relative_path(
            args.checkpoint,
            root=REPO_ROOT,
        ),
        split_manifest_path=resolve_relative_path(
            args.split_manifest,
            root=REPO_ROOT,
        ),
        output_dir=resolve_relative_path(args.output_dir, root=REPO_ROOT),
        input_mode=str(args.input_mode),
        options=options,
    )


if __name__ == "__main__":
    main()
