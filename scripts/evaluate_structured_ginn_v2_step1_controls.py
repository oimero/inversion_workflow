"""Evaluate paired full/no-seismic and state-duration Stage 1 controls.

Usage::

    python scripts/evaluate_structured_ginn_v2_step1_controls.py \
        --config experiments/ginn_v2/stage1_step1.yaml \
        --full-run experiments/ginn_v2/results/<full> \
        --no-seismic-run experiments/ginn_v2/results/<no-seismic> \
        --mean-pooling-run experiments/ginn_v2/results/20260725_v2 \
        --output-dir experiments/ginn_v2/results/<controls>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, resolve_relative_path  # noqa: E402
from ginn_v2.controls import run_stage1_step1_controls  # noqa: E402
from ginn_v2.data import ParentSplitManifest, TeacherForcingDataModule  # noqa: E402
from ginn_v2.runtime import resolve_device  # noqa: E402
from ginn_v2.training import Stage1Step1Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_step1.yaml"),
    )
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--no-seismic-run", type=Path, required=True)
    parser.add_argument(
        "--mean-pooling-run",
        type=Path,
        default=None,
        help="Optional frozen mean-pooling full run used as the architecture reference.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fit/evaluate one parent with at most four trace-zone samples.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_relative_path(path, root=REPO_ROOT)


def main() -> None:
    args = parse_args()
    payload = load_yaml_config(_resolve(args.config))
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(
        section.get("stage1_step1"), dict
    ):
        raise ValueError("config requires structured_ginn_v2.stage1_step1.")
    config = Stage1Step1Config.from_mapping(
        section["stage1_step1"],
        root=REPO_ROOT,
    )
    full_run = _resolve(args.full_run)
    no_seismic_run = _resolve(args.no_seismic_run)
    manifest = ParentSplitManifest.from_dict(
        json.loads(
            (full_run / "parent_split_manifest.json").read_text(encoding="utf-8")
        )
    )
    data = TeacherForcingDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        condition_limit=config.training.condition_limit,
    )
    device, _ = resolve_device("cpu" if args.smoke else config.training.device)
    run_stage1_step1_controls(
        data=data,
        full_run_dir=full_run,
        no_seismic_run_dir=no_seismic_run,
        mean_pooling_run_dir=(
            _resolve(args.mean_pooling_run)
            if args.mean_pooling_run is not None
            else None
        ),
        output_dir=_resolve(args.output_dir),
        device=device,
        training_samples_per_zone=config.training.samples_per_zone_per_parent or 8,
        batch_size=2 if args.smoke else config.training.batch_size,
        seed=config.training.seed,
        bootstrap_samples=200 if args.smoke else 2000,
        maximum_training_parents=1 if args.smoke else None,
        maximum_tuning_parents=1 if args.smoke else None,
        maximum_samples_per_parent=4 if args.smoke else None,
    )


if __name__ == "__main__":
    main()
