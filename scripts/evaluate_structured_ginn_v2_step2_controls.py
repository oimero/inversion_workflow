"""Evaluate paired Stage 1 Step 2 structured evidence controls.

Usage::

    python scripts/evaluate_structured_ginn_v2_step2_controls.py \
        --config experiments/ginn_v2/stage1_step2.yaml \
        --full-run experiments/ginn_v2/results/<full> \
        --no-seismic-run experiments/ginn_v2/results/<no-seismic> \
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
from ginn_v2.data import ParentSplitManifest, TeacherForcingDataModule  # noqa: E402
from ginn_v2.runtime import resolve_device  # noqa: E402
from ginn_v2.structure_controls import run_stage1_step2_controls  # noqa: E402
from ginn_v2.structure_training import Stage1Step2Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_step2.yaml"),
    )
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--no-seismic-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-zone", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--maximum-parents", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate one parent with two traces per zone on CPU.",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return resolve_relative_path(path, root=REPO_ROOT)


def main() -> None:
    args = parse_args()
    payload = load_yaml_config(_resolve(args.config))
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(
        section.get("stage1_step2"), dict
    ):
        raise ValueError("config requires structured_ginn_v2.stage1_step2.")
    config = Stage1Step2Config.from_mapping(
        section["stage1_step2"],
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
        condition_limit=config.training.base.condition_limit,
    )
    device, _ = resolve_device(
        "cpu" if args.smoke else config.training.base.device
    )
    run_stage1_step2_controls(
        data=data,
        full_run_dir=full_run,
        no_seismic_run_dir=no_seismic_run,
        output_dir=_resolve(args.output_dir),
        device=device,
        seed=config.training.base.seed + 700_000,
        samples_per_zone_per_parent=(
            2 if args.smoke else int(args.samples_per_zone)
        ),
        bootstrap_samples=(
            200 if args.smoke else int(args.bootstrap_samples)
        ),
        maximum_parents=1 if args.smoke else args.maximum_parents,
    )


if __name__ == "__main__":
    main()
