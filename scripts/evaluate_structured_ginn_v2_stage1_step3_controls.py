"""Evaluate Stage 1 Step 3 lateral and evidence controls."""

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

import torch  # noqa: E402

from cup.utils.io import load_yaml_config, resolve_relative_path  # noqa: E402
from ginn_v2.data import ParentSplitManifest  # noqa: E402
from ginn_v2.lateral import LateralPatchDataModule  # noqa: E402
from ginn_v2.lateral_controls import run_stage1_step3_controls  # noqa: E402
from ginn_v2.runtime import resolve_device  # noqa: E402
from ginn_v2.lateral_training import Stage1Step3Config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/ginn_v2/stage1_step3.yaml"),
    )
    parser.add_argument("--full-run-dir", type=Path, required=True)
    parser.add_argument("--no-seismic-run-dir", type=Path, required=True)
    parser.add_argument("--single-trace-full-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-parents", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_relative_path(args.config, root=REPO_ROOT)
    payload = load_yaml_config(config_path)
    section = payload.get("structured_ginn_v2")
    if not isinstance(section, dict) or not isinstance(section.get("stage1_step3"), dict):
        raise ValueError("config requires structured_ginn_v2.stage1_step3.")
    config = Stage1Step3Config.from_mapping(section["stage1_step3"], root=REPO_ROOT)
    full_run = resolve_relative_path(args.full_run_dir, root=REPO_ROOT)
    manifest_path = full_run / "parent_split_manifest.json"
    manifest = ParentSplitManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    device, _ = resolve_device(config.training.base.device)
    data = LateralPatchDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        patch_width=config.model.patch_width,
        augmentation_profile=config.augmentation,
        dirty_probability=config.dirty_probability,
        condition_limit=config.training.base.condition_limit,
    )
    maximum_parents = 1 if args.smoke else args.maximum_parents
    bootstrap_samples = 50 if args.smoke else args.bootstrap_samples
    output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    run_stage1_step3_controls(
        data=data,
        full_run_dir=full_run,
        no_seismic_run_dir=resolve_relative_path(args.no_seismic_run_dir, root=REPO_ROOT),
        single_trace_full_run_dir=resolve_relative_path(
            args.single_trace_full_run_dir,
            root=REPO_ROOT,
        ),
        output_dir=output_dir,
        device=device,
        seed=args.seed,
        bootstrap_samples=bootstrap_samples,
        maximum_parents=maximum_parents,
    )


if __name__ == "__main__":
    main()
