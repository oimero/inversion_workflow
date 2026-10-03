"""Run GINN v2 self-supervised pretraining and/or well-supervised finetuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ginn_v2 import train_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ginn_v2/ginn_v2.yaml"))
    parser.add_argument("--stage", choices=("all", "pretrain", "finetune"), default="all")
    parser.add_argument("--pretrain-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--lfm-run-dir", type=Path, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--well-control-run-dir", type=Path, default=None)
    parser.add_argument("--forward-model-inputs-run-dir", type=Path, default=None,
                        help="Depth forward-input run directory containing forward_model_inputs.json.")
    parser.add_argument("--wavelet-generation-run-dir", type=Path, default=None,
                        help="Time-domain Step-5 run directory containing selected_wavelet.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_body(
        args.config,
        stage=args.stage,
        pretrain_checkpoint=args.pretrain_checkpoint,
        output_dir=args.output_dir,
        lfm_run_dir=args.lfm_run_dir,
        variant_id=args.variant_id,
        well_control_run_dir=args.well_control_run_dir,
        forward_model_inputs_run_dir=args.forward_model_inputs_run_dir,
        wavelet_generation_run_dir=args.wavelet_generation_run_dir,
    )
    print("=== GINN v2 body training ===")
    print(f"Output: {result.output_dir}")
    print(f"Pretrain checkpoint: {result.pretrain_checkpoint}")
    if result.selected_checkpoint is not None:
        print(f"Selected checkpoint: {result.selected_checkpoint}")
    if result.warnings:
        print(f"Warnings: {', '.join(result.warnings)}")


if __name__ == "__main__":
    main()
