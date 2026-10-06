"""Train the independent one-dimensional PIAI inversion model.

Usage::

    python scripts/piai_train.py --config experiments/ginn_v3/ginn_v3.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ginn_v3.workflow import train_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/ginn_v3/ginn_v3.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lfm-run-dir", type=Path, default=None)
    parser.add_argument("--variant-id", type=str, default=None)
    parser.add_argument("--well-control-run-dir", type=Path, default=None)
    parser.add_argument("--forward-model-inputs-run-dir", type=Path, default=None)
    parser.add_argument("--wavelet-generation-run-dir", type=Path, default=None)
    parser.add_argument("--trusted-well-name", action="append", dest="trusted_well_names", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_body(
        args.config,
        output_dir=args.output_dir,
        updates=args.updates,
        device=args.device,
        lfm_run_dir=args.lfm_run_dir,
        variant_id=args.variant_id,
        well_control_run_dir=args.well_control_run_dir,
        forward_model_inputs_run_dir=args.forward_model_inputs_run_dir,
        wavelet_generation_run_dir=args.wavelet_generation_run_dir,
        trusted_well_names=args.trusted_well_names,
    )
    print("=== PIAI v3 training ===")
    print(f"Output: {result.output_dir}")
    print(f"Selected checkpoint: {result.selected_checkpoint}")
    print(f"Last checkpoint: {result.last_checkpoint}")
    print(f"Updates completed: {result.updates_completed}")


if __name__ == "__main__":
    main()
