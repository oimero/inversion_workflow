"""Train a GINN model from a YAML experiment config.

This is the eighth step of the time-domain workflow.  It reads the seismic
volume, a precomputed AI low-frequency model (step 7), an optional well
anchor bundle (step 6), and a global wavelet (step 5), then trains a
physics-informed neural network to predict AI residuals.

Usage::

    python scripts/ginn_train.py
    python scripts/ginn_train.py --config experiments/ginn/train.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ginn.config import GINNConfig
from ginn.trainer import Trainer

# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments" / "ginn" / "train.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = GINNConfig.from_yaml(args.config, base_dir=REPO_ROOT)
    logging.info("Loaded config from %s", args.config.resolve())

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
