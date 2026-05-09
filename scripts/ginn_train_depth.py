"""Depth-domain GINN training entry point."""

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

from ginn_depth.config import DepthGINNConfig
from ginn_depth.trainer import Trainer

# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train depth-domain GINN from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments" / "ginn_depth" / "train.yaml",
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
    cfg = DepthGINNConfig.from_yaml(args.config, base_dir=REPO_ROOT)
    logging.info("Loaded config from %s", args.config.resolve())

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
