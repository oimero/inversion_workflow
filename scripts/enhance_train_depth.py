"""Train stage-2 depth resolution enhancement from well-guided synthetic samples."""

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

# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/enhance_depth/train.yaml"))
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    from enhance.config import EnhancementConfig
    from enhance.trainer import EnhancementTrainer
    from ginn_depth.enhance import build_depth_enhancement_bundle

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = EnhancementConfig.from_yaml(config_path, base_dir=REPO_ROOT)
    bundle = build_depth_enhancement_bundle(cfg)
    trainer = EnhancementTrainer(
        cfg,
        bundle.synthetic_dataset,
        normalization={
            "seis_rms": bundle.dataset_bundle.train_dataset.seis_rms,
            "base_ai_scale": bundle.dataset_bundle.train_dataset.lfm_scale,
            "dynamic_gain_median": bundle.dataset_bundle.train_dataset.dynamic_gain_median,
        },
        metadata=bundle.metadata,
    )
    trainer.train()


if __name__ == "__main__":
    main()
