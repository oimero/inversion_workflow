"""GINN 训练入口脚本。

使用方法
--------
在仓库根目录下运行：

    python scripts/ginn_train.py
    python scripts/ginn_train.py --config experiments/ginn/train.yaml

需要将 ``src`` 加入 PYTHONPATH，或通过 ``train_network.ps1`` 类似脚本启动。
"""

import argparse
import logging
import sys
from pathlib import Path

# ── 确保 src/ 在搜索路径中 ──
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ginn.config import GINNConfig
from ginn.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GINN from a YAML experiment config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_repo_root / "experiments" / "ginn" / "train.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    # ── 日志 ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = GINNConfig.from_yaml(args.config, base_dir=_repo_root)
    logging.info("Loaded config from %s", args.config.resolve())

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
