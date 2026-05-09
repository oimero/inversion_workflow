"""Predict a depth enhanced-AI volume from a stage-2 enhancement checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=None, help="Override config path; defaults to checkpoint config."
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    from enhance.config import EnhancementConfig
    from enhance.loss import compose_enhanced_ai
    from enhance.model import DilatedResNet1D
    from ginn_depth.enhance import build_depth_enhancement_data_bundle

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.config is None:
        cfg = EnhancementConfig.from_dict(payload["config"], base_dir=REPO_ROOT)
    else:
        config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        cfg = EnhancementConfig.from_yaml(config_path, base_dir=REPO_ROOT)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    bundle = build_depth_enhancement_data_bundle(cfg)
    dataset = bundle.dataset_bundle.inference_dataset
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    model = DilatedResNet1D(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.out_channels,
        dilations=cfg.dilations,
        kernel_size=cfg.kernel_size,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    enhanced_flat = dataset.ai_lfm_flat.astype(np.float32, copy=True)
    delta_flat = np.zeros_like(enhanced_flat, dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            base_ai = batch["lfm_raw"].to(device)
            taper = batch["taper_weight"].to(device)
            delta = model(x)
            if cfg.zero_delta_outside_mask:
                delta = delta * taper.to(dtype=delta.dtype)
            enhanced = compose_enhanced_ai(base_ai, delta, ai_min=cfg.ai_min, ai_max=cfg.ai_max)
            n_batch = enhanced.shape[0]
            indices = dataset.valid_indices[offset : offset + n_batch]
            enhanced_flat[indices] = enhanced.squeeze(1).cpu().numpy()
            delta_flat[indices] = delta.squeeze(1).cpu().numpy()
            offset += n_batch

    n_il = int(bundle.dataset_bundle.geometry["n_il"])
    n_xl = int(bundle.dataset_bundle.geometry["n_xl"])
    n_sample = int(bundle.dataset_bundle.geometry["n_sample"])
    enhanced_volume = enhanced_flat.reshape(n_il, n_xl, n_sample)
    delta_volume = delta_flat.reshape(n_il, n_xl, n_sample)

    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geometry = bundle.dataset_bundle.geometry
    ilines = float(geometry["inline_min"]) + np.arange(n_il, dtype=np.float32) * float(geometry["inline_step"])
    xlines = float(geometry["xline_min"]) + np.arange(n_xl, dtype=np.float32) * float(geometry["xline_step"])
    np.savez_compressed(
        output_path,
        volume=enhanced_volume.astype(np.float32),
        delta_log_ai=delta_volume.astype(np.float32),
        ilines=ilines,
        xlines=xlines,
        samples=bundle.dataset_bundle.depth_axis_m.astype(np.float32),
        geometry_json=np.asarray(json.dumps(geometry, ensure_ascii=False)),
        metadata_json=np.asarray(
            json.dumps({"checkpoint": str(checkpoint_path), "config": cfg.to_json_dict()}, ensure_ascii=False)
        ),
    )
    logging.info("Enhanced depth AI saved: %s", output_path)


if __name__ == "__main__":
    main()
