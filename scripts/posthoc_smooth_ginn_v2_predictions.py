"""Post-process GINN-v2 patch predictions with lateral smoothing.

This script is intentionally small: it reads an existing ``ginn_v2 predict``
directory, smooths only ``pred_log_ai`` along the lateral axis, and writes a
prediction directory that remains consumable by ``ginn_v2.py report``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import repo_relative_path, resolve_relative_path, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, required=True, help="Odd lateral kernel width.")
    parser.add_argument("--sigma", type=float, required=True, help="Gaussian sigma in traces.")
    parser.add_argument(
        "--mode",
        choices=("mask_aware", "plain"),
        default="mask_aware",
        help="Whether to renormalize the lateral convolution by valid_mask_model.",
    )
    parser.add_argument(
        "--model-suffix",
        default=None,
        help="Suffix appended to prediction_index.model_id. Defaults to gaussian{width}.",
    )
    return parser.parse_args()


def _timestamped_output(prefix: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return REPO_ROOT / "scripts" / "output" / f"{prefix}_{timestamp}"


def _gaussian_kernel(width: int, sigma: float) -> np.ndarray:
    if width < 3 or width % 2 == 0:
        raise ValueError("width must be an odd integer >= 3.")
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("sigma must be a positive finite value.")
    offsets = np.arange(width, dtype=np.float64) - (width // 2)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    return kernel.astype(np.float64)


def _lateral_smooth(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    kernel: np.ndarray,
    mask_aware: bool,
) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("pred_log_ai must have shape [patch, lateral, twt].")
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask_model shape must match pred_log_ai.")
    radius = int(len(kernel) // 2)
    values32 = values.astype(np.float32, copy=False)
    weights_source = (
        valid_mask.astype(np.float32)
        if mask_aware
        else np.ones_like(values32, dtype=np.float32)
    )
    padded_values = np.pad(values32, ((0, 0), (radius, radius), (0, 0)), mode="constant")
    padded_weights = np.pad(weights_source, ((0, 0), (radius, radius), (0, 0)), mode="constant")
    numerator = np.zeros_like(values32, dtype=np.float32)
    denominator = np.zeros_like(values32, dtype=np.float32)
    for offset, weight in enumerate(kernel.astype(np.float32)):
        source = padded_values[:, offset : offset + values.shape[1], :]
        source_weight = padded_weights[:, offset : offset + values.shape[1], :] * weight
        numerator += source * source_weight
        denominator += source_weight
    smoothed = values32.copy()
    ok = denominator > 0.0
    smoothed[ok] = numerator[ok] / denominator[ok]
    return smoothed.astype(np.float32, copy=False)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    prediction_dir = resolve_relative_path(args.prediction_dir, root=REPO_ROOT)
    output_dir = _timestamped_output("ginn_v2_predict_posthoc_smooth", args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    kernel = _gaussian_kernel(int(args.width), float(args.sigma))
    arrays = np.load(prediction_dir / "predictions.npz", allow_pickle=True)
    pred = arrays["pred_log_ai"].astype(np.float32)
    mask = arrays["valid_mask_model"].astype(bool)
    smoothed = _lateral_smooth(
        pred,
        mask,
        kernel=kernel,
        mask_aware=str(args.mode) == "mask_aware",
    )

    array_payload = {key: arrays[key] for key in arrays.files}
    array_payload["pred_log_ai"] = smoothed
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(npz_path, **array_payload)

    index = pd.read_csv(prediction_dir / "prediction_index.csv")
    suffix = args.model_suffix or f"posthoc_gaussian{int(args.width)}"
    if "model_id" in index:
        index["model_id"] = index["model_id"].astype(str) + f"__{suffix}"
    else:
        index["model_id"] = suffix
    index_path = output_dir / "prediction_index.csv"
    index.to_csv(index_path, index=False)

    for optional_name in ["eval_patch_index.csv"]:
        source = prediction_dir / optional_name
        if source.is_file():
            shutil.copy2(source, output_dir / optional_name)

    source_manifest = _load_json(prediction_dir / "prediction_manifest.json")
    manifest = dict(source_manifest)
    manifest.update(
        {
            "schema_version": "ginn_v2_prediction_posthoc_smoothing_v1",
            "source_prediction_dir": repo_relative_path(prediction_dir, root=REPO_ROOT),
            "posthoc_smoothing": {
                "family": "lateral_gaussian",
                "mode": str(args.mode),
                "width_traces": int(args.width),
                "sigma_traces": float(args.sigma),
                "kernel": [float(value) for value in kernel],
                "smoothed_array": "pred_log_ai",
                "unchanged_arrays": [
                    key for key in arrays.files if key != "pred_log_ai"
                ],
            },
            "model_id": str(manifest.get("model_id", "")) + f"__{suffix}",
            "outputs": {
                "predictions": repo_relative_path(npz_path, root=REPO_ROOT),
                "prediction_index": repo_relative_path(index_path, root=REPO_ROOT),
            },
            "prediction_sha256": sha256_file(npz_path),
            "source_prediction_sha256": source_manifest.get("prediction_sha256", ""),
        }
    )
    write_json(output_dir / "prediction_manifest.json", manifest)
    write_json(output_dir / "prediction_summary.json", manifest)

    print("=== GINN-v2 posthoc lateral smoothing ===")
    print(f"Output: {output_dir}")
    print(f"Source: {prediction_dir}")
    print(f"Kernel: {kernel.tolist()}")


if __name__ == "__main__":
    main()
