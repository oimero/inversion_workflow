"""Device, loss, wavelet, and differentiable forward primitives for ablation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from cup.physics.torch_backend import forward_depth, forward_time, velocity_from_ai
from cup.synthetic.schemas import require_science_contract
from cup.utils.logging import configure_run_logger


def configure_training_logger(output_dir: Path) -> logging.Logger:
    """Create the logger shared by the composable runner and CLI."""
    return configure_run_logger(
        output_dir,
        logger_name="ablation",
        file_name="training.log",
    )


def resolve_device(device_name: str) -> tuple[torch.device, dict[str, Any]]:
    requested = str(device_name or "auto")
    cuda_available = bool(torch.cuda.is_available())
    resolved = requested if requested != "auto" else ("cuda" if cuda_available else "cpu")
    if requested.startswith("cuda") and not cuda_available:
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false.")
    device = torch.device(resolved)
    return device, {
        "requested_device": requested,
        "resolved_device": str(device),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch_version": str(torch.__version__),
    }


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if torch.count_nonzero(valid) == 0:
        raise ValueError("Batch has no valid samples.")
    residual = prediction[valid] - target[valid]
    return torch.mean(residual.square())


def _required_file(value: object, *, label: str, base_dir: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Benchmark manifest lacks {label}.")
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_benchmark_wavelet(
    benchmark_dir: Path,
    *,
    device: torch.device,
    artifact_root: Path,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
    manifest_path = Path(benchmark_dir) / "benchmark_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    require_science_contract(manifest, label="ablation Synthoseis benchmark")
    domain = str(manifest.get("sample_domain") or "")
    if domain == "depth":
        with _required_file(
            manifest.get("forward_model_inputs_path"), label="forward_model_inputs_path",
            base_dir=artifact_root,
        ).open("r", encoding="utf-8") as handle:
            forward_inputs = json.load(handle)
        if forward_inputs.get("sample_domain") != "depth" or forward_inputs.get("depth_basis") != "tvdss":
            raise ValueError("Depth forward inputs must declare depth/TVDSS.")
        raw_relation = dict(forward_inputs.get("ai_velocity_relation") or {})
        if raw_relation.get("ai_unit") != "m/s*g/cm3" or raw_relation.get("vp_unit") != "m/s":
            raise ValueError("Depth AI–Vp relation units are invalid.")
        relation: dict[str, float] | None = {
            "a": float(raw_relation["a"]), "b": float(raw_relation["b"])
        }
        csv_path = _required_file(
            dict(forward_inputs.get("wavelet") or {}).get("path"),
            label="forward_model_inputs.wavelet.path",
            base_dir=artifact_root,
        )
    elif domain == "time":
        relation = None
        wavelet_dir = dict(manifest.get("source_runs") or {}).get("wavelet_generation_dir")
        if not wavelet_dir:
            raise ValueError("Time benchmark lacks source_runs.wavelet_generation_dir.")
        csv_path = Path(str(wavelet_dir)) / "selected_wavelet.csv"
        if not csv_path.is_absolute():
            csv_path = artifact_root / csv_path
    else:
        raise ValueError(f"Unsupported benchmark sample_domain: {domain!r}")
    frame = pd.read_csv(csv_path)
    missing = sorted({"time_s", "amplitude"} - set(frame.columns))
    if missing:
        raise ValueError(f"Wavelet CSV lacks columns {missing}: {csv_path}")
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    amplitude = frame["amplitude"].to_numpy(dtype=np.float32)
    if amplitude.size < 3 or amplitude.size % 2 == 0:
        raise ValueError("Wavelet must have odd length >= 3.")
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(amplitude)):
        raise ValueError("Wavelet contains non-finite values.")
    return (
        torch.as_tensor(time_s, dtype=torch.float32, device=device),
        torch.as_tensor(amplitude, dtype=torch.float32, device=device),
        relation,
    )


def forward_physics_batch(
    log_ai: torch.Tensor,
    *,
    sample_axes: torch.Tensor,
    sample_domain: str,
    wavelet_time_s: torch.Tensor,
    wavelet_amp: torch.Tensor,
    ai_velocity_relation: Mapping[str, float] | None,
) -> torch.Tensor:
    if log_ai.ndim != 3:
        raise ValueError("Physics forward expects [batch, lateral, sample] logAI.")
    if sample_domain == "time":
        return forward_time(log_ai, wavelet_time_s, wavelet_amp)
    if sample_domain != "depth":
        raise ValueError(f"Unsupported physics sample domain: {sample_domain!r}")
    if sample_axes.shape != (log_ai.shape[0], log_ai.shape[-1]):
        raise ValueError("Depth physics axes must have shape [batch, sample].")
    if ai_velocity_relation is None:
        raise ValueError("Depth physics requires the frozen AI–Vp relation.")
    velocity_mps = velocity_from_ai(
        torch.exp(log_ai),
        a=float(ai_velocity_relation["a"]),
        b=float(ai_velocity_relation["b"]),
    )
    return torch.stack(
        [
            forward_depth(
                log_ai[index], velocity_mps[index], sample_axes[index],
                wavelet_time_s, wavelet_amp,
            )
            for index in range(log_ai.shape[0])
        ],
        dim=0,
    )


__all__ = [
    "configure_training_logger",
    "forward_physics_batch",
    "load_benchmark_wavelet",
    "masked_mse",
    "resolve_device",
]
