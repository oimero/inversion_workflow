"""Small runtime helpers for the independent Structured GINN V2 package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from cup.utils.logging import configure_run_logger


def configure_training_logger(output_dir: Path) -> logging.Logger:
    """Create the logger used by a Structured GINN V2 run."""
    return configure_run_logger(
        Path(output_dir),
        logger_name="ginn_v2",
        file_name="training.log",
    )


def resolve_device(device_name: str) -> tuple[torch.device, dict[str, Any]]:
    """Resolve one explicit PyTorch device for the structured runtime."""
    requested = str(device_name or "auto").strip()
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


__all__ = ["configure_training_logger", "resolve_device"]
