"""Execution policy for non-differentiable depth-domain forward modeling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cup.physics.numpy_backend import forward_depth as numpy_forward_depth


class DepthForwardExecutor:
    """Resolve NumPy/CUDA once and execute equal-axis trace batches."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        requested = str(config.get("backend") or "auto").strip().casefold()
        dtype = str(config.get("dtype") or "float64").strip().casefold()
        batch_size = int(config.get("batch_size", 64))
        if requested not in {"auto", "numpy", "torch_cuda"}:
            raise ValueError("Depth forward backend must be auto, numpy, or torch_cuda.")
        if dtype != "float64":
            raise ValueError("Depth forward dtype is fixed to float64.")
        if batch_size <= 0:
            raise ValueError("Depth forward batch_size must be positive.")
        self.requested = requested
        self.dtype = dtype
        self.batch_size = batch_size
        self._torch = None
        self._torch_forward = None
        if requested != "numpy":
            try:
                import torch
                from cup.physics.torch_backend import forward_depth as torch_forward_depth
            except Exception as exc:
                if requested == "torch_cuda":
                    raise RuntimeError(
                        "torch_cuda backend requires PyTorch with CUDA support."
                    ) from exc
            else:
                if bool(torch.cuda.is_available()):
                    self._torch = torch
                    self._torch_forward = torch_forward_depth
        if requested == "torch_cuda" and self._torch is None:
            raise RuntimeError("Requested torch_cuda backend, but CUDA is unavailable.")
        self.resolved = "torch_cuda" if self._torch is not None else "numpy"

    @property
    def operator_id(self) -> str:
        return (
            "cup.physics.torch_backend.forward_depth"
            if self.resolved == "torch_cuda"
            else "cup.physics.numpy_backend.forward_depth"
        )

    @property
    def manifest_fields(self) -> dict[str, str | int]:
        return {
            "requested_backend": self.requested,
            "resolved_backend": self.resolved,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "operator": self.operator_id,
        }

    def __call__(
        self,
        log_ai: np.ndarray,
        velocity_mps: np.ndarray,
        depth_m: np.ndarray,
        wavelet_time_s: np.ndarray,
        wavelet_amp: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(log_ai, dtype=np.float64)
        velocity = np.asarray(velocity_mps, dtype=np.float64)
        if values.shape != velocity.shape:
            raise ValueError("Depth forward logAI/velocity shape mismatch.")
        n_samples = values.shape[-1]
        original_shape = values.shape
        flat_values = values.reshape((-1, n_samples))
        flat_velocity = velocity.reshape((-1, n_samples))
        chunks: list[np.ndarray] = []
        for start in range(0, flat_values.shape[0], self.batch_size):
            stop = min(start + self.batch_size, flat_values.shape[0])
            if self.resolved == "numpy":
                result = numpy_forward_depth(
                    flat_values[start:stop],
                    flat_velocity[start:stop],
                    depth_m,
                    wavelet_time_s,
                    wavelet_amp,
                )
                chunks.append(np.asarray(result, dtype=np.float64))
                continue
            torch = self._torch
            assert torch is not None and self._torch_forward is not None
            device = torch.device("cuda")
            with torch.inference_mode():
                result = self._torch_forward(
                    torch.as_tensor(flat_values[start:stop], dtype=torch.float64, device=device),
                    torch.as_tensor(flat_velocity[start:stop], dtype=torch.float64, device=device),
                    torch.as_tensor(depth_m, dtype=torch.float64, device=device),
                    torch.as_tensor(wavelet_time_s, dtype=torch.float64, device=device),
                    torch.as_tensor(wavelet_amp, dtype=torch.float64, device=device),
                )
            chunks.append(result.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(chunks, axis=0).reshape(original_shape)


__all__ = ["DepthForwardExecutor"]
