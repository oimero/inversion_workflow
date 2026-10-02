"""Frozen forward-input loading and non-differentiable depth execution policy."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from cup.physics.calibration import AIVelocityRelation
from cup.physics.numpy_backend import forward_depth as numpy_forward_depth
from cup.seismic.wavelet import load_wavelet_csv, validate_wavelet_normalization
from cup.utils.io import resolve_relative_path


def load_forward_inputs(
    run_dir: Path,
    *,
    repo_root: Path,
    domain: str,
    depth_basis: str | None,
) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation | None, dict[str, Any]]:
    """Read ``forward_model_inputs.json`` from a forward-input run directory."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    path = run_dir / "forward_model_inputs.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "forward_model_inputs_v3":
        raise ValueError("Body inversion requires forward_model_inputs_v3.")
    if payload.get("sample_domain") != domain or payload.get("depth_basis") != depth_basis:
        raise ValueError("Frozen forward inputs do not match the seismic SampleAxis domain.")
    wavelet_info = payload.get("wavelet")
    if not isinstance(wavelet_info, Mapping):
        raise ValueError("forward_model_inputs.wavelet must be a mapping.")
    wavelet_path = resolve_relative_path(
        str(wavelet_info.get("path") or ""),
        root=Path(repo_root),
    )
    time_s, amplitude = load_wavelet_csv(wavelet_path)
    amplitude, qc = validate_wavelet_normalization(
        time_s,
        amplitude,
        allow_small_renormalization=False,
    )
    if qc.status != "ok":
        raise ValueError(f"Frozen wavelet failed normalization QC: {qc.reasons}")
    relation_info = payload.get("ai_velocity_relation")
    if domain == "depth" and not isinstance(relation_info, Mapping):
        raise ValueError("Depth forward inputs must contain ai_velocity_relation.")
    relation = AIVelocityRelation.from_mapping(relation_info) if relation_info is not None else None
    return time_s, amplitude, relation, payload


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


__all__ = ["DepthForwardExecutor", "load_forward_inputs"]
