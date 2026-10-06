"""Streaming trace and volume inference for the independent PIAI workflow.

Inference keeps the network output at the physical log-impedance scale.  The
only construction applied here is the model contract itself:
``log_ai = lfm_log_ai + raw_correction``.  Unsupported traces remain NaN in
the output memmap and are represented by a separate boolean mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ginn_v3.model import PIAINetwork
from ginn_v3.types import TraceKey, TracePrediction


def _chunks(values: Sequence[TraceKey], size: int) -> Iterable[tuple[TraceKey, ...]]:
    if size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


@dataclass(frozen=True)
class VolumePrediction:
    """Paths and aggregate information produced by streaming inference."""

    log_ai_path: Path
    valid_mask_path: Path
    shape: tuple[int, int, int]
    predicted_trace_count: int
    wavelet_mean_normalized: np.ndarray


class Inverter:
    """Run one-dimensional PIAI predictions over explicit trace keys."""

    def __init__(
        self,
        model: PIAINetwork,
        reader: object,
        device: str | torch.device,
        batch_size: int,
        min_support_samples: int = 2,
    ) -> None:
        if not isinstance(model, PIAINetwork):
            raise TypeError("model must be a PIAINetwork instance.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(min_support_samples) < 2:
            raise ValueError("min_support_samples must be at least two.")
        self.model = model
        self.reader = reader
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.min_support_samples = int(min_support_samples)
        self.model.to(self.device)
        self.model.eval()

    def _predict_chunk(
        self,
        keys: tuple[TraceKey, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not keys:
            raise ValueError("Cannot predict an empty trace chunk.")
        batch = self.reader.batch(keys, self.device)
        with torch.no_grad():
            prediction = self.model(batch.features, batch.initial_log_ai)
        log_ai = prediction.log_ai.detach().cpu().numpy().astype(np.float32, copy=True)
        valid = batch.lfm_mask.detach().cpu().numpy().astype(bool, copy=True)
        finite = np.isfinite(log_ai)
        if np.any(valid & ~finite):
            raise FloatingPointError("PIAI produced a non-finite value on finite LFM support.")
        log_ai[~valid] = np.nan
        wavelets = prediction.wavelets.detach().cpu().numpy().astype(np.float64, copy=True)
        if wavelets.ndim != 2 or wavelets.shape[0] != len(keys):
            raise ValueError("PIAI wavelet output must have shape (batch, wavelet_samples).")
        if not np.all(np.isfinite(wavelets)):
            raise FloatingPointError("PIAI wavelet output contains non-finite values.")
        supported_trace = np.any(valid, axis=1)
        return log_ai, valid, wavelets, supported_trace

    def predict_traces(self, keys: Sequence[TraceKey]) -> TracePrediction:
        """Predict explicit traces and return the unfiltered physical log-AI."""

        selected = tuple(keys)
        if not selected:
            raise ValueError("keys must contain at least one TraceKey.")
        log_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        wavelet_parts: list[np.ndarray] = []
        for chunk in _chunks(selected, self.batch_size):
            log_ai, valid, wavelets, supported = self._predict_chunk(chunk)
            log_parts.append(log_ai)
            mask_parts.append(valid)
            if np.any(supported):
                wavelet_parts.append(wavelets[supported])
        log_ai = np.concatenate(log_parts, axis=0)
        valid = np.concatenate(mask_parts, axis=0)
        support_counts = np.count_nonzero(valid, axis=1)
        if np.any(support_counts < self.min_support_samples):
            bad = np.flatnonzero(support_counts < self.min_support_samples).tolist()
            raise ValueError(
                "Requested traces do not meet min_support_samples: "
                f"rows={bad}, minimum={self.min_support_samples}."
            )
        if not wavelet_parts:
            raise ValueError("No requested trace has finite LFM support.")
        wavelet_mean = np.concatenate(wavelet_parts, axis=0).mean(axis=0)
        return TracePrediction(
            keys=selected,
            log_ai=log_ai,
            valid_mask=valid,
            wavelet_mean_normalized=wavelet_mean,
        )

    def predict_volume(
        self,
        *,
        shape: tuple[int, int, int],
        support_mask: np.ndarray,
        output_path: str | Path,
        inline_indices: Sequence[int] | None = None,
        xline_indices: Sequence[int] | None = None,
    ) -> VolumePrediction:
        """Stream a regular volume into an ``.npy`` memmap.

        ``support_mask`` is the authoritative LFM support.  Traces without
        support are never sent through the network and remain NaN/False in the
        two output memmaps.  No spatial fill or nearest-trace fallback is
        performed.
        """

        if len(shape) != 3 or any(int(value) <= 0 for value in shape):
            raise ValueError("shape must be a positive three-dimensional volume shape.")
        n_inline, n_xline, n_sample = (int(value) for value in shape)
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != shape:
            raise ValueError("support_mask must match the requested volume shape.")
        ilines = np.arange(n_inline, dtype=np.int64) if inline_indices is None else np.asarray(inline_indices, dtype=np.int64)
        xlines = np.arange(n_xline, dtype=np.int64) if xline_indices is None else np.asarray(xline_indices, dtype=np.int64)
        if ilines.ndim != 1 or xlines.ndim != 1 or ilines.size == 0 or xlines.size == 0:
            raise ValueError("inline_indices and xline_indices must be non-empty one-dimensional arrays.")
        if np.any(ilines < 0) or np.any(ilines >= n_inline) or np.any(xlines < 0) or np.any(xlines >= n_xline):
            raise ValueError("Volume trace indices are outside the supplied shape.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path = output_path.with_name(f"{output_path.stem}_valid_mask.npy")
        values = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(ilines.size, xlines.size, n_sample)
        )
        masks = np.lib.format.open_memmap(
            mask_path, mode="w+", dtype=np.bool_, shape=(ilines.size, xlines.size, n_sample)
        )
        values[...] = np.nan
        masks[...] = False

        keys: list[TraceKey] = []
        locations: list[tuple[int, int]] = []
        for local_i, global_i in enumerate(ilines.tolist()):
            for local_j, global_j in enumerate(xlines.tolist()):
                if np.count_nonzero(support[int(global_i), int(global_j)]) >= self.min_support_samples:
                    keys.append(TraceKey(int(global_i), int(global_j)))
                    locations.append((local_i, local_j))

        wavelet_sum: np.ndarray | None = None
        predicted_count = 0
        for start in range(0, len(keys), self.batch_size):
            chunk = tuple(keys[start : start + self.batch_size])
            log_ai, valid, wavelets, supported = self._predict_chunk(chunk)
            for row, (local_i, local_j) in enumerate(locations[start : start + len(chunk)]):
                trace_support = valid[row]
                values[local_i, local_j, trace_support] = log_ai[row, trace_support]
                masks[local_i, local_j, trace_support] = True
                if supported[row]:
                    predicted_count += 1
            supported_waves = wavelets[supported]
            if supported_waves.size:
                current = supported_waves.sum(axis=0)
                wavelet_sum = current if wavelet_sum is None else wavelet_sum + current

        values.flush()
        masks.flush()
        if wavelet_sum is None or predicted_count == 0:
            raise ValueError("No finite LFM-supported trace was available for volume inference.")
        wavelet_mean = wavelet_sum / float(predicted_count)
        return VolumePrediction(
            log_ai_path=output_path,
            valid_mask_path=mask_path,
            shape=(int(ilines.size), int(xlines.size), n_sample),
            predicted_trace_count=predicted_count,
            wavelet_mean_normalized=wavelet_mean,
        )


__all__ = ["Inverter", "VolumePrediction"]
