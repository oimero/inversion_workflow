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

from cup.seismic.geometry import SampleAxis
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


def _exact_line_index(axis: np.ndarray, requested: float, *, name: str) -> int:
    """Return the index of one physical line number already present on ``axis``.

    The small absolute tolerance only absorbs decimal representation noise in
    an axis read from a file.  It does not snap a request to a neighboring
    line, which is important for line axes whose step is greater than one.
    """

    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} axis must be a non-empty finite one-dimensional array.")
    values = values.reshape(-1)
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} axis must be strictly increasing.")
    target = float(requested)
    if not np.isfinite(target):
        raise ValueError(f"Requested {name} line must be finite, got {requested!r}.")
    matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1.0e-8))
    if matches.size != 1:
        raise ValueError(
            f"Requested {name} line {target:g} is not present on the supplied axis; "
            "line requests must match an existing physical line exactly."
        )
    return int(matches[0])


def _line_token(value: float) -> str:
    """Format a physical line number for a stable output filename."""

    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".12g").replace("-", "m").replace(".", "p")


def _section_values(
    log_ai: np.ndarray,
    valid_mask: np.ndarray,
    *,
    direction: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one masked log-AI section to finite linear-AI values."""

    if direction == "inline":
        log_section = np.asarray(log_ai[index, :, :])
        mask_section = np.asarray(valid_mask[index, :, :], dtype=bool)
    elif direction == "xline":
        log_section = np.asarray(log_ai[:, index, :])
        mask_section = np.asarray(valid_mask[:, index, :], dtype=bool)
    else:
        raise ValueError(f"Unsupported section direction: {direction!r}.")
    if log_section.ndim != 2 or mask_section.shape != log_section.shape:
        raise ValueError("A volume section must have shape [lateral, sample].")

    with np.errstate(over="ignore", invalid="ignore"):
        linear = np.exp(log_section.astype(np.float64, copy=False))
    invalid = mask_section & (
        ~np.isfinite(log_section) | ~np.isfinite(linear) | (linear <= 0.0)
    )
    if np.any(invalid):
        raise ValueError(
            f"{direction} section contains non-finite or non-positive linear-AI values "
            "inside the valid prediction mask."
        )
    finite = mask_section & np.isfinite(log_section) & np.isfinite(linear) & (linear > 0.0)
    values = np.full(log_section.shape, np.nan, dtype=np.float64)
    values[finite] = linear[finite]
    support_by_sample = np.any(finite, axis=0)
    supported = np.flatnonzero(support_by_sample)
    if supported.size == 0:
        raise ValueError(f"Requested {direction} section has no finite valid prediction samples.")
    return values, np.asarray(support_by_sample, dtype=bool)


def validate_volume_section_requests(
    *,
    ilines: Sequence[float],
    xlines: Sequence[float],
    inline_numbers: Sequence[float] = (),
    xline_numbers: Sequence[float] = (),
) -> dict[str, tuple[int, ...]]:
    """Validate physical section line requests before starting inference.

    The returned indices are array indices corresponding to the supplied
    physical axes.  Requests are exact axis members; a nearby line is never
    selected implicitly.
    """

    requested_inline = tuple(float(value) for value in inline_numbers)
    requested_xline = tuple(float(value) for value in xline_numbers)
    if not requested_inline and not requested_xline:
        raise ValueError("At least one inline or xline must be requested for section QC.")
    return {
        "inline": tuple(_exact_line_index(np.asarray(ilines), value, name="inline") for value in requested_inline),
        "xline": tuple(_exact_line_index(np.asarray(xlines), value, name="xline") for value in requested_xline),
    }


def write_volume_section_qc(
    *,
    log_ai_path: str | Path,
    valid_mask_path: str | Path,
    ilines: Sequence[float],
    xlines: Sequence[float],
    sample_axis: SampleAxis,
    output_dir: str | Path,
    inline_numbers: Sequence[float] = (),
    xline_numbers: Sequence[float] = (),
) -> dict[str, object]:
    """Write shared-scale inline and xline linear-AI section QC figures.

    ``ilines`` and ``xlines`` are physical survey axes, not array indices.
    Requested lines must be members of those axes; no nearest-line fallback is
    applied.  The returned payload contains PNG and SVG paths for each figure.
    Invalid samples remain masked in the image, and the vertical range is
    cropped to the joint supported sample interval of the requested sections.
    """

    requested_inline = tuple(float(value) for value in inline_numbers)
    requested_xline = tuple(float(value) for value in xline_numbers)
    validated = validate_volume_section_requests(
        ilines=ilines,
        xlines=xlines,
        inline_numbers=requested_inline,
        xline_numbers=requested_xline,
    )

    inline_axis = np.asarray(ilines, dtype=np.float64).reshape(-1)
    xline_axis = np.asarray(xlines, dtype=np.float64).reshape(-1)
    log_path = Path(log_ai_path)
    mask_path = Path(valid_mask_path)
    log_volume = np.load(log_path, mmap_mode="r")
    valid_volume = np.load(mask_path, mmap_mode="r")
    if log_volume.ndim != 3 or valid_volume.shape != log_volume.shape:
        raise ValueError("Inference log-AI and valid-mask arrays must be matching three-dimensional volumes.")
    if log_volume.shape != (inline_axis.size, xline_axis.size, log_volume.shape[2]):
        raise ValueError(
            "Inference volume shape does not match supplied inline/xline axes: "
            f"shape={log_volume.shape}, axes=({inline_axis.size}, {xline_axis.size})."
        )

    if not isinstance(sample_axis, SampleAxis):
        raise TypeError("sample_axis must be an explicit SampleAxis instance.")
    sample_values = np.asarray(sample_axis.values, dtype=np.float64).reshape(-1)
    if sample_values.size != log_volume.shape[2] or sample_values.size == 0:
        raise ValueError("sample_axis must contain one value for every volume sample.")
    if np.any(~np.isfinite(sample_values)) or (
        sample_values.size > 1 and np.any(np.diff(sample_values) <= 0.0)
    ):
        raise ValueError("sample_axis must be finite and strictly increasing.")
    domain = str(sample_axis.domain).strip().casefold()
    unit = str(sample_axis.unit).strip()
    if domain not in {"time", "depth"}:
        raise ValueError(f"Unsupported sample-axis domain: {domain!r}.")

    sections: list[dict[str, object]] = []
    for index in validated["inline"]:
        values, support = _section_values(log_volume, valid_volume, direction="inline", index=index)
        sections.append(
            {
                "direction": "inline",
                "line": float(inline_axis[index]),
                "index": index,
                "values": values,
                "support": support,
            }
        )
    for index in validated["xline"]:
        values, support = _section_values(log_volume, valid_volume, direction="xline", index=index)
        sections.append(
            {
                "direction": "xline",
                "line": float(xline_axis[index]),
                "index": index,
                "values": values,
                "support": support,
            }
        )

    supported_indices = [
        np.flatnonzero(np.asarray(section["support"], dtype=bool)) for section in sections
    ]
    sample_start = min(int(indices[0]) for indices in supported_indices)
    sample_stop = max(int(indices[-1]) + 1 for indices in supported_indices)
    finite_values = np.concatenate(
        [
            np.asarray(section["values"], dtype=np.float64)[:, sample_start:sample_stop].ravel()
            for section in sections
        ]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Requested section QC has no finite linear-AI values.")
    vmin, vmax = (float(value) for value in np.nanpercentile(finite_values, [2.0, 98.0]))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Cannot determine a finite linear-AI color scale for section QC.")
    if vmax <= vmin:
        delta = max(abs(vmin) * 0.01, 1.0)
        vmin, vmax = vmin - delta, vmax + delta

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0.0)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_payload: list[dict[str, object]] = []
    for section in sections:
        direction = str(section["direction"])
        line = float(section["line"])
        index = int(section["index"])
        values = np.asarray(section["values"], dtype=np.float64)[:, sample_start:sample_stop]
        vertical = sample_values[sample_start:sample_stop]
        horizontal = xline_axis if direction == "inline" else inline_axis
        image_values = np.ma.masked_invalid(values.T)

        fig, axis = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
        mesh = axis.pcolormesh(
            horizontal,
            vertical,
            image_values,
            shading="nearest",
            rasterized=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_xlim(float(horizontal[0]), float(horizontal[-1]))
        axis.set_ylim(float(vertical[-1]), float(vertical[0]))
        axis.set_xlabel("Xline" if direction == "inline" else "Inline")
        if domain == "depth":
            ylabel = f"{str(sample_axis.depth_basis).upper()} ({unit})"
        else:
            ylabel = f"{domain.title()} ({unit})"
        axis.set_ylabel(ylabel)
        axis.set_title(f"{direction.title()} {line:g} — predicted linear AI")
        colorbar = fig.colorbar(mesh, ax=axis, pad=0.02, fraction=0.035, extend="both")
        colorbar.set_label("Linear AI (m/s*g/cm³)")

        token = _line_token(line)
        stem = f"{direction}_{token}"
        png_path = output / f"{stem}.png"
        svg_path = output / f"{stem}.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_payload.append(
            {
                "direction": direction,
                "line": line,
                "index": index,
                "png": str(png_path),
                "svg": str(svg_path),
                "sample_start": float(vertical[0]),
                "sample_stop": float(vertical[-1]),
            }
        )

    return {
        "sample_domain": domain,
        "sample_unit": unit,
        "color_scale": {
            "quantity": "linear_ai",
            "unit": "m/s*g/cm3",
            "method": "finite requested-section values, 2nd-98th percentiles",
            "vmin": vmin,
            "vmax": vmax,
        },
        "figures": figure_payload,
    }


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
        valid = (batch.lfm_mask & batch.output_mask).detach().cpu().numpy().astype(bool, copy=True)
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


__all__ = [
    "Inverter",
    "VolumePrediction",
    "validate_volume_section_requests",
    "write_volume_section_qc",
]
