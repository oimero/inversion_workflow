"""Reusable SEG-Y/ZGY volume export helpers.

The functions in this module are intentionally business-agnostic: callers pass
a regular ``[inline, xline, sample]`` volume plus explicit axes and a source
seismic file.  The output format follows the source seismic type so research
artifacts can be loaded into interpretation software without every script
reimplementing header/geometry handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import build_segy_textual_header, sha256_file


def log_ai_to_ai_volume(log_ai: np.ndarray) -> np.ndarray:
    """Convert a log(AI) export volume to finite positive linear AI."""

    values = np.asarray(log_ai, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        ai = np.exp(values)
    invalid = np.isfinite(values) & (
        ~np.isfinite(ai)
        | (ai <= 0.0)
        | (ai > np.finfo(np.float32).max)
    )
    if np.any(invalid):
        raise ValueError("Cannot export AI: exp(log_ai) produced non-finite or non-positive values.")
    output = ai.astype(np.float32)
    invalid_output = np.isfinite(values) & (~np.isfinite(output) | (output <= 0.0))
    if np.any(invalid_output):
        raise ValueError("Cannot export AI: exp(log_ai) is outside the float32 positive range.")
    return output


def export_volume_like_source(
    *,
    output_base: Path,
    volume: np.ndarray,
    ilines: Sequence[float],
    xlines: Sequence[float],
    samples: Sequence[float],
    source_seismic_file: Path,
    source_seismic_type: str,
    title: str,
    details: Sequence[str] | None = None,
    seismic_options: Mapping[str, Any] | None = None,
    inline_chunk_size: int = 16,
    nan_fill: float | None = None,
) -> dict[str, Any]:
    """Export a volume in the same format as the source seismic.

    Parameters
    ----------
    output_base:
        Output path without suffix, or with a suffix that will be replaced.
    volume:
        Regular volume with shape ``[n_inline, n_xline, n_sample]``.
    ilines, xlines, samples:
        Explicit physical axes.  ``samples`` is in seconds for time-domain ZGY
        and is passed through as sample coordinates for SEG-Y header sharing.
    source_seismic_file, source_seismic_type:
        Source seismic used for geometry/header provenance.
    nan_fill:
        Optional replacement for non-finite samples.  ``None`` preserves NaN.
    """

    source_type = str(source_seismic_type).casefold()
    output_base = Path(output_base)
    if source_type == "zgy":
        target = output_base.with_suffix(".zgy")
        _write_zgy(
            target,
            volume=volume,
            ilines=ilines,
            xlines=xlines,
            samples=samples,
            source_seismic_file=Path(source_seismic_file),
            inline_chunk_size=int(inline_chunk_size),
            nan_fill=nan_fill,
        )
        return _export_payload(target, "zgy")
    if source_type == "segy":
        target = output_base.with_suffix(".segy")
        _write_segy(
            target,
            volume=volume,
            source_seismic_file=Path(source_seismic_file),
            title=title,
            details=details or [],
            seismic_options=seismic_options or {},
            nan_fill=nan_fill,
        )
        return _export_payload(target, "segy")
    raise ValueError(f"Unsupported source seismic type for volume export: {source_seismic_type!r}")


def _export_payload(path: Path, fmt: str) -> dict[str, Any]:
    return {
        "status": "written",
        "format": fmt,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _prepared_volume(volume: np.ndarray, *, nan_fill: float | None) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Export volume must be 3D [inline, xline, sample], got shape {values.shape}.")
    out = np.ascontiguousarray(values)
    if nan_fill is not None:
        out = np.where(np.isfinite(out), out, np.float32(nan_fill)).astype(np.float32, copy=False)
    return np.ascontiguousarray(out, dtype=np.float32)


def _validate_axes(
    *,
    volume: np.ndarray,
    ilines: Sequence[float],
    xlines: Sequence[float],
    samples: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    il_axis = np.asarray(ilines, dtype=np.float64).reshape(-1)
    xl_axis = np.asarray(xlines, dtype=np.float64).reshape(-1)
    sample_axis = np.asarray(samples, dtype=np.float64).reshape(-1)
    if volume.shape != (il_axis.size, xl_axis.size, sample_axis.size):
        raise ValueError(
            "Export volume shape does not match axes: "
            f"volume={volume.shape}, ilines={il_axis.size}, xlines={xl_axis.size}, samples={sample_axis.size}."
        )
    if sample_axis.size < 2:
        raise ValueError("Volume export requires at least two samples.")
    if il_axis.size < 1 or xl_axis.size < 1:
        raise ValueError("Volume export requires non-empty inline and xline axes.")
    for name, axis in [("ilines", il_axis), ("xlines", xl_axis), ("samples", sample_axis)]:
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} axis contains non-finite values.")
        if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
            raise ValueError(f"{name} axis must be strictly increasing.")
    return il_axis, xl_axis, sample_axis


def _axis_step(axis: np.ndarray, *, name: str) -> float:
    if axis.size <= 1:
        return 0.0
    step = float(np.median(np.diff(axis)))
    if not np.allclose(np.diff(axis), step, rtol=1e-6, atol=1e-9):
        raise ValueError(f"{name} axis must be regular for volume export.")
    return step


def _zgy_corners_from_survey(survey: Any, ilines: np.ndarray, xlines: np.ndarray) -> tuple[tuple[float, float], ...]:
    geometry = survey.line_geometry
    il0 = float(ilines[0])
    iln = float(ilines[-1])
    xl0 = float(xlines[0])
    xln = float(xlines[-1])
    return (
        tuple(float(v) for v in geometry.line_to_coord(il0, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(il0, xln)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xln)),
    )


def _write_zgy(
    path: Path,
    *,
    volume: np.ndarray,
    ilines: Sequence[float],
    xlines: Sequence[float],
    samples: Sequence[float],
    source_seismic_file: Path,
    inline_chunk_size: int,
    nan_fill: float | None,
) -> None:
    from pyzgy.write import SeismicWriter

    export_volume = _prepared_volume(volume, nan_fill=nan_fill)
    il_axis, xl_axis, sample_axis = _validate_axes(
        volume=export_volume,
        ilines=ilines,
        xlines=xlines,
        samples=samples,
    )
    sample_step_s = _axis_step(sample_axis, name="samples")
    inline_inc = _axis_step(il_axis, name="ilines") if il_axis.size > 1 else 0.0
    xline_inc = _axis_step(xl_axis, name="xlines") if xl_axis.size > 1 else 0.0
    survey = open_survey(source_seismic_file, seismic_type="zgy")
    corners = _zgy_corners_from_survey(survey, il_axis, xl_axis)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    chunk = max(1, int(inline_chunk_size))
    with SeismicWriter(
        path,
        tuple(int(v) for v in export_volume.shape),
        float(sample_axis[0]) * 1000.0,
        sample_step_s * 1000.0,
        (float(il_axis[0]), float(xl_axis[0])),
        (inline_inc, xline_inc),
        corners=corners,
    ) as writer:
        for il_start in range(0, export_volume.shape[0], chunk):
            il_end = min(export_volume.shape[0], il_start + chunk)
            writer.write_subvolume(export_volume[il_start:il_end], il_start, 0, 0)


def _write_segy(
    path: Path,
    *,
    volume: np.ndarray,
    source_seismic_file: Path,
    title: str,
    details: Sequence[str],
    seismic_options: Mapping[str, Any],
    nan_fill: float | None,
) -> None:
    import cigsegy

    export_volume = _prepared_volume(volume, nan_fill=nan_fill)
    options = segy_options_from_config(dict(seismic_options))
    keylocs = [options.get(key) for key in ("iline", "xline", "istep", "xstep")]
    if any(value is None for value in keylocs):
        raise ValueError(
            "SEG-Y volume export requires iline/xline/istep/xstep key locations "
            "in seismic_options."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    textual = build_segy_textual_header(title, list(details))
    cigsegy.create_by_sharing_header(
        str(path),
        str(source_seismic_file),
        export_volume,
        keylocs=[int(value) for value in keylocs],
        textual=textual,
    )
