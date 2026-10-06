"""Reusable SEG-Y/ZGY/NPZ volume export helpers.

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


def build_segy_textual_header(title: str, lines: list[str] | None = None) -> str:
    """Construct the forty fixed-width rows of a SEG-Y textual header."""
    all_lines = [title] + (lines or [])
    rows = [f"C{i:>2d} {text}"[:80].ljust(80) for i, text in enumerate(all_lines, start=1)]
    rows.extend([f"C{i:>2d}".ljust(80) for i in range(len(rows) + 1, 41)])
    textual = "".join(rows)
    if len(textual) != 3200:
        raise ValueError(f"Expected 3200-char textual header, got {len(textual)}")
    return textual


def log_ai_to_ai_volume(log_ai: np.ndarray) -> np.ndarray:
    """Convert log(AI) to float32 AI with bounded float64 exponential blocks."""

    values = np.asarray(log_ai)
    output = np.empty(values.shape, dtype=np.float32)
    with np.nditer(
        [values, output],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["writeonly"]],
        op_dtypes=[np.float64, np.float32],
        casting="unsafe",
        buffersize=1 << 20,
    ) as blocks:
        for block, target in blocks:
            with np.errstate(over="ignore", invalid="ignore"):
                ai = np.exp(block)
            finite = np.isfinite(block)
            invalid = finite & (
                ~np.isfinite(ai)
                | (ai <= 0.0)
                | (ai > np.finfo(np.float32).max)
            )
            if np.any(invalid):
                raise ValueError("Cannot export AI: exp(log_ai) produced non-finite or non-positive values.")
            converted = ai.astype(np.float32)
            if np.any(finite & (~np.isfinite(converted) | (converted <= 0.0))):
                raise ValueError("Cannot export AI: exp(log_ai) is outside the float32 positive range.")
            target[...] = converted
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
    sample_domain: str,
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
        Explicit physical axes.  ``samples`` is in seconds for the time domain
        and metres for the depth domain.
    source_seismic_file, source_seismic_type:
        Source seismic used for geometry/header provenance.
    sample_domain:
        Explicit sample domain, either ``"time"`` or ``"depth"``.  It is
        required for both output formats so ZGY header units cannot be inferred
        from the file type.
    nan_fill:
        Optional replacement for non-finite samples.  ``None`` preserves NaN.
    """

    source_type = str(source_seismic_type).casefold()
    domain = str(sample_domain).strip().casefold()
    if domain not in {"time", "depth"}:
        raise ValueError("sample_domain must be 'time' or 'depth'.")
    output_base = Path(output_base)
    if source_type == "zgy":
        target = output_base.with_suffix(".zgy")
        _write_zgy(
            target,
            volume=volume,
            ilines=ilines,
            xlines=xlines,
            samples=samples,
            sample_domain=domain,
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
    if source_type == "npz":
        target = output_base.with_suffix(".npz")
        unit = _write_npz(
            target,
            volume=volume,
            ilines=ilines,
            xlines=xlines,
            samples=samples,
            sample_domain=domain,
            source_seismic_file=Path(source_seismic_file),
            nan_fill=nan_fill,
        )
        payload = _export_payload(target, "npz")
        payload.update(
            {
                "sample_domain": domain,
                "sample_unit": unit,
                "shape": list(np.asarray(volume).shape),
            }
        )
        return payload
    raise ValueError(f"Unsupported source seismic type for volume export: {source_seismic_type!r}")


def _export_payload(path: Path, fmt: str) -> dict[str, Any]:
    return {
        "status": "written",
        "format": fmt,
        "path": str(path),
    }


def _write_npz(
    path: Path,
    *,
    volume: np.ndarray,
    ilines: Sequence[float],
    xlines: Sequence[float],
    samples: Sequence[float],
    sample_domain: str,
    source_seismic_file: Path,
    nan_fill: float | None,
) -> str:
    """Write an explicit NPZ survey while preserving source XY geometry."""

    source = open_survey(source_seismic_file, seismic_type="npz")
    source_axis = source.sample_axis(sample_domain)
    export_volume = _prepared_volume(volume, nan_fill=nan_fill)
    il_axis, xl_axis, sample_axis = _validate_axes(
        volume=export_volume,
        ilines=ilines,
        xlines=xlines,
        samples=samples,
    )
    source_ilines = source.line_geometry.inline_axis.values()
    source_xlines = source.line_geometry.xline_axis.values()
    if not np.all(np.isin(il_axis, source_ilines)) or not np.all(np.isin(xl_axis, source_xlines)):
        raise ValueError("NPZ export axes must be subsets of the source NPZ geometry axes.")
    if not np.all(np.isin(sample_axis, source_axis.values)):
        raise ValueError("NPZ export samples must be a subset of the source NPZ sample axis.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seismic=export_volume,
        sample_values=sample_axis.astype(np.float64),
        sample_domain=np.asarray(source_axis.domain),
        sample_unit=np.asarray(source_axis.unit),
        depth_basis=np.asarray("" if source_axis.depth_basis is None else source_axis.depth_basis),
        ilines=il_axis.astype(np.float64),
        xlines=xl_axis.astype(np.float64),
        origin_xy_m=np.asarray(
            [source.line_geometry.x0, source.line_geometry.y0],
            dtype=np.float64,
        ),
        inline_step_xy_m=np.asarray(
            [source.line_geometry.dx_inline, source.line_geometry.dy_inline],
            dtype=np.float64,
        ),
        xline_step_xy_m=np.asarray(
            [source.line_geometry.dx_xline, source.line_geometry.dy_xline],
            dtype=np.float64,
        ),
    )
    return source_axis.unit


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
    sample_domain: str,
    source_seismic_file: Path,
    inline_chunk_size: int,
    nan_fill: float | None,
) -> None:
    from openzgy.api import SampleDataType, UnitDimension, ZgyWriter

    export_volume = _prepared_volume(volume, nan_fill=nan_fill)
    il_axis, xl_axis, sample_axis = _validate_axes(
        volume=export_volume,
        ilines=ilines,
        xlines=xlines,
        samples=samples,
    )
    sample_step = _axis_step(sample_axis, name="samples")
    inline_inc = _axis_step(il_axis, name="ilines") if il_axis.size > 1 else 0.0
    xline_inc = _axis_step(xl_axis, name="xlines") if xl_axis.size > 1 else 0.0
    unit_dimension, unit_name, unit_factor = {
        "time": (UnitDimension.time, "ms", 0.001),
        "depth": (UnitDimension.length, "m", 1.0),
    }.get(sample_domain, (None, None, None))
    if unit_dimension is None:
        raise ValueError("sample_domain must be 'time' or 'depth'.")
    survey = open_survey(source_seismic_file, seismic_type="zgy")
    corners = _zgy_corners_from_survey(survey, il_axis, xl_axis)

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    chunk = max(1, int(inline_chunk_size))
    with ZgyWriter(
        str(path),
        size=tuple(int(v) for v in export_volume.shape),
        datatype=SampleDataType.float,
        zunitdim=unit_dimension,
        zunitname=unit_name,
        zunitfactor=unit_factor,
        zstart=float(sample_axis[0]) * (1000.0 if sample_domain == "time" else 1.0),
        zinc=sample_step * (1000.0 if sample_domain == "time" else 1.0),
        annotstart=(float(il_axis[0]), float(xl_axis[0])),
        annotinc=(inline_inc, xline_inc),
        corners=corners,
    ) as writer:
        for il_start in range(0, export_volume.shape[0], chunk):
            il_end = min(export_volume.shape[0], il_start + chunk)
            writer.write((il_start, 0, 0), export_volume[il_start:il_end])


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
