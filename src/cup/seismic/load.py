"""地震数据加载工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from wtie.processing import grid


def _domain_to_basis_type(domain: str) -> str:
    domain_lower = domain.lower()
    if domain_lower == "time":
        return "twt"
    if domain_lower == "depth":
        return "md"
    raise ValueError(f"Unsupported domain: {domain}. Expect 'time' or 'depth'.")


def _resolve_sample_window(samples: np.ndarray, start: Optional[float], end: Optional[float]) -> Tuple[int, int]:
    if start is None:
        start = float(samples[0])
    if end is None:
        end = float(samples[-1])
    if start >= end:
        raise ValueError(f"Invalid window: start={start}, end={end}")

    sample_idx_start = int(np.searchsorted(samples, start, side="left"))
    sample_idx_end = int(np.searchsorted(samples, end, side="right"))
    sample_idx_start = max(0, sample_idx_start)
    sample_idx_end = min(samples.size, sample_idx_end)

    if sample_idx_start >= sample_idx_end:
        raise ValueError("Selected sample window is empty.")

    return sample_idx_start, sample_idx_end


def _interpolate_trace_from_4_neighbors(
    i: float,
    j: float,
    trace00: np.ndarray,
    trace01: np.ndarray,
    trace10: np.ndarray,
    trace11: np.ndarray,
) -> np.ndarray:
    i_floor = int(np.floor(i))
    j_floor = int(np.floor(j))
    wi = i - i_floor
    wj = j - j_floor
    return (1 - wi) * (1 - wj) * trace00 + (1 - wi) * wj * trace01 + wi * (1 - wj) * trace10 + wi * wj * trace11


def _segy_build_samples(meta: Dict, domain: str) -> np.ndarray:
    domain_lower = domain.lower()
    nt = int(meta["nt"])
    if domain_lower == "time":
        start_time_ms = float(meta.get("start_time", 0.0))
        dt_ms = float(meta["dt"]) / 1000.0
        return start_time_ms + np.arange(nt, dtype=np.float64) * dt_ms

    if domain_lower == "depth":
        # 深度域数据中，cigsegy 常仍复用 start_time/dt 这组字段保存采样轴。
        start_depth = float(meta.get("start_depth", meta.get("start_time", 0.0)))
        dz = float(meta.get("dz", meta["dt"])) / 1000.0
        return start_depth + np.arange(nt, dtype=np.float64) * dz

    raise ValueError(f"Unsupported domain: {domain}. Expect 'time' or 'depth'.")


def _segy_pick_affine_reference_points_from_geom(
    geom: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    ni, nx = geom.shape

    for i0 in range(ni):
        for j0 in range(nx):
            if geom[i0, j0] < 0:
                continue
            if i0 + 1 >= ni or j0 + 1 >= nx:
                continue
            if geom[i0 + 1, j0] < 0 or geom[i0, j0 + 1] < 0:
                continue
            return (i0, j0), (i0 + 1, j0), (i0, j0 + 1)

    raise ValueError("Cannot find valid neighboring traces to build SEG-Y coordinate transform.")


def _segy_coord_to_index_from_3points(
    x: float,
    y: float,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    i0: float,
    j0: float,
) -> Tuple[float, float]:
    dx_il = p1[0] - p0[0]
    dy_il = p1[1] - p0[1]
    dx_xl = p2[0] - p0[0]
    dy_xl = p2[1] - p0[1]

    det = dx_il * dy_xl - dy_il * dx_xl
    if abs(det) < 1e-10:
        raise ValueError("Coordinate transform is degenerate for current SEG-Y geometry.")

    dx = x - p0[0]
    dy = y - p0[1]

    di = (dx * dy_xl - dy * dx_xl) / det
    dj = (dy * dx_il - dx * dy_il) / det
    return i0 + di, j0 + dj


def _segy_read_trace_by_index(segy, trace_idx: int, sample_idx_start: int, sample_idx_end: int) -> np.ndarray:
    return segy.collect(trace_idx, trace_idx + 1, sample_idx_start, sample_idx_end).squeeze()


def _import_seismic_at_well_segy(
    segy_file: Path,
    well_x: float,
    well_y: float,
    sample_start: Optional[float],
    sample_end: Optional[float],
    domain: str,
) -> grid.Seismic:
    import cigsegy

    segy = cigsegy.Pysegy(str(segy_file))
    try:
        meta = cigsegy.tools.get_metaInfo(segy)

        offset_keyloc = int(meta.get("offset", 37))
        ostep = int(meta.get("ostep", 1))
        segy.setLocations(int(meta["iline"]), int(meta["xline"]), offset_keyloc)
        segy.setSteps(int(meta["istep"]), int(meta["xstep"]), ostep)
        segy.setXYLocations(int(meta["xloc"]), int(meta["yloc"]))
        segy.set_segy_type(3)
        segy.scan()

        geominfo = cigsegy.tools.full_scan(
            segy,
            iline=int(meta["iline"]),
            xline=int(meta["xline"]),
            offset=offset_keyloc,
            is4d=False,
        )
        geom = np.asarray(geominfo["geom"])
        if geom.ndim != 2:
            raise ValueError("Only 3D post-stack SEG-Y is supported.")

        (i0, j0), (i1, j1), (i2, j2) = _segy_pick_affine_reference_points_from_geom(geom)
        idx0 = int(geom[i0, j0])
        idx1 = int(geom[i1, j1])
        idx2 = int(geom[i2, j2])

        p0 = (float(segy.coordx(idx0)), float(segy.coordy(idx0)))
        p1 = (float(segy.coordx(idx1)), float(segy.coordy(idx1)))
        p2 = (float(segy.coordx(idx2)), float(segy.coordy(idx2)))

        i, j = _segy_coord_to_index_from_3points(well_x, well_y, p0, p1, p2, float(i0), float(j0))

        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))

        ni, nx = geom.shape
        if not (0 <= i_floor < ni and 0 <= i_ceil < ni):
            raise ValueError(f"Well is outside seismic inline range: {i_floor}, {i_ceil}")
        if not (0 <= j_floor < nx and 0 <= j_ceil < nx):
            raise ValueError(f"Well is outside seismic crossline range: {j_floor}, {j_ceil}")

        neighbor_indices = [
            int(geom[i_floor, j_floor]),
            int(geom[i_floor, j_ceil]),
            int(geom[i_ceil, j_floor]),
            int(geom[i_ceil, j_ceil]),
        ]
        if any(idx < 0 for idx in neighbor_indices):
            raise ValueError("Well neighborhood contains missing traces, cannot apply bilinear interpolation.")

        raw_samples = _segy_build_samples(meta, domain=domain)
        sample_idx_start, sample_idx_end = _resolve_sample_window(raw_samples, sample_start, sample_end)

        t00 = _segy_read_trace_by_index(segy, neighbor_indices[0], sample_idx_start, sample_idx_end)
        t01 = _segy_read_trace_by_index(segy, neighbor_indices[1], sample_idx_start, sample_idx_end)
        t10 = _segy_read_trace_by_index(segy, neighbor_indices[2], sample_idx_start, sample_idx_end)
        t11 = _segy_read_trace_by_index(segy, neighbor_indices[3], sample_idx_start, sample_idx_end)

        trace_data = _interpolate_trace_from_4_neighbors(i, j, t00, t01, t10, t11)

        trace_axis = raw_samples[sample_idx_start:sample_idx_end]
        if domain.lower() == "time":
            trace_axis = trace_axis / 1000.0

        basis_type = _domain_to_basis_type(domain)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        return grid.Seismic(values=trace_data, basis=trace_axis, basis_type=basis_type, name=trace_name)
    finally:
        segy.close()


def _zgy_coord_to_index(reader, x: float, y: float) -> Tuple[float, float]:
    """将 XY 坐标转换为 0-based 的 (inline_index, crossline_index)。"""
    dx_il = reader.easting_inc_il
    dy_il = reader.northing_inc_il
    dx_xl = reader.easting_inc_xl
    dy_xl = reader.northing_inc_xl

    x0 = reader.corners[0][0]
    y0 = reader.corners[0][1]

    det = dx_il * dy_xl - dy_il * dx_xl
    if abs(det) < 1e-10:
        raise ValueError("Coordinate system is degenerate (determinant is zero)")

    dx = x - x0
    dy = y - y0

    i = (dx * dy_xl - dy * dx_xl) / det
    j = (dy * dx_il - dx * dy_il) / det

    return i, j


def _zgy_coord_to_line(reader, x: float, y: float) -> Tuple[float, float]:
    """将 XY 坐标转换为 (inline_no, crossline_no)。"""
    i, j = _zgy_coord_to_index(reader, x, y)

    il_no = reader.annotstart[0] + i * reader.annotinc[0]
    xl_no = reader.annotstart[1] + j * reader.annotinc[1]

    return il_no, xl_no


def _zgy_line_to_coord(reader, il_no: float, xl_no: float) -> Tuple[float, float]:
    """将 (inline_no, crossline_no) 转换为 XY 坐标。"""
    i = (il_no - reader.annotstart[0]) / reader.annotinc[0]
    j = (xl_no - reader.annotstart[1]) / reader.annotinc[1]

    x = reader.gen_cdp_x(i, j)
    y = reader.gen_cdp_y(i, j)

    return x, y


def _import_seismic_at_well_zgy(
    zgy_file: Path,
    well_x: float,
    well_y: float,
    sample_start: Optional[float],
    sample_end: Optional[float],
    domain: str,
) -> grid.Seismic:
    import pyzgy

    with pyzgy.open(str(zgy_file), mode="r") as reader:
        i, j = _zgy_coord_to_index(reader, well_x, well_y)

        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))

        if not (0 <= i_floor < reader.n_ilines and 0 <= i_ceil < reader.n_ilines):
            raise ValueError(f"Well is outside seismic inline range: {i_floor}, {i_ceil}")
        if not (0 <= j_floor < reader.n_xlines and 0 <= j_ceil < reader.n_xlines):
            raise ValueError(f"Well is outside seismic crossline range: {j_floor}, {j_ceil}")

        raw_samples = np.asarray(reader.samples, dtype=np.float64)
        sample_idx_start, sample_idx_end = _resolve_sample_window(raw_samples, sample_start, sample_end)

        t00 = reader.get_trace(i_floor * reader.n_xlines + j_floor)[sample_idx_start:sample_idx_end]
        t01 = reader.get_trace(i_floor * reader.n_xlines + j_ceil)[sample_idx_start:sample_idx_end]
        t10 = reader.get_trace(i_ceil * reader.n_xlines + j_floor)[sample_idx_start:sample_idx_end]
        t11 = reader.get_trace(i_ceil * reader.n_xlines + j_ceil)[sample_idx_start:sample_idx_end]

        trace_data = _interpolate_trace_from_4_neighbors(i, j, t00, t01, t10, t11)

        trace_axis = raw_samples[sample_idx_start:sample_idx_end]
        if domain.lower() == "time":
            trace_axis = trace_axis / 1000.0

    basis_type = _domain_to_basis_type(domain)
    trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
    return grid.Seismic(values=trace_data, basis=trace_axis, basis_type=basis_type, name=trace_name)


def import_seismic_at_well(
    seismic_file: Path,
    well_x: float,
    well_y: float,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    domain: str = "time",
    seismic_type: str = "segy",
) -> grid.Seismic:
    """提取井位处的复合地震道（双线性插值）。

    Parameters
    ----------
    seismic_file : Path
            地震文件路径，支持 .zgy / .sgy / .segy。
    well_x : float
            井位 X 坐标。
    well_y : float
            井位 Y 坐标。
    sample_start : float, optional
            采样轴起点。domain='time' 时单位为 ms；domain='depth' 时视为深度轴单位。
    sample_end : float, optional
            采样轴终点。domain='time' 时单位为 ms；domain='depth' 时视为深度轴单位。
    domain : str, optional
            采样域类型，'time' 或 'depth'，默认 'time'。
    seismic_type : str, optional
            地震文件类型，支持 'segy' 或 'zgy'，默认 'segy'。

    Returns
    -------
    grid.Seismic
            井旁地震道对象。
    """
    seismic_type_lower = seismic_type.lower()
    if seismic_type_lower == "zgy":
        return _import_seismic_at_well_zgy(seismic_file, well_x, well_y, sample_start, sample_end, domain)
    if seismic_type_lower == "segy":
        return _import_seismic_at_well_segy(seismic_file, well_x, well_y, sample_start, sample_end, domain)

    raise ValueError(f"Unsupported seismic_type: {seismic_type}. Expect 'segy' or 'zgy'.")
