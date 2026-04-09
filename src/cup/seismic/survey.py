"""地震工区信息查询与数据处理模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from wtie.processing import grid


class SurveyContext(Protocol):
    """统一地震工区上下文接口。"""

    def query_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]: ...

    def coord_to_line(self, x: float, y: float) -> Tuple[float, float]: ...

    def line_to_coord(self, il_no: float, xl_no: float) -> Tuple[float, float]: ...

    def import_seismic_at_well(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic: ...


def _axis_stats(axis: np.ndarray) -> Dict[str, float]:
    if axis.size == 0:
        raise ValueError("Axis is empty.")
    if axis.size == 1:
        return {"min": float(axis[0]), "max": float(axis[0]), "step": 0.0}
    return {
        "min": float(axis[0]),
        "max": float(axis[-1]),
        "step": float(axis[1] - axis[0]),
    }


def _normalize_domain(domain: Optional[str]) -> str:
    if domain is None:
        return "time"
    domain_lower = domain.lower()
    if domain_lower not in {"time", "depth"}:
        raise ValueError(f"Unsupported domain: {domain}. Expect 'time' or 'depth'.")
    return domain_lower


def _domain_to_basis_type(domain: str) -> str:
    domain_lower = _normalize_domain(domain)
    if domain_lower == "time":
        return "twt"
    return "md"


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


def _segy_build_samples(meta: Dict[str, Any], domain: str) -> np.ndarray:
    domain_lower = _normalize_domain(domain)
    nt = int(meta["nt"])
    if domain_lower == "time":
        start_time_ms = float(meta.get("start_time", 0.0))
        dt_ms = float(meta["dt"]) / 1000.0
        return start_time_ms + np.arange(nt, dtype=np.float64) * dt_ms

    start_depth = float(meta.get("start_depth", meta.get("start_time", 0.0)))
    dz = float(meta.get("dz", meta["dt"])) / 1000.0
    return start_depth + np.arange(nt, dtype=np.float64) * dz


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


def _segy_coord_scalar_to_factor(coord_scalar: float) -> float:
    """Convert SEG-Y coordinate scalar to multiplicative factor."""
    if coord_scalar == 0:
        return 1.0
    if coord_scalar > 0:
        return float(coord_scalar)
    return 1.0 / abs(float(coord_scalar))


@dataclass(frozen=True)
class SegySurveyContext:
    """SEG-Y 工区上下文，封装坐标映射与井旁道提取。"""

    seismic_file: Path
    meta: Dict[str, Any]
    geom: np.ndarray
    i0: float
    j0: float
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    min_iline: float
    min_xline: float
    istep: float
    xstep: float

    @classmethod
    def from_file(
        cls,
        seismic_file: Path,
        iline: Optional[int] = None,
        xline: Optional[int] = None,
        istep: Optional[int] = None,
        xstep: Optional[int] = None,
    ) -> "SegySurveyContext":
        import cigsegy

        segy = cigsegy.Pysegy(str(seismic_file))
        try:
            meta = cigsegy.tools.get_metaInfo(segy, apply_scalar=True)

            offset_keyloc = int(meta.get("offset", 37))
            ostep = int(meta.get("ostep", 1))
            iline_keyloc = int(meta["iline"]) if iline is None else int(iline)
            xline_keyloc = int(meta["xline"]) if xline is None else int(xline)
            il_step = int(meta["istep"]) if istep is None else int(istep)
            xl_step = int(meta["xstep"]) if xstep is None else int(xstep)

            segy.setLocations(iline_keyloc, xline_keyloc, offset_keyloc)
            segy.setSteps(il_step, xl_step, ostep)
            segy.setXYLocations(int(meta["xloc"]), int(meta["yloc"]))
            segy.set_segy_type(3)
            segy.scan()

            geominfo = cigsegy.tools.full_scan(
                segy,
                iline=iline_keyloc,
                xline=xline_keyloc,
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

            coord_factor = _segy_coord_scalar_to_factor(float(meta.get("scalar", 1.0)))

            p0 = (float(segy.coordx(idx0)) * coord_factor, float(segy.coordy(idx0)) * coord_factor)
            p1 = (float(segy.coordx(idx1)) * coord_factor, float(segy.coordy(idx1)) * coord_factor)
            p2 = (float(segy.coordx(idx2)) * coord_factor, float(segy.coordy(idx2)) * coord_factor)

            return cls(
                seismic_file=Path(seismic_file),
                meta=meta,
                geom=geom,
                i0=float(i0),
                j0=float(j0),
                p0=p0,
                p1=p1,
                p2=p2,
                min_iline=float(geominfo["iline"]["min_iline"]),
                min_xline=float(geominfo["xline"]["min_xline"]),
                istep=float(geominfo["iline"]["istep"]),
                xstep=float(geominfo["xline"]["xstep"]),
            )
        finally:
            segy.close()

    def query_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]:
        domain_value = _normalize_domain(domain)

        ilines = self.min_iline + np.arange(self.geom.shape[0], dtype=np.float64) * self.istep
        xlines = self.min_xline + np.arange(self.geom.shape[1], dtype=np.float64) * self.xstep
        samples = _segy_build_samples(self.meta, domain_value)

        inline_stats = _axis_stats(ilines)
        xline_stats = _axis_stats(xlines)
        sample_stats = _axis_stats(samples)

        sample_unit = "ms" if domain_value == "time" else "m"
        return {
            "inline_min": inline_stats["min"],
            "inline_max": inline_stats["max"],
            "inline_step": inline_stats["step"],
            "xline_min": xline_stats["min"],
            "xline_max": xline_stats["max"],
            "xline_step": xline_stats["step"],
            "sample_min": sample_stats["min"],
            "sample_max": sample_stats["max"],
            "sample_step": sample_stats["step"],
            "sample_domain": domain_value,
            "sample_unit": sample_unit,
        }

    def coord_to_index(self, x: float, y: float) -> Tuple[float, float]:
        i, j = _segy_coord_to_index_from_3points(x, y, self.p0, self.p1, self.p2, self.i0, self.j0)

        ni, nx = self.geom.shape
        if not (0 <= i <= ni - 1):
            raise ValueError(f"Point is outside SEG-Y inline range: {i}")
        if not (0 <= j <= nx - 1):
            raise ValueError(f"Point is outside SEG-Y crossline range: {j}")
        return i, j

    def coord_to_line(self, x: float, y: float) -> Tuple[float, float]:
        i, j = self.coord_to_index(x, y)
        il_no = self.min_iline + i * self.istep
        xl_no = self.min_xline + j * self.xstep
        return il_no, xl_no

    def line_to_index(self, il_no: float, xl_no: float) -> Tuple[float, float]:
        i = (il_no - self.min_iline) / self.istep
        j = (xl_no - self.min_xline) / self.xstep

        ni, nx = self.geom.shape
        if not (0 <= i <= ni - 1):
            raise ValueError(f"Inline is outside SEG-Y range: {il_no}")
        if not (0 <= j <= nx - 1):
            raise ValueError(f"Crossline is outside SEG-Y range: {xl_no}")
        return i, j

    def line_to_coord(self, il_no: float, xl_no: float) -> Tuple[float, float]:
        i, j = self.line_to_index(il_no, xl_no)

        dx_il = self.p1[0] - self.p0[0]
        dy_il = self.p1[1] - self.p0[1]
        dx_xl = self.p2[0] - self.p0[0]
        dy_xl = self.p2[1] - self.p0[1]
        di = i - self.i0
        dj = j - self.j0

        x = self.p0[0] + di * dx_il + dj * dx_xl
        y = self.p0[1] + di * dy_il + dj * dy_xl
        return x, y

    def import_seismic_at_well(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic:
        domain_value = _normalize_domain(domain)

        i, j = self.coord_to_index(well_x, well_y)
        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))

        ni, nx = self.geom.shape
        if not (0 <= i_floor < ni and 0 <= i_ceil < ni):
            raise ValueError(f"Well is outside seismic inline range: {i_floor}, {i_ceil}")
        if not (0 <= j_floor < nx and 0 <= j_ceil < nx):
            raise ValueError(f"Well is outside seismic crossline range: {j_floor}, {j_ceil}")

        neighbor_indices = [
            int(self.geom[i_floor, j_floor]),
            int(self.geom[i_floor, j_ceil]),
            int(self.geom[i_ceil, j_floor]),
            int(self.geom[i_ceil, j_ceil]),
        ]
        if any(idx < 0 for idx in neighbor_indices):
            raise ValueError("Well neighborhood contains missing traces, cannot apply bilinear interpolation.")

        raw_samples = _segy_build_samples(self.meta, domain=domain_value)
        sample_idx_start, sample_idx_end = _resolve_sample_window(raw_samples, sample_start, sample_end)

        import cigsegy

        segy = cigsegy.Pysegy(str(self.seismic_file))
        try:
            t00 = segy.collect(neighbor_indices[0], neighbor_indices[0] + 1, sample_idx_start, sample_idx_end).squeeze()
            t01 = segy.collect(neighbor_indices[1], neighbor_indices[1] + 1, sample_idx_start, sample_idx_end).squeeze()
            t10 = segy.collect(neighbor_indices[2], neighbor_indices[2] + 1, sample_idx_start, sample_idx_end).squeeze()
            t11 = segy.collect(neighbor_indices[3], neighbor_indices[3] + 1, sample_idx_start, sample_idx_end).squeeze()
        finally:
            segy.close()

        trace_data = _interpolate_trace_from_4_neighbors(i, j, t00, t01, t10, t11)
        trace_axis = raw_samples[sample_idx_start:sample_idx_end]
        if domain_value == "time":
            trace_axis = trace_axis / 1000.0

        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        return grid.Seismic(values=trace_data, basis=trace_axis, basis_type=basis_type, name=trace_name)


@dataclass(frozen=True)
class ZgySurveyContext:
    """ZGY 工区上下文，封装几何查询与井旁道提取。"""

    seismic_file: Path
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    n_ilines: int
    n_xlines: int
    annotstart_il: float
    annotstart_xl: float
    annotinc_il: float
    annotinc_xl: float
    x0: float
    y0: float
    dx_il: float
    dy_il: float
    dx_xl: float
    dy_xl: float

    @classmethod
    def from_file(cls, seismic_file: Path) -> "ZgySurveyContext":
        import pyzgy

        with pyzgy.open(str(seismic_file), mode="r") as reader:
            return cls(
                seismic_file=Path(seismic_file),
                ilines=np.asarray(reader.ilines, dtype=np.float64),
                xlines=np.asarray(reader.xlines, dtype=np.float64),
                samples=np.asarray(reader.samples, dtype=np.float64),
                n_ilines=int(reader.n_ilines),
                n_xlines=int(reader.n_xlines),
                annotstart_il=float(reader.annotstart[0]),
                annotstart_xl=float(reader.annotstart[1]),
                annotinc_il=float(reader.annotinc[0]),
                annotinc_xl=float(reader.annotinc[1]),
                x0=float(reader.corners[0][0]),
                y0=float(reader.corners[0][1]),
                dx_il=float(reader.easting_inc_il),
                dy_il=float(reader.northing_inc_il),
                dx_xl=float(reader.easting_inc_xl),
                dy_xl=float(reader.northing_inc_xl),
            )

    def query_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]:
        domain_value = _normalize_domain(domain)
        inline_stats = _axis_stats(self.ilines)
        xline_stats = _axis_stats(self.xlines)
        sample_stats = _axis_stats(self.samples)

        sample_unit = "ms" if domain_value == "time" else "m"
        return {
            "inline_min": inline_stats["min"],
            "inline_max": inline_stats["max"],
            "inline_step": inline_stats["step"],
            "xline_min": xline_stats["min"],
            "xline_max": xline_stats["max"],
            "xline_step": xline_stats["step"],
            "sample_min": sample_stats["min"],
            "sample_max": sample_stats["max"],
            "sample_step": sample_stats["step"],
            "sample_domain": domain_value,
            "sample_unit": sample_unit,
        }

    def coord_to_index(self, x: float, y: float) -> Tuple[float, float]:
        det = self.dx_il * self.dy_xl - self.dy_il * self.dx_xl
        if abs(det) < 1e-10:
            raise ValueError("Coordinate system is degenerate (determinant is zero)")

        dx = x - self.x0
        dy = y - self.y0
        i = (dx * self.dy_xl - dy * self.dx_xl) / det
        j = (dy * self.dx_il - dx * self.dy_il) / det

        if not (0 <= i <= self.n_ilines - 1):
            raise ValueError(f"Point is outside ZGY inline range: {i}")
        if not (0 <= j <= self.n_xlines - 1):
            raise ValueError(f"Point is outside ZGY crossline range: {j}")
        return i, j

    def coord_to_line(self, x: float, y: float) -> Tuple[float, float]:
        i, j = self.coord_to_index(x, y)
        il_no = self.annotstart_il + i * self.annotinc_il
        xl_no = self.annotstart_xl + j * self.annotinc_xl
        return il_no, xl_no

    def line_to_coord(self, il_no: float, xl_no: float) -> Tuple[float, float]:
        i = (il_no - self.annotstart_il) / self.annotinc_il
        j = (xl_no - self.annotstart_xl) / self.annotinc_xl
        if not (0 <= i <= self.n_ilines - 1):
            raise ValueError(f"Inline is outside ZGY range: {il_no}")
        if not (0 <= j <= self.n_xlines - 1):
            raise ValueError(f"Crossline is outside ZGY range: {xl_no}")

        x = self.x0 + i * self.dx_il + j * self.dx_xl
        y = self.y0 + i * self.dy_il + j * self.dy_xl
        return x, y

    def import_seismic_at_well(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic:
        import pyzgy

        domain_value = _normalize_domain(domain)

        i, j = self.coord_to_index(well_x, well_y)
        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))

        if not (0 <= i_floor < self.n_ilines and 0 <= i_ceil < self.n_ilines):
            raise ValueError(f"Well is outside seismic inline range: {i_floor}, {i_ceil}")
        if not (0 <= j_floor < self.n_xlines and 0 <= j_ceil < self.n_xlines):
            raise ValueError(f"Well is outside seismic crossline range: {j_floor}, {j_ceil}")

        raw_samples = self.samples
        sample_idx_start, sample_idx_end = _resolve_sample_window(raw_samples, sample_start, sample_end)

        with pyzgy.open(str(self.seismic_file), mode="r") as reader:
            t00 = reader.get_trace(i_floor * self.n_xlines + j_floor)[sample_idx_start:sample_idx_end]
            t01 = reader.get_trace(i_floor * self.n_xlines + j_ceil)[sample_idx_start:sample_idx_end]
            t10 = reader.get_trace(i_ceil * self.n_xlines + j_floor)[sample_idx_start:sample_idx_end]
            t11 = reader.get_trace(i_ceil * self.n_xlines + j_ceil)[sample_idx_start:sample_idx_end]

        trace_data = _interpolate_trace_from_4_neighbors(i, j, t00, t01, t10, t11)

        trace_axis = raw_samples[sample_idx_start:sample_idx_end]
        if domain_value == "time":
            trace_axis = trace_axis / 1000.0

        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        return grid.Seismic(values=trace_data, basis=trace_axis, basis_type=basis_type, name=trace_name)


def _resolve_context(
    seismic_file: Path,
    seismic_type: str,
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> SurveyContext:
    return open_survey(
        seismic_file,
        seismic_type=seismic_type,
        segy_options=_build_segy_options(iline=iline, xline=xline, istep=istep, xstep=xstep),
    )


def _build_segy_options(
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> Dict[str, int]:
    options: Dict[str, int] = {}
    if iline is not None:
        options["iline"] = int(iline)
    if xline is not None:
        options["xline"] = int(xline)
    if istep is not None:
        options["istep"] = int(istep)
    if xstep is not None:
        options["xstep"] = int(xstep)
    return options


def open_survey(
    seismic_file: Path,
    seismic_type: str = "segy",
    *,
    segy_options: Optional[Dict[str, int]] = None,
) -> SurveyContext:
    """打开地震工区并返回可复用的上下文对象。

    建议批量调用场景优先使用本函数：先打开一次，再复用上下文执行多次查询。
    """
    seismic_type_lower = seismic_type.lower()
    if seismic_type_lower == "segy":
        options = dict(segy_options or {})
        unsupported = set(options) - {"iline", "xline", "istep", "xstep"}
        if unsupported:
            unsupported_keys = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported SEG-Y options: {unsupported_keys}")
        return SegySurveyContext.from_file(
            seismic_file,
            iline=options.get("iline"),
            xline=options.get("xline"),
            istep=options.get("istep"),
            xstep=options.get("xstep"),
        )
    if seismic_type_lower == "zgy":
        if segy_options:
            raise ValueError("segy_options is only valid when seismic_type='segy'.")
        return ZgySurveyContext.from_file(seismic_file)
    raise ValueError(f"Unsupported seismic_type: {seismic_type}. Expect 'segy' or 'zgy'.")


def query_seismic_geometry(
    seismic_file: Path,
    seismic_type: str = "segy",
    domain: Optional[str] = "time",
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> Dict[str, Any]:
    """查询地震几何信息（工区范围与采样轴）。

    便捷函数：批量查询建议先调用 open_survey 后复用上下文。
    """
    ctx = _resolve_context(
        seismic_file,
        seismic_type,
        iline=iline,
        xline=xline,
        istep=istep,
        xstep=xstep,
    )
    return ctx.query_geometry(domain=domain)


def import_seismic_at_well(
    seismic_file: Path,
    well_x: float,
    well_y: float,
    sample_start: Optional[float] = None,
    sample_end: Optional[float] = None,
    domain: str = "time",
    seismic_type: str = "segy",
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> grid.Seismic:
    """提取井位处的复合地震道（双线性插值）。

    便捷函数：批量井提取建议先调用 open_survey 后复用上下文。
    """
    ctx = _resolve_context(
        seismic_file,
        seismic_type,
        iline=iline,
        xline=xline,
        istep=istep,
        xstep=xstep,
    )
    return ctx.import_seismic_at_well(
        well_x=well_x,
        well_y=well_y,
        sample_start=sample_start,
        sample_end=sample_end,
        domain=domain,
    )


def coord_to_line(
    seismic_file: Path,
    x: float,
    y: float,
    seismic_type: str = "segy",
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> Tuple[float, float]:
    """将 XY 坐标转换为 (inline_no, crossline_no)。

    便捷函数：批量坐标转换建议先调用 open_survey 后复用上下文。
    """
    ctx = _resolve_context(
        seismic_file,
        seismic_type,
        iline=iline,
        xline=xline,
        istep=istep,
        xstep=xstep,
    )
    return ctx.coord_to_line(x, y)


def line_to_coord(
    seismic_file: Path,
    il_no: float,
    xl_no: float,
    seismic_type: str = "segy",
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> Tuple[float, float]:
    """将 (inline_no, crossline_no) 转换为 XY 坐标。

    便捷函数：批量坐标转换建议先调用 open_survey 后复用上下文。
    """
    ctx = _resolve_context(
        seismic_file,
        seismic_type,
        iline=iline,
        xline=xline,
        istep=istep,
        xstep=xstep,
    )
    return ctx.line_to_coord(il_no, xl_no)
