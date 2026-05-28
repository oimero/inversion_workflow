"""cup.seismic.survey: SEG-Y/ZGY 地震体 Adapter。

本模块提供地震体文件的统一打开入口，负责 SEG-Y/ZGY 元数据读取、
采样轴构造和井旁道双线性插值提取。inline/xline 与 XY 的几何计算由
``cup.seismic.geometry`` 承担。

边界说明
--------
- 本模块负责文件 Adapter，不承载通用几何数学。
- ``open_survey`` 是唯一公开工厂入口。
- ``domain='depth'`` 的支持依赖底层数据提供深度采样信息；该模块不做速度换算。

核心公开对象
------------
1. SurveyContext: 地震体 Adapter 协议。
2. SegySurveyContext: SEG-Y Adapter。
3. ZgySurveyContext: ZGY Adapter。
4. open_survey: 根据文件类型打开地震体。
5. segy_options_from_config: 从配置段构建 SEG-Y 读取参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from cup.seismic.geometry import LineAxis, SampleAxis, SurveyLineGeometry
from wtie.processing import grid


class SurveyContext(Protocol):
    """统一地震体 Adapter 接口。"""

    line_geometry: SurveyLineGeometry

    def sample_axis(self, domain: Optional[str] = "time") -> SampleAxis: ...

    def describe_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]: ...

    def read_trace_at_xy(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic: ...

    def trace_flat_index(self, inline_index: int, xline_index: int) -> int: ...

    def read_traces_at_indices(
        self,
        indices: list[tuple[int, int]],
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> dict[tuple[int, int], grid.Seismic]: ...


def _normalize_domain(domain: Optional[str]) -> str:
    """规范化采样域标识。"""
    if domain is None:
        return "time"
    domain_lower = domain.lower()
    if domain_lower not in {"time", "depth"}:
        raise ValueError(f"Unsupported domain: {domain}. Expect 'time' or 'depth'.")
    return domain_lower


def _domain_to_basis_type(domain: str) -> str:
    """将采样域映射为曲线基准类型。"""
    domain_lower = _normalize_domain(domain)
    if domain_lower == "time":
        return "twt"
    return "md"


def _interpolate_trace_from_4_neighbors(
    i: float,
    j: float,
    trace00: np.ndarray,
    trace01: np.ndarray,
    trace10: np.ndarray,
    trace11: np.ndarray,
) -> np.ndarray:
    """对四邻道执行双线性插值。"""
    i_floor = int(np.floor(i))
    j_floor = int(np.floor(j))
    wi = i - i_floor
    wj = j - j_floor
    return (1 - wi) * (1 - wj) * trace00 + (1 - wi) * wj * trace01 + wi * (1 - wj) * trace10 + wi * wj * trace11


def _segy_build_sample_axis(meta: Dict[str, Any], domain: str) -> SampleAxis:
    """根据 SEG-Y 元信息构建采样轴。"""
    domain_lower = _normalize_domain(domain)
    nt = int(meta["nt"])
    if domain_lower == "time":
        start_time_s = float(meta.get("start_time", 0.0)) / 1000.0
        dt_s = float(meta["dt"]) / 1_000_000.0
        values = start_time_s + np.arange(nt, dtype=np.float64) * dt_s
        return SampleAxis(values=values, domain=domain_lower, unit="s")

    start_depth = float(meta.get("start_depth", meta.get("start_time", 0.0)))
    dz = float(meta.get("dz", meta["dt"])) / 1000.0
    values = start_depth + np.arange(nt, dtype=np.float64) * dz
    return SampleAxis(values=values, domain=domain_lower, unit="m")


def _segy_pick_affine_reference_points_from_geom(
    geom: np.ndarray,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """在几何网格中选择仿射参考点。"""
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


def _segy_coord_scalar_to_factor(coord_scalar: float) -> float:
    """将 SEG-Y 坐标缩放因子转换为乘法系数。"""
    if coord_scalar == 0:
        return 1.0
    if coord_scalar > 0:
        return float(coord_scalar)
    return 1.0 / abs(float(coord_scalar))


@dataclass(frozen=True)
class SegySurveyContext:
    """SEG-Y 地震体 Adapter。"""

    seismic_file: Path
    meta: Dict[str, Any]
    geom: np.ndarray
    line_geometry: SurveyLineGeometry

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

            dx_inline = p1[0] - p0[0]
            dy_inline = p1[1] - p0[1]
            dx_xline = p2[0] - p0[0]
            dy_xline = p2[1] - p0[1]
            x_origin = p0[0] - i0 * dx_inline - j0 * dx_xline
            y_origin = p0[1] - i0 * dy_inline - j0 * dy_xline

            line_geometry = SurveyLineGeometry(
                inline_axis=LineAxis(
                    minimum=float(geominfo["iline"]["min_iline"]),
                    step=float(geominfo["iline"]["istep"]),
                    count=int(geom.shape[0]),
                    name="inline",
                ),
                xline_axis=LineAxis(
                    minimum=float(geominfo["xline"]["min_xline"]),
                    step=float(geominfo["xline"]["xstep"]),
                    count=int(geom.shape[1]),
                    name="xline",
                ),
                x0=float(x_origin),
                y0=float(y_origin),
                dx_inline=float(dx_inline),
                dy_inline=float(dy_inline),
                dx_xline=float(dx_xline),
                dy_xline=float(dy_xline),
            )

            return cls(
                seismic_file=Path(seismic_file),
                meta=meta,
                geom=geom,
                line_geometry=line_geometry,
            )
        finally:
            segy.close()

    def sample_axis(self, domain: Optional[str] = "time") -> SampleAxis:
        """返回指定采样域的采样轴。"""
        return _segy_build_sample_axis(self.meta, _normalize_domain(domain))

    def describe_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]:
        """返回历史几何字典格式。"""
        return self.line_geometry.describe(sample_axis=self.sample_axis(domain))

    def read_trace_at_xy(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic:
        """读取井位处四邻道双线性插值后的地震道。"""
        domain_value = _normalize_domain(domain)

        i, j = self.line_geometry.coord_to_index(well_x, well_y)
        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))

        ni, nx = self.geom.shape
        if not (0 <= i_floor < ni and 0 <= i_ceil < ni):
            raise ValueError(f"Well is outside seismic inline range: {i_floor}, {i_ceil}")
        if not (0 <= j_floor < nx and 0 <= j_ceil < nx):
            raise ValueError(f"Well is outside seismic xline range: {j_floor}, {j_ceil}")

        neighbor_indices = [
            int(self.geom[i_floor, j_floor]),
            int(self.geom[i_floor, j_ceil]),
            int(self.geom[i_ceil, j_floor]),
            int(self.geom[i_ceil, j_ceil]),
        ]
        if any(idx < 0 for idx in neighbor_indices):
            raise ValueError("Well neighborhood contains missing traces, cannot apply bilinear interpolation.")

        sample_axis = self.sample_axis(domain_value)
        sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)

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
        trace_axis = sample_axis.values[sample_idx_start:sample_idx_end]

        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        return grid.Seismic(values=trace_data, basis=trace_axis, basis_type=basis_type, name=trace_name)

    def trace_flat_index(self, inline_index: int, xline_index: int) -> int:
        """Return the underlying SEG-Y trace index for integer grid indices."""
        i = int(inline_index)
        j = int(xline_index)
        ni, nx = self.geom.shape
        if not (0 <= i < ni and 0 <= j < nx):
            raise ValueError(f"Trace indices are outside survey range: {(i, j)}")
        flat_idx = int(self.geom[i, j])
        if flat_idx < 0:
            raise ValueError(f"Trace indices reference a missing SEG-Y trace: {(i, j)}")
        return flat_idx

    def read_traces_at_indices(
        self,
        indices: list[tuple[int, int]],
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> dict[tuple[int, int], grid.Seismic]:
        """Read multiple traces by integer inline/xline indices."""
        domain_value = _normalize_domain(domain)
        unique_indices = sorted({(int(i), int(j)) for i, j in indices})
        if not unique_indices:
            return {}

        sample_axis = self.sample_axis(domain_value)
        sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)
        trace_axis = sample_axis.values[sample_idx_start:sample_idx_end]
        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"

        import cigsegy

        out: dict[tuple[int, int], grid.Seismic] = {}
        segy = cigsegy.Pysegy(str(self.seismic_file))
        try:
            for key in unique_indices:
                flat_idx = self.trace_flat_index(*key)
                values = segy.collect(flat_idx, flat_idx + 1, sample_idx_start, sample_idx_end).squeeze()
                out[key] = grid.Seismic(
                    np.atleast_1d(np.asarray(values, dtype=np.float64)),
                    trace_axis,
                    basis_type,
                    name=trace_name,
                )
        finally:
            segy.close()
        return out


@dataclass(frozen=True)
class ZgySurveyContext:
    """ZGY 地震体 Adapter。"""

    seismic_file: Path
    samples: np.ndarray
    n_ilines: int
    n_xlines: int
    line_geometry: SurveyLineGeometry

    @classmethod
    def from_file(cls, seismic_file: Path) -> "ZgySurveyContext":
        import pyzgy

        with pyzgy.open(str(seismic_file), mode="r") as reader:
            line_geometry = SurveyLineGeometry(
                inline_axis=LineAxis(
                    minimum=float(reader.annotstart[0]),
                    step=float(reader.annotinc[0]),
                    count=int(reader.n_ilines),
                    name="inline",
                ),
                xline_axis=LineAxis(
                    minimum=float(reader.annotstart[1]),
                    step=float(reader.annotinc[1]),
                    count=int(reader.n_xlines),
                    name="xline",
                ),
                x0=float(reader.corners[0][0]),
                y0=float(reader.corners[0][1]),
                dx_inline=float(reader.easting_inc_il),
                dy_inline=float(reader.northing_inc_il),
                dx_xline=float(reader.easting_inc_xl),
                dy_xline=float(reader.northing_inc_xl),
            )
            return cls(
                seismic_file=Path(seismic_file),
                samples=np.asarray(reader.samples, dtype=np.float64),
                n_ilines=int(reader.n_ilines),
                n_xlines=int(reader.n_xlines),
                line_geometry=line_geometry,
            )

    def sample_axis(self, domain: Optional[str] = "time") -> SampleAxis:
        """返回指定采样域的采样轴。"""
        domain_value = _normalize_domain(domain)
        if domain_value == "time":
            return SampleAxis(values=self.samples / 1000.0, domain=domain_value, unit="s")
        return SampleAxis(values=self.samples, domain=domain_value, unit="m")

    def describe_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]:
        """返回历史几何字典格式。"""
        return self.line_geometry.describe(sample_axis=self.sample_axis(domain))

    def read_trace_at_xy(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic:
        """读取井位处四邻道双线性插值后的地震道。"""
        import pyzgy

        domain_value = _normalize_domain(domain)
        i, j = self.line_geometry.coord_to_index(well_x, well_y)
        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))

        if not (0 <= i_floor < self.n_ilines and 0 <= i_ceil < self.n_ilines):
            raise ValueError(f"Well is outside seismic inline range: {i_floor}, {i_ceil}")
        if not (0 <= j_floor < self.n_xlines and 0 <= j_ceil < self.n_xlines):
            raise ValueError(f"Well is outside seismic xline range: {j_floor}, {j_ceil}")

        sample_axis = self.sample_axis(domain_value)
        sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)

        with pyzgy.open(str(self.seismic_file), mode="r") as reader:
            t00 = reader.get_trace(i_floor * self.n_xlines + j_floor)[sample_idx_start:sample_idx_end]
            t01 = reader.get_trace(i_floor * self.n_xlines + j_ceil)[sample_idx_start:sample_idx_end]
            t10 = reader.get_trace(i_ceil * self.n_xlines + j_floor)[sample_idx_start:sample_idx_end]
            t11 = reader.get_trace(i_ceil * self.n_xlines + j_ceil)[sample_idx_start:sample_idx_end]

        trace_data = _interpolate_trace_from_4_neighbors(i, j, t00, t01, t10, t11)
        trace_axis = sample_axis.values[sample_idx_start:sample_idx_end]

        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        return grid.Seismic(values=trace_data, basis=trace_axis, basis_type=basis_type, name=trace_name)

    def trace_flat_index(self, inline_index: int, xline_index: int) -> int:
        """Return the underlying ZGY trace index for integer grid indices."""
        i = int(inline_index)
        j = int(xline_index)
        if not (0 <= i < self.n_ilines and 0 <= j < self.n_xlines):
            raise ValueError(f"Trace indices are outside survey range: {(i, j)}")
        return i * self.n_xlines + j

    def read_traces_at_indices(
        self,
        indices: list[tuple[int, int]],
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> dict[tuple[int, int], grid.Seismic]:
        """Read multiple traces by integer inline/xline indices."""
        import pyzgy

        domain_value = _normalize_domain(domain)
        unique_indices = sorted({(int(i), int(j)) for i, j in indices})
        if not unique_indices:
            return {}

        sample_axis = self.sample_axis(domain_value)
        sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)
        trace_axis = sample_axis.values[sample_idx_start:sample_idx_end]
        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"

        out: dict[tuple[int, int], grid.Seismic] = {}
        with pyzgy.open(str(self.seismic_file), mode="r") as reader:
            for key in unique_indices:
                flat_idx = self.trace_flat_index(*key)
                values = reader.get_trace(flat_idx)[sample_idx_start:sample_idx_end]
                out[key] = grid.Seismic(
                    np.asarray(values, dtype=np.float64),
                    trace_axis,
                    basis_type,
                    name=trace_name,
                )
        return out


def segy_options_from_config(seismic_cfg: dict[str, Any]) -> dict[str, int]:
    """从配置段构建 SEG-Y 读取参数字典。

    将 ``iline``、``xline``、``istep``、``xstep``、``iline_byte``、
    ``xline_byte`` 映射为底层读取器需要的整数参数名。
    """
    mapping = {
        "iline": "iline",
        "xline": "xline",
        "istep": "istep",
        "xstep": "xstep",
        "iline_byte": "iline",
        "xline_byte": "xline",
    }
    options: dict[str, int] = {}
    for key, target in mapping.items():
        value = seismic_cfg.get(key)
        if value is not None:
            options[target] = int(value)
    return options


def open_survey(
    seismic_file: Path,
    seismic_type: str = "segy",
    *,
    segy_options: Optional[Dict[str, int]] = None,
) -> SurveyContext:
    """打开地震体文件并返回可复用 Adapter。"""
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
