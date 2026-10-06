"""cup.seismic.survey: SEG-Y/ZGY/NPZ 地震体 Adapter。

本模块提供地震体文件的统一打开入口，负责 SEG-Y/ZGY/NPZ 元数据读取、
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
4. NpzSurveyContext: 不复制二维横向数据的显式 NPZ Adapter。
5. open_survey: 根据文件类型打开地震体。
6. segy_options_from_config: 从配置段构建 SEG-Y 读取参数。
7. import_seismic: 读取完整的三维地震数组。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from cup.seismic.geometry import LineAxis, SampleAxis, SurveyLineGeometry
from wtie.processing import grid


def import_seismic(
    seismic_file: Path,
    seismic_type: str = "segy",
    iline: int | None = None,
    xline: int | None = None,
    istep: int | None = None,
    xstep: int | None = None,
) -> np.ndarray:
    """Read a complete SEG-Y, ZGY, or NPZ volume as ``[inline, xline, sample]``."""
    seismic_type_lower = str(seismic_type).lower()
    if seismic_type_lower == "segy":
        import cigsegy

        segy_kwargs = {}
        if iline is not None:
            segy_kwargs["iline"] = int(iline)
        if xline is not None:
            segy_kwargs["xline"] = int(xline)
        if istep is not None:
            segy_kwargs["istep"] = int(istep)
        if xstep is not None:
            segy_kwargs["xstep"] = int(xstep)
        volume = cigsegy.fromfile(str(seismic_file), **segy_kwargs)
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Only 3D post-stack SEG-Y is supported, got ndim={volume.ndim}")
        return volume

    if seismic_type_lower == "zgy":
        from pyzgy.read import SeismicReader

        with SeismicReader(str(seismic_file)) as reader:
            volume = reader.read_volume()
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Only 3D ZGY volume is supported, got ndim={volume.ndim}")
        return volume

    if seismic_type_lower == "npz":
        context = NpzSurveyContext.from_file(Path(seismic_file))
        return np.asarray(context.seismic, dtype=np.float32).copy()

    raise ValueError(f"Unsupported seismic_type: {seismic_type}. Expect 'segy', 'zgy', or 'npz'.")


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
    return "tvdss"


_NPZ_SURVEY_KEYS = frozenset(
    {
        "seismic",
        "sample_values",
        "sample_domain",
        "sample_unit",
        "depth_basis",
        "ilines",
        "xlines",
        "origin_xy_m",
        "inline_step_xy_m",
        "xline_step_xy_m",
    }
)


def _npz_scalar_text(data: Any, *, name: str, allow_empty: bool = False) -> str:
    """Read one scalar UTF string from an NPZ field without pickle."""

    value = np.asarray(data)
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"NPZ {name} must be a scalar string without pickle.")
    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    text = str(scalar).strip()
    if not text and not allow_empty:
        raise ValueError(f"NPZ {name} must be a non-empty string.")
    return text


def _regular_axis_step(axis: np.ndarray, *, name: str) -> float:
    if axis.ndim != 1 or axis.size == 0 or np.any(~np.isfinite(axis)):
        raise ValueError(f"NPZ {name} must be a non-empty finite one-dimensional axis.")
    if axis.size == 1:
        return 0.0
    differences = np.diff(axis)
    if np.any(differences <= 0.0):
        raise ValueError(f"NPZ {name} must be strictly increasing.")
    step = float(differences[0])
    if not np.allclose(differences, step, rtol=1.0e-6, atol=max(1.0e-12, abs(step) * 1.0e-6)):
        raise ValueError(f"NPZ {name} must be regularly sampled.")
    return step


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

        flat_items = sorted(
            (self.trace_flat_index(*key), key)
            for key in unique_indices
        )
        runs: list[list[tuple[int, tuple[int, int]]]] = []
        for item in flat_items:
            if not runs or item[0] != runs[-1][-1][0] + 1:
                runs.append([item])
            else:
                runs[-1].append(item)

        out: dict[tuple[int, int], grid.Seismic] = {}
        segy = cigsegy.Pysegy(str(self.seismic_file))
        try:
            for run in runs:
                block = np.asarray(
                    segy.collect(
                        run[0][0],
                        run[-1][0] + 1,
                        sample_idx_start,
                        sample_idx_end,
                    ),
                    dtype=np.float64,
                ).reshape((len(run), trace_axis.size))
                for row, (_flat_idx, key) in enumerate(run):
                    out[key] = grid.Seismic(
                        block[row],
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


@dataclass(frozen=True)
class NpzSurveyContext:
    """Explicit-array survey adapter used for synthetic and 2-D inputs.

    The NPZ contract stores the volume exactly as ``[inline, xline, sample]``.
    A two-dimensional line is represented by a singleton spatial axis; no
    traces are copied to fabricate a second direction.  The XY basis vectors
    remain explicit so a singleton axis still has a non-degenerate coordinate
    transform for line-position lookup.
    """

    seismic_file: Path
    seismic: np.ndarray
    _sample_values: np.ndarray
    _sample_domain: str
    _sample_unit: str
    _depth_basis: str | None
    line_geometry: SurveyLineGeometry

    @classmethod
    def from_file(cls, seismic_file: Path) -> "NpzSurveyContext":
        path = Path(seismic_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != _NPZ_SURVEY_KEYS:
                    missing = sorted(_NPZ_SURVEY_KEYS - set(data.files))
                    extra = sorted(set(data.files) - _NPZ_SURVEY_KEYS)
                    raise ValueError(
                        f"NPZ survey keys do not match the frozen contract; missing={missing}, extra={extra}."
                    )
                seismic = np.asarray(data["seismic"])
                sample_values = np.asarray(data["sample_values"])
                sample_domain = _npz_scalar_text(data["sample_domain"], name="sample_domain").casefold()
                sample_unit = _npz_scalar_text(data["sample_unit"], name="sample_unit").casefold()
                depth_basis_text = _npz_scalar_text(
                    data["depth_basis"], name="depth_basis", allow_empty=True
                )
                ilines = np.asarray(data["ilines"])
                xlines = np.asarray(data["xlines"])
                origin_xy_m = np.asarray(data["origin_xy_m"])
                inline_step_xy_m = np.asarray(data["inline_step_xy_m"])
                xline_step_xy_m = np.asarray(data["xline_step_xy_m"])
        except ValueError as exc:
            if "Object arrays cannot be loaded" in str(exc):
                raise ValueError("NPZ survey cannot contain object arrays; pickle loading is disabled.") from exc
            raise
        except Exception as exc:
            raise ValueError(f"Failed to read NPZ survey contract: {path}") from exc

        if seismic.dtype != np.dtype("float32") or seismic.ndim != 3:
            raise ValueError("NPZ seismic must be float32 with shape [inline, xline, sample].")
        if np.any(np.isinf(seismic)):
            raise ValueError("NPZ seismic must not contain infinite values.")
        for name, value in (
            ("sample_values", sample_values),
            ("ilines", ilines),
            ("xlines", xlines),
            ("origin_xy_m", origin_xy_m),
            ("inline_step_xy_m", inline_step_xy_m),
            ("xline_step_xy_m", xline_step_xy_m),
        ):
            if value.dtype != np.dtype("float64"):
                raise ValueError(f"NPZ {name} must have dtype float64.")
        if sample_domain not in {"time", "depth"}:
            raise ValueError("NPZ sample_domain must be 'time' or 'depth'.")
        expected_unit = "s" if sample_domain == "time" else "m"
        if sample_unit != expected_unit:
            raise ValueError(
                f"NPZ sample_unit must be {expected_unit!r} for sample_domain={sample_domain!r}."
            )
        if sample_domain == "time":
            if depth_basis_text:
                raise ValueError("NPZ time survey must have an empty depth_basis.")
            depth_basis = None
        else:
            if depth_basis_text.casefold() != "tvdss":
                raise ValueError("NPZ depth survey must have depth_basis='tvdss'.")
            depth_basis = "tvdss"

        if any(value.ndim != 1 for value in (sample_values, ilines, xlines)):
            raise ValueError("NPZ sample_values, ilines, and xlines must be one-dimensional arrays.")
        sample_values = np.asarray(sample_values, dtype=np.float64)
        ilines = np.asarray(ilines, dtype=np.float64)
        xlines = np.asarray(xlines, dtype=np.float64)
        _regular_axis_step(sample_values, name="sample_values")
        # A singleton line has no observed line-number difference.  Keep a
        # positive nominal step for downstream axis/grid contracts; the real
        # XY spacing remains exclusively in the explicit XY basis vectors.
        inline_step = 1.0 if ilines.size == 1 else _regular_axis_step(ilines, name="ilines")
        xline_step = 1.0 if xlines.size == 1 else _regular_axis_step(xlines, name="xlines")
        if seismic.shape != (ilines.size, xlines.size, sample_values.size):
            raise ValueError(
                "NPZ seismic shape must match ilines/xlines/sample_values: "
                f"got {seismic.shape}, expected {(ilines.size, xlines.size, sample_values.size)}."
            )
        for name, value in (
            ("origin_xy_m", origin_xy_m),
            ("inline_step_xy_m", inline_step_xy_m),
            ("xline_step_xy_m", xline_step_xy_m),
        ):
            if value.shape != (2,) or np.any(~np.isfinite(value)):
                raise ValueError(f"NPZ {name} must be a finite float64 vector of shape (2,).")
        determinant = float(
            inline_step_xy_m[0] * xline_step_xy_m[1]
            - inline_step_xy_m[1] * xline_step_xy_m[0]
        )
        if not np.isfinite(determinant) or abs(determinant) <= 1.0e-12:
            raise ValueError("NPZ XY basis vectors must define a non-degenerate coordinate transform.")

        sample_axis = SampleAxis(
            values=sample_values,
            domain=sample_domain,
            unit=sample_unit,
            depth_basis=depth_basis,
        )
        geometry = SurveyLineGeometry(
            inline_axis=LineAxis(
                minimum=float(ilines[0]),
                step=inline_step,
                count=int(ilines.size),
                name="inline",
            ),
            xline_axis=LineAxis(
                minimum=float(xlines[0]),
                step=xline_step,
                count=int(xlines.size),
                name="xline",
            ),
            x0=float(origin_xy_m[0]),
            y0=float(origin_xy_m[1]),
            dx_inline=float(inline_step_xy_m[0]),
            dy_inline=float(inline_step_xy_m[1]),
            dx_xline=float(xline_step_xy_m[0]),
            dy_xline=float(xline_step_xy_m[1]),
        )
        # Store independent copies so callers cannot mutate the loaded arrays
        # through an external view after the contract has been validated.
        seismic = np.asarray(seismic, dtype=np.float32).copy()
        seismic.setflags(write=False)
        sample_values = sample_values.copy()
        sample_values.setflags(write=False)
        return cls(
            seismic_file=path,
            seismic=seismic,
            _sample_values=sample_values,
            _sample_domain=sample_domain,
            _sample_unit=sample_unit,
            _depth_basis=depth_basis,
            line_geometry=geometry,
        )

    def sample_axis(self, domain: Optional[str] = "time") -> SampleAxis:
        domain_value = _normalize_domain(domain)
        if domain_value != self._sample_domain:
            raise ValueError(
                f"NPZ survey stores sample_domain={self._sample_domain!r}, not {domain_value!r}."
            )
        return SampleAxis(
            values=self._sample_values,
            domain=self._sample_domain,
            unit=self._sample_unit,
            depth_basis=self._depth_basis,
        )

    def describe_geometry(self, domain: Optional[str] = "time") -> Dict[str, Any]:
        return self.line_geometry.describe(sample_axis=self.sample_axis(domain))

    def _interpolated_trace(self, inline: float, xline: float) -> np.ndarray:
        i, j = self.line_geometry.line_to_index(inline, xline)
        i_floor = int(np.floor(i))
        j_floor = int(np.floor(j))
        i_ceil = min(i_floor + 1, self.seismic.shape[0] - 1)
        j_ceil = min(j_floor + 1, self.seismic.shape[1] - 1)
        wi = float(i - i_floor)
        wj = float(j - j_floor)
        trace00 = self.seismic[i_floor, j_floor].astype(np.float64, copy=False)
        trace01 = self.seismic[i_floor, j_ceil].astype(np.float64, copy=False)
        trace10 = self.seismic[i_ceil, j_floor].astype(np.float64, copy=False)
        trace11 = self.seismic[i_ceil, j_ceil].astype(np.float64, copy=False)
        return (
            (1.0 - wi) * (1.0 - wj) * trace00
            + (1.0 - wi) * wj * trace01
            + wi * (1.0 - wj) * trace10
            + wi * wj * trace11
        )

    def read_trace_at_xy(
        self,
        well_x: float,
        well_y: float,
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> grid.Seismic:
        domain_value = _normalize_domain(domain)
        i, j = self.line_geometry.coord_to_index(float(well_x), float(well_y))
        trace = self._interpolated_trace(
            self.line_geometry.inline_axis.line_at_index(i),
            self.line_geometry.xline_axis.line_at_index(j),
        )
        sample_axis = self.sample_axis(domain_value)
        sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)
        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        return grid.Seismic(
            values=trace[sample_idx_start:sample_idx_end],
            basis=sample_axis.values[sample_idx_start:sample_idx_end],
            basis_type=basis_type,
            name=trace_name,
        )

    def trace_flat_index(self, inline_index: int, xline_index: int) -> int:
        i = int(inline_index)
        j = int(xline_index)
        if not (0 <= i < self.seismic.shape[0] and 0 <= j < self.seismic.shape[1]):
            raise ValueError(f"Trace indices are outside survey range: {(i, j)}")
        return i * int(self.seismic.shape[1]) + j

    def read_traces_at_indices(
        self,
        indices: list[tuple[int, int]],
        sample_start: Optional[float] = None,
        sample_end: Optional[float] = None,
        domain: str = "time",
    ) -> dict[tuple[int, int], grid.Seismic]:
        domain_value = _normalize_domain(domain)
        sample_axis = self.sample_axis(domain_value)
        sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)
        trace_axis = sample_axis.values[sample_idx_start:sample_idx_end]
        basis_type = _domain_to_basis_type(domain_value)
        trace_name = "Seismic Trace" if basis_type == "twt" else "Seismic Trace (Depth)"
        output: dict[tuple[int, int], grid.Seismic] = {}
        for raw_i, raw_j in indices:
            key = (int(raw_i), int(raw_j))
            self.trace_flat_index(*key)
            output[key] = grid.Seismic(
                values=self.seismic[key[0], key[1], sample_idx_start:sample_idx_end].astype(
                    np.float64, copy=False
                ),
                basis=trace_axis,
                basis_type=basis_type,
                name=trace_name,
            )
        return output


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
    if seismic_type_lower == "npz":
        if segy_options:
            raise ValueError("segy_options is not valid when seismic_type='npz'.")
        return NpzSurveyContext.from_file(Path(seismic_file))
    raise ValueError(f"Unsupported seismic_type: {seismic_type}. Expect 'segy', 'zgy', or 'npz'.")
