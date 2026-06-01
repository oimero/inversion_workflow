"""cup.well.trajectory: 井轨迹模型与 Petrel 轨迹解析。

本模块持有项目级的井轨迹表示，将 Petrel 轨迹解析、TVDSS 口径换算
与 wtie 适配逻辑集中于此，供斜井路径各步复用同一几何事实。

边界说明
--------
- 本模块不负责井型判定（直/斜），判定逻辑在 ``cup.well.assets`` 与
  ``well_trajectory.py`` 中。
- TVDSS 换算约定在本模块内唯一实现，其他模块不应独立散写。

核心公开对象
------------
1. WellTrajectory: 井轨迹完整表示（MD、TVD、XY、Z、井斜、方位）。
2. WellSpatialSampleSet / sample_trajectory_on_twt: 在 TWT 轴上采样轨迹位置。
3. trajectory_summary: 轨迹几何摘要。
4. z_tvd_residual_m: Z 与 TVD 一致性残差。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from wtie.processing import grid

SPATIAL_SAMPLE_COLUMNS = [
    "well_name",
    "sample_index",
    "twt_s",
    "md_m",
    "tvd_kb_m",
    "tvdss_m",
    "z_m",
    "x_m",
    "y_m",
]

_HEADER_PATTERNS = {
    "well_name": re.compile(r"^#\s*WELL NAME:\s*(?P<value>.+?)\s*$", re.IGNORECASE),
    "well_head_x_m": re.compile(r"^#\s*WELL HEAD X-COORDINATE:\s*(?P<value>[-+0-9.eE]+)", re.IGNORECASE),
    "well_head_y_m": re.compile(r"^#\s*WELL HEAD Y-COORDINATE:\s*(?P<value>[-+0-9.eE]+)", re.IGNORECASE),
    "kb_m": re.compile(r"^#\s*WELL DATUM.*?:\s*(?P<value>[-+0-9.eE]+)", re.IGNORECASE),
}


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _read_petrel_header(path: Path) -> tuple[dict[str, Any], list[str], int]:
    metadata: dict[str, Any] = {}
    columns: list[str] | None = None
    column_line_index: int | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line_index, line in enumerate(fp):
            stripped = line.strip()
            if stripped.startswith("#"):
                for key, pattern in _HEADER_PATTERNS.items():
                    match = pattern.match(stripped)
                    if not match:
                        continue
                    raw_value = match.group("value").strip()
                    metadata[key] = raw_value if key == "well_name" else _optional_float(raw_value)
                continue

            parts = stripped.split()
            if parts and parts[0].casefold() == "md":
                columns = parts
                column_line_index = line_index
                break

    if columns is None or column_line_index is None:
        raise ValueError(f"Petrel trajectory file has no column header: {path}")
    return metadata, columns, column_line_index


def _column_lookup(columns: list[str]) -> dict[str, int]:
    return {column.strip().casefold(): index for index, column in enumerate(columns)}


@dataclass(frozen=True)
class WellSpatialSampleSet:
    """A well trajectory sampled on a workflow axis."""

    well_name: str
    rows: pd.DataFrame

    @property
    def sample_count(self) -> int:
        return int(len(self.rows))

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()


@dataclass(frozen=True)
class WellTrajectory:
    """项目级井轨迹对象，包含 MD、TVDSS 与 XY 几何信息。"""

    well_name: str
    header_well_name: str | None
    source_file: Path
    md_m: np.ndarray
    tvd_kb_m: np.ndarray
    tvdss_m: np.ndarray
    z_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    dx_m: np.ndarray
    dy_m: np.ndarray
    azim_deg: np.ndarray
    incl_deg: np.ndarray
    dls: np.ndarray
    kb_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_petrel_trace(cls, path: str | Path) -> "WellTrajectory":
        """读取 Petrel 导出的井轨迹文本。

        当前项目约定为 ``tvdss_m = tvd_kb_m - kb_m``。必需列为
        ``MD``、``X``、``Y``、``Z`` 与 ``TVD``；可选列缺失时填 NaN。
        """
        source_file = Path(path).resolve()
        metadata, columns, column_line_index = _read_petrel_header(source_file)
        lookup = _column_lookup(columns)

        required = {"md", "x", "y", "z", "tvd"}
        missing = sorted(required - set(lookup))
        if missing:
            raise ValueError(f"Petrel trajectory file {source_file} is missing required columns: {missing}")

        try:
            data = np.loadtxt(source_file, comments="#", skiprows=column_line_index + 1, ndmin=2)
        except Exception as exc:  # np.loadtxt raises several parsing-specific exception types.
            raise ValueError(f"Failed to parse Petrel trajectory numeric data from {source_file}: {exc}") from exc

        if data.size == 0:
            raise ValueError(f"Petrel trajectory file has no numeric rows: {source_file}")
        if data.shape[1] < len(columns):
            raise ValueError(
                f"Petrel trajectory file {source_file} has {data.shape[1]} numeric columns, "
                f"but header declares {len(columns)}."
            )

        def col(name: str, default: float = np.nan) -> np.ndarray:
            index = lookup.get(name.casefold())
            if index is None:
                return np.full(data.shape[0], default, dtype=np.float64)
            return np.asarray(data[:, index], dtype=np.float64)

        md_m = col("md")
        x_m = col("x")
        y_m = col("y")
        z_m = col("z")
        tvd_kb_m = col("tvd")
        valid_required = (
            np.isfinite(md_m) & np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(z_m) & np.isfinite(tvd_kb_m)
        )
        valid_count = int(np.count_nonzero(valid_required))
        if valid_count < 2:
            raise ValueError(f"Petrel trajectory file has fewer than 2 valid trajectory points: {source_file}")

        invalid_required_row_count = int(md_m.size - valid_count)
        md_m = md_m[valid_required]
        x_m = x_m[valid_required]
        y_m = y_m[valid_required]
        z_m = z_m[valid_required]
        tvd_kb_m = tvd_kb_m[valid_required]
        dx_m = col("dx")[valid_required]
        dy_m = col("dy")[valid_required]
        azim_deg = col("azim")[valid_required]
        incl_deg = col("incl")[valid_required]
        dls = col("dls")[valid_required]

        if not np.all(np.diff(md_m) > 0.0):
            raise ValueError(f"Petrel trajectory MD is not strictly increasing: {source_file}")
        if not (np.any(np.isfinite(x_m)) and np.any(np.isfinite(y_m))):
            raise ValueError(f"Petrel trajectory XY values are all invalid: {source_file}")

        kb_m = _optional_float(metadata.get("kb_m"))
        if kb_m is None:
            raise ValueError(f"Petrel trajectory header is missing a valid KB datum: {source_file}")

        tvdss_m = tvd_kb_m - float(kb_m)
        well_name = str(metadata.get("well_name") or source_file.stem).strip()
        metadata = dict(metadata)
        metadata.update(
            {
                "columns": columns,
                "valid_point_count": int(md_m.size),
                "invalid_required_row_count": invalid_required_row_count,
            }
        )

        return cls(
            well_name=well_name,
            header_well_name=str(metadata.get("well_name")).strip() if metadata.get("well_name") is not None else None,
            source_file=source_file,
            md_m=md_m,
            tvd_kb_m=tvd_kb_m,
            tvdss_m=tvdss_m,
            z_m=z_m,
            x_m=x_m,
            y_m=y_m,
            dx_m=dx_m,
            dy_m=dy_m,
            azim_deg=azim_deg,
            incl_deg=incl_deg,
            dls=dls,
            kb_m=float(kb_m),
            metadata=metadata,
        )

    @property
    def point_count(self) -> int:
        return int(self.md_m.size)

    def to_wtie_wellpath(self) -> grid.WellPath:
        """返回用于 wtie MD/TVDSS 转换的 WellPath 适配对象。"""
        return grid.WellPath(md=self.md_m, tvdss=self.tvdss_m, kb=self.kb_m)

    def with_well_name(self, well_name: str) -> "WellTrajectory":
        """返回替换显示井名后的副本。"""
        return replace(self, well_name=str(well_name).strip())

    def position_at_md(self, md_m: float) -> dict[str, float]:
        """按米制 MD 插值得到轨迹位置。"""
        md = float(md_m)
        if md < float(self.md_m[0]) or md > float(self.md_m[-1]):
            raise ValueError(f"MD {md} is outside trajectory range [{self.md_m[0]}, {self.md_m[-1]}].")
        return {
            "md_m": md,
            "x_m": float(np.interp(md, self.md_m, self.x_m)),
            "y_m": float(np.interp(md, self.md_m, self.y_m)),
            "tvd_kb_m": float(np.interp(md, self.md_m, self.tvd_kb_m)),
            "tvdss_m": float(np.interp(md, self.md_m, self.tvdss_m)),
            "z_m": float(np.interp(md, self.md_m, self.z_m)),
        }

    def representative_position(self, policy: str = "surface") -> dict[str, float]:
        """返回一组代表性轨迹位置。

        支持策略为 ``surface``、``bottom`` 与 ``max_offset``。
        """
        policy_key = policy.strip().casefold()
        if policy_key == "surface":
            index = 0
        elif policy_key == "bottom":
            index = self.point_count - 1
        elif policy_key == "max_offset":
            offsets = np.hypot(self.x_m - self.x_m[0], self.y_m - self.y_m[0])
            index = int(np.nanargmax(offsets))
        else:
            raise ValueError(f"Unsupported representative position policy: {policy}")
        return self._position_at_index(index)

    def _position_at_index(self, index: int) -> dict[str, float]:
        return {
            "md_m": float(self.md_m[index]),
            "x_m": float(self.x_m[index]),
            "y_m": float(self.y_m[index]),
            "tvd_kb_m": float(self.tvd_kb_m[index]),
            "tvdss_m": float(self.tvdss_m[index]),
            "z_m": float(self.z_m[index]),
        }


def _validate_twt_axis(twt_axis: np.ndarray) -> np.ndarray:
    twt = np.asarray(twt_axis, dtype=np.float64)
    if twt.ndim != 1 or twt.size == 0:
        raise ValueError("twt_axis must be a non-empty 1D array.")
    if not np.all(np.isfinite(twt)):
        raise ValueError("twt_axis contains non-finite values.")
    if np.any(np.diff(twt) <= 0.0):
        raise ValueError("twt_axis must be strictly increasing.")
    return twt


def _validate_md_table(table: grid.TimeDepthTable) -> None:
    if not table.is_md_domain:
        raise ValueError("sample_trajectory_on_twt expects an MD-domain TimeDepthTable.")
    if table.size < 2:
        raise ValueError("Time-depth table must contain at least two samples.")


def sample_trajectory_on_twt(
    trajectory: WellTrajectory,
    table: grid.TimeDepthTable,
    twt_axis: np.ndarray,
) -> WellSpatialSampleSet:
    """Sample a trajectory at TWT positions using an MD-domain time-depth table."""
    _validate_md_table(table)
    twt = _validate_twt_axis(twt_axis)

    table_twt = np.asarray(table.twt, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    if float(twt[0]) < float(table_twt[0]) or float(twt[-1]) > float(table_twt[-1]):
        raise ValueError(
            f"TWT axis [{twt[0]}, {twt[-1]}] is outside table range [{table_twt[0]}, {table_twt[-1]}]."
        )

    md = np.interp(twt, table_twt, table_md)
    if not np.all(np.isfinite(md)):
        raise ValueError("Sampled MD values contain non-finite values.")
    if float(np.nanmin(md)) < float(trajectory.md_m[0]) or float(np.nanmax(md)) > float(trajectory.md_m[-1]):
        raise ValueError(
            f"Sampled MD range [{np.nanmin(md)}, {np.nanmax(md)}] is outside trajectory range "
            f"[{trajectory.md_m[0]}, {trajectory.md_m[-1]}]."
        )

    rows: list[dict[str, Any]] = []
    for index, (twt_s, md_m) in enumerate(zip(twt, md)):
        position = trajectory.position_at_md(float(md_m))
        rows.append(
            {
                "well_name": trajectory.well_name,
                "sample_index": index,
                "twt_s": float(twt_s),
                "md_m": float(position["md_m"]),
                "tvd_kb_m": float(position["tvd_kb_m"]),
                "tvdss_m": float(position["tvdss_m"]),
                "z_m": float(position["z_m"]),
                "x_m": float(position["x_m"]),
                "y_m": float(position["y_m"]),
            }
        )
    return WellSpatialSampleSet(
        well_name=trajectory.well_name,
        rows=pd.DataFrame.from_records(rows, columns=SPATIAL_SAMPLE_COLUMNS),
    )


def z_tvd_residual_m(trajectory: WellTrajectory) -> np.ndarray:
    """返回米制 ``Z - (KB - TVD)`` 残差。"""
    return trajectory.z_m - (trajectory.kb_m - trajectory.tvd_kb_m)


def horizontal_offsets_m(trajectory: WellTrajectory) -> np.ndarray:
    """返回相对第一个轨迹点的水平偏移。"""
    return np.hypot(trajectory.x_m - trajectory.x_m[0], trajectory.y_m - trajectory.y_m[0])


def finite_max(values: np.ndarray) -> float | None:
    """返回数组中的最大有限值；若没有有限值则返回 ``None``。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def trajectory_summary(trajectory: WellTrajectory) -> Mapping[str, float | int | None]:
    """返回单条轨迹的核心数值摘要。"""
    offsets = horizontal_offsets_m(trajectory)
    return {
        "point_count": trajectory.point_count,
        "md_min_m": float(np.nanmin(trajectory.md_m)),
        "md_max_m": float(np.nanmax(trajectory.md_m)),
        "tvd_kb_min_m": float(np.nanmin(trajectory.tvd_kb_m)),
        "tvd_kb_max_m": float(np.nanmax(trajectory.tvd_kb_m)),
        "tvdss_min_m": float(np.nanmin(trajectory.tvdss_m)),
        "tvdss_max_m": float(np.nanmax(trajectory.tvdss_m)),
        "surface_x_m": float(trajectory.x_m[0]),
        "surface_y_m": float(trajectory.y_m[0]),
        "bottom_x_m": float(trajectory.x_m[-1]),
        "bottom_y_m": float(trajectory.y_m[-1]),
        "surface_to_bottom_offset_m": float(
            np.hypot(trajectory.x_m[-1] - trajectory.x_m[0], trajectory.y_m[-1] - trajectory.y_m[0])
        ),
        "max_horizontal_offset_m": float(np.nanmax(offsets)),
        "max_incl_deg": finite_max(trajectory.incl_deg),
        "max_dls": finite_max(trajectory.dls),
    }

