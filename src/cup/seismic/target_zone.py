"""cup.seismic.target_zone: 多层位目标层段构建与 QC。

本模块以原始层位拾取为输入，构建插值支撑掩码、trace 级 QC 掩码，
并通过厚度插值重建全覆盖层位网格。

边界说明
--------
- 不负责反演优化、训练流程或可视化渲染。
- 不处理下游属性建模或测井重采样。

核心公开对象
------------
1. TargetZone: 目标层段构建与 QC 汇总。
2. TargetZone.with_boundary_extension: 生成上下外延层位副本。
3. TargetZone.to_mask: 生成三维样点掩码。

Examples
--------
>>> from cup.seismic.target_zone import TargetZone
>>> # raw_horizon_dfs 与 geometry 需提前准备
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cup.seismic import horizon as horizon_tools
from cup.seismic.horizon import HorizonSurface


def _validate_geometry_keys(geometry: Dict[str, Any]) -> None:
    """校验几何参数必要键。"""
    required = {
        "inline_min",
        "inline_max",
        "inline_step",
        "xline_min",
        "xline_max",
        "xline_step",
        "sample_min",
        "sample_max",
        "sample_step",
    }
    missing = required - set(geometry)
    if missing:
        raise ValueError(f"geometry is missing required keys: {sorted(missing)}")


class TargetZone:
    """从有序层位解释准备目标层段与 QC。

    Parameters
    ----------
    raw_horizon_dfs : dict[str, pandas.DataFrame]
        层位名称到原始拾取表的映射，需包含 ``inline``、``xline``、``interpretation`` 列。
    geometry : dict[str, Any]
        工区几何参数，要求包含 inline/xline/sample 轴的最小、最大与步长。
    horizon_names : list[str]
        层位顺序，从浅到深。
    qc_output_dir : str or Path or None, default=None
        QC 输出目录；为 None 时仅保存在内存。
    min_thickness : float or None, default=None
        最小允许厚度，None 时使用 ``sample_step``。
    nearest_distance_limit : float or None, default=None
        最近邻补全的最大距离（trace 索引单位）。
    outlier_threshold : float or None, default=None
        孤立异常点阈值，None 时禁用剔除。
    outlier_min_neighbor_count : int, default=2
        判断孤立异常点所需的最小邻居数量。

    Notes
    -----
    单层位建面由 ``cup.seismic.horizon`` 负责；本类只处理多层位顺序、
    厚度约束、目标层段掩码和 QC 汇总。
    """

    def __init__(
        self,
        raw_horizon_dfs: Dict[str, pd.DataFrame],
        geometry: Dict[str, Any],
        horizon_names: list[str],
        *,
        qc_output_dir: Optional[str | Path] = None,
        min_thickness: Optional[float] = None,
        nearest_distance_limit: Optional[float] = None,
        outlier_threshold: Optional[float] = None,
        outlier_min_neighbor_count: int = 2,
    ) -> None:
        if len(raw_horizon_dfs) < 2:
            raise ValueError("raw_horizon_dfs must contain at least two horizons.")
        if len(horizon_names) < 2:
            raise ValueError("horizon_names must contain at least two ordered horizons.")
        if len(set(horizon_names)) != len(horizon_names):
            raise ValueError("horizon_names must be unique.")
        missing = [name for name in horizon_names if name not in raw_horizon_dfs]
        if missing:
            raise ValueError(f"horizon_names not found in raw_horizon_dfs: {missing}")

        _validate_geometry_keys(geometry)
        self.geometry = dict(geometry)
        self.horizon_names = list(horizon_names)
        self.qc_output_dir = None if qc_output_dir is None else Path(qc_output_dir)
        self.min_thickness = float(self.geometry["sample_step"]) if min_thickness is None else float(min_thickness)
        if self.min_thickness <= 0.0:
            raise ValueError(f"min_thickness must be positive, got {self.min_thickness}.")
        self.nearest_distance_limit = None if nearest_distance_limit is None else float(nearest_distance_limit)
        if self.nearest_distance_limit is not None and self.nearest_distance_limit <= 0.0:
            raise ValueError(f"nearest_distance_limit must be positive, got {self.nearest_distance_limit}.")
        self.outlier_threshold = None if outlier_threshold is None else float(outlier_threshold)
        self.outlier_min_neighbor_count = int(outlier_min_neighbor_count)
        if self.outlier_min_neighbor_count < 1:
            raise ValueError(f"outlier_min_neighbor_count must be >= 1, got {self.outlier_min_neighbor_count}.")

        self._il_axis, self._xl_axis, self._sample_axis = self._build_axes()
        self.raw_horizon_dfs = {
            name: horizon_tools.normalize_interpretation_unit_for_geometry(raw_horizon_dfs[name], self.geometry)
            for name in self.horizon_names
        }
        for name, df in self.raw_horizon_dfs.items():
            horizon_tools.require_interpretation_columns(df, name)

        self._surface_interpolations = {}
        self.horizon_surfaces: dict[str, HorizonSurface] = {}
        value_domain = str(self.geometry.get("sample_domain", ""))
        value_unit = str(self.geometry.get("sample_unit", ""))
        for name in self.horizon_names:
            surface, interpolation = horizon_tools.build_horizon_surface(
                self.raw_horizon_dfs[name],
                self._il_axis,
                self._xl_axis,
                name=name,
                nearest_distance_limit=self.nearest_distance_limit,
                outlier_threshold=self.outlier_threshold,
                outlier_min_neighbor_count=self.outlier_min_neighbor_count,
                value_domain=value_domain,
                value_unit=value_unit,
            )
            self._surface_interpolations[name] = interpolation
            self.horizon_surfaces[name] = surface
        self.initial_horizon_grids = {
            name: interp.linear_grid.copy() for name, interp in self._surface_interpolations.items()
        }
        self.independent_filled_horizon_grids = {
            name: interp.nearest_grid.copy() for name, interp in self._surface_interpolations.items()
        }
        self.interpolation_support_masks = {
            name: interp.linear_support_mask.copy() for name, interp in self._surface_interpolations.items()
        }
        self.raw_pick_masks = {name: interp.raw_mask.copy() for name, interp in self._surface_interpolations.items()}
        self.nearest_distance_grids = {
            name: interp.nearest_distance_grid.copy() for name, interp in self._surface_interpolations.items()
        }
        self.outlier_stats = {name: dict(interp.outlier_stats) for name, interp in self._surface_interpolations.items()}

        self._build_trace_qc_masks()
        self._horizon_grids = self._build_final_horizon_grids()
        self.horizon_surfaces = {
            name: HorizonSurface.from_grid(
                name=name,
                inline_axis=self._il_axis,
                xline_axis=self._xl_axis,
                values=grid,
                value_domain=value_domain,
                value_unit=value_unit,
                support_mask=self.interpolation_support_masks.get(name),
                source_mask=self.raw_pick_masks.get(name),
                nearest_distance_grid=self.nearest_distance_grids.get(name),
                metadata={"outlier_stats": dict(self.outlier_stats.get(name, {}))},
            )
            for name, grid in self._horizon_grids.items()
        }
        self.interpolated_horizon_dfs = {
            name: self._grid_to_horizon_df(grid) for name, grid in self._horizon_grids.items()
        }
        self._build_qc_dataframes()
        if self.qc_output_dir is not None:
            self.write_qc(self.qc_output_dir)

    def _build_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建 inline/xline/sample 轴。"""
        il_axis = horizon_tools.build_axis(
            self.geometry["inline_min"],
            self.geometry["inline_max"],
            self.geometry["inline_step"],
            "inline",
        )
        xl_axis = horizon_tools.build_axis(
            self.geometry["xline_min"],
            self.geometry["xline_max"],
            self.geometry["xline_step"],
            "xline",
        )
        sample_axis = horizon_tools.build_axis(
            self.geometry["sample_min"],
            self.geometry["sample_max"],
            self.geometry["sample_step"],
            "sample",
        )
        return il_axis, xl_axis, sample_axis

    def _build_trace_qc_masks(self) -> None:
        """构建 trace 级 QC 掩码。"""
        shape = (self._il_axis.size, self._xl_axis.size)
        no_support_any = np.zeros(shape, dtype=bool)
        for support_mask in self.interpolation_support_masks.values():
            no_support_any |= ~support_mask

        crossing_any = np.zeros(shape, dtype=bool)
        thin_any = np.zeros(shape, dtype=bool)
        pair_records = []
        self._pair_qc_masks: dict[tuple[str, str], dict[str, np.ndarray]] = {}

        for top_name, bottom_name in self.iter_zones():
            top_grid = self.initial_horizon_grids[top_name]
            bottom_grid = self.initial_horizon_grids[bottom_name]
            finite_pair = np.isfinite(top_grid) & np.isfinite(bottom_grid)
            thickness = bottom_grid - top_grid
            crossing = finite_pair & (top_grid >= bottom_grid)
            thin = finite_pair & ~crossing & (thickness < self.min_thickness)
            pair_no_support = ~(
                self.interpolation_support_masks[top_name] & self.interpolation_support_masks[bottom_name]
            )

            crossing_any |= crossing
            thin_any |= thin
            self._pair_qc_masks[(top_name, bottom_name)] = {
                "finite_pair": finite_pair,
                "no_support": pair_no_support,
                "crossing": crossing,
                "thin": thin,
                "thickness": thickness,
            }

            pair_records.append(
                {
                    "horizon_pair": f"{top_name}->{bottom_name}",
                    "top_name": top_name,
                    "bottom_name": bottom_name,
                    "total_traces": int(np.prod(shape)),
                    "pair_no_support_count": int(np.count_nonzero(pair_no_support)),
                    "crossing_count": int(np.count_nonzero(crossing)),
                    "thin_count": int(np.count_nonzero(thin)),
                    "pair_valid_count": int(np.count_nonzero(finite_pair & ~pair_no_support & ~crossing & ~thin)),
                    "min_thickness": self.min_thickness,
                }
            )

        self.no_support_mask = no_support_any
        self.crossing_mask = crossing_any
        self.thin_mask = thin_any
        self.valid_control_mask = ~(no_support_any | crossing_any | thin_any)
        self.masked_trace_mask = ~self.valid_control_mask
        self._summary_pair_records = pair_records

    def _build_final_horizon_grids(self) -> dict[str, np.ndarray]:
        """基于厚度插值生成最终层位网格。"""
        final_grids = {}
        top_name = self.horizon_names[0]
        top_grid = self.independent_filled_horizon_grids[top_name].copy()
        final_grids[top_name] = top_grid
        previous_grid = top_grid

        for top_name, bottom_name in self.iter_zones():
            pair_qc = self._pair_qc_masks[(top_name, bottom_name)]
            thickness = pair_qc["thickness"]
            control_mask = self.valid_control_mask & np.isfinite(thickness) & (thickness >= self.min_thickness)
            if not np.any(control_mask):
                positive_mask = np.isfinite(thickness) & (thickness >= self.min_thickness)
                if not np.any(positive_mask):
                    raise ValueError(f"Zone '{top_name}' -> '{bottom_name}' has no positive thickness controls.")
                control_mask = positive_mask

            thickness_control_grid = np.full(thickness.shape, np.nan, dtype=float)
            thickness_control_grid[control_mask] = thickness[control_mask]
            thickness_interp = horizon_tools.linear_then_nearest_from_grid(
                thickness_control_grid,
                nearest_distance_limit=self.nearest_distance_limit,
            ).nearest_grid
            finite = np.isfinite(thickness_interp)
            thickness_interp[finite] = np.maximum(thickness_interp[finite], self.min_thickness)
            bottom_grid = previous_grid + thickness_interp
            final_grids[bottom_name] = bottom_grid
            previous_grid = bottom_grid

        filled_model_mask = np.ones_like(self.valid_control_mask, dtype=bool)
        for grid in final_grids.values():
            filled_model_mask &= np.isfinite(grid)
        self.filled_model_mask = filled_model_mask
        return final_grids

    def _grid_to_horizon_df(self, grid: np.ndarray) -> pd.DataFrame:
        """将层位网格转为 DataFrame。"""
        il_grid, xl_grid = np.meshgrid(self._il_axis, self._xl_axis, indexing="ij")
        return pd.DataFrame(
            {
                "inline": il_grid.ravel(),
                "xline": xl_grid.ravel(),
                "interpretation": grid.ravel(),
            }
        )

    def _trace_qc_dataframe(self) -> pd.DataFrame:
        """生成 trace 级 QC DataFrame。"""
        rows = []
        for i, inline in enumerate(self._il_axis):
            for j, xline in enumerate(self._xl_axis):
                no_support_horizons = [
                    name for name in self.horizon_names if not bool(self.interpolation_support_masks[name][i, j])
                ]
                crossing_pairs = []
                thin_pairs = []
                for top_name, bottom_name in self.iter_zones():
                    pair_qc = self._pair_qc_masks[(top_name, bottom_name)]
                    if bool(pair_qc["crossing"][i, j]):
                        crossing_pairs.append(f"{top_name}->{bottom_name}")
                    if bool(pair_qc["thin"][i, j]):
                        thin_pairs.append(f"{top_name}->{bottom_name}")

                rows.append(
                    {
                        "inline": float(inline),
                        "xline": float(xline),
                        "valid_control": bool(self.valid_control_mask[i, j]),
                        "filled_model": bool(self.filled_model_mask[i, j]),
                        "masked_trace": bool(self.masked_trace_mask[i, j]),
                        "filled_by_thickness_interpolation": bool(
                            self.filled_model_mask[i, j] and not self.valid_control_mask[i, j]
                        ),
                        "no_support": bool(self.no_support_mask[i, j]),
                        "crossing": bool(self.crossing_mask[i, j]),
                        "thin": bool(self.thin_mask[i, j]),
                        "no_support_horizons": ";".join(no_support_horizons),
                        "crossing_pairs": ";".join(crossing_pairs),
                        "thin_pairs": ";".join(thin_pairs),
                    }
                )
        return pd.DataFrame.from_records(rows)

    def _pair_qc_dataframe(self) -> pd.DataFrame:
        """生成层对 QC DataFrame。"""
        records = []
        for top_name, bottom_name in self.iter_zones():
            pair_qc = self._pair_qc_masks[(top_name, bottom_name)]
            top_grid = self.initial_horizon_grids[top_name]
            bottom_grid = self.initial_horizon_grids[bottom_name]
            invalid = pair_qc["no_support"] | pair_qc["crossing"] | pair_qc["thin"]
            invalid_indices = np.argwhere(invalid)
            for i, j in invalid_indices:
                records.append(
                    {
                        "top_name": top_name,
                        "bottom_name": bottom_name,
                        "inline": float(self._il_axis[i]),
                        "xline": float(self._xl_axis[j]),
                        "interpretation_top": float(top_grid[i, j]) if np.isfinite(top_grid[i, j]) else np.nan,
                        "interpretation_bottom": float(bottom_grid[i, j]) if np.isfinite(bottom_grid[i, j]) else np.nan,
                        "thickness": float(pair_qc["thickness"][i, j])
                        if np.isfinite(pair_qc["thickness"][i, j])
                        else np.nan,
                        "no_support": bool(pair_qc["no_support"][i, j]),
                        "crossing": bool(pair_qc["crossing"][i, j]),
                        "thin": bool(pair_qc["thin"][i, j]),
                    }
                )
        columns = [
            "top_name",
            "bottom_name",
            "inline",
            "xline",
            "interpretation_top",
            "interpretation_bottom",
            "thickness",
            "no_support",
            "crossing",
            "thin",
        ]
        return pd.DataFrame.from_records(records, columns=columns)

    def _summary_dataframe(self) -> pd.DataFrame:
        """生成 QC 汇总 DataFrame。"""
        records = []
        for horizon_name in self.horizon_names:
            stats = self.outlier_stats.get(horizon_name, {})
            records.append(
                {
                    "record_type": "horizon",
                    "horizon_pair": horizon_name,
                    "top_name": horizon_name,
                    "bottom_name": "",
                    "total_traces": int(self.valid_control_mask.size),
                    "pair_no_support_count": int(np.count_nonzero(~self.interpolation_support_masks[horizon_name])),
                    "crossing_count": 0,
                    "thin_count": 0,
                    "pair_valid_count": int(np.count_nonzero(self.interpolation_support_masks[horizon_name])),
                    "min_thickness": self.min_thickness,
                    "outlier_enabled": bool(stats.get("enabled", False)),
                    "outlier_threshold": stats.get("threshold"),
                    "outlier_min_neighbor_count": stats.get("min_neighbor_count"),
                    "outlier_total_points": stats.get("total_points"),
                    "outlier_removed_points": stats.get("removed_points"),
                    "outlier_removed_ratio": stats.get("removed_ratio"),
                }
            )

        for record in self._summary_pair_records:
            out_record = dict(record)
            out_record["record_type"] = "pair"
            records.append(out_record)

        records.append(
            {
                "record_type": "global",
                "horizon_pair": "__trace_global__",
                "top_name": "",
                "bottom_name": "",
                "total_traces": int(self.valid_control_mask.size),
                "pair_no_support_count": int(np.count_nonzero(self.no_support_mask)),
                "crossing_count": int(np.count_nonzero(self.crossing_mask)),
                "thin_count": int(np.count_nonzero(self.thin_mask)),
                "pair_valid_count": int(np.count_nonzero(self.valid_control_mask)),
                "min_thickness": self.min_thickness,
                "filled_model_count": int(np.count_nonzero(self.filled_model_mask)),
                "filled_by_thickness_interpolation_count": int(
                    np.count_nonzero(self.filled_model_mask & ~self.valid_control_mask)
                ),
                "nearest_distance_limit": self.nearest_distance_limit,
                "outlier_threshold": self.outlier_threshold,
                "outlier_min_neighbor_count": self.outlier_min_neighbor_count,
            }
        )
        return pd.DataFrame.from_records(records)

    def _build_qc_dataframes(self) -> None:
        """构建全部 QC DataFrame。"""
        self.trace_qc_df = self._trace_qc_dataframe()
        self.pair_qc_df = self._pair_qc_dataframe()
        self.qc_summary_df = self._summary_dataframe()
        self.trace_qc_path: Optional[Path] = None
        self.pair_qc_path: Optional[Path] = None
        self.qc_summary_path: Optional[Path] = None

    def write_qc(self, qc_output_dir: str | Path) -> dict[str, Path]:
        """写出 QC CSV 文件。

        Parameters
        ----------
        qc_output_dir : str or Path
            输出目录。

        Returns
        -------
        dict[str, Path]
            写出文件路径字典。
        """
        qc_output_path = Path(qc_output_dir)
        qc_output_path.mkdir(parents=True, exist_ok=True)
        self.qc_output_dir = qc_output_path
        self.trace_qc_path = qc_output_path / "target_layer_trace_qc.csv"
        self.pair_qc_path = qc_output_path / "target_layer_pair_qc.csv"
        self.qc_summary_path = qc_output_path / "target_layer_qc_summary.csv"
        self.trace_qc_df.to_csv(self.trace_qc_path, index=False, encoding="utf-8-sig")
        self.pair_qc_df.to_csv(self.pair_qc_path, index=False, encoding="utf-8-sig")
        self.qc_summary_df.to_csv(self.qc_summary_path, index=False, encoding="utf-8-sig")
        return {
            "trace_qc": self.trace_qc_path,
            "pair_qc": self.pair_qc_path,
            "qc_summary": self.qc_summary_path,
        }

    @property
    def ilines(self) -> np.ndarray:
        """返回 inline 轴坐标。

        Returns
        -------
        np.ndarray
            inline 轴取值。
        """
        return self._il_axis.copy()

    @property
    def xlines(self) -> np.ndarray:
        """返回 xline 轴坐标。

        Returns
        -------
        np.ndarray
            xline 轴取值。
        """
        return self._xl_axis.copy()

    @property
    def samples(self) -> np.ndarray:
        """返回采样轴坐标。

        Returns
        -------
        np.ndarray
            采样轴取值。
        """
        return self._sample_axis.copy()

    def iter_zones(self) -> list[tuple[str, str]]:
        """返回相邻层位对列表。

        Returns
        -------
        list[tuple[str, str]]
            相邻层位对 (top, bottom)。
        """
        return list(zip(self.horizon_names[:-1], self.horizon_names[1:]))

    def get_trace_valid_mask(self) -> np.ndarray:
        """返回可靠控制的 trace 掩码。

        Returns
        -------
        np.ndarray
            可靠控制 trace 的布尔掩码。
        """
        return self.valid_control_mask.copy()

    def get_filled_model_mask(self) -> np.ndarray:
        """返回填充后层位网格的有效掩码。

        Returns
        -------
        np.ndarray
            填充后层位网格的布尔掩码。
        """
        return self.filled_model_mask.copy()

    def get_zone_valid_mask(self, zone: tuple[str, str], *, use_valid_control_mask: bool = True) -> np.ndarray:
        """返回指定层段的有效掩码。

        Parameters
        ----------
        zone : tuple[str, str]
            层段名称对 (top, bottom)。
        use_valid_control_mask : bool, default=True
            是否仅使用可靠控制 trace。

        Returns
        -------
        np.ndarray
            指定层段的布尔掩码。
        """
        top_name, bottom_name = self._resolve_zone(zone)
        top_grid, bottom_grid = self.get_zone_sample_index_grids((top_name, bottom_name))
        valid = np.isfinite(top_grid) & np.isfinite(bottom_grid) & (bottom_grid > top_grid)
        if use_valid_control_mask:
            valid &= self.valid_control_mask
        return valid

    def _resolve_zone(self, zone: tuple[str, str]) -> tuple[str, str]:
        """解析并校验层段名称对。"""
        top_name, bottom_name = zone
        if top_name not in self.horizon_names or bottom_name not in self.horizon_names:
            raise ValueError(f"zone contains unknown horizons: {zone}")
        top_idx = self.horizon_names.index(top_name)
        bottom_idx = self.horizon_names.index(bottom_name)
        if bottom_idx != top_idx + 1:
            raise ValueError(f"zone must contain adjacent horizons, got {zone}")
        return top_name, bottom_name

    def get_horizon_grid(self, horizon_name: str) -> np.ndarray:
        """返回指定层位的插值网格。

        Parameters
        ----------
        horizon_name : str
            层位名称。

        Returns
        -------
        np.ndarray
            层位网格。

        Raises
        ------
        ValueError
            当层位名称不存在时。
        """
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")
        return self._horizon_grids[horizon_name].copy()

    def get_horizon_surface(self, horizon_name: str) -> HorizonSurface:
        """返回指定层位的 ``HorizonSurface``。"""
        if horizon_name not in self.horizon_surfaces:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")
        return self.horizon_surfaces[horizon_name]

    def get_horizon_interpretation_at_location(
        self,
        horizon_name: str,
        il_float: float,
        xl_float: float,
    ) -> float:
        """获取指定层位在位置处的插值值。

        Parameters
        ----------
        horizon_name : str
            层位名称。
        il_float : float
            inline 坐标（浮点）。
        xl_float : float
            xline 坐标（浮点）。

        Returns
        -------
        float
            双线性插值结果。

        Raises
        ------
        ValueError
            当层位名称不存在时。
        """
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")
        return float(self.horizon_surfaces[horizon_name].sample_at_line(il_float, xl_float).value)

    def get_interpretation_values_at_location(self, il_float: float, xl_float: float) -> Dict[str, float]:
        """获取所有层位在位置处的插值值。

        Parameters
        ----------
        il_float : float
            inline 坐标（浮点）。
        xl_float : float
            xline 坐标（浮点）。

        Returns
        -------
        dict[str, float]
            层位名称到插值值的映射。
        """
        return {
            horizon_name: self.get_horizon_interpretation_at_location(horizon_name, il_float, xl_float)
            for horizon_name in self.horizon_names
        }

    def convert_horizon_to_relative_sample_index(self, horizon_name: str) -> pd.DataFrame:
        """将层位解释值转换为相对样点索引。

        Parameters
        ----------
        horizon_name : str
            层位名称。

        Returns
        -------
        pandas.DataFrame
            包含 ``sample_index`` 列的层位解释表。

        Raises
        ------
        ValueError
            当层位名称不存在或解释值越界时。
        """
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")

        sample_min = float(self._sample_axis[0])
        sample_max = float(self._sample_axis[-1])
        sample_step = float(self.geometry["sample_step"])
        grid = self._horizon_grids[horizon_name]
        finite = np.isfinite(grid)
        if np.any(finite):
            tol = 1e-6
            out_of_range = finite & ((grid < sample_min - tol) | (grid > sample_max + tol))
            if np.any(out_of_range):
                bad_indices = np.argwhere(out_of_range)[:5]
                examples = [
                    {
                        "inline": float(self._il_axis[i]),
                        "xline": float(self._xl_axis[j]),
                        "interpretation": float(grid[i, j]),
                    }
                    for i, j in bad_indices
                ]
                raise ValueError(
                    "Horizon values are out of sample range. "
                    f"Expected within [{sample_min}, {sample_max}], "
                    f"found {int(np.count_nonzero(out_of_range))} out-of-range points. "
                    f"Examples: {examples}"
                )

        out_df = self._grid_to_horizon_df(grid)
        sample_index = np.full(grid.size, np.nan, dtype=float)
        values = out_df["interpretation"].to_numpy(dtype=float, copy=False)
        value_finite = np.isfinite(values)
        sample_index[value_finite] = (values[value_finite] - sample_min) / sample_step
        out_df["sample_index"] = sample_index
        return out_df

    def _get_horizon_sample_index_grid(self, horizon_name: str) -> np.ndarray:
        """获取层位相对样点索引网格。"""
        df = self.convert_horizon_to_relative_sample_index(horizon_name)
        return df["sample_index"].to_numpy(dtype=float).reshape((self._il_axis.size, self._xl_axis.size))

    def get_zone_sample_index_grids(self, zone: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
        """返回层段顶/底界样点索引网格。

        Parameters
        ----------
        zone : tuple[str, str]
            层段名称对 (top, bottom)。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            顶界与底界样点索引网格。
        """
        top_name, bottom_name = self._resolve_zone(zone)
        return self._get_horizon_sample_index_grid(top_name), self._get_horizon_sample_index_grid(bottom_name)

    def _clone_with_horizon_grids(
        self,
        *,
        horizon_names: list[str],
        horizon_grids: dict[str, np.ndarray],
    ) -> "TargetZone":
        """基于已有 QC 状态和新的层位网格创建派生副本。"""
        out = object.__new__(TargetZone)
        out.geometry = dict(self.geometry)
        out.horizon_names = list(horizon_names)
        out.qc_output_dir = None
        out.min_thickness = self.min_thickness
        out.nearest_distance_limit = self.nearest_distance_limit
        out.outlier_threshold = self.outlier_threshold
        out.outlier_min_neighbor_count = self.outlier_min_neighbor_count
        out._il_axis = self._il_axis.copy()
        out._xl_axis = self._xl_axis.copy()
        out._sample_axis = self._sample_axis.copy()
        out.raw_horizon_dfs = {name: df.copy() for name, df in self.raw_horizon_dfs.items()}
        out._surface_interpolations = dict(self._surface_interpolations)
        out.initial_horizon_grids = {name: grid.copy() for name, grid in self.initial_horizon_grids.items()}
        out.independent_filled_horizon_grids = {
            name: grid.copy() for name, grid in self.independent_filled_horizon_grids.items()
        }
        out.interpolation_support_masks = {
            name: mask.copy() for name, mask in self.interpolation_support_masks.items()
        }
        out.raw_pick_masks = {name: mask.copy() for name, mask in self.raw_pick_masks.items()}
        out.nearest_distance_grids = {name: grid.copy() for name, grid in self.nearest_distance_grids.items()}
        out.outlier_stats = {name: dict(stats) for name, stats in self.outlier_stats.items()}
        out.no_support_mask = self.no_support_mask.copy()
        out.crossing_mask = self.crossing_mask.copy()
        out.thin_mask = self.thin_mask.copy()
        out.valid_control_mask = self.valid_control_mask.copy()
        out.masked_trace_mask = self.masked_trace_mask.copy()
        out.filled_model_mask = self.filled_model_mask.copy()
        out._pair_qc_masks = dict(self._pair_qc_masks)
        out._summary_pair_records = list(self._summary_pair_records)
        out._horizon_grids = {name: grid.copy() for name, grid in horizon_grids.items()}

        value_domain = str(out.geometry.get("sample_domain", ""))
        value_unit = str(out.geometry.get("sample_unit", ""))
        out.horizon_surfaces = {
            name: HorizonSurface.from_grid(
                name=name,
                inline_axis=out._il_axis,
                xline_axis=out._xl_axis,
                values=grid,
                value_domain=value_domain,
                value_unit=value_unit,
            )
            for name, grid in out._horizon_grids.items()
        }
        out.interpolated_horizon_dfs = {
            name: out._grid_to_horizon_df(grid) for name, grid in out._horizon_grids.items()
        }
        out.trace_qc_df = self.trace_qc_df.copy()
        out.pair_qc_df = self.pair_qc_df.copy()
        out.qc_summary_df = self.qc_summary_df.copy()
        out.trace_qc_path = None
        out.pair_qc_path = None
        out.qc_summary_path = None
        return out

    def with_boundary_extension(
        self,
        extension_samples: int,
        *,
        top_extension_name: str = "top_extension",
        bottom_extension_name: str = "bottom_extension",
    ) -> "TargetZone":
        """返回包含上下外延层位的轻量副本。

        Parameters
        ----------
        extension_samples : int
            外延样点数。
        top_extension_name : str, default="top_extension"
            顶外延层位名称。
        bottom_extension_name : str, default="bottom_extension"
            底外延层位名称。

        Returns
        -------
        TargetZone
            带外延层位的副本。

        Raises
        ------
        ValueError
            当参数非法或层位名称冲突时。
        """
        if extension_samples < 0:
            raise ValueError(f"extension_samples must be >= 0, got {extension_samples}.")
        if extension_samples == 0:
            return self
        if top_extension_name == bottom_extension_name:
            raise ValueError("top_extension_name and bottom_extension_name must be different.")
        duplicate_names = {name for name in (top_extension_name, bottom_extension_name) if name in self.horizon_names}
        if duplicate_names:
            raise ValueError(f"extension horizon names already exist: {sorted(duplicate_names)}")

        sample_min = float(self._sample_axis[0])
        sample_max = float(self._sample_axis[-1])
        offset = float(extension_samples) * float(self.geometry["sample_step"])

        top_source = self._horizon_grids[self.horizon_names[0]]
        bottom_source = self._horizon_grids[self.horizon_names[-1]]
        top_extension = np.where(np.isfinite(top_source), np.clip(top_source - offset, sample_min, sample_max), np.nan)
        bottom_extension = np.where(
            np.isfinite(bottom_source),
            np.clip(bottom_source + offset, sample_min, sample_max),
            np.nan,
        )

        return self._clone_with_horizon_grids(
            horizon_names=[top_extension_name, *self.horizon_names, bottom_extension_name],
            horizon_grids={
                top_extension_name: top_extension,
                **{name: grid.copy() for name, grid in self._horizon_grids.items()},
                bottom_extension_name: bottom_extension,
            },
        )

    def to_mask(
        self,
        zone: Optional[tuple[str, str]] = None,
        *,
        use_valid_control_mask: bool = True,
    ) -> np.ndarray:
        """构建三维样点掩码。

        Parameters
        ----------
        zone : tuple[str, str] or None, default=None
            指定层段 (top, bottom)。为 None 时使用全部层段。
        use_valid_control_mask : bool, default=True
            是否仅使用可靠控制 trace；为 False 时使用填充后的全覆盖层位。

        Returns
        -------
        np.ndarray
            形状为 (n_il, n_xl, n_sample) 的布尔掩码。
        """
        n_il = int(self.geometry.get("n_il", self._il_axis.size))
        n_xl = int(self.geometry.get("n_xl", self._xl_axis.size))
        n_sample = int(self.geometry.get("n_sample", self._sample_axis.size))
        if n_il != self._il_axis.size:
            raise ValueError(f"geometry n_il={n_il} does not match axis size {self._il_axis.size}.")
        if n_xl != self._xl_axis.size:
            raise ValueError(f"geometry n_xl={n_xl} does not match axis size {self._xl_axis.size}.")
        if n_sample != self._sample_axis.size:
            raise ValueError(f"geometry n_sample={n_sample} does not match axis size {self._sample_axis.size}.")

        mask = np.zeros((n_il, n_xl, n_sample), dtype=bool)
        zones = [self._resolve_zone(zone)] if zone is not None else self.iter_zones()
        for top_name, bottom_name in zones:
            top_grid, bottom_grid = self.get_zone_sample_index_grids((top_name, bottom_name))
            valid = self.get_zone_valid_mask((top_name, bottom_name), use_valid_control_mask=use_valid_control_mask)
            for i in range(n_il):
                for j in range(n_xl):
                    if not valid[i, j]:
                        continue
                    idx_top = max(0, int(np.round(top_grid[i, j])))
                    idx_bottom = min(n_sample, int(np.round(bottom_grid[i, j])) + 1)
                    if idx_top < idx_bottom:
                        mask[i, j, idx_top:idx_bottom] = True
        return mask
