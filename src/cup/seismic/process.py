"""cup.seismic.process: 旧层位插值入口。

本模块保留 ``interpolate_interpretation_surface`` 作为旧 notebook 的适配层。
新代码应使用 ``cup.seismic.horizon.build_horizon_surface`` 或
``HorizonSurface``，避免在多个模块维护重复的层位建面算法。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd

from cup.seismic import horizon as horizon_tools

OUTLIER_REMOVAL_WARNING_RATIO = horizon_tools.OUTLIER_REMOVAL_WARNING_RATIO
_build_axis = horizon_tools.build_axis


def interpolate_interpretation_surface(
    interpretation_df: pd.DataFrame,
    geometry: Dict[str, Any],
    outlier_threshold: float,
    min_neighbor_count: int = 2,
    keep_nan: bool = True,
) -> pd.DataFrame:
    """旧入口：清洗并插值单个层位面。

    新代码请直接使用 ``cup.seismic.horizon.build_horizon_surface``。本函数仅
    保持旧返回格式：``inline``、``xline``、``interpretation`` 三列，并把
    异常点统计写入 ``df.attrs["outlier_removal"]``。
    """
    warnings.warn(
        "cup.seismic.process.interpolate_interpretation_surface is deprecated; "
        "use cup.seismic.horizon.build_horizon_surface instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    required = {"inline_min", "inline_max", "inline_step", "xline_min", "xline_max", "xline_step"}
    missing = required - set(geometry)
    if missing:
        raise ValueError(f"geometry is missing required keys: {sorted(missing)}")

    il_axis = horizon_tools.build_axis(
        float(geometry["inline_min"]),
        float(geometry["inline_max"]),
        float(geometry["inline_step"]),
        "inline",
    )
    xl_axis = horizon_tools.build_axis(
        float(geometry["xline_min"]),
        float(geometry["xline_max"]),
        float(geometry["xline_step"]),
        "xline",
    )
    normalized_df = horizon_tools.normalize_interpretation_unit_for_geometry(interpretation_df, geometry)
    input_values = normalized_df[["inline", "xline", "interpretation"]].to_numpy(dtype=float, copy=False)
    valid_input_count = int(np.count_nonzero(np.isfinite(input_values).all(axis=1)))

    interpolation = horizon_tools.build_surface_interpolation(
        normalized_df,
        il_axis,
        xl_axis,
        nearest_distance_limit=None,
        outlier_threshold=float(outlier_threshold),
        outlier_min_neighbor_count=int(min_neighbor_count),
    )
    outlier_stats = dict(interpolation.outlier_stats)
    outlier_stats["input_valid_points"] = valid_input_count
    outlier_stats["gridded_valid_points"] = outlier_stats.get("total_points", 0)
    if outlier_stats.get("total_points") and float(outlier_stats.get("removed_ratio", 0.0)) > OUTLIER_REMOVAL_WARNING_RATIO:
        warnings.warn(
            "Isolated outlier removal dropped "
            f"{outlier_stats.get('removed_points')}/{outlier_stats.get('total_points')} interpretation point(s) "
            f"({float(outlier_stats.get('removed_ratio', 0.0)):.2%}), exceeding the "
            f"{OUTLIER_REMOVAL_WARNING_RATIO:.0%} warning threshold.",
            RuntimeWarning,
            stacklevel=2,
        )

    il_grid, xl_grid = np.meshgrid(il_axis, xl_axis, indexing="ij")
    out_df = pd.DataFrame(
        {
            "inline": il_grid.ravel(),
            "xline": xl_grid.ravel(),
            "interpretation": interpolation.nearest_grid.ravel(),
        }
    )
    out_df.attrs["outlier_removal"] = outlier_stats
    if keep_nan:
        return out_df
    out_df_valid = out_df[np.isfinite(out_df["interpretation"])].reset_index(drop=True)
    out_df_valid.attrs["outlier_removal"] = outlier_stats
    return out_df_valid
