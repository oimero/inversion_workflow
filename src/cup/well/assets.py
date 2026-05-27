"""cup.well.assets: 井资产盘点辅助工具。

本模块将第一步井资产盘点逻辑从工作流脚本中抽离为可复用的数据结构与函数。
仅处理轻量元数据：井头、文件存在性、井口/底孔坐标与粗粒度空间 QC 标签。

边界说明
--------
- 本模块不读取 LAS 曲线内容，也不解析斜井轨迹文件。
- 井名匹配、文件查找、井型初分与平台聚类均在本模块完成。

核心公开对象
------------
1. WellHead / WellInventoryRecord / WellInventory: 井资产盘点数据结构。
2. NeighborPair / WellClusterRow: 井间关系与平台聚类输出。
3. build_file_lookup / build_name_lookup: 井名与文件查找。
4. classify_wellbore / determine_inventory_status: 井型初分与状态判定。
5. normalize_well_name: 项目统一的井名规范化键。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def normalize_well_name(name: object) -> str:
    """返回项目统一的井名匹配键。"""
    return str(name).strip().casefold()


def _display_well_name(name: object) -> str:
    return str(name).strip()


def _is_missing_well_name(name: str) -> bool:
    normalized = name.strip().casefold()
    return normalized == "" or normalized in {"nan", "none", "null", "<na>"}


def build_name_lookup(names: Iterable[object], *, asset_label: str) -> dict[str, str]:
    """构建大小写不敏感查找表，并在大小写冲突时失败。"""
    lookup: dict[str, str] = {}
    for raw_name in names:
        display = _display_well_name(raw_name)
        if _is_missing_well_name(display):
            continue
        key = normalize_well_name(display)
        previous = lookup.get(key)
        if previous is not None and previous != display:
            raise ValueError(
                f"Case-insensitive well name collision in {asset_label}: "
                f"{previous!r} and {display!r}. Please fix names before inventory."
            )
        lookup[key] = display
    return lookup


def build_file_lookup(files: Iterable[Path], *, asset_label: str) -> dict[str, Path]:
    """按文件 stem 构建大小写不敏感的资产文件查找表。"""
    lookup: dict[str, Path] = {}
    display_by_key: dict[str, str] = {}
    for file_path in files:
        if not file_path.is_file():
            continue
        display = _display_well_name(file_path.stem)
        if _is_missing_well_name(display):
            continue
        key = normalize_well_name(display)
        previous_display = display_by_key.get(key)
        previous_path = lookup.get(key)
        if previous_display is not None and (previous_display != display or previous_path != file_path):
            raise ValueError(
                f"Case-insensitive well file collision in {asset_label}: "
                f"{previous_path} and {file_path}. Please disambiguate files before inventory."
            )
        lookup[key] = file_path
        display_by_key[key] = display
    return lookup


def is_finite_number(value: object) -> bool:
    """判断输入是否为有限数值。"""
    try:
        return bool(np.isfinite(float(value)))  # type: ignore
    except (TypeError, ValueError):
        return False


def optional_float(value: object) -> float | None:
    """将输入转为 float，非有限时返回 None。"""
    if not is_finite_number(value):
        return None
    return float(value)  # type: ignore


def classify_wellbore(
    surface_x: object,
    surface_y: object,
    bottom_x: object,
    bottom_y: object,
    *,
    vertical_bottom_offset_threshold_m: float,
) -> tuple[str, float | None]:
    """根据井口到底孔的水平偏移初分井型。"""
    coords = [optional_float(v) for v in (surface_x, surface_y, bottom_x, bottom_y)]
    if any(v is None for v in coords):
        return "unknown", None

    sx, sy, bx, by = (float(v) for v in coords)  # type: ignore
    offset = float(np.hypot(bx - sx, by - sy))
    if offset <= float(vertical_bottom_offset_threshold_m):
        return "vertical", offset
    return "deviated", offset


@dataclass(frozen=True)
class WellHead:
    """结构化的 Petrel 井头记录。"""

    well_name: str
    surface_x: float | None
    surface_y: float | None
    bottom_x: float | None
    bottom_y: float | None
    kb_m: float | None

    @classmethod
    def from_petrel_row(cls, row: Mapping[str, Any]) -> "WellHead":
        return cls(
            well_name=_display_well_name(row["Name"]),
            surface_x=optional_float(row.get("Surface X")),
            surface_y=optional_float(row.get("Surface Y")),
            bottom_x=optional_float(row.get("Bottom hole X")),
            bottom_y=optional_float(row.get("Bottom hole Y")),
            kb_m=optional_float(row.get("Well datum value")),
        )


@dataclass
class WellInventoryRecord:
    """``well_inventory.csv`` 中的一井一行记录。"""

    well_name: str
    has_well_head: bool
    has_las: bool
    has_well_trace: bool
    has_time_depth: bool
    has_well_tops: bool
    surface_x: float | None = None
    surface_y: float | None = None
    bottom_x: float | None = None
    bottom_y: float | None = None
    kb_m: float | None = None
    inline_float: float | None = None
    xline_float: float | None = None
    nearest_inline: float | None = None
    nearest_xline: float | None = None
    survey_position: str = "invalid_xy"
    distance_to_survey_m: float | None = None
    bottom_offset_m: float | None = None
    wellbore_class: str = "unknown"
    inventory_status: str = "unknown"
    reasons: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["reasons"] = ";".join(self.reasons)
        return row


@dataclass(frozen=True)
class NeighborPair:
    well_a: str
    well_b: str
    distance_m: float
    same_surface_nearest_trace: bool
    same_surface_platform: bool
    class_pair: str
    risk: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WellClusterRow:
    cluster_id: str
    well_name: str
    surface_x: float
    surface_y: float
    wellbore_class: str
    survey_position: str
    nearest_inline: float | None
    nearest_xline: float | None
    cluster_size: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WellInventory:
    records: list[WellInventoryRecord]
    neighbor_pairs: list[NeighborPair]
    cluster_rows: list[WellClusterRow]
    summary: dict[str, Any]

    def records_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([record.to_row() for record in self.records])

    def neighbor_pairs_dataframe(self) -> pd.DataFrame:
        columns = [
            "well_a",
            "well_b",
            "distance_m",
            "same_surface_nearest_trace",
            "same_surface_platform",
            "class_pair",
            "risk",
        ]
        return pd.DataFrame.from_records([pair.to_row() for pair in self.neighbor_pairs], columns=columns)

    def clusters_dataframe(self) -> pd.DataFrame:
        columns = [
            "cluster_id",
            "well_name",
            "surface_x",
            "surface_y",
            "wellbore_class",
            "survey_position",
            "nearest_inline",
            "nearest_xline",
            "cluster_size",
        ]
        return pd.DataFrame.from_records([row.to_row() for row in self.cluster_rows], columns=columns)


def determine_inventory_status(*, has_las: bool, has_well_head: bool) -> tuple[str, list[str]]:
    """返回 LAS 筛选资产状态及原因。"""
    reasons: list[str] = []
    if has_las and has_well_head:
        return "usable_for_las_screen", reasons
    if has_las:
        reasons.append("no_well_head")
        return "las_only", reasons
    if has_well_head:
        reasons.append("no_las")
        return "head_only", reasons
    reasons.append("no_las")
    reasons.append("no_well_head")
    return "unknown", reasons


def build_platform_clusters(
    records: Sequence[WellInventoryRecord],
    *,
    platform_cluster_threshold_m: float,
) -> list[list[WellInventoryRecord]]:
    """用连通分量将有效井口聚合为平台井簇。"""
    valid = [
        record
        for record in records
        if record.surface_x is not None
        and record.surface_y is not None
        and is_finite_number(record.surface_x)
        and is_finite_number(record.surface_y)
    ]
    if not valid:
        return []

    parent = list(range(len(valid)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    threshold = float(platform_cluster_threshold_m)
    for i, left in enumerate(valid):
        for j in range(i + 1, len(valid)):
            right = valid[j]
            distance = float(
                np.hypot(float(right.surface_x) - float(left.surface_x), float(right.surface_y) - float(left.surface_y))  # type: ignore
            )
            if distance <= threshold:
                union(i, j)

    groups: dict[int, list[WellInventoryRecord]] = {}
    for index, record in enumerate(valid):
        groups.setdefault(find(index), []).append(record)

    clusters = [group for group in groups.values() if len(group) > 1]
    clusters.sort(key=lambda group: (-len(group), group[0].well_name))
    return clusters


def build_cluster_rows(
    records: Sequence[WellInventoryRecord],
    *,
    platform_cluster_threshold_m: float,
) -> list[WellClusterRow]:
    rows: list[WellClusterRow] = []
    clusters = build_platform_clusters(records, platform_cluster_threshold_m=platform_cluster_threshold_m)
    for cluster_index, cluster in enumerate(clusters, start=1):
        cluster_id = f"platform_{cluster_index:03d}"
        for record in sorted(cluster, key=lambda item: item.well_name.casefold()):
            rows.append(
                WellClusterRow(
                    cluster_id=cluster_id,
                    well_name=record.well_name,
                    surface_x=float(record.surface_x),  # type: ignore
                    surface_y=float(record.surface_y),  # type: ignore
                    wellbore_class=record.wellbore_class,
                    survey_position=record.survey_position,
                    nearest_inline=record.nearest_inline,
                    nearest_xline=record.nearest_xline,
                    cluster_size=len(cluster),
                )
            )
    return rows


def build_neighbor_pairs(
    records: Sequence[WellInventoryRecord],
    *,
    dense_well_neighbor_threshold_m: float,
    platform_cluster_threshold_m: float,
) -> tuple[list[NeighborPair], dict[str, Any]]:
    """构建非同平台同最近道冲突井对，并返回完整近邻统计。"""
    valid = [
        record
        for record in records
        if record.surface_x is not None
        and record.surface_y is not None
        and is_finite_number(record.surface_x)
        and is_finite_number(record.surface_y)
    ]
    dense_threshold = float(dense_well_neighbor_threshold_m)
    platform_threshold = float(platform_cluster_threshold_m)

    dense_pair_count = 0
    same_trace_pair_count = 0
    same_platform_pair_count = 0
    same_trace_platform_pair_count = 0
    exported_pairs: list[NeighborPair] = []

    for i, left in enumerate(valid):
        for j in range(i + 1, len(valid)):
            right = valid[j]
            distance = float(
                np.hypot(float(right.surface_x) - float(left.surface_x), float(right.surface_y) - float(left.surface_y))  # type: ignore
            )
            same_platform = distance <= platform_threshold
            same_trace = (
                left.nearest_inline is not None
                and left.nearest_xline is not None
                and right.nearest_inline is not None
                and right.nearest_xline is not None
                and left.nearest_inline == right.nearest_inline
                and left.nearest_xline == right.nearest_xline
            )

            if distance <= dense_threshold:
                dense_pair_count += 1
            if same_platform:
                same_platform_pair_count += 1
            if same_trace:
                same_trace_pair_count += 1
            if same_trace and same_platform:
                same_trace_platform_pair_count += 1
            if same_trace and not same_platform:
                exported_pairs.append(
                    NeighborPair(
                        well_a=left.well_name,
                        well_b=right.well_name,
                        distance_m=distance,
                        same_surface_nearest_trace=True,
                        same_surface_platform=False,
                        class_pair=f"{left.wellbore_class}/{right.wellbore_class}",
                        risk="same_trace_conflict",
                    )
                )

    exported_pairs.sort(key=lambda pair: (pair.distance_m, pair.well_a.casefold(), pair.well_b.casefold()))
    summary = {
        "valid_surface_well_count": len(valid),
        "dense_neighbor_pair_count": dense_pair_count,
        "same_surface_nearest_trace_pair_count": same_trace_pair_count,
        "same_platform_pair_count": same_platform_pair_count,
        "same_trace_platform_pair_count": same_trace_platform_pair_count,
        "exported_neighbor_pair_count": len(exported_pairs),
    }
    return exported_pairs, summary


def value_counts(records: Sequence[WellInventoryRecord], field_name: str) -> dict[str, int]:
    """按字段统计各取值的出现次数。

    Parameters
    ----------
    records : Sequence[WellInventoryRecord]
        井资产记录列表。
    field_name : str
        目标字段名。

    Returns
    -------
    dict[str, int]
        取值到计数的映射，按键排序。
    """
    values = [str(getattr(record, field_name)) for record in records]
    return dict(sorted(pd.Series(values, dtype="object").value_counts(dropna=False).astype(int).to_dict().items()))  # type: ignore
