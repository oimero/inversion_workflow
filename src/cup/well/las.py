"""cup.well.las: LAS 文件头扫描与筛选曲线导出。

本模块提供 LAS 文件头的轻量扫描、曲线存在性检查，以及将筛选后的曲线
导出为标准 LAS 文件的工具。

边界说明
--------
- 本模块不负责曲线分类与主曲线选择，这些由 ``cup.well.curves`` 处理。
- 导出时不修改曲线数值，输入清洗应在上游完成。

核心公开对象
------------
1. scan_las_curves: 扫描 LAS 文件头与曲线列表。
2. export_selected_curves_to_las: 将筛选后的曲线集合导出为 LAS。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import lasio
import numpy as np

from cup.well.curves import CurveInfo, exact_mnemonic, normalize_mnemonic


@dataclass(frozen=True)
class LasHeader:
    """Lightweight LAS header summary."""

    well_name: str
    index_mnemonic: str
    index_unit: str
    start: float | None
    stop: float | None
    step: float | None
    null_value: float | None
    curve_count: int
    estimated_sample_count: int | None
    index_is_monotonic: bool | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _header_value(las: lasio.LASFile, section: str, mnemonic: str) -> Any:
    try:
        return getattr(las, section)[mnemonic].value
    except (KeyError, AttributeError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _estimated_sample_count(start: float | None, stop: float | None, step: float | None) -> int | None:
    if start is None or stop is None or step is None or step == 0.0:
        return None
    count = int(abs((stop - start) / step)) + 1
    return count if count > 0 else None


def _index_is_monotonic(start: float | None, stop: float | None, step: float | None) -> bool | None:
    if start is None or stop is None or step is None or step == 0.0:
        return None
    return (stop - start) * step >= 0.0


def _build_las_header(las: lasio.LASFile, *, fallback_well_name: str) -> LasHeader:
    index_curve = las.curves[0] if las.curves else None
    start = _optional_float(_header_value(las, "well", "STRT"))
    stop = _optional_float(_header_value(las, "well", "STOP"))
    step = _optional_float(_header_value(las, "well", "STEP"))
    null_value = _optional_float(_header_value(las, "well", "NULL"))
    well_name = str(_header_value(las, "well", "WELL") or fallback_well_name).strip()
    return LasHeader(
        well_name=well_name,
        index_mnemonic=str(index_curve.mnemonic if index_curve is not None else "DEPT"),
        index_unit=str(index_curve.unit or "") if index_curve is not None else "",
        start=start,
        stop=stop,
        step=step,
        null_value=null_value,
        curve_count=len(las.curves),
        estimated_sample_count=_estimated_sample_count(start, stop, step),
        index_is_monotonic=_index_is_monotonic(start, stop, step),
    )


def scan_las_header(path: Path) -> LasHeader:
    """不加载数据样点，仅读取 LAS 元数据。"""
    las = lasio.read(str(path), ignore_data=True)
    return _build_las_header(las, fallback_well_name=Path(path).stem)


def scan_las_curves(path: Path) -> tuple[LasHeader, list[CurveInfo]]:
    """不加载数据样点，仅读取 LAS 曲线头信息。"""
    las = lasio.read(str(path), ignore_data=True)
    header = _build_las_header(las, fallback_well_name=Path(path).stem)
    curves = [
        CurveInfo(
            mnemonic=str(curve.mnemonic),
            unit=str(curve.unit or ""),
            description=str(curve.descr or ""),
            index=index,
        )
        for index, curve in enumerate(las.curves)
    ]
    return header, curves


def _resolve_curve_index(
    curve_index: dict[str, int], normalized_index: dict[str, list[int]], mnemonic: str
) -> int | None:
    exact = exact_mnemonic(mnemonic)
    if exact in curve_index:
        return curve_index[exact]
    candidates = normalized_index.get(normalize_mnemonic(mnemonic), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[0]
    return None


def _validate_write_format(write_fmt: str) -> None:
    if not isinstance(write_fmt, str) or not write_fmt.strip():
        raise ValueError("write_fmt must be a non-empty printf-style float format.")
    try:
        _ = write_fmt % 1.2345
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid write_fmt: {write_fmt}") from exc


def export_selected_curves_to_las(
    source_las: Path,
    output_las: Path,
    selected_mnemonics: Sequence[str],
    *,
    null_value: float = -999.25,
    write_fmt: str = "%.6f",
) -> tuple[Path, list[dict[str, str]], list[str]]:
    """导出 LAS 索引曲线和选中的原始曲线。"""
    _validate_write_format(write_fmt)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    las = lasio.read(str(source_las))
    if not las.curves:
        raise ValueError(f"LAS file has no curves: {source_las}")
    data = np.asarray(las.data)
    if data.ndim != 2 or data.shape[1] != len(las.curves):
        raise ValueError(f"LAS data shape does not match curve headers: {source_las}")

    curve_index = {exact_mnemonic(curve.mnemonic): index for index, curve in enumerate(las.curves)}
    normalized_index: dict[str, list[int]] = {}
    for index, curve in enumerate(las.curves):
        normalized_index.setdefault(normalize_mnemonic(curve.mnemonic), []).append(index)
    requested = list(dict.fromkeys(str(mnemonic) for mnemonic in selected_mnemonics))

    out = lasio.LASFile()
    for item in las.well:
        out.well[item.mnemonic] = item
    out.well["NULL"].value = float(null_value)
    source_well_name = str(_header_value(las, "well", "WELL") or Path(source_las).stem).strip()
    out.well["WELL"].value = source_well_name

    index_curve = las.curves[0]
    out.append_curve(
        str(index_curve.mnemonic),
        data[:, 0],
        unit=str(index_curve.unit or ""),
        descr=str(index_curve.descr or ""),
    )

    skipped: list[dict[str, str]] = []
    exported_mnemonics: list[str] = []
    for mnemonic in requested:
        index = _resolve_curve_index(curve_index, normalized_index, mnemonic)
        if index is None:
            skipped.append({"curve": mnemonic, "reason": "selected_curve_missing_in_las"})
            continue
        if index == 0:
            continue
        curve = las.curves[index]
        try:
            out.append_curve(
                str(curve.mnemonic),
                data[:, index],
                unit=str(curve.unit or ""),
                descr=str(curve.descr or ""),
            )
            exported_mnemonics.append(str(curve.mnemonic))
        except Exception as exc:
            skipped.append({"curve": mnemonic, "reason": str(exc)})

    if len(out.curves) <= 1:
        raise ValueError(f"No selected curves were exported from {source_las}")

    out.write(str(output_las), version=2.0, wrap=False, fmt=write_fmt)
    return output_las, skipped, exported_mnemonics
