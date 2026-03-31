"""cup.utils.replace: 文本替换与 LAS 曲线段规范化工具。

本模块提供面向文本内容的正则替换能力，当前聚焦 SMI 软件导出的
LAS 文件 ``~Curve`` 段规范化：识别带后缀的曲线简称并归一为基础
mnemonic，同时按曲线类型补齐标准单位。

边界说明
--------
- 本模块仅处理文本层面的替换，不负责 LAS 数值质量控制或插值。
- 本模块仅对 ``~Curve`` 段生效，不修改其他分段内容。

核心公开对象
------------
1. replace_smi_las_curve_section: 处理字符串形式的 LAS 文本。
2. replace_smi_las_curve_section_in_file: 输出到同目录 output 子目录。

Examples
--------
>>> from cup.utils.replace import replace_smi_las_curve_section
>>> text = "~Curve\\nDTCO_QYZ . :\\nRHOZ_QYZ . :\\n"
>>> out = replace_smi_las_curve_section(text)
>>> "DTCO .us/ft" in out
True
"""

from __future__ import annotations

import re
from pathlib import Path

from cup.well.mnemonics import _RHO_MNEMONICS, _VP_MNEMONICS, _VS_MNEMONICS


def _resolve_curve_base_name(raw_name: str) -> str | None:
    """解析 SMI 曲线简称为基础 mnemonic。

    Parameters
    ----------
    raw_name : str
        原始曲线名，可能包含后缀（如 ``DTCO_QYZ``）。

    Returns
    -------
    str or None
        若识别成功则返回基础简称（大写，如 ``DTCO``），
        否则返回 ``None``。

    Notes
    -----
    - 识别集合来自 ``cup.well.mnemonics`` 中的 Vp/Vs/Rho 常量。
    - 按长度降序匹配，避免 ``DT`` 抢先匹配 ``DTCO``。
    """
    name = raw_name.strip().upper()
    candidates = tuple(set(_RHO_MNEMONICS + _VP_MNEMONICS + _VS_MNEMONICS))

    # 优先匹配更长前缀，避免 DT 误匹配 DTCO。
    for candidate in sorted(candidates, key=len, reverse=True):
        if name == candidate or name.startswith(candidate + "_"):
            return candidate
    return None


def _unit_for_curve(base_name: str, sonic_unit: str, density_unit: str) -> str:
    """根据基础曲线简称选择目标单位。

    Parameters
    ----------
    base_name : str
        基础曲线简称（大写）。
    sonic_unit : str
        声波相关曲线（Vp/Vs）目标单位。
    density_unit : str
        密度曲线目标单位。
    Returns
    -------
    str
        与 ``base_name`` 对应的目标单位。

    Raises
    ------
    ValueError
        当 ``base_name`` 不在受支持集合中时触发。
    """
    if base_name in _VP_MNEMONICS or base_name in _VS_MNEMONICS:
        return sonic_unit
    if base_name in _RHO_MNEMONICS:
        return density_unit
    raise ValueError(f"Unsupported base mnemonic: {base_name}")


def replace_smi_las_curve_section(
    text: str,
    sonic_unit: str = "us/ft",
    density_unit: str = "g/cm3",
) -> str:
    """替换 SMI LAS 文本 ``~Curve`` 段内的曲线简称与单位。

    Parameters
    ----------
    text : str
        输入 LAS 文本内容。
    sonic_unit : str, default="us/ft"
        声波曲线（Vp/Vs）默认单位。
    density_unit : str, default="g/cm3"
        密度曲线默认单位。

    Returns
    -------
    str
        替换后的 LAS 文本内容。

    Notes
    -----
    - 仅在 ``~Curve`` 与下一段 ``~`` 标记之间处理。
    - 识别 ``<MNEMONIC>_...`` 形式（如 ``DTCO_QYZ``、``RHOZ_QYZ``），并归一为基础简称。
    - 输出曲线简称统一为大写。
    - 同类曲线全部保留，不做去重。
    """
    lines = text.splitlines(keepends=True)
    in_curve = False
    curve_line_pattern = re.compile(r"^(?P<indent>\s*)(?P<name>[^\s\.]+)\s*\.\s*(?P<unit>[^:\s]*)\s*:\s*(?P<desc>.*)$")

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("~CURVE"):
            in_curve = True
            new_lines.append(line)
            continue

        if in_curve and stripped.startswith("~") and not stripped.upper().startswith("~CURVE"):
            in_curve = False
            new_lines.append(line)
            continue

        if not in_curve:
            new_lines.append(line)
            continue

        # 在 ~Curve 段保留空行和注释行。
        if stripped == "" or stripped.startswith("#"):
            new_lines.append(line)
            continue

        match = curve_line_pattern.match(line.rstrip("\r\n"))
        if not match:
            new_lines.append(line)
            continue

        raw_name = match.group("name")
        base_name = _resolve_curve_base_name(raw_name)
        if base_name is None:
            new_lines.append(line)
            continue

        unit = _unit_for_curve(base_name, sonic_unit=sonic_unit, density_unit=density_unit)
        indent = match.group("indent")
        newline = "\n" if line.endswith("\n") else ""
        new_lines.append(f"{indent}{base_name} .{unit:<20}: {base_name}{newline}")

    return "".join(new_lines)


def replace_smi_las_curve_section_in_file(
    file_path: Path,
    sonic_unit: str = "us/ft",
    density_unit: str = "g/cm3",
    encoding: str = "utf-8",
) -> Path:
    """对单个 LAS 文件执行 ``~Curve`` 段替换并输出到 output 子目录。

    Parameters
    ----------
    file_path : Path
        待处理 LAS 文件路径。
    sonic_unit : str, default="us/ft"
        声波曲线（Vp/Vs）默认单位。
    density_unit : str, default="g/cm3"
        密度曲线默认单位。
    encoding : str, default="utf-8"
        文件读写编码。

    Returns
    -------
    Path
        输出文件路径：``<原目录>/output/<原文件名>``。
    """
    file_path = Path(file_path)
    raw_text = file_path.read_text(encoding=encoding)
    new_text = replace_smi_las_curve_section(
        raw_text,
        sonic_unit=sonic_unit,
        density_unit=density_unit,
    )
    output_dir = file_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_path.name
    output_path.write_text(new_text, encoding=encoding)
    return output_path
