"""cup.utils.io: shared path, configuration, filename, and JSON helpers.

The functions in this module are independent of the geophysical domain.  They
operate on paths and basic serialized values without importing ``cup.seismic``,
``cup.well``, or ``wtie``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def resolve_relative_path(relative: str | Path, *, root: Path) -> Path:
    """返回绝对路径；相对路径会在 ``root`` 下解析。"""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def repo_relative_path(path: str | Path, *, root: Path) -> str:
    """返回相对 ``root`` 的可移植 POSIX 风格路径。

    仓库产物应使用本函数保存路径，避免写入本机专属绝对路径。
    """
    root = Path(root).resolve()
    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (root / p).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repository root and cannot be stored portably: {resolved}") from exc


def load_yaml_config(config_path: str | Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置文件，并可按 ``base_dir`` 解析相对路径。"""
    path = Path(config_path)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def sanitize_filename(name: str) -> str:
    """将文件名中的不安全字符替换为下划线。"""
    bad = {"/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"}
    return "".join("_" if c in bad else c for c in name)


def to_json_compatible(value: Any) -> Any:
    """递归地将输入转换为可 JSON 序列化的类型。

    支持 ``Path``、``numpy`` 标量/数组以及常见容器；非有限浮点数会转为
    ``null``。
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_json_compatible(value.item())
        return [to_json_compatible(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """用 UTF-8 和 2 空格缩进将 ``payload`` 写为 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(to_json_compatible(payload), fp, ensure_ascii=False, indent=2)
