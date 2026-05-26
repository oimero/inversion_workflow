"""cup.utils.config: 配置合并辅助工具。

本模块提供脚本配置段的默认值合并逻辑。

边界说明
--------
- 不依赖任何地球物理库。
- 合并是就地修改，调用方如需保留原始配置应自行深拷贝。

核心公开对象
------------
1. merge_dict_defaults: 将默认值字典合并到配置 key 下。
"""

from __future__ import annotations

from typing import Any


def merge_dict_defaults(config: dict[str, Any], key: str, defaults: dict[str, Any]) -> None:
    """Merge *defaults* into ``config[key]`` in place.

    If ``config[key]`` is ``None`` or missing it is set to a copy of
    *defaults*.  If it is already a mapping, *defaults* are applied
    underneath it (existing keys kept).
    """
    value = config.get(key)
    if value is None:
        config[key] = dict(defaults)
        return
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping, got {type(value).__name__}.")
    merged = dict(defaults)
    merged.update(value)
    config[key] = merged
