"""cup.utils.io: 项目共享基础设施与 I/O 辅助工具。

本模块的函数与地球物理库无关，仅依赖 Python 标准库与基础三方库。

边界说明
--------
- 本模块不依赖 ``wtie``、``cup.seismic`` 或 ``cup.well`` 中的任何模块。
- 路径工具统一以 ``root`` 参数为基准，不在模块内保存全局状态。

核心公开对象
------------
1. resolve_relative_path / repo_relative_path: 路径解析与可移植路径表示。
2. load_yaml_config / write_json: 配置文件加载与 JSON 写出。
3. sanitize_filename: 文件名安全化。
4. build_segy_textual_header: SEG-Y 文本头构造。
5. latest_run / resolve_timestamped_output_dir: 工作流 run 目录发现与构造。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Path resolution ──


def resolve_optional_path(value: str | Path | None, *, root: Path) -> Path | None:
    """Parse a config-level optional path value.

    Returns ``None`` for empty strings, ``"none"``, ``"null"``, ``"nan"``
    (case-insensitive), or ``None``.  Otherwise resolves the value relative
    to *root* and returns the resolved absolute path.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return None
    return resolve_relative_path(text, root=root)


def resolve_artifact_path(value: Any, *, root: Path, run_dir: Path) -> Path | None:
    """Resolve a repo-relative artifact path from metadata, falling back to
    *run_dir*-relative when the absolute path does not exist.

    *root* is typically ``REPO_ROOT``.  Returns ``None`` for empty / sentinel
    strings.
    """
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [root / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def latest_run(output_root: str | Path, prefix: str, required_file: str) -> Path:
    """Return the latest ``<prefix>_*`` run directory containing ``required_file``.

    This helper only discovers directories. Callers must still validate that
    the required file conforms to the expected CSV/NPZ schema.
    """
    root = Path(output_root)
    pattern = f"{prefix}_*"
    candidates = [p for p in root.glob(pattern) if p.is_dir() and (p / required_file).exists()]
    if not candidates:
        raise FileNotFoundError(f"No run found under {root} for {pattern} containing {required_file!r}.")
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def resolve_timestamped_output_dir(
    output_root: str | Path,
    prefix: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Build ``<output_root>/<prefix>_<timestamp>`` without creating it."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / f"{prefix}_{ts}"


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


def resolve_repo_metadata_path(value: str | Path, *, root: Path) -> Path:
    """解析产物元数据中保存的仓库相对路径。

    本函数会拒绝绝对路径，让过期的本机路径明确失败。
    """
    p = Path(value)
    if p.is_absolute():
        raise ValueError(
            f"Artifact metadata contains an absolute path, which is not portable: {p}. "
            "Regenerate the artifact with repo-relative metadata."
        )
    resolved = (Path(root).resolve() / p).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact metadata path does not exist: {value} -> {resolved}")
    return resolved


# ── Config loading ──


def load_yaml_config(config_path: str | Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置文件，并可按 ``base_dir`` 解析相对路径。"""
    path = Path(config_path)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


# ── String sanitisation ──


def sanitize_filename(name: str) -> str:
    """将文件名中的不安全字符替换为下划线。"""
    bad = {"/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"}
    return "".join("_" if c in bad else c for c in name)


# ── SEG-Y ──


def build_segy_textual_header(title: str, lines: list[str] | None = None) -> str:
    """根据标题和附加行构造 3200 字节 SEG-Y 文本头。"""
    all_lines = [title] + (lines or [])
    rows = [f"C{i:>2d} {text}"[:80].ljust(80) for i, text in enumerate(all_lines, start=1)]
    rows.extend([f"C{i:>2d}".ljust(80) for i in range(len(rows) + 1, 41)])
    textual = "".join(rows)
    if len(textual) != 3200:
        raise ValueError(f"Expected 3200-char textual header, got {len(textual)}")
    return textual


# ── JSON serialization ──


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
