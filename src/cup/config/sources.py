"""Workflow source-run discovery and validation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cup.utils.io import latest_checked_run, resolve_relative_path


def _path_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _run_prefix(value: str) -> str:
    return value[:-1] if value.endswith("_") else value


def require_source_files(directory: Path, names: Sequence[str], *, label: str) -> None:
    """Require a source run directory and its key files."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} directory is missing required files {missing}: {directory}")


def load_summary(
    path: Path,
    *,
    schema_version: str | None = None,
    allowed_status: set[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Load a JSON summary and optionally validate schema/status."""
    if not path.is_file():
        raise FileNotFoundError(f"{label or path.name} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if schema_version is not None and str(payload.get("schema_version") or "") != schema_version:
        raise ValueError(f"{label or path.name} schema_version is not {schema_version}")
    if allowed_status is not None and str(payload.get("status") or "") not in allowed_status:
        raise ValueError(f"{label or path.name} status is not consumable: {payload.get('status')}")
    return payload


def assert_same_path(left: str | Path, right: str | Path, *, root: Path, message: str) -> None:
    """Raise when two repo-relative/absolute paths do not resolve to the same path."""
    if resolve_relative_path(left, root=root).resolve() != resolve_relative_path(right, root=root).resolve():
        raise ValueError(message)


def assert_recorded_source_matches(
    recorded: dict[str, Any],
    key: str,
    actual: Path,
    *,
    root: Path,
    message: str | None = None,
) -> None:
    """Validate that a summary-recorded source path matches a resolved source path."""
    text = _path_text(recorded.get(key))
    if not text:
        raise ValueError(f"source_run_mismatch: summary lacks {key}")
    assert_same_path(
        text,
        actual,
        root=root,
        message=message or f"source_run_mismatch:{key}",
    )


def resolve_source_run(
    explicit: str | Path | None,
    *,
    output_root: Path,
    prefix: str,
    required_files: Sequence[str],
    root: Path,
    label: str | None = None,
    summary_file: str | None = None,
    schema_version: str | None = None,
    allowed_status: set[str] | None = None,
) -> Path:
    """Resolve an explicit or latest checked workflow source run."""
    source_label = label or prefix
    text = _path_text(explicit)
    if text:
        path = resolve_relative_path(text, root=root)
        require_source_files(path, required_files, label=source_label)
        if summary_file is not None:
            load_summary(
                path / summary_file,
                schema_version=schema_version,
                allowed_status=allowed_status,
                label=summary_file,
            )
        return path
    validator = None
    if summary_file is not None:
        def validator(path: Path) -> None:
            load_summary(
                path / summary_file,
                schema_version=schema_version,
                allowed_status=allowed_status,
                label=summary_file,
            )
    return latest_checked_run(
        output_root,
        _run_prefix(prefix),
        required_files=required_files,
        validator=validator,
    )


def resolve_source_file_from_run(
    explicit: str | Path | None,
    *,
    output_root: Path,
    prefix: str,
    file_name: str,
    root: Path,
    run_required_files: Sequence[str] | None = None,
    label: str | None = None,
    summary_file: str | None = None,
    schema_version: str | None = None,
    allowed_status: set[str] | None = None,
) -> Path:
    """Resolve an explicit source file or a file inside the latest checked run."""
    text = _path_text(explicit)
    if text:
        path = resolve_relative_path(text, root=root)
        if not path.is_file():
            raise FileNotFoundError(f"{label or file_name} not found: {path}")
        return path
    required = list(run_required_files or [file_name])
    if file_name not in required:
        required.append(file_name)
    run_dir = resolve_source_run(
        None,
        output_root=output_root,
        prefix=prefix,
        required_files=required,
        root=root,
        label=label or prefix,
        summary_file=summary_file,
        schema_version=schema_version,
        allowed_status=allowed_status,
    )
    return run_dir / file_name
