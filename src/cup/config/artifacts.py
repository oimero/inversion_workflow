"""Workflow artifact location and published-contract helpers.

This module owns run-directory discovery, artifact path resolution, and the
published contract status/fingerprint fields shared by workflow producers and
consumers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cup.utils.io import repo_relative_path


__all__ = [
    "CONTRACT_FINGERPRINT_SCHEMA",
    "CONSUMABLE_CONTRACT_STATUSES",
    "contract_fingerprint_sha256",
    "is_consumable_contract_status",
    "latest_checked_run",
    "published_contract_reference",
    "require_contract_fingerprint",
    "resolve_artifact_path",
    "resolve_repo_metadata_path",
    "resolve_timestamped_output_dir",
]


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


def latest_checked_run(
    output_root: str | Path,
    prefix: str,
    *,
    required_files: Sequence[str],
    validator: Callable[[Path], None] | None = None,
) -> Path:
    """Return the latest run directory that has required files and passes validation."""
    root = Path(output_root)
    candidates = [
        path
        for path in root.glob(f"{prefix}_*")
        if path.is_dir() and all((path / name).is_file() for name in required_files)
    ]
    checked = sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    rejected: list[str] = []
    for path in checked:
        if validator is None:
            return path
        try:
            validator(path)
        except Exception as exc:
            rejected.append(f"{path.name}: {exc}")
            continue
        return path
    detail = f" Rejected candidates: {'; '.join(rejected)}" if rejected else ""
    raise FileNotFoundError(
        f"No valid run found under {root} for {prefix}_* containing {list(required_files)!r}.{detail}"
    )


def resolve_timestamped_output_dir(
    output_root: str | Path,
    prefix: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Build ``<output_root>/<prefix>_<timestamp>`` without creating it."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / f"{prefix}_{ts}"


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


CONTRACT_FINGERPRINT_SCHEMA = "contract_fingerprint_v1"
CONSUMABLE_CONTRACT_STATUSES = frozenset(
    {
        "success",
        "ok",
        "completed_with_warnings",
        "development_limited",
        "needs_forward_diagnostic",
    }
)


def is_consumable_contract_status(value: Any) -> bool:
    """Return whether a published run may be consumed downstream."""
    return str(value or "") in CONSUMABLE_CONTRACT_STATUSES


_NON_BUSINESS_CONFIG_KEYS = {
    "created_at",
    "completed_at",
    "device",
    "devices",
    "diagnostic",
    "diagnostics",
    "figure",
    "figures",
    "log",
    "logging",
    "log_level",
    "num_threads",
    "num_workers",
    "output",
    "output_dir",
    "output_root",
    "preprocessed_las",
    "report_card",
    "run_id",
    "thread_count",
    "threads",
    "timestamp",
    "updated_at",
    "visualization",
    "visualizations",
    "workers",
}
_NON_BUSINESS_CONFIG_SUFFIXES = (
    "_path",
    "_paths",
    "_dir",
    "_dirs",
    "_directory",
    "_directories",
    "_file",
    "_files",
    "_root",
)

_ARTIFACT_LOCATION_KEYS = {
    "completed_at",
    "config_file",
    "config_provenance",
    "created_at",
    "device",
    "devices",
    "log",
    "logging",
    "log_level",
    "output_dir",
    "output_root",
    "preprocessed_las",
    "report_card",
    "run_id",
    "timestamp",
    "updated_at",
}


def _canonical_contract_value(value: Any) -> Any:
    """Convert a contract fingerprint input to strict canonical JSON values."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _canonical_contract_value(value.item())
        return [_canonical_contract_value(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Contract fingerprint inputs must not contain NaN or Infinity.")
        return result
    if isinstance(value, np.generic):
        return _canonical_contract_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_contract_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_contract_value(item) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(
        "Contract fingerprint inputs must be JSON values, Path objects, or numpy values; "
        f"got {type(value).__name__}."
    )


def _canonical_business_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove location/runtime-only keys before canonical JSON encoding."""

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                normalized = key.casefold().replace("-", "_")
                if (
                    normalized in _NON_BUSINESS_CONFIG_KEYS
                    or normalized.startswith(
                        ("debug_", "diagnostic_", "figure_", "log_", "plot_")
                    )
                    or normalized.endswith(_NON_BUSINESS_CONFIG_SUFFIXES)
                ):
                    continue
                result[key] = visit(raw_value)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(value) for value in item]
        if isinstance(item, Path):
            # Paths should normally be attached to a *_path/*_dir key. Keeping a
            # neutral marker here also prevents an unusually named Path value from
            # making an otherwise identical contract location-dependent.
            return None
        return _canonical_contract_value(item)

    return visit(value)


def _is_artifact_location_key(key: str) -> bool:
    normalized = str(key).casefold().replace("-", "_")
    return (
        normalized in _ARTIFACT_LOCATION_KEYS
        or normalized.startswith("log_")
        or normalized.endswith(_NON_BUSINESS_CONFIG_SUFFIXES)
    )


def _canonical_artifact_value(value: Any) -> Any:
    """Remove filesystem/runtime provenance from structured primary content."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_artifact_value(item)
            for key, item in value.items()
            if not _is_artifact_location_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_artifact_value(item) for item in value]
    return _canonical_contract_value(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_artifact_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    """SHA-256 digest of a file, streamed in fixed-size blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _structured_primary_artifact_sha256(path: Path) -> str:
    """Hash logical structured content while ignoring stored filesystem locations."""
    suffix = path.suffix.casefold()
    digest = hashlib.sha256()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            digest.update(b"json\0")
            digest.update(_canonical_json_bytes(json.load(handle)))
        return digest.hexdigest()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Primary CSV has no header: {path}")
            fields = [
                name
                for name in reader.fieldnames
                if not _is_artifact_location_key(name)
            ]
            rows = [{name: row.get(name, "") for name in fields} for row in reader]
        digest.update(b"csv\0")
        digest.update(_canonical_json_bytes({"fields": fields, "rows": rows}))
        return digest.hexdigest()
    if suffix == ".npz":
        digest.update(b"npz\0")
        with np.load(path, allow_pickle=False) as arrays:
            for name in sorted(arrays.files):
                value = np.asarray(arrays[name])
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                if (
                    name.endswith("metadata_json")
                    and value.ndim == 0
                    and value.dtype.kind in {"S", "U"}
                ):
                    # metadata_json is a logical JSON document. Its numpy string
                    # storage width only reflects serialized text length and must
                    # not leak path length into the published contract identity.
                    digest.update(b"canonical-json\0")
                    raw_metadata = value.item()
                    if isinstance(raw_metadata, bytes):
                        raw_metadata = raw_metadata.decode("utf-8")
                    digest.update(_canonical_json_bytes(json.loads(str(raw_metadata))))
                else:
                    digest.update(value.dtype.str.encode("ascii"))
                    digest.update(b"\0")
                    digest.update(
                        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
                    )
                    digest.update(b"\0")
                    digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()
    return _sha256_file(path)


def require_contract_fingerprint(payload: Mapping[str, Any], *, label: str) -> str:
    """Return the single published fingerprint from a successful contract."""
    status = str(payload.get("status") or "").casefold()
    if status not in CONSUMABLE_CONTRACT_STATUSES:
        raise ValueError(
            f"{label} is not a consumable published contract: status={status!r}."
        )
    if str(payload.get("contract_fingerprint_schema") or "") != CONTRACT_FINGERPRINT_SCHEMA:
        raise ValueError(f"{label} does not use {CONTRACT_FINGERPRINT_SCHEMA}.")
    digest = str(payload.get("contract_fingerprint_sha256") or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} has an invalid contract_fingerprint_sha256.")
    return digest


def published_contract_reference(
    summary_path: str | Path,
    *,
    root: Path,
    label: str,
) -> dict[str, str]:
    """Load one upstream publish manifest without re-hashing any artifact."""
    path = Path(summary_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} publish manifest must be a JSON object: {path}")
    return {
        "path": repo_relative_path(path, root=root),
        "contract_fingerprint_sha256": require_contract_fingerprint(payload, label=label),
    }


def contract_fingerprint_sha256(
    *,
    contract_schema_version: str,
    semantics: Mapping[str, Any],
    business_config: Mapping[str, Any],
    input_contracts: Mapping[str, str | Mapping[str, Any]],
    primary_artifacts: Mapping[str, str | Path],
) -> str:
    """Compute one producer-side fingerprint for an immutable published contract.

    Per-file digests are deliberately transient: only the returned aggregate digest
    belongs in the published manifest. Consumers must copy that digest into their
    own ``input_contracts`` and must not call this function for upstream files.
    """
    schema = str(contract_schema_version).strip()
    if not schema:
        raise ValueError("contract_schema_version must be explicit.")
    inputs: dict[str, str] = {}
    for role, value in input_contracts.items():
        if isinstance(value, Mapping):
            digest = str(value.get("contract_fingerprint_sha256") or "")
        else:
            digest = str(value)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"input_contracts.{role} must be a lowercase SHA-256 digest.")
        inputs[str(role)] = digest
    artifacts: dict[str, str] = {}
    for logical_name, path_value in primary_artifacts.items():
        name = str(logical_name).strip()
        if not name or name in artifacts:
            raise ValueError("Primary artifact logical names must be non-empty and unique.")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Primary contract artifact does not exist: {name} -> {path}")
        artifacts[name] = _structured_primary_artifact_sha256(path)
    if not artifacts:
        raise ValueError("A published contract requires at least one primary artifact.")
    fingerprint_payload = {
        "fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_schema_version": schema,
        "semantics": _canonical_contract_value(semantics),
        "business_config": _canonical_business_config(business_config),
        "input_contracts": _canonical_contract_value(inputs),
        "primary_artifacts": _canonical_contract_value(artifacts),
    }
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
