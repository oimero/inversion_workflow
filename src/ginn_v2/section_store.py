"""Atomic on-disk cache for one-dimensional volume-prediction sections.

The section cache is deliberately small in scope.  A manifest freezes the
run contract, while each ``orientation_fixed-index.npz`` stores one complete
prediction section and its exact ordered patch identities.  A missing section
is a normal cache miss; an existing but malformed section is an error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

from ginn_v2.data import PatchKey


_MANIFEST_NAME = "manifest.json"
_SCHEMA_VERSION = 1
_SECTION_FIELDS = frozenset({"keys", "body_log_ai", "valid_mask"})


def _canonical_json(value: Any) -> str:
    """Return the stable JSON representation used for contract comparison."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Section prediction contract must be JSON-serializable JSON.") from exc


def _sample_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("Section prediction sample_count must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError("Section prediction sample_count must be a positive integer.")
    return result


class SectionPredictionStore:
    """Read and atomically write cached body-prediction sections.

    ``root`` may be ``None`` to disable caching.  A disabled or not-yet-created
    cache returns ``None`` from :meth:`read`; :meth:`write` requires a concrete
    directory so an accidental disabled write cannot silently discard output.
    """

    def __init__(
        self,
        root: Path | None,
        contract: Mapping[str, Any],
        *,
        sample_count: int | None = None,
    ) -> None:
        if not isinstance(contract, Mapping):
            raise TypeError("Section prediction contract must be a mapping.")
        contract_mapping = dict(contract)
        contract_json = _canonical_json(contract_mapping)
        resolved_sample_count = (
            sample_count
            if sample_count is not None
            else contract_mapping.get("sample_count")
        )
        if resolved_sample_count is None:
            raise ValueError("Section prediction sample_count must be passed or included in the contract.")
        self.sample_count = _sample_count(resolved_sample_count)
        if "sample_count" in contract_mapping:
            contract_sample_count = _sample_count(contract_mapping["sample_count"])
            if contract_sample_count != self.sample_count:
                raise ValueError("Section prediction sample_count disagrees with the contract.")
        self.root = None if root is None else Path(root)
        self.contract = json.loads(contract_json)
        self._contract_json = contract_json
        if self.root is not None:
            self._verify_existing_manifest()

    @property
    def manifest_path(self) -> Path | None:
        """The manifest path, or ``None`` when caching is disabled."""

        return None if self.root is None else self.root / _MANIFEST_NAME

    @staticmethod
    def _validate_orientation(orientation: str) -> str:
        if orientation not in {"inline", "xline"}:
            raise ValueError("Section orientation must be 'inline' or 'xline'.")
        return orientation

    @staticmethod
    def _validate_fixed_index(fixed_index: int) -> int:
        if isinstance(fixed_index, bool) or not isinstance(fixed_index, (int, np.integer)):
            raise TypeError("Section fixed_index must be an integer.")
        result = int(fixed_index)
        if result < 0:
            raise ValueError("Section fixed_index must be non-negative.")
        return result

    def _section_path(self, orientation: str, fixed_index: int) -> Path:
        orientation = self._validate_orientation(orientation)
        fixed_index = self._validate_fixed_index(fixed_index)
        if self.root is None:
            raise ValueError("Section prediction cache is disabled.")
        return self.root / f"{orientation}_{fixed_index}.npz"

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "sample_count": self.sample_count,
            "contract": self.contract,
        }

    def _read_manifest(self) -> dict[str, Any]:
        if self.root is None:
            raise ValueError("Section prediction cache is disabled.")
        manifest_path = self.root / _MANIFEST_NAME
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            raise ValueError(f"Corrupt section prediction manifest: {manifest_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Section prediction manifest must contain a JSON object.")
        if set(payload) != {"schema_version", "sample_count", "contract"}:
            raise ValueError("Section prediction manifest fields do not match the frozen schema.")
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported section prediction manifest schema version.")
        if _sample_count(payload.get("sample_count")) != self.sample_count:
            raise ValueError("Section prediction manifest sample_count differs from the requested contract.")
        try:
            stored_contract_json = _canonical_json(payload["contract"])
        except ValueError as exc:
            raise ValueError("Section prediction manifest contract is not canonical JSON.") from exc
        if stored_contract_json != self._contract_json:
            raise ValueError("Section prediction manifest contract differs from the requested contract.")
        return payload

    def _verify_existing_manifest(self) -> None:
        if self.root is None or not self.root.exists():
            return
        if not self.root.is_dir():
            raise ValueError(f"Section prediction cache root is not a directory: {self.root}")
        manifest_path = self.root / _MANIFEST_NAME
        if manifest_path.is_file():
            self._read_manifest()
            return
        # A directory containing only interrupted temporary writes is a normal
        # cache miss.  A final section without its manifest is an orphan and
        # must not be used as an unverified fallback.
        if any(path.is_file() and path.suffix == ".npz" for path in self.root.iterdir()):
            raise ValueError("Section prediction cache contains sections but no manifest.json.")

    def _ensure_manifest(self) -> None:
        if self.root is None:
            raise ValueError("Cannot write a section prediction when the cache is disabled.")
        self.root.mkdir(parents=True, exist_ok=True)
        self._verify_existing_manifest()
        manifest_path = self.root / _MANIFEST_NAME
        if manifest_path.is_file():
            return
        temp_path = self.root / f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    self._manifest_payload(),
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(manifest_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _validate_keys(
        self,
        orientation: str,
        fixed_index: int,
        keys: Sequence[PatchKey],
    ) -> tuple[tuple[PatchKey, ...], np.ndarray]:
        orientation = self._validate_orientation(orientation)
        fixed_index = self._validate_fixed_index(fixed_index)
        selected = tuple(keys)
        rows: list[tuple[int, int]] = []
        for key in selected:
            if not isinstance(key, PatchKey):
                raise TypeError("Section prediction keys must contain PatchKey values.")
            if key.orientation != orientation:
                raise ValueError("Section prediction key orientation differs from the section orientation.")
            key_fixed = key.inline_index if orientation == "inline" else key.xline_index
            if int(key_fixed) != fixed_index:
                raise ValueError("Section prediction key fixed index differs from the section fixed_index.")
            rows.append((int(key.inline_index), int(key.xline_index)))
        key_array = np.asarray(rows, dtype=np.int64).reshape((-1, 2))
        return selected, key_array

    def write(
        self,
        orientation: str,
        fixed_index: int,
        keys: Sequence[PatchKey],
        body_log_ai: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        """Atomically write one complete section and its ordered patch keys."""

        selected, key_array = self._validate_keys(orientation, fixed_index, keys)
        body = np.asarray(body_log_ai)
        support = np.asarray(valid_mask)
        if body.dtype != np.dtype(np.float32):
            raise ValueError("Section body_log_ai must have dtype float32.")
        if support.dtype != np.dtype(bool):
            raise ValueError("Section valid_mask must have dtype bool.")
        expected_shape = (len(selected), self.sample_count)
        if body.shape != expected_shape or support.shape != expected_shape:
            raise ValueError(f"Section arrays must both have shape {expected_shape}.")
        if np.any(support & ~np.isfinite(body)):
            raise ValueError("Section body_log_ai must be finite on valid_mask support.")
        self._ensure_manifest()
        path = self._section_path(orientation, fixed_index)
        temp_path = self.root / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temp_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    keys=key_array,
                    body_log_ai=body,
                    valid_mask=support,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def read(
        self,
        orientation: str,
        fixed_index: int,
        keys: Sequence[PatchKey],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Read a validated section, returning ``None`` for a normal miss."""

        _selected, expected_keys = self._validate_keys(orientation, fixed_index, keys)
        if self.root is None or not self.root.exists():
            return None
        self._verify_existing_manifest()
        path = self._section_path(orientation, fixed_index)
        if not path.is_file():
            return None
        try:
            archive = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Corrupt section prediction archive: {path}") from exc
        try:
            with archive:
                if frozenset(archive.files) != _SECTION_FIELDS:
                    raise ValueError("Section prediction archive fields do not match the frozen schema.")
                stored_keys = np.asarray(archive["keys"])
                body = np.asarray(archive["body_log_ai"])
                support = np.asarray(archive["valid_mask"])
                if stored_keys.dtype != np.dtype(np.int64):
                    raise ValueError("Section prediction keys must have dtype int64.")
                if stored_keys.ndim != 2 or stored_keys.shape != expected_keys.shape:
                    raise ValueError("Section prediction key array has an unexpected shape.")
                if body.dtype != np.dtype(np.float32):
                    raise ValueError("Stored section body_log_ai must have dtype float32.")
                if support.dtype != np.dtype(bool):
                    raise ValueError("Stored section valid_mask must have dtype bool.")
                expected_shape = (expected_keys.shape[0], self.sample_count)
                if body.shape != expected_shape or support.shape != expected_shape:
                    raise ValueError(f"Stored section arrays must both have shape {expected_shape}.")
                if not np.array_equal(stored_keys, expected_keys):
                    raise ValueError("Stored section PatchKey order or identities differ from the request.")
                if np.any(support & ~np.isfinite(body)):
                    raise ValueError("Stored section body_log_ai is non-finite on valid_mask support.")
                return np.array(body, dtype=np.float32, copy=True), np.array(support, dtype=bool, copy=True)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Corrupt section prediction archive: {path}") from exc


__all__ = ["SectionPredictionStore"]
