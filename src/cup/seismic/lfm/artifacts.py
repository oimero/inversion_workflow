"""Strict resolver for published unified LFM v2 variants."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.seismic.lfm.pipeline import RUN_SCHEMA, VARIANT_SCHEMA
from cup.utils.io import resolve_relative_path, sha256_file
from cup.well.real_field_controls import SCHEMA_VERSION as WELL_CONTROL_SCHEMA


@dataclass(frozen=True)
class ResolvedLfmVariant:
    run_dir: Path
    variant_id: str
    variant_dir: Path
    lfm_path: Path
    lfm_sha256: str
    run_summary_path: Path
    run_summary: Mapping[str, Any]
    variant_metadata: Mapping[str, Any]
    well_control_run_dir: Path
    well_control_run_sha256: str


def _required_text(config: Mapping[str, Any], key: str) -> str:
    text = str(config.get(key) or "").strip()
    if not text or text.casefold() == "auto":
        raise ValueError(f"real_field_inputs.{key} must be an explicit non-auto value.")
    return text


def resolve_lfm_variant(inputs: Mapping[str, Any], *, repo_root: Path) -> ResolvedLfmVariant:
    run_dir = resolve_relative_path(_required_text(inputs, "lfm_run_dir"), root=repo_root)
    variant_id = _required_text(inputs, "variant_id")
    well_control_run = resolve_relative_path(_required_text(inputs, "well_control_run_dir"), root=repo_root)
    run_summary_path = run_dir / "lfm_run_summary.json"
    manifest_path = run_dir / "variant_manifest.csv"
    if not run_summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete unified LFM run: {run_dir}")
    with run_summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != RUN_SCHEMA or summary.get("status") != "ok":
        raise ValueError("R0 accepts only successful real_field_lfm_run_v2 runs.")
    _validate_path_hash(dict(summary.get("seismic") or {}), repo_root=repo_root, label="seismic")
    for index, horizon in enumerate(summary.get("horizons") or []):
        _validate_path_hash(dict(horizon), repo_root=repo_root, label=f"horizon[{index}]")
    manifest_info = dict(dict(summary.get("outputs") or {}).get("variant_manifest") or {})
    recorded_manifest_path = resolve_relative_path(str(manifest_info.get("path") or ""), root=repo_root)
    if (
        recorded_manifest_path.resolve() != manifest_path.resolve()
        or str(manifest_info.get("sha256") or "") != sha256_file(manifest_path)
    ):
        raise ValueError("variant_manifest.csv SHA-256 mismatch.")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "variant_id",
        "baseline_id",
        "baseline_method",
        "modifier_chain",
        "status",
        "lfm_path",
        "lfm_sha256",
        "method_fields_path",
        "method_fields_sha256",
        "modifier_fields_path",
        "modifier_fields_sha256",
        "variant_summary_path",
        "variant_summary_sha256",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"variant_manifest.csv is missing columns: {missing}")
    manifest_ids = manifest["variant_id"].astype(str)
    if manifest_ids.duplicated().any() or not manifest["status"].astype(str).eq("ok").all():
        raise ValueError("Published variant manifest must contain unique, all-successful variants.")
    requested_ids = [str(value) for value in summary.get("requested_variant_ids") or []]
    if set(manifest_ids) != set(requested_ids) or len(manifest_ids) != len(requested_ids):
        raise ValueError("variant_manifest.csv does not exactly match the run's requested variants.")
    selected = manifest[manifest["variant_id"].astype(str).eq(variant_id)]
    if len(selected) != 1 or str(selected.iloc[0]["status"]) != "ok":
        raise ValueError(f"Requested variant_id is absent or not successful: {variant_id!r}")
    row = selected.iloc[0]
    lfm_path = resolve_relative_path(str(row["lfm_path"]), root=repo_root)
    variant_summary_path = resolve_relative_path(str(row["variant_summary_path"]), root=repo_root)
    if not lfm_path.is_file() or sha256_file(lfm_path) != str(row["lfm_sha256"]):
        raise ValueError(f"Selected variant LFM SHA-256 mismatch: {variant_id}")
    if not variant_summary_path.is_file() or sha256_file(variant_summary_path) != str(row["variant_summary_sha256"]):
        raise ValueError(f"Selected variant summary SHA-256 mismatch: {variant_id}")
    with variant_summary_path.open("r", encoding="utf-8") as handle:
        variant_summary = json.load(handle)
    summary_lfm = dict(dict(variant_summary.get("outputs") or {}).get("lfm") or {})
    if (
        variant_summary.get("schema_version") != VARIANT_SCHEMA
        or variant_summary.get("status") != "ok"
        or variant_summary.get("variant_id") != variant_id
        or str(summary_lfm.get("sha256") or "") != str(row["lfm_sha256"])
        or resolve_relative_path(str(summary_lfm.get("path") or ""), root=repo_root).resolve()
        != lfm_path.resolve()
    ):
        raise ValueError(f"Selected variant summary does not match its manifest row: {variant_id}")
    with np.load(lfm_path, allow_pickle=False) as data:
        if set(data.files) != {"log_ai", "valid_mask_model", "ilines", "xlines", "samples", "metadata_json"}:
            raise ValueError("Selected variant lfm.npz does not match the minimal v2 primary schema.")
        if data["log_ai"].dtype != np.dtype("float32") or data["valid_mask_model"].dtype != np.dtype("bool"):
            raise ValueError("Selected variant log_ai/valid_mask_model dtypes must be float32/bool.")
        if any(data[key].dtype != np.dtype("float64") for key in ("ilines", "xlines", "samples")):
            raise ValueError("Selected variant axis dtypes must be float64.")
        metadata_array = np.asarray(data["metadata_json"])
        if metadata_array.ndim != 0 or metadata_array.dtype.kind not in {"U", "S"}:
            raise ValueError("Selected variant metadata_json must be a scalar string without pickle.")
        log_ai = np.asarray(data["log_ai"], dtype=np.float64)
        valid_mask = np.asarray(data["valid_mask_model"], dtype=bool)
        ilines = np.asarray(data["ilines"], dtype=np.float64)
        xlines = np.asarray(data["xlines"], dtype=np.float64)
        samples = np.asarray(data["samples"], dtype=np.float64)
        metadata = json.loads(str(metadata_array.item()))
    if log_ai.shape != valid_mask.shape or log_ai.ndim not in {2, 3}:
        raise ValueError("Selected variant log_ai/mask shape is invalid.")
    if not np.all(np.isfinite(log_ai[valid_mask])) or np.any(np.isfinite(log_ai[~valid_mask])):
        raise ValueError("Selected variant violates finite-inside/NaN-outside mask semantics.")
    for name, axis in (("ilines", ilines), ("xlines", xlines), ("samples", samples)):
        if axis.ndim != 1 or axis.size == 0 or np.any(~np.isfinite(axis)):
            raise ValueError(f"Selected variant {name} axis is invalid.")
    if np.any(np.diff(samples) <= 0.0):
        raise ValueError("Selected variant samples axis must be strictly increasing.")
    expected_shape = (ilines.size, samples.size) if log_ai.ndim == 2 else (ilines.size, xlines.size, samples.size)
    if log_ai.shape != expected_shape or (log_ai.ndim == 2 and xlines.size != ilines.size):
        raise ValueError("Selected variant axes do not match log_ai shape.")
    output_mode = str(dict(metadata.get("output_geometry") or {}).get("mode") or "")
    if log_ai.ndim == 2:
        if output_mode != "section":
            raise ValueError("A two-dimensional LFM primary must declare section output geometry.")
    elif output_mode not in {"window", "volume"}:
        raise ValueError("A three-dimensional LFM primary must declare window or volume output geometry.")
    elif np.any(np.diff(ilines) <= 0.0) or np.any(np.diff(xlines) <= 0.0):
        raise ValueError("Volume/window inline and xline axes must be strictly increasing.")
    if metadata.get("schema_version") != VARIANT_SCHEMA:
        if metadata.get("schema_version") == "real_field_lfm_v1":
            raise ValueError("real_field_lfm_v1 is retired; rebuild Step 6/7 with unified LFM v2.")
        raise ValueError(f"Unsupported LFM variant schema: {metadata.get('schema_version')!r}")
    if metadata.get("variant_id") != variant_id or metadata.get("value_key") != "log_ai":
        raise ValueError("Selected LFM metadata does not match requested variant/value contract.")
    if (
        metadata.get("value_domain") != "log(AI)"
        or metadata.get("linear_ai_unit") != "m/s*g/cm3"
        or metadata.get("valid_mask_key") != "valid_mask_model"
    ):
        raise ValueError("Selected LFM metadata has invalid value-domain/unit/mask semantics.")
    sample_domain = str(metadata.get("sample_domain") or "")
    expected_unit = "s" if sample_domain == "time" else "m" if sample_domain == "depth" else None
    if expected_unit is None or metadata.get("sample_unit") != expected_unit:
        raise ValueError("Selected LFM metadata has an invalid sample domain/unit pair.")
    if (sample_domain == "depth" and metadata.get("depth_basis") != "tvdss") or (
        sample_domain == "time" and metadata.get("depth_basis") is not None
    ):
        raise ValueError("Selected LFM metadata depth_basis is inconsistent with sample_domain.")
    if not str(metadata.get("baseline_id") or "").strip() or not str(
        metadata.get("baseline_method") or ""
    ).strip() or not isinstance(metadata.get("modifier_chain"), list):
        raise ValueError("Selected LFM metadata lacks baseline/modifier identity.")
    expected_modifier_chain = ";".join(str(value) for value in metadata["modifier_chain"])
    method_sidecar = dict(metadata.get("method_sidecar_path_and_hash") or {})
    modifier_sidecar = metadata.get("modifier_sidecar_path_and_hash")
    if (
        str(row["baseline_id"]) != str(metadata["baseline_id"])
        or str(row["baseline_method"]) != str(metadata["baseline_method"])
        or str(row["modifier_chain"]) != expected_modifier_chain
        or str(row["method_fields_path"]) != str(method_sidecar.get("path") or "")
        or str(row["method_fields_sha256"]) != str(method_sidecar.get("sha256") or "")
        or str(row["modifier_fields_path"])
        != ("" if modifier_sidecar is None else str(dict(modifier_sidecar).get("path") or ""))
        or str(row["modifier_fields_sha256"])
        != ("" if modifier_sidecar is None else str(dict(modifier_sidecar).get("sha256") or ""))
    ):
        raise ValueError("Selected variant manifest row disagrees with lfm.npz metadata.")
    if dict(variant_summary.get("metadata") or {}) != metadata:
        raise ValueError("Selected variant summary metadata does not match lfm.npz metadata_json.")
    if dict(metadata.get("seismic_path_and_hash") or {}) != dict(summary.get("seismic") or {}):
        raise ValueError("Selected variant and run summary record different seismic provenance.")
    if list(metadata.get("horizon_paths_and_hashes") or []) != list(summary.get("horizons") or []):
        raise ValueError("Selected variant and run summary record different horizon provenance.")
    _validate_path_hash(
        dict(metadata.get("method_sidecar_path_and_hash") or {}),
        repo_root=repo_root,
        label="method sidecar",
    )
    modifier_sidecar = metadata.get("modifier_sidecar_path_and_hash")
    if modifier_sidecar is not None:
        _validate_path_hash(dict(modifier_sidecar), repo_root=repo_root, label="modifier sidecar")
    for modifier_id, source in dict(metadata.get("modifier_source_paths_and_hashes") or {}).items():
        body_source = dict(source)
        _validate_path_hash(
            {"path": body_source.get("bodies_file"), "sha256": body_source.get("bodies_file_sha256")},
            repo_root=repo_root,
            label=f"modifier source {modifier_id}",
        )
    controls_summary_path = well_control_run / "run_summary.json"
    if not controls_summary_path.is_file():
        raise FileNotFoundError(controls_summary_path)
    with controls_summary_path.open("r", encoding="utf-8") as handle:
        controls_summary = json.load(handle)
    if controls_summary.get("schema_version") != WELL_CONTROL_SCHEMA or controls_summary.get("status") != "ok":
        raise ValueError("Configured well_control_run_dir is not a successful v2 run.")
    controls_manifest = well_control_run / "well_control_manifest.csv"
    control_outputs = dict(controls_summary.get("outputs") or {})
    expected_controls_manifest_hash = str(control_outputs.get("well_control_manifest_sha256") or "")
    recorded_controls_manifest = resolve_relative_path(
        str(control_outputs.get("well_control_manifest") or ""), root=repo_root
    )
    if not controls_manifest.is_file() or not expected_controls_manifest_hash:
        raise ValueError("Configured WellControlSet manifest provenance is incomplete.")
    if (
        recorded_controls_manifest.resolve() != controls_manifest.resolve()
        or sha256_file(controls_manifest) != expected_controls_manifest_hash
    ):
        raise ValueError("Configured WellControlSet manifest SHA-256 mismatch.")
    actual_control_hash = sha256_file(controls_summary_path)
    recorded_control = dict(metadata.get("well_control_run_path_and_hash") or {})
    recorded_path = resolve_relative_path(str(recorded_control.get("path") or ""), root=repo_root)
    if recorded_path.resolve() != well_control_run.resolve() or recorded_control.get("run_summary_sha256") != actual_control_hash:
        raise ValueError("Selected LFM variant and configured WellControlSet hash chain do not match.")
    run_control = dict(summary.get("well_control_run") or {})
    run_control_path = resolve_relative_path(str(run_control.get("path") or ""), root=repo_root)
    if (
        run_control_path.resolve() != well_control_run.resolve()
        or str(run_control.get("run_summary_sha256") or "") != actual_control_hash
    ):
        raise ValueError("LFM run summary and selected variant record different WellControlSet provenance.")
    return ResolvedLfmVariant(
        run_dir=run_dir,
        variant_id=variant_id,
        variant_dir=lfm_path.parent,
        lfm_path=lfm_path,
        lfm_sha256=str(row["lfm_sha256"]),
        run_summary_path=run_summary_path,
        run_summary=summary,
        variant_metadata=metadata,
        well_control_run_dir=well_control_run,
        well_control_run_sha256=actual_control_hash,
    )


def _validate_path_hash(payload: Mapping[str, Any], *, repo_root: Path, label: str) -> None:
    path_text = str(payload.get("path") or "").strip()
    expected = str(payload.get("sha256") or "").strip()
    if not path_text or not expected:
        raise ValueError(f"{label} path/hash provenance is incomplete.")
    path = resolve_relative_path(path_text, root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch.")


__all__ = ["ResolvedLfmVariant", "resolve_lfm_variant"]
