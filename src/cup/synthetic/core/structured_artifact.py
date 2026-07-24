"""Producer-owned structured truth artifacts for the new inversion seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from cup.synthetic.core.records import SampleAxis, StructuredSampleRecord


ARTIFACT_TYPE = "structured_truth_v1"
ARTIFACT_VERSION = 1
ARRAY_NAMES = (
    "observed_axis",
    "observed_seismic",
    "model_consistent_seismic",
    "observed_lfm",
    "observed_valid",
    "latent_axis",
    "latent_log_ai_highres_truth",
    "latent_valid",
    "latent_state_id",
    "latent_object_id",
    "latent_object_xi",
    "latent_zone_id",
    "latent_clipping_mask",
    "zone_valid",
)
IDENTITY_KEYS = ("producer", "calibration", "projection", "forward")
STRUCTURED_SAMPLE_INDEX_COLUMNS = (
    "realization_id",
    "scenario_id",
    "sample_domain",
    "sample_unit",
    "depth_basis",
    "lateral_index",
    "zone_id",
    "artifact_path",
    "lateral_m",
    "inline",
    "xline",
    "xline_step",
)


def _safe_component(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("structured artifact path component must be non-empty")
    safe = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    )
    return safe or "_"


def _axis_manifest(axis: SampleAxis) -> dict[str, Any]:
    return {
        "sample_domain": axis.sample_domain,
        "unit": axis.unit,
        "sample_interval": float(axis.sample_interval),
        "positive_direction": axis.positive_direction,
        "depth_basis": axis.depth_basis,
    }


def _required_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("structured_identity must be a mapping")
    missing = sorted(set(IDENTITY_KEYS).difference(value))
    if missing:
        raise ValueError(f"structured_identity is missing required fields: {missing}")
    result = dict(value)
    json.dumps(result, allow_nan=False)
    return result


def _segment_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "zone_id",
        "object_id",
        "state",
        "state_id",
        "top",
        "bottom",
        "duration_fraction",
        "duration_samples",
        "c0_raw",
        "c1_raw",
        "c2_raw",
        "c0_projected",
        "c1_projected",
        "c2_projected",
        "c0_effective",
        "c1_effective",
        "c2_effective",
        "segment_supervision_valid",
    )
    missing = sorted(set(required).difference(row))
    if missing:
        raise ValueError(f"structured segment truth is missing fields: {missing}")
    result: dict[str, Any] = {
        "zone_id": str(row["zone_id"]),
        "object_id": int(row["object_id"]),
        "state": str(row["state"]),
        "state_id": int(row["state_id"]),
        "top": float(row["top"]),
        "bottom": float(row["bottom"]),
        "duration_fraction": float(row["duration_fraction"]),
        "duration_samples": int(row["duration_samples"]),
        "segment_supervision_valid": bool(row["segment_supervision_valid"]),
    }
    for name in (
        "c0_raw",
        "c1_raw",
        "c2_raw",
        "c0_projected",
        "c1_projected",
        "c2_projected",
        "c0_effective",
        "c1_effective",
        "c2_effective",
    ):
        result[name] = float(row[name])
    return result


def _validate_forward_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "structured artifact requires explicit forward context metadata"
        )
    result = dict(value)
    required = {"wavelet_time_s", "wavelet_amplitude", "ai_velocity_relation", "output_chunk_size"}
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"structured forward context is missing fields: {missing}")
    wavelet_time = np.asarray(result["wavelet_time_s"], dtype=np.float64).reshape(-1)
    wavelet_amplitude = np.asarray(result["wavelet_amplitude"], dtype=np.float64).reshape(-1)
    if (
        wavelet_time.size < 3
        or wavelet_time.size != wavelet_amplitude.size
        or wavelet_time.size % 2 == 0
        or np.any(~np.isfinite(wavelet_time))
        or np.any(~np.isfinite(wavelet_amplitude))
    ):
        raise ValueError("structured forward context wavelet is invalid")
    relation = result["ai_velocity_relation"]
    if relation is not None and not isinstance(relation, Mapping):
        raise TypeError("structured forward context ai_velocity_relation must be a mapping or null")
    result["wavelet_time_s"] = wavelet_time.tolist()
    result["wavelet_amplitude"] = wavelet_amplitude.tolist()
    result["ai_velocity_relation"] = None if relation is None else dict(relation)
    result["output_chunk_size"] = int(result["output_chunk_size"])
    if result["output_chunk_size"] <= 0:
        raise ValueError("structured forward context output_chunk_size must be positive")
    json.dumps(result, allow_nan=False)
    return result


def _forward_context_manifest(record: StructuredSampleRecord) -> dict[str, Any]:
    return _validate_forward_context(
        record.forward.metadata.get("structured_forward_context")
    )


def _remove_exact_directory(path: Path) -> None:
    """Remove one known artifact directory without following a directory link."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def remove_structured_truth_v1(
    root: str | Path,
    realization_id: str,
) -> None:
    """Rollback exactly one published structured realization.

    The shared pipeline calls this when a later seismic-view operation rejects
    a parent after the structured writer has already committed its own
    realization directory.  The target is validated to remain directly under
    ``root/realizations`` before removal.
    """
    realizations = (Path(root) / "realizations").resolve()
    target = realizations / _safe_component(realization_id)
    if target.parent != realizations:
        raise ValueError("structured realization rollback escaped its artifact root")
    _remove_exact_directory(target)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read structured {label}: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"structured {label} must contain a JSON object: {path}")
    return value


def validate_structured_truth_v1(
    root: str | Path,
    *,
    expected_realization_ids: Sequence[str],
) -> dict[str, Any]:
    """Audit the published structured tree from disk and write its sample index.

    This is deliberately kept in ``cup.synthetic`` and does not import the
    inversion package.  It closes the producer-side persistence boundary:
    accepted parents must have exactly one realization manifest, every listed
    trace must exist, and no rejected/orphan parent may be published.
    """
    root_path = Path(root)
    realizations = root_path / "realizations"
    if not realizations.is_dir():
        raise FileNotFoundError(f"structured truth realizations directory is missing: {realizations}")

    expected = {_safe_component(value) for value in expected_realization_ids}
    actual_dirs = {
        path.name
        for path in realizations.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    orphan_ids = sorted(actual_dirs.difference(expected))
    missing_ids = sorted(expected.difference(actual_dirs))
    if missing_ids or orphan_ids:
        raise ValueError(
            "structured truth parent set differs from accepted parent set: "
            f"missing={missing_ids}, orphan={orphan_ids}"
        )

    rows: list[dict[str, Any]] = []
    primary_keys: set[tuple[str, int, str]] = set()
    duplicate_count = 0
    for realization_dir in sorted(realizations.iterdir(), key=lambda path: path.name):
        if not realization_dir.is_dir() or realization_dir.name.startswith("."):
            continue
        realization_manifest = _read_json(
            realization_dir / "realization_manifest.json",
            label="realization manifest",
        )
        if realization_manifest.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError(f"unsupported structured realization artifact: {realization_dir}")
        if int(realization_manifest.get("artifact_version", -1)) != ARTIFACT_VERSION:
            raise ValueError(f"structured realization version mismatch: {realization_dir}")
        realization_id = str(realization_manifest.get("realization_id") or "")
        if _safe_component(realization_id) != realization_dir.name:
            raise ValueError(f"structured realization identity disagrees with its path: {realization_dir}")
        trace_artifacts = realization_manifest.get("trace_artifacts")
        if not isinstance(trace_artifacts, list) or not trace_artifacts:
            raise ValueError(f"structured realization has no trace artifacts: {realization_dir}")
        for relative in trace_artifacts:
            trace_path = realization_dir / str(relative)
            try:
                trace_path.resolve().relative_to(realization_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"structured trace path escapes its realization: {trace_path}"
                ) from exc
            trace_manifest = _read_json(trace_path / "manifest.json", label="trace manifest")
            array_path = trace_path / "arrays.npz"
            if not array_path.is_file():
                raise FileNotFoundError(f"structured trace arrays are missing: {trace_path}")
            if trace_manifest.get("artifact_type") != ARTIFACT_TYPE:
                raise ValueError(f"unsupported structured trace artifact: {trace_path}")
            if int(trace_manifest.get("artifact_version", -1)) != ARTIFACT_VERSION:
                raise ValueError(f"structured trace version mismatch: {trace_path}")
            if str(trace_manifest.get("realization_id")) != realization_id:
                raise ValueError(f"structured trace parent identity mismatch: {trace_path}")
            if list(trace_manifest.get("array_names") or []) != list(ARRAY_NAMES):
                raise ValueError(f"structured trace array contract is invalid: {trace_path}")
            _required_identity(
                trace_manifest.get("identity"),
            )
            _validate_forward_context(trace_manifest.get("forward_context"))
            with np.load(array_path, allow_pickle=False) as data:
                if set(data.files) != set(ARRAY_NAMES):
                    raise ValueError(f"structured trace arrays do not match its manifest: {trace_path}")
                observed_axis = np.asarray(data["observed_axis"], dtype=np.float64)
                for name in (
                    "observed_seismic",
                    "model_consistent_seismic",
                    "observed_lfm",
                    "observed_valid",
                ):
                    if np.asarray(data[name]).shape != observed_axis.shape:
                        raise ValueError(f"structured trace array shape mismatch: {trace_path}/{name}")
            geometry = _required_mapping(
                trace_manifest.get("geometry"),
                ("lateral_m", "inline", "xline", "xline_step"),
                label="structured trace geometry",
            )
            observed_axis = _required_mapping(
                trace_manifest.get("observed_axis"),
                (
                    "sample_domain",
                    "unit",
                    "sample_interval",
                    "positive_direction",
                    "depth_basis",
                ),
                label="structured trace observed axis",
            )
            latent_axis = _required_mapping(
                trace_manifest.get("latent_axis"),
                (
                    "sample_domain",
                    "unit",
                    "sample_interval",
                    "positive_direction",
                    "depth_basis",
                ),
                label="structured trace latent axis",
            )
            if (
                observed_axis["sample_domain"] != latent_axis["sample_domain"]
                or observed_axis["unit"] != latent_axis["unit"]
                or observed_axis["depth_basis"] != latent_axis["depth_basis"]
            ):
                raise ValueError(f"structured trace axis identity mismatch: {trace_path}")
            zone = _required_mapping(
                trace_manifest.get("zone"),
                ("zone_id", "top", "bottom", "background_a", "background_b"),
                label="structured trace zone",
            )
            segments = trace_manifest.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"structured trace has no segment truth: {trace_path}")
            for segment in segments:
                _segment_manifest(segment)
            producer_fields = _required_mapping(
                trace_manifest.get("producer_fields"),
                (
                    "raw_projected_effective_coefficients",
                    "realization_zone_background",
                    "clipping_mask_available_in_producer_record",
                ),
                label="structured trace producer fields",
            )
            if not all(bool(value) for value in producer_fields.values()):
                raise ValueError(f"structured trace producer fields are incomplete: {trace_path}")
            lateral_index = int(trace_manifest.get("lateral_index"))
            zone_id = str(zone["zone_id"])
            key = (realization_id, lateral_index, zone_id)
            if key in primary_keys:
                duplicate_count += 1
            primary_keys.add(key)
            rows.append(
                {
                    "realization_id": realization_id,
                    "scenario_id": str(trace_manifest.get("scenario_id") or ""),
                    "sample_domain": str(observed_axis["sample_domain"]),
                    "sample_unit": str(observed_axis["unit"]),
                    "depth_basis": str(observed_axis["depth_basis"] or ""),
                    "lateral_index": lateral_index,
                    "zone_id": zone_id,
                    "artifact_path": str(trace_path.relative_to(root_path)).replace("\\", "/"),
                    "lateral_m": float(geometry["lateral_m"]),
                    "inline": float(geometry["inline"]),
                    "xline": float(geometry["xline"]),
                    "xline_step": float(geometry["xline_step"]),
                }
            )
    if duplicate_count:
        raise ValueError(f"structured truth contains {duplicate_count} duplicate primary keys")
    if not rows:
        raise ValueError("structured truth contains no trace artifacts")

    rows.sort(key=lambda row: (row["realization_id"], row["lateral_index"], row["zone_id"]))
    index_path = root_path / "sample_index.csv"
    temporary_index = root_path / f".{index_path.name}.{uuid.uuid4().hex}.staging"
    with temporary_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURED_SAMPLE_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_index.replace(index_path)
    return {
        "schema": "structured_truth_v1_publication_report",
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "passed": True,
        "expected_parent_count": len(expected),
        "structured_parent_count": len(actual_dirs),
        "structured_trace_count": len(rows),
        "orphan_parent_count": len(orphan_ids),
        "missing_parent_count": len(missing_ids),
        "duplicate_primary_key_count": duplicate_count,
        "sample_index": "sample_index.csv",
    }


def _write_trace(
    record: StructuredSampleRecord,
    *,
    lateral_index: int,
    zone_id: str,
    target: Path,
) -> None:
    truth = record.truth
    model_axis = record.projected.model_axis
    zone_rows = [
        row
        for row in truth.structured_zone_truth
        if str(row.get("zone_id")) == zone_id
        and int(row.get("lateral_index", -1)) == lateral_index
    ]
    if len(zone_rows) != 1:
        raise ValueError(
            f"structured truth requires exactly one zone row for {zone_id!r}/"
            f"lateral {lateral_index}, got {len(zone_rows)}"
        )
    zone_row = zone_rows[0]
    zone_grid_value = int(zone_row["zone_grid_value"])
    zone_grid = np.asarray(truth.zone_id_highres)[lateral_index]
    zone_valid = zone_grid == zone_grid_value
    log_ai = np.asarray(truth.log_ai_highres, dtype=np.float64)[lateral_index]
    latent_valid = zone_valid & np.isfinite(log_ai)
    if not np.any(latent_valid):
        raise ValueError("structured truth zone has no finite high-resolution samples")

    segment_rows = sorted(
        (
            row
            for row in truth.structured_segment_truth
            if str(row.get("zone_id")) == zone_id
            and int(row.get("lateral_index", -1)) == lateral_index
        ),
        key=lambda row: (float(row["top"]), int(row["object_id"])),
    )
    if not segment_rows:
        raise ValueError("structured truth zone has no segment rows")
    if any(
        int(row["duration_samples"]) <= 0
        or not bool(row["segment_supervision_valid"])
        for row in segment_rows
    ):
        raise ValueError(
            "structured truth contains a segment without complete supervision"
        )
    tolerance = max(float(truth.highres_sample_interval) * 1e-6, 1e-9)
    if not np.isclose(
        float(segment_rows[0]["top"]),
        float(zone_row["top"]),
        rtol=0.0,
        atol=tolerance,
    ) or not np.isclose(
        float(segment_rows[-1]["bottom"]),
        float(zone_row["bottom"]),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("structured segment endpoints do not cover the selected zone")
    for previous, current in zip(segment_rows, segment_rows[1:], strict=False):
        if not np.isclose(
            float(previous["bottom"]),
            float(current["top"]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("structured segment endpoints are not contiguous")

    seismic = np.asarray(record.forward.seismic_observed, dtype=np.float64)[lateral_index]
    model_consistent_seismic = np.asarray(
        record.forward.seismic_model_consistent,
        dtype=np.float64,
    )[lateral_index]
    lfm = np.asarray(record.lfm.values, dtype=np.float64)[lateral_index]
    observed_valid = np.asarray(record.valid_mask, dtype=bool)[lateral_index].copy()
    if observed_valid.shape != model_axis.coordinates.shape:
        raise ValueError("structured observed mask does not match model axis")
    if np.any(
        observed_valid
        & (
            ~np.isfinite(seismic)
            | ~np.isfinite(model_consistent_seismic)
            | ~np.isfinite(lfm)
        )
    ):
        raise ValueError("structured observed valid samples must be finite")

    identity = _required_identity(record.domain_metadata.get("structured_identity"))
    xline_step = float(record.domain_metadata.get("xline_step"))
    if not np.isfinite(xline_step) or xline_step == 0.0:
        raise ValueError("structured artifact requires a finite non-zero xline_step")
    geometry = {
        "lateral_m": float(truth.lateral_m[lateral_index]),
        "inline": float(truth.inline_float[lateral_index]),
        "xline": float(truth.xline_float[lateral_index]),
        "xline_step": xline_step,
    }
    arrays = {
        "observed_axis": model_axis.coordinates,
        "observed_seismic": seismic,
        "model_consistent_seismic": model_consistent_seismic,
        "observed_lfm": lfm,
        "observed_valid": observed_valid,
        "latent_axis": np.asarray(truth.highres_axis, dtype=np.float64),
        "latent_log_ai_highres_truth": log_ai,
        "latent_valid": latent_valid,
        "latent_state_id": np.asarray(truth.state_id_highres)[lateral_index],
        "latent_object_id": np.asarray(truth.object_id_highres)[lateral_index],
        "latent_object_xi": np.asarray(truth.object_xi_highres)[lateral_index],
        "latent_zone_id": np.asarray(truth.zone_id_highres)[lateral_index],
        "latent_clipping_mask": np.asarray(truth.clipping_mask_highres)[lateral_index],
        "zone_valid": zone_valid,
    }
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": ARTIFACT_VERSION,
        "realization_id": str(truth.realization_id),
        "scenario_id": str(truth.scenario.scenario_id),
        "lateral_index": int(lateral_index),
        "geometry": geometry,
        "identity": identity,
        "forward_context": _forward_context_manifest(record),
        "lfm": {
            "source_identity": dict(record.lfm.source_identity),
        },
        "observed_axis": _axis_manifest(model_axis),
        "latent_axis": _axis_manifest(
            SampleAxis(
                sample_domain=truth.sample_domain,
                unit=truth.axis_unit,
                coordinates=truth.highres_axis,
                sample_interval=float(truth.highres_sample_interval),
                positive_direction=(
                    "down" if truth.sample_domain == "depth" else "increasing_time"
                ),
                depth_basis="tvdss" if truth.sample_domain == "depth" else None,
            )
        ),
        "zone": {
            "zone_id": zone_id,
            "zone_grid_value": zone_grid_value,
            "top": float(zone_row["top"]),
            "bottom": float(zone_row["bottom"]),
            "background_a": float(zone_row["background_a"]),
            "background_b": float(zone_row["background_b"]),
        },
        "segments": [_segment_manifest(row) for row in segment_rows],
        "array_file": "arrays.npz",
        "array_names": list(ARRAY_NAMES),
        "producer_fields": {
            "raw_projected_effective_coefficients": True,
            "realization_zone_background": True,
            "clipping_mask_available_in_producer_record": True,
        },
    }
    json.dumps(manifest, allow_nan=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    temporary_path.mkdir(parents=True, exist_ok=False)
    try:
        np.savez_compressed(temporary_path / "arrays.npz", **arrays)
        (temporary_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(target)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)


def write_structured_truth_v1(
    record: StructuredSampleRecord,
    root: str | Path,
) -> tuple[Path, ...]:
    """Publish one realization as strict per-zone/per-trace artifacts."""
    if not isinstance(record, StructuredSampleRecord):
        raise TypeError("write_structured_truth_v1 requires StructuredSampleRecord")
    root_path = Path(root)
    realizations = root_path / "realizations"
    realizations.mkdir(parents=True, exist_ok=True)
    realization_target = realizations / _safe_component(record.truth.realization_id)
    if realization_target.exists():
        raise FileExistsError(f"structured realization already exists: {realization_target}")
    zone_ids = sorted(
        {str(row["zone_id"]) for row in record.truth.structured_zone_truth}
    )
    if not zone_ids:
        raise ValueError("SyntheticTruth has no explicit structured zone truth")
    for zone_id in zone_ids:
        rows = [
            row
            for row in record.truth.structured_zone_truth
            if str(row["zone_id"]) == zone_id
        ]
        if not rows:
            raise ValueError(f"structured zone {zone_id!r} has no producer rows")
        for name in ("background_a", "background_b"):
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            if not np.all(np.isfinite(values)) or not np.allclose(
                values,
                values[0],
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"structured {zone_id!r} {name} is not realization-zone constant"
                )
        segment_keys_by_lateral = {
            lateral_index: tuple(
                sorted(
                    int(row["object_id"])
                    for row in record.truth.structured_segment_truth
                    if str(row.get("zone_id")) == zone_id
                    and int(row.get("lateral_index", -1)) == lateral_index
                )
            )
            for lateral_index in range(record.truth.lateral_m.size)
        }
        if len(set(segment_keys_by_lateral.values())) != 1:
            raise ValueError(
                f"structured {zone_id!r} segment/object keys differ across lateral traces"
            )
    written: list[Path] = []
    staging = realizations / f".{realization_target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for lateral_index in range(record.truth.lateral_m.size):
            for zone_id in zone_ids:
                target = (
                    staging
                    / "traces"
                    / f"lateral_{lateral_index:04d}"
                    / f"zone_{_safe_component(zone_id)}"
                )
                _write_trace(
                    record,
                    lateral_index=lateral_index,
                    zone_id=zone_id,
                    target=target,
                )
                written.append(
                    target.relative_to(staging)
                )
        realization_manifest = {
            "artifact_type": ARTIFACT_TYPE,
            "artifact_version": ARTIFACT_VERSION,
            "realization_id": str(record.truth.realization_id),
            "scenario_id": str(record.truth.scenario.scenario_id),
            "sample_domain": record.truth.sample_domain,
            "sample_unit": record.truth.axis_unit,
            "depth_basis": (
                "tvdss" if record.truth.sample_domain == "depth" else None
            ),
            "trace_artifacts": [
                str(path).replace("\\", "/") for path in written
            ],
        }
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "realization_manifest.json").write_text(
            json.dumps(realization_manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staging.replace(realization_target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return tuple(realization_target / path for path in written)


__all__ = [
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "ARRAY_NAMES",
    "STRUCTURED_SAMPLE_INDEX_COLUMNS",
    "remove_structured_truth_v1",
    "validate_structured_truth_v1",
    "write_structured_truth_v1",
]
