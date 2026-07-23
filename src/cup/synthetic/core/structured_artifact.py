"""Producer-owned structured truth artifacts for the new inversion seam."""

from __future__ import annotations

from collections.abc import Mapping
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
    lfm = np.asarray(record.lfm.values, dtype=np.float64)[lateral_index]
    observed_valid = np.asarray(record.valid_mask, dtype=bool)[lateral_index].copy()
    if observed_valid.shape != model_axis.coordinates.shape:
        raise ValueError("structured observed mask does not match model axis")
    if np.any(observed_valid & (~np.isfinite(seismic) | ~np.isfinite(lfm))):
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
    "write_structured_truth_v1",
]
