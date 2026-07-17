"""Domain-neutral helpers shared by Synthoseis-lite time/depth pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd



def evaluation_role(geometry_family: str, held_out_geometry_family: str) -> str:
    return (
        "geometry_holdout"
        if str(geometry_family) == str(held_out_geometry_family)
        else "development_pool"
    )


def build_attempt_plan(
    *,
    section_ids: Sequence[str],
    scenarios: Sequence[Any],
    attempts_per_scenario: int,
    held_out_geometry_family: str,
    geometry_families: Iterable[str] | None = None,
) -> pd.DataFrame:
    selected = (
        None
        if geometry_families is None
        else {str(value) for value in geometry_families}
    )
    rows: list[dict[str, Any]] = []
    for section_id in section_ids:
        for scenario in scenarios:
            family = str(getattr(scenario, "geometry_family"))
            if selected is not None and family not in selected:
                continue
            for attempt_id in range(int(attempts_per_scenario)):
                parent = f"{section_id}__{scenario.scenario_id}__a{attempt_id:03d}"
                rows.append(
                    {
                        "section_id": str(section_id),
                        "scenario_id": str(scenario.scenario_id),
                        "duration_mode": str(getattr(scenario, "duration_mode", "")),
                        "geometry_family": family,
                        "geometry_direction": str(
                            getattr(scenario, "geometry_direction", "")
                        ),
                        "attempt_id": int(attempt_id),
                        "parent_realization_id": parent,
                        "evaluation_role": evaluation_role(
                            family, held_out_geometry_family
                        ),
                    }
                )
    return pd.DataFrame.from_records(rows)


def validate_debug_attempt_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or int(limit) <= 0 or float(limit) != float(int(limit)):
        raise ValueError("debug_attempt_limit must be a positive integer.")
    return int(limit)


def limit_attempt_plan(plan: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Apply the development attempt cap identically in both domains."""
    parsed_limit = validate_debug_attempt_limit(limit)
    if parsed_limit is None:
        return plan.reset_index(drop=True)
    return (
        plan.groupby(["section_id", "scenario_id"], sort=False)
        .head(parsed_limit)
        .reset_index(drop=True)
    )


def write_dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    *,
    unit: str,
    sample_domain: str,
    axis_path: str,
    axis_order: str | Sequence[str],
) -> h5py.Dataset:
    data = np.asarray(values)
    if data.ndim == 0:
        dataset = group.create_dataset(name, data=data)
    else:
        dataset = group.create_dataset(name, data=data, compression="gzip", shuffle=True)
    dataset.attrs["unit"] = unit
    dataset.attrs["sample_domain"] = str(sample_domain)
    dataset.attrs["axis_path"] = str(axis_path)
    if isinstance(axis_order, str):
        dataset.attrs["axis_order"] = axis_order
    else:
        dataset.attrs["axis_order"] = ",".join(str(value) for value in axis_order)
    dataset.attrs["shape_json"] = json.dumps(list(data.shape))
    dataset.attrs["dtype"] = str(data.dtype)
    return dataset


def validate_dataset_metadata(value: h5py.Dataset, *, sample_domain: str) -> None:
    required = {
        "unit",
        "sample_domain",
        "axis_path",
        "axis_order",
        "shape_json",
        "dtype",
    }
    missing = required - set(value.attrs)
    if missing:
        raise ValueError(
            f"HDF5 dataset {value.name} lacks metadata attrs: {sorted(missing)}"
        )
    if str(value.attrs["sample_domain"]) != str(sample_domain):
        raise ValueError(f"HDF5 dataset {value.name} has stale sample_domain metadata.")
    if json.loads(str(value.attrs["shape_json"])) != list(value.shape):
        raise ValueError(f"HDF5 dataset {value.name} shape metadata is stale.")
    if str(value.attrs["dtype"]) != str(value.dtype):
        raise ValueError(f"HDF5 dataset {value.name} dtype metadata is stale.")
    axis_path = str(value.attrs["axis_path"])
    if value.ndim == 0 and str(value.attrs.get("unit")) == "json":
        if axis_path:
            raise ValueError(f"JSON metadata dataset {value.name} must not reference an axis")
        return
    if not axis_path:
        if Path(value.parent.name).name.casefold() != "axes":
            raise ValueError(
                f"HDF5 dataset {value.name} has no axis reference outside an axes group."
            )
        return
    if axis_path not in value.file or not isinstance(value.file[axis_path], h5py.Dataset):
        raise ValueError(
            f"HDF5 dataset {value.name} references a missing axis: {axis_path}"
        )
    axis_order = [
        item.strip() for item in str(value.attrs["axis_order"]).split(",")
    ]
    axis_name = Path(axis_path).name.casefold()
    if axis_name == "lateral_m":
        candidates = [index for index, label in enumerate(axis_order) if label == "lateral"]
    elif "tvdss" in axis_name:
        candidates = [
            index for index, label in enumerate(axis_order) if label.startswith("tvdss")
        ]
    elif "twt" in axis_name or "time" in axis_name:
        candidates = [
            index
            for index, label in enumerate(axis_order)
            if label.startswith("twt") or label.startswith("time")
        ]
    else:
        raise ValueError(
            f"HDF5 dataset {value.name} references an unknown axis: {axis_path}"
        )
    if (
        len(candidates) != 1
        or value.shape[candidates[0]] != value.file[axis_path].shape[0]
    ):
        raise ValueError(
            f"HDF5 dataset {value.name} shape is inconsistent with axis {axis_path}."
        )


def validate_training_manifest(
    manifest: Mapping[str, Any], *, sample_domain: str
) -> None:
    """Apply the shared status and qc-only consumption contract."""
    status = str(manifest.get("status") or "")
    if status not in {
        "ok",
        "success",
        "completed_with_warnings",
        "development_limited",
    }:
        raise ValueError(
            f"Synthoseis v5 {sample_domain} manifest is not consumable: status={status!r}."
        )
    if bool(manifest.get("qc_only", False)) or manifest.get(
        "training_consumable"
    ) is False:
        raise ValueError(
            "Synthoseis v5 qc-only benchmark is not training-consumable; "
            "regenerate without --qc-only."
        )


def geometry_feasibility_rows(
    *,
    sections: Sequence[Any],
    ordered_horizons: Sequence[str],
    vertical_axis_name: str,
    minimum_highres_cells: int,
    highres_step: float,
    duration_reference: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section in sections:
        if hasattr(section, "horizon_twt_s"):
            raw_horizons = getattr(section, "horizon_twt_s")
        else:
            raw_horizons = getattr(section, "horizon_tvdss_m")
        horizons = np.asarray(raw_horizons, dtype=np.float64)
        for zone_index, (top, bottom) in enumerate(
            zip(ordered_horizons[:-1], ordered_horizons[1:])
        ):
            thickness = horizons[:, zone_index + 1] - horizons[:, zone_index]
            min_value = float(np.min(thickness))
            rows.append(
                {
                    "section_id": str(section.section_id),
                    "zone_id": f"{top}__to__{bottom}",
                    "top_horizon": str(top),
                    "bottom_horizon": str(bottom),
                    "vertical_axis": vertical_axis_name,
                    "thickness_min": min_value,
                    "thickness_p05": float(np.quantile(thickness, 0.05)),
                    "thickness_median": float(np.median(thickness)),
                    "thickness_p95": float(np.quantile(thickness, 0.95)),
                    "thickness_max": float(np.max(thickness)),
                    "highres_step": float(highres_step),
                    "minimum_highres_cells": int(minimum_highres_cells),
                    "minimum_required_thickness": float(minimum_highres_cells)
                    * float(highres_step),
                    "minimum_available_highres_cells": int(
                        np.floor(min_value / float(highres_step))
                    ),
                    "sequence_minimum_duration_reference": str(duration_reference),
                    "feasible": bool(
                        min_value >= int(minimum_highres_cells) * float(highres_step)
                    ),
                }
            )
    return rows


def _primary_reason(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return ""
    return text.split(";", maxsplit=1)[0].split(":", maxsplit=1)[0]


def rejection_reason_summary(
    details: pd.DataFrame, index: pd.DataFrame | None = None
) -> pd.DataFrame:
    reasons: list[str] = []
    if not details.empty and "reason" in details:
        reasons.extend(_primary_reason(value) for value in details["reason"].tolist())
    if index is not None and not index.empty and "reasons" in index:
        rejected = index[
            index.get("status", pd.Series(dtype=str)).astype(str).eq("rejected")
        ]
        reasons.extend(_primary_reason(value) for value in rejected["reasons"].tolist())
    cleaned = [reason for reason in reasons if reason]
    if not cleaned:
        return pd.DataFrame(columns=["reason", "count", "fraction"])
    counts = (
        pd.Series(cleaned, dtype=str)
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
    )
    counts["fraction"] = counts["count"] / float(counts["count"].sum())
    return counts


__all__ = [
    "build_attempt_plan",
    "evaluation_role",
    "geometry_feasibility_rows",
    "limit_attempt_plan",
    "rejection_reason_summary",
    "validate_dataset_metadata",
    "validate_debug_attempt_limit",
    "validate_training_manifest",
    "write_dataset",
]
