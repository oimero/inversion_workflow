"""Atomic tabular publication helpers for Synthoseis-lite v5."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterable, Mapping

import pandas as pd


REALIZATION_INDEX_COLUMNS = (
    "realization_id", "sample_domain", "sample_unit", "depth_basis",
    "section_id", "scenario_id", "geometry_family", "duration_mode",
    "suite", "evaluation_role", "parent_realization_id", "hdf5_group",
    "base_seismic_dataset", "model_consistent_seismic_dataset",
    "target_log_ai_dataset", "canonical_background_dataset",
    "target_increment_dataset", "valid_mask_dataset", "n_valid",
)
VIEW_INDEX_COLUMNS = (
    "realization_id", "parent_realization_id", "view_id", "sample_domain", "sample_unit",
    "evaluation_role", "hdf5_group", "seismic_observed_dataset",
    "seismic_input_dataset", "model_consistent_seismic_dataset",
    "seismic_model_consistent_dataset", "valid_mask_dataset",
    "operator_ids_json", "operator_kinds_json", "operator_parameters_json",
    "operator_contract_versions_json", "view_spec_canonical_json",
    "view_spec_sha256", "random_stream_identity_json",
    "operator_trace_dataset", "n_valid",
)


def _frame(rows: Iterable[Mapping[str, Any]], columns: tuple[str, ...], sort_by: tuple[str, ...]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if not str(row.get("sample_domain") or "").strip():
            raise ValueError("v5 index rows require sample_domain")
        if not str(row.get("sample_unit") or "").strip():
            raise ValueError("v5 index rows require sample_unit")
        row.setdefault("realization_id", row.get("parent_realization_id", ""))
        row.setdefault("model_consistent_seismic_dataset", row.get("seismic_model_consistent_dataset", ""))
        row.setdefault("seismic_model_consistent_dataset", row.get("model_consistent_seismic_dataset", ""))
        row.setdefault("target_log_ai_dataset", row.get("target_dataset", ""))
        row.setdefault("seismic_observed_dataset", row.get("seismic_input_dataset", ""))
        row.setdefault("seismic_input_dataset", row.get("seismic_observed_dataset", ""))
        row.setdefault("n_valid", row.get("valid_sample_count", ""))
        row.setdefault("operator_parameters_json", row.get("operator_parameters", ""))
        if isinstance(row.get("operator_parameters_json"), Mapping):
            row["operator_parameters_json"] = json.dumps(row["operator_parameters_json"], sort_keys=True)
        row.setdefault("operator_contract_versions_json", row.get("operator_contract_versions", ""))
        if isinstance(row.get("operator_contract_versions_json"), Mapping):
            row["operator_contract_versions_json"] = json.dumps(row["operator_contract_versions_json"], sort_keys=True)
        row.setdefault("random_stream_identity_json", row.get("random_stream_identity", ""))
        if isinstance(row.get("random_stream_identity_json"), Mapping):
            row["random_stream_identity_json"] = json.dumps(row["random_stream_identity_json"], sort_keys=True)
        normalized.append(row)
    frame = pd.DataFrame.from_records(normalized)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    frame = frame.loc[:, list(columns)]
    if not frame.empty:
        frame = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    return frame


def publish_v5_indexes(
    output_dir: str | Path,
    realization_rows: Iterable[Mapping[str, Any]],
    view_rows: Iterable[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Write only successful parent/view rows using the fixed v5 columns."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    realization_path = directory / "realization_index.csv"
    view_path = directory / "seismic_view_index.csv"
    def successful(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [
            row
            for row in rows
            if str(row.get("status", "ok") or "ok").casefold()
            in {"ok", "accepted", "success"}
        ]

    parent_rows = successful(realization_rows)
    parent_frame = _frame(
        parent_rows,
        REALIZATION_INDEX_COLUMNS,
        ("sample_domain", "realization_id"),
    )
    if parent_frame["realization_id"].duplicated().any():
        raise ValueError("realization_index.csv contains duplicate realization_id")
    parent_ids = set(parent_frame["realization_id"].astype(str))
    successful_views = successful(view_rows)
    unknown_parents = sorted(
        {
            str(row.get("parent_realization_id") or row.get("realization_id") or "")
            for row in successful_views
        }
        - parent_ids
    )
    if unknown_parents:
        raise ValueError(
            "seismic_view_index.csv references unknown parent realizations: "
            f"{unknown_parents[:5]}"
        )
    materialized_views = successful_views
    view_frame = _frame(
        materialized_views,
        VIEW_INDEX_COLUMNS,
        ("parent_realization_id", "view_id"),
    )
    if view_frame.duplicated(subset=["parent_realization_id", "view_id"]).any():
        raise ValueError(
            "seismic_view_index.csv contains duplicate parent_realization_id/view_id"
        )
    realization_tmp = directory / f".{realization_path.name}.tmp"
    view_tmp = directory / f".{view_path.name}.tmp"
    try:
        parent_frame.to_csv(realization_tmp, index=False)
        view_frame.to_csv(view_tmp, index=False)
        realization_tmp.replace(realization_path)
        view_tmp.replace(view_path)
    finally:
        realization_tmp.unlink(missing_ok=True)
        view_tmp.unlink(missing_ok=True)
    return realization_path, view_path


__all__ = [
    "REALIZATION_INDEX_COLUMNS",
    "VIEW_INDEX_COLUMNS",
    "publish_v5_indexes",
]
