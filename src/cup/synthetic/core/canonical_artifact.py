"""Publication helpers for the canonical structured synthetic artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


REALIZATION_INDEX_COLUMNS = (
    "realization_id",
    "sample_domain",
    "sample_unit",
    "depth_basis",
    "section_id",
    "scenario_id",
    "geometry_family",
    "duration_mode",
    "suite",
    "evaluation_role",
    "attempt_id",
    "hdf5_group",
    "seismic_dataset",
    "lfm_dataset",
    "model_consistent_seismic_dataset",
    "model_log_ai_dataset",
    "valid_mask_dataset",
    "n_valid",
)


def publish_realization_index(
    output_dir: str | Path,
    rows: Iterable[Mapping[str, object]],
) -> Path:
    normalized = [dict(row) for row in rows]
    frame = pd.DataFrame.from_records(normalized)
    for column in REALIZATION_INDEX_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame = frame.loc[:, list(REALIZATION_INDEX_COLUMNS)]
    if not frame.empty:
        frame = frame.sort_values(
            ["sample_domain", "realization_id"], kind="mergesort"
        ).reset_index(drop=True)
    if frame["realization_id"].duplicated().any():
        raise ValueError("realization index contains duplicate realization_id")
    path = Path(output_dir) / "realization_index.csv"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = ["REALIZATION_INDEX_COLUMNS", "publish_realization_index"]
