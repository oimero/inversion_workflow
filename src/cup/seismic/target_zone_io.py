"""Load the workflow target interval into one shared :class:`TargetZone`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.petrel.load import import_interpretation_petrel
from cup.seismic.horizon import normalize_interpretation_unit_for_geometry
from cup.seismic.target_zone import TargetZone
from cup.utils.io import repo_relative_path, resolve_relative_path


def build_workflow_target_zone(
    *,
    raw_config: Mapping[str, Any],
    survey: Any,
    data_root: Path,
    repo_root: Path,
) -> tuple[TargetZone, list[dict[str, str]]]:
    """Load the ordered ``target_interval.horizons`` workflow contract."""

    sample_axis = survey.sample_axis(str(raw_config["seismic"]["domain"]))
    geometry = survey.describe_geometry(sample_axis.domain)
    target = raw_config.get("target_interval")
    if (
        not isinstance(target, Mapping)
        or not isinstance(target.get("horizons"), list)
        or len(target["horizons"]) < 2
    ):
        raise ValueError("target_interval.horizons must contain at least two ordered horizons.")

    raw_horizons: dict[str, pd.DataFrame] = {}
    sources: list[dict[str, str]] = []
    names: list[str] = []
    for index, item in enumerate(target["horizons"]):
        if not isinstance(item, Mapping):
            raise ValueError(f"target_interval.horizons[{index}] must be a mapping.")
        name = str(item.get("name") or "").strip()
        file_text = str(item.get("file") or "").strip()
        if not name or not file_text or name in raw_horizons:
            raise ValueError(f"Invalid/duplicate target horizon at index {index}.")
        path = resolve_relative_path(file_text, root=data_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = normalize_interpretation_unit_for_geometry(
            import_interpretation_petrel(path),
            geometry,
        )
        values = pd.to_numeric(frame["interpretation"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        frame = frame.copy()
        frame["interpretation"] = np.abs(values)
        raw_horizons[name] = frame
        names.append(name)
        sources.append({"name": name, "path": repo_relative_path(path, root=repo_root)})

    return (
        TargetZone(
            raw_horizons,
            geometry,
            names,
            min_thickness=float(sample_axis.step),
        ),
        sources,
    )


__all__ = ["build_workflow_target_zone"]
