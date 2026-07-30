"""Read-only interface for the canonical structured synthetic benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from cup.synthetic.core.records import SampleAxis
from cup.synthetic.schemas import (
    STRUCTURED_ARTIFACT_TYPE,
    STRUCTURED_ARTIFACT_VERSION,
)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_attr(group: h5py.Group, name: str) -> Any:
    if name not in group.attrs:
        raise ValueError(f"missing HDF5 attribute {group.name}:{name}")
    return json.loads(_text(group.attrs[name]))


def _table(group: h5py.Group) -> tuple[dict[str, Any], ...]:
    columns = sorted(group.keys())
    if not columns:
        raise ValueError(f"empty structured table: {group.name}")
    arrays = {name: np.asarray(group[name]) for name in columns}
    row_count = int(group.attrs.get("row_count", -1))
    if row_count < 0 or any(values.shape != (row_count,) for values in arrays.values()):
        raise ValueError(f"invalid structured table shape: {group.name}")
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row: dict[str, Any] = {}
        for name, values in arrays.items():
            value = values[index]
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8")
            elif isinstance(value, np.generic):
                value = value.item()
            row[name] = value
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class ParentIdentity:
    realization_id: str
    scenario_id: str
    section_id: str
    corpus_role: str
    split_role: str
    generalization_role: str


@dataclass(frozen=True)
class StructuredParent:
    identity: ParentIdentity
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    highres_axis: SampleAxis
    model_axis: SampleAxis
    lateral_m: np.ndarray
    inline: np.ndarray
    xline: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    xline_step: float
    seismic: np.ndarray
    model_consistent_seismic: np.ndarray
    lfm: np.ndarray
    observed_valid: np.ndarray
    log_ai_highres: np.ndarray
    truth_valid_highres: np.ndarray
    state_id_highres: np.ndarray
    object_id_highres: np.ndarray
    object_xi_highres: np.ndarray
    zone_id_highres: np.ndarray
    clipping_mask_highres: np.ndarray
    model_log_ai: np.ndarray
    zones: tuple[Mapping[str, Any], ...]
    segments: tuple[Mapping[str, Any], ...]
    structured_identity: Mapping[str, Any]
    lfm_source_identity: Mapping[str, Any]
    forward_context: Mapping[str, Any]


class StructuredSyntheticBenchmark:
    """Deep reader module for one canonical HDF5 plus its parent index."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.h5_path = self.run_dir / "synthetic_benchmark.h5"
        self.index_path = self.run_dir / "realization_index.csv"
        if not self.h5_path.is_file():
            raise FileNotFoundError(self.h5_path)
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        self.index = pd.read_csv(self.index_path, dtype=str, keep_default_na=False)
        required = {
            "realization_id",
            "section_id",
            "scenario_id",
            "corpus_role",
            "split_role",
            "generalization_role",
            "hdf5_group",
        }
        missing = sorted(required.difference(self.index.columns))
        if missing:
            raise ValueError(f"realization index is missing columns: {missing}")
        if self.index["realization_id"].duplicated().any():
            raise ValueError("realization index contains duplicate realization_id")
        with h5py.File(self.h5_path, "r") as h5:
            if _text(h5.attrs.get("artifact_type", "")) != STRUCTURED_ARTIFACT_TYPE:
                raise ValueError("HDF5 artifact_type is not structured_synthetic_benchmark")
            if int(h5.attrs.get("artifact_version", -1)) != STRUCTURED_ARTIFACT_VERSION:
                raise ValueError("unsupported structured synthetic artifact version")
            self.sample_domain = _text(h5.attrs.get("sample_domain", "")).casefold()
            self.sample_unit = _text(h5.attrs.get("sample_unit", ""))
            depth = h5.attrs.get("depth_basis")
            self.depth_basis = None if depth is None else _text(depth)
            parent_ids = set(h5.get("/realizations", {}).keys())
            indexed = set(self.index["realization_id"])
            if parent_ids != indexed:
                raise ValueError("HDF5 parent set differs from realization_index.csv")
            incomplete = [
                parent_id
                for parent_id in sorted(parent_ids)
                if not bool(h5[f"/realizations/{parent_id}"].attrs.get("complete", False))
            ]
            if incomplete:
                raise ValueError(f"HDF5 contains incomplete parents: {incomplete[:5]}")
            if "/__staging__" in h5 and len(h5["/__staging__"]) != 0:
                raise ValueError("HDF5 contains uncommitted staging parents")

    def list_parents(self, split: str | None = None) -> list[ParentIdentity]:
        rows = self.index
        if split is not None:
            rows = rows.loc[rows["split_role"] == str(split)]
        return [
            ParentIdentity(
                realization_id=str(row.realization_id),
                scenario_id=str(row.scenario_id),
                section_id=str(row.section_id),
                corpus_role=str(row.corpus_role),
                split_role=str(row.split_role),
                generalization_role=str(row.generalization_role),
            )
            for row in rows.itertuples(index=False)
        ]

    def read_parent(self, parent_id: str) -> StructuredParent:
        selected = self.index.loc[self.index["realization_id"] == str(parent_id)]
        if len(selected) != 1:
            raise KeyError(f"unknown realization_id: {parent_id}")
        row = selected.iloc[0]
        with h5py.File(self.h5_path, "r") as h5:
            root = h5[str(row["hdf5_group"])]
            identity = root["identity"]
            axes = root["axes"]
            observed = root["observed"]
            truth = root["truth"]
            forward = root["forward"]
            domain = _text(root.attrs["sample_domain"])
            unit = _text(root.attrs["sample_unit"])
            depth = root.attrs.get("depth_basis")
            depth_basis = None if depth is None else _text(depth)
            high_name = "tvdss_highres_m" if domain == "depth" else "twt_highres_s"
            model_name = "tvdss_model_m" if domain == "depth" else "twt_model_s"
            high_values = np.asarray(axes[high_name], dtype=np.float64)
            model_values = np.asarray(axes[model_name], dtype=np.float64)
            positive = "down" if domain == "depth" else "increasing_time"
            context_group = forward["context"]
            context = {
                "wavelet_time_s": np.asarray(
                    context_group["wavelet_time_s"], dtype=np.float64
                ).tolist(),
                "wavelet_amplitude": np.asarray(
                    context_group["wavelet_amplitude"], dtype=np.float64
                ).tolist(),
                "ai_velocity_relation": _json_attr(
                    context_group, "ai_velocity_relation_json"
                ),
                "output_chunk_size": int(context_group.attrs["output_chunk_size"]),
            }
            return StructuredParent(
                identity=ParentIdentity(
                    realization_id=str(row["realization_id"]),
                    scenario_id=str(row["scenario_id"]),
                    section_id=str(row["section_id"]),
                    corpus_role=str(row["corpus_role"]),
                    split_role=str(row["split_role"]),
                    generalization_role=str(row["generalization_role"]),
                ),
                sample_domain=domain,
                sample_unit=unit,
                depth_basis=depth_basis,
                highres_axis=SampleAxis(
                    sample_domain=domain,
                    unit=unit,
                    coordinates=high_values,
                    sample_interval=float(np.diff(high_values)[0]),
                    positive_direction=positive,
                    depth_basis=depth_basis,
                ),
                model_axis=SampleAxis(
                    sample_domain=domain,
                    unit=unit,
                    coordinates=model_values,
                    sample_interval=float(np.diff(model_values)[0]),
                    positive_direction=positive,
                    depth_basis=depth_basis,
                ),
                lateral_m=np.asarray(axes["lateral_m"], dtype=np.float64),
                inline=np.asarray(axes["inline_float"], dtype=np.float64),
                xline=np.asarray(axes["xline_float"], dtype=np.float64),
                x_m=np.asarray(axes["x_m"], dtype=np.float64),
                y_m=np.asarray(axes["y_m"], dtype=np.float64),
                xline_step=float(identity.attrs["xline_step"]),
                seismic=np.asarray(observed["seismic"], dtype=np.float64),
                model_consistent_seismic=np.asarray(
                    forward["model_consistent_seismic"], dtype=np.float64
                ),
                lfm=np.asarray(observed["lfm"], dtype=np.float64),
                observed_valid=np.asarray(observed["valid"], dtype=bool),
                log_ai_highres=np.asarray(truth["log_ai_highres"], dtype=np.float64),
                truth_valid_highres=np.asarray(
                    truth["truth_valid_highres"], dtype=bool
                ),
                state_id_highres=np.asarray(truth["state_id_highres"]),
                object_id_highres=np.asarray(truth["object_id_highres"]),
                object_xi_highres=np.asarray(truth["object_xi_highres"]),
                zone_id_highres=np.asarray(truth["zone_id_highres"]),
                clipping_mask_highres=np.asarray(
                    truth["clipping_mask_highres"], dtype=bool
                ),
                model_log_ai=np.asarray(truth["model_log_ai"], dtype=np.float64),
                zones=_table(truth["zones"]),
                segments=_table(truth["segments"]),
                structured_identity=_json_attr(
                    identity, "structured_identity_json"
                ),
                lfm_source_identity=_json_attr(
                    identity, "lfm_source_identity_json"
                ),
                forward_context=context,
            )


__all__ = [
    "ParentIdentity",
    "StructuredParent",
    "StructuredSyntheticBenchmark",
]
