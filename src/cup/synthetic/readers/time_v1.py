"""Reader for frozen Synthoseis-lite v1 time-domain artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION = "synthoseis_lite_v1"
SEISMIC_VARIANT_KINDS = {"seismic_variant", "frequency_probe_seismic_variant"}


@dataclass(frozen=True)
class TimeSyntheticSample:
    sample_id: str
    sample_kind: str
    row: dict[str, Any]
    sample_domain: str
    target_log_ai: np.ndarray
    seismic_input: np.ndarray
    valid_mask: np.ndarray
    lateral_m: np.ndarray
    twt_model_s: np.ndarray
    priors: dict[str, np.ndarray]


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _clean_path(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _root_from_group(group_path: str) -> str:
    if "/probes/" in group_path:
        return group_path.split("/probes/", maxsplit=1)[0]
    if "/seismic_variants/" in group_path:
        return group_path.split("/seismic_variants/", maxsplit=1)[0]
    return group_path


class TimeV1Benchmark:
    """Read-only accessor around ``synthoseis_lite_v1`` time artifacts."""

    schema = SCHEMA_VERSION
    sample_domain = "time"

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.h5_path = self.run_dir / "synthetic_benchmark.h5"
        self.index_path = self.run_dir / "sample_index.csv"
        self.manifest_path = self.run_dir / "benchmark_manifest.json"
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"synthetic_benchmark.h5 not found: {self.h5_path}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"sample_index.csv not found: {self.index_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"benchmark_manifest.json not found: {self.manifest_path}")

        self.manifest = _json(self.manifest_path)
        actual_schema = str(self.manifest.get("schema") or self.manifest.get("schema_version") or "")
        if actual_schema != SCHEMA_VERSION:
            raise ValueError(f"Unsupported time Synthoseis schema {actual_schema!r}; expected {SCHEMA_VERSION!r}.")
        sample_domain = str(self.manifest.get("sample_domain") or "time").casefold()
        if sample_domain not in {"", "time"}:
            raise ValueError(f"Time v1 reader requires sample_domain=time or an omitted domain; got {sample_domain!r}.")

        self.index = pd.read_csv(self.index_path)
        if self.index.empty:
            raise ValueError(f"empty sample_index.csv: {self.index_path}")
        if "sample_id" not in self.index:
            raise ValueError("sample_index.csv lacks sample_id.")
        duplicates = self.index["sample_id"].duplicated()
        if bool(duplicates.any()):
            names = self.index.loc[duplicates, "sample_id"].head(5).tolist()
            raise ValueError(f"duplicate sample_id values: {names}")
        self._rows = {
            str(row["sample_id"]): row.to_dict()
            for _, row in self.index.iterrows()
        }

    def sample_ids(
        self,
        *,
        kinds: set[str] | None = None,
        status: str = "ok",
        split: str | None = None,
    ) -> list[str]:
        frame = self.index
        if status:
            frame = frame[frame["status"].astype(str).eq(status)]
        if kinds:
            frame = frame[frame["sample_kind"].astype(str).isin(kinds)]
        if split is not None:
            frame = frame[frame["split"].astype(str).eq(split)]
        return [str(value) for value in frame["sample_id"].tolist()]

    def row(self, sample_id: str) -> dict[str, Any]:
        try:
            return dict(self._rows[str(sample_id)])
        except KeyError as exc:
            raise KeyError(f"Unknown sample_id: {sample_id}") from exc

    def load_sample(self, sample_id: str) -> TimeSyntheticSample:
        row = self.row(sample_id)
        sample_kind = str(row.get("sample_kind", "base"))
        group_path = _clean_path(row.get("hdf5_group"))
        if not group_path:
            raise ValueError(f"sample has empty hdf5_group: {sample_id}")
        source_row = row
        source_kind = sample_kind
        if sample_kind in SEISMIC_VARIANT_KINDS:
            source_id = _clean_path(row.get("source_sample_id"))
            if not source_id:
                raise ValueError(f"{sample_kind} lacks source_sample_id: {sample_id}")
            source_row = self.row(source_id)
            source_kind = str(source_row.get("sample_kind", "base"))
        source_group = _clean_path(source_row.get("hdf5_group"))
        if not source_group:
            raise ValueError(f"source sample has empty hdf5_group: {sample_id}")

        root_path = _root_from_group(source_group)
        with h5py.File(self.h5_path, "r") as h5:
            target = self._read_target(h5, source_group, source_kind)
            seismic = self._read_seismic(h5, group_path, sample_kind)
            valid = np.asarray(h5[f"{root_path}/truth/valid_mask_model"][()], dtype=bool)
            lateral = np.asarray(h5[f"{root_path}/axes/lateral_m"][()], dtype=np.float64)
            twt = np.asarray(h5[f"{root_path}/axes/twt_model_s"][()], dtype=np.float64)
            priors = {
                "lfm_ideal": self._read_optional_dataset(
                    h5,
                    source_row,
                    "lfm_ideal_dataset",
                    f"{source_group}/priors/lfm_ideal",
                ),
                "lfm_controlled_degraded": self._read_optional_dataset(
                    h5,
                    source_row,
                    "lfm_controlled_degraded_dataset",
                    f"{source_group}/priors/lfm_controlled_degraded",
                ),
            }
        if target.shape != valid.shape:
            raise ValueError(f"target/valid shape mismatch for {sample_id}: {target.shape} vs {valid.shape}")
        return TimeSyntheticSample(
            sample_id=str(sample_id),
            sample_kind=sample_kind,
            row=row,
            sample_domain="time",
            target_log_ai=np.asarray(target, dtype=np.float64),
            seismic_input=np.asarray(seismic, dtype=np.float64),
            valid_mask=valid,
            lateral_m=lateral,
            twt_model_s=twt,
            priors={key: np.asarray(value, dtype=np.float64) for key, value in priors.items()},
        )

    @staticmethod
    def _read_target(h5: h5py.File, group_path: str, sample_kind: str) -> np.ndarray:
        if sample_kind == "frequency_probe":
            root_path = _root_from_group(group_path)
            base = np.asarray(h5[f"{root_path}/truth/model_target_log_ai"][()])
            increment = np.asarray(h5[f"{group_path}/truth/probe_log_ai_model_grid"][()])
            return base + increment
        return np.asarray(h5[f"{group_path}/truth/model_target_log_ai"][()])

    @staticmethod
    def _read_seismic(h5: h5py.File, group_path: str, sample_kind: str) -> np.ndarray:
        if sample_kind in SEISMIC_VARIANT_KINDS:
            return np.asarray(h5[f"{group_path}/seismic_observed"][()])
        return np.asarray(h5[f"{group_path}/seismic/seismic_model_consistent"][()])

    @staticmethod
    def _read_optional_dataset(
        h5: h5py.File,
        row: Mapping[str, Any],
        column: str,
        fallback: str,
    ) -> np.ndarray:
        path = _clean_path(row.get(column)) or fallback
        if path not in h5:
            raise KeyError(f"Missing HDF5 dataset for {column}: {path}")
        return np.asarray(h5[path][()])


__all__ = ["SCHEMA_VERSION", "SEISMIC_VARIANT_KINDS", "TimeSyntheticSample", "TimeV1Benchmark"]
