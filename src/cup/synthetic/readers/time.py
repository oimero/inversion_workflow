"""Reader for Synthoseis-lite v4 time-domain artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from cup.impedance import (
    CanonicalIncrementContract,
    validate_contract_compatibility,
    validate_lfm_producer_contract,
    validate_sample_axis,
)
from cup.synthetic.core import (
    validate_dataset_metadata,
    validate_training_manifest,
)
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION
from cup.utils.io import require_contract_fingerprint


SCHEMA_VERSION = BENCHMARK_SCHEMA_VERSION
SEISMIC_VARIANT_KINDS = {"seismic_variant"}


@dataclass(frozen=True)
class TimeV2SyntheticSample:
    sample_id: str
    sample_kind: str
    row: dict[str, Any]
    sample_domain: str
    target_log_ai: np.ndarray
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    input_lfm_log_ai: np.ndarray
    seismic_input: np.ndarray
    seismic_model_consistent: np.ndarray
    valid_mask: np.ndarray
    physics_valid_mask: np.ndarray
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
    if "/seismic_variants/" in group_path:
        return group_path.split("/seismic_variants/", maxsplit=1)[0]
    return group_path


def _pad_forward_to_model(
    values: np.ndarray, model_shape: tuple[int, int]
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == model_shape:
        return array
    if array.shape == (model_shape[0], model_shape[1] - 1):
        padded = np.zeros(model_shape, dtype=np.float64)
        padded[:, 1:] = array
        return padded
    raise ValueError(
        f"Cannot align forward grid {array.shape} to model grid {model_shape}."
    )


class TimeV2Benchmark:
    """Read-only accessor around ``synthoseis_lite_v4`` time artifacts."""

    schema = SCHEMA_VERSION
    sample_domain = "time"

    def __init__(
        self,
        run_dir: str | Path,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.h5_path = self.run_dir / "synthetic_benchmark.h5"
        self.index_path = self.run_dir / "sample_index.csv"
        self.manifest_path = self.run_dir / "benchmark_manifest.json"
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"synthetic_benchmark.h5 not found: {self.h5_path}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"sample_index.csv not found: {self.index_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"benchmark_manifest.json not found: {self.manifest_path}"
            )

        self.manifest = _json(self.manifest_path)
        actual_schema = str(
            self.manifest.get("schema") or self.manifest.get("schema_version") or ""
        )
        if actual_schema != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported time Synthoseis schema {actual_schema!r}; expected {SCHEMA_VERSION!r}."
            )
        if str(self.manifest.get("sample_domain") or "").casefold() != "time":
            raise ValueError("Time v4 reader requires sample_domain=time.")
        self.increment_contract = CanonicalIncrementContract.from_mapping(
            self.manifest.get("increment_contract") or {}
        )
        if self.increment_contract.sample_domain != "time":
            raise ValueError("Time benchmark increment_contract must use sample_domain=time.")
        self.lfm_contract = validate_lfm_producer_contract(
            self.manifest.get("lfm_contract") or {}
        )
        validate_contract_compatibility(self.increment_contract, self.lfm_contract)
        validate_training_manifest(self.manifest, sample_domain="time")
        require_contract_fingerprint(self.manifest, label=f"benchmark {self.run_dir}")

        self.index = pd.read_csv(self.index_path, dtype=str, keep_default_na=False)
        if self.index.empty:
            raise ValueError(f"empty sample_index.csv: {self.index_path}")
        required = {
            "sample_id",
            "parent_realization_id",
            "sample_kind",
            "status",
            "hdf5_group",
            "evaluation_role",
        }
        missing = required - set(self.index)
        if missing:
            raise ValueError(f"sample_index.csv lacks columns: {sorted(missing)}")
        forbidden_probe_kinds = {
            "frequency_probe",
            "frequency_probe_seismic_variant",
        }.intersection(set(self.index["sample_kind"]))
        if forbidden_probe_kinds:
            raise ValueError(
                "frequency_probe samples do not belong to the v4 canonical benchmark: "
                f"{sorted(forbidden_probe_kinds)}"
            )
        if bool(self.index["sample_id"].duplicated().any()):
            raise ValueError("sample_index.csv contains duplicate sample_id values.")
        if not set(self.index["evaluation_role"]) <= {
            "development_pool",
            "geometry_holdout",
        }:
            raise ValueError(
                "sample_index.csv contains unknown evaluation_role values."
            )
        held_out = str(
            dict(self.manifest.get("split_policy") or {}).get(
                "held_out_geometry_family"
            )
            or ""
        )
        if held_out:
            invalid = self.index["geometry_family"].eq(held_out) & ~self.index[
                "evaluation_role"
            ].eq("geometry_holdout")
            if bool(invalid.any()):
                raise ValueError(
                    "Held-out geometry rows must carry evaluation_role=geometry_holdout."
                )
        self._rows = {
            str(row["sample_id"]): row.to_dict() for _, row in self.index.iterrows()
        }
        for row in self._rows.values():
            if (
                str(row.get("status") or "") != "ok"
                or str(row.get("sample_kind") or "") not in SEISMIC_VARIANT_KINDS
            ):
                continue
            source_id = _clean_path(row.get("source_sample_id"))
            source = self._rows.get(source_id)
            if source is None:
                raise ValueError(f"Variant has an invalid source: {row['sample_id']}")
            if (
                str(source.get("parent_realization_id"))
                != str(row.get("parent_realization_id"))
                or str(source.get("evaluation_role"))
                != str(row.get("evaluation_role"))
            ):
                raise ValueError(
                    f"Variant crosses parent realization or evaluation role: {row['sample_id']}"
                )

        with h5py.File(self.h5_path, "r") as h5:
            if (
                h5.attrs.get("schema") != SCHEMA_VERSION
                and h5.attrs.get("schema_version") != SCHEMA_VERSION
            ):
                raise ValueError(f"HDF5 schema does not match {SCHEMA_VERSION}.")
            if h5.attrs.get("sample_domain") != "time":
                raise ValueError("HDF5 sample_domain does not match time.")
            if bool(h5.attrs.get("qc_only", False)) != bool(
                self.manifest.get("qc_only", False)
            ):
                raise ValueError("HDF5 qc_only does not match manifest.")
            for key in ("suite", "global_seed"):
                if key in self.manifest and str(h5.attrs.get(key)) != str(
                    self.manifest[key]
                ):
                    raise ValueError(f"HDF5 {key} does not match manifest.")

            def visit(_: str, value: Any) -> None:
                if isinstance(value, h5py.Dataset):
                    validate_dataset_metadata(value, sample_domain="time")

            h5.visititems(visit)

            successful = self.index[self.index["status"].astype(str).eq("ok")]
            for _, row in successful.iterrows():
                group_path = _clean_path(row.get("hdf5_group"))
                if not group_path or group_path not in h5:
                    raise KeyError(
                        "sample_index.csv references missing HDF5 group "
                        f"{group_path!r}."
                    )

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
            if "split" not in frame:
                raise ValueError(
                    "Time v4 train/validation/test split is derived by the training adapter."
                )
            frame = frame[frame["split"].astype(str).eq(split)]
        return [str(value) for value in frame["sample_id"].tolist()]

    def row(self, sample_id: str) -> dict[str, Any]:
        try:
            return dict(self._rows[str(sample_id)])
        except KeyError as exc:
            raise KeyError(f"Unknown sample_id: {sample_id}") from exc

    def load_sample(self, sample_id: str) -> TimeV2SyntheticSample:
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
            model_shape = tuple(np.asarray(target).shape)
            seismic = _pad_forward_to_model(
                self._read_seismic(h5, group_path, sample_kind), model_shape
            )
            consistent = _pad_forward_to_model(
                h5[f"{root_path}/seismic/seismic_model_consistent"][()], model_shape
            )
            valid = np.asarray(
                h5[f"{root_path}/masks/valid_mask_model"][()], dtype=bool
            )
            physics_valid = np.asarray(
                h5[f"{root_path}/masks/physics_valid_mask"][()], dtype=bool
            )
            lateral = np.asarray(
                h5[f"{root_path}/axes/lateral_m"][()], dtype=np.float64
            )
            twt = np.asarray(h5[f"{root_path}/axes/twt_model_s"][()], dtype=np.float64)
            validate_sample_axis(twt, self.increment_contract)
            canonical_background = np.asarray(
                h5[f"{root_path}/priors/canonical_background_log_ai"][()],
                dtype=np.float64,
            )
            target_increment = np.asarray(
                h5[f"{root_path}/targets/target_increment_log_ai"][()],
                dtype=np.float64,
            )
            variant_id = _clean_path(row.get("lfm_variant_id")) or "controlled_default"
            input_lfm_path = _clean_path(row.get("input_lfm_log_ai_dataset"))
            if not input_lfm_path:
                input_lfm_path = (
                    f"{root_path}/priors/input_lfm_variants/{variant_id}/log_ai"
                )
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
                "canonical_background_log_ai": canonical_background,
                "input_lfm_log_ai": np.asarray(h5[input_lfm_path][()], dtype=np.float64),
            }
        for name, values in {
            "valid": valid,
            "physics_valid": physics_valid,
            "seismic": seismic,
            "consistent": consistent,
            "canonical_background": canonical_background,
            "target_increment": target_increment,
            "input_lfm": priors["input_lfm_log_ai"],
        }.items():
            if values.shape != model_shape:
                raise ValueError(
                    f"{name}/target shape mismatch for {sample_id}: {values.shape} vs {model_shape}"
                )
        finite = np.isfinite(target) & np.isfinite(canonical_background) & np.isfinite(target_increment)
        if np.any(finite & (np.abs(target - canonical_background - target_increment) > 1e-5)):
            raise ValueError(f"Canonical increment decomposition mismatch for {sample_id}.")
        return TimeV2SyntheticSample(
            sample_id=str(sample_id),
            sample_kind=sample_kind,
            row=row,
            sample_domain="time",
            target_log_ai=np.asarray(target, dtype=np.float64),
            canonical_background_log_ai=canonical_background,
            target_increment_log_ai=target_increment,
            input_lfm_log_ai=priors["input_lfm_log_ai"],
            seismic_input=seismic,
            seismic_model_consistent=consistent,
            valid_mask=valid,
            physics_valid_mask=physics_valid,
            lateral_m=lateral,
            twt_model_s=twt,
            priors={
                key: np.asarray(value, dtype=np.float64)
                for key, value in priors.items()
            },
        )

    @staticmethod
    def _read_target(h5: h5py.File, group_path: str, sample_kind: str) -> np.ndarray:
        if sample_kind in {"frequency_probe", "frequency_probe_seismic_variant"}:
            raise ValueError(
                "frequency_probe samples do not belong to the v4 canonical benchmark."
            )
        return np.asarray(h5[f"{group_path}/truth/model_target_log_ai"][()])

    @staticmethod
    def _read_seismic(h5: h5py.File, group_path: str, sample_kind: str) -> np.ndarray:
        if sample_kind in SEISMIC_VARIANT_KINDS:
            return np.asarray(h5[f"{group_path}/seismic_observed"][()])
        return np.asarray(h5[f"{group_path}/seismic/seismic_model_consistent"][()])

    @staticmethod
    def _read_optional_dataset(
        h5: h5py.File, row: Mapping[str, Any], column: str, fallback: str
    ) -> np.ndarray:
        path = _clean_path(row.get(column)) or fallback
        if path not in h5:
            raise KeyError(f"Missing HDF5 dataset for {column}: {path}")
        return np.asarray(h5[path][()])


__all__ = [
    "SCHEMA_VERSION",
    "SEISMIC_VARIANT_KINDS",
    "TimeV2Benchmark",
    "TimeV2SyntheticSample",
]
