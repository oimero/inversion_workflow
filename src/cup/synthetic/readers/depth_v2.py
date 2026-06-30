"""Strict reader for frozen Synthoseis-lite v2 depth benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from cup.synthetic.depth.config import SCHEMA_VERSION
from cup.synthetic.core import (
    validate_dataset_metadata,
    validate_manifest_files,
    validate_training_manifest,
)


@dataclass(frozen=True)
class DepthSyntheticSample:
    sample_id: str
    sample_kind: str
    row: dict[str, Any]
    sample_domain: str
    depth_basis: str
    target_log_ai: np.ndarray
    vp_model_mps: np.ndarray
    seismic_observed: np.ndarray
    seismic_model_consistent: np.ndarray
    valid_mask: np.ndarray
    observed_valid_mask: np.ndarray
    physics_valid_mask: np.ndarray
    lateral_m: np.ndarray
    tvdss_model_m: np.ndarray
    priors: dict[str, np.ndarray]

    @property
    def seismic_input(self) -> np.ndarray:
        """Explicit network input selected by the v2 contract."""
        return self.seismic_observed


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


class DepthV2Benchmark:
    """Validate and read ``synthoseis_lite_v2`` depth artifacts only."""

    schema = SCHEMA_VERSION
    sample_domain = "depth"

    def __init__(self, run_dir: str | Path, *, expected_forward_model_inputs_sha256: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.h5_path = self.run_dir / "synthetic_benchmark.h5"
        self.index_path = self.run_dir / "sample_index.csv"
        self.manifest_path = self.run_dir / "benchmark_manifest.json"
        for path in (self.h5_path, self.index_path, self.manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required Synthoseis v2 artifact not found: {path}")
        self.manifest = _json(self.manifest_path)
        actual_schema = str(self.manifest.get("schema") or self.manifest.get("schema_version") or "missing")
        if actual_schema != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported Synthoseis schema {actual_schema!r}; expected {SCHEMA_VERSION!r}. "
                "Re-run calibrate/generate to rebuild the benchmark."
            )
        if self.manifest.get("sample_domain") != "depth" or self.manifest.get("depth_basis") != "tvdss":
            raise ValueError("Synthoseis v2 reader requires sample_domain=depth and depth_basis=tvdss.")
        validate_training_manifest(self.manifest, sample_domain="depth")
        validate_manifest_files(self.run_dir, dict(self.manifest.get("files") or {}))
        recorded_forward = str(self.manifest.get("forward_model_inputs_sha256") or "")
        if not recorded_forward:
            raise ValueError("Benchmark manifest lacks forward_model_inputs_sha256; rebuild required.")
        if expected_forward_model_inputs_sha256 is not None and recorded_forward != str(expected_forward_model_inputs_sha256):
            raise ValueError("Benchmark forward_model_inputs_sha256 does not match the requested model run.")
        self.index = pd.read_csv(self.index_path, dtype=str, keep_default_na=False)
        required = {
            "sample_id", "parent_realization_id", "sample_kind", "source_sample_id",
            "hdf5_group", "sample_domain", "depth_basis", "suite", "evaluation_role",
            "forward_model_inputs_sha256",
        }
        missing = sorted(required - set(self.index))
        if missing:
            raise ValueError(f"sample_index.csv lacks v2 columns: {missing}")
        if self.index.empty or self.index["sample_id"].duplicated().any():
            raise ValueError("sample_index.csv must be non-empty with unique sample_id values.")
        if not self.index["sample_domain"].eq("depth").all() or not self.index["depth_basis"].eq("tvdss").all():
            raise ValueError("sample_index.csv contains a wrong domain or depth basis.")
        if not self.index["suite"].eq("field_conditioned").all():
            raise ValueError("Depth v2 sample_index.csv must contain only field_conditioned samples.")
        if not set(self.index["sample_kind"]) <= {"base", "seismic_variant"}:
            raise ValueError("Depth v2 sample_index.csv contains canonical/probe/unknown sample kinds.")
        forbidden_columns = {
            "sample_axis", "twt_model_s", "twt_highres_s", "probe_frequency_hz",
            "probe_phase", "probe_amplitude_multiplier", "probe_lateral_shape",
        }
        present_forbidden = sorted(forbidden_columns.intersection(self.index.columns))
        if present_forbidden:
            raise ValueError(f"Depth v2 sample_index.csv contains forbidden time/probe fields: {present_forbidden}")
        if not self.index["forward_model_inputs_sha256"].eq(recorded_forward).all():
            raise ValueError("sample_index.csv forward hash differs from benchmark manifest.")
        if not set(self.index["evaluation_role"]) <= {"development_pool", "geometry_holdout"}:
            raise ValueError("sample_index.csv contains unknown evaluation_role values.")
        held_out = str(dict(self.manifest.get("split_policy") or {}).get("held_out_geometry_family") or "")
        if held_out:
            invalid = self.index["geometry_family"].eq(held_out) & ~self.index["evaluation_role"].eq("geometry_holdout")
            if invalid.any():
                raise ValueError("Held-out geometry family appears outside geometry_holdout.")
        self._rows = {str(row["sample_id"]): row.to_dict() for _, row in self.index.iterrows()}
        for row in self._rows.values():
            if str(row.get("status") or "") != "ok":
                continue
            if row["sample_kind"] == "seismic_variant":
                source = self._rows.get(str(row["source_sample_id"]))
                if source is None or source["sample_kind"] != "base":
                    raise ValueError(f"Variant has an invalid base source: {row['sample_id']}")
                if source["parent_realization_id"] != row["parent_realization_id"] or source["evaluation_role"] != row["evaluation_role"]:
                    raise ValueError(f"Variant crosses parent realization or evaluation role: {row['sample_id']}")
        self._validate_hdf5(recorded_forward)

    def _validate_hdf5(self, forward_hash: str) -> None:
        with h5py.File(self.h5_path, "r") as h5:
            if h5.attrs.get("schema") != SCHEMA_VERSION:
                raise ValueError("HDF5 schema does not match synthoseis_lite_v2.")
            if h5.attrs.get("sample_domain") != "depth" or h5.attrs.get("depth_basis") != "tvdss":
                raise ValueError("HDF5 root has the wrong sample domain or basis.")
            if h5.attrs.get("forward_model_inputs_sha256") != forward_hash:
                raise ValueError("HDF5 forward hash differs from benchmark manifest.")
            if bool(h5.attrs.get("qc_only", False)) != bool(
                self.manifest.get("qc_only", False)
            ):
                raise ValueError("HDF5 qc_only does not match manifest.")
            for key in ("suite", "global_seed", "impedance_calibration_sha256"):
                if key in self.manifest and str(h5.attrs.get(key)) != str(
                    self.manifest[key]
                ):
                    raise ValueError(f"HDF5 {key} does not match manifest.")

            def validate_dataset(name: str, value: h5py.Dataset | h5py.Group) -> None:
                if not isinstance(value, h5py.Dataset):
                    return
                validate_dataset_metadata(value, sample_domain="depth")

            h5.visititems(validate_dataset)
            factor = None
            for row in self._rows.values():
                if str(row.get("status") or "") != "ok":
                    continue
                base_path = f"/realizations/{row['parent_realization_id']}"
                if base_path not in h5:
                    raise KeyError(f"Missing base realization group: {base_path}")
                for field in ("hdf5_group", "seismic_input_dataset", "physics_target_dataset"):
                    referenced = str(row.get(field) or "")
                    if not referenced or referenced not in h5:
                        raise KeyError(f"sample_index.csv references missing HDF5 path {field}={referenced!r}.")
                high = np.asarray(h5[f"{base_path}/axes/tvdss_highres_m"][()], dtype=np.float64)
                model = np.asarray(h5[f"{base_path}/axes/tvdss_model_m"][()], dtype=np.float64)
                if high.size < 2 or model.size < 2 or np.any(np.diff(high) <= 0.0) or np.any(np.diff(model) <= 0.0):
                    raise ValueError(f"Invalid TVDSS axes: {base_path}")
                local_factor = int(round(np.diff(model[:2])[0] / np.diff(high[:2])[0]))
                if local_factor < 1 or not np.array_equal(high[::local_factor], model):
                    raise ValueError(f"Highres/model TVDSS axes are not nested: {base_path}")
                factor = local_factor if factor is None else factor
                if factor != local_factor:
                    raise ValueError("Benchmark mixes vertical oversampling factors.")

    def sample_ids(self, *, kinds: set[str] | None = None, status: str = "ok", split: str | None = None) -> list[str]:
        frame = self.index
        if status:
            frame = frame[frame["status"].eq(status)]
        if kinds:
            frame = frame[frame["sample_kind"].isin(kinds)]
        if split is not None:
            if "split" not in frame:
                raise ValueError("Depth v2 train/validation/test split is derived by the training adapter.")
            frame = frame[frame["split"].eq(split)]
        return frame["sample_id"].tolist()

    def row(self, sample_id: str) -> dict[str, Any]:
        if str(sample_id) not in self._rows:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        return dict(self._rows[str(sample_id)])

    def load_sample(self, sample_id: str) -> DepthSyntheticSample:
        row = self.row(sample_id)
        base_path = f"/realizations/{row['parent_realization_id']}"
        input_path = str(row["seismic_input_dataset"])
        with h5py.File(self.h5_path, "r") as h5:
            arrays = {
                "target": np.asarray(h5[f"{base_path}/truth/model_target_log_ai"][()], dtype=np.float64),
                "vp": np.asarray(h5[f"{base_path}/truth/vp_model_mps"][()], dtype=np.float64),
                "observed": np.asarray(h5[input_path][()], dtype=np.float64),
                "consistent": np.asarray(h5[f"{base_path}/seismic/seismic_model_consistent"][()], dtype=np.float64),
                "valid": np.asarray(h5[f"{base_path}/masks/valid_mask_model"][()], dtype=bool),
                "observed_valid": np.asarray(h5[f"{base_path}/masks/observed_valid_mask"][()], dtype=bool),
                "physics_valid": np.asarray(h5[f"{base_path}/masks/physics_valid_mask"][()], dtype=bool),
                "lateral": np.asarray(h5[f"{base_path}/axes/lateral_m"][()], dtype=np.float64),
                "axis": np.asarray(h5[f"{base_path}/axes/tvdss_model_m"][()], dtype=np.float64),
                "lfm_ideal": np.asarray(h5[f"{base_path}/priors/lfm_ideal"][()], dtype=np.float64),
                "lfm_degraded": np.asarray(h5[f"{base_path}/priors/lfm_controlled_degraded"][()], dtype=np.float64),
            }
            if row["sample_kind"] == "seismic_variant":
                arrays["observed_valid"] = np.asarray(h5[f"{row['hdf5_group']}/observed_valid_mask"][()], dtype=bool)
        shape = arrays["target"].shape
        for name in ("vp", "observed", "consistent", "valid", "observed_valid", "physics_valid", "lfm_ideal", "lfm_degraded"):
            if arrays[name].shape != shape:
                raise ValueError(f"N-point shape contract failed for {sample_id}: {name}={arrays[name].shape}, target={shape}")
        return DepthSyntheticSample(
            sample_id=str(sample_id), sample_kind=str(row["sample_kind"]), row=row,
            sample_domain="depth", depth_basis="tvdss",
            target_log_ai=arrays["target"], vp_model_mps=arrays["vp"],
            seismic_observed=arrays["observed"], seismic_model_consistent=arrays["consistent"],
            valid_mask=arrays["valid"], observed_valid_mask=arrays["observed_valid"],
            physics_valid_mask=arrays["physics_valid"], lateral_m=arrays["lateral"],
            tvdss_model_m=arrays["axis"], priors={"lfm_ideal": arrays["lfm_ideal"], "lfm_controlled_degraded": arrays["lfm_degraded"]},
        )


__all__ = ["DepthSyntheticSample", "DepthV2Benchmark"]
