"""Strict Synthoseis-lite v5 parent/view reader.

The reader deliberately has no compatibility branch for the former flat
``sample_index.csv``/variant layout.  A benchmark is a set of parent
realizations plus an optional set of materialized seismic views.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from cup.impedance import CanonicalIncrementContract
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION, require_science_contract
from cup.synthetic.core import validate_dataset_metadata, validate_training_manifest
from cup.synthetic.core.views import sha256_text, validate_seismic_view_metadata
from cup.synthetic.core.v5_artifacts import REALIZATION_INDEX_COLUMNS, VIEW_INDEX_COLUMNS


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def validate_benchmark_header(
    value: Mapping[str, Any], *, sample_domain: str, label: str
) -> None:
    """Validate the shared v5 manifest header before opening HDF5."""
    schema = str(value.get("schema") or value.get("schema_version") or "")
    if schema != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"{label} schema {schema!r} does not match {BENCHMARK_SCHEMA_VERSION!r}"
        )
    if str(value.get("sample_domain") or "").casefold() != sample_domain:
        raise ValueError(f"{label} requires sample_domain={sample_domain}")
    if sample_domain == "depth" and value.get("depth_basis") != "tvdss":
        raise ValueError(f"{label} requires depth_basis=tvdss")
    require_science_contract(value, label=label)


_LEGACY_FIELD_NAMES = frozenset(
    {
        "sample_index",
        "seismic_variant_id",
        "seismic_variant_contract_version",
        "seismic_variants",
        "lfm_degradation_contract_version",
        "lfm_degradation",
        "lfm_controlled_degraded",
        "controlled_degraded",
        "controlled_default",
        "combined_moderate",
        "seismic_input_dataset",
        "seismic_model_consistent_dataset",
    }
)


def _legacy_field_paths(value: Any, *, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in _LEGACY_FIELD_NAMES:
                found.append(child_path)
            found.extend(_legacy_field_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_legacy_field_paths(child, path=f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class V5SyntheticSample:
    sample_id: str
    sample_kind: str
    row: dict[str, Any]
    sample_domain: str
    depth_basis: str | None
    target_log_ai: np.ndarray
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    input_lfm_log_ai: np.ndarray
    seismic_input: np.ndarray
    seismic_observed: np.ndarray
    seismic_model_consistent: np.ndarray
    valid_mask: np.ndarray
    lateral_m: np.ndarray
    sample_axis: np.ndarray
    priors: dict[str, np.ndarray]

    @property
    def tvdss_model_m(self) -> np.ndarray:
        if self.sample_domain != "depth":
            raise AttributeError("tvdss_model_m is only available for depth samples")
        return self.sample_axis

    @property
    def twt_model_s(self) -> np.ndarray:
        if self.sample_domain != "time":
            raise AttributeError("twt_model_s is only available for time samples")
        return self.sample_axis


@dataclass(frozen=True)
class V5SeismicView:
    """One materialized view without duplicating parent target semantics."""

    realization_id: str
    view_id: str
    row: dict[str, Any]
    seismic_observed: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    metadata: dict[str, Any]
    qc: dict[str, Any]


class V5Benchmark:
    """Shared parent/view reader implementation used by both domain adapters."""

    schema = BENCHMARK_SCHEMA_VERSION

    def __init__(self, run_dir: str | Path, *, sample_domain: str) -> None:
        self.run_dir = Path(run_dir)
        self.sample_domain = str(sample_domain).casefold()
        if self.sample_domain not in {"time", "depth"}:
            raise ValueError("sample_domain must be time or depth")
        self.h5_path = self.run_dir / "synthetic_benchmark.h5"
        self.manifest_path = self.run_dir / "benchmark_manifest.json"
        self.realization_index_path = self.run_dir / "realization_index.csv"
        self.view_index_path = self.run_dir / "seismic_view_index.csv"
        if self.run_dir.joinpath("sample_index.csv").exists():
            raise ValueError("v5 benchmark must not contain legacy sample_index.csv")
        for path in (self.h5_path, self.manifest_path, self.realization_index_path, self.view_index_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required Synthoseis v5 artifact not found: {path}")
        self.manifest = _json(self.manifest_path)
        legacy_manifest_fields = _legacy_field_paths(self.manifest)
        if legacy_manifest_fields:
            raise ValueError(
                "v5 manifest contains legacy fields: "
                f"{legacy_manifest_fields[:5]}"
            )
        for legacy_key in (
            "lfm_degradation_contract_version",
            "seismic_variant_contract_version",
        ):
            if legacy_key in self.manifest:
                raise ValueError(f"v5 manifest contains legacy field {legacy_key!r}")
        validate_benchmark_header(self.manifest, sample_domain=self.sample_domain, label="benchmark manifest")
        validate_training_manifest(
            self.manifest,
            sample_domain=self.sample_domain,
        )
        expected_unit = "s" if self.sample_domain == "time" else "m"
        if _text(self.manifest.get("sample_unit")) != expected_unit:
            raise ValueError(
                f"benchmark manifest requires sample_unit={expected_unit!r}"
            )
        actual = _text(self.manifest.get("schema") or self.manifest.get("schema_version"))
        if actual != BENCHMARK_SCHEMA_VERSION:
            raise ValueError(f"v5 reader requires {BENCHMARK_SCHEMA_VERSION}; got {actual!r}")
        require_science_contract(self.manifest, label="benchmark manifest")
        self.increment_contract = CanonicalIncrementContract.from_mapping(
            self.manifest.get("increment_contract") or {}
        )
        self.realizations = pd.read_csv(self.realization_index_path, dtype=str, keep_default_na=False)
        self.views = pd.read_csv(self.view_index_path, dtype=str, keep_default_na=False)
        self._validate_indexes()
        self._rows: dict[str, dict[str, Any]] = {}
        for _, raw in self.realizations.iterrows():
            row = raw.to_dict()
            rid = _text(row.get("realization_id"))
            self._rows[rid] = {
                **row,
                "sample_id": rid,
                "sample_kind": "base",
                "status": "ok",
                "source_sample_id": "",
                "hdf5_group": f"/realizations/{rid}",
                "seismic_observed_dataset": f"/realizations/{rid}/seismic/seismic_observed",
                "model_consistent_seismic_dataset": f"/realizations/{rid}/seismic/seismic_model_consistent",
                "valid_mask_dataset": f"/realizations/{rid}/masks/valid_mask",
            }
        for _, raw in self.views.iterrows():
            row = raw.to_dict()
            rid = _text(row.get("parent_realization_id"))
            view_id = _text(row.get("view_id"))
            sid = f"{rid}__view__{view_id}"
            self._rows[sid] = {
                **row,
                "sample_id": sid,
                "sample_kind": "seismic_view",
                "status": "ok",
                "source_sample_id": rid,
                "hdf5_group": f"/realizations/{rid}/seismic_views/{view_id}",
                "seismic_observed_dataset": f"/realizations/{rid}/seismic_views/{view_id}/seismic_observed",
                "model_consistent_seismic_dataset": f"/realizations/{rid}/seismic/seismic_model_consistent",
                "valid_mask_dataset": f"/realizations/{rid}/masks/valid_mask",
            }
        self.index = pd.DataFrame.from_records(list(self._rows.values()))
        self._validate_hdf5()

    def _validate_indexes(self) -> None:
        required_realization = set(REALIZATION_INDEX_COLUMNS)
        required_view = set(VIEW_INDEX_COLUMNS)
        missing_real = sorted(required_realization - set(self.realizations.columns))
        missing_view = sorted(required_view - set(self.views.columns))
        if missing_real or missing_view:
            raise ValueError(f"v5 index columns missing: realizations={missing_real}, views={missing_view}")
        legacy_columns = sorted(
            _LEGACY_FIELD_NAMES.intersection(self.realizations.columns)
            | _LEGACY_FIELD_NAMES.intersection(self.views.columns)
        )
        if legacy_columns:
            raise ValueError(
                f"v5 indexes contain legacy columns: {legacy_columns}"
            )
        if "parent_realization_id" in self.realizations.columns:
            raise ValueError(
                "realization_index.csv must not contain the parent_realization_id alias column"
            )
        if self.realizations.empty:
            raise ValueError("realization_index.csv must contain at least one parent realization")
        if self.realizations["realization_id"].duplicated().any():
            raise ValueError("realization_index.csv contains duplicate realization_id")
        if not self.views.empty and self.views.duplicated(subset=["parent_realization_id", "view_id"]).any():
            raise ValueError("seismic_view_index.csv contains duplicate parent/view pairs")
        if not self.realizations["sample_domain"].eq(self.sample_domain).all():
            raise ValueError("realization_index.csv contains a wrong sample_domain")
        expected_unit = "s" if self.sample_domain == "time" else "m"
        if not self.realizations["sample_unit"].eq(expected_unit).all():
            raise ValueError("realization_index.csv contains a wrong sample_unit")
        if not self.views.empty and not self.views["sample_domain"].eq(self.sample_domain).all():
            raise ValueError("seismic_view_index.csv contains a wrong sample_domain")
        if not self.views.empty and not self.views["sample_unit"].eq(expected_unit).all():
            raise ValueError("seismic_view_index.csv contains a wrong sample_unit")
        known = set(self.realizations["realization_id"].astype(str))
        if not set(self.views["parent_realization_id"].astype(str)).issubset(known):
            raise ValueError("seismic_view_index.csv references an unknown parent realization")
        if not self.realizations["evaluation_role"].isin({"development_pool", "geometry_holdout"}).all():
            raise ValueError("unknown evaluation_role in realization_index.csv")
        if not self.views.empty:
            parent_rows = self.realizations.set_index("realization_id", drop=False)
            for _, view in self.views.iterrows():
                parent_id = _text(view["parent_realization_id"])
                if _text(view.get("realization_id")) != parent_id:
                    raise ValueError(
                        "seismic_view_index.csv realization_id must equal its parent_realization_id"
                    )
                parent = parent_rows.loc[parent_id]
                if _text(view.get("n_valid")) and _text(parent.get("n_valid")):
                    if _text(view["n_valid"]) != _text(parent["n_valid"]):
                        raise ValueError(
                            f"seismic view {view['view_id']!r} has a different n_valid from its parent"
                        )
                value = view["view_spec_canonical_json"]
                try:
                    parsed = json.loads(str(value))
                except json.JSONDecodeError as exc:
                    raise ValueError("view_spec_canonical_json must be valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("view_spec_canonical_json must encode an object")
                if sha256_text(str(value)) != _text(view["view_spec_sha256"]):
                    raise ValueError(
                        f"view {view['view_id']!r} has an invalid view_spec_sha256"
                    )

    def _validate_hdf5(self) -> None:
        expected_unit = "s" if self.sample_domain == "time" else "m"
        with h5py.File(self.h5_path, "r") as h5:
            legacy_tokens = {
                "lfm_degradation_contract_version",
                "seismic_variant_contract_version",
                "controlled_default",
                "lfm_controlled_degraded",
            }
            for key in h5.attrs:
                if str(key) in legacy_tokens:
                    raise ValueError(f"HDF5 benchmark contains legacy field {key!r}")
            require_science_contract(h5.attrs, label="benchmark HDF5 root")
            if _text(h5.attrs.get("schema") or h5.attrs.get("schema_version")) != BENCHMARK_SCHEMA_VERSION:
                raise ValueError("HDF5 schema does not match v5")
            if _text(h5.attrs.get("sample_domain")) != self.sample_domain:
                raise ValueError("HDF5 sample_domain does not match index")
            if _text(h5.attrs.get("sample_unit")) != expected_unit:
                raise ValueError("HDF5 sample_unit does not match index")
            for _, row in self.realizations.iterrows():
                rid = _text(row["realization_id"])
                root = f"/realizations/{rid}"
                if root not in h5:
                    raise KeyError(f"Missing HDF5 parent realization: {root}")
                expected_parent_paths = {
                    "hdf5_group": root,
                    "base_seismic_dataset": f"{root}/seismic/seismic_observed",
                    "model_consistent_seismic_dataset": f"{root}/seismic/seismic_model_consistent",
                    "target_log_ai_dataset": f"{root}/truth/model_target_log_ai",
                    "canonical_background_dataset": f"{root}/priors/canonical_background_log_ai",
                    "target_increment_dataset": f"{root}/targets/target_increment_log_ai",
                    "valid_mask_dataset": f"{root}/masks/valid_mask",
                }
                for field, expected in expected_parent_paths.items():
                    recorded = _text(row.get(field))
                    if recorded and recorded != expected:
                        raise ValueError(
                            f"realization_index.csv {field} for {rid!r} does not match HDF5 contract"
                        )
                    if expected not in h5:
                        raise KeyError(f"Missing HDF5 parent dataset: {expected}")
            for _, row in self.views.iterrows():
                parent_id = _text(row["parent_realization_id"])
                view_id = _text(row["view_id"])
                path = f"/realizations/{parent_id}/seismic_views/{view_id}"
                if path not in h5:
                    raise KeyError(f"Missing HDF5 seismic view: {path}")
                expected_view_paths = {
                    "hdf5_group": path,
                    "seismic_observed_dataset": f"{path}/seismic_observed",
                    "model_consistent_seismic_dataset": f"/realizations/{parent_id}/seismic/seismic_model_consistent",
                    "valid_mask_dataset": f"/realizations/{parent_id}/masks/valid_mask",
                    "operator_trace_dataset": f"{path}/operator_trace_json",
                }
                for field, expected in expected_view_paths.items():
                    recorded = _text(row.get(field))
                    if recorded and recorded != expected:
                        raise ValueError(
                            f"seismic_view_index.csv {field} for ({parent_id!r}, {view_id!r}) "
                            "does not match HDF5 contract"
                        )
                    if expected not in h5:
                        raise KeyError(f"Missing HDF5 seismic view dataset: {expected}")
                parent_path = f"/realizations/{parent_id}"
                parent_mask = h5[f"{parent_path}/masks/valid_mask"]
                observed = h5[f"{path}/seismic_observed"]
                if observed.shape != parent_mask.shape:
                    raise ValueError(
                        f"HDF5 seismic view {path} shape differs from its parent mask"
                    )
                if _text(row.get("n_valid")) and int(float(_text(row["n_valid"]))) != int(np.count_nonzero(parent_mask[()])):
                    raise ValueError(
                        f"seismic view {path} n_valid differs from its parent mask"
                    )
                group = h5[path]
                if _text(group.attrs.get("view_spec_sha256")) != _text(row["view_spec_sha256"]):
                    raise ValueError(f"seismic view {path} has a stale view_spec_sha256 attribute")
                if _text(group.attrs.get("view_spec_canonical_json")) != _text(row["view_spec_canonical_json"]):
                    raise ValueError(f"seismic view {path} has a stale canonical view spec attribute")
            def validate_item(name: str, value: h5py.Dataset | h5py.Group) -> None:
                path_tokens = set(str(name).split("/"))
                if path_tokens.intersection(_LEGACY_FIELD_NAMES):
                    raise ValueError(f"HDF5 benchmark contains legacy path: {name}")
                if isinstance(value, h5py.Dataset):
                    validate_dataset_metadata(value, sample_domain=self.sample_domain)

            h5.visititems(validate_item)

    def sample_ids(self, *, kinds: set[str] | None = None, status: str = "ok", split: str | None = None) -> list[str]:
        frame = self.index
        if status and status != "ok":
            return []
        if kinds:
            frame = frame[frame["sample_kind"].isin(kinds)]
        if split is not None:
            if "split" not in frame:
                raise ValueError("train/validation/test split must be supplied by split_assignment.csv")
            frame = frame[frame["split"].eq(split)]
        return [str(value) for value in frame["sample_id"].tolist()]

    def realization_ids(self, *, evaluation_role: str | None = None) -> list[str]:
        frame = self.realizations
        if evaluation_role is not None:
            frame = frame[frame["evaluation_role"].eq(str(evaluation_role))]
        return [str(value) for value in frame["realization_id"].tolist()]

    def available_view_ids(self, realization_id: str) -> list[str]:
        rid = str(realization_id)
        if rid not in set(self.realizations["realization_id"].astype(str)):
            raise KeyError(f"Unknown realization_id: {rid}")
        frame = self.views[self.views["parent_realization_id"].astype(str).eq(rid)]
        return [str(value) for value in frame["view_id"].tolist()]

    def row(self, sample_id: str) -> dict[str, Any]:
        try:
            return dict(self._rows[str(sample_id)])
        except KeyError as exc:
            raise KeyError(f"Unknown sample_id: {sample_id}") from exc

    def load_sample(self, sample_id: str) -> V5SyntheticSample:
        row = self.row(sample_id)
        rid = _text(row.get("parent_realization_id") or row.get("realization_id"))
        root = f"/realizations/{rid}"
        with h5py.File(self.h5_path, "r") as h5:
            target = np.asarray(h5[f"{root}/truth/model_target_log_ai"][()], dtype=np.float64)
            canonical = np.asarray(h5[f"{root}/priors/canonical_background_log_ai"][()], dtype=np.float64)
            increment = np.asarray(h5[f"{root}/targets/target_increment_log_ai"][()], dtype=np.float64)
            vp_path = f"{root}/truth/vp_model_mps"
            vp = np.asarray(h5[vp_path][()], dtype=np.float64) if vp_path in h5 else np.full_like(target, np.nan)
            seismic_path = row.get("seismic_observed_dataset") or row.get("base_seismic_dataset")
            consistent_path = row.get("model_consistent_seismic_dataset")
            if not seismic_path or not consistent_path:
                raise ValueError(
                    f"v5 realization row lacks canonical seismic dataset paths: {sample_id}"
                )
            seismic = np.asarray(h5[str(seismic_path)][()], dtype=np.float64)
            consistent = np.asarray(h5[str(consistent_path)][()], dtype=np.float64)
            valid = np.asarray(h5[row["valid_mask_dataset"]][()], dtype=bool)
            lateral = np.asarray(h5[f"{root}/axes/lateral_m"][()], dtype=np.float64)
            axis_name = "tvdss_model_m" if self.sample_domain == "depth" else "twt_model_s"
            axis = np.asarray(h5[f"{root}/axes/{axis_name}"][()], dtype=np.float64)
        expected = target.shape
        for name, value in {"canonical_background": canonical, "target_increment": increment, "vp": vp, "seismic": seismic, "consistent": consistent, "valid": valid}.items():
            if value.shape != expected:
                raise ValueError(f"v5 shape mismatch for {sample_id}: {name}={value.shape}, target={expected}")
        if np.any(valid & (~np.isfinite(target) | ~np.isfinite(canonical) | ~np.isfinite(increment) | ~np.isfinite(seismic) | ~np.isfinite(consistent))):
            raise ValueError(f"non-finite values inside valid_mask: {sample_id}")
        finite = np.isfinite(target) & np.isfinite(canonical) & np.isfinite(increment)
        if np.any(finite & (np.abs(target - canonical - increment) > 1e-5)):
            raise ValueError(f"canonical increment decomposition mismatch: {sample_id}")
        return V5SyntheticSample(
            sample_id=str(sample_id), sample_kind=str(row["sample_kind"]), row=row,
            sample_domain=self.sample_domain, depth_basis="tvdss" if self.sample_domain == "depth" else None,
            target_log_ai=target, canonical_background_log_ai=canonical,
            target_increment_log_ai=increment, input_lfm_log_ai=canonical,
            seismic_input=seismic, seismic_observed=seismic,
            seismic_model_consistent=consistent, valid_mask=valid,
            lateral_m=lateral, sample_axis=axis,
            priors={"canonical_background_log_ai": canonical, "input_lfm_log_ai": canonical},
        )

    def load_realization(self, realization_id: str) -> V5SyntheticSample:
        """Load one parent realization and its base seismic input."""
        rid = str(realization_id)
        if rid not in set(self.realizations["realization_id"].astype(str)):
            raise KeyError(f"Unknown realization_id: {rid}")
        return self.load_sample(rid)

    def load_seismic_view(self, realization_id: str, view_id: str) -> V5SeismicView:
        rid = str(realization_id)
        vid = str(view_id)
        matches = self.views[
            self.views["parent_realization_id"].astype(str).eq(rid)
            & self.views["view_id"].astype(str).eq(vid)
        ]
        if len(matches) != 1:
            raise KeyError(f"Unknown seismic view: ({rid}, {vid})")
        row = matches.iloc[0].to_dict()
        path = f"/realizations/{rid}/seismic_views/{vid}"
        with h5py.File(self.h5_path, "r") as h5:
            group = h5[path]
            observed = np.asarray(group["seismic_observed"][()], dtype=np.float64)
            positive_gain = np.asarray(group["positive_gain"][()], dtype=np.float64)
            additive_noise = np.asarray(group["additive_noise"][()], dtype=np.float64)
            metadata: dict[str, Any] = {}
            for key, value in group.attrs.items():
                text_key = str(key)
                if text_key.endswith("_json"):
                    try:
                        metadata[text_key[:-5]] = json.loads(_text(value))
                    except json.JSONDecodeError:
                        metadata[text_key[:-5]] = _text(value)
                else:
                    metadata[text_key] = _text(value)
            qc = {
                str(key): value.item() if isinstance(value, np.generic) else value
                for key, value in group["qc"].attrs.items()
            }
        validate_seismic_view_metadata(metadata)
        return V5SeismicView(
            realization_id=rid,
            view_id=vid,
            row=row,
            seismic_observed=observed,
            positive_gain=positive_gain,
            additive_noise=additive_noise,
            metadata=metadata,
            qc=qc,
        )


__all__ = ["V5Benchmark", "V5SeismicView", "V5SyntheticSample"]
