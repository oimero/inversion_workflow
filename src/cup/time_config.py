"""Shared project configuration for the time-domain workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _required_text(config: Mapping[str, Any], key: str, *, path: str) -> str:
    value = config.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


def _optional_int(config: Mapping[str, Any], key: str, *, path: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}.{key} must be an integer or null.") from exc


def _category_tuple(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list.")
    categories = tuple(str(item).strip() for item in value)
    if not categories or any(not item for item in categories):
        raise ValueError(f"{path} must contain non-empty category names.")
    if len(set(categories)) != len(categories):
        raise ValueError(f"{path} must not contain duplicates.")
    return categories


@dataclass(frozen=True)
class AssetsConfig:
    well_heads_file: str
    las_dir: str
    well_trace_dir: str
    well_tops_file: str
    time_depth_dir: str

    @classmethod
    def from_mapping(cls, value: Any) -> "AssetsConfig":
        config = _mapping(value, path="assets")
        return cls(
            well_heads_file=_required_text(config, "well_heads_file", path="assets"),
            las_dir=_required_text(config, "las_dir", path="assets"),
            well_trace_dir=_required_text(config, "well_trace_dir", path="assets"),
            well_tops_file=_required_text(config, "well_tops_file", path="assets"),
            time_depth_dir=_required_text(config, "time_depth_dir", path="assets"),
        )


@dataclass(frozen=True)
class SeismicConfig:
    file: str
    type: str
    zgy_inline_chunk_size: int
    iline: int | None = None
    xline: int | None = None
    istep: int | None = None
    xstep: int | None = None
    iline_byte: int | None = None
    xline_byte: int | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> "SeismicConfig":
        config = _mapping(value, path="seismic")
        seismic_type = _required_text(config, "type", path="seismic").casefold()
        if seismic_type not in {"zgy", "segy"}:
            raise ValueError("seismic.type must be 'zgy' or 'segy'.")
        chunk_size = _optional_int(config, "zgy_inline_chunk_size", path="seismic")
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("seismic.zgy_inline_chunk_size must be a positive integer.")
        return cls(
            file=_required_text(config, "file", path="seismic"),
            type=seismic_type,
            zgy_inline_chunk_size=chunk_size,
            iline=_optional_int(config, "iline", path="seismic"),
            xline=_optional_int(config, "xline", path="seismic"),
            istep=_optional_int(config, "istep", path="seismic"),
            xstep=_optional_int(config, "xstep", path="seismic"),
            iline_byte=_optional_int(config, "iline_byte", path="seismic"),
            xline_byte=_optional_int(config, "xline_byte", path="seismic"),
        )

    def as_dict(self) -> dict[str, Any]:
        values = {
            "file": self.file,
            "type": self.type,
            "zgy_inline_chunk_size": self.zgy_inline_chunk_size,
            "iline": self.iline,
            "xline": self.xline,
            "istep": self.istep,
            "xstep": self.xstep,
            "iline_byte": self.iline_byte,
            "xline_byte": self.xline_byte,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class WellCurveContract:
    required_categories: tuple[str, ...]
    selected_categories: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "WellCurveContract":
        config = _mapping(value, path="well_curves")
        required = _category_tuple(config.get("required_categories"), path="well_curves.required_categories")
        selected = _category_tuple(config.get("selected_categories"), path="well_curves.selected_categories")
        missing = sorted(set(required) - set(selected))
        if missing:
            raise ValueError(
                "well_curves.required_categories must be included in selected_categories; "
                f"missing={missing}."
            )
        return cls(required_categories=required, selected_categories=selected)


@dataclass(frozen=True)
class SpatialDebiasConfig:
    cluster_radius_m: float

    @classmethod
    def from_mapping(cls, value: Any) -> "SpatialDebiasConfig":
        config = _mapping(value, path="spatial_debias")
        try:
            radius = float(config.get("cluster_radius_m"))
        except (TypeError, ValueError) as exc:
            raise ValueError("spatial_debias.cluster_radius_m must be a positive number.") from exc
        if not radius > 0.0:
            raise ValueError("spatial_debias.cluster_radius_m must be a positive number.")
        return cls(cluster_radius_m=radius)


@dataclass(frozen=True)
class TimeWorkflowConfig:
    data_root: str
    output_root: str
    assets: AssetsConfig
    seismic: SeismicConfig
    well_curves: WellCurveContract
    spatial_debias: SpatialDebiasConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TimeWorkflowConfig":
        config = dict(value)
        validate_retired_time_config_keys(config)
        return cls(
            data_root=str(config.get("data_root", "data")),
            output_root=str(config.get("output_root", "scripts/output")),
            assets=AssetsConfig.from_mapping(config.get("assets")),
            seismic=SeismicConfig.from_mapping(config.get("seismic")),
            well_curves=WellCurveContract.from_mapping(config.get("well_curves")),
            spatial_debias=SpatialDebiasConfig.from_mapping(config.get("spatial_debias")),
        )


_RETIRED_KEYS: dict[str, str] = {
    "well_inventory.source_data": "use top-level assets",
    "well_inventory.seismic": "use top-level seismic",
    "well_screen.source_data": "use top-level assets.las_dir",
    "well_screen.curve_selection": "use top-level well_curves",
    "well_screen.llm": "LLM classification is not implemented and the placeholder config was removed",
    "well_screen.export": "LAS output naming and formatting are fixed workflow contracts",
    "well_preprocess.output_las_dir": "the output directory is fixed to preprocessed_las",
    "well_preprocess.required_categories": "use top-level well_curves.required_categories",
    "well_preprocess.selected_categories": "use top-level well_curves.selected_categories",
    "well_preprocess.mnemonic_standardization": "mnemonic standardization is mandatory",
    "well_preprocess.unit_standardization": "unit standardization and its QC report are mandatory",
    "well_preprocess.constant_runs.replacement": "constant runs are always replaced with missing values",
    "well_preprocess.outliers.strategy": "the global-quantile-with-override strategy is fixed",
    "well_preprocess.outliers.replacement": "outliers are always replaced with missing values",
    "well_preprocess.export": "LAS NULL and write format are fixed workflow contracts",
    "well_trajectory.well_trace_dir": "use top-level assets.well_trace_dir",
    "well_trajectory.seismic": "use top-level seismic",
    "well_trajectory.survey_qc.enabled": "survey QC is mandatory",
    "well_trajectory.output.sampled_trajectory_dir": "the output directory is fixed to trajectory_points",
    "well_auto_tie.time_depth_dir": "use top-level assets.time_depth_dir",
    "well_auto_tie.well_trace_dir": "use top-level assets.well_trace_dir",
    "well_auto_tie.well_tops_file": "use top-level assets.well_tops_file",
    "well_auto_tie.seismic": "use top-level seismic",
    "wavelet_generation.evaluation_wells.status": "evaluation wells are fixed to tie_status=success",
    "wavelet_generation.generation.mode": "consensus optimization is the fixed generation mode",
    "wavelet_generation.generation.pca.include_mean_wavelet": "the unused option was removed",
    "wavelet_generation.generation.optimizer.strategy": "random-then-Powell is the fixed optimizer",
    "wavelet_generation.generation.objective.name": "the objective is defined by its weights",
    "wavelet_generation.spatial_debias.cluster_radius_m": "use top-level spatial_debias.cluster_radius_m",
    "wavelet_generation.export": "selected output names are fixed; use --debug for debug artifacts",
}

_SOURCE_RUN_SECTIONS = (
    "well_screen",
    "well_preprocess",
    "well_trajectory",
    "well_auto_tie",
    "wavelet_generation",
)

_TIME_WORKFLOW_SECTIONS = (
    "well_inventory",
    *_SOURCE_RUN_SECTIONS,
)


def _has_path(config: Mapping[str, Any], path: str) -> bool:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return False
        current = current[key]
    return True


def validate_retired_time_config_keys(config: Mapping[str, Any]) -> None:
    """Reject removed keys instead of silently accepting stale configuration."""
    retired = dict(_RETIRED_KEYS)
    for section in _TIME_WORKFLOW_SECTIONS:
        retired.setdefault(f"{section}.assets", "use top-level assets")
        retired.setdefault(f"{section}.seismic", "use top-level seismic")
        retired.setdefault(f"{section}.well_curves", "use top-level well_curves")
        retired.setdefault(
            f"{section}.spatial_debias.cluster_radius_m",
            "use top-level spatial_debias.cluster_radius_m",
        )
    for section in _SOURCE_RUN_SECTIONS:
        retired[f"{section}.source_runs.mode"] = "latest-run discovery is now implicit"
    found = [(path, message) for path, message in retired.items() if _has_path(config, path)]
    if not found:
        return
    details = "; ".join(f"{path}: {message}" for path, message in found)
    raise ValueError(f"Retired time-workflow configuration key(s): {details}")
