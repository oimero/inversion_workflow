"""Shared project configuration for the seismic workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def merge_dict_defaults(config: dict[str, Any], key: str, defaults: dict[str, Any]) -> None:
    """Merge *defaults* into ``config[key]`` in place."""
    value = config.get(key)
    if value is None:
        config[key] = dict(defaults)
        return
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping, got {type(value).__name__}.")
    merged = dict(defaults)
    merged.update(value)
    config[key] = merged


def deep_merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict that recursively merges *updates* into *base*."""
    out = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


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
    domain: str
    zgy_inline_chunk_size: int | None = None
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
        domain = _required_text(config, "domain", path="seismic").casefold()
        if domain not in {"time", "depth"}:
            raise ValueError("seismic.domain must be 'time' or 'depth'.")
        chunk_size = _optional_int(config, "zgy_inline_chunk_size", path="seismic")
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("seismic.zgy_inline_chunk_size must be a positive integer or null.")
        return cls(
            file=_required_text(config, "file", path="seismic"),
            type=seismic_type,
            domain=domain,
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
            "domain": self.domain,
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
        return cls(
            data_root=str(config.get("data_root", "data")),
            output_root=str(config.get("output_root", "scripts/output")),
            assets=AssetsConfig.from_mapping(config.get("assets")),
            seismic=SeismicConfig.from_mapping(config.get("seismic")),
            well_curves=WellCurveContract.from_mapping(config.get("well_curves")),
            spatial_debias=SpatialDebiasConfig.from_mapping(config.get("spatial_debias")),
        )
