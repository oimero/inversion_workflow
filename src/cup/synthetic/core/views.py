"""Contracts for materialized seismic views in Synthoseis v5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Any, Mapping, Sequence

from cup.synthetic.schemas import (
    RANDOM_STREAM_CONTRACT_VERSION,
    SCIENCE_REVISION,
    SEISMIC_OPERATOR_CONTRACT_VERSION,
    SEISMIC_VIEW_CONTRACT_VERSION,
)


FORWARD_PARAMETER_KINDS = frozenset(
    {"wavelet_phase_rotation", "wavelet_time_shift"}
)
SAMPLED_SEISMIC_KINDS = frozenset(
    {
        "axis_static",
        "global_gain",
        "tracewise_gain",
        "axis_lateral_gain",
        "rgt_lateral_gain",
        "empirical_rgt_gain",
        "additive_white_noise",
        "additive_colored_noise",
    }
)
SUPPORTED_OPERATOR_KINDS = FORWARD_PARAMETER_KINDS | SAMPLED_SEISMIC_KINDS
SEISMIC_VIEW_REQUIRED_FIELDS = ("view_id", "view_spec_sha256")


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with one stable scientific spelling."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(element) for element in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("view spec cannot contain non-finite floats")
            return float(item)
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        raise TypeError(f"view spec value is not JSON-compatible: {type(item)!r}")

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_seismic_view_metadata(
    *, view_id: str, view_spec_sha256: str,
    operator_ids: Sequence[str], operator_kinds: Sequence[str],
    operator_contract_versions: Mapping[str, str],
    view_spec_canonical_json: str, parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    view = str(view_id).strip()
    fingerprint = str(view_spec_sha256).strip()
    if not view or not fingerprint:
        raise ValueError("view_id and view_spec_sha256 must be non-empty")
    return {
        "contract_version": SEISMIC_VIEW_CONTRACT_VERSION,
        "view_id": view,
        "view_spec_sha256": fingerprint,
        "view_spec_canonical_json": str(view_spec_canonical_json),
        "operator_ids": list(operator_ids),
        "operator_kinds": list(operator_kinds),
        "operator_contract_versions": dict(operator_contract_versions),
        "operator_parameters": dict(parameters or {}),
        "random_stream_contract_version": RANDOM_STREAM_CONTRACT_VERSION,
    }


def validate_seismic_view_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("seismic view metadata must be a mapping")
    missing = [key for key in SEISMIC_VIEW_REQUIRED_FIELDS if key not in value]
    if missing:
        raise ValueError(f"seismic view metadata lacks fields: {missing}")
    if value.get("contract_version") != SEISMIC_VIEW_CONTRACT_VERSION:
        raise ValueError("seismic view metadata does not use seismic_views_v1")
    canonical = value.get("view_spec_canonical_json")
    if canonical is not None:
        try:
            json.loads(str(canonical))
        except json.JSONDecodeError as exc:
            raise ValueError("view_spec_canonical_json must be valid JSON") from exc
        if sha256_text(str(canonical)) != str(value.get("view_spec_sha256")):
            raise ValueError("view_spec_sha256 does not match canonical view spec")
    return dict(value)


def validate_operator_sequence(
    operator_ids: Sequence[str],
    operators: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Validate and return an immutable operator sequence.

    Forward operators form a prefix and each forward kind occurs at most once.
    Sampled-seismic operators may follow in the declared order.
    """

    ids = tuple(str(value) for value in operator_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("a seismic view cannot repeat an operator ID")
    seen_sampled = False
    seen_forward_kinds: set[str] = set()
    for operator_id in ids:
        if operator_id not in operators:
            raise ValueError(f"seismic view references unknown operator: {operator_id}")
        kind = str(operators[operator_id].get("kind") or "")
        if kind not in SUPPORTED_OPERATOR_KINDS:
            raise ValueError(f"unsupported seismic operator kind: {kind!r}")
        if kind in FORWARD_PARAMETER_KINDS:
            if seen_sampled:
                raise ValueError(
                    "forward parameter operators must precede sampled seismic operators"
                )
            if kind in seen_forward_kinds:
                raise ValueError(
                    f"a view may contain at most one forward operator of kind {kind!r}"
                )
            seen_forward_kinds.add(kind)
        else:
            seen_sampled = True
    return ids


def _validate_operator_parameters(operator_id: str, spec: Mapping[str, Any]) -> None:
    """Validate numeric operator values before they enter a view fingerprint."""
    kind = str(spec.get("kind") or "")

    def finite_number(key: str, *, positive: bool = False) -> float:
        try:
            value = float(spec[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"seismic operator {operator_id!r}.{key} must be numeric"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f"seismic operator {operator_id!r}.{key} must be finite")
        if positive and value <= 0.0:
            raise ValueError(f"seismic operator {operator_id!r}.{key} must be positive")
        return value

    if kind in {"wavelet_phase_rotation", "wavelet_time_shift"}:
        finite_number("degrees" if kind == "wavelet_phase_rotation" else "seconds")
    elif kind == "axis_static":
        shift = spec.get("shift")
        if not isinstance(shift, Mapping) or not str(shift.get("unit") or "").strip():
            raise ValueError(f"seismic operator {operator_id!r}.shift requires value and unit")
        try:
            value = float(shift["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"seismic operator {operator_id!r}.shift.value must be numeric") from exc
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(f"seismic operator {operator_id!r}.shift.value must be finite and non-zero")
    elif kind in {"global_gain", "tracewise_gain", "axis_lateral_gain"}:
        finite_number("log_sigma", positive=True)
        if kind != "global_gain":
            value = finite_number("lateral_correlation_fraction", positive=True)
            if value > 1.0:
                raise ValueError(f"seismic operator {operator_id!r}.lateral_correlation_fraction must be <= 1")
        if kind == "axis_lateral_gain":
            value = finite_number("axis_correlation_fraction", positive=True)
            if value > 1.0:
                raise ValueError(f"seismic operator {operator_id!r}.axis_correlation_fraction must be <= 1")
    elif kind == "rgt_lateral_gain":
        sigma_keys = (
            "rgt_log_sigma",
            "lateral_log_sigma",
            "interaction_log_sigma",
        )
        sigma_values = [finite_number(key) for key in sigma_keys]
        if any(value < 0.0 for value in sigma_values):
            raise ValueError(
                f"seismic operator {operator_id!r} gain sigmas must be non-negative"
            )
        if not any(value > 0.0 for value in sigma_values):
            raise ValueError(
                f"seismic operator {operator_id!r} requires at least one positive gain sigma"
            )
        for key in (
            "rgt_correlation_length",
            "lateral_correlation_length_m",
            "interaction_rgt_correlation_length",
            "interaction_lateral_correlation_length_m",
            "max_abs_log_gain",
        ):
            finite_number(key, positive=True)
        try:
            rank = int(spec["interaction_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"seismic operator {operator_id!r}.interaction_rank must be an integer"
            ) from exc
        if isinstance(spec["interaction_rank"], bool) or rank != spec["interaction_rank"] or rank < 1:
            raise ValueError(
                f"seismic operator {operator_id!r}.interaction_rank must be a positive integer"
            )
    elif kind == "empirical_rgt_gain":
        source = str(spec.get("parameter_source") or "")
        if source != "seismic_amplitude_calibration":
            raise ValueError(
                f"seismic operator {operator_id!r}.parameter_source must be "
                "'seismic_amplitude_calibration'"
            )
        scale = finite_number("mean_pattern_scale", positive=True)
        if scale > 1.0:
            raise ValueError(
                f"seismic operator {operator_id!r}.mean_pattern_scale must be <= 1"
            )
        resolved_keys = {
            "calibration_schema_version",
            "calibration_artifact_sha256",
            "calibration_contract_fingerprint_sha256",
            "template_sha256",
            "rgt_knots",
            "mean_log_gain_rgt",
        }
        present = resolved_keys.intersection(spec)
        if present and present != resolved_keys:
            raise ValueError(
                f"seismic operator {operator_id!r} has an incomplete resolved amplitude template"
            )
        if present:
            knots = [float(value) for value in spec["rgt_knots"]]
            template = [float(value) for value in spec["mean_log_gain_rgt"]]
            if len(knots) < 2 or len(knots) != len(template):
                raise ValueError(
                    f"seismic operator {operator_id!r} resolved template arrays are misaligned"
                )
            if any(not math.isfinite(value) for value in knots + template):
                raise ValueError(
                    f"seismic operator {operator_id!r} resolved template must be finite"
                )
            if any(right <= left for left, right in zip(knots, knots[1:])):
                raise ValueError(
                    f"seismic operator {operator_id!r}.rgt_knots must be strictly increasing"
                )
            if abs(float(statistics.median(template))) > 1e-10:
                raise ValueError(
                    f"seismic operator {operator_id!r} resolved template violates its zero-median gauge"
                )
    elif kind == "additive_white_noise":
        finite_number("rms_fraction", positive=True)
    elif kind == "additive_colored_noise":
        finite_number("rms_fraction", positive=True)
        correlation = spec.get("axis_correlation")
        if not isinstance(correlation, Mapping) or not str(correlation.get("unit") or "").strip():
            raise ValueError(f"seismic operator {operator_id!r}.axis_correlation requires value and unit")
        try:
            value = float(correlation["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"seismic operator {operator_id!r}.axis_correlation.value must be numeric") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"seismic operator {operator_id!r}.axis_correlation.value must be positive")


@dataclass(frozen=True)
class SeismicViewSpec:
    view_id: str
    operator_ids: tuple[str, ...]
    operators: Mapping[str, Mapping[str, Any]]
    canonical_json: str
    sha256: str

    @property
    def operator_kinds(self) -> tuple[str, ...]:
        return tuple(str(self.operators[item]["kind"]) for item in self.operator_ids)

    @property
    def forward_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            item
            for item in self.operator_ids
            if self.operators[item]["kind"] in FORWARD_PARAMETER_KINDS
        )

    @property
    def sampled_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            item
            for item in self.operator_ids
            if self.operators[item]["kind"] in SAMPLED_SEISMIC_KINDS
        )

    def operator_spec_sha256(self, operator_id: str) -> str:
        operator = self.operators[operator_id]
        return sha256_text(
            canonical_json(
                {
                    "operator_id": operator_id,
                    "operator": operator,
                    "science_revision": SCIENCE_REVISION,
                    "operator_contract_version": SEISMIC_OPERATOR_CONTRACT_VERSION,
                }
            )
        )

    def metadata(self) -> dict[str, Any]:
        operator_parameters = {
            item: dict(self.operators[item]) for item in self.operator_ids
        }
        random_identity = {
            item: {
                "operator_id": item,
                "operator_spec_sha256": self.operator_spec_sha256(item),
                "coefficient_namespace": "seismic_view_operator",
            }
            for item in self.operator_ids
        }
        return {
            "view_id": self.view_id,
            "operator_ids": list(self.operator_ids),
            "operator_kinds": list(self.operator_kinds),
            "operator_contract_versions": {
                item: SEISMIC_OPERATOR_CONTRACT_VERSION for item in self.operator_ids
            },
            "operator_parameters": operator_parameters,
            "view_spec_canonical_json": self.canonical_json,
            "view_spec_sha256": self.sha256,
            "science_revision": SCIENCE_REVISION,
            "seismic_view_contract_version": SEISMIC_VIEW_CONTRACT_VERSION,
            "random_stream_contract_version": RANDOM_STREAM_CONTRACT_VERSION,
            "random_stream_identity": random_identity,
        }


def resolve_view_specs(config: Mapping[str, Any]) -> tuple[SeismicViewSpec, ...]:
    """Parse the operator directory and view list without guessing defaults."""

    if not isinstance(config, Mapping):
        raise ValueError("seismic_views must be a mapping")
    unknown_config = sorted(set(config) - {"operators", "views"})
    if unknown_config:
        raise ValueError(
            "seismic_views contains unknown keys: "
            f"{unknown_config}"
        )
    operators_raw = config.get("operators")
    views_raw = config.get("views")
    if not isinstance(operators_raw, Mapping) or not isinstance(views_raw, Sequence) or isinstance(views_raw, (str, bytes)):
        raise ValueError("seismic_views requires operators mapping and views list")
    operators: dict[str, dict[str, Any]] = {}
    for key, value in operators_raw.items():
        operator_id = str(key)
        if not operator_id or operator_id in operators:
            raise ValueError(f"duplicate or empty seismic operator ID: {operator_id!r}")
        if not isinstance(value, Mapping):
            raise ValueError(f"seismic operator {operator_id!r} must be a mapping")
        spec = {str(name): item for name, item in value.items()}
        kind = str(spec.get("kind") or "")
        if kind not in SUPPORTED_OPERATOR_KINDS:
            raise ValueError(f"unsupported seismic operator kind: {kind!r}")
        allowed_by_kind = {
            "wavelet_phase_rotation": {"kind", "degrees"},
            "wavelet_time_shift": {"kind", "seconds"},
            "axis_static": {"kind", "shift"},
            "global_gain": {"kind", "log_sigma"},
            "tracewise_gain": {"kind", "log_sigma", "lateral_correlation_fraction"},
            "axis_lateral_gain": {"kind", "log_sigma", "lateral_correlation_fraction", "axis_correlation_fraction"},
            "rgt_lateral_gain": {
                "kind", "rgt_log_sigma", "rgt_correlation_length",
                "lateral_log_sigma", "lateral_correlation_length_m",
                "interaction_log_sigma", "interaction_rank",
                "interaction_rgt_correlation_length",
                "interaction_lateral_correlation_length_m",
                "max_abs_log_gain",
            },
            "empirical_rgt_gain": {
                "kind", "parameter_source", "mean_pattern_scale",
                "calibration_schema_version", "calibration_artifact_sha256",
                "calibration_contract_fingerprint_sha256", "template_sha256",
                "rgt_knots", "mean_log_gain_rgt",
            },
            "additive_white_noise": {"kind", "rms_fraction"},
            "additive_colored_noise": {"kind", "rms_fraction", "axis_correlation"},
        }
        unknown = sorted(set(spec) - allowed_by_kind[kind])
        if unknown:
            raise ValueError(f"seismic operator {operator_id!r} contains unknown keys: {unknown}")
        required_by_kind = {
            "wavelet_phase_rotation": {"degrees"},
            "wavelet_time_shift": {"seconds"},
            "axis_static": {"shift"},
            "global_gain": {"log_sigma"},
            "tracewise_gain": {"log_sigma", "lateral_correlation_fraction"},
            "axis_lateral_gain": {"log_sigma", "lateral_correlation_fraction", "axis_correlation_fraction"},
            "rgt_lateral_gain": {
                "rgt_log_sigma", "rgt_correlation_length",
                "lateral_log_sigma", "lateral_correlation_length_m",
                "interaction_log_sigma", "interaction_rank",
                "interaction_rgt_correlation_length",
                "interaction_lateral_correlation_length_m",
                "max_abs_log_gain",
            },
            "empirical_rgt_gain": {"parameter_source", "mean_pattern_scale"},
            "additive_white_noise": {"rms_fraction"},
            "additive_colored_noise": {"rms_fraction", "axis_correlation"},
        }
        missing = sorted(required_by_kind[kind] - set(spec))
        if missing:
            raise ValueError(f"seismic operator {operator_id!r} lacks required keys: {missing}")
        _validate_operator_parameters(operator_id, spec)
        operators[operator_id] = spec

    result: list[SeismicViewSpec] = []
    seen_views: set[str] = set()
    for raw_view in views_raw:
        if not isinstance(raw_view, Mapping):
            raise ValueError("each seismic view must be a mapping")
        unknown_view = sorted(set(raw_view) - {"view_id", "operator_ids"})
        if unknown_view:
            raise ValueError(
                "seismic view contains unknown keys: "
                f"{unknown_view}"
            )
        view_id = str(raw_view.get("view_id") or "")
        if not view_id or view_id in seen_views:
            raise ValueError(f"duplicate or empty seismic view ID: {view_id!r}")
        operator_ids_raw = raw_view.get("operator_ids")
        if not isinstance(operator_ids_raw, Sequence) or isinstance(operator_ids_raw, (str, bytes)):
            raise ValueError(f"seismic view {view_id!r} requires operator_ids list")
        if not operator_ids_raw:
            raise ValueError(
                f"seismic view {view_id!r} must reference at least one operator; "
                "base is represented by the empty views list"
            )
        operator_ids = validate_operator_sequence(operator_ids_raw, operators)
        view_payload = {
            "view_id": view_id,
            "operator_ids": list(operator_ids),
            "operators": {item: operators[item] for item in operator_ids},
            "science_revision": SCIENCE_REVISION,
            "seismic_view_contract_version": SEISMIC_VIEW_CONTRACT_VERSION,
            "seismic_operator_contract_version": SEISMIC_OPERATOR_CONTRACT_VERSION,
            "random_stream_contract_version": RANDOM_STREAM_CONTRACT_VERSION,
        }
        serialized = canonical_json(view_payload)
        result.append(
            SeismicViewSpec(
                view_id=view_id,
                operator_ids=operator_ids,
                operators=operators,
                canonical_json=serialized,
                sha256=sha256_text(serialized),
            )
        )
        seen_views.add(view_id)
    return tuple(result)


def validate_view_units(config: Mapping[str, Any], *, axis_unit: str) -> None:
    """Reject an operator directory whose axis-valued parameters use another domain."""
    operators = config.get("operators")
    if not isinstance(operators, Mapping):
        raise ValueError("seismic_views.operators must be a mapping")
    for operator_id, operator in operators.items():
        kind = str(operator.get("kind") or "")
        if kind == "axis_static":
            shift = operator.get("shift")
            if not isinstance(shift, Mapping) or str(shift.get("unit")) != axis_unit:
                raise ValueError(f"seismic operator {operator_id!r} axis_static unit must be {axis_unit!r}")
        if kind == "additive_colored_noise":
            correlation = operator.get("axis_correlation")
            if not isinstance(correlation, Mapping) or str(correlation.get("unit")) != axis_unit:
                raise ValueError(f"seismic operator {operator_id!r} axis_correlation unit must be {axis_unit!r}")


__all__ = [
    "FORWARD_PARAMETER_KINDS",
    "SAMPLED_SEISMIC_KINDS",
    "SUPPORTED_OPERATOR_KINDS",
    "SEISMIC_VIEW_REQUIRED_FIELDS",
    "SeismicViewSpec",
    "canonical_json",
    "resolve_view_specs",
    "sha256_text",
    "build_seismic_view_metadata",
    "validate_seismic_view_metadata",
    "validate_operator_sequence",
    "validate_view_units",
]
