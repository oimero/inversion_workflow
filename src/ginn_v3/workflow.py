"""Compose the independent PIAI training and inference workflow.

The workflow owns only input assembly and artifact loading.  The network,
data reader, acoustic operator, and joint trainer are separate v3 contracts,
which keeps time and depth runs on the same path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.config.artifacts import is_consumable_contract_status, latest_checked_run
from cup.lfm.artifacts import load_lfm_input
from cup.lfm.contracts import RUN_SCHEMA as LFM_RUN_SCHEMA
from cup.physics.relations import AIVelocityRelation
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from cup.well.controls import load_evaluation_support_for_run, load_well_control_set
from ginn_v3.config import InferenceConfig, KernelConfig, NetworkConfig, TrainingConfig
from ginn_v3.data import SurveyTraceSource, prepare_training_data
from ginn_v3.infer import Inverter, VolumePrediction
from ginn_v3.model import PIAINetwork
from ginn_v3.physics import AcousticPhysics
from ginn_v3.qc import write_well_qc
from ginn_v3.train import JointTrainer, load_checkpoint
from ginn_v3.types import Normalization, TrainingResult, TraceKey


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("experiments/ginn_v3/ginn_v3.yaml")


@dataclass(frozen=True)
class _Overrides:
    lfm_run_dir: Path | None = None
    variant_id: str | None = None
    well_control_run_dir: Path | None = None
    velocity_volume: Path | None = None
    trusted_well_names: tuple[str, ...] | None = None
    device: str | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class _Runtime:
    raw: Mapping[str, Any]
    section: Mapping[str, Any]
    workflow: WorkflowConfig
    config_path: Path
    lfm_run_dir: Path
    well_control_run_dir: Path
    variant_id: str
    lfm: Any
    survey: Any
    sample_axis: Any
    controls: Any
    evaluation_supports: Mapping[str, Any]
    source: Any
    data: Any
    physics: AcousticPhysics
    wavelet_time_s: np.ndarray
    normalization: Normalization
    velocity_mps: np.ndarray | None
    velocity_source: str
    trusted_well_names: tuple[str, ...]


@dataclass(frozen=True)
class LoadedBody:
    """Loaded v3 model and its explicit survey/data contracts."""

    checkpoint: Path
    checkpoint_payload: Mapping[str, Any]
    workflow: WorkflowConfig
    lfm: Any
    survey: Any
    sample_axis: Any
    controls: Any
    reader: Any
    data: Any
    model: PIAINetwork
    physics: AcousticPhysics
    normalization: Normalization
    inference_config: InferenceConfig
    output_dir: Path | None = None

    def _inverter(self, batch_size: int | None = None) -> Inverter:
        size = self.inference_config.batch_size if batch_size is None else batch_size
        return Inverter(
            self.model,
            self.reader,
            device=next(self.model.parameters()).device,
            batch_size=size,
            min_support_samples=self.inference_config.min_support_samples,
        )

    def predict_traces(
        self,
        keys: Sequence[TraceKey],
        *,
        batch_size: int | None = None,
    ) -> Any:
        """Return unfiltered log-AI for explicit trace keys."""

        return self._inverter(batch_size).predict_traces(keys)

    def predict_volume(
        self,
        *,
        output_path: str | Path | None = None,
        batch_size: int | None = None,
        inline_slice: slice | None = None,
        xline_slice: slice | None = None,
    ) -> VolumePrediction:
        """Stream volume predictions while preserving unsupported samples."""

        if np.asarray(self.lfm.log_ai).ndim != 3:
            raise ValueError("Volume inference requires a three-dimensional LFM variant.")
        shape = tuple(int(value) for value in np.asarray(self.lfm.log_ai).shape)
        # ``None`` is not a no-op index: ``array[None]`` adds an axis, so an
        # omitted slice must become an explicit full slice before indexing.
        inline_slice = slice(None) if inline_slice is None else inline_slice
        xline_slice = slice(None) if xline_slice is None else xline_slice
        ilines = np.arange(shape[0], dtype=np.int64)[inline_slice]
        xlines = np.arange(shape[1], dtype=np.int64)[xline_slice]
        if ilines.size == 0 or xlines.size == 0:
            raise ValueError("Requested volume slice is empty.")
        if output_path is None:
            base = self.output_dir or REPO_ROOT / "scripts" / "output"
            output_path = base / "piai_log_ai.npy"
        return self._inverter(batch_size).predict_volume(
            shape=shape,
            support_mask=np.asarray(self.lfm.valid_mask, dtype=bool),
            output_path=output_path,
            inline_indices=ilines,
            xline_indices=xlines,
        )


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load one experiment overlay and its common workflow configuration."""

    resolved = resolve_relative_path(path, root=REPO_ROOT)
    return _load_config_recursive(resolved, root=REPO_ROOT)


def _load_config_recursive(
    path: str | Path,
    *,
    root: Path,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Compose arbitrarily nested workflow overlays with cycle detection."""

    resolved = resolve_relative_path(path, root=root).resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"workflow_config cycle detected: {chain}")
    payload = load_yaml_config(resolved)
    parent_ref = payload.get("workflow_config")
    if parent_ref in (None, ""):
        return dict(payload)
    parent = _load_config_recursive(
        str(parent_ref),
        root=root,
        stack=(*stack, resolved),
    )
    overlay = {key: value for key, value in payload.items() if key != "workflow_config"}
    return deep_merge_dict(parent, overlay)


def _section(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    section = raw.get("ginn_v3_body_inversion")
    if not isinstance(section, Mapping):
        raise ValueError("Config lacks explicit ginn_v3_body_inversion section.")
    unknown = set(section) - {"inputs", "kernel", "network", "training", "inference", "trusted_well_names"}
    if unknown:
        raise ValueError(f"Unknown ginn_v3_body_inversion settings: {sorted(unknown)}")
    inputs = section.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("ginn_v3_body_inversion.inputs must be a mapping.")
    unknown_inputs = set(inputs) - {
        "lfm_run_dir", "variant_id", "well_control_run_dir", "velocity_volume", "ai_velocity_relation"
    }
    if unknown_inputs:
        raise ValueError(f"Unknown ginn_v3_body_inversion.inputs settings: {sorted(unknown_inputs)}")
    return section


def _required_input(section: Mapping[str, Any], key: str, override: object = None) -> str:
    if override is not None:
        value = str(override).strip()
    else:
        inputs = section.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("ginn_v3_body_inversion.inputs must be a mapping.")
        value = str(inputs.get(key) or "").strip()
    if not value:
        raise ValueError(f"ginn_v3_body_inversion input {key!r} must be explicit.")
    return value


def _resolve_inputs(
    section: Mapping[str, Any], overrides: _Overrides, *, workflow: WorkflowConfig
) -> tuple[Path, Path, str]:
    inputs = section.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("ginn_v3_body_inversion.inputs must be a mapping.")
    lfm_value = overrides.lfm_run_dir
    if lfm_value is None:
        lfm_value = str(inputs.get("lfm_run_dir") or "").strip()
    if str(lfm_value or "").strip():
        lfm_run_dir = resolve_relative_path(str(lfm_value), root=REPO_ROOT)
    else:
        output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
        def validate_lfm(path: Path) -> None:
            summary_path = path / "lfm_run_summary.json"
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if summary.get("schema_version") != LFM_RUN_SCHEMA or not is_consumable_contract_status(summary.get("status")):
                raise ValueError("LFM run is not a successful real_field_lfm_run_v3 artifact.")
            axis = dict(summary.get("sample_axis") or {})
            if axis.get("sample_domain") != workflow.seismic.domain or summary.get("depth_basis") != workflow.seismic.depth_basis:
                raise ValueError("LFM run domain does not match the workflow seismic domain.")
        lfm_run_dir = latest_checked_run(
            output_root,
            "real_field_lfm",
            required_files=["lfm_run_summary.json", "variant_manifest.csv"],
            validator=validate_lfm,
        )
    controls_value = overrides.well_control_run_dir
    if controls_value is None:
        controls_value = str(inputs.get("well_control_run_dir") or "").strip()
    if str(controls_value or "").strip():
        well_control_run_dir = resolve_relative_path(str(controls_value), root=REPO_ROOT)
    else:
        output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
        from cup.well.controls import SCHEMA_VERSION as WELL_CONTROL_SCHEMA
        def validate_controls(path: Path) -> None:
            with (path / "run_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if summary.get("schema_version") != WELL_CONTROL_SCHEMA or not is_consumable_contract_status(summary.get("status")):
                raise ValueError("Well-control run is not a successful current Step-6 artifact.")
            axis = dict(summary.get("sample_axis") or {})
            if axis.get("sample_domain") != workflow.seismic.domain or summary.get("depth_basis") != workflow.seismic.depth_basis:
                raise ValueError("Well-control run domain does not match the workflow seismic domain.")
        well_control_run_dir = latest_checked_run(
            output_root,
            "real_field_well_controls",
            required_files=["run_summary.json", "well_control_manifest.csv"],
            validator=validate_controls,
        )
    variant_id = _required_input(section, "variant_id", overrides.variant_id)
    return lfm_run_dir, well_control_run_dir, variant_id


def _resolve_trusted_names(
    section: Mapping[str, Any],
    overrides: _Overrides,
    *,
    controls: Any,
    workflow: WorkflowConfig,
) -> tuple[str, ...]:
    central_configured = workflow.well_usage.trusted_well_names
    central_control_names = {str(name).casefold() for name in controls.trusted_well_names}
    if central_configured is not None:
        missing_central = sorted(
            str(name) for name in central_configured
            if str(name).casefold() not in central_control_names
        )
        if missing_central:
            raise ValueError(
                "Central trusted wells are missing from the Step-6 control set: "
                f"{missing_central}"
            )
    value: object = overrides.trusted_well_names
    if value is None:
        value = section.get("trusted_well_names")
    if value is None:
        names = tuple(str(name) for name in controls.trusted_well_names)
    else:
        if not isinstance(value, (list, tuple)):
            raise ValueError("ginn_v3_body_inversion.trusted_well_names must be a list when provided.")
        names = tuple(str(item).strip() for item in value)
    if not names or any(not item for item in names) or len(set(name.casefold() for name in names)) != len(names):
        raise ValueError("trusted_well_names must contain unique non-empty names.")
    central = (
        {str(name).casefold() for name in central_configured}
        if central_configured is not None else central_control_names
    )
    if any(str(name).casefold() not in central for name in names):
        invalid = sorted(str(name) for name in names if str(name).casefold() not in central)
        raise ValueError(f"GINN trusted_well_names must be a subset of central trusted wells: {invalid}")
    invalid_roles = sorted(
        str(name) for name in names if workflow.well_usage.role_for(str(name)) != "trusted"
    )
    if invalid_roles:
        raise ValueError(f"GINN supervision may only use trusted wells: {invalid_roles}")
    return names


def _axis_values(axis: Any) -> np.ndarray:
    values = np.asarray(getattr(axis, "values", getattr(axis, "coordinates", axis)), dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("A SampleAxis must expose at least two finite values.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("A SampleAxis must be strictly increasing.")
    return values


def _require_matching_axes(*pairs: tuple[str, Any]) -> None:
    if not pairs:
        return
    reference_name, reference = pairs[0]
    reference_values = _axis_values(reference)
    reference_semantics = (
        str(getattr(reference, "domain", "")),
        str(getattr(reference, "unit", "")),
        getattr(reference, "depth_basis", None),
    )
    for name, axis in pairs[1:]:
        semantics = (
            str(getattr(axis, "domain", "")),
            str(getattr(axis, "unit", "")),
            getattr(axis, "depth_basis", None),
        )
        values = _axis_values(axis)
        if semantics != reference_semantics or not np.array_equal(values, reference_values):
            raise ValueError(f"{name} SampleAxis differs from {reference_name}.")


def _build_velocity(
    lfm: Any,
    section: Mapping[str, Any],
    velocity_volume: str | Path | None = None,
) -> tuple[np.ndarray | None, str]:
    """Read a velocity volume or derive it from an independent AI--Vp relation."""
    inputs = section["inputs"]
    path_value = velocity_volume if velocity_volume is not None else inputs.get("velocity_volume")
    relation_info = None if velocity_volume is not None else inputs.get("ai_velocity_relation")
    if lfm.sample_axis.domain == "time":
        if path_value is not None or relation_info is not None:
            raise ValueError("Time inversion does not accept velocity_volume or ai_velocity_relation.")
        return None, "time_no_velocity"
    if lfm.sample_axis.domain != "depth":
        raise ValueError(f"Unsupported seismic domain: {lfm.sample_axis.domain!r}.")
    if (path_value is None) == (relation_info is None):
        raise ValueError("Depth inversion requires exactly one of velocity_volume and ai_velocity_relation.")
    values = np.asarray(lfm.log_ai)
    support = np.asarray(lfm.valid_mask, dtype=bool)
    if path_value is not None:
        path = resolve_relative_path(str(path_value), root=REPO_ROOT)
        velocity = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
        if velocity.shape != values.shape:
            raise ValueError("velocity_volume must match the complete LFM volume shape.")
        if np.any(support & (~np.isfinite(velocity) | (velocity <= 0.0))):
            raise ValueError("velocity_volume must be finite and positive on LFM support.")
        return velocity, "velocity_volume"
    if not isinstance(relation_info, Mapping):
        raise ValueError("inputs.ai_velocity_relation must be a unit-explicit relation mapping.")
    relation = AIVelocityRelation.from_mapping(relation_info)
    velocity = np.full(values.shape, np.nan, dtype=np.float32)
    for start in range(0, values.shape[0], 8):
        block = values[start:start + 8]
        block_mask = support[start:start + 8]
        converted = relation.velocity_from_ai(np.exp(block[block_mask].astype(np.float64)))
        if np.any(~np.isfinite(converted)) or np.any(converted <= 0.0):
            raise ValueError("The fixed AI-to-velocity relation produced invalid velocities.")
        velocity[start:start + 8][block_mask] = converted.astype(np.float32)
    return velocity, "fixed_ai_velocity_relation"


def _normalization_from(value: Any) -> Normalization:
    if isinstance(value, Normalization):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload = {name: getattr(value, name) for name in (
            "seismic_mean", "seismic_std", "lfm_mean", "lfm_std", "impedance_mean", "impedance_std"
        )}
    return Normalization(**{name: float(payload[name]) for name in (
        "seismic_mean", "seismic_std", "lfm_mean", "lfm_std", "impedance_mean", "impedance_std"
    )})


def _network_config(section: Mapping[str, Any], *, sample_count: int, wavelet_samples: int) -> NetworkConfig:
    value = section.get("network")
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("network must be a mapping.")
    payload = dict(value or {})
    for key, expected in (("sample_count", sample_count), ("wavelet_samples", wavelet_samples)):
        if key in payload and int(payload[key]) != expected:
            raise ValueError(f"network.{key} must match the supplied data axis ({expected}).")
        payload[key] = expected
    return NetworkConfig.from_mapping(payload)


def _training_config(
    section: Mapping[str, Any],
    *,
    updates: int | None,
    device: str | None,
) -> TrainingConfig:
    raw = section.get("training")
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("training must be a mapping.")
    payload = dict(raw or {})
    if updates is not None:
        payload["updates"] = int(updates)
    if device is not None:
        payload["device"] = str(device)
    return TrainingConfig.from_mapping(payload)


def _inference_config(section: Mapping[str, Any], *, batch_size: int | None) -> InferenceConfig:
    raw = section.get("inference")
    if raw is not None and not isinstance(raw, Mapping):
        raise ValueError("inference must be a mapping.")
    payload = dict(raw or {})
    if batch_size is not None:
        payload["batch_size"] = int(batch_size)
    return InferenceConfig.from_mapping(payload)


def _resolve_runtime(
    config_path: Path,
    raw: Mapping[str, Any],
    *,
    overrides: _Overrides,
    training_config: TrainingConfig,
    normalization: Normalization | None = None,
    expected_axis: Any | None = None,
    kernel_time_s: np.ndarray | None = None,
) -> _Runtime:
    section = _section(raw)
    wavelet_time_s = (
        KernelConfig.from_mapping(section.get("kernel")).time_axis_s()
        if kernel_time_s is None else np.asarray(kernel_time_s, dtype=np.float64).copy()
    )
    workflow = WorkflowConfig.from_mapping(raw)
    lfm_run_dir, well_control_run_dir, variant_id = _resolve_inputs(section, overrides, workflow=workflow)
    lfm = load_lfm_input(
        {
            "lfm_run_dir": str(lfm_run_dir),
            "variant_id": variant_id,
            "well_control_run_dir": str(well_control_run_dir),
        },
        repo_root=REPO_ROOT,
    )
    if np.asarray(lfm.log_ai).ndim != 3:
        raise ValueError("PIAI body inversion requires a three-dimensional LFM variant.")
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    survey_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=survey_options or None)
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    _require_matching_axes(("survey", sample_axis), ("LFM", lfm.sample_axis))
    controls = load_well_control_set(well_control_run_dir, repo_root=REPO_ROOT)
    _require_matching_axes(("survey", sample_axis), ("well controls", controls.sample_axis))
    expected_shape = (
        survey.line_geometry.inline_axis.count,
        survey.line_geometry.xline_axis.count,
        _axis_values(sample_axis).size,
    )
    if tuple(np.asarray(lfm.log_ai).shape) != expected_shape:
        raise ValueError(f"LFM shape {np.asarray(lfm.log_ai).shape} differs from survey geometry {expected_shape}.")
    if not np.array_equal(np.asarray(lfm.ilines), survey.line_geometry.inline_axis.values()):
        raise ValueError("LFM inline axis differs from the survey line geometry.")
    if not np.array_equal(np.asarray(lfm.xlines), survey.line_geometry.xline_axis.values()):
        raise ValueError("LFM xline axis differs from the survey line geometry.")
    if expected_axis is not None:
        _require_matching_axes(("checkpoint", expected_axis), ("survey", sample_axis))

    velocity_mps, velocity_source = _build_velocity(lfm, section, overrides.velocity_volume)
    evaluation_supports = load_evaluation_support_for_run(
        well_control_run_dir, sample_axis=sample_axis, repo_root=REPO_ROOT
    )
    source = SurveyTraceSource(
        survey=survey,
        sample_axis=sample_axis,
        geometry=survey.line_geometry,
    )
    trusted_names = _resolve_trusted_names(section, overrides, controls=controls, workflow=workflow)
    data = prepare_training_data(
        source,
        np.asarray(lfm.log_ai),
        np.asarray(lfm.valid_mask, dtype=bool),
        controls,
        evaluation_supports,
        trusted_names,
        training_config,
        velocity_mps=velocity_mps,
        normalization=normalization,
    )
    runtime_normalization = _normalization_from(data.reader.normalization)
    physics = AcousticPhysics(sample_axis, wavelet_time_s)
    return _Runtime(
        raw=raw,
        section=section,
        workflow=workflow,
        config_path=config_path,
        lfm_run_dir=lfm_run_dir,
        well_control_run_dir=well_control_run_dir,
        variant_id=variant_id,
        lfm=lfm,
        survey=survey,
        sample_axis=sample_axis,
        controls=controls,
        evaluation_supports=evaluation_supports,
        source=source,
        data=data,
        physics=physics,
        wavelet_time_s=wavelet_time_s,
        normalization=runtime_normalization,
        velocity_mps=velocity_mps,
        velocity_source=velocity_source,
        trusted_well_names=trusted_names,
    )


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_output_dir(value: str | Path | None, workflow: WorkflowConfig) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_relative_path(workflow.output_root, root=REPO_ROOT) / f"ginn_v3_piai_{timestamp}"


def _checkpoint_context(runtime: _Runtime, *, config_path: Path) -> dict[str, Any]:
    return {
        "config_path": repo_relative_path(config_path, root=REPO_ROOT),
        "lfm_run_dir": repo_relative_path(runtime.lfm_run_dir, root=REPO_ROOT),
        "variant_id": runtime.variant_id,
        "well_control_run_dir": repo_relative_path(runtime.well_control_run_dir, root=REPO_ROOT),
        "trusted_well_names": list(runtime.trusted_well_names),
        "sample_axis": {
            "domain": str(runtime.sample_axis.domain),
            "unit": str(runtime.sample_axis.unit),
            "values": _axis_values(runtime.sample_axis).tolist(),
        },
        "velocity_source": runtime.velocity_source,
    }


def train_body(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_dir: str | Path | None = None,
    updates: int | None = None,
    lfm_run_dir: str | Path | None = None,
    variant_id: str | None = None,
    well_control_run_dir: str | Path | None = None,
    velocity_volume: str | Path | None = None,
    trusted_well_names: Sequence[str] | None = None,
    device: str | None = None,
) -> TrainingResult:
    """Train the independent PIAI model from explicit workflow artifacts."""

    resolved_config = resolve_relative_path(config_path, root=REPO_ROOT)
    raw = load_config(resolved_config)
    section = _section(raw)
    overrides = _Overrides(
        lfm_run_dir=None if lfm_run_dir is None else Path(lfm_run_dir),
        variant_id=variant_id,
        well_control_run_dir=None if well_control_run_dir is None else Path(well_control_run_dir),
        velocity_volume=None if velocity_volume is None else Path(velocity_volume),
        trusted_well_names=None if trusted_well_names is None else tuple(trusted_well_names),
        device=device,
    )
    workflow = WorkflowConfig.from_mapping(raw)
    training_config = _training_config(section, updates=updates, device=device)
    # Axes are needed to derive both network dimensions, so assemble input data
    # before constructing the model.  The seed is set before model creation.
    runtime = _resolve_runtime(
        resolved_config,
        raw,
        overrides=overrides,
        training_config=training_config,
    )
    network_config = _network_config(
        section,
        sample_count=_axis_values(runtime.sample_axis).size,
        wavelet_samples=runtime.wavelet_time_s.size,
    )
    _seed(training_config.seed)
    model = PIAINetwork(network_config)
    output = _resolve_output_dir(output_dir, workflow)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(output, logger_name="ginn_v3_piai", file_name="training.log")
    write_json(
        output / "input_contract.json",
        {
            "schema": "ginn_v3_piai_input_contract_v2",
            "config_path": repo_relative_path(resolved_config, root=REPO_ROOT),
            "sample_axis": runtime.sample_axis.describe(),
            "network": asdict(network_config),
            "training": asdict(training_config),
            "normalization": asdict(runtime.normalization),
            "trusted_well_names": list(runtime.trusted_well_names),
            "learned_wavelet_time_s": runtime.wavelet_time_s.tolist(),
            "kernel": asdict(KernelConfig.from_mapping(section.get("kernel"))),
            "interface_alignment": "midpoint",
            "velocity_source": runtime.velocity_source,
        },
    )
    trainer = JointTrainer(runtime.data, model, runtime.physics, training_config, logger=logger)
    result = trainer.fit(output, checkpoint_context=_checkpoint_context(runtime, config_path=resolved_config))
    selected_payload = load_checkpoint(result.selected_checkpoint, device="cpu")
    qc_metrics = write_well_qc(
        selected_payload["model"],
        runtime.data,
        runtime.physics,
        np.asarray(selected_payload["mean_wavelet_normalized"], dtype=np.float32),
        output / "well_qc",
        training_config.device,
    )
    write_json(output / "well_qc" / "metrics.json", qc_metrics)
    return result


def _checkpoint_path(value: str | Path) -> Path:
    path = resolve_relative_path(value, root=REPO_ROOT)
    if path.is_dir():
        path = path / "selected_checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_body(
    config_path: str | Path = DEFAULT_CONFIG,
    checkpoint: str | Path | None = None,
    *,
    lfm_run_dir: str | Path | None = None,
    variant_id: str | None = None,
    well_control_run_dir: str | Path | None = None,
    velocity_volume: str | Path | None = None,
    trusted_well_names: Sequence[str] | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> LoadedBody:
    """Load a v3 checkpoint and rebuild its explicit input contracts."""

    if checkpoint is None:
        raise ValueError("load_body requires an explicit v3 checkpoint path or output directory.")
    resolved_config = resolve_relative_path(config_path, root=REPO_ROOT)
    raw = load_config(resolved_config)
    section = _section(raw)
    checkpoint_path = _checkpoint_path(checkpoint)
    inference_mapping = section.get("inference")
    configured_device = (
        str(inference_mapping.get("device"))
        if isinstance(inference_mapping, Mapping) and inference_mapping.get("device")
        else "cpu"
    )
    requested_device = device or configured_device
    payload = load_checkpoint(checkpoint_path, device=requested_device)
    model = payload["model"]
    training_payload = payload.get("training_config") or {}
    if not isinstance(training_payload, Mapping):
        raise ValueError("Checkpoint training_config must be a mapping.")
    training_config = TrainingConfig.from_mapping(training_payload)
    if device is not None and training_config.device != device:
        training_override = dict(training_payload)
        training_override["device"] = str(device)
        training_config = TrainingConfig.from_mapping(training_override)
    normalization = _normalization_from(payload.get("normalization"))
    axis_payload = payload.get("sample_axis")
    if not isinstance(axis_payload, Mapping) or "values" not in axis_payload:
        raise ValueError("Checkpoint lacks sample-axis metadata.")
    from cup.seismic.geometry import SampleAxis

    checkpoint_axis = SampleAxis(
        values=np.asarray(axis_payload["values"], dtype=np.float64),
        domain=str(axis_payload.get("sample_domain", axis_payload.get("domain", ""))),
        unit=str(axis_payload.get("sample_unit", axis_payload.get("unit", ""))),
        depth_basis=axis_payload.get("depth_basis"),
    )
    wavelet_time = np.asarray(payload.get("wavelet_time_s"), dtype=np.float64)
    if wavelet_time.ndim != 1 or wavelet_time.size != model.config.wavelet_samples:
        raise ValueError("Checkpoint wavelet_time_s does not match the model configuration.")
    overrides = _Overrides(
        lfm_run_dir=None if lfm_run_dir is None else Path(lfm_run_dir),
        variant_id=variant_id,
        well_control_run_dir=None if well_control_run_dir is None else Path(well_control_run_dir),
        velocity_volume=None if velocity_volume is None else Path(velocity_volume),
        trusted_well_names=None if trusted_well_names is None else tuple(trusted_well_names),
        device=device,
        batch_size=batch_size,
    )
    runtime = _resolve_runtime(
        resolved_config,
        raw,
        overrides=overrides,
        training_config=training_config,
        normalization=normalization,
        expected_axis=checkpoint_axis,
        kernel_time_s=wavelet_time,
    )
    if model.config.sample_count != _axis_values(runtime.sample_axis).size:
        raise ValueError("Checkpoint model sample_count differs from the seismic SampleAxis.")
    inference_config = _inference_config(section, batch_size=batch_size)
    return LoadedBody(
        checkpoint=checkpoint_path,
        checkpoint_payload=payload,
        workflow=runtime.workflow,
        lfm=runtime.lfm,
        survey=runtime.survey,
        sample_axis=runtime.sample_axis,
        controls=runtime.controls,
        reader=runtime.data.reader,
        data=runtime.data,
        model=model,
        physics=runtime.physics,
        normalization=runtime.normalization,
        inference_config=inference_config,
    )


__all__ = ["LoadedBody", "load_body", "load_config", "train_body"]
