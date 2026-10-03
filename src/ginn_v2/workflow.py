"""Compose GINN v2 training stages behind one workflow interface.

The command requires explicit Step-6, Step-7, and domain-specific wavelet inputs.  A
configuration section named ``ginn_v2_body_inversion`` carries the settings;
the input identities can also be supplied as command-line overrides.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import torch


from cup.config.workflow import WorkflowConfig, load_workflow_config
from cup.lfm.math import parse_lowpass_spec
from cup.physics.relations import AIVelocityRelation
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.forward_inputs import load_forward_inputs as load_seismic_forward_inputs
from cup.seismic.wavelet import load_wavelet_csv, validate_wavelet_normalization
from cup.utils.io import repo_relative_path, resolve_relative_path, write_json
from cup.utils.logging import configure_run_logger
from cup.well.controls import load_well_control_set
from ginn_v2.physics import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.data import PatchKey, PatchReader, SurveyTraceSource, candidate_patch_keys, fit_lfm_normalization
from ginn_v2.train import (
    BodyInversionConfig,
    BodyInversionTrainer,
    build_body_inversion_data,
    load_checkpoint,
)
from cup.lfm.artifacts import load_lfm_input
from ginn_v2.model import CenterTraceBodyNet
from ginn_v2.model import BodySmoother
from ginn_v2.infer import (
    BodyInverter,
    BodyVolumeInverter,
    BodyVolumeResult,
    VolumeInferenceConfig,
    centered_tile_bounds,
)
from ginn_v2.diagnose import write_well_waveform_qc as write_well_waveform_qc_artifact
from ginn_v2.section_store import SectionPredictionStore

REPO_ROOT = Path(__file__).resolve().parents[2]
Stage = str


@dataclass(frozen=True)
class BodyRun:
    """Paths produced by one body-inversion workflow invocation."""

    output_dir: Path
    pretrain_checkpoint: Path
    selected_checkpoint: Path | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LoadedBody:
    """Loaded body model with trace-reader and volume scheduling hidden inside."""

    checkpoint: Path
    checkpoint_payload: Mapping[str, Any]
    lfm_run_dir: Path
    well_control_run_dir: Path
    forward_model_inputs_run_dir: Path | None
    wavelet_generation_run_dir: Path | None
    workflow: WorkflowConfig
    training_config: BodyInversionConfig
    inference_config: Mapping[str, Any]
    lfm: Any
    survey: Any
    sample_axis: Any
    seismic_path: Path
    reader: PatchReader
    model: CenterTraceBodyNet
    adapter: DepthDomainAdapter | TimeDomainAdapter
    lfm_lowpass_spec: Any

    def _inverter(self, batch_size: int | None = None) -> BodyInverter:
        resolved_batch_size = int(
            batch_size
            or self.inference_config.get("batch_size")
            or self.training_config.batch_size
        )
        return BodyInverter(
            self.model,
            self.reader,
            self.adapter,
            smoother=BodySmoother(
                smoothing_fwhm=self.training_config.body_smoothing_fwhm,
            ),
            lfm_lowpass_spec=self.lfm_lowpass_spec,
            device=next(self.model.parameters()).device,
            batch_size=resolved_batch_size,
        )

    def predict_traces(self, keys: tuple[PatchKey, ...], *, batch_size: int | None = None) -> Any:
        """Predict body traces on explicit patch keys."""

        return self._inverter(batch_size).predict_body(keys, center_visible=True)

    def validation_section_keys(
        self,
        orientation: str,
        *,
        max_traces: int = 128,
    ) -> tuple[PatchKey, ...]:
        """Return the longest contiguous section recorded in the checkpoint split."""

        if orientation not in {"inline", "xline"}:
            raise ValueError("orientation must be inline or xline.")
        split = self.checkpoint_payload.get("split")
        if not isinstance(split, Mapping):
            raise ValueError("Checkpoint split description must be a mapping.")
        raw_keys = split.get("review_patch_keys")
        if not isinstance(raw_keys, list):
            raise ValueError("Checkpoint split lacks review_patch_keys.")
        grouped: dict[int, list[PatchKey]] = {}
        for value in raw_keys:
            if not isinstance(value, Mapping) or value.get("orientation") != orientation:
                continue
            key = PatchKey(
                int(value["inline_index"]),
                int(value["xline_index"]),
                orientation,
            )
            fixed = key.inline_index if orientation == "inline" else key.xline_index
            grouped.setdefault(fixed, []).append(key)
        runs: list[tuple[PatchKey, ...]] = []
        for values in grouped.values():
            ordered = sorted(
                values,
                key=(
                    (lambda item: item.xline_index)
                    if orientation == "inline"
                    else (lambda item: item.inline_index)
                ),
            )
            current: list[PatchKey] = []
            previous: int | None = None
            for key in ordered:
                varying = key.xline_index if orientation == "inline" else key.inline_index
                if previous is not None and varying != previous + 1:
                    runs.append(tuple(current))
                    current = []
                current.append(key)
                previous = varying
            if current:
                runs.append(tuple(current))
        if not runs:
            raise ValueError(f"Checkpoint review split has no {orientation} section.")
        selected = max(
            runs,
            key=lambda item: (len(item), -item[0].inline_index, -item[0].xline_index),
        )
        if len(selected) > int(max_traces):
            start = (len(selected) - int(max_traces)) // 2
            selected = selected[start : start + int(max_traces)]
        return selected

    def predict_volume(
        self,
        *,
        batch_size: int | None = None,
        smoke_tile_size: int | None = None,
        smoke_tile_origin: str = "center",
        logger: Any = None,
        section_cache_dir: Path | None = None,
    ) -> BodyVolumeResult:
        """Predict a smoke tile or full survey and fill the complete target zone."""

        resolved_batch_size = int(
            batch_size
            or self.inference_config.get("batch_size")
            or self.training_config.batch_size
        )
        inverter = self._inverter(resolved_batch_size)
        orientations = tuple(
            self.inference_config.get("orientations")
            or self.training_config.orientations
        )
        volume = BodyVolumeInverter(
            inverter,
            VolumeInferenceConfig(
                batch_size=resolved_batch_size,
                log_every_sections=int(self.inference_config.get("log_every_sections") or 10),
                min_lfm_support=int(self.inference_config.get("min_lfm_support") or 8),
                orientations=orientations,
            ),
            logger=logger,
        )
        inline_bounds = None
        xline_bounds = None
        if smoke_tile_size is not None:
            if smoke_tile_origin == "northwest":
                inline_bounds = (0, min(int(smoke_tile_size), int(self.lfm.ilines.size)))
                xline_bounds = (0, min(int(smoke_tile_size), int(self.lfm.xlines.size)))
            elif smoke_tile_origin == "center":
                inline_bounds = centered_tile_bounds(self.lfm.ilines.size, smoke_tile_size)
                xline_bounds = centered_tile_bounds(self.lfm.xlines.size, smoke_tile_size)
            else:
                raise ValueError("smoke_tile_origin must be 'center' or 'northwest'.")
        section_store = None
        if section_cache_dir is not None:
            def file_digest(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                return digest.hexdigest()

            extra_digests = {}
            for name, values in self.reader.domain_extras.items():
                digest = hashlib.sha256()
                digest.update(str(values.dtype).encode())
                digest.update(str(values.shape).encode())
                with np.nditer(values, flags=["external_loop", "buffered"],
                               op_flags=["readonly"], order="C", buffersize=1 << 18) as blocks:
                    for block in blocks:
                        digest.update(block.tobytes())
                extra_digests[name] = digest.hexdigest()
            contract = {
                "schema": "body_raw_sections_v1",
                "checkpoint_sha256": file_digest(self.checkpoint),
                "seismic_sha256": file_digest(self.seismic_path),
                "lfm_sha256": file_digest(self.lfm.variant.lfm_path),
                "domain_extras_sha256": extra_digests,
                "seismic_settings": self.workflow.seismic.as_dict(),
                "training_config": self.training_config.to_json_dict(),
                "batch_size": resolved_batch_size,
                "sample_axis": self.sample_axis.describe(),
                "geometry": asdict(self.survey.line_geometry),
                "inline_bounds": inline_bounds,
                "xline_bounds": xline_bounds,
                "orientations": list(orientations),
                "min_lfm_support": volume.config.min_lfm_support,
                "implementation_sha256": {
                    name: file_digest(Path(__file__).parent / name)
                    for name in ("data.py", "model.py", "infer.py", "workflow.py")
                },
            }
            section_store = SectionPredictionStore(
                section_cache_dir, contract, sample_count=self.sample_axis.values.size,
            )
        return volume.predict(inline_bounds=inline_bounds, xline_bounds=xline_bounds,
                              section_store=section_store)


@dataclass(frozen=True)
class _BodyOptions:
    output_dir: Path | None = None
    lfm_run_dir: Path | None = None
    variant_id: str | None = None
    well_control_run_dir: Path | None = None
    forward_model_inputs_run_dir: Path | None = None
    wavelet_generation_run_dir: Path | None = None


def load_config(path: Path) -> dict[str, Any]:
    return load_workflow_config(path, repo_root=REPO_ROOT)


def _required_input(stage_config: Mapping[str, Any], key: str, override: object) -> str:
    if override is not None:
        value = str(override).strip()
    else:
        inputs = stage_config.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("ginn_v2_body_inversion.inputs must explicitly contain the required input paths.")
        value = str(inputs.get(key) or "").strip()
    if not value:
        raise ValueError(f"ginn_v2_body_inversion input {key!r} must be explicit.")
    return value


def load_forward_inputs(run_dir: Path, *, domain: str, depth_basis: str | None) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation | None, dict[str, Any]]:
    return load_seismic_forward_inputs(run_dir, repo_root=REPO_ROOT, domain=domain, depth_basis=depth_basis)


def _resolve_forward_source(section: Mapping[str, Any], *, domain: str, options: _BodyOptions) -> Path:
    inputs = section.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("ginn_v2_body_inversion.inputs must be a mapping.")
    if domain == "time":
        if inputs.get("forward_model_inputs_run_dir") is not None or options.forward_model_inputs_run_dir is not None:
            raise ValueError("Time body inversion reads wavelet_generation_run_dir, not forward_model_inputs_run_dir.")
        if inputs.get("velocity_volume") is not None:
            raise ValueError("Time body inversion uses TWT coordinates and does not accept velocity_volume.")
        key = "wavelet_generation_run_dir"
        override = options.wavelet_generation_run_dir
    elif domain == "depth":
        if inputs.get("wavelet_generation_run_dir") is not None or options.wavelet_generation_run_dir is not None:
            raise ValueError("Depth body inversion requires forward_model_inputs_run_dir.")
        key = "forward_model_inputs_run_dir"
        override = options.forward_model_inputs_run_dir
    else:
        raise ValueError(f"Unsupported body inversion domain: {domain!r}.")
    return resolve_relative_path(_required_input(section, key, override), root=REPO_ROOT)


def _load_domain_forward_inputs(
    source_dir: Path, *, domain: str, depth_basis: str | None,
) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation | None, dict[str, Any]]:
    if domain == "depth":
        return load_forward_inputs(source_dir, domain=domain, depth_basis=depth_basis)
    if domain != "time":
        raise ValueError(f"Unsupported body inversion domain: {domain!r}.")
    wavelet_path = source_dir / "selected_wavelet.csv"
    time_s, amplitude = load_wavelet_csv(wavelet_path)
    amplitude, qc = validate_wavelet_normalization(time_s, amplitude, allow_small_renormalization=False)
    if qc.status != "ok":
        raise ValueError(f"Selected wavelet failed normalization QC: {qc.reasons}")
    return time_s, amplitude, None, {"wavelet": {"path": repo_relative_path(wavelet_path, root=REPO_ROOT)}}


def _domain_runtime(
    raw: Mapping[str, Any], lfm: Any, sample_axis: Any,
    wavelet_time_s: np.ndarray, wavelet_amplitude: np.ndarray,
    relation: AIVelocityRelation | None,
) -> tuple[DepthDomainAdapter | TimeDomainAdapter, dict[str, np.ndarray]]:
    """Build domain-specific forward inputs shared by training and inference."""

    if sample_axis.domain == "time":
        return TimeDomainAdapter(
            torch.as_tensor(wavelet_time_s, dtype=torch.float32),
            torch.as_tensor(wavelet_amplitude, dtype=torch.float32),
        ), {}
    if sample_axis.domain != "depth":
        raise ValueError(f"Unsupported body inversion domain: {sample_axis.domain!r}.")
    inputs = raw["ginn_v2_body_inversion"].get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("ginn_v2_body_inversion.inputs must be a mapping.")
    velocity_path = inputs.get("velocity_volume")
    if velocity_path is not None:
        velocity = np.load(resolve_relative_path(velocity_path, root=REPO_ROOT), mmap_mode="r", allow_pickle=False)
        if velocity.shape != lfm.log_ai.shape:
            raise ValueError("velocity_volume must match the complete LFM volume shape.")
        if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0.0):
            raise ValueError("velocity_volume must contain finite positive velocities in m/s.")
    elif relation is not None:
        velocity = np.full(lfm.log_ai.shape, np.nan, dtype=np.float64)
        for start in range(0, lfm.log_ai.shape[0], 8):
            valid = lfm.valid_mask[start:start + 8]
            block = velocity[start:start + 8]
            block[valid] = relation.velocity_from_ai(np.exp(lfm.log_ai[start:start + 8][valid].astype(np.float64)))
    else:
        raise ValueError("Depth forward modelling requires velocity_volume or an ai_velocity_relation.")
    adapter = DepthDomainAdapter(
        torch.as_tensor(wavelet_time_s, dtype=torch.float32),
        torch.as_tensor(wavelet_amplitude, dtype=torch.float32),
    )
    return adapter, {"velocity_mps": velocity}


def _require_matching_axes(sample_axis: Any, *other_axes: Any) -> None:
    """Compare coordinate values together with their sampling-domain meaning."""
    for other in other_axes:
        if (
            (sample_axis.domain, sample_axis.unit, sample_axis.depth_basis)
            != (other.domain, other.unit, other.depth_basis)
            or not np.array_equal(sample_axis.values, other.values)
        ):
            raise ValueError("Body workflow SampleAxis domain, unit, depth basis and values must match.")


def _resolve_output_dir(value: Path | None, workflow: WorkflowConfig) -> Path:
    if value is not None:
        return resolve_relative_path(value, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_relative_path(workflow.output_root, root=REPO_ROOT) / f"ginn_v2_body_inversion_{timestamp}"


def _build_runtime(raw: Mapping[str, Any], args: _BodyOptions) -> tuple[WorkflowConfig, BodyInversionConfig, Path, Path, Path, Path, str]:
    workflow = WorkflowConfig.from_mapping(raw)
    section = raw.get("ginn_v2_body_inversion")
    if not isinstance(section, Mapping):
        raise ValueError("Config lacks explicit ginn_v2_body_inversion section.")
    lfm_run_dir = resolve_relative_path(_required_input(section, "lfm_run_dir", args.lfm_run_dir), root=REPO_ROOT)
    variant_id = _required_input(section, "variant_id", args.variant_id)
    well_control_run_dir = resolve_relative_path(
        _required_input(section, "well_control_run_dir", args.well_control_run_dir),
        root=REPO_ROOT,
    )
    forward_source_dir = _resolve_forward_source(section, domain=workflow.seismic.domain, options=args)
    training_mapping = section.get("training")
    if not isinstance(training_mapping, Mapping):
        raise ValueError("ginn_v2_body_inversion.training must be a mapping.")
    config = BodyInversionConfig.from_mapping(training_mapping, sample_domain=workflow.seismic.domain)
    output_dir = _resolve_output_dir(args.output_dir, workflow)
    return workflow, config, lfm_run_dir, well_control_run_dir, forward_source_dir, output_dir, variant_id


def _write_review_package(trainer: BodyInversionTrainer, model: CenterTraceBodyNet, output_dir: Path) -> dict[str, Any]:
    """Render deterministic well profiles and blind validation sections."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    review_dir = output_dir / "review_package"
    profiles_dir = review_dir / "fixed_well_profiles"
    blind_dir = review_dir / "blind_sections"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    blind_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    well_rows: dict[str, dict[int, list[float]]] = {}
    well_baselines: dict[str, dict[int, list[float]]] = {}
    well_targets: dict[str, dict[int, float]] = {}
    well_patch_keys: dict[str, dict[int, Any]] = {}
    with torch.no_grad():
        items = trainer.data.trusted_well_patches
        for start in range(0, len(items), trainer.config.batch_size):
            local = items[start : start + trainer.config.batch_size]
            batch = trainer.data.reader.batch(
                tuple(item.patch_key for item in local),
                center_visible=True,
                device=trainer.device,
            )
            body, _, _, baseline = trainer._predict(model, batch)
            for row, item in enumerate(local):
                for sample_index in np.flatnonzero(item.target_mask):
                    index = int(sample_index)
                    well_rows.setdefault(item.well_name, {}).setdefault(index, []).append(
                        float(body[row, index].cpu())
                    )
                    well_baselines.setdefault(item.well_name, {}).setdefault(index, []).append(
                        float(baseline[row, index].cpu())
                    )
                    well_targets.setdefault(item.well_name, {})[index] = float(item.target_values[index])
                    well_patch_keys.setdefault(item.well_name, {}).setdefault(index, item.patch_key)
    figure, axes = plt.subplots(max(1, len(well_rows)), 1, figsize=(8, max(3, 2.5 * len(well_rows))), squeeze=False)
    axis_values = trainer.data.reader.sample_axis.values
    for row, name in enumerate(sorted(well_rows)):
        indices = sorted(well_rows[name])
        predicted = [np.mean(well_rows[name][index]) for index in indices]
        target = [well_targets[name][index] for index in indices]
        lfm = [
            trainer.data.reader.lfm_log_ai[
                well_patch_keys[name][index].inline_index,
                well_patch_keys[name][index].xline_index,
                index,
            ]
            for index in indices
        ]
        current_axis = axes[row, 0]
        current_axis.plot(target, axis_values[indices], label="well body target")
        current_axis.plot(predicted, axis_values[indices], label="GINN body")
        current_axis.plot(lfm, axis_values[indices], label="LFM")
        current_axis.plot([np.mean(well_baselines[name][index]) for index in indices], axis_values[indices],
                          label="smoothed initial model", linestyle="--")
        current_axis.set_title(name)
        current_axis.set_xlabel("log-AI")
        current_axis.set_ylabel(trainer.data.reader.sample_axis.unit)
        current_axis.invert_yaxis()
        current_axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    well_figure = profiles_dir / "trusted_well_profiles.png"
    figure.savefig(well_figure, dpi=150)
    plt.close(figure)

    section_keys = {
        orientation: trainer.validation_section_keys(orientation)
        for orientation in trainer.config.orientations
    }
    review_keys = tuple(dict.fromkeys(key for keys in section_keys.values() for key in keys))
    trace_eval = trainer._trace_evaluation(model, review_keys, center_visible=True)
    section_files: list[str] = []
    for orientation in trainer.config.orientations:
        keys = section_keys[orientation]
        lateral_m = trainer.lateral_distance_m(keys)
        values = np.stack([trace_eval["bodies"][item] for item in keys])
        baseline_values = np.stack([trace_eval["baselines"][item] for item in keys])
        support = np.stack([trace_eval["supports"][item] for item in keys])
        values = np.where(support, values, np.nan)
        residual = np.where(support, values - baseline_values, np.nan)
        vertical_support = np.any(support, axis=0)
        sample_indices = np.flatnonzero(vertical_support)
        if sample_indices.size < 2:
            raise ValueError("Blind section has fewer than two target-zone samples.")
        sample_start, sample_stop = int(sample_indices[0]), int(sample_indices[-1])
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        extent = [lateral_m[0], lateral_m[-1], axis_values[sample_stop], axis_values[sample_start]]
        axes[0].imshow(values[:, sample_start : sample_stop + 1].T, aspect="auto", origin="upper", extent=extent)
        axes[0].set_title(f"Blind validation section — {orientation}")
        axes[0].set_ylabel(trainer.data.reader.sample_axis.unit)
        axes[1].imshow(residual[:, sample_start : sample_stop + 1].T, aspect="auto", origin="upper", extent=extent, cmap="RdBu_r")
        axes[1].set_title("GINN body minus smoothed initial model")
        axes[1].set_xlabel("lateral distance (m)")
        axes[1].set_ylabel(trainer.data.reader.sample_axis.unit)
        figure.tight_layout()
        path = blind_dir / f"blind_section_{orientation}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        section_files.append(repo_relative_path(path, root=REPO_ROOT))
    well_waveform_qc = _write_well_waveform_qc(trainer, model, output_dir)
    manifest = {
        "fixed_well_profile": repo_relative_path(well_figure, root=REPO_ROOT),
        "blind_sections": section_files,
        "well_waveform_qc": well_waveform_qc,
        "validation_patch_key_count": len(trainer.data.spatial_split.review_keys),
        "validation_centers": [list(item) for item in trainer.data.spatial_split.validation_centers],
    }
    write_json(review_dir / "review_manifest.json", manifest)
    return manifest


def _write_well_waveform_qc(
    trainer: BodyInversionTrainer,
    model: CenterTraceBodyNet,
    output_dir: Path,
) -> dict[str, Any]:
    return write_well_waveform_qc_artifact(
        trainer,
        model,
        output_dir / "review_package" / "well_waveform_qc",
        root=REPO_ROOT,
    )


def train_body(
    config_path: str | Path = "experiments/ginn_v2/ginn_v2.yaml",
    *,
    stage: Stage = "all",
    pretrain_checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    lfm_run_dir: str | Path | None = None,
    variant_id: str | None = None,
    well_control_run_dir: str | Path | None = None,
    forward_model_inputs_run_dir: str | Path | None = None,
    wavelet_generation_run_dir: str | Path | None = None,
) -> BodyRun:
    """Run reusable self-supervised pretraining, semi-supervised finetuning, or both."""

    if stage not in {"all", "pretrain", "finetune"}:
        raise ValueError("stage must be 'all', 'pretrain', or 'finetune'.")
    if stage == "finetune" and pretrain_checkpoint is None:
        raise ValueError("finetune requires an explicit pretrain_checkpoint.")
    args = _BodyOptions(
        output_dir=None if output_dir is None else Path(output_dir),
        lfm_run_dir=None if lfm_run_dir is None else Path(lfm_run_dir),
        variant_id=variant_id,
        well_control_run_dir=None if well_control_run_dir is None else Path(well_control_run_dir),
        forward_model_inputs_run_dir=None if forward_model_inputs_run_dir is None else Path(forward_model_inputs_run_dir),
        wavelet_generation_run_dir=None if wavelet_generation_run_dir is None else Path(wavelet_generation_run_dir),
    )
    config_path = resolve_relative_path(config_path, root=REPO_ROOT)
    raw = load_config(config_path)
    workflow, config, lfm_run_dir, well_control_run_dir, forward_source_dir, output_dir, variant_id = _build_runtime(raw, args)
    if output_dir.exists():
        raise FileExistsError(f"Body-inversion output directory already exists: {output_dir}; use a new output directory.")
    else:
        output_dir.mkdir(parents=True)
    logger = configure_run_logger(
        output_dir,
        logger_name="ginn_v2_body_inversion",
        file_name="training.log",
    )
    logger.info("body inversion start | config=%s | output=%s", config_path, output_dir)

    controls = load_well_control_set(well_control_run_dir, repo_root=REPO_ROOT)
    lfm = load_lfm_input(
        {
            "lfm_run_dir": str(lfm_run_dir),
            "variant_id": variant_id,
            "well_control_run_dir": str(well_control_run_dir),
        },
        repo_root=REPO_ROOT,
    )
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    survey_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=survey_options or None)
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    _require_matching_axes(sample_axis, lfm.sample_axis, controls.sample_axis)
    if lfm.log_ai.ndim != 3:
        raise ValueError("Body inversion requires a volume LFM variant, not a section variant.")
    if lfm.log_ai.shape != (survey.line_geometry.inline_axis.count, survey.line_geometry.xline_axis.count, sample_axis.values.size):
        raise ValueError("LFM volume shape differs from current survey geometry.")
    target_zone_mask = np.asarray(lfm.valid_mask, dtype=bool)
    baseline_config = dict(lfm.variant.variant_metadata.get("resolved_baseline_config") or {})
    lfm_lowpass_spec = parse_lowpass_spec(
        dict(baseline_config.get("filter") or {}),
        sample_axis,
    )
    wavelet_time_s, wavelet_amplitude, relation, forward_payload = _load_domain_forward_inputs(
        forward_source_dir,
        domain=workflow.seismic.domain,
        depth_basis=workflow.seismic.depth_basis,
    )
    adapter, domain_extras = _domain_runtime(
        raw, lfm, sample_axis, wavelet_time_s, wavelet_amplitude, relation,
    )
    normalization = fit_lfm_normalization(lfm.log_ai, lfm.valid_mask, geometry=survey.line_geometry)
    source = SurveyTraceSource(survey=survey, sample_axis=sample_axis, geometry=survey.line_geometry)
    reader = PatchReader(
        source,
        lfm_log_ai=lfm.log_ai,
        lfm_valid_mask=lfm.valid_mask,
        ilines=lfm.ilines,
        xlines=lfm.xlines,
        sample_axis=sample_axis,
        normalization=normalization,
        patch_radius=config.patch_radius,
        domain_extras=domain_extras,
        cache_size=config.cache_size,
        seismic_feature_mode=config.seismic_feature_mode,
        seismic_balance_window_samples=config.seismic_balance_window_samples,
        seismic_balance_floor_fraction=config.seismic_balance_floor_fraction,
    )
    candidates = candidate_patch_keys(
        lfm.log_ai,
        lfm.valid_mask,
        patch_radius=config.patch_radius,
        orientations=config.orientations,
    )
    data = build_body_inversion_data(
        reader,
        controls,
        config=config,
        lfm_lowpass_spec=lfm_lowpass_spec,
        candidate_keys=candidates,
        target_zone_mask=target_zone_mask,
    )
    logger.info(
        "data ready | train_patches=%d | validation_patches=%d | trusted_well_samples=%d",
        len(data.spatial_split.train_keys),
        len(data.spatial_split.validation_keys),
        sum(int(np.count_nonzero(item.target_mask)) for item in data.trusted_well_patches),
    )
    trainer = BodyInversionTrainer(
        data,
        adapter=adapter,
        config=config,
        lfm_lowpass_spec=lfm_lowpass_spec,
        output_dir=output_dir,
        artifact_root=REPO_ROOT,
        logger=logger,
    )
    logger.info("LFM-only baseline evaluation start")
    baseline = trainer.evaluate(None)
    logger.info(
        "LFM-only baseline ready | masked_corr=%.4f | visible_corr=%.4f | well_rmse=%.5f",
        float(np.median(baseline.masked_correlation)),
        float(np.median(baseline.visible_correlation)),
        baseline.well_pooled_rmse,
    )
    all_well_names = [control.well_name for control in controls.controls]
    trusted_well_names = list(config.trusted_well_names)
    trusted_lookup = {name.casefold() for name in trusted_well_names}
    diagnostic_well_names = [name for name in all_well_names if name.casefold() not in trusted_lookup]
    write_json(output_dir / "baseline_metrics.json", baseline.to_json_dict())
    write_json(output_dir / "split.json", data.split_description())
    write_json(
        output_dir / "well_roles.json",
        {
            "lfm_well_names": all_well_names,
            "body_well_names": trusted_well_names,
            "diagnostic_only_well_names": diagnostic_well_names,
        },
    )
    write_json(
        output_dir / "input_contract.json",
        {
            "sample_axis": sample_axis.describe(),
            "depth_basis": workflow.seismic.depth_basis,
            f"body_smoothing_fwhm_{config.sample_unit}": config.body_smoothing_fwhm,
            "prediction_definition": "b0 + (I - L)(gaussian_smooth(initial + correction) - b0); b0 = gaussian_smooth(initial)",
            "lfm_anchor_weight": config.loss_weights.lfm_anchor,
            "low_frequency_baseline": "gaussian_smooth(initial)",
            "well_target_definition": "native_filtered_log_ai resampled then gaussian_smooth once",
            "seismic_objective": "normalized waveform shape",
            "seismic_feature": {
                "mode": config.seismic_feature_mode,
                "balance_window_samples": config.seismic_balance_window_samples,
                "balance_floor_fraction": config.seismic_balance_floor_fraction,
            },
            "lfm_input_normalization": {
                "lfm_mean": normalization.lfm_mean,
                "lfm_scale": normalization.lfm_scale,
                "geometry_scale_m": normalization.geometry_scale_m,
            },
            "lfm_run_dir": repo_relative_path(lfm_run_dir, root=REPO_ROOT),
            "lfm_variant_id": variant_id,
            "lfm_artifact_lowpass": {
                "value": lfm.lowpass_value,
                "unit": lfm.lowpass_unit,
                "order": lfm_lowpass_spec.order,
                "buffer_mode": lfm_lowpass_spec.buffer_mode,
                "buffer_axis_units": lfm_lowpass_spec.buffer_axis_units,
            },
            "well_control_run_dir": repo_relative_path(well_control_run_dir, root=REPO_ROOT),
            ("wavelet_generation_run_dir" if workflow.seismic.domain == "time" else "forward_model_inputs_run_dir"): repo_relative_path(forward_source_dir, root=REPO_ROOT),
            "forward_adapter": getattr(adapter, "adapter_id", type(adapter).__name__),
            "wavelet": forward_payload.get("wavelet"),
            "well_roles": {
                "lfm_well_names": all_well_names,
                "body_well_names": trusted_well_names,
                "diagnostic_only_well_names": diagnostic_well_names,
            },
        },
    )

    warnings: list[str] = []
    if stage in {"all", "pretrain"}:
        logger.info("shared masked pretraining start")
        shared_pretrain_checkpoint, pretrain_metrics = trainer.run_pretraining()
        write_json(output_dir / "pretrain_metrics.json", pretrain_metrics.to_json_dict())
        logger.info(
            "shared masked pretraining ready | checkpoint=%s | masked_corr=%.4f | well_rmse=%.5f",
            shared_pretrain_checkpoint,
            float(np.median(pretrain_metrics.masked_correlation)),
            pretrain_metrics.well_pooled_rmse,
        )
        if float(np.median(pretrain_metrics.masked_correlation)) < (
            float(np.median(baseline.masked_correlation)) + config.warnings.pretrain_masked_corr_improvement
        ):
            warnings.append("pretrain_masked_correlation_below_reference")
        if float(np.median(pretrain_metrics.masked_shape_loss)) > (
            config.warnings.pretrain_masked_shape_ratio * float(np.median(baseline.masked_shape_loss))
        ):
            warnings.append("pretrain_masked_shape_above_reference")
        if stage == "pretrain":
            write_json(
                output_dir / "body_inversion_status.json",
                {
                    "status": "completed_with_warnings" if warnings else "completed",
                    "stage": "pretrain",
                    "warnings": warnings,
                    "checkpoint": repo_relative_path(shared_pretrain_checkpoint, root=REPO_ROOT),
                },
            )
            return BodyRun(output_dir, shared_pretrain_checkpoint, None, tuple(warnings))
    else:
        shared_pretrain_checkpoint = resolve_relative_path(pretrain_checkpoint, root=REPO_ROOT)
        if not shared_pretrain_checkpoint.is_file():
            raise FileNotFoundError(shared_pretrain_checkpoint)

    logger.info("single semi-supervised finetune start")
    selected = trainer.run_finetuning(
        baseline=baseline,
        resume_checkpoint=shared_pretrain_checkpoint,
    )
    write_json(output_dir / "finetune_result.json", selected.to_json_dict())
    selected_path = output_dir / "selected_checkpoint.pt"
    shutil.copyfile(selected.selected_checkpoint, selected_path)
    warnings.extend(f"finetune_{name}" for name in selected.warnings.warnings)
    write_json(
        output_dir / "selected_checkpoint.json",
        {
            "status": "completed_with_warnings" if warnings else "completed",
            "warnings": warnings,
            "selected_checkpoint": repo_relative_path(selected_path, root=REPO_ROOT),
            "epoch": selected.selected_epoch,
            "selection_rule": "best_recorded_finetune_checkpoint",
            "quality_warnings": selected.warnings.to_json_dict(),
        },
    )
    selected_model = CenterTraceBodyNet(config.network).to(trainer.device)
    load_checkpoint(
        selected_path,
        model=selected_model,
        expected_network_config=config.network,
        map_location=trainer.device,
    )
    review = _write_review_package(trainer, selected_model, output_dir)
    write_json(
        output_dir / "body_inversion_status.json",
        {
            "status": "completed_with_warnings" if warnings else "completed",
            "stage": stage,
            "warnings": warnings,
            "selected_checkpoint": repo_relative_path(selected_path, root=REPO_ROOT),
            "review_package": review,
            "finetune_epochs": config.finetune_epochs,
            "quality_warnings": selected.warnings.to_json_dict(),
        },
    )
    return BodyRun(output_dir, shared_pretrain_checkpoint, selected_path, tuple(warnings))


def load_body(
    config_path: str | Path = "experiments/ginn_v2/ginn_v2.yaml",
    *,
    checkpoint: str | Path | None = None,
    lfm_run_dir: str | Path | None = None,
    variant_id: str | None = None,
    well_control_run_dir: str | Path | None = None,
    forward_model_inputs_run_dir: str | Path | None = None,
    wavelet_generation_run_dir: str | Path | None = None,
    batch_size: int | None = None,
) -> LoadedBody:
    """Load a trained body facade without exposing its internal assembly."""

    config_path = resolve_relative_path(config_path, root=REPO_ROOT)
    raw = load_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    section = raw.get("ginn_v2_body_inversion")
    if not isinstance(section, Mapping):
        raise ValueError("Config lacks explicit ginn_v2_body_inversion section.")
    inputs = section.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("ginn_v2_body_inversion.inputs must be a mapping.")
    training = section.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("ginn_v2_body_inversion.training must be a mapping.")
    config = BodyInversionConfig.from_mapping(training, sample_domain=workflow.seismic.domain)
    inference = raw.get("ginn_v2_volume_inference")
    if not isinstance(inference, Mapping):
        raise ValueError("ginn_v2_volume_inference must be a mapping.")
    checkpoint_path = resolve_relative_path(
        checkpoint if checkpoint is not None else str(inference.get("checkpoint") or ""),
        root=REPO_ROOT,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    lfm_path = resolve_relative_path(_required_input(section, "lfm_run_dir", lfm_run_dir), root=REPO_ROOT)
    well_path = resolve_relative_path(
        _required_input(section, "well_control_run_dir", well_control_run_dir),
        root=REPO_ROOT,
    )
    forward_source_dir = _resolve_forward_source(
        section,
        domain=workflow.seismic.domain,
        options=_BodyOptions(
            forward_model_inputs_run_dir=None if forward_model_inputs_run_dir is None else Path(forward_model_inputs_run_dir),
            wavelet_generation_run_dir=None if wavelet_generation_run_dir is None else Path(wavelet_generation_run_dir),
        ),
    )
    selected_variant = _required_input(section, "variant_id", variant_id)
    lfm = load_lfm_input(
        {
            "lfm_run_dir": str(lfm_path),
            "variant_id": selected_variant,
            "well_control_run_dir": str(well_path),
        },
        repo_root=REPO_ROOT,
    )
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey_options = (
        segy_options_from_config(workflow.seismic.as_dict())
        if workflow.seismic.type == "segy"
        else {}
    )
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options=survey_options or None,
    )
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    _require_matching_axes(sample_axis, lfm.sample_axis)
    baseline = dict(lfm.variant.variant_metadata.get("resolved_baseline_config") or {})
    lowpass = parse_lowpass_spec(dict(baseline.get("filter") or {}), sample_axis)
    wavelet_time_s, wavelet_amplitude, relation, _payload = _load_domain_forward_inputs(
        forward_source_dir,
        domain=workflow.seismic.domain,
        depth_basis=workflow.seismic.depth_basis,
    )
    adapter, domain_extras = _domain_runtime(
        raw, lfm, sample_axis, wavelet_time_s, wavelet_amplitude, relation,
    )
    normalization = fit_lfm_normalization(
        lfm.log_ai,
        lfm.valid_mask,
        geometry=survey.line_geometry,
    )
    reader = PatchReader(
        SurveyTraceSource(survey=survey, sample_axis=sample_axis, geometry=survey.line_geometry),
        lfm_log_ai=lfm.log_ai,
        lfm_valid_mask=lfm.valid_mask,
        ilines=lfm.ilines,
        xlines=lfm.xlines,
        sample_axis=sample_axis,
        normalization=normalization,
        patch_radius=config.patch_radius,
        cache_size=max(config.cache_size, 4 * config.patch_radius + 2),
        domain_extras=domain_extras,
        seismic_feature_mode=config.seismic_feature_mode,
        seismic_balance_window_samples=config.seismic_balance_window_samples,
        seismic_balance_floor_fraction=config.seismic_balance_floor_fraction,
    )
    device = torch.device(str(inference.get("device") or config.device))
    model = CenterTraceBodyNet(config.network).to(device)
    payload = load_checkpoint(
        checkpoint_path,
        model=model,
        expected_network_config=config.network,
        map_location=device,
    )
    if dict(payload["run_config"]) != config.to_json_dict():
        raise ValueError("Checkpoint model semantics differ from the current GINN configuration.")
    resolved_inference = dict(inference)
    if batch_size is not None:
        resolved_inference["batch_size"] = int(batch_size)
    return LoadedBody(
        checkpoint=checkpoint_path,
        checkpoint_payload=payload,
        lfm_run_dir=lfm_path,
        well_control_run_dir=well_path,
        forward_model_inputs_run_dir=forward_source_dir if workflow.seismic.domain == "depth" else None,
        wavelet_generation_run_dir=forward_source_dir if workflow.seismic.domain == "time" else None,
        workflow=workflow,
        training_config=config,
        inference_config=resolved_inference,
        lfm=lfm,
        survey=survey,
        sample_axis=sample_axis,
        seismic_path=seismic_path,
        reader=reader,
        model=model,
        adapter=adapter,
        lfm_lowpass_spec=lowpass,
    )


def pretrain_body(
    config_path: str | Path = "experiments/ginn_v2/ginn_v2.yaml",
    **kwargs: Any,
) -> BodyRun:
    """Create a reusable self-supervised body checkpoint."""

    return train_body(config_path, stage="pretrain", **kwargs)


def finetune_body(
    config_path: str | Path = "experiments/ginn_v2/ginn_v2.yaml",
    *,
    pretrained: BodyRun | str | Path,
    **kwargs: Any,
) -> BodyRun:
    """Apply well supervision to a reusable self-supervised checkpoint."""

    checkpoint = pretrained.pretrain_checkpoint if isinstance(pretrained, BodyRun) else pretrained
    return train_body(
        config_path,
        stage="finetune",
        pretrain_checkpoint=checkpoint,
        **kwargs,
    )


__all__ = [
    "BodyRun",
    "LoadedBody",
    "finetune_body",
    "load_config",
    "load_body",
    "load_forward_inputs",
    "pretrain_body",
    "train_body",
]
