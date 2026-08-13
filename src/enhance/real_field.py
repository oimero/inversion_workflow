"""Real-field residual library and paired samples for Enhance V2.

This module owns the stage-2 data seam.  It reads the published Step-6/Step-7
contracts, runs a frozen GINN body checkpoint through the existing forward
adapter, and returns simple tensors to the training module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from cup.config.workflow import WorkflowConfig, deep_merge_dict
from cup.lfm.artifacts import load_lfm_input
from cup.lfm.math import parse_lowpass_spec
from cup.physics.calibration import AIVelocityRelation
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, write_json
from cup.utils.masks import true_runs
from cup.well.real_field_controls import WellControlSet, load_well_control_set
from ginn_v2.adapters import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.checkpoint import load_checkpoint
from ginn_v2.contracts import CommonObservationBatch
from ginn_v2.data import PatchReader, SurveyTraceSource, fit_lfm_normalization
from ginn_v2.model import BodyNetworkConfig, CenterTraceBodyNet
from ginn_v2.projector import BodyScaleProjector
from ginn_v2.split import build_well_body_target, well_target_zone_mask

from enhance.config import EnhanceV2Config
from enhance.projector import ResidualScaleProjector


@dataclass(frozen=True)
class ResidualRecord:
    well_name: str
    native_axis: np.ndarray
    native_reference: np.ndarray
    native_body: np.ndarray
    native_residual: np.ndarray
    model_axis: np.ndarray
    model_reference: np.ndarray
    model_body: np.ndarray
    model_residual: np.ndarray
    model_mask: np.ndarray
    inline: float
    xline: float
    x_m: float
    y_m: float


@dataclass(frozen=True)
class ResidualLibrary:
    records: tuple[ResidualRecord, ...]
    sample_axis: np.ndarray
    sample_domain: str
    sample_unit: str
    metadata: dict[str, Any]

    @property
    def well_names(self) -> tuple[str, ...]:
        return tuple(item.well_name for item in self.records)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        max_native = max(item.native_axis.size for item in self.records)
        max_model = self.sample_axis.size
        native_axis = np.full((len(self.records), max_native), np.nan, dtype=np.float32)
        native_reference = np.full_like(native_axis, np.nan)
        native_body = np.full_like(native_axis, np.nan)
        native_residual = np.full_like(native_axis, np.nan)
        native_mask = np.zeros_like(native_axis, dtype=bool)
        names = []
        model_reference = []
        model_body = []
        model_residual = []
        model_mask = []
        positions = []
        for row, item in enumerate(self.records):
            count = item.native_axis.size
            native_axis[row, :count] = item.native_axis
            native_reference[row, :count] = item.native_reference
            native_body[row, :count] = item.native_body
            native_residual[row, :count] = item.native_residual
            native_mask[row, :count] = np.isfinite(item.native_residual)
            names.append(item.well_name)
            model_reference.append(item.model_reference)
            model_body.append(item.model_body)
            model_residual.append(item.model_residual)
            model_mask.append(item.model_mask)
            positions.append((item.inline, item.xline, item.x_m, item.y_m))
        np.savez_compressed(
            path,
            sample_axis=self.sample_axis.astype(np.float64),
            sample_domain=np.asarray(self.sample_domain),
            sample_unit=np.asarray(self.sample_unit),
            well_names=np.asarray(names),
            inline=np.asarray(positions, dtype=np.float64)[:, 0],
            xline=np.asarray(positions, dtype=np.float64)[:, 1],
            x_m=np.asarray(positions, dtype=np.float64)[:, 2],
            y_m=np.asarray(positions, dtype=np.float64)[:, 3],
            model_reference=np.asarray(model_reference, dtype=np.float32),
            model_body=np.asarray(model_body, dtype=np.float32),
            model_residual=np.asarray(model_residual, dtype=np.float32),
            model_mask=np.asarray(model_mask, dtype=bool),
            native_axis=native_axis,
            native_reference=native_reference,
            native_body=native_body,
            native_residual=native_residual,
            native_mask=native_mask,
            metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        )


@dataclass(frozen=True)
class RealFieldRuntime:
    config: EnhanceV2Config
    workflow: WorkflowConfig
    controls: WellControlSet
    lfm: Any
    survey: Any
    reader: PatchReader
    adapter: Any
    lowpass_spec: Any
    projector: BodyScaleProjector
    residual_projector: ResidualScaleProjector
    model: CenterTraceBodyNet
    body_volume: np.ndarray
    seismic_volume: np.ndarray
    velocity_volume: np.ndarray | None
    body_cache: dict[tuple[int, int, str], np.ndarray]
    seismic_cache: dict[tuple[int, int], np.ndarray]


def load_runtime(config: EnhanceV2Config, *, repo_root: Path) -> RealFieldRuntime:
    experiment = load_yaml_config(config.ginn_config)
    workflow_name = str(experiment.get("workflow_config") or config.workflow_config)
    common = load_yaml_config(resolve_relative_path(workflow_name, root=repo_root))
    raw = deep_merge_dict(common, {key: value for key, value in experiment.items() if key != "workflow_config"})
    workflow = WorkflowConfig.from_mapping(raw)
    controls = load_well_control_set(config.well_control_run_dir, repo_root=repo_root)
    lfm = load_lfm_input(
        {
            "lfm_run_dir": str(config.lfm_run_dir),
            "variant_id": config.lfm_variant_id,
            "well_control_run_dir": str(config.well_control_run_dir),
        },
        repo_root=repo_root,
    )
    if lfm.log_ai.ndim != 3:
        raise ValueError("Enhance V2 requires a volume LFM.")
    seismic_path = resolve_relative_path(workflow.seismic.file, root=resolve_relative_path(workflow.data_root, root=repo_root))
    survey_options = segy_options_from_config(workflow.seismic.as_dict()) if workflow.seismic.type == "segy" else {}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=survey_options or None)
    sample_axis = survey.sample_axis(workflow.seismic.domain)
    if not np.array_equal(sample_axis.values, lfm.sample_axis.values) or not np.array_equal(sample_axis.values, controls.sample_axis.values):
        raise ValueError("Enhance V2 seismic, LFM, and well controls must share one SampleAxis.")
    baseline = dict(lfm.variant.variant_metadata.get("resolved_baseline_config") or {})
    lowpass_spec = parse_lowpass_spec(dict(baseline.get("filter") or {}), sample_axis)
    wavelet_payload = json.loads(config.forward_model_inputs.read_text(encoding="utf-8"))
    wavelet_info = dict(wavelet_payload.get("wavelet") or {})
    wavelet_path = resolve_relative_path(str(wavelet_info.get("path") or ""), root=repo_root)
    from cup.seismic.wavelet import load_wavelet_csv, validate_wavelet_normalization

    wavelet_time, wavelet_amp = load_wavelet_csv(wavelet_path)
    wavelet_amp, wavelet_qc = validate_wavelet_normalization(wavelet_time, wavelet_amp, allow_small_renormalization=False)
    if wavelet_qc.status != "ok":
        raise ValueError(f"Frozen wavelet failed normalization: {wavelet_qc.reasons}")
    relation = None
    domain_extras: dict[str, np.ndarray] = {}
    if workflow.seismic.domain == "depth":
        relation_info = wavelet_payload.get("ai_velocity_relation")
        if not isinstance(relation_info, dict):
            raise ValueError("Depth Enhance V2 requires ai_velocity_relation in forward inputs.")
        relation = AIVelocityRelation.from_mapping(relation_info)
        velocity = np.full(lfm.log_ai.shape, np.nan, dtype=np.float64)
        valid = lfm.valid_mask
        velocity[valid] = relation.velocity_from_ai(np.exp(lfm.log_ai[valid]))
        domain_extras["velocity_mps"] = velocity
        adapter = DepthDomainAdapter(torch.from_numpy(wavelet_time).float(), torch.from_numpy(wavelet_amp).float())
    else:
        adapter = TimeDomainAdapter(torch.from_numpy(wavelet_time).float(), torch.from_numpy(wavelet_amp).float())
    normalization = fit_lfm_normalization(lfm.log_ai, lfm.valid_mask, geometry=survey.line_geometry)
    reader = PatchReader(
        SurveyTraceSource(survey=survey, sample_axis=sample_axis, geometry=survey.line_geometry),
        lfm_log_ai=lfm.log_ai,
        lfm_valid_mask=lfm.valid_mask,
        ilines=lfm.ilines,
        xlines=lfm.xlines,
        sample_axis=sample_axis,
        normalization=normalization,
        patch_radius=config.patch_radius,
        domain_extras=domain_extras,
        cache_size=64,
    )
    checkpoint_payload = torch.load(config.ginn_checkpoint, map_location="cpu", weights_only=False)
    network_config = BodyNetworkConfig(**dict(checkpoint_payload["network_config"]))
    model = CenterTraceBodyNet(network_config)
    load_checkpoint(config.ginn_checkpoint, model=model, expected_network_config=network_config, map_location="cpu")
    model.eval()
    projector = BodyScaleProjector(
        smoothing_fwhm_m=config.body_smoothing_fwhm_m,
        sample_step=float(sample_axis.step),
        lowpass_spec=lowpass_spec,
    )
    return RealFieldRuntime(
        config=config,
        workflow=workflow,
        controls=controls,
        lfm=lfm,
        survey=survey,
        reader=reader,
        adapter=adapter,
        lowpass_spec=lowpass_spec,
        projector=projector,
        residual_projector=ResidualScaleProjector(smoothing_fwhm_m=config.body_smoothing_fwhm_m),
        model=model,
        body_volume=np.empty((0,), dtype=np.float32),
        seismic_volume=np.empty((0,), dtype=np.float32),
        velocity_volume=domain_extras.get("velocity_mps"),
        body_cache={},
        seismic_cache={},
    )


def build_residual_library(runtime: RealFieldRuntime) -> ResidualLibrary:
    controls = {item.well_name: item for item in runtime.controls.controls}
    records: list[ResidualRecord] = []
    target_mask = np.asarray(runtime.lfm.valid_mask, dtype=bool)
    for name in runtime.config.trusted_well_names:
        control = controls.get(name)
        if control is None:
            raise ValueError(f"Trusted Enhance well is absent: {name}")
        target = build_well_body_target(
            control,
            body_smoothing_fwhm_m=runtime.config.body_smoothing_fwhm_m,
            target_zone_support=well_target_zone_mask(control, geometry=runtime.reader.geometry, target_zone_mask=target_mask),
            lfm_log_ai=runtime.lfm.log_ai,
            lfm_valid_mask=runtime.lfm.valid_mask,
            geometry=runtime.reader.geometry,
            projector=runtime.projector,
        )
        native_values = np.asarray(control.native.full_log_ai, dtype=np.float64)
        native_axis = np.asarray(control.native.coordinates, dtype=np.float64)
        native_mask = np.isfinite(native_values)
        from cup.well.scale_separation import gaussian_smooth_finite_runs_numpy

        native_body = gaussian_smooth_finite_runs_numpy(native_values, native_axis, fwhm_m=runtime.config.body_smoothing_fwhm_m)
        native_residual = native_values - native_body
        model_body = np.full(runtime.lfm.sample_axis.values.shape, np.nan, dtype=np.float64)
        model_body[target.valid_target_mask] = target.model_axis_target[target.valid_target_mask]
        model_reference = np.asarray(control.log_ai.values, dtype=np.float64)
        model_mask = np.asarray(control.observed_valid_mask & target.valid_target_mask, dtype=bool)
        model_residual = _resample_native_residual(
            native_axis,
            native_residual,
            runtime.lfm.sample_axis.values,
        )
        model_residual[~model_mask] = np.nan
        valid = np.flatnonzero(model_mask)
        if valid.size < 2:
            raise ValueError(f"{name}: residual library has fewer than two model samples.")
        records.append(
            ResidualRecord(
                well_name=name,
                native_axis=native_axis,
                native_reference=np.where(native_mask, native_values, np.nan),
                native_body=native_body,
                native_residual=native_residual,
                model_axis=runtime.lfm.sample_axis.values.copy(),
                model_reference=model_reference,
                model_body=model_body,
                model_residual=model_residual,
                model_mask=model_mask,
                inline=float(control.inline_by_sample[valid[0]]),
                xline=float(control.xline_by_sample[valid[0]]),
                x_m=float(control.x_m_by_sample[valid[0]]),
                y_m=float(control.y_m_by_sample[valid[0]]),
            )
        )
    return ResidualLibrary(
        records=tuple(records),
        sample_axis=runtime.lfm.sample_axis.values.copy(),
        sample_domain=runtime.lfm.sample_axis.domain,
        sample_unit=runtime.lfm.sample_axis.unit,
        metadata={
            "schema_version": "enhance_v2_residual_library_v1",
            "body_smoothing_fwhm_m": runtime.config.body_smoothing_fwhm_m,
            "well_names": list(runtime.config.trusted_well_names),
            "sample_domain": runtime.lfm.sample_axis.domain,
            "sample_unit": runtime.lfm.sample_axis.unit,
            "lfm_variant_id": runtime.config.lfm_variant_id,
        },
    )


def _resample_native_residual(native_axis: np.ndarray, native_residual: np.ndarray, model_axis: np.ndarray) -> np.ndarray:
    result = np.full(model_axis.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(native_residual)
    for start, stop in _finite_runs(finite):
        inside = (model_axis >= native_axis[start]) & (model_axis <= native_axis[stop - 1])
        result[inside] = np.interp(model_axis[inside], native_axis[start:stop], native_residual[start:stop])
    return result


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    return [(int(start), int(stop)) for start, stop in np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2) if stop > start]


def infer_body_trace(runtime: RealFieldRuntime, i: int, j: int, orientation: str = "inline") -> np.ndarray:
    from ginn_v2.data import PatchKey

    cache_key = (int(i), int(j), str(orientation))
    if cache_key in runtime.body_cache:
        return runtime.body_cache[cache_key].copy()
    batch = runtime.reader.batch((PatchKey(int(i), int(j), orientation),), center_visible=True, device="cpu")
    common = CommonObservationBatch(
        sample_axis=runtime.reader.sample_axis,
        observed_seismic=batch.observed_seismic,
        observed_valid_mask=batch.observed_valid_mask,
        lfm_log_ai=batch.lfm_log_ai,
        lfm_valid_mask=batch.lfm_valid_mask,
        xy_m=batch.xy_m,
        domain_extras=batch.domain_extras,
    )
    with torch.no_grad():
        raw = runtime.model(batch.features, center_index=runtime.reader.patch_radius)
        coords = runtime.adapter.vertical_coordinates_m(common)
        correction = runtime.projector.project(raw, coords, batch.lfm_valid_mask)
        body = (batch.lfm_log_ai + correction)[0].numpy().astype(np.float32)
    runtime.body_cache[cache_key] = body.copy()
    return body


def read_seismic_trace(runtime: RealFieldRuntime, i: int, j: int) -> np.ndarray:
    key = (int(i), int(j))
    if key not in runtime.seismic_cache:
        from ginn_v2.data import PatchKey

        sample = runtime.reader.read(PatchKey(int(i), int(j), "inline"), center_visible=True)
        runtime.seismic_cache[key] = sample.observed_seismic.copy()
    return runtime.seismic_cache[key].copy()


class PairedResidualDataset(Dataset):
    """Deterministic paired seismic/residual samples from frozen real-field inputs."""

    def __init__(self, runtime: RealFieldRuntime, library: ResidualLibrary, *, count: int, seed: int) -> None:
        self.runtime = runtime
        self.library = library
        self.count = int(count)
        self.seed = int(seed)
        self.radius = runtime.config.patch_radius
        self.keys = self._candidate_keys()
        if not self.keys:
            raise ValueError("No valid Enhance V2 center patches are available.")

    def _candidate_keys(self) -> tuple[tuple[int, int, str], ...]:
        radius = self.radius
        mask = np.asarray(self.runtime.lfm.valid_mask, dtype=bool)
        result = []
        for i in range(radius, mask.shape[0] - radius):
            for j in range(radius, mask.shape[1] - radius):
                if np.count_nonzero(mask[i, j]) < 2:
                    continue
                result.append((i, j, "inline"))
                result.append((i, j, "xline"))
        return tuple(result)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | float | int]:
        rng = np.random.default_rng(self.seed + int(index))
        i, j, orientation = self.keys[int(index) % len(self.keys)]
        from ginn_v2.data import PatchKey

        key = PatchKey(i, j, orientation)  # type: ignore[arg-type]
        batch = self.runtime.reader.batch((key,), center_visible=True, device="cpu")
        support = batch.lfm_valid_mask[0].numpy().astype(bool)
        center_body = infer_body_trace(self.runtime, i, j, orientation)
        seismic = read_seismic_trace(self.runtime, i, j)
        if not np.all(np.isfinite(seismic)):
            raise ValueError("Observed seismic patch contains non-finite center samples.")
        record = self.library.records[int(rng.integers(len(self.library.records)))]
        target = np.zeros_like(center_body)
        target_mask = np.zeros_like(support, dtype=bool)
        active = support & np.isfinite(record.model_residual)
        if np.any(active):
            span = float(rng.uniform(self.runtime.config.residual_patch_min_m, self.runtime.config.residual_patch_max_m))
            active_indices = np.flatnonzero(active)
            center = int(rng.integers(active_indices.size))
            center_coord = self.library.sample_axis[active_indices[center]]
            run = active_indices[np.abs(self.library.sample_axis[active_indices] - center_coord) <= span / 2.0]
            scale = float(rng.uniform(self.runtime.config.residual_scale_min, self.runtime.config.residual_scale_max))
            target[run] = np.interp(
                self.library.sample_axis[run],
                self.library.sample_axis[active_indices],
                np.nan_to_num(record.model_residual[active_indices], nan=0.0),
            ).astype(np.float32) * scale
            target_mask[run] = True
        if rng.random() < self.runtime.config.zero_injection_fraction:
            target[:] = 0.0
            target_mask[:] = support
        base = np.nan_to_num(
            self.runtime.lfm.log_ai[i, j].astype(np.float32),
            nan=float(self.runtime.reader.normalization.lfm_mean),
        )
        target_body = center_body + target
        base_seismic = _forward_center(self.runtime, center_body, i, j, support)
        target_seismic = _forward_center(self.runtime, target_body, i, j, support)
        gain = _nonnegative_gain(seismic, base_seismic, support)
        paired = seismic + gain * (target_seismic - base_seismic)
        # The network sees the frozen body, never the injected target body.
        features = _make_features(
            paired,
            center_body,
            support,
            self.runtime,
            reference_seismic=gain * base_seismic,
        )
        return {
            "input": torch.from_numpy(features),
            "target_residual": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "support": torch.from_numpy(support),
            "base_body": torch.from_numpy(center_body),
            "target_body": torch.from_numpy(target_body),
            "observed_seismic": torch.from_numpy(seismic),
            "paired_seismic": torch.from_numpy(paired.astype(np.float32)),
            "gain": float(gain),
        }


def _forward_center(runtime: RealFieldRuntime, body: np.ndarray, i: int, j: int, support: np.ndarray) -> np.ndarray:
    from ginn_v2.data import PatchKey

    key = PatchKey(i, j, "inline")
    batch = runtime.reader.batch((key,), center_visible=True, device="cpu")
    common = CommonObservationBatch(
        sample_axis=runtime.reader.sample_axis,
        observed_seismic=batch.observed_seismic,
        observed_valid_mask=batch.observed_valid_mask,
        lfm_log_ai=batch.lfm_log_ai,
        lfm_valid_mask=batch.lfm_valid_mask,
        xy_m=batch.xy_m,
        domain_extras=batch.domain_extras,
    )
    result = runtime.adapter.close_body(torch.from_numpy(body[None, :]), common).synthetic_seismic[0].numpy()
    return result.astype(np.float32)


def _nonnegative_gain(observed: np.ndarray, predicted: np.ndarray, support: np.ndarray) -> float:
    denom = float(np.sum(np.square(predicted[support])))
    if denom <= 0.0:
        return 0.0
    return max(0.0, float(np.sum(observed[support] * predicted[support]) / denom))


def _make_features(
    seismic: np.ndarray,
    body: np.ndarray,
    support: np.ndarray,
    runtime: RealFieldRuntime,
    *,
    reference_seismic: np.ndarray | None = None,
) -> np.ndarray:
    """Build the four-channel real-field input with explicit outside-support zeros."""
    seismic = np.asarray(seismic, dtype=np.float32)
    body = np.asarray(body, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    selected = seismic[support & np.isfinite(seismic)]
    if selected.size < 2:
        raise ValueError("Enhance feature seismic trace has fewer than two finite support samples.")
    seismic_centered = seismic - float(np.mean(selected))
    seismic_scale = float(np.sqrt(np.mean(np.square(seismic_centered[support]))))
    if not np.isfinite(seismic_scale) or seismic_scale <= 0.0:
        raise ValueError("Enhance feature seismic trace has zero support variance.")
    seismic_norm = np.where(support, seismic_centered / seismic_scale, 0.0)
    mean = float(runtime.reader.normalization.lfm_mean)
    scale = max(float(runtime.reader.normalization.lfm_scale), 1e-8)
    body_safe = np.nan_to_num(body, nan=mean)
    body_norm = np.where(support, (body_safe - mean) / scale, 0.0)
    derivative = np.gradient(body_safe, runtime.reader.sample_axis.values).astype(np.float32) / scale
    derivative = np.where(support, derivative, 0.0)
    if reference_seismic is None:
        reference_seismic = np.zeros_like(seismic)
    reference = np.asarray(reference_seismic, dtype=np.float32)
    if reference.shape != seismic.shape:
        raise ValueError("reference_seismic must match seismic shape.")
    reference_selected = reference[support & np.isfinite(reference)]
    if reference_selected.size < 2:
        raise ValueError("Reference seismic trace has fewer than two finite support samples.")
    reference_centered = reference - float(np.mean(reference_selected))
    reference_scale = float(np.sqrt(np.mean(np.square(reference_centered[support]))))
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        raise ValueError("Reference seismic trace has zero support variance.")
    reference_norm = np.where(support, reference_centered / reference_scale, 0.0)
    difference = np.where(support, seismic - reference, 0.0)
    difference_norm = np.where(support, difference / max(seismic_scale, 1e-8), 0.0)
    return np.stack((seismic_norm, body_norm, derivative, support.astype(np.float32), reference_norm, difference_norm), axis=0).astype(np.float32)


def write_residual_atlas(library: ResidualLibrary, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for record in library.records:
        target = np.isfinite(record.native_residual)
        if not np.any(target):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
        axis = record.native_axis[target]
        axes[0].plot(record.native_reference[target], axis, color="black")
        axes[0].plot(record.native_body[target], axis, color="tab:blue")
        axes[0].set_title("filtered full / body")
        axes[1].plot(record.native_residual[target], axis, color="tab:red")
        axes[1].set_title("residual")
        model_valid = record.model_mask
        axes[2].plot(record.model_reference[model_valid], record.model_axis[model_valid], color="black", label="full")
        axes[2].plot(record.model_body[model_valid], record.model_axis[model_valid], color="tab:blue", label="body")
        axes[2].legend(fontsize=8)
        axes[2].set_title("model-axis anchor")
        for current in axes:
            current.invert_yaxis()
            current.grid(alpha=0.2)
        fig.tight_layout()
        path = output_dir / record.well_name / "residual_atlas.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


__all__ = [
    "PairedResidualDataset",
    "RealFieldRuntime",
    "ResidualLibrary",
    "ResidualRecord",
    "build_residual_library",
    "infer_body_trace",
    "load_runtime",
    "read_seismic_trace",
    "write_residual_atlas",
]
