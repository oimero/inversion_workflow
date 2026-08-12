"""GINN V2 body-inversion curriculum, evaluation, and run orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from cup.lfm.math import LowpassSpec
from ginn_v2.adapters import DomainAdapter
from ginn_v2.checkpoint import load_checkpoint, save_epoch_checkpoint
from ginn_v2.contracts import CommonObservationBatch
from ginn_v2.data import PatchBatch, PatchKey, PatchReader
from ginn_v2.evaluation import (
    EvaluationMetrics,
    GateReport,
    GateThresholds,
    evaluate_gates,
)
from ginn_v2.losses import (
    analytic_gain_diagnostic,
    lfm_anchor_loss,
    masked_lfm_lowpass,
    short_wave_energy_ratio,
    waveform_shape_loss,
)
from ginn_v2.model import BodyNetworkConfig, CenterTraceBodyNet
from ginn_v2.projector import BodyScaleProjector
from ginn_v2.qc import write_well_waveform_qc
from ginn_v2.split import (
    SpatialSplit,
    WellPatchTarget,
    WellSampleSplit,
    WellTarget,
    build_well_body_target,
    build_well_patch_targets,
    build_well_splits,
    make_spatial_split,
    well_target_zone_mask,
)
from cup.well.real_field_controls import WellControlSet


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class BodyInversionLossWeights:
    seismic_shape: float = 1.0
    lfm_anchor: float = 1.0
    trusted_well_body: float = 1.0
    trusted_well_derivative: float = 0.5
    lambda_shape: float = 0.25

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Loss weight {name} must be finite and non-negative.")


@dataclass(frozen=True)
class CheckpointSelectionWeights:
    well_rmse: float
    roughness: float
    short_wave: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Checkpoint selection weight {name} must be finite and non-negative.")
        if not any(float(value) > 0.0 for value in asdict(self).values()):
            raise ValueError("At least one checkpoint selection weight must be positive.")


@dataclass(frozen=True)
class BodyInversionConfig:
    """Frozen body-inversion business and training configuration."""

    body_smoothing_fwhm_m: float
    selection_weights: CheckpointSelectionWeights
    waveform_qc_dynamic_window_m: float
    gates: GateThresholds
    patch_radius: int = 8
    batch_size: int = 8
    pretrain_epochs: int = 1
    finetune_epochs: int = 3
    pretrain_learning_rate: float = 1e-3
    finetune_learning_rate: float = 2e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    max_train_centers: int = 512
    max_validation_centers: int = 128
    review_fraction: float = 0.04
    validation_gap_m: float = 300.0
    validation_anchor: str = "maxmin"
    well_batch_multiplier: int = 2
    trusted_well_names: tuple[str, ...] = ()
    orientations: tuple[str, ...] = ("inline", "xline")
    device: str = "cpu"
    seed: int = 20260812
    cache_size: int = 256
    log_every_batches: int = 20
    loss_weights: BodyInversionLossWeights = BodyInversionLossWeights()
    network: BodyNetworkConfig = BodyNetworkConfig()

    def __post_init__(self) -> None:
        for name in (
            "body_smoothing_fwhm_m",
            "pretrain_learning_rate",
            "finetune_learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "validation_gap_m",
            "waveform_qc_dynamic_window_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.body_smoothing_fwhm_m <= 0.0 or self.pretrain_learning_rate <= 0.0 or self.finetune_learning_rate <= 0.0:
            raise ValueError("Body scale and learning rates must be positive.")
        if self.waveform_qc_dynamic_window_m <= 0.0:
            raise ValueError("waveform_qc_dynamic_window_m must be positive.")
        _positive_int(self.patch_radius, name="patch_radius")
        _positive_int(self.batch_size, name="batch_size")
        _positive_int(self.max_train_centers, name="max_train_centers")
        _positive_int(self.max_validation_centers, name="max_validation_centers")
        if not 1 <= int(self.pretrain_epochs) <= 3 or not 1 <= int(self.finetune_epochs) <= 3:
            raise ValueError("pretrain_epochs and finetune_epochs must be within [1, 3].")
        _positive_int(self.well_batch_multiplier, name="well_batch_multiplier")
        if not 0.0 < float(self.review_fraction) < 1.0:
            raise ValueError("review_fraction must be within (0, 1).")
        if self.validation_anchor not in {"maxmax", "maxmin", "minmax", "minmin", "center"}:
            raise ValueError("Unsupported validation_anchor.")
        if not self.trusted_well_names:
            raise ValueError("trusted_well_names must be explicitly frozen and non-empty.")
        names = tuple(str(name).strip() for name in self.trusted_well_names)
        if any(not name for name in names) or len({name.casefold() for name in names}) != len(names):
            raise ValueError("trusted_well_names must be unique non-empty names.")
        if not self.orientations or any(name not in {"inline", "xline"} for name in self.orientations):
            raise ValueError("orientations must contain inline and/or xline.")
        if isinstance(self.seed, bool) or int(self.seed) < 0 or isinstance(self.cache_size, bool) or int(self.cache_size) < 0:
            raise ValueError("seed must be non-negative and cache_size must be non-negative.")
        _positive_int(self.log_every_batches, name="log_every_batches")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BodyInversionConfig":
        config = dict(value)
        loss_value = config.pop("loss_weights", config.pop("loss", {}))
        gate_value = config.pop("gates", {})
        network_value = config.pop("network", {})
        selection_value = config.pop("selection_weights", {})
        if (
            not isinstance(loss_value, Mapping)
            or not isinstance(gate_value, Mapping)
            or not isinstance(network_value, Mapping)
            or not isinstance(selection_value, Mapping)
        ):
            raise ValueError("ginn_v2_body_inversion loss/gates/network/selection_weights must be mappings.")
        if "trusted_well_names" in config:
            config["trusted_well_names"] = tuple(config["trusted_well_names"])
        if "orientations" in config:
            config["orientations"] = tuple(config["orientations"])
        return cls(
            **config,
            loss_weights=BodyInversionLossWeights(**dict(loss_value)),
            gates=GateThresholds(**dict(gate_value)),
            network=BodyNetworkConfig(**dict(network_value)),
            selection_weights=CheckpointSelectionWeights(**dict(selection_value)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "body_smoothing_fwhm_m": self.body_smoothing_fwhm_m,
            "selection_weights": asdict(self.selection_weights),
            "patch_radius": self.patch_radius,
            "batch_size": self.batch_size,
            "pretrain_epochs": self.pretrain_epochs,
            "finetune_epochs": self.finetune_epochs,
            "pretrain_learning_rate": self.pretrain_learning_rate,
            "finetune_learning_rate": self.finetune_learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "max_train_centers": self.max_train_centers,
            "max_validation_centers": self.max_validation_centers,
            "review_fraction": self.review_fraction,
            "validation_gap_m": self.validation_gap_m,
            "validation_anchor": self.validation_anchor,
            "well_batch_multiplier": self.well_batch_multiplier,
            "waveform_qc_dynamic_window_m": self.waveform_qc_dynamic_window_m,
            "trusted_well_names": list(self.trusted_well_names),
            "orientations": list(self.orientations),
            "device": self.device,
            "seed": self.seed,
            "cache_size": self.cache_size,
            "log_every_batches": self.log_every_batches,
            "loss_weights": asdict(self.loss_weights),
            "gates": asdict(self.gates),
            "network": asdict(self.network),
        }


@dataclass(frozen=True)
class BodyInversionData:
    reader: PatchReader
    spatial_split: SpatialSplit
    well_targets: Mapping[str, WellTarget]
    well_splits: tuple[WellSampleSplit, ...]
    trusted_well_patches: tuple[WellPatchTarget, ...]
    trusted_well_names: tuple[str, ...]

    def split_description(self) -> dict[str, Any]:
        return {
            "train_patch_keys": [asdict(item) for item in self.spatial_split.train_keys],
            "validation_patch_keys": [asdict(item) for item in self.spatial_split.validation_keys],
            "review_patch_keys": [asdict(item) for item in self.spatial_split.review_keys],
            "validation_centers": [list(item) for item in self.spatial_split.validation_centers],
            "validation_block_xy_m": list(self.spatial_split.block_xy_m),
            "validation_gap_m": self.spatial_split.gap_m,
            "validation_anchor": self.spatial_split.anchor,
            "well_splits": [asdict(item) for item in self.well_splits],
            "trusted_well_names": list(self.trusted_well_names),
        }


def _sample_center_keys(
    keys: tuple[PatchKey, ...],
    *,
    max_centers: int,
    seed: int,
) -> tuple[PatchKey, ...]:
    centers = sorted({(item.inline_index, item.xline_index) for item in keys})
    if len(centers) <= max_centers:
        return keys
    selected_indices = np.random.default_rng(seed).choice(
        len(centers),
        size=max_centers,
        replace=False,
    )
    selected = {centers[int(index)] for index in selected_indices}
    return tuple(
        item
        for item in keys
        if (item.inline_index, item.xline_index) in selected
    )


def build_body_inversion_data(
    reader: PatchReader,
    controls: WellControlSet,
    *,
    config: BodyInversionConfig,
    lfm_lowpass_spec: LowpassSpec,
    candidate_keys: Iterable[PatchKey],
    target_zone_mask: np.ndarray,
) -> BodyInversionData:
    """Freeze the spatial split and trusted-well sample identities once."""

    control_by_name = {item.well_name: item for item in controls.controls}
    trusted: list[Any] = []
    for name in config.trusted_well_names:
        control = control_by_name.get(name)
        if control is None:
            raise ValueError(f"Configured trusted well is absent from WellControlSet: {name}")
        trusted.append(control)
    trusted_set = WellControlSet(
        sample_axis=controls.sample_axis,
        controls=tuple(trusted),
        sample_domain=controls.sample_domain,
        sample_unit=controls.sample_unit,
        depth_basis=controls.depth_basis,
        source_run_type=controls.source_run_type,
        provenance=controls.provenance,
    )
    projector = BodyScaleProjector(
        smoothing_fwhm_m=config.body_smoothing_fwhm_m,
        sample_step=float(reader.sample_axis.step),
        lowpass_spec=lfm_lowpass_spec,
    )
    targets = {
        control.well_name: build_well_body_target(
            control,
            body_smoothing_fwhm_m=config.body_smoothing_fwhm_m,
            target_zone_support=well_target_zone_mask(
                control,
                geometry=reader.geometry,
                target_zone_mask=target_zone_mask,
            ),
            lfm_log_ai=reader.lfm_log_ai,
            lfm_valid_mask=reader.lfm_valid_mask,
            geometry=reader.geometry,
            projector=projector,
        )
        for control in trusted
    }
    well_splits = build_well_splits(
        trusted_set,
        targets,
    )
    full_spatial = make_spatial_split(
        tuple(candidate_keys),
        geometry=reader.geometry,
        validation_fraction=config.review_fraction,
        gap_m=config.validation_gap_m,
        anchor=config.validation_anchor,  # type: ignore[arg-type]
    )
    sampled_train = _sample_center_keys(
            full_spatial.train_keys,
            max_centers=config.max_train_centers,
            seed=config.seed,
        )
    sampled_validation = _sample_center_keys(
        full_spatial.validation_keys,
        max_centers=config.max_validation_centers,
        seed=config.seed + 1,
    )
    spatial = SpatialSplit(
        train_keys=sampled_train,
        validation_keys=sampled_validation,
        review_keys=full_spatial.review_keys,
        validation_centers=full_spatial.validation_centers,
        block_xy_m=full_spatial.block_xy_m,
        gap_m=full_spatial.gap_m,
        anchor=full_spatial.anchor,
    )
    trusted_well_patches = build_well_patch_targets(
        trusted_set,
        targets,
        well_splits,
        geometry=reader.geometry,
        subset="train",
        orientations=config.orientations,  # type: ignore[arg-type]
    )
    return BodyInversionData(
        reader=reader,
        spatial_split=spatial,
        well_targets=targets,
        well_splits=well_splits,
        trusted_well_patches=trusted_well_patches,
        trusted_well_names=tuple(config.trusted_well_names),
    )


@dataclass(frozen=True)
class TrialAdjustment:
    trial_id: int
    action: str
    learning_rate: float
    loss_weights: BodyInversionLossWeights

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "action": self.action,
            "learning_rate": self.learning_rate,
            "loss_weights": asdict(self.loss_weights),
        }


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    adjustment: TrialAdjustment
    checkpoints: tuple[str, ...]
    selected_checkpoint: str | None
    selected_epoch: int | None
    metrics: EvaluationMetrics
    gate: GateReport
    stop_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "adjustment": self.adjustment.to_json_dict(),
            "checkpoints": list(self.checkpoints),
            "selected_checkpoint": self.selected_checkpoint,
            "selected_epoch": self.selected_epoch,
            "metrics": self.metrics.to_json_dict(),
            "gate": self.gate.to_json_dict(),
            "stop_reason": self.stop_reason,
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _chunks(values: tuple[Any, ...], batch_size: int, *, seed: int) -> list[tuple[Any, ...]]:
    order = np.arange(len(values), dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    return [tuple(values[int(index)] for index in order[start : start + batch_size]) for start in range(0, len(order), batch_size)]


def _finite_mean(values: list[Tensor], *, name: str) -> float:
    if not values:
        raise ValueError(f"No values were produced for {name}.")
    result = torch.cat([item.detach().reshape(-1) for item in values])
    if not bool(torch.all(torch.isfinite(result)).item()):
        raise ValueError(f"{name} contains non-finite values.")
    return float(torch.mean(result).cpu())


class BodyInversionTrainer:
    """Deep training module behind the body-inversion command."""

    def __init__(
        self,
        data: BodyInversionData,
        *,
        adapter: DomainAdapter,
        config: BodyInversionConfig,
        lfm_lowpass_spec: LowpassSpec,
        output_dir: Path,
        artifact_root: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.data = data
        self.adapter = adapter
        self.config = config
        self.lfm_lowpass_spec = lfm_lowpass_spec
        self.projector = BodyScaleProjector(
            smoothing_fwhm_m=config.body_smoothing_fwhm_m,
            sample_step=float(data.reader.sample_axis.step),
            lowpass_spec=lfm_lowpass_spec,
        )
        self.output_dir = Path(output_dir)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else self.output_dir
        self.log = logger or logging.getLogger(__name__)
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Body inversion device='cuda' was requested but CUDA is unavailable.")
        if data.reader.input_channels != config.network.input_channels:
            raise ValueError(
                f"PatchReader input channels {data.reader.input_channels} differ from network input channels {config.network.input_channels}."
            )

    def _common(self, batch: PatchBatch) -> CommonObservationBatch:
        return CommonObservationBatch(
            sample_axis=self.data.reader.sample_axis,
            observed_seismic=batch.observed_seismic,
            observed_valid_mask=batch.observed_valid_mask,
            lfm_log_ai=batch.lfm_log_ai,
            lfm_valid_mask=batch.lfm_valid_mask,
            xy_m=batch.xy_m,
            domain_extras=batch.domain_extras,
        )

    def _coordinates(self, common: CommonObservationBatch) -> Tensor:
        return self.adapter.vertical_coordinates_m(common)

    def _predict(
        self,
        model: CenterTraceBodyNet | None,
        batch: PatchBatch,
    ) -> tuple[Tensor, Tensor, CommonObservationBatch]:
        common = self._common(batch)
        if model is None:
            closure = self.adapter.close_body(
                batch.lfm_log_ai,
                common,
            )
            body = closure.body_log_ai
            synthetic = closure.synthetic_seismic
        else:
            raw_correction = model(
                batch.features,
                center_index=self.data.reader.patch_radius,
            )
            body_correction = self.projector.project(
                raw_correction,
                self._coordinates(common),
                batch.lfm_valid_mask,
            )
            body = batch.lfm_log_ai + body_correction
            closure = self.adapter.close_body(
                body,
                common,
            )
            body = closure.body_log_ai
            synthetic = closure.synthetic_seismic
        if not bool(torch.all(torch.isfinite(body)).item()) or not bool(torch.all(torch.isfinite(synthetic)).item()):
            raise ValueError("Body-inversion model or forward output contains non-finite values.")
        return body, synthetic, common

    def _loss(
        self,
        body: Tensor,
        synthetic: Tensor,
        common: CommonObservationBatch,
        *,
        well_items: tuple[WellPatchTarget, ...] = (),
        weights: BodyInversionLossWeights,
        include_seismic: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        shape_loss = torch.zeros((), dtype=body.dtype, device=body.device)
        if include_seismic:
            shape_loss = waveform_shape_loss(
                common.observed_seismic,
                synthetic,
                common.observed_valid_mask,
                lambda_shape=weights.lambda_shape,
            ).loss
        if weights.lfm_anchor > 0.0:
            anchor = lfm_anchor_loss(
                body,
                common.lfm_log_ai,
                common.lfm_valid_mask,
                sample_step=float(common.sample_axis.step),
                lowpass_spec=self.lfm_lowpass_spec,
            )
        else:
            anchor = torch.zeros((), dtype=body.dtype, device=body.device)
        well_loss = torch.zeros((), dtype=body.dtype, device=body.device)
        well_derivative_loss = torch.zeros((), dtype=body.dtype, device=body.device)
        if well_items:
            target = torch.zeros_like(body)
            target_mask = torch.zeros_like(body, dtype=torch.bool)
            for row, item in enumerate(well_items):
                target[row] = torch.as_tensor(item.target_values, device=body.device, dtype=body.dtype)
                target_mask[row] = torch.as_tensor(item.target_mask, device=body.device, dtype=torch.bool)
            per_sample_scale = torch.ones_like(body)
            for row, item in enumerate(well_items):
                per_sample_scale[row] = float(item.target_scale)
            well_loss = F.smooth_l1_loss(
                body[target_mask] / per_sample_scale[target_mask],
                target[target_mask] / per_sample_scale[target_mask],
                reduction="mean",
            )
            coordinates = self._coordinates(common)
            if coordinates.ndim == 1:
                coordinates = coordinates[None, :].expand_as(body)
            pair_mask = target_mask[:, 1:] & target_mask[:, :-1]
            if bool(torch.any(pair_mask).item()):
                spacing = coordinates[:, 1:] - coordinates[:, :-1]
                if bool(torch.any(spacing <= 0.0).item()):
                    raise ValueError("Well derivative coordinates must be strictly increasing.")
                body_derivative = (body[:, 1:] - body[:, :-1]) / spacing
                target_derivative = (target[:, 1:] - target[:, :-1]) / spacing
                derivative_scale = per_sample_scale[:, 1:] / torch.clamp(
                    torch.abs(spacing),
                    min=torch.finfo(body.dtype).eps,
                )
                well_derivative_loss = F.smooth_l1_loss(
                    body_derivative[pair_mask] / derivative_scale[pair_mask],
                    target_derivative[pair_mask] / derivative_scale[pair_mask],
                    reduction="mean",
                )
        total = (
            weights.seismic_shape * shape_loss
            + weights.lfm_anchor * anchor
            + weights.trusted_well_body * well_loss
            + weights.trusted_well_derivative * well_derivative_loss
        )
        if not bool(torch.isfinite(total).item()):
            raise ValueError("Body-inversion loss is non-finite.")
        return total, {
            "seismic_shape": shape_loss.detach(),
            "lfm_anchor": anchor.detach(),
            "trusted_well_body": well_loss.detach(),
            "trusted_well_derivative": well_derivative_loss.detach(),
        }

    def _scheduled_batches(
        self,
        *,
        epoch: int,
        trial_id: int,
        adjustment: TrialAdjustment,
    ) -> Iterable[tuple[str, tuple[Any, ...], bool]]:
        base_seed = self.config.seed + 1000 * trial_id + epoch
        masked = _chunks(self.data.spatial_split.train_keys, self.config.batch_size, seed=base_seed)
        visible = _chunks(self.data.spatial_split.train_keys, self.config.batch_size, seed=base_seed + 1)
        by_well: dict[str, list[WellPatchTarget]] = {}
        for item in self.data.trusted_well_patches:
            by_well.setdefault(item.well_name, []).append(item)
        if not by_well:
            raise ValueError("Trusted-well training has no patch targets.")
        rng = np.random.default_rng(base_seed + 2)
        per_well = max(len(items) for items in by_well.values())
        balanced_wells: list[WellPatchTarget] = []
        for name in sorted(by_well):
            items = by_well[name]
            indices = rng.choice(len(items), size=per_well, replace=len(items) < per_well)
            balanced_wells.extend(items[int(index)] for index in indices)
        rng.shuffle(balanced_wells)
        wells = _chunks(tuple(balanced_wells), self.config.batch_size, seed=base_seed + 3)
        n_steps = max(len(masked), len(visible), len(wells))
        for step in range(n_steps):
            yield "masked", masked[step % len(masked)], False
            yield "visible_center", visible[step % len(visible)], True
            for _ in range(self.config.well_batch_multiplier):
                yield "trusted_well", wells[step % len(wells)], True

    def train_epoch(
        self,
        model: CenterTraceBodyNet,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        trial_id: int,
        adjustment: TrialAdjustment,
        pretrain: bool = False,
    ) -> dict[str, float]:
        model.train()
        totals: dict[str, list[Tensor]] = {
            "total": [],
            "seismic_shape": [],
            "lfm_anchor": [],
            "trusted_well_body": [],
            "trusted_well_derivative": [],
        }
        started = time.monotonic()
        schedule: Iterable[tuple[str, tuple[Any, ...], bool]]
        if pretrain:
            schedule = (
                ("masked", values, False)
                for values in _chunks(
                    self.data.spatial_split.train_keys,
                    self.config.batch_size,
                    seed=self.config.seed + epoch,
                )
            )
        else:
            schedule = self._scheduled_batches(epoch=epoch, trial_id=trial_id, adjustment=adjustment)
        for batch_index, (kind, values, center_visible) in enumerate(
            schedule,
            start=1,
        ):
            if kind == "trusted_well":
                well_items = tuple(values)
                keys = tuple(item.patch_key for item in well_items)
            else:
                well_items = ()
                keys = tuple(values)
            batch = self.data.reader.batch(keys, center_visible=center_visible, device=self.device)
            optimizer.zero_grad(set_to_none=True)
            body, synthetic, common = self._predict(model, batch)
            total, components = self._loss(
                body,
                synthetic,
                common,
                well_items=well_items,
                weights=adjustment.loss_weights,
                include_seismic=kind != "trusted_well",
            )
            total.backward()
            for parameter_name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.all(torch.isfinite(parameter.grad)).item()):
                    raise ValueError(f"Body-inversion gradient contains non-finite values: {kind} {parameter_name}.")
            if self.config.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            totals["total"].append(total.detach())
            for name, value in components.items():
                totals[name].append(value)
            if batch_index % self.config.log_every_batches == 0:
                self.log.info(
                    "run %d | epoch %d/%d | batch %d | kind=%s | loss=%.6f | elapsed=%.1fs",
                    trial_id,
                    epoch,
                    self.config.pretrain_epochs if pretrain else self.config.finetune_epochs,
                    batch_index,
                    kind,
                    _finite_mean(totals["total"], name="running_train_total"),
                    time.monotonic() - started,
                )
        return {name: _finite_mean(values, name=f"train_{name}") for name, values in totals.items()}

    def _trace_evaluation(
        self,
        model: CenterTraceBodyNet | None,
        keys: tuple[PatchKey, ...],
        *,
        center_visible: bool,
    ) -> dict[str, Any]:
        if not keys:
            raise ValueError("Trace evaluation requires non-empty fixed identities.")
        correlations: list[np.ndarray] = []
        shape_losses: list[np.ndarray] = []
        gains: list[np.ndarray] = []
        raw_residuals: list[np.ndarray] = []
        bodies: dict[PatchKey, np.ndarray] = {}
        lfm_values: dict[PatchKey, np.ndarray] = {}
        supports: dict[PatchKey, np.ndarray] = {}
        with torch.no_grad():
            for start in range(0, len(keys), self.config.batch_size):
                local_keys = keys[start : start + self.config.batch_size]
                batch = self.data.reader.batch(local_keys, center_visible=center_visible, device=self.device)
                body, synthetic, common = self._predict(model, batch)
                shape = waveform_shape_loss(common.observed_seismic, synthetic, common.observed_valid_mask)
                diagnostic = analytic_gain_diagnostic(common.observed_seismic, synthetic, common.observed_valid_mask)
                correlations.append(shape.correlation.cpu().numpy())
                shape_losses.append(shape.normalized_shape_loss.cpu().numpy())
                gains.append(diagnostic.gain.cpu().numpy())
                raw_residuals.append(diagnostic.raw_amplitude_residual.cpu().numpy())
                for row, key in enumerate(local_keys):
                    bodies[key] = body[row].cpu().numpy()
                    lfm_values[key] = batch.lfm_log_ai[row].cpu().numpy()
                    supports[key] = common.observed_valid_mask[row].cpu().numpy()
        return {
            "correlation": np.concatenate(correlations),
            "shape_loss": np.concatenate(shape_losses),
            "gain": np.concatenate(gains),
            "raw_residual": np.concatenate(raw_residuals),
            "bodies": bodies,
            "lfm": lfm_values,
            "supports": supports,
        }

    def _well_evaluation(self, model: CenterTraceBodyNet | None) -> dict[str, Any]:
        values: dict[str, dict[int, list[float]]] = {}
        targets: dict[str, dict[int, float]] = {}
        with torch.no_grad():
            for start in range(0, len(self.data.trusted_well_patches), self.config.batch_size):
                items = self.data.trusted_well_patches[start : start + self.config.batch_size]
                batch = self.data.reader.batch(
                    tuple(item.patch_key for item in items),
                    center_visible=True,
                    device=self.device,
                )
                body, _, _ = self._predict(model, batch)
                for row, item in enumerate(items):
                    for sample_index in np.flatnonzero(item.target_mask):
                        index = int(sample_index)
                        values.setdefault(item.well_name, {}).setdefault(index, []).append(
                            float(body[row, index].cpu())
                        )
                        targets.setdefault(item.well_name, {})[index] = float(item.target_values[index])
        rmse_by_well: dict[str, float] = {}
        bias_by_well: dict[str, float] = {}
        correlation_by_well: dict[str, float] = {}
        all_residuals: list[float] = []
        roughness_by_well: dict[str, float] = {}
        axis = np.asarray(self.data.reader.sample_axis.values, dtype=np.float64)
        for name in sorted(values):
            indices = sorted(values[name])
            predicted = np.asarray([np.mean(values[name][index]) for index in indices], dtype=np.float64)
            target = np.asarray([targets[name][index] for index in indices], dtype=np.float64)
            residual = predicted - target
            rmse_by_well[name] = float(np.sqrt(np.mean(np.square(residual))))
            bias_by_well[name] = float(np.mean(residual))
            correlation = float(np.corrcoef(target, predicted)[0, 1])
            if not np.isfinite(correlation):
                raise ValueError(f"Well body correlation is non-finite: {name}")
            correlation_by_well[name] = correlation
            all_residuals.extend(residual.tolist())
            if len(indices) >= 3:
                coords = axis[np.asarray(indices)]
                differences = np.diff(coords)
                contiguous = np.isclose(differences, np.median(differences), rtol=0.0, atol=1e-8)
                if not np.any(contiguous):
                    raise ValueError(f"Well target has no adjacent physical samples: {name}")
                predicted_rough = float(np.sqrt(np.mean(np.square(np.diff(predicted)[contiguous] / differences[contiguous]))))
                target_rough = float(np.sqrt(np.mean(np.square(np.diff(target)[contiguous] / differences[contiguous]))))
                if target_rough <= 0.0:
                    raise ValueError(f"Well body target has zero roughness: {name}")
                roughness_by_well[name] = predicted_rough / target_rough
        if not all_residuals or set(roughness_by_well) != set(rmse_by_well):
            raise ValueError("Well evaluation did not produce residual and roughness support.")
        return {
            "rmse_by_well": rmse_by_well,
            "bias_by_well": bias_by_well,
            "correlation_by_well": correlation_by_well,
            "pooled_rmse": float(np.sqrt(np.mean(np.square(all_residuals)))),
            "pooled_bias": float(np.mean(all_residuals)),
            "roughness_by_well": roughness_by_well,
            "roughness_ratio": float(np.median(np.asarray(list(roughness_by_well.values()), dtype=np.float64))),
        }

    def _lfm_drift(self, trace_eval: Mapping[str, Any], keys: tuple[PatchKey, ...]) -> float:
        differences: list[float] = []
        with torch.no_grad():
            for start in range(0, len(keys), self.config.batch_size):
                local = keys[start : start + self.config.batch_size]
                body = torch.as_tensor(
                    np.stack([trace_eval["bodies"][key] for key in local]),
                    device=self.device,
                    dtype=torch.float32,
                )
                lfm = torch.as_tensor(
                    np.stack([trace_eval["lfm"][key] for key in local]),
                    device=self.device,
                    dtype=torch.float32,
                )
                mask = torch.as_tensor(
                    np.stack(
                        [
                            self.data.reader.lfm_valid_mask[
                                key.inline_index,
                                key.xline_index,
                                :,
                            ]
                            for key in local
                        ]
                    ),
                    device=self.device,
                    dtype=torch.bool,
                )
                residual_low, residual_support = masked_lfm_lowpass(
                    body - lfm,
                    mask,
                    sample_step=float(self.data.reader.sample_axis.step),
                    spec=self.lfm_lowpass_spec,
                )
                differences.extend(residual_low[mask & residual_support].cpu().numpy().tolist())
        if not differences:
            raise ValueError("LFM drift has no valid support.")
        return float(np.sqrt(np.mean(np.square(differences))))

    def validation_section_keys(
        self,
        orientation: str,
        *,
        max_traces: int = 128,
    ) -> tuple[PatchKey, ...]:
        """Choose one contiguous blind profile from the validation block."""

        if orientation not in {"inline", "xline"}:
            raise ValueError("orientation must be inline or xline.")
        if isinstance(max_traces, bool) or int(max_traces) <= 0:
            raise ValueError("max_traces must be a positive integer.")
        grouped: dict[int, list[PatchKey]] = {}
        for key in self.data.spatial_split.review_keys:
            if key.orientation != orientation:
                continue
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
            raise ValueError(f"Validation block has no {orientation} profile.")
        selected = max(runs, key=lambda item: (len(item), -item[0].inline_index, -item[0].xline_index))
        if len(selected) > int(max_traces):
            start = (len(selected) - int(max_traces)) // 2
            selected = selected[start : start + int(max_traces)]
        return selected

    def lateral_distance_m(self, keys: tuple[PatchKey, ...]) -> np.ndarray:
        if not keys:
            raise ValueError("lateral_distance_m requires at least one key.")
        xy = np.asarray(
            [
                self.data.reader.geometry.line_to_coord(
                    self.data.reader.ilines[key.inline_index],
                    self.data.reader.xlines[key.xline_index],
                )
                for key in keys
            ],
            dtype=np.float64,
        )
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

    def _orientation_disagreement(self, trace_eval: Mapping[str, Any], keys: tuple[PatchKey, ...]) -> float:
        bodies: Mapping[PatchKey, np.ndarray] = trace_eval["bodies"]
        pairs: list[float] = []
        for key in keys:
            other_orientation: PatchKey = PatchKey(
                key.inline_index,
                key.xline_index,
                "xline" if key.orientation == "inline" else "inline",
            )
            if other_orientation in bodies:
                left = bodies[key]
                right = bodies[other_orientation]
                pairs.extend((left - right).tolist())
        if not pairs:
            return 0.0
        denominator = max(float(np.std(np.concatenate([bodies[key] for key in keys]))), 1e-12)
        return float(np.sqrt(np.mean(np.square(pairs))) / denominator)

    def _write_epoch_validation_artifacts(
        self,
        model: CenterTraceBodyNet,
        *,
        metrics: EvaluationMetrics,
        gate: GateReport,
        trial_id: int,
        epoch: int,
    ) -> None:
        """Publish fixed-identity metrics and figures beside each checkpoint."""

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stage_dir = "pretraining" if trial_id == 0 else f"trial_{trial_id:02d}"
        epoch_dir = self.output_dir / stage_dir / "validation" / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        with (epoch_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"metrics": metrics.to_json_dict(), "gate": gate.to_json_dict()},
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        section_keys = {
            orientation: self.validation_section_keys(orientation)
            for orientation in self.config.orientations
        }
        review_keys = tuple(
            dict.fromkeys(key for keys in section_keys.values() for key in keys)
        )
        trace_eval = self._trace_evaluation(model, review_keys, center_visible=True)
        axis_values = np.asarray(self.data.reader.sample_axis.values, dtype=np.float64)
        for orientation in self.config.orientations:
            keys = section_keys[orientation]
            lateral_m = self.lateral_distance_m(keys)
            bodies = np.stack([trace_eval["bodies"][item] for item in keys])
            lfm_values = np.stack([trace_eval["lfm"][item] for item in keys])
            support = np.stack([trace_eval["supports"][item] for item in keys])
            body_plot = np.where(support, bodies, np.nan)
            residual_plot = np.where(support, bodies - lfm_values, np.nan)
            vertical_support = np.any(support, axis=0)
            sample_indices = np.flatnonzero(vertical_support)
            if sample_indices.size < 2:
                raise ValueError("Validation section has fewer than two target-zone samples.")
            start, stop = int(sample_indices[0]), int(sample_indices[-1])
            figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            extent = [lateral_m[0], lateral_m[-1], axis_values[stop], axis_values[start]]
            axes[0].imshow(body_plot[:, start : stop + 1].T, aspect="auto", origin="upper", extent=extent)
            axes[0].set_title(f"Fixed validation body — {orientation}")
            axes[0].set_ylabel(self.data.reader.sample_axis.unit)
            axes[1].imshow(residual_plot[:, start : stop + 1].T, aspect="auto", origin="upper", extent=extent, cmap="RdBu_r")
            axes[1].set_title("GINN body minus LFM")
            axes[1].set_xlabel("lateral distance (m)")
            axes[1].set_ylabel(self.data.reader.sample_axis.unit)
            figure.tight_layout()
            figure.savefig(epoch_dir / f"fixed_validation_{orientation}.png", dpi=120)
            plt.close(figure)

    def evaluate(self, model: CenterTraceBodyNet | None) -> EvaluationMetrics:
        model_eval = self._trace_evaluation(model, self.data.spatial_split.validation_keys, center_visible=True)
        masked_eval = self._trace_evaluation(model, self.data.spatial_split.validation_keys, center_visible=False)
        wells = self._well_evaluation(model)
        coordinates = torch.as_tensor(self.data.reader.sample_axis.values, device=self.device, dtype=torch.float32)
        short_wave_values: list[np.ndarray] = []
        support_counts: list[int] = []
        support_is_contiguous: list[bool] = []
        validation_keys = self.data.spatial_split.validation_keys
        with torch.no_grad():
            for start in range(0, len(validation_keys), self.config.batch_size):
                local = validation_keys[start : start + self.config.batch_size]
                body = torch.as_tensor(
                    np.stack([model_eval["bodies"][key] for key in local]),
                    device=self.device,
                    dtype=torch.float32,
                )
                support_array = np.stack([model_eval["supports"][key] for key in local])
                support = torch.as_tensor(support_array, device=self.device, dtype=torch.bool)
                body_short_wave = short_wave_energy_ratio(
                    body,
                    coordinates,
                    support,
                    body_smoothing_fwhm_m=self.config.body_smoothing_fwhm_m,
                )
                lfm_short_wave = short_wave_energy_ratio(
                    torch.as_tensor(
                        np.stack([model_eval["lfm"][key] for key in local]),
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    coordinates,
                    support,
                    body_smoothing_fwhm_m=self.config.body_smoothing_fwhm_m,
                )
                short_wave_values.append(
                    (body_short_wave / torch.clamp(lfm_short_wave, min=torch.finfo(body_short_wave.dtype).eps))
                    .cpu()
                    .numpy()
                )
                for row in support_array:
                    indices = np.flatnonzero(row)
                    support_counts.append(int(indices.size))
                    support_is_contiguous.append(bool(indices.size and np.all(np.diff(indices) == 1)))
        support_count_array = np.asarray(support_counts, dtype=np.int64)
        if np.any(support_count_array < 2):
            raise ValueError("Validation support has a trace with fewer than two samples.")
        return EvaluationMetrics(
            masked_correlation=np.asarray(masked_eval["correlation"], dtype=np.float64),
            masked_shape_loss=np.asarray(masked_eval["shape_loss"], dtype=np.float64),
            visible_correlation=np.asarray(model_eval["correlation"], dtype=np.float64),
            visible_shape_loss=np.asarray(model_eval["shape_loss"], dtype=np.float64),
            well_rmse_by_well=wells["rmse_by_well"],
            well_bias_by_well=wells["bias_by_well"],
            well_pooled_rmse=wells["pooled_rmse"],
            well_pooled_bias=wells["pooled_bias"],
            well_body_correlation_by_well=wells["correlation_by_well"],
            lfm_drift_rmse=self._lfm_drift(model_eval, validation_keys),
            short_wave_energy_fraction=float(np.mean(np.concatenate(short_wave_values))),
            roughness_ratio=wells["roughness_ratio"],
            roughness_ratio_by_well=wells["roughness_by_well"],
            analytic_gain_mean=float(np.mean(model_eval["gain"])),
            raw_amplitude_residual_mean=float(np.mean(model_eval["raw_residual"])),
            support_contiguous_fraction=float(np.mean(support_is_contiguous)),
            orientation_disagreement_rms_ratio=self._orientation_disagreement(
                model_eval,
                self.data.spatial_split.validation_keys,
            ),
            sample_count=int(np.sum(support_count_array)),
        )

    def _model_and_optimizer(self, learning_rate: float) -> tuple[CenterTraceBodyNet, torch.optim.Optimizer]:
        _seed_everything(self.config.seed)
        model = CenterTraceBodyNet(self.config.network).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return model, optimizer

    def run_pretraining(self) -> tuple[Path, EvaluationMetrics]:
        """Train the shared masked model once and persist its final branch point."""

        model, optimizer = self._model_and_optimizer(self.config.pretrain_learning_rate)
        checkpoint_dir = self.output_dir / "pretraining" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path: Path | None = None
        final_metrics: EvaluationMetrics | None = None
        diagnostic_gate = GateReport(
            passed=True,
            failed_gates=(),
            first_failed_gate=None,
            details={"stage": 0},
        )
        adjustment = TrialAdjustment(
            trial_id=0,
            action="masked_pretraining",
            learning_rate=self.config.pretrain_learning_rate,
            loss_weights=self.config.loss_weights,
        )
        for epoch in range(1, self.config.pretrain_epochs + 1):
            self.log.info(
                "pretraining | epoch %d/%d start | lr=%.6g",
                epoch,
                self.config.pretrain_epochs,
                self.config.pretrain_learning_rate,
            )
            train_metrics = self.train_epoch(
                model,
                optimizer,
                epoch=epoch,
                trial_id=0,
                adjustment=adjustment,
                pretrain=True,
            )
            metrics = self.evaluate(model)
            final_path = save_epoch_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                trial_id=1,
                network_config=self.config.network,
                run_config=self.config.to_json_dict(),
                split_description=self.data.split_description(),
                metrics=metrics,
                gate=diagnostic_gate,
            )
            self._write_epoch_validation_artifacts(
                model,
                metrics=metrics,
                gate=diagnostic_gate,
                trial_id=0,
                epoch=epoch,
            )
            write_well_waveform_qc(
                self,
                model,
                self.output_dir / f"trial_{adjustment.trial_id:02d}" / "validation" / f"epoch_{epoch:03d}" / "well_waveform_qc",
                root=self.artifact_root,
            )
            self.log.info(
                "pretraining | epoch %d complete | train_loss=%.6f | masked_corr=%.4f | well_rmse=%.5f | checkpoint=%s",
                epoch,
                train_metrics["total"],
                float(np.median(metrics.masked_correlation)),
                metrics.well_pooled_rmse,
                final_path.name,
            )
            final_metrics = metrics
        if final_path is None or final_metrics is None:
            raise ValueError("Masked pretraining produced no checkpoint.")
        shared_path = self.output_dir / "pretraining" / "shared_checkpoint.pt"
        shutil.copyfile(final_path, shared_path)
        return shared_path, final_metrics

    def run_finetuning(
        self,
        *,
        baseline: EvaluationMetrics,
        resume_checkpoint: Path,
        adjustment: TrialAdjustment | None = None,
    ) -> TrialResult:
        """Run one fixed semi-supervised finetune and select its best epoch."""

        if adjustment is None:
            adjustment = TrialAdjustment(
                trial_id=1,
                action="single_semi_supervised_finetune",
                learning_rate=self.config.finetune_learning_rate,
                loss_weights=self.config.loss_weights,
            )
        model, optimizer = self._model_and_optimizer(adjustment.learning_rate)
        trial_dir = self.output_dir / f"trial_{adjustment.trial_id:02d}"
        checkpoint_dir = trial_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = load_checkpoint(
            resume_checkpoint,
            model=model,
            expected_network_config=self.config.network,
            map_location=self.device,
        )
        if dict(payload["run_config"]) != self.config.to_json_dict():
            raise ValueError("Pretraining checkpoint run configuration differs from the current body-inversion configuration.")
        if dict(payload["split"]) != self.data.split_description():
            raise ValueError("Pretraining checkpoint split differs from the current fixed body-inversion split.")
        pretrain_metrics = EvaluationMetrics.from_json_dict(payload["metrics"])
        checkpoints: list[Path] = []
        metrics_by_epoch: dict[int, EvaluationMetrics] = {}
        gates_by_epoch: dict[int, GateReport] = {}
        for epoch in range(1, self.config.finetune_epochs + 1):
            self.log.info(
                "finetune | epoch %d/%d start | action=%s | lr=%.6g",
                epoch,
                self.config.finetune_epochs,
                adjustment.action,
                adjustment.learning_rate,
            )
            train_metrics = self.train_epoch(
                model,
                optimizer,
                epoch=epoch,
                trial_id=adjustment.trial_id,
                adjustment=adjustment,
            )
            metrics = self.evaluate(model)
            gate = evaluate_gates(
                metrics,
                baseline,
                pretrain_metrics,
                thresholds=self.config.gates,
            )
            checkpoint_path = save_epoch_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                trial_id=adjustment.trial_id,
                network_config=self.config.network,
                run_config=self.config.to_json_dict(),
                split_description=self.data.split_description(),
                metrics=metrics,
                gate=gate,
            )
            self._write_epoch_validation_artifacts(
                model,
                metrics=metrics,
                gate=gate,
                trial_id=adjustment.trial_id,
                epoch=epoch,
            )
            write_well_waveform_qc(
                self,
                model,
                self.output_dir / f"trial_{adjustment.trial_id:02d}" / "validation" / f"epoch_{epoch:03d}" / "well_waveform_qc",
                root=self.artifact_root,
            )
            checkpoints.append(checkpoint_path)
            metrics_by_epoch[epoch] = metrics
            gates_by_epoch[epoch] = gate
            self.log.info(
                "finetune | epoch %d complete | train_loss=%.6f | seismic_shape=%.6f | lfm_anchor=%.6f | well_loss=%.6f | well_derivative=%.6f | masked_corr=%.4f | visible_corr=%.4f | well_rmse=%.5f | lfm_drift=%.5f | roughness_median=%.4f | short_wave_ratio=%.4f | failed=%s | checkpoint=%s",
                epoch,
                train_metrics["total"],
                train_metrics["seismic_shape"],
                train_metrics["lfm_anchor"],
                train_metrics["trusted_well_body"],
                train_metrics["trusted_well_derivative"],
                float(np.median(metrics.masked_correlation)),
                float(np.median(metrics.visible_correlation)),
                metrics.well_pooled_rmse,
                metrics.lfm_drift_rmse,
                metrics.roughness_ratio,
                metrics.short_wave_energy_fraction,
                ",".join(gate.failed_gates) if gate.failed_gates else "none",
                checkpoint_path.name,
            )

        if not checkpoints:
            raise ValueError("Fine-tuning produced no evaluation checkpoint.")
        safe_epochs = [
            epoch
            for epoch, metrics in metrics_by_epoch.items()
            if float(np.median(metrics.masked_correlation))
            >= float(np.median(pretrain_metrics.masked_correlation)) - self.config.gates.masked_corr_drop_tolerance
        ]
        candidate_epochs = safe_epochs or list(metrics_by_epoch)
        selected_epoch = min(
            candidate_epochs,
            key=lambda epoch: (
                self.config.selection_weights.well_rmse
                * metrics_by_epoch[epoch].well_pooled_rmse
                / max(abs(pretrain_metrics.well_pooled_rmse), torch.finfo(torch.float32).eps)
                + self.config.selection_weights.roughness
                * abs(metrics_by_epoch[epoch].roughness_ratio - 1.0)
                + self.config.selection_weights.short_wave
                * abs(metrics_by_epoch[epoch].short_wave_energy_fraction - 1.0),
                epoch,
            ),
        )
        selected_checkpoint = trial_dir / "selected_checkpoint.pt"
        shutil.copyfile(checkpoints[selected_epoch - 1], selected_checkpoint)
        selected_gate = gates_by_epoch[selected_epoch]
        return TrialResult(
            trial_id=adjustment.trial_id,
            adjustment=adjustment,
            checkpoints=tuple(str(path) for path in checkpoints),
            selected_checkpoint=str(selected_checkpoint),
            selected_epoch=selected_epoch,
            metrics=metrics_by_epoch[selected_epoch],
            gate=selected_gate,
            stop_reason="all_finetune_epochs_completed_best_quality_checkpoint",
        )


__all__ = [
    "BodyInversionConfig",
    "BodyInversionData",
    "BodyInversionLossWeights",
    "BodyInversionTrainer",
    "TrialAdjustment",
    "TrialResult",
    "build_body_inversion_data",
]
