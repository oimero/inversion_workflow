"""Training and evaluation for the real-field Enhance V2 residual module."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from cup.utils.io import to_json_compatible, write_json
from enhance.config import EnhanceV2Config
from enhance.loss import _moving_average_1d
from enhance.model import DilatedResNet1D
from enhance.real_field import (
    PairedResidualDataset,
    RealFieldRuntime,
    ResidualLibrary,
    _nonnegative_gain,
    infer_body_trace,
    read_seismic_trace,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    monitor_loss: float
    monitor_rmse: float
    monitor_rms_ratio: float
    well_residual_rmse: dict[str, float]
    well_residual_corr: dict[str, float]
    well_full_rmse: dict[str, float]
    well_body_rmse: dict[str, float]
    well_rms_ratio: dict[str, float]
    improved_wells: int
    candidate: bool

    def to_json_dict(self) -> dict[str, Any]:
        return to_json_compatible(self.__dict__)


class EnhanceV2Trainer:
    """Deep training module behind the one-entry-point stage-2 command."""

    def __init__(
        self,
        config: EnhanceV2Config,
        runtime: RealFieldRuntime,
        library: ResidualLibrary,
        output_dir: Path,
        rms_weight_multiplier: float = 1.0,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.library = library
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not np.isfinite(float(rms_weight_multiplier)) or float(rms_weight_multiplier) <= 0.0:
            raise ValueError("rms_weight_multiplier must be finite and positive.")
        self.rms_weight_multiplier = float(rms_weight_multiplier)
        requested = torch.device(config.device)
        self.device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
        self.train_dataset = PairedResidualDataset(
            runtime,
            library,
            count=config.training_pairs,
            seed=config.seed,
        )
        self.monitor_dataset = PairedResidualDataset(
            runtime,
            library,
            count=config.fixed_monitor_pairs,
            seed=config.seed + 100003,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=config.pin_memory if hasattr(config, "pin_memory") else self.device.type == "cuda",
        )
        self.monitor_loader = DataLoader(
            self.monitor_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.model = DilatedResNet1D(
            in_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            out_channels=1,
            dilations=config.dilations,
            kernel_size=config.kernel_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.formal_epochs),
            eta_min=config.learning_rate * 0.01,
        )
        self.coordinates = torch.from_numpy(runtime.reader.sample_axis.values.astype(np.float32)).to(self.device)
        self.metrics_path = self.output_dir / "metrics.csv"
        self._write_metrics_header()
        self.zero_monitor_rmse = self._zero_monitor_rmse()
        (
            self.anchor_inputs,
            self.anchor_targets,
            self.anchor_masks,
            self.anchor_base_bodies,
            self.anchor_names,
        ) = self._build_well_anchors()
        self.best_diagnostic_path: Path | None = None
        self.selected_path: Path | None = None
        self.epoch_metrics: list[EpochMetrics] = []

    def _write_metrics_header(self) -> None:
        with self.metrics_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "train_loss",
                    "monitor_loss",
                    "monitor_rmse",
                    "monitor_rms_ratio",
                    "improved_wells",
                    "candidate",
                    "well_residual_rmse_json",
                    "well_residual_corr_json",
                    "well_full_rmse_json",
                    "well_body_rmse_json",
                ]
            )

    def _zero_monitor_rmse(self) -> float:
        values = []
        for index in range(len(self.monitor_dataset)):
            item = self.monitor_dataset[index]
            mask = item["target_mask"].numpy().astype(bool)
            target = item["target_residual"].numpy()
            if np.count_nonzero(mask) >= 1:
                values.append(float(np.sqrt(np.mean(np.square(target[mask])))))
        if not values:
            raise ValueError("Fixed monitor has no residual support.")
        return float(np.mean(values))

    def _build_well_anchors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[str, ...]]:
        features: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        base_bodies: list[np.ndarray] = []
        names: list[str] = []
        controls = {item.well_name: item for item in self.runtime.controls.controls}
        for record in self.library.records:
            control = controls[record.well_name]
            valid = np.flatnonzero(control.observed_valid_mask & record.model_mask)
            if valid.size < 2:
                continue
            i_float, j_float = self.runtime.reader.geometry.line_to_index(
                float(control.inline_by_sample[valid[0]]),
                float(control.xline_by_sample[valid[0]]),
            )
            i, j = int(round(i_float)), int(round(j_float))
            body = infer_body_trace(self.runtime, i, j, "inline")
            seismic = read_seismic_trace(self.runtime, i, j)
            support = np.asarray(record.model_mask, dtype=bool) & np.isfinite(body) & np.isfinite(seismic)
            base_seismic = self._base_seismic(body, i, j, support)
            gain = _nonnegative_gain(seismic, base_seismic, support)
            features.append(
                _make_features(
                    seismic,
                    body,
                    support,
                    self.runtime,
                    reference_seismic=gain * base_seismic,
                )
            )
            targets.append(np.nan_to_num(record.model_residual, nan=0.0).astype(np.float32))
            masks.append(support)
            base_bodies.append(np.nan_to_num(body, nan=0.0).astype(np.float32))
            names.append(record.well_name)
        if not features:
            raise ValueError("No trusted-well residual anchors could be constructed.")
        return (
            torch.from_numpy(np.stack(features)).float(),
            torch.from_numpy(np.stack(targets)).float(),
            torch.from_numpy(np.stack(masks)).bool(),
            torch.from_numpy(np.stack(base_bodies)).float(),
            tuple(names),
        )

    def _project(self, raw: Tensor, support: Tensor) -> Tensor:
        return self.runtime.residual_projector.project(raw, self.coordinates, support)

    def smoke(self) -> dict[str, Any]:
        """Run one reversible optimizer step and checkpoint round trip."""
        batch = next(iter(self.train_loader))
        saved_model = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        saved_optimizer = self.optimizer.state_dict()
        x = batch["input"].to(self.device)
        target = batch["target_residual"].to(self.device)
        support = batch["support"].to(self.device)
        mask = batch["target_mask"].to(self.device)
        prediction = self._project(self.model(x)[:, 0, :], support)
        loss = self._loss(prediction, target, mask) + self._anchor_loss()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        smoke_path = self.output_dir / "smoke_checkpoint.pt"
        torch.save({"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, smoke_path)
        payload = torch.load(smoke_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.model.load_state_dict(saved_model, strict=True)
        self.optimizer.load_state_dict(saved_optimizer)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError("Enhance V2 smoke loss is non-finite.")
        return {"batch_size": int(x.shape[0]), "sample_count": int(x.shape[-1]), "loss": float(loss.detach().cpu()), "checkpoint": str(smoke_path)}

    @torch.no_grad()
    def predict_trace(self, i: int, j: int, orientation: str = "inline") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from enhance.real_field import read_seismic_trace

        body = infer_body_trace(self.runtime, int(i), int(j), orientation)
        seismic = read_seismic_trace(self.runtime, int(i), int(j))
        support = np.asarray(self.runtime.lfm.valid_mask[int(i), int(j)], dtype=bool)
        base_seismic = self._base_seismic(body, i, j, support)
        gain = _nonnegative_gain(seismic, base_seismic, support)
        features = _make_features(
            seismic,
            body,
            support,
            self.runtime,
            reference_seismic=gain * base_seismic,
        )
        raw = self.model(torch.from_numpy(features[None]).to(self.device))[:, 0, :]
        predicted = self._project(raw, torch.from_numpy(support[None]).to(self.device))[0].cpu().numpy()
        return body, predicted.astype(np.float32), support

    def _base_seismic(self, body: np.ndarray, i: int, j: int, support: np.ndarray) -> np.ndarray:
        from enhance.real_field import _forward_center

        return _forward_center(self.runtime, body, i, j, support)

    def _loss(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        support = mask.to(dtype=prediction.dtype)
        point_weight = torch.where(
            target.abs() > 1e-8,
            torch.full_like(target, self.config.nonzero_target_weight),
            torch.ones_like(target),
        )
        weighted_support = support * point_weight
        denom = weighted_support.sum().clamp(min=1.0)
        direct = (F.smooth_l1_loss(prediction, target, reduction="none") * weighted_support).sum() / denom
        high_prediction = prediction - _moving_average_1d(prediction, self.config.highpass_window_samples)
        high_target = target - _moving_average_1d(target, self.config.highpass_window_samples)
        high = (F.smooth_l1_loss(high_prediction, high_target, reduction="none") * weighted_support).sum() / denom
        pred_rms = torch.sqrt((prediction.square() * support).sum(dim=-1) / support.sum(dim=-1).clamp(min=1.0) + 1e-8)
        target_rms = torch.sqrt((target.square() * support).sum(dim=-1) / support.sum(dim=-1).clamp(min=1.0) + 1e-8)
        trace_weight = torch.where(
            target_rms > 1e-8,
            torch.full_like(target_rms, self.config.nonzero_target_weight),
            torch.ones_like(target_rms),
        )
        rms = (F.smooth_l1_loss(pred_rms, target_rms, reduction="none") * trace_weight).sum() / trace_weight.sum().clamp(min=1.0)
        return (
            self.config.lambda_direct * direct
            + self.config.lambda_highpass * high
            + self.config.lambda_rms * self.rms_weight_multiplier * rms
        )

    def _anchor_loss(self) -> Tensor:
        if self.config.well_anchor_weight <= 0.0:
            return torch.zeros((), device=self.device)
        features = self.anchor_inputs.to(self.device)
        target = self.anchor_targets.to(self.device)
        mask = self.anchor_masks.to(self.device)
        raw = self.model(features)[:, 0, :]
        prediction = self._project(raw, mask)
        if self.config.well_anchor_normalize:
            target_rms = torch.sqrt(
                (target.square() * mask.to(dtype=target.dtype)).sum(dim=-1)
                / mask.sum(dim=-1).clamp(min=1).to(dtype=target.dtype)
                + 1e-8
            ).detach().clamp(min=1e-6)
            prediction = prediction / target_rms[:, None]
            target = target / target_rms[:, None]
        # The anchor is evaluated on every synthetic batch, so its configured
        # weight is defined at epoch scale rather than multiplied by the
        # number of synthetic batches.
        return (
            self.config.well_anchor_weight
            * self._loss(prediction, target, mask)
            / max(1, len(self.train_loader))
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train(True)
        totals: list[float] = []
        for batch_index, batch in enumerate(self.train_loader, start=1):
            x = batch["input"].to(self.device, non_blocking=True)
            target = batch["target_residual"].to(self.device, non_blocking=True)
            support = batch["support"].to(self.device, non_blocking=True)
            mask = batch["target_mask"].to(self.device, non_blocking=True)
            raw = self.model(x)[:, 0, :]
            prediction = self._project(raw, support)
            loss = self._loss(prediction, target, mask) + self._anchor_loss()
            if not bool(torch.isfinite(loss).item()):
                raise ValueError(f"Enhance V2 loss is non-finite at epoch={epoch}, batch={batch_index}.")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            totals.append(float(loss.detach().cpu()))
            if batch_index % self.config.log_interval == 0:
                logger.info("enhance epoch=%d batch=%d/%d loss=%.6g", epoch, batch_index, len(self.train_loader), totals[-1])
        return float(np.mean(totals)) if totals else float("nan")

    @torch.no_grad()
    def _evaluate_monitor(self) -> tuple[float, float, float]:
        self.model.eval()
        losses: list[float] = []
        trace_rmse: list[float] = []
        prediction_rms: list[float] = []
        target_rms: list[float] = []
        for batch in self.monitor_loader:
            x = batch["input"].to(self.device)
            target = batch["target_residual"].to(self.device)
            support = batch["support"].to(self.device)
            mask = batch["target_mask"].to(self.device)
            prediction = self._project(self.model(x)[:, 0, :], support)
            losses.append(float(self._loss(prediction, target, mask).cpu()))
            prediction_np = prediction.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)
            for row in range(prediction_np.shape[0]):
                valid = mask_np[row]
                if not np.any(valid):
                    continue
                trace_rmse.append(float(np.sqrt(np.mean(np.square(prediction_np[row, valid] - target_np[row, valid])))))
                target_rms_row = float(np.sqrt(np.mean(np.square(target_np[row, valid]))))
                prediction_rms.append(float(np.sqrt(np.mean(np.square(prediction_np[row, valid])))))
                target_rms.append(target_rms_row)
        rmse = float(np.mean(trace_rmse))
        nonzero = np.asarray(target_rms) > 1e-8
        ratio = float(np.mean(np.asarray(prediction_rms)[nonzero]) / max(np.mean(np.asarray(target_rms)[nonzero]), 1e-8))
        return float(np.mean(losses)), rmse, ratio

    @torch.no_grad()
    def _evaluate_wells(self) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float], int]:
        self.model.eval()
        prediction = self._project(
            self.model(self.anchor_inputs.to(self.device))[:, 0, :],
            self.anchor_masks.to(self.device),
        ).cpu().numpy()
        target = self.anchor_targets.numpy()
        mask = self.anchor_masks.numpy().astype(bool)
        residual_rmse: dict[str, float] = {}
        residual_corr: dict[str, float] = {}
        full_rmse: dict[str, float] = {}
        body_rmse: dict[str, float] = {}
        rms_ratio: dict[str, float] = {}
        for row, name in enumerate(self.anchor_names):
            values = mask[row]
            target_row = target[row, values]
            pred_row = prediction[row, values]
            body = self.anchor_base_bodies.numpy()[row, values]
            reference = np.asarray(self.library.records[row].model_reference)[values]
            residual_rmse[name] = float(np.sqrt(np.mean(np.square(pred_row - target_row))))
            residual_corr[name] = float(np.corrcoef(target_row, pred_row)[0, 1]) if np.std(pred_row) > 0 and np.std(target_row) > 0 else 0.0
            full_rmse[name] = float(np.sqrt(np.mean(np.square(body + pred_row - reference))))
            body_rmse[name] = float(np.sqrt(np.mean(np.square(body - reference))))
            rms_ratio[name] = float(np.sqrt(np.mean(np.square(pred_row))) / max(np.sqrt(np.mean(np.square(target_row))), 1e-8))
        improved = sum(full_rmse[name] < body_rmse[name] for name in full_rmse)
        return residual_rmse, residual_corr, full_rmse, body_rmse, rms_ratio, int(improved)

    def _save_checkpoint(self, path: Path, epoch: int, metrics: EpochMetrics) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "enhance_v2_real_field_checkpoint_v1",
                "epoch": int(epoch),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config.to_json_dict(),
                "metrics": metrics.to_json_dict(),
            },
            path,
        )
        return path

    def train(self, *, epochs: int | None = None, start_checkpoint: Path | None = None) -> dict[str, Any]:
        if start_checkpoint is not None:
            payload = torch.load(start_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        count = int(epochs or self.config.formal_epochs)
        candidates: list[tuple[int, Path, EpochMetrics]] = []
        best_score = float("inf")
        for epoch in range(1, count + 1):
            train_loss = self._train_epoch(epoch)
            monitor_loss, monitor_rmse, monitor_ratio = self._evaluate_monitor()
            residual_rmse, residual_corr, full_rmse, body_rmse, rms_ratio, improved = self._evaluate_wells()
            candidate = (
                monitor_rmse < self.zero_monitor_rmse
                and self.config.min_prediction_rms_ratio <= monitor_ratio <= self.config.max_prediction_rms_ratio
                and improved >= self.config.min_wells_improved
            )
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                monitor_loss=monitor_loss,
                monitor_rmse=monitor_rmse,
                monitor_rms_ratio=monitor_ratio,
                well_residual_rmse=residual_rmse,
                well_residual_corr=residual_corr,
                well_full_rmse=full_rmse,
                well_body_rmse=body_rmse,
                well_rms_ratio=rms_ratio,
                improved_wells=improved,
                candidate=candidate,
            )
            self.epoch_metrics.append(metrics)
            checkpoint = self._save_checkpoint(self.output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", epoch, metrics)
            score = monitor_rmse + float(np.mean(list(residual_rmse.values())))
            if score < best_score:
                best_score = score
                self.best_diagnostic_path = checkpoint
            if candidate and not candidates:
                candidates.append((epoch, checkpoint, metrics))
            with self.metrics_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch,
                        train_loss,
                        monitor_loss,
                        monitor_rmse,
                        monitor_ratio,
                        improved,
                        candidate,
                        json.dumps(residual_rmse, ensure_ascii=False, sort_keys=True),
                        json.dumps(residual_corr, ensure_ascii=False, sort_keys=True),
                        json.dumps(full_rmse, ensure_ascii=False, sort_keys=True),
                        json.dumps(body_rmse, ensure_ascii=False, sort_keys=True),
                    ]
                )
            logger.info(
                "enhance epoch=%d/%d train=%.6g monitor_rmse=%.6g ratio=%.3f improved=%d candidate=%s",
                epoch,
                count,
                train_loss,
                monitor_rmse,
                monitor_ratio,
                improved,
                candidate,
            )
            self.scheduler.step()
        selected = candidates[0][1] if candidates else None
        if selected is not None:
            self.selected_path = self.output_dir / "selected_checkpoint.pt"
            selected_payload = torch.load(selected, map_location="cpu", weights_only=False)
            torch.save(selected_payload, self.selected_path)
        elif self.best_diagnostic_path is not None:
            self.selected_path = self.output_dir / "best_diagnostic_checkpoint.pt"
            payload = torch.load(self.best_diagnostic_path, map_location="cpu", weights_only=False)
            torch.save(payload, self.selected_path)
        last = self.output_dir / "last.pt"
        if self.epoch_metrics:
            torch.save(
                {
                    "schema": "enhance_v2_real_field_checkpoint_v1",
                    "epoch": self.epoch_metrics[-1].epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "config": self.config.to_json_dict(),
                    "metrics": self.epoch_metrics[-1].to_json_dict(),
                },
                last,
            )
        return {
            "status": "accepted" if selected is not None else "completed_not_accepted",
            "selected_checkpoint": self.selected_path,
            "last": last,
            "best_diagnostic_checkpoint": self.best_diagnostic_path,
            "zero_monitor_rmse": self.zero_monitor_rmse,
            "epochs": [item.to_json_dict() for item in self.epoch_metrics],
        }


def enhance_sections(
    runtime: RealFieldRuntime,
    checkpoint: Path,
    frozen_body_sections: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply one frozen checkpoint to deterministic body section inputs."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DilatedResNet1D(
        in_channels=runtime.config.input_channels,
        hidden_channels=runtime.config.hidden_channels,
        out_channels=1,
        dilations=runtime.config.dilations,
        kernel_size=runtime.config.kernel_size,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    result: dict[str, np.ndarray] = {}
    for name, body_section in frozen_body_sections.items():
        values = np.asarray(body_section, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("frozen_body_sections values must have shape (traces, samples).")
        output = values.copy()
        for row in range(values.shape[0]):
            support = np.isfinite(values[row])
            if np.count_nonzero(support) < 2:
                continue
            seismic = np.zeros_like(values[row])
            features = _make_features(seismic, np.nan_to_num(values[row], nan=0.0), support, runtime)
            with torch.no_grad():
                raw = model(torch.from_numpy(features[None]))[:, 0, :]
                delta = runtime.projector.project(raw, torch.from_numpy(support[None]), torch.from_numpy(support[None]))[0].numpy()
            output[row, support] = values[row, support] + delta[support]
        result[name] = output
    return result


def _make_features(
    seismic: np.ndarray,
    body: np.ndarray,
    support: np.ndarray,
    runtime: RealFieldRuntime,
    *,
    reference_seismic: np.ndarray | None = None,
) -> np.ndarray:
    seismic = np.asarray(seismic, dtype=np.float32)
    body = np.asarray(body, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    seismic_norm = _normalize(seismic, support)
    scale = float(runtime.reader.normalization.lfm_scale)
    mean = float(runtime.reader.normalization.lfm_mean)
    body_safe = np.nan_to_num(body, nan=mean)
    body_norm = (body_safe - mean) / max(scale, 1e-8)
    derivative = np.gradient(body_safe, runtime.reader.sample_axis.values).astype(np.float32)
    derivative /= max(scale, 1e-8)
    body_norm = np.where(support, body_norm, 0.0)
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
    if reference_scale <= 0.0 or not np.isfinite(reference_scale):
        raise ValueError("Reference seismic trace has zero support variance.")
    reference_norm = np.where(support, reference_centered / reference_scale, 0.0)
    difference = np.where(support, seismic - reference, 0.0)
    difference_norm = np.where(support, difference / max(scale, 1e-8), 0.0)
    return np.stack((seismic_norm, body_norm, derivative, support.astype(np.float32), reference_norm, difference_norm), axis=0).astype(np.float32)


def _normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask & np.isfinite(values)]
    if selected.size < 2:
        raise ValueError("Feature trace has fewer than two finite support samples.")
    centered = values - float(np.mean(selected))
    scale = float(np.sqrt(np.mean(np.square(centered[mask]))))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Feature trace has zero support variance.")
    return np.where(mask, centered / scale, 0.0).astype(np.float32)


__all__ = ["EnhanceV2Trainer", "EpochMetrics"]
