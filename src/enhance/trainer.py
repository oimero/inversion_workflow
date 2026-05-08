"""Synthetic-only trainer for stage-2 resolution enhancement."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from enhance.config import EnhancementConfig
from enhance.loss import EnhancementLoss, compose_enhanced_ai
from enhance.model import DilatedResNet1D

logger = logging.getLogger(__name__)

METRICS_FIELDNAMES = [
    "epoch",
    "global_step",
    "lr",
    "epoch_time_s",
    "train_loss",
    "train_delta_lowpass",
    "train_delta_highpass",
    "train_delta_rms",
    "train_delta_rms_underfit",
    "train_lowpass_term",
    "train_highpass_term",
    "train_rms_term",
    "train_rms_underfit_term",
    "train_pred_delta_rms",
    "train_target_delta_rms",
    "train_delta_rms_ratio",
    "train_pred_highpass_rms",
    "train_target_highpass_rms",
    "train_highpass_rms_ratio",
]


class EnhancementTrainer:
    """Train an enhancement delta model from synthetic target deltas."""

    def __init__(
        self,
        cfg: EnhancementConfig,
        train_dataset: Dataset,
        *,
        normalization: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.train_dataset = train_dataset
        self.normalization = normalization or {}
        self.metadata = metadata or {}

        batch_size = cfg.synthetic_batch_size or cfg.batch_size
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
        )
        self.model = DilatedResNet1D(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            out_channels=cfg.out_channels,
            dilations=cfg.dilations,
            kernel_size=cfg.kernel_size,
        ).to(self.device)
        self.criterion = EnhancementLoss(
            lambda_lowpass=cfg.lambda_delta_lowpass,
            lambda_highpass=cfg.lambda_delta_highpass,
            lambda_rms=cfg.lambda_delta_rms,
            lambda_rms_underfit=cfg.lambda_delta_rms_underfit,
            rms_floor=cfg.delta_rms_floor,
            lowpass_samples=cfg.delta_lowpass_samples,
            highpass_samples=cfg.delta_highpass_samples,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, cfg.epochs),
            eta_min=cfg.lr * 0.01,
        )
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = cfg.checkpoint_dir / "metrics.csv"
        self.run_summary_path = cfg.checkpoint_dir / "run_summary.json"
        _initialize_metrics_csv(self.metrics_path)
        self.global_step = 0
        self.epoch = 0
        self._write_run_summary()

    def _write_run_summary(self) -> None:
        payload = {
            "created_at_unix": time.time(),
            "config": self.cfg.to_json_dict(),
            "device": {"requested": self.cfg.device, "resolved": str(self.device)},
            "normalization": self.normalization,
            "metadata": self.metadata,
            "train_dataset_len": len(self.train_dataset),
            "train_batches": len(self.train_dataloader),
            "model_parameters": self.model.count_parameters(),
        }
        _write_json(self.run_summary_path, payload)

    def _run_epoch(self) -> dict[str, float]:
        self.model.train(True)
        totals: dict[str, float] = {}
        n_batches = 0
        for batch_idx, batch in enumerate(self.train_dataloader):
            x = batch["input"].to(self.device)
            target_delta = batch["target_delta_log_ai"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)

            pred_delta = self.model(x)
            if self.cfg.zero_delta_outside_mask:
                taper = batch.get("taper_weight")
                if taper is not None:
                    pred_delta = pred_delta * taper.to(self.device, dtype=pred_delta.dtype)
            loss, metrics = self.criterion(pred_delta, target_delta, loss_mask)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.global_step += 1

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
            if (batch_idx + 1) % self.cfg.log_interval == 0:
                logger.info(
                    "  [Enhance batch %d/%d] loss=%.6f high=%.3e rms_ratio=%.3f",
                    batch_idx + 1,
                    len(self.train_dataloader),
                    metrics["total"],
                    metrics["delta_highpass"],
                    metrics["delta_rms_ratio"],
                )

        n_batches = max(n_batches, 1)
        return {key: value / n_batches for key, value in totals.items()}

    def train(self) -> None:
        logger.info(
            "Start stage-2 enhancement training: epochs=%d, traces_per_epoch=%d",
            self.cfg.epochs,
            len(self.train_dataset),
        )
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            start = time.time()
            metrics = self._run_epoch()
            self.scheduler.step()
            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Enhance epoch %d/%d loss=%.6f high=%.3e rms_ratio=%.3f lr=%.2e time=%.1fs",
                epoch + 1,
                self.cfg.epochs,
                metrics["total"],
                metrics["delta_highpass"],
                metrics["delta_rms_ratio"],
                lr,
                elapsed,
            )
            row = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "lr": lr,
                "epoch_time_s": elapsed,
                "train_loss": metrics["total"],
                **{f"train_{key}": value for key, value in metrics.items() if key != "total"},
            }
            _append_metrics_csv(self.metrics_path, row)
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint()
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, filename: str | None = None) -> Path:
        if filename is None:
            filename = f"checkpoint_epoch{self.epoch + 1:03d}.pt"
        path = self.cfg.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.cfg.to_json_dict(),
                "normalization": _json_compatible(self.normalization),
                "metadata": _json_compatible(self.metadata),
            },
            path,
        )
        _write_json(path.with_suffix(".config.json"), self.cfg.to_json_dict())
        logger.info("Enhancement checkpoint saved: %s", path)
        return path

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        x = batch["input"].to(self.device)
        base_ai = batch["base_ai_raw"].to(self.device)
        taper = batch.get("taper_weight")
        delta = self.model(x)
        if self.cfg.zero_delta_outside_mask and taper is not None:
            delta = delta * taper.to(self.device, dtype=delta.dtype)
        enhanced_ai = compose_enhanced_ai(base_ai, delta, ai_min=self.cfg.ai_min, ai_max=self.cfg.ai_max)
        return enhanced_ai.cpu(), delta.cpu()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(_json_compatible(payload), fp, ensure_ascii=False, indent=2)


def _initialize_metrics_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.DictWriter(fp, fieldnames=METRICS_FIELDNAMES).writeheader()


def _append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    normalized = {field: _json_compatible(row.get(field, "")) for field in METRICS_FIELDNAMES}
    with path.open("a", encoding="utf-8", newline="") as fp:
        csv.DictWriter(fp, fieldnames=METRICS_FIELDNAMES).writerow(normalized)
