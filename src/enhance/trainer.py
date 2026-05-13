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

from cup.utils.io import to_json_compatible, write_json
from enhance.config import EnhancementConfig
from enhance.loss import EnhancementLoss, compose_enhanced_ai
from enhance.model import DilatedResNet1D

logger = logging.getLogger(__name__)

LOSS_METRIC_KEYS = [
    "delta_lowpass",
    "delta_highpass",
    "delta_rms",
    "delta_rms_underfit",
    "lowpass_term",
    "highpass_term",
    "rms_term",
    "rms_underfit_term",
    "pred_delta_rms",
    "target_delta_rms",
    "delta_rms_ratio",
    "pred_highpass_rms",
    "target_highpass_rms",
    "highpass_rms_ratio",
]

METRICS_FIELDNAMES = [
    "epoch",
    "global_step",
    "lr",
    "epoch_time_s",
    "train_loss",
    *(f"train_{key}" for key in LOSS_METRIC_KEYS),
]

MONITOR_METRICS_FIELDNAMES = [
    "epoch",
    "global_step",
    "lr",
    "monitor_loss",
    *(f"monitor_{key}" for key in LOSS_METRIC_KEYS),
    "monitor_synthetic_rms_scale_mean",
    "monitor_resample_attempts_mean",
    "monitor_quality_gate_pass_fraction",
    "monitor_quality_gate_forced_accept_fraction",
    "monitor_quality_gate_max_attempt_fraction",
    "monitor_well_patch_fraction",
    "monitor_unresolved_cluster_fraction",
    "monitor_base_target_waveform_corr_mean",
    "monitor_base_target_waveform_delta_rms_to_target_rms_mean",
    "monitor_core_mask_fraction",
    "monitor_waveform_mask_fraction",
    "monitor_delta_mask_fraction",
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
        self.monitor_metrics_path = cfg.checkpoint_dir / "monitor_metrics.csv"
        self.run_summary_path = cfg.checkpoint_dir / "run_summary.json"
        self.training_diagnostics_path = cfg.checkpoint_dir / "training_diagnostics.json"
        self.best_checkpoint_path = cfg.checkpoint_dir / "best.pt"
        _initialize_metrics_csv(self.metrics_path)
        _initialize_csv(self.monitor_metrics_path, MONITOR_METRICS_FIELDNAMES)
        self.global_step = 0
        self.epoch = 0
        self.monitor_samples = _build_fixed_monitor_samples(
            train_dataset,
            n_samples=cfg.monitor_samples,
            seed=cfg.monitor_seed,
        )
        self.monitor_dataloader = DataLoader(
            self.monitor_samples,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        self.best_monitor_loss = float("inf")
        self.best_monitor_epoch = 0
        self.history_tail: list[dict[str, Any]] = []
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
            "monitor_samples": len(self.monitor_samples),
            "monitor_batches": len(self.monitor_dataloader),
            "model_parameters": self.model.count_parameters(),
            "outputs": {
                "metrics_csv": self.metrics_path,
                "monitor_metrics_csv": self.monitor_metrics_path,
                "training_diagnostics_json": self.training_diagnostics_path,
                "run_summary_json": self.run_summary_path,
                "best_checkpoint": self.best_checkpoint_path,
            },
        }
        write_json(self.run_summary_path, payload)

    def _run_epoch(self) -> dict[str, float]:
        self.model.train(True)
        totals: dict[str, float] = {}
        n_batches = 0
        for batch_idx, batch in enumerate(self.train_dataloader):
            x = batch["input"].to(self.device)
            target_delta = batch["target_delta_log_ai"].to(self.device)
            loss_mask = batch["delta_loss_mask"].to(self.device)

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
                    "  [Enhance batch %d/%d] loss=%.6f high=%.3e rms_ratio=%.3f high_ratio=%.3f",
                    batch_idx + 1,
                    len(self.train_dataloader),
                    metrics["total"],
                    metrics["delta_highpass"],
                    metrics["delta_rms_ratio"],
                    metrics["highpass_rms_ratio"],
                )

        n_batches = max(n_batches, 1)
        return {key: value / n_batches for key, value in totals.items()}

    @torch.no_grad()
    def _evaluate_monitor(self) -> dict[str, float]:
        if not self.monitor_samples:
            return {}
        self.model.eval()
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in self.monitor_dataloader:
            x = batch["input"].to(self.device)
            target_delta = batch["target_delta_log_ai"].to(self.device)
            loss_mask = batch["delta_loss_mask"].to(self.device)

            pred_delta = self.model(x)
            if self.cfg.zero_delta_outside_mask:
                taper = batch.get("taper_weight")
                if taper is not None:
                    pred_delta = pred_delta * taper.to(self.device, dtype=pred_delta.dtype)
            _, metrics = self.criterion(pred_delta, target_delta, loss_mask)

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            _accumulate_optional_batch_stats(totals, batch)
            n_batches += 1
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
            monitor_metrics = self._evaluate_monitor()
            self.scheduler.step()
            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]
            monitor_loss = monitor_metrics.get("total")
            if monitor_loss is not None and monitor_loss < self.best_monitor_loss:
                self.best_monitor_loss = float(monitor_loss)
                self.best_monitor_epoch = epoch + 1
                self.save_checkpoint("best.pt")
            logger.info(
                "Enhance epoch %d/%d train_loss=%.6f monitor_loss=%s high=%.3e rms_ratio=%.3f high_ratio=%.3f lr=%.2e time=%.1fs",
                epoch + 1,
                self.cfg.epochs,
                metrics["total"],
                _fmt_float(monitor_loss),
                metrics["delta_highpass"],
                metrics["delta_rms_ratio"],
                metrics["highpass_rms_ratio"],
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
            if monitor_metrics:
                monitor_row = {
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                    "lr": lr,
                    "monitor_loss": monitor_metrics["total"],
                    **{f"monitor_{key}": value for key, value in monitor_metrics.items() if key != "total"},
                }
                _append_csv(self.monitor_metrics_path, monitor_row, MONITOR_METRICS_FIELDNAMES)
            self._write_training_diagnostics(
                epoch=epoch + 1,
                lr=lr,
                elapsed=elapsed,
                train_metrics=metrics,
                monitor_metrics=monitor_metrics,
            )
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint()
        self.save_checkpoint("final.pt")

    def _write_training_diagnostics(
        self,
        *,
        epoch: int,
        lr: float,
        elapsed: float,
        train_metrics: dict[str, float],
        monitor_metrics: dict[str, float],
    ) -> None:
        flags, actions = _training_flags_and_actions(train_metrics, monitor_metrics)
        latest = {
            "epoch": epoch,
            "global_step": self.global_step,
            "lr": lr,
            "epoch_time_s": elapsed,
            "train": train_metrics,
            "monitor": monitor_metrics,
        }
        self.history_tail.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.get("total"),
                "monitor_loss": monitor_metrics.get("total"),
                "train_delta_rms_ratio": train_metrics.get("delta_rms_ratio"),
                "monitor_delta_rms_ratio": monitor_metrics.get("delta_rms_ratio"),
                "train_highpass_rms_ratio": train_metrics.get("highpass_rms_ratio"),
                "monitor_highpass_rms_ratio": monitor_metrics.get("highpass_rms_ratio"),
            }
        )
        self.history_tail = self.history_tail[-10:]
        payload = {
            "updated_at_unix": time.time(),
            "overall_status": "WARN" if flags else "OK",
            "flag_counts": {"WARN": len(flags)},
            "flags": flags,
            "recommended_actions": actions,
            "latest": latest,
            "best": {
                "monitor_loss": None if not np.isfinite(self.best_monitor_loss) else self.best_monitor_loss,
                "monitor_epoch": self.best_monitor_epoch,
                "checkpoint": self.best_checkpoint_path if self.best_monitor_epoch > 0 else None,
            },
            "history_tail": self.history_tail,
            "llm_entrypoints": {
                "run_summary_json": self.run_summary_path,
                "train_metrics_csv": self.metrics_path,
                "monitor_metrics_csv": self.monitor_metrics_path,
                "training_diagnostics_json": self.training_diagnostics_path,
            },
        }
        write_json(self.training_diagnostics_path, payload)

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
                "normalization": to_json_compatible(self.normalization),
                "metadata": to_json_compatible(self.metadata),
            },
            path,
        )
        write_json(path.with_suffix(".config.json"), self.cfg.to_json_dict())
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


def _initialize_metrics_csv(path: Path) -> None:
    _initialize_csv(path, METRICS_FIELDNAMES)


def _append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    _append_csv(path, row, METRICS_FIELDNAMES)


def _initialize_csv(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.DictWriter(fp, fieldnames=fieldnames).writeheader()


def _append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    normalized = {field: to_json_compatible(row.get(field, "")) for field in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as fp:
        csv.DictWriter(fp, fieldnames=fieldnames).writerow(normalized)


def _build_fixed_monitor_samples(dataset: Dataset, *, n_samples: int, seed: int) -> list[dict[str, Any]]:
    if n_samples <= 0:
        return []
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        return [_freeze_sample(dataset[index % len(dataset)]) for index in range(int(n_samples))]
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _freeze_sample(sample: dict[str, Any]) -> dict[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            frozen[key] = value.detach().cpu().clone()
        else:
            frozen[key] = value
    return frozen


def _accumulate_optional_batch_stats(totals: dict[str, float], batch: dict[str, Any]) -> None:
    if "synthetic_rms_scale" in batch:
        totals["synthetic_rms_scale_mean"] = totals.get("synthetic_rms_scale_mean", 0.0) + float(
            batch["synthetic_rms_scale"].float().mean().item()
        )
    if "synthetic_resample_attempts" in batch:
        totals["resample_attempts_mean"] = totals.get("resample_attempts_mean", 0.0) + float(
            batch["synthetic_resample_attempts"].float().mean().item()
        )
    if "synthetic_quality_gate_passed" in batch:
        totals["quality_gate_pass_fraction"] = totals.get("quality_gate_pass_fraction", 0.0) + float(
            batch["synthetic_quality_gate_passed"].float().mean().item()
        )
    if "synthetic_quality_gate_forced_accept" in batch:
        totals["quality_gate_forced_accept_fraction"] = totals.get(
            "quality_gate_forced_accept_fraction", 0.0
        ) + float(batch["synthetic_quality_gate_forced_accept"].float().mean().item())
    if "synthetic_quality_gate_max_attempt_reached" in batch:
        totals["quality_gate_max_attempt_fraction"] = totals.get("quality_gate_max_attempt_fraction", 0.0) + float(
            batch["synthetic_quality_gate_max_attempt_reached"].float().mean().item()
        )
    if "synthetic_mode" in batch:
        mode = batch["synthetic_mode"].long()
        totals["well_patch_fraction"] = totals.get("well_patch_fraction", 0.0) + float((mode == 0).float().mean().item())
        totals["unresolved_cluster_fraction"] = totals.get("unresolved_cluster_fraction", 0.0) + float(
            (mode == 1).float().mean().item()
        )
    if "base_seismic" in batch and "target_seismic" in batch:
        waveform_mask = batch.get("loss_mask")
        if waveform_mask is None:
            waveform_mask = torch.ones_like(batch["target_seismic"], dtype=torch.bool)
        corr, delta_ratio = _batch_waveform_corr_and_delta_ratio(
            batch["base_seismic"].float(),
            batch["target_seismic"].float(),
            waveform_mask.bool(),
        )
        totals["base_target_waveform_corr_mean"] = totals.get("base_target_waveform_corr_mean", 0.0) + corr
        totals["base_target_waveform_delta_rms_to_target_rms_mean"] = totals.get(
            "base_target_waveform_delta_rms_to_target_rms_mean", 0.0
        ) + delta_ratio
    if "mask" in batch:
        totals["core_mask_fraction"] = totals.get("core_mask_fraction", 0.0) + float(batch["mask"].float().mean().item())
    if "loss_mask" in batch:
        totals["waveform_mask_fraction"] = totals.get("waveform_mask_fraction", 0.0) + float(
            batch["loss_mask"].float().mean().item()
        )
    if "delta_loss_mask" in batch:
        totals["delta_mask_fraction"] = totals.get("delta_mask_fraction", 0.0) + float(
            batch["delta_loss_mask"].float().mean().item()
        )


def _training_flags_and_actions(
    train_metrics: dict[str, float],
    monitor_metrics: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flags: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    def add_flag(
        name: str,
        message: str,
        *,
        metric: str,
        actual: float | None,
        threshold: str,
        action: str,
        related_config_keys: list[str],
    ) -> None:
        flags.append(
            {
                "level": "WARN",
                "name": name,
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "message": message,
                "related_config_keys": related_config_keys,
            }
        )
        actions.append(
            {
                "name": name,
                "suggestion": action,
                "related_config_keys": related_config_keys,
            }
        )

    metrics = monitor_metrics or train_metrics
    prefix = "monitor" if monitor_metrics else "train"
    delta_ratio = _maybe_float(metrics.get("delta_rms_ratio"))
    high_ratio = _maybe_float(metrics.get("highpass_rms_ratio"))
    target_high = _maybe_float(metrics.get("target_highpass_rms"))
    resample_attempts = _maybe_float(metrics.get("resample_attempts_mean"))
    forced_accept_fraction = _maybe_float(metrics.get("quality_gate_forced_accept_fraction"))
    max_attempt_fraction = _maybe_float(metrics.get("quality_gate_max_attempt_fraction"))
    waveform_corr = _maybe_float(metrics.get("base_target_waveform_corr_mean"))

    if delta_ratio is not None and delta_ratio < 0.55:
        add_flag(
            "delta_underfit",
            f"{prefix} delta RMS ratio is low; predicted delta is much weaker than target.",
            metric=f"{prefix}_delta_rms_ratio",
            actual=delta_ratio,
            threshold=">= 0.55",
            action="Prediction amplitude is conservative; check lr/epochs first, then consider increasing lambda_delta_rms or enabling lambda_delta_rms_underfit.",
            related_config_keys=["epochs", "lr", "lambda_delta_rms", "lambda_delta_rms_underfit", "delta_rms_floor"],
        )
    if high_ratio is not None and high_ratio < 0.45:
        add_flag(
            "highpass_underfit",
            f"{prefix} highpass RMS ratio is low; high-frequency details are not being recovered.",
            metric=f"{prefix}_highpass_rms_ratio",
            actual=high_ratio,
            threshold=">= 0.45",
            action="High-frequency detail is underfit; first verify synthetic targets contain highpass energy, then consider increasing lambda_delta_highpass or training longer.",
            related_config_keys=["lambda_delta_highpass", "delta_highpass_samples", "epochs"],
        )
    if target_high is not None and target_high < 0.01:
        add_flag(
            "target_too_smooth",
            f"{prefix} target highpass RMS is low; synthetic target may be too smooth.",
            metric=f"{prefix}_target_highpass_rms",
            actual=target_high,
            threshold=">= 0.01",
            action="Synthetic targets may be too smooth; inspect residual highpass window, well patch scaling, and cluster amplitude settings.",
            related_config_keys=[
                "synthetic_residual_highpass_samples",
                "synthetic_well_patch_scale_min",
                "synthetic_well_patch_scale_max",
                "synthetic_cluster_amp_abs_p95_min",
                "synthetic_cluster_amp_abs_p99_max",
            ],
        )
    if resample_attempts is not None and resample_attempts > 2.0:
        add_flag(
            "synthetic_rejection_high",
            f"{prefix} fixed monitor samples needed many resample attempts.",
            metric=f"{prefix}_resample_attempts_mean",
            actual=resample_attempts,
            threshold="<= 2.0",
            action="Quality gates reject many samples; inspect synthetic_max_* gate thresholds and synthetic amplitude settings together.",
            related_config_keys=[
                "synthetic_max_residual_near_clip_fraction",
                "synthetic_max_seismic_rms_ratio",
                "synthetic_max_seismic_abs_p99_ratio",
                "synthetic_max_resample_attempts",
            ],
        )
    if forced_accept_fraction is not None and forced_accept_fraction > 0.0:
        add_flag(
            "synthetic_forced_accept_present",
            f"{prefix} monitor samples include traces that failed every quality-gate retry and were still accepted.",
            metric=f"{prefix}_quality_gate_forced_accept_fraction",
            actual=forced_accept_fraction,
            threshold="== 0.0",
            action="Quality gates are filtering beyond the configured retry budget; inspect gate thresholds, residual amplitudes, and waveform realism before trusting the synthetic mix.",
            related_config_keys=[
                "synthetic_max_residual_near_clip_fraction",
                "synthetic_max_seismic_rms_ratio",
                "synthetic_max_seismic_abs_p99_ratio",
                "synthetic_min_base_target_waveform_corr",
                "synthetic_max_resample_attempts",
            ],
        )
    if max_attempt_fraction is not None and max_attempt_fraction > 0.25:
        add_flag(
            "synthetic_retry_budget_pressure",
            f"{prefix} many monitor samples consume the full quality-gate retry budget.",
            metric=f"{prefix}_quality_gate_max_attempt_fraction",
            actual=max_attempt_fraction,
            threshold="<= 0.25",
            action="Retry budget pressure is high; inspect whether the gate is too strict or the synthetic generator is producing overly aggressive traces.",
            related_config_keys=[
                "synthetic_max_residual_near_clip_fraction",
                "synthetic_max_seismic_rms_ratio",
                "synthetic_max_seismic_abs_p99_ratio",
                "synthetic_min_base_target_waveform_corr",
                "synthetic_max_resample_attempts",
            ],
        )
    if waveform_corr is not None and waveform_corr < 0.5:
        add_flag(
            "base_target_waveform_corr_low",
            f"{prefix} base/target waveform correlation is low.",
            metric=f"{prefix}_base_target_waveform_corr_mean",
            actual=waveform_corr,
            threshold=">= 0.50",
            action="Synthetic target seismic may drift far from base forward seismic; inspect QC samples before enabling a hard gate.",
            related_config_keys=["synthetic_min_base_target_waveform_corr"],
        )
    return flags, actions


def _batch_waveform_corr_and_delta_ratio(base: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    base_flat = base.reshape(base.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    mask_flat = mask.reshape(mask.shape[0], -1).to(dtype=base_flat.dtype)
    denom = mask_flat.sum(dim=1).clamp(min=1.0)
    base_mean = (base_flat * mask_flat).sum(dim=1, keepdim=True) / denom.view(-1, 1)
    target_mean = (target_flat * mask_flat).sum(dim=1, keepdim=True) / denom.view(-1, 1)
    base_centered = (base_flat - base_mean) * mask_flat
    target_centered = (target_flat - target_mean) * mask_flat
    corr_denom = torch.sqrt(base_centered.pow(2).sum(dim=1) * target_centered.pow(2).sum(dim=1)).clamp(min=1e-8)
    corr = (base_centered * target_centered).sum(dim=1) / corr_denom
    delta_rms = torch.sqrt(((target_flat - base_flat).pow(2) * mask_flat).sum(dim=1) / denom + 1e-8)
    target_rms = torch.sqrt((target_flat.pow(2) * mask_flat).sum(dim=1) / denom + 1e-8)
    ratio = delta_rms / target_rms.clamp(min=1e-8)
    return float(corr.mean().item()), float(ratio.mean().item())


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def _fmt_float(value: Any) -> str:
    value_f = _maybe_float(value)
    return "None" if value_f is None else f"{value_f:.6f}"
