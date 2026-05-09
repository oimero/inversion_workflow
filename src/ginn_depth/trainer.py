"""Depth-domain GINN trainer."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ginn.trainer import (
    append_metrics_csv,
    build_common_run_summary,
    initialize_metrics_csv,
    prefix_metrics,
    summarize_array,
    write_json,
)
from ginn.well_anchor import WellLogAIAnchor, disabled_well_anchor_summary, zero_well_anchor_metrics
from ginn_depth.config import DepthGINNConfig
from ginn_depth.data import build_dataset
from ginn_depth.loss import GINNLoss
from ginn_depth.model import DilatedResNet1D
from ginn_depth.physics import DepthForwardModel

logger = logging.getLogger(__name__)


class Trainer:
    """Depth-domain GINN trainer with log-AI residual output."""

    def __init__(self, cfg: DepthGINNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        logger.info("Building depth-domain dataset...")
        dataset_bundle = build_dataset(cfg)
        self.dataset = dataset_bundle.inference_dataset
        self.train_dataset = dataset_bundle.train_dataset
        self.val_dataset = dataset_bundle.val_dataset
        self.wavelet_time_s = dataset_bundle.wavelet_time_s
        self.wavelet_amp = dataset_bundle.wavelet_amp
        self.depth_axis_m = dataset_bundle.depth_axis_m
        self.geometry = dataset_bundle.geometry
        self.split_metadata = dataset_bundle.split_metadata

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )
        logger.info("Train DataLoader: %d batches/epoch", len(self.train_dataloader))
        self.val_dataloader = None
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                drop_last=False,
            )
            logger.info("Validation DataLoader: %d batches/epoch", len(self.val_dataloader))
        logger.info("Split metadata: %s", self.split_metadata)

        self.model = DilatedResNet1D(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            out_channels=cfg.out_channels,
            dilations=cfg.dilations,
            kernel_size=cfg.kernel_size,
        ).to(self.device)
        logger.info("Model: DilatedResNet1D, %d trainable parameters", self.model.count_parameters())

        self.forward_model = DepthForwardModel(
            self.wavelet_time_s,
            self.wavelet_amp,
            depth_axis_m=self.depth_axis_m,
            amplitude_threshold=cfg.wavelet_amplitude_threshold,
        ).to(self.device)

        self.criterion = GINNLoss(lambda_l2=cfg.lambda_l2, lambda_tv=cfg.lambda_tv)
        self.well_anchor = self._build_well_anchor()
        logger.info(
            "Loss domain: normalized depth seismic amplitude, lambda_l2=%.3e, lambda_tv=%.3e, lambda_well_log_ai=%.3e",
            self.cfg.lambda_l2,
            self.cfg.lambda_tv,
            self.cfg.lambda_well_log_ai,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.lr * 0.01,
        )

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = cfg.checkpoint_dir / "metrics.csv"
        self.run_summary_path = cfg.checkpoint_dir / "run_summary.json"
        initialize_metrics_csv(self.metrics_path)
        logger.info("Metrics CSV initialized: %s", self.metrics_path)

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self._es_best = float("inf")
        logger.info("Residual taper zero_outside_mask=%s", self.cfg.zero_residual_outside_mask)
        if self.val_dataloader is not None:
            logger.info(
                "Early stopping enabled: patience=%d, min_delta=%.2e, warmup=%d",
                self.cfg.early_stopping_patience,
                self.cfg.early_stopping_min_delta,
                self.cfg.early_stopping_warmup,
            )
        else:
            logger.info("Early stopping disabled because no validation dataset is configured.")

        if self.run_summary_path.exists():
            logger.info("Run summary already exists, preserving: %s", self.run_summary_path)
        else:
            self._write_run_summary()

    def _write_run_summary(self) -> None:
        summary = build_common_run_summary(
            domain="depth",
            cfg=self.cfg,
            device=self.device,
            geometry=self.geometry,
            split_metadata=self.split_metadata,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            inference_dataset=self.dataset,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            extra={
                "wavelet": {
                    "source": self.cfg.wavelet_source,
                    "type": self.cfg.wavelet_type,
                    "time_s": summarize_array(self.wavelet_time_s),
                    "amplitude": summarize_array(self.wavelet_amp),
                    "amplitude_threshold": self.cfg.wavelet_amplitude_threshold,
                },
                "depth_axis_m": summarize_array(self.depth_axis_m),
                "gain": {
                    "source": self.cfg.gain_source,
                    "fixed_gain": self.cfg.fixed_gain,
                    "dynamic_gain_model": self.cfg.dynamic_gain_model,
                },
                "lfm": {
                    "ai_lfm_file": self.cfg.ai_lfm_file,
                    "vp_lfm_file": self.cfg.vp_lfm_file,
                },
                "well_anchor": self._well_anchor_summary(),
            },
        )
        write_json(self.run_summary_path, summary)
        logger.info("Run summary saved: %s", self.run_summary_path)

    def _build_well_anchor(self) -> WellLogAIAnchor | None:
        n_traces = int(self.geometry["n_il"]) * int(self.geometry["n_xl"])
        return WellLogAIAnchor.build(
            prior_file=self.cfg.well_anchor_prior_file,
            lambda_weight=self.cfg.lambda_well_log_ai,
            batch_size=self.cfg.well_anchor_batch_size,
            use_prior_weight=self.cfg.well_anchor_use_prior_weight,
            sample_domain="depth",
            n_sample=self.depth_axis_m.size,
            n_traces=n_traces,
            valid_indices=self.dataset.valid_indices,
        )

    def _well_anchor_summary(self) -> dict[str, Any]:
        if self.well_anchor is None:
            return disabled_well_anchor_summary(
                prior_file=self.cfg.well_anchor_prior_file,
                lambda_weight=self.cfg.lambda_well_log_ai,
            )
        return self.well_anchor.summary()

    def _compose_impedance(
        self,
        x: torch.Tensor,
        lfm_raw: torch.Tensor,
        taper_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_residual = self.model(x)
        safe_lfm = torch.clamp(lfm_raw, min=1e-6)
        residual = raw_residual
        if self.cfg.zero_residual_outside_mask:
            if taper_weight is None:
                raise ValueError("taper_weight is required when zero_residual_outside_mask is enabled.")
            residual = residual * taper_weight.to(dtype=residual.dtype)
        ai = safe_lfm * torch.exp(residual)
        return ai, residual

    def _compute_well_anchor_loss(self, *, training: bool) -> tuple[torch.Tensor, dict[str, float]]:
        if self.well_anchor is None:
            return zero_well_anchor_metrics(self.device)
        return self.well_anchor.compute_loss(
            dataset=self.dataset,
            device=self.device,
            compose_impedance=self._compose_impedance,
            training=training,
        )

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        total_waveform_mae = 0.0
        total_residual_l2 = 0.0
        total_l2_term = 0.0
        total_tv_term = 0.0
        total_residual_tv = 0.0
        total_residual_mean = 0.0
        total_well_log_ai = 0.0
        total_well_log_ai_term = 0.0
        total_well_anchor_traces = 0.0
        n_batches = 0

        context = torch.enable_grad if training else torch.no_grad
        with context():
            for batch_idx, batch in enumerate(dataloader):
                x = batch["input"].to(self.device)
                d_obs = batch["obs"].to(self.device)
                core_mask = batch["mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                taper_weight = batch["taper_weight"].to(self.device)
                lfm_raw = batch["lfm_raw"].to(self.device)
                velocity_raw = batch["velocity_raw"].to(self.device)
                dynamic_gain = batch.get("dynamic_gain")
                if dynamic_gain is not None:
                    dynamic_gain = dynamic_gain.to(self.device)

                ai, residual = self._compose_impedance(x, lfm_raw, taper_weight)
                d_syn = self.forward_model(ai, velocity_raw, gain=dynamic_gain)
                loss, loss_dict = self.criterion(d_syn, d_obs, loss_mask, core_mask, residual, taper_weight)
                well_term, well_dict = self._compute_well_anchor_loss(training=training)
                loss = loss + well_term
                loss_dict["total"] = float(loss.detach().cpu().item())

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.global_step += 1

                residual_mean = residual.abs().mean().item()
                total_loss += loss_dict["total"]
                total_waveform_mae += loss_dict["waveform_mae"]
                total_residual_l2 += loss_dict["residual_l2"]
                total_l2_term += loss_dict["l2_term"]
                total_tv_term += loss_dict["tv_term"]
                total_residual_tv += loss_dict["residual_tv"]
                total_residual_mean += residual_mean
                total_well_log_ai += well_dict["well_log_ai"]
                total_well_log_ai_term += well_dict["well_log_ai_term"]
                total_well_anchor_traces += well_dict["well_anchor_traces"]
                n_batches += 1

                if training and (batch_idx + 1) % self.cfg.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "  [Epoch %d | Batch %d/%d] loss=%.6f (mae=%.6f l2_raw=%.3e l2=%.3e tv=%.3e tv_raw=%.3e res=%.3e well=%.3e) lr=%.2e",
                        self.epoch + 1,
                        batch_idx + 1,
                        len(dataloader),
                        loss_dict["total"],
                        loss_dict["waveform_mae"],
                        loss_dict["residual_l2"],
                        loss_dict["l2_term"],
                        loss_dict["tv_term"],
                        loss_dict["residual_tv"],
                        residual_mean,
                        well_dict["well_log_ai_term"],
                        lr,
                    )

        n_batches = max(n_batches, 1)
        return {
            "loss": total_loss / n_batches,
            "waveform_mae": total_waveform_mae / n_batches,
            "residual_l2": total_residual_l2 / n_batches,
            "l2_term": total_l2_term / n_batches,
            "tv_term": total_tv_term / n_batches,
            "residual_tv": total_residual_tv / n_batches,
            "residual_mean": total_residual_mean / n_batches,
            "well_log_ai": total_well_log_ai / n_batches,
            "well_log_ai_term": total_well_log_ai_term / n_batches,
            "well_anchor_traces": total_well_anchor_traces / n_batches,
        }

    def train_one_epoch(self) -> dict[str, float]:
        return self._run_epoch(self.train_dataloader, training=True)

    def validate(self) -> dict[str, float]:
        if self.val_dataloader is None:
            raise RuntimeError("validate() called without a validation dataloader.")
        return self._run_epoch(self.val_dataloader, training=False)

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"checkpoint_epoch{self.epoch:03d}.pt"
        path = self.cfg.checkpoint_dir / filename
        config_payload = self.cfg.to_json_dict()
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "config": config_payload,
                "normalization": {
                    "seis_rms": self.dataset.seis_rms,
                    "lfm_scale": self.dataset.lfm_scale,
                },
                "split_metadata": self.split_metadata,
            },
            path,
        )
        config_path = path.with_suffix(".config.json")
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(config_payload, fp, ensure_ascii=False, indent=2)
        logger.info("Checkpoint saved: %s", path)
        logger.info("Checkpoint config saved: %s", config_path)
        return path

    def train(self) -> None:
        logger.info("=" * 60)
        logger.info("Start depth GINN training: %d epochs, batch_size=%d", self.cfg.epochs, self.cfg.batch_size)
        logger.info("=" * 60)

        total_start = time.time()
        epochs_without_improvement = 0
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            train_metrics = self.train_one_epoch()
            self.scheduler.step()
            val_metrics = self.validate() if self.val_dataloader is not None else None
            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            if val_metrics is None:
                logger.info(
                    "Epoch %d/%d  train_loss=%.6f (mae=%.6f l2_raw=%.3e l2=%.3e tv=%.3e tv_raw=%.3e res=%.3e well=%.3e)  lr=%.2e  time=%.1fs",
                    epoch + 1,
                    self.cfg.epochs,
                    train_metrics["loss"],
                    train_metrics["waveform_mae"],
                    train_metrics["residual_l2"],
                    train_metrics["l2_term"],
                    train_metrics["tv_term"],
                    train_metrics["residual_tv"],
                    train_metrics["residual_mean"],
                    train_metrics["well_log_ai_term"],
                    lr,
                    elapsed,
                )
                monitor_value = train_metrics["loss"]
                monitor_name = "train_loss"
            else:
                logger.info(
                    "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  val_mae=%.6f  val_l2=%.3e  val_l2_raw=%.3e  "
                    "val_tv=%.3e  val_tv_raw=%.3e  val_res=%.3e  val_well=%.3e  lr=%.2e  time=%.1fs",
                    epoch + 1,
                    self.cfg.epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["waveform_mae"],
                    val_metrics["l2_term"],
                    val_metrics["residual_l2"],
                    val_metrics["tv_term"],
                    val_metrics["residual_tv"],
                    val_metrics["residual_mean"],
                    val_metrics["well_log_ai_term"],
                    lr,
                    elapsed,
                )
                monitor_value = val_metrics["loss"]
                monitor_name = "val_loss"

            if monitor_value < self.best_loss:
                self.best_loss = monitor_value
                self.best_epoch = epoch + 1
                is_best = True
                self.save_checkpoint("best.pt")
                logger.info("New best model at epoch %d: %s=%.6f", epoch + 1, monitor_name, monitor_value)
            else:
                is_best = False

            early_stop_triggered = False
            if self.val_dataloader is not None and (epoch + 1) >= self.cfg.early_stopping_warmup:
                if monitor_value < (self._es_best - self.cfg.early_stopping_min_delta):
                    self._es_best = monitor_value
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    logger.info(
                        "No significant validation improvement for %d epoch(s) "
                        "(best %s=%.6f at epoch %d, es_ref=%.6f).",
                        epochs_without_improvement,
                        monitor_name,
                        self.best_loss,
                        self.best_epoch,
                        self._es_best,
                    )
                    if self.cfg.early_stopping_patience > 0 and epochs_without_improvement >= self.cfg.early_stopping_patience:
                        early_stop_triggered = True
                        logger.info(
                            "Early stopping triggered at epoch %d after %d stale validation epochs.",
                            epoch + 1,
                            epochs_without_improvement,
                        )

            append_metrics_csv(
                self.metrics_path,
                {
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                    "lr": lr,
                    "epoch_time_s": elapsed,
                    **prefix_metrics("train", train_metrics),
                    **prefix_metrics("val", val_metrics),
                    "monitor_name": monitor_name,
                    "monitor_value": monitor_value,
                    "best_loss": self.best_loss,
                    "best_epoch": self.best_epoch,
                    "is_best": is_best,
                    "epochs_without_improvement": epochs_without_improvement,
                    "early_stop_triggered": early_stop_triggered,
                },
            )

            if early_stop_triggered:
                break

            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint("final.pt")
        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("Depth training complete. Total time: %.1f s (%.1f min)", total_time, total_time / 60.0)
        logger.info("Best monitored loss: %.6f (epoch %d)", self.best_loss, self.best_epoch)
        logger.info("=" * 60)

    @torch.no_grad()
    def predict_volume(self) -> np.ndarray:
        self.model.eval()
        n_il = int(self.geometry["n_il"])
        n_xl = int(self.geometry["n_xl"])
        n_sample = int(self.geometry["n_sample"])

        full_loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        predictions = []
        for batch in full_loader:
            x = batch["input"].to(self.device)
            lfm_raw = batch["lfm_raw"].to(self.device)
            taper_weight = batch["taper_weight"].to(self.device)
            ai, _ = self._compose_impedance(x, lfm_raw, taper_weight)
            predictions.append(ai.squeeze(1).cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        volume = self.dataset.ai_lfm_flat.astype(np.float32, copy=True)
        valid_indices = self.dataset.valid_indices
        volume[valid_indices] = predictions
        return volume.reshape(n_il, n_xl, n_sample)
