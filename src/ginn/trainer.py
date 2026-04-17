"""ginn.trainer — 主训练循环。

串联 DilatedResNet1D + ForwardModel + MaskedMAELoss，执行标准的
前向传播 → 物理正演 → 损失计算 → 反向传播训练流程。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ginn.config import GINNConfig
from ginn.data import build_dataset
from ginn.loss import GINNLoss
from ginn.model import DilatedResNet1D
from ginn.physics import ForwardModel

logger = logging.getLogger(__name__)


class Trainer:
    """GINN 训练器。

    Parameters
    ----------
    cfg : GINNConfig
        完整配置对象。
    """

    def __init__(self, cfg: GINNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # ── 数据 ──
        logger.info("Building dataset...")
        dataset_bundle = build_dataset(cfg)
        self.dataset = dataset_bundle.inference_dataset
        self.train_dataset = dataset_bundle.train_dataset
        self.val_dataset = dataset_bundle.val_dataset
        wavelet = dataset_bundle.wavelet
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

        # ── 模型 ──
        self.model = DilatedResNet1D(
            in_channels=cfg.in_channels,
            hidden_channels=cfg.hidden_channels,
            out_channels=cfg.out_channels,
            dilations=cfg.dilations,
            kernel_size=cfg.kernel_size,
        ).to(self.device)

        n_params = self.model.count_parameters()
        logger.info("Model: DilatedResNet1D, %d trainable parameters", n_params)

        # ── 物理正演 ──
        self.forward_model = ForwardModel(wavelet).to(self.device)

        # ── 损失 ──
        self.criterion = GINNLoss(lambda_l2=cfg.lambda_l2, lambda_tv=cfg.lambda_tv)
        logger.info(
            "Loss domain: normalized seismic amplitude (obs pre-divided by RMS), lambda_l2=%.3e, lambda_tv=%.3e",
            self.cfg.lambda_l2,
            self.cfg.lambda_tv,
        )

        # ── 优化器 ──
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.lr * 0.01,
        )

        # ── 输出目录 ──
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── 日志 ──
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self._es_best = float("inf")
        logger.info(
            "AI bounding: ai_min=%.2f, ai_max=%.2f, zero_outside_mask=%s",
            self.cfg.ai_min,
            self.cfg.ai_max,
            self.cfg.zero_residual_outside_mask,
        )
        if self.val_dataloader is not None:
            logger.info(
                "Early stopping enabled: patience=%d, min_delta=%.2e, warmup=%d",
                self.cfg.early_stopping_patience,
                self.cfg.early_stopping_min_delta,
                self.cfg.early_stopping_warmup,
            )
        else:
            logger.info("Early stopping disabled because no validation dataset is configured.")

    def _compose_impedance(
        self,
        x: torch.Tensor,
        lfm_raw: torch.Tensor,
        taper_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """网络输出区间有界的对数残差并与 LFM 合成阻抗。

        Notes
        -----
        - 使用 ``sigmoid(raw)`` 将目标层内 AI 严格限制到 ``[ai_min, ai_max]``。
        - 先由 ``log(ai / lfm_raw)`` 反算有界残差，再在 halo 区通过 taper 平滑收口，
          使层外阻抗退回到 ``lfm_raw``。
        - 若启用 ``zero_residual_outside_mask``，则用 core+halo taper 将层外残差
          平滑压回 0，避免在目的层边界处形成新的硬切。
        """
        raw_residual = self.model(x)
        safe_lfm = torch.clamp(lfm_raw, min=1e-6)
        residual_lo = torch.log(torch.full_like(safe_lfm, float(self.cfg.ai_min)) / safe_lfm)
        residual_hi = torch.log(torch.full_like(safe_lfm, float(self.cfg.ai_max)) / safe_lfm)
        alpha = torch.sigmoid(raw_residual)
        residual = residual_lo + (residual_hi - residual_lo) * alpha

        if self.cfg.zero_residual_outside_mask:
            if taper_weight is None:
                raise ValueError("taper_weight is required when zero_residual_outside_mask is enabled.")
            residual = residual * taper_weight.to(dtype=residual.dtype)

        ai = lfm_raw * torch.exp(residual)
        return ai, residual

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> dict[str, float]:
        """执行一个 train/validation epoch，并返回聚合指标。"""
        self.model.train(training)
        total_loss = 0.0
        total_waveform_mae = 0.0
        total_residual_l2 = 0.0
        total_l2_term = 0.0
        total_tv_term = 0.0
        total_residual_tv = 0.0
        total_residual_mean = 0.0
        n_batches = 0

        context = torch.enable_grad if training else torch.no_grad
        with context():
            for batch_idx, batch in enumerate(dataloader):
                x = batch["input"].to(self.device)  # (B, 2, T)
                d_obs = batch["obs"].to(self.device)  # (B, 1, T)
                core_mask = batch["mask"].to(self.device)  # (B, 1, T)
                loss_mask = batch["loss_mask"].to(self.device)  # (B, 1, T)
                taper_weight = batch["taper_weight"].to(self.device)  # (B, 1, T)
                lfm_raw = batch["lfm_raw"].to(self.device)  # (B, 1, T)

                # 1. 网络前向 + 阻抗合成
                ai, residual = self._compose_impedance(x, lfm_raw, taper_weight)

                # 2. 物理正演
                d_syn = self.forward_model(ai)  # (B, 1, T)

                # 3. 损失
                loss, loss_dict = self.criterion(d_syn, d_obs, loss_mask, core_mask, residual, taper_weight)

                if training:
                    # 4. 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪
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
                n_batches += 1

                if training and (batch_idx + 1) % self.cfg.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "  [Epoch %d | Batch %d/%d] loss=%.6f (mae=%.6f l2_raw=%.3e l2=%.3e tv=%.3e tv_raw=%.3e res=%.3e) lr=%.2e",
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
        }

    def train_one_epoch(self) -> dict[str, float]:
        """训练一个 epoch，返回 epoch 平均指标。"""
        return self._run_epoch(self.train_dataloader, training=True)

    def validate(self) -> dict[str, float]:
        """评估一个 validation epoch，返回平均指标。"""
        if self.val_dataloader is None:
            raise RuntimeError("validate() called without a validation dataloader.")
        return self._run_epoch(self.val_dataloader, training=False)

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        """保存模型 checkpoint。"""
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
        """完整训练流程。"""
        logger.info("=" * 60)
        logger.info("Start training: %d epochs, batch_size=%d", self.cfg.epochs, self.cfg.batch_size)
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
                    "Epoch %d/%d  train_loss=%.6f (mae=%.6f l2_raw=%.3e l2=%.3e tv=%.3e tv_raw=%.3e res=%.3e)  lr=%.2e  time=%.1fs",
                    epoch + 1,
                    self.cfg.epochs,
                    train_metrics["loss"],
                    train_metrics["waveform_mae"],
                    train_metrics["residual_l2"],
                    train_metrics["l2_term"],
                    train_metrics["tv_term"],
                    train_metrics["residual_tv"],
                    train_metrics["residual_mean"],
                    lr,
                    elapsed,
                )
                monitor_value = train_metrics["loss"]
                monitor_name = "train_loss"
            else:
                logger.info(
                    "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  val_mae=%.6f  val_l2_raw=%.3e  val_l2=%.3e  "
                    "val_tv=%.3e  val_tv_raw=%.3e  val_res=%.3e  lr=%.2e  time=%.1fs",
                    epoch + 1,
                    self.cfg.epochs,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    val_metrics["waveform_mae"],
                    val_metrics["residual_l2"],
                    val_metrics["l2_term"],
                    val_metrics["tv_term"],
                    val_metrics["residual_tv"],
                    val_metrics["residual_mean"],
                    lr,
                    elapsed,
                )
                monitor_value = val_metrics["loss"]
                monitor_name = "val_loss"

            # 保存最优模型（任何改善都保存，不受 min_delta 约束）
            if monitor_value < self.best_loss:
                self.best_loss = monitor_value
                self.best_epoch = epoch + 1
                self.save_checkpoint("best.pt")
                logger.info("New best model at epoch %d: %s=%.6f", epoch + 1, monitor_name, monitor_value)

            # Early stopping（独立跟踪，用 min_delta 判定“有效改善”）
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
                    if (
                        self.cfg.early_stopping_patience > 0
                        and epochs_without_improvement >= self.cfg.early_stopping_patience
                    ):
                        logger.info(
                            "Early stopping triggered at epoch %d after %d stale validation epochs.",
                            epoch + 1,
                            epochs_without_improvement,
                        )
                        break

            # 定期保存
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint()

        # 训练结束，保存最终模型
        self.save_checkpoint("final.pt")

        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("Training complete. Total time: %.1f s (%.1f min)", total_time, total_time / 60.0)
        logger.info("Best monitored loss: %.6f (epoch %d)", self.best_loss, self.best_epoch)
        logger.info("=" * 60)

    @torch.no_grad()
    def predict_volume(self) -> np.ndarray:
        """推理整个数据体，返回阻抗体。

        Returns
        -------
        np.ndarray
            预测阻抗体，shape ``(n_il, n_xl, n_sample)``。
        """
        self.model.eval()

        n_il = int(self.geometry["n_il"])
        n_xl = int(self.geometry["n_xl"])
        n_sample = int(self.geometry["n_sample"])

        full_loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size * 4,  # 推理可用更大 batch
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

        predictions = np.concatenate(predictions, axis=0)  # (N_valid, T)

        # 放回原始 3D 位置
        volume = np.zeros((n_il * n_xl, n_sample), dtype=np.float32)
        valid_indices = self.dataset.valid_indices
        volume[valid_indices] = predictions

        return volume.reshape(n_il, n_xl, n_sample)
