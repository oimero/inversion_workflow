"""ginn.trainer — 主训练循环。

串联 DilatedResNet1D + ForwardModel + MaskedMAELoss，执行标准的
前向传播 → 物理正演 → 损失计算 → 反向传播训练流程。
"""

from __future__ import annotations

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
        self.dataset, wavelet, self.meta = build_dataset(cfg)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
        )
        logger.info("DataLoader: %d batches/epoch", len(self.dataloader))

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
        self.criterion = GINNLoss(lambda_reg=cfg.lambda_reg)

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

    def train_one_epoch(self) -> float:
        """训练一个 epoch，返回 epoch 平均损失。"""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            x = batch["input"].to(self.device)            # (B, 2, T)
            d_obs = batch["obs"].to(self.device)           # (B, 1, T)
            mask = batch["mask"].to(self.device)            # (B, 1, T)
            lmf_raw = batch["lmf_raw"].to(self.device)     # (B, 1, T)

            # 1. 网络前向：输出阻抗残差
            residual = self.model(x)                        # (B, 1, T)

            # 2. 恢复完整阻抗
            ai = residual + lmf_raw                         # (B, 1, T)

            # 3. 物理正演
            d_syn = self.forward_model(ai)                  # (B, 1, T)

            # 4. 损失（包含波形 MAE + 残差 L2 正则化）
            loss, loss_dict = self.criterion(d_syn, d_obs, mask, residual)

            # 5. 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if (batch_idx + 1) % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "  [Epoch %d | Batch %d/%d] loss=%.6f (mae=%.6f reg=%.6f) lr=%.2e",
                    self.epoch + 1, batch_idx + 1, len(self.dataloader),
                    loss_dict["total"], loss_dict["waveform_mae"],
                    loss_dict["residual_l2"], lr,
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        return avg_loss

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        """保存模型 checkpoint。"""
        if filename is None:
            filename = f"checkpoint_epoch{self.epoch:03d}.pt"
        path = self.cfg.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_loss": self.best_loss,
                "config": self.cfg,
                "normalization": {
                    "seis_rms": self.dataset.seis_rms,
                    "lmf_scale": self.dataset.lmf_scale,
                },
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)
        return path

    def train(self) -> None:
        """完整训练流程。"""
        logger.info("=" * 60)
        logger.info("Start training: %d epochs, batch_size=%d", self.cfg.epochs, self.cfg.batch_size)
        logger.info("=" * 60)

        total_start = time.time()

        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            avg_loss = self.train_one_epoch()
            self.scheduler.step()

            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.2e  time=%.1fs",
                epoch + 1, self.cfg.epochs, avg_loss, lr, elapsed,
            )

            # 保存最优模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint("best.pt")

            # 定期保存
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint()

        # 训练结束，保存最终模型
        self.save_checkpoint("final.pt")

        total_time = time.time() - total_start
        logger.info("=" * 60)
        logger.info("Training complete. Total time: %.1f s (%.1f min)", total_time, total_time / 60.0)
        logger.info("Best loss: %.6f", self.best_loss)
        logger.info("=" * 60)

    @torch.no_grad()
    def predict_volume(self) -> np.ndarray:
        """推理整个数据体，返回阻抗体。

        Returns
        -------
        np.ndarray
            预测阻抗体，shape ``(n_il, n_xl, n_t)``。
        """
        self.model.eval()

        n_il = self.meta["n_il"]
        n_xl = self.meta["n_xl"]
        n_t = self.meta["n_t"]

        # 使用全部道（包含无效道）进行推理
        full_loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size * 4,  # 推理可用更大 batch
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        predictions = []
        indices = []

        for batch in full_loader:
            x = batch["input"].to(self.device)
            lmf_raw = batch["lmf_raw"].to(self.device)

            residual = self.model(x)
            ai = residual + lmf_raw  # (B, 1, T)

            predictions.append(ai.squeeze(1).cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)  # (N_valid, T)

        # 放回原始 3D 位置
        volume = np.zeros((n_il * n_xl, n_t), dtype=np.float32)
        valid_indices = self.dataset._valid_indices
        volume[valid_indices] = predictions

        return volume.reshape(n_il, n_xl, n_t)
