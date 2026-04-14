"""ginn.trainer — 主训练循环。

串联 DilatedResNet1D + ForwardModel + MaskedMAELoss，执行标准的
前向传播 → 物理正演 → 损失计算 → 反向传播训练流程。
"""

from __future__ import annotations

import json
import logging
import math
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
        self.dataset, wavelet, self.geometry = build_dataset(cfg)

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
        logger.info("Loss domain: normalized seismic amplitude (obs pre-divided by RMS)")

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
        self.residual_tanh_scale = self._resolve_residual_tanh_scale()
        logger.info(
            "Residual bounding: tanh_scale=%.4f, zero_outside_mask=%s",
            self.residual_tanh_scale,
            self.cfg.zero_residual_outside_mask,
        )

    def _resolve_residual_tanh_scale(self) -> float:
        """解析对数扰动上界。

        若配置中显式给出 ``residual_tanh_scale``，则直接使用。
        否则根据有效区 LMF 上限和 ``residual_max_ai_offset`` 自动换算：

        ``scale = log((lmf_upper + offset) / lmf_upper)``.
        """
        if self.cfg.residual_tanh_scale is not None:
            return float(self.cfg.residual_tanh_scale)

        lmf_upper = float(self.dataset.lmf_scale)
        if lmf_upper <= 0.0:
            raise ValueError(f"LMF upper bound must be positive to auto-compute tanh scale, got {lmf_upper}.")

        offset = float(self.cfg.residual_max_ai_offset)
        target_ai_upper = lmf_upper + offset
        scale = math.log(target_ai_upper / lmf_upper)

        logger.info(
            "Auto residual tanh scale from LMF upper bound: lmf_upper=%.2f, ai_offset=%.2f, target_ai_upper=%.2f, tanh_scale=%.4f",
            lmf_upper,
            offset,
            target_ai_upper,
            scale,
        )
        return scale

    def _compose_impedance(
        self,
        x: torch.Tensor,
        lmf_raw: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """网络输出对数扰动并与 LMF 合成阻抗。

        Notes
        -----
        - 使用 ``scale * tanh(raw)`` 限制对数扰动幅度，防止 ``exp`` 放大失控。
        - 若启用 ``zero_residual_outside_mask``，则将目的层外残差压回 0，
          使层外阻抗退回到 ``lmf_raw``。
        """
        raw_residual = self.model(x)
        residual = self.residual_tanh_scale * torch.tanh(raw_residual)

        if self.cfg.zero_residual_outside_mask:
            if mask is None:
                raise ValueError("Mask is required when zero_residual_outside_mask is enabled.")
            residual = residual * mask.to(dtype=residual.dtype)

        ai = lmf_raw * torch.exp(residual)
        return ai, residual

    def train_one_epoch(self) -> float:
        """训练一个 epoch，返回 epoch 平均损失。"""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.dataloader):
            x = batch["input"].to(self.device)  # (B, 2, T)
            d_obs = batch["obs"].to(self.device)  # (B, 1, T)
            mask = batch["mask"].to(self.device)  # (B, 1, T)
            lmf_raw = batch["lmf_raw"].to(self.device)  # (B, 1, T)

            # 1. 网络前向 + 阻抗合成
            ai, residual = self._compose_impedance(x, lmf_raw, mask)

            # 2. 物理正演
            d_syn = self.forward_model(ai)  # (B, 1, T)

            # 3. 损失
            loss, loss_dict = self.criterion(d_syn, d_obs, mask, residual)

            # 4. 反向传播
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
                residual_mean = residual.abs().mean().item()
                logger.info(
                    "  [Epoch %d | Batch %d/%d] loss=%.6f (mae=%.6f reg=%.3e res=%.3e) lr=%.2e",
                    self.epoch + 1,
                    batch_idx + 1,
                    len(self.dataloader),
                    loss_dict["total"],
                    loss_dict["waveform_mae"],
                    loss_dict["reg_term"],
                    residual_mean,
                    lr,
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        return avg_loss

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
                "config": config_payload,
                "normalization": {
                    "seis_rms": self.dataset.seis_rms,
                    "lmf_scale": self.dataset.lmf_scale,
                },
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

        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            avg_loss = self.train_one_epoch()
            self.scheduler.step()

            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d  loss=%.6f  lr=%.2e  time=%.1fs",
                epoch + 1,
                self.cfg.epochs,
                avg_loss,
                lr,
                elapsed,
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
            预测阻抗体，shape ``(n_il, n_xl, n_sample)``。
        """
        self.model.eval()

        n_il = int(self.geometry["n_il"])
        n_xl = int(self.geometry["n_xl"])
        n_sample = int(self.geometry["n_sample"])

        # 使用全部道（包含无效道）进行推理
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
            lmf_raw = batch["lmf_raw"].to(self.device)
            mask = batch["mask"].to(self.device)

            ai, _ = self._compose_impedance(x, lmf_raw, mask)

            predictions.append(ai.squeeze(1).cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)  # (N_valid, T)

        # 放回原始 3D 位置
        volume = np.zeros((n_il * n_xl, n_sample), dtype=np.float32)
        valid_indices = self.dataset._valid_indices
        volume[valid_indices] = predictions

        return volume.reshape(n_il, n_xl, n_sample)
