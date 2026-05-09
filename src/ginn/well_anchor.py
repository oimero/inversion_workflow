"""Reusable well log-AI anchor constraint for GINN trainers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ginn.well_prior import load_well_resolution_prior_npz, validate_well_resolution_prior

logger = logging.getLogger(__name__)

ComposeImpedanceFn = Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]


@dataclass
class WellLogAIAnchor:
    """Well log-AI supervision sampled on a GINN trace axis."""

    prior_file: Path
    lambda_weight: float
    batch_size: int
    use_prior_weight: bool
    dataset_indices: np.ndarray
    flat_indices: np.ndarray
    well_names: np.ndarray
    target_log_ai: Tensor
    mask_weight: Tensor

    @classmethod
    def build(
        cls,
        *,
        prior_file: Path | None,
        lambda_weight: float,
        batch_size: int,
        use_prior_weight: bool,
        sample_domain: str,
        n_sample: int,
        n_traces: int,
        valid_indices: np.ndarray,
    ) -> "WellLogAIAnchor | None":
        if lambda_weight <= 0.0:
            return None
        if prior_file is None:
            logger.warning("lambda_well_log_ai > 0 but well_anchor_prior_file is empty; well anchor disabled.")
            return None

        prior = load_well_resolution_prior_npz(prior_file)
        validate_well_resolution_prior(
            prior,
            sample_domain=sample_domain,
            n_sample=n_sample,
            n_traces=n_traces,
        )

        flat_to_dataset = {int(flat_idx): idx for idx, flat_idx in enumerate(valid_indices)}
        rows: list[int] = []
        dataset_indices: list[int] = []
        for row, flat_idx in enumerate(prior.flat_indices):
            dataset_idx = flat_to_dataset.get(int(flat_idx))
            if dataset_idx is None:
                continue
            mask = np.asarray(prior.well_mask[row], dtype=bool)
            ai = np.asarray(prior.well_ai[row], dtype=np.float32)
            if np.any(mask & np.isfinite(ai) & (ai > 0.0)):
                rows.append(row)
                dataset_indices.append(dataset_idx)

        if not rows:
            logger.warning("No usable well anchor traces from %s; well anchor disabled.", prior_file)
            return None

        target_ai = np.asarray(prior.well_ai[rows], dtype=np.float32)
        target_log_ai = np.zeros_like(target_ai, dtype=np.float32)
        valid = np.asarray(prior.well_mask[rows], dtype=bool) & np.isfinite(target_ai) & (target_ai > 0.0)
        target_log_ai[valid] = np.log(np.clip(target_ai[valid], 1e-6, None)).astype(np.float32)

        weights = np.ones_like(target_log_ai, dtype=np.float32)
        if use_prior_weight:
            weights = np.asarray(prior.well_weight[rows], dtype=np.float32)
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
        weights = weights * valid.astype(np.float32)

        anchor = cls(
            prior_file=Path(prior_file),
            lambda_weight=float(lambda_weight),
            batch_size=int(batch_size),
            use_prior_weight=bool(use_prior_weight),
            dataset_indices=np.asarray(dataset_indices, dtype=np.int64),
            flat_indices=np.asarray(prior.flat_indices[rows], dtype=np.int64),
            well_names=np.asarray(prior.well_names[rows]).astype(str),
            target_log_ai=torch.from_numpy(target_log_ai).float(),
            mask_weight=torch.from_numpy(weights).float(),
        )
        logger.info(
            "Well log-AI anchor enabled: wells=%d, lambda=%.3e, prior=%s",
            len(rows),
            float(lambda_weight),
            prior_file,
        )
        return anchor

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "prior_file": self.prior_file,
            "lambda_well_log_ai": self.lambda_weight,
            "batch_size": self.batch_size,
            "use_prior_weight": self.use_prior_weight,
            "n_wells": int(self.dataset_indices.size),
            "well_names": self.well_names.tolist(),
            "flat_indices": self.flat_indices.tolist(),
        }

    def sample_rows(self, *, training: bool) -> np.ndarray:
        n_wells = int(self.dataset_indices.size)
        if self.batch_size <= 0 or self.batch_size >= n_wells:
            return np.arange(n_wells, dtype=np.int64)
        if training:
            return np.random.choice(n_wells, size=self.batch_size, replace=False).astype(np.int64)
        return np.arange(min(self.batch_size, n_wells), dtype=np.int64)

    def compute_loss(
        self,
        *,
        dataset: Dataset,
        device: torch.device,
        compose_impedance: ComposeImpedanceFn,
        training: bool,
    ) -> tuple[Tensor, dict[str, float]]:
        rows = self.sample_rows(training=training)
        dataset_indices = self.dataset_indices[rows]
        items = [dataset[int(idx)] for idx in dataset_indices]
        x = torch.stack([item["input"] for item in items], dim=0).to(device)
        lfm_raw = torch.stack([item["lfm_raw"] for item in items], dim=0).to(device)
        taper_weight = torch.stack([item["taper_weight"] for item in items], dim=0).to(device)

        ai, _ = compose_impedance(x, lfm_raw, taper_weight)
        pred_log_ai = torch.log(torch.clamp(ai.squeeze(1), min=1e-6))
        target_log_ai = self.target_log_ai[rows].to(device)
        mask_weight = self.mask_weight[rows].to(device)
        valid_weight = mask_weight.to(dtype=pred_log_ai.dtype)
        denom = valid_weight.sum().clamp(min=1.0)
        raw = (F.smooth_l1_loss(pred_log_ai, target_log_ai, reduction="none") * valid_weight).sum() / denom
        term = self.lambda_weight * raw
        return term, {
            "well_log_ai": float(raw.detach().cpu().item()),
            "well_log_ai_term": float(term.detach().cpu().item()),
            "well_anchor_traces": float(rows.size),
        }


def disabled_well_anchor_summary(
    *,
    prior_file: Path | None,
    lambda_weight: float,
) -> dict[str, Any]:
    return {
        "enabled": False,
        "prior_file": prior_file,
        "lambda_well_log_ai": lambda_weight,
    }


def zero_well_anchor_metrics(device: torch.device) -> tuple[Tensor, dict[str, float]]:
    zero = torch.zeros((), device=device)
    return zero, {"well_log_ai": 0.0, "well_log_ai_term": 0.0, "well_anchor_traces": 0.0}
