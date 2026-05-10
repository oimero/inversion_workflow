"""Reusable well log-AI anchor constraint for GINN trainers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from ginn.well_prior import load_well_resolution_prior_npz, validate_well_resolution_prior
from wtie.processing.spectral import apply_butter_lowpass_filter

logger = logging.getLogger(__name__)

ComposeImpedanceFn = Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]


@dataclass
class WellLogAIAnchor:
    """Well log-AI supervision sampled on a GINN trace axis.

    Supports optional lowpass filtering of the target AI (to constrain only the
    low-frequency baseline) and optional spatial neighbourhood (to spread the
    constraint across traces near the well with distance-decay weights).
    """

    prior_file: Path
    lambda_weight: float
    batch_size: int
    use_prior_weight: bool
    dataset_indices: np.ndarray
    flat_indices: np.ndarray
    well_names: np.ndarray
    target_log_ai: Tensor
    mask_weight: Tensor

    # ── neighbourhood ──
    neighborhood_radius: int = 0
    neighbor_inputs: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_lfm_raw: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_taper: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_weights: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_ranges: list[tuple[int, int]] = field(default_factory=list)
    lowpass_cutoff_wavelength_m: float | None = None
    lowpass_cutoff_hz: float | None = None

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
        dataset: Dataset | None = None,
        neighborhood_radius: int = 0,
        geometry: dict | None = None,
        lowpass_cutoff_wavelength_m: float | None = None,
        lowpass_cutoff_hz: float | None = None,
        lowpass_filter_order: int = 6,
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

        # ── lowpass filter target AI (before log) ──
        actual_cutoff: float | None = None
        cutoff_is_wavelength: bool = True
        if lowpass_cutoff_wavelength_m is not None and lowpass_cutoff_wavelength_m > 0.0:
            actual_cutoff = float(lowpass_cutoff_wavelength_m)
            cutoff_is_wavelength = True
        elif lowpass_cutoff_hz is not None and lowpass_cutoff_hz > 0.0:
            actual_cutoff = float(lowpass_cutoff_hz)
            cutoff_is_wavelength = False

        if actual_cutoff is not None:
            dz = float(np.median(np.diff(prior.samples)))
            if dz > 0.0:
                fs = 1.0 / dz
                highcut = (1.0 / actual_cutoff) if cutoff_is_wavelength else actual_cutoff
                order = int(lowpass_filter_order)
                pad = max(1, int(np.ceil(3.0 * order * dz)))
                for row in range(target_ai.shape[0]):
                    row_mask = np.asarray(prior.well_mask[rows][row], dtype=bool)
                    row_valid = row_mask & np.isfinite(target_ai[row]) & (target_ai[row] > 0.0)
                    if row_valid.sum() < 2:
                        continue
                    values = target_ai[row].astype(np.float64)
                    padded = np.pad(values, (pad, pad), mode="reflect")
                    filtered = apply_butter_lowpass_filter(
                        padded, highcut, fs, order=order, zero_phase=True,
                    )
                    target_ai[row] = filtered[pad:pad + values.size].astype(np.float32)
                cutoff_label = f"{actual_cutoff:.0f} m" if cutoff_is_wavelength else f"{actual_cutoff:.1f} Hz"
                logger.info(
                    "Well anchor lowpass: cutoff=%s (%.4f cycles/unit), order=%d, pad=%d samples",
                    cutoff_label, highcut, order, pad,
                )

        # ── build log-AI target ──
        target_log_ai = np.zeros_like(target_ai, dtype=np.float32)
        valid = np.asarray(prior.well_mask[rows], dtype=bool) & np.isfinite(target_ai) & (target_ai > 0.0)
        target_log_ai[valid] = np.log(np.clip(target_ai[valid], 1e-6, None)).astype(np.float32)

        weights = np.ones_like(target_log_ai, dtype=np.float32)
        if use_prior_weight:
            weights = np.asarray(prior.well_weight[rows], dtype=np.float32)
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
        weights = weights * valid.astype(np.float32)

        # ── neighbourhood precomputation ──
        n_wells = len(rows)
        dataset_indices_arr = np.asarray(dataset_indices, dtype=np.int64)
        neighbor_inputs = torch.empty(0)
        neighbor_lfm_raw = torch.empty(0)
        neighbor_taper = torch.empty(0)
        neighbor_weights_flat = torch.empty(0)
        neighbor_ranges: list[tuple[int, int]] = []

        if neighborhood_radius > 0:
            if geometry is None:
                raise ValueError("geometry dict is required when neighborhood_radius > 0.")
            n_il = int(geometry["n_il"])
            n_xl = int(geometry["n_xl"])
            il_step = float(geometry.get("inline_step", 1.0))
            xl_step = float(geometry.get("xline_step", 1.0))
            phys_radius = float(neighborhood_radius) * max(il_step, xl_step)
            sigma = max(phys_radius / 2.0, 0.5 * max(il_step, xl_step))

            all_inputs: list[Tensor] = []
            all_lfm: list[Tensor] = []
            all_taper: list[Tensor] = []
            all_w: list[Tensor] = []

            for w in range(n_wells):
                flat_ref = int(prior.flat_indices[rows[w]])
                il_ref = flat_ref // n_xl
                xl_ref = flat_ref % n_xl
                candidates: list[tuple[float, int, float]] = []
                for dil in range(-neighborhood_radius, neighborhood_radius + 1):
                    for dxl in range(-neighborhood_radius, neighborhood_radius + 1):
                        dist = math.hypot(dil * il_step, dxl * xl_step)
                        if dist > phys_radius + 1e-8:
                            continue
                        il = il_ref + dil
                        xl = xl_ref + dxl
                        if il < 0 or il >= n_il or xl < 0 or xl >= n_xl:
                            continue
                        flat = il * n_xl + xl
                        ds_idx = flat_to_dataset.get(int(flat))
                        if ds_idx is None:
                            continue
                        dw = 1.0 if dist < 1e-8 else math.exp(-dist ** 2 / (2.0 * sigma ** 2))
                        candidates.append((dist, ds_idx, dw))
                candidates.sort(key=lambda t: t[0])

                inputs_w = []
                lfm_w = []
                taper_w = []
                weights_w = []
                for _, ds_idx, dw in candidates:
                    item = dataset[int(ds_idx)]  # type: ignore[index]
                    inputs_w.append(torch.from_numpy(item["input"]))
                    lfm_w.append(torch.from_numpy(item["lfm_raw"]))
                    taper_w.append(torch.from_numpy(item["taper_weight"]))
                    weights_w.append(dw)

                if inputs_w:
                    all_inputs.append(torch.stack(inputs_w))
                    all_lfm.append(torch.stack(lfm_w))
                    all_taper.append(torch.stack(taper_w))
                    all_w.append(torch.tensor(weights_w, dtype=torch.float32))
                    neighbor_ranges.append((neighbor_weights_flat.numel(), neighbor_weights_flat.numel() + len(inputs_w)))
                else:
                    neighbor_ranges.append((neighbor_weights_flat.numel(), neighbor_weights_flat.numel()))

                if all_w:
                    neighbor_weights_flat = torch.cat([neighbor_weights_flat] + [all_w[-1]]) if neighbor_weights_flat.numel() > 0 else all_w[-1]

            if all_inputs:
                neighbor_inputs = torch.cat(all_inputs, dim=0)
                neighbor_lfm_raw = torch.cat(all_lfm, dim=0)
                neighbor_taper = torch.cat(all_taper, dim=0)

            max_nbr = max((e - s) for s, e in neighbor_ranges) if neighbor_ranges else 0
            logger.info(
                "Well anchor neighbourhood: radius=%d grid (%.0f step-units), sigma=%.1f, max_neighbors=%d, total_nbr=%d",
                neighborhood_radius, phys_radius, sigma, max_nbr, int(neighbor_weights_flat.numel()),
            )
        else:
            if dataset is None:
                raise ValueError("dataset is required for well anchor precomputation.")
            for w in range(n_wells):
                ds_idx = dataset_indices_arr[w]
                item = dataset[int(ds_idx)]
                neighbor_inputs = torch.cat(
                    [neighbor_inputs, torch.from_numpy(item["input"]).unsqueeze(0)]
                ) if neighbor_inputs.numel() > 0 else torch.from_numpy(item["input"]).unsqueeze(0)
                neighbor_lfm_raw = torch.cat(
                    [neighbor_lfm_raw, torch.from_numpy(item["lfm_raw"]).unsqueeze(0)]
                ) if neighbor_lfm_raw.numel() > 0 else torch.from_numpy(item["lfm_raw"]).unsqueeze(0)
                neighbor_taper = torch.cat(
                    [neighbor_taper, torch.from_numpy(item["taper_weight"]).unsqueeze(0)]
                ) if neighbor_taper.numel() > 0 else torch.from_numpy(item["taper_weight"]).unsqueeze(0)
                neighbor_ranges.append((w, w + 1))
            neighbor_weights_flat = torch.ones(n_wells, dtype=torch.float32)

        anchor = cls(
            prior_file=Path(prior_file),
            lambda_weight=float(lambda_weight),
            batch_size=int(batch_size),
            use_prior_weight=bool(use_prior_weight),
            dataset_indices=dataset_indices_arr,
            flat_indices=np.asarray(prior.flat_indices[rows], dtype=np.int64),
            well_names=np.asarray(prior.well_names[rows]).astype(str),
            target_log_ai=torch.from_numpy(target_log_ai),
            mask_weight=torch.from_numpy(weights),
            neighborhood_radius=int(neighborhood_radius),
            neighbor_inputs=neighbor_inputs,
            neighbor_lfm_raw=neighbor_lfm_raw,
            neighbor_taper=neighbor_taper,
            neighbor_weights=neighbor_weights_flat,
            neighbor_ranges=neighbor_ranges,
            lowpass_cutoff_wavelength_m=actual_cutoff if cutoff_is_wavelength else None,
            lowpass_cutoff_hz=actual_cutoff if not cutoff_is_wavelength else None,
        )
        logger.info(
            "Well log-AI anchor enabled: wells=%d, lambda=%.3e, prior=%s",
            n_wells,
            float(lambda_weight),
            prior_file,
        )
        return anchor

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": True,
            "prior_file": self.prior_file,
            "lambda_well_log_ai": self.lambda_weight,
            "batch_size": self.batch_size,
            "use_prior_weight": self.use_prior_weight,
            "n_wells": int(self.dataset_indices.size),
            "well_names": self.well_names.tolist(),
            "flat_indices": self.flat_indices.tolist(),
            "neighborhood_radius": self.neighborhood_radius,
            "lowpass_cutoff_wavelength_m": self.lowpass_cutoff_wavelength_m,
            "lowpass_cutoff_hz": self.lowpass_cutoff_hz,
        }
        if self.neighbor_ranges:
            result["max_neighbors"] = max(e - s for s, e in self.neighbor_ranges)
        return result

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
        if rows.size == 0:
            return zero_well_anchor_metrics(device)

        # ── gather neighbour slices for selected wells ──
        slices = [self.neighbor_ranges[r] for r in rows]
        total_nbr = sum(e - s for s, e in slices)
        if total_nbr == 0:
            return zero_well_anchor_metrics(device)

        # ── build flat batch via index cat ──
        all_idx = torch.cat([torch.arange(s, e) for s, e in slices])
        x = self.neighbor_inputs[all_idx].to(device)
        lfm_raw = self.neighbor_lfm_raw[all_idx].to(device)
        taper_weight = self.neighbor_taper[all_idx].to(device)
        w_flat = self.neighbor_weights[all_idx].to(device)

        ai, _ = compose_impedance(x, lfm_raw, taper_weight)
        pred_log_ai = torch.log(torch.clamp(ai.squeeze(1), min=1e-6))  # (total_nbr, n_sample)

        # ── per-well loss with distance weighting ──
        total_loss = torch.zeros((), device=device)
        total_denom = torch.zeros((), device=device)
        cursor = 0
        for b, r in enumerate(rows):
            n_valid = slices[b][1] - slices[b][0]
            if n_valid == 0:
                continue

            pred = pred_log_ai[cursor:cursor + n_valid]
            target = self.target_log_ai[r].expand(n_valid, -1).to(device)
            mask = self.mask_weight[r].to(device)
            w = w_flat[cursor:cursor + n_valid]

            raw = F.smooth_l1_loss(pred, target, reduction="none")  # (n_valid, n_sample)
            weighted = raw * w.unsqueeze(-1) * mask.unsqueeze(0)
            total_loss = total_loss + weighted.sum()
            total_denom = total_denom + (w.unsqueeze(-1) * mask.unsqueeze(0)).sum()
            cursor += n_valid

        raw_loss = total_loss / total_denom.clamp(min=1.0)
        term = self.lambda_weight * raw_loss
        return term, {
            "well_log_ai": float(raw_loss.detach().cpu().item()),
            "well_log_ai_term": float(term.detach().cpu().item()),
            "well_anchor_traces": float(rows.size),
            "well_anchor_neighbors": float(total_nbr),
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
    return zero, {"well_log_ai": 0.0, "well_log_ai_term": 0.0, "well_anchor_traces": 0.0, "well_anchor_neighbors": 0.0}
