"""Log-AI anchor schema and runtime constraint for GINN trainers.

This module merges the former ``log_ai_anchor`` (NPZ schema layer) and
``well_anchor`` (runtime loss layer) into a single public entrypoint:
``from ginn.anchor import ...``.
"""

from __future__ import annotations

import json
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

from cup.utils.io import to_json_compatible

logger = logging.getLogger(__name__)

# Schema constants.

SCHEMA_VERSION = "ginn_log_ai_anchor_v1"
VALID_ANCHOR_TYPES = {"well", "facies_control"}
VALID_SAMPLE_DOMAINS = {"time", "depth"}
VALID_SAMPLE_UNITS = {"s", "m"}


# Schema dataclass.


@dataclass(frozen=True)
class LogAIAnchorBundle:
    """Target log-AI traces sampled on a GINN trace axis.

    Unlike the well-resolution prior used by the enhancement stage, this bundle
    contains no residual or LFM-difference fields. It is only a first-stage
    supervision target: where, over which samples, and with what weight the
    predicted AI should be anchored.
    """

    sample_domain: str
    sample_unit: str
    samples: np.ndarray
    flat_indices: np.ndarray
    target_ai: np.ndarray
    target_log_ai: np.ndarray
    anchor_mask: np.ndarray
    anchor_weight: np.ndarray
    anchor_names: np.ndarray
    anchor_types: np.ndarray
    inline: np.ndarray
    xline: np.ndarray
    summary: dict[str, Any]
    metadata: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @property
    def n_anchors(self) -> int:
        return int(self.flat_indices.size)

    @property
    def n_sample(self) -> int:
        return int(self.samples.size)


# Schema helpers.


def build_log_ai_anchor_bundle(
    *,
    sample_domain: str,
    sample_unit: str,
    samples: np.ndarray,
    flat_indices: np.ndarray,
    target_ai: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_weight: np.ndarray,
    anchor_names: np.ndarray,
    anchor_types: np.ndarray,
    inline: np.ndarray,
    xline: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> LogAIAnchorBundle:
    """Create and validate a log-AI anchor bundle from positive AI targets."""
    target_ai_arr = np.asarray(target_ai, dtype=np.float32)
    mask = np.asarray(anchor_mask, dtype=bool)
    target_log_ai = np.zeros_like(target_ai_arr, dtype=np.float32)
    valid = mask & np.isfinite(target_ai_arr) & (target_ai_arr > 0.0)
    target_log_ai[valid] = np.log(np.clip(target_ai_arr[valid], 1e-6, None)).astype(np.float32)
    summary = summarize_log_ai_anchor(target_log_ai, mask, anchor_weight=anchor_weight)
    bundle = LogAIAnchorBundle(
        sample_domain=sample_domain,
        sample_unit=sample_unit,
        samples=np.asarray(samples, dtype=np.float32),
        flat_indices=np.asarray(flat_indices, dtype=np.int64),
        target_ai=target_ai_arr,
        target_log_ai=target_log_ai,
        anchor_mask=mask,
        anchor_weight=np.asarray(anchor_weight, dtype=np.float32),
        anchor_names=np.asarray(anchor_names).astype(str),
        anchor_types=np.asarray(anchor_types).astype(str),
        inline=np.asarray(inline, dtype=np.float32),
        xline=np.asarray(xline, dtype=np.float32),
        summary=summary,
        metadata={} if metadata is None else dict(metadata),
    )
    validate_log_ai_anchor(bundle)
    return bundle


def load_log_ai_anchor_npz(path: str | Path) -> LogAIAnchorBundle:
    """Load a ``ginn_log_ai_anchor_v1`` NPZ file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sample_domain",
            "sample_unit",
            "samples",
            "flat_indices",
            "target_ai",
            "target_log_ai",
            "anchor_mask",
            "anchor_weight",
            "anchor_names",
            "anchor_types",
            "inline",
            "xline",
            "summary_json",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Log-AI anchor NPZ is missing keys: {sorted(missing)}")

        bundle = LogAIAnchorBundle(
            schema_version=_as_str(data["schema_version"]),
            sample_domain=_as_str(data["sample_domain"]),
            sample_unit=_as_str(data["sample_unit"]),
            samples=np.asarray(data["samples"], dtype=np.float32),
            flat_indices=np.asarray(data["flat_indices"], dtype=np.int64),
            target_ai=np.asarray(data["target_ai"], dtype=np.float32),
            target_log_ai=np.asarray(data["target_log_ai"], dtype=np.float32),
            anchor_mask=np.asarray(data["anchor_mask"], dtype=bool),
            anchor_weight=np.asarray(data["anchor_weight"], dtype=np.float32),
            anchor_names=np.asarray(data["anchor_names"]).astype(str),
            anchor_types=np.asarray(data["anchor_types"]).astype(str),
            inline=np.asarray(data["inline"], dtype=np.float32),
            xline=np.asarray(data["xline"], dtype=np.float32),
            summary=_json_to_dict(data["summary_json"]),
            metadata=_json_to_dict(data["metadata_json"]),
        )

    validate_log_ai_anchor(bundle)
    return bundle


def save_log_ai_anchor_npz(path: str | Path, bundle: LogAIAnchorBundle) -> Path:
    """Validate and save a log-AI anchor bundle as compressed NPZ."""
    validate_log_ai_anchor(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(bundle.schema_version),
        sample_domain=np.asarray(bundle.sample_domain),
        sample_unit=np.asarray(bundle.sample_unit),
        samples=np.asarray(bundle.samples, dtype=np.float32),
        flat_indices=np.asarray(bundle.flat_indices, dtype=np.int64),
        target_ai=np.asarray(bundle.target_ai, dtype=np.float32),
        target_log_ai=np.asarray(bundle.target_log_ai, dtype=np.float32),
        anchor_mask=np.asarray(bundle.anchor_mask, dtype=bool),
        anchor_weight=np.asarray(bundle.anchor_weight, dtype=np.float32),
        anchor_names=np.asarray(bundle.anchor_names).astype(str),
        anchor_types=np.asarray(bundle.anchor_types).astype(str),
        inline=np.asarray(bundle.inline, dtype=np.float32),
        xline=np.asarray(bundle.xline, dtype=np.float32),
        summary_json=np.asarray(json.dumps(to_json_compatible(bundle.summary), ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(to_json_compatible(bundle.metadata), ensure_ascii=False)),
    )
    return path


def validate_log_ai_anchor(
    bundle: LogAIAnchorBundle,
    sample_domain: str | None = None,
    n_sample: int | None = None,
    n_traces: int | None = None,
) -> None:
    """Validate schema, shapes, finite values, and workflow compatibility."""
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported log-AI anchor schema_version={bundle.schema_version!r}.")
    if bundle.sample_domain not in VALID_SAMPLE_DOMAINS:
        raise ValueError(f"sample_domain must be one of {sorted(VALID_SAMPLE_DOMAINS)}, got {bundle.sample_domain!r}.")
    if bundle.sample_unit not in VALID_SAMPLE_UNITS:
        raise ValueError(f"sample_unit must be one of {sorted(VALID_SAMPLE_UNITS)}, got {bundle.sample_unit!r}.")
    if sample_domain is not None and bundle.sample_domain != sample_domain:
        raise ValueError(f"Log-AI anchor is for {bundle.sample_domain!r}, expected {sample_domain!r}.")

    samples = np.asarray(bundle.samples)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty 1D array.")
    if samples.size > 1 and np.any(np.diff(samples.astype(np.float64)) <= 0.0):
        raise ValueError("samples must be strictly increasing.")
    if n_sample is not None and samples.size != int(n_sample):
        raise ValueError(f"Log-AI anchor n_sample={samples.size} does not match expected {int(n_sample)}.")

    flat_indices = np.asarray(bundle.flat_indices)
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be a 1D array.")
    n_anchors = int(flat_indices.size)
    if np.unique(flat_indices).size != n_anchors:
        raise ValueError("Duplicate flat_indices are not supported in log-AI anchor v1.")
    if n_traces is not None and n_anchors:
        if flat_indices.min() < 0 or flat_indices.max() >= int(n_traces):
            raise ValueError(
                f"flat_indices must be within [0, {int(n_traces)}), "
                f"got min={int(flat_indices.min())}, max={int(flat_indices.max())}."
            )

    expected_2d = (n_anchors, samples.size)
    for name in ("target_ai", "target_log_ai", "anchor_mask", "anchor_weight"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != expected_2d:
            raise ValueError(f"{name} shape {value.shape} does not match expected {expected_2d}.")

    for name in ("anchor_names", "anchor_types", "inline", "xline"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != (n_anchors,):
            raise ValueError(f"{name} shape {value.shape} does not match expected {(n_anchors,)}.")

    unknown_types = set(np.asarray(bundle.anchor_types).astype(str)) - VALID_ANCHOR_TYPES
    if unknown_types:
        raise ValueError(f"anchor_types contains unsupported values: {sorted(unknown_types)}.")

    mask = np.asarray(bundle.anchor_mask, dtype=bool)
    weight = np.asarray(bundle.anchor_weight, dtype=np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError("anchor_weight must be finite.")
    if np.any(weight < 0.0):
        raise ValueError("anchor_weight must be non-negative.")

    target_ai = np.asarray(bundle.target_ai, dtype=np.float64)
    target_log_ai = np.asarray(bundle.target_log_ai, dtype=np.float64)
    if np.any(~np.isfinite(target_ai[mask])) or np.any(target_ai[mask] <= 0.0):
        raise ValueError("target_ai must be finite and positive inside anchor_mask.")
    if np.any(~np.isfinite(target_log_ai[mask])):
        raise ValueError("target_log_ai must be finite inside anchor_mask.")


def summarize_log_ai_anchor(
    target_log_ai: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    anchor_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute lightweight global statistics for a log-AI anchor bundle."""
    target = np.asarray(target_log_ai, dtype=np.float64)
    mask = np.asarray(anchor_mask, dtype=bool)
    if target.shape != mask.shape:
        raise ValueError(f"target shape {target.shape} does not match mask shape {mask.shape}.")
    valid = mask & np.isfinite(target)
    summary: dict[str, Any] = {
        "n_anchors": int(target.shape[0]) if target.ndim == 2 else 0,
        "n_valid_samples": int(np.count_nonzero(valid)),
        "target_log_ai": _robust_stats(target[valid]),
    }
    if anchor_weight is not None:
        weight = np.asarray(anchor_weight, dtype=np.float64)
        if weight.shape != target.shape:
            raise ValueError("anchor_weight shape must match target_log_ai shape.")
        summary["weight"] = _robust_stats(weight[valid])
    return summary


def _robust_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _as_str(value: object) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.reshape(-1)[0])


def _json_to_dict(value: object) -> dict[str, Any]:
    text = _as_str(value)
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON scalar must decode to an object.")
    return parsed


# Runtime constraint.

ComposeImpedanceFn = Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]


def _as_tensor(value: np.ndarray | Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.from_numpy(value)


@dataclass
class LogAIAnchor:
    """Log-AI supervision sampled on a GINN trace axis.

    Supports optional spatial neighbourhood to spread an anchor across nearby
    traces with distance-decay weights. The anchor file is responsible for any
    frequency split; this loss uses the stored target_log_ai as-is.
    """

    anchor_file: Path
    lambda_weight: float
    batch_size: int
    use_anchor_weight: bool
    dataset_indices: np.ndarray
    flat_indices: np.ndarray
    anchor_names: np.ndarray
    anchor_types: np.ndarray
    target_log_ai: Tensor
    mask_weight: Tensor

    # Neighbourhood.
    neighborhood_radius: int = 0
    neighbor_inputs: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_lfm_raw: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_taper: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_weights: Tensor = field(default_factory=lambda: torch.empty(0))
    neighbor_ranges: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        anchor_file: Path | None,
        lambda_weight: float,
        batch_size: int,
        use_anchor_weight: bool,
        sample_domain: str,
        n_sample: int,
        n_traces: int,
        valid_indices: np.ndarray,
        dataset: Dataset | None = None,
        neighborhood_radius: int = 0,
        geometry: dict | None = None,
    ) -> "LogAIAnchor | None":
        if lambda_weight <= 0.0:
            return None
        if anchor_file is None:
            logger.warning("lambda_log_ai_anchor > 0 but log_ai_anchor_file is empty; log-AI anchor disabled.")
            return None

        anchor_bundle = load_log_ai_anchor_npz(anchor_file)
        validate_log_ai_anchor(
            anchor_bundle,
            sample_domain=sample_domain,
            n_sample=n_sample,
            n_traces=n_traces,
        )

        flat_to_dataset = {int(flat_idx): idx for idx, flat_idx in enumerate(valid_indices)}
        rows: list[int] = []
        dataset_indices: list[int] = []
        for row, flat_idx in enumerate(anchor_bundle.flat_indices):
            dataset_idx = flat_to_dataset.get(int(flat_idx))
            if dataset_idx is None:
                continue
            mask = np.asarray(anchor_bundle.anchor_mask[row], dtype=bool)
            ai = np.asarray(anchor_bundle.target_ai[row], dtype=np.float32)
            if np.any(mask & np.isfinite(ai) & (ai > 0.0)):
                rows.append(row)
                dataset_indices.append(dataset_idx)

        if not rows:
            logger.warning("No usable log-AI anchor traces from %s; anchor disabled.", anchor_file)
            return None

        target_log_ai = np.asarray(anchor_bundle.target_log_ai[rows], dtype=np.float32)
        valid = np.asarray(anchor_bundle.anchor_mask[rows], dtype=bool) & np.isfinite(target_log_ai)

        weights = np.ones_like(target_log_ai, dtype=np.float32)
        if use_anchor_weight:
            weights = np.asarray(anchor_bundle.anchor_weight[rows], dtype=np.float32)
            weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0).astype(np.float32)
        weights = weights * valid.astype(np.float32)

        # Neighbourhood precomputation.
        n_anchors = len(rows)
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

            for w in range(n_anchors):
                flat_ref = int(anchor_bundle.flat_indices[rows[w]])
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
                        dw = 1.0 if dist < 1e-8 else math.exp(-(dist**2) / (2.0 * sigma**2))
                        candidates.append((dist, ds_idx, dw))
                candidates.sort(key=lambda t: t[0])

                inputs_w = []
                lfm_w = []
                taper_w = []
                weights_w = []
                for _, ds_idx, dw in candidates:
                    item = dataset[int(ds_idx)]  # type: ignore[index]
                    inputs_w.append(_as_tensor(item["input"]))
                    lfm_w.append(_as_tensor(item["lfm_raw"]))
                    taper_w.append(_as_tensor(item["taper_weight"]))
                    weights_w.append(dw)

                if inputs_w:
                    all_inputs.append(torch.stack(inputs_w))
                    all_lfm.append(torch.stack(lfm_w))
                    all_taper.append(torch.stack(taper_w))
                    all_w.append(torch.tensor(weights_w, dtype=torch.float32))
                    neighbor_ranges.append(
                        (neighbor_weights_flat.numel(), neighbor_weights_flat.numel() + len(inputs_w))
                    )
                else:
                    neighbor_ranges.append((neighbor_weights_flat.numel(), neighbor_weights_flat.numel()))

                if all_w:
                    neighbor_weights_flat = (
                        torch.cat([neighbor_weights_flat] + [all_w[-1]])
                        if neighbor_weights_flat.numel() > 0
                        else all_w[-1]
                    )

            if all_inputs:
                neighbor_inputs = torch.cat(all_inputs, dim=0)
                neighbor_lfm_raw = torch.cat(all_lfm, dim=0)
                neighbor_taper = torch.cat(all_taper, dim=0)

            max_nbr = max((e - s) for s, e in neighbor_ranges) if neighbor_ranges else 0
            logger.info(
                "Log-AI anchor neighbourhood: radius=%d grid (%.0f step-units), sigma=%.1f, max_neighbors=%d, total_nbr=%d",
                neighborhood_radius,
                phys_radius,
                sigma,
                max_nbr,
                int(neighbor_weights_flat.numel()),
            )
        else:
            if dataset is None:
                raise ValueError("dataset is required for log-AI anchor precomputation.")
            for w in range(n_anchors):
                ds_idx = dataset_indices_arr[w]
                item = dataset[int(ds_idx)]
                inp = _as_tensor(item["input"]).unsqueeze(0)
                lfm = _as_tensor(item["lfm_raw"]).unsqueeze(0)
                tap = _as_tensor(item["taper_weight"]).unsqueeze(0)
                neighbor_inputs = torch.cat([neighbor_inputs, inp]) if neighbor_inputs.numel() > 0 else inp
                neighbor_lfm_raw = torch.cat([neighbor_lfm_raw, lfm]) if neighbor_lfm_raw.numel() > 0 else lfm
                neighbor_taper = torch.cat([neighbor_taper, tap]) if neighbor_taper.numel() > 0 else tap
                neighbor_ranges.append((w, w + 1))
            neighbor_weights_flat = torch.ones(n_anchors, dtype=torch.float32)

        anchor = cls(
            anchor_file=Path(anchor_file),
            lambda_weight=float(lambda_weight),
            batch_size=int(batch_size),
            use_anchor_weight=bool(use_anchor_weight),
            dataset_indices=dataset_indices_arr,
            flat_indices=np.asarray(anchor_bundle.flat_indices[rows], dtype=np.int64),
            anchor_names=np.asarray(anchor_bundle.anchor_names[rows]).astype(str),
            anchor_types=np.asarray(anchor_bundle.anchor_types[rows]).astype(str),
            target_log_ai=torch.from_numpy(target_log_ai),
            mask_weight=torch.from_numpy(weights),
            neighborhood_radius=int(neighborhood_radius),
            neighbor_inputs=neighbor_inputs,
            neighbor_lfm_raw=neighbor_lfm_raw,
            neighbor_taper=neighbor_taper,
            neighbor_weights=neighbor_weights_flat,
            neighbor_ranges=neighbor_ranges,
        )
        logger.info(
            "Log-AI anchor enabled: anchors=%d, lambda=%.3e, file=%s",
            n_anchors,
            float(lambda_weight),
            anchor_file,
        )
        return anchor

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enabled": True,
            "anchor_file": self.anchor_file,
            "lambda_log_ai_anchor": self.lambda_weight,
            "batch_size": self.batch_size,
            "use_anchor_weight": self.use_anchor_weight,
            "n_anchors": int(self.dataset_indices.size),
            "anchor_names": self.anchor_names.tolist(),
            "anchor_types": self.anchor_types.tolist(),
            "flat_indices": self.flat_indices.tolist(),
            "neighborhood_radius": self.neighborhood_radius,
        }
        if self.neighbor_ranges:
            result["max_neighbors"] = max(e - s for s, e in self.neighbor_ranges)
        return result

    def sample_rows(self, *, training: bool) -> np.ndarray:
        n_anchors = int(self.dataset_indices.size)
        if self.batch_size <= 0 or self.batch_size >= n_anchors:
            return np.arange(n_anchors, dtype=np.int64)
        if training:
            return np.random.choice(n_anchors, size=self.batch_size, replace=False).astype(np.int64)
        return np.arange(min(self.batch_size, n_anchors), dtype=np.int64)

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
            return zero_log_ai_anchor_metrics(device)

        # Gather neighbour slices for selected anchors.
        slices = [self.neighbor_ranges[r] for r in rows]
        total_nbr = sum(e - s for s, e in slices)
        if total_nbr == 0:
            return zero_log_ai_anchor_metrics(device)

        # Build flat batch via index cat.
        all_idx = torch.cat([torch.arange(s, e) for s, e in slices])
        x = self.neighbor_inputs[all_idx].to(device)
        lfm_raw = self.neighbor_lfm_raw[all_idx].to(device)
        taper_weight = self.neighbor_taper[all_idx].to(device)
        w_flat = self.neighbor_weights[all_idx].to(device)

        ai, _ = compose_impedance(x, lfm_raw, taper_weight)
        pred_log_ai = torch.log(torch.clamp(ai.squeeze(1), min=1e-6))  # (total_nbr, n_sample)

        # Per-anchor loss with distance weighting.
        total_loss = torch.zeros((), device=device)
        total_denom = torch.zeros((), device=device)
        total_sample_count = torch.zeros((), device=device)
        cursor = 0
        for b, r in enumerate(rows):
            n_valid = slices[b][1] - slices[b][0]
            if n_valid == 0:
                continue

            pred = pred_log_ai[cursor : cursor + n_valid]
            target = self.target_log_ai[r].expand(n_valid, -1).to(device)
            mask = self.mask_weight[r].to(device)
            w = w_flat[cursor : cursor + n_valid]

            raw = F.smooth_l1_loss(pred, target, reduction="none")  # (n_valid, n_sample)
            weighted = raw * w.unsqueeze(-1) * mask.unsqueeze(0)
            total_loss = total_loss + weighted.sum()
            sample_weight = w.unsqueeze(-1) * mask.unsqueeze(0)
            total_denom = total_denom + sample_weight.sum()
            total_sample_count = total_sample_count + (sample_weight > 0.0).sum()
            cursor += n_valid

        raw_loss = total_loss / total_denom.clamp(min=1.0)
        term = self.lambda_weight * raw_loss
        return term, {
            "log_ai_anchor": float(raw_loss.detach().cpu().item()),
            "log_ai_anchor_term": float(term.detach().cpu().item()),
            "log_ai_anchor_traces": float(rows.size),
            "log_ai_anchor_neighbors": float(total_nbr),
            "anchor_sample_count": float(total_sample_count.detach().cpu().item()),
        }


def disabled_log_ai_anchor_summary(
    *,
    anchor_file: Path | None,
    lambda_weight: float,
) -> dict[str, Any]:
    return {
        "enabled": False,
        "anchor_file": anchor_file,
        "lambda_log_ai_anchor": lambda_weight,
    }


def zero_log_ai_anchor_metrics(device: torch.device) -> tuple[Tensor, dict[str, float]]:
    zero = torch.zeros((), device=device)
    return zero, {
        "log_ai_anchor": 0.0,
        "log_ai_anchor_term": 0.0,
        "log_ai_anchor_traces": 0.0,
        "log_ai_anchor_neighbors": 0.0,
        "anchor_sample_count": 0.0,
    }
