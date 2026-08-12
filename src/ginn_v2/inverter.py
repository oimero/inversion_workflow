"""Deterministic body-inference seam used by review and later delivery."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ginn_v2.adapters import DomainAdapter
from ginn_v2.contracts import CommonObservationBatch
from ginn_v2.data import PatchKey, PatchReader
from ginn_v2.model import CenterTraceBodyNet


@dataclass(frozen=True)
class BodyResult:
    keys: tuple[PatchKey, ...]
    body_log_ai: Tensor
    synthetic_seismic: Tensor
    valid_mask: Tensor


class BodyInverter:
    """Small inference interface hiding patch batching and the domain adapter."""

    def __init__(
        self,
        model: CenterTraceBodyNet,
        reader: PatchReader,
        adapter: DomainAdapter,
        *,
        body_smoothing_fwhm_m: float,
        device: torch.device | str = "cpu",
        batch_size: int = 8,
    ) -> None:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.model = model.to(device)
        self.reader = reader
        self.adapter = adapter
        self.body_smoothing_fwhm_m = float(body_smoothing_fwhm_m)
        if self.body_smoothing_fwhm_m <= 0.0:
            raise ValueError("body_smoothing_fwhm_m must be positive.")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

    def _common(self, batch) -> CommonObservationBatch:
        return CommonObservationBatch(
            sample_axis=self.reader.sample_axis,
            observed_seismic=batch.observed_seismic,
            observed_valid_mask=batch.observed_valid_mask,
            lfm_log_ai=batch.lfm_log_ai,
            lfm_valid_mask=batch.lfm_valid_mask,
            xy_m=batch.xy_m,
            domain_extras=batch.domain_extras,
        )

    @torch.no_grad()
    def predict(self, keys: tuple[PatchKey, ...] | list[PatchKey], *, center_visible: bool = True) -> BodyResult:
        selected = tuple(keys)
        if not selected:
            raise ValueError("BodyInverter.predict requires at least one PatchKey.")
        self.model.eval()
        bodies: list[Tensor] = []
        synthetics: list[Tensor] = []
        supports: list[Tensor] = []
        for start in range(0, len(selected), self.batch_size):
            local = selected[start : start + self.batch_size]
            batch = self.reader.batch(local, center_visible=center_visible, device=self.device)
            common = self._common(batch)
            raw = self.model(
                batch.features,
                center_index=self.reader.patch_radius,
                center_lfm_log_ai=batch.lfm_log_ai,
            )
            closure = self.adapter.close_body(
                raw,
                common,
                body_smoothing_fwhm_m=self.body_smoothing_fwhm_m,
            )
            bodies.append(closure.body_log_ai)
            synthetics.append(closure.synthetic_seismic)
            supports.append(closure.valid_mask)
        return BodyResult(
            keys=selected,
            body_log_ai=torch.cat(bodies, dim=0),
            synthetic_seismic=torch.cat(synthetics, dim=0),
            valid_mask=torch.cat(supports, dim=0),
        )


__all__ = ["BodyInverter", "BodyResult"]
