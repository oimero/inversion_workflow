"""Reserved microtexture emission interface for Synthoseis-lite v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class MicrotextureEmission:
    """Object-local high-resolution log(AI) texture and its QC metadata."""

    values_log_ai: np.ndarray
    mode: str
    source_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MicrotextureEmitter(Protocol):
    mode: str

    def emit(
        self,
        *,
        zone_id: str,
        state_id: str,
        object_top: float,
        object_bottom: float,
        sample_axis_highres: np.ndarray,
        macro_profile: np.ndarray,
        seed: int,
    ) -> MicrotextureEmission:
        """Emit an object-local perturbation on the supplied physical axis."""


@dataclass(frozen=True)
class NoneMicrotextureEmitter:
    """Baseline emitter: no object-local microtexture."""

    mode: str = "none"

    def emit(
        self,
        *,
        zone_id: str,
        state_id: str,
        object_top: float,
        object_bottom: float,
        sample_axis_highres: np.ndarray,
        macro_profile: np.ndarray,
        seed: int,
    ) -> MicrotextureEmission:
        profile = np.asarray(macro_profile, dtype=np.float64)
        return MicrotextureEmission(
            values_log_ai=np.zeros_like(profile),
            mode=self.mode,
            source_id="none",
            metadata={
                "zone_id": str(zone_id),
                "state_id": str(state_id),
                "physical_thickness": float(object_bottom - object_top),
                "seed": int(seed),
            },
        )


def build_microtexture_emitter(mode: str = "none") -> MicrotextureEmitter:
    normalized = str(mode).strip().lower()
    if normalized == "none":
        return NoneMicrotextureEmitter()
    if normalized in {"thin_bed_cluster", "canonical_well_texture"}:
        raise NotImplementedError(
            f"Microtexture mode {normalized!r} is reserved for the next generation stage."
        )
    raise ValueError(f"Unsupported microtexture mode: {mode!r}")


__all__ = [
    "MicrotextureEmission",
    "MicrotextureEmitter",
    "NoneMicrotextureEmitter",
    "build_microtexture_emitter",
]
