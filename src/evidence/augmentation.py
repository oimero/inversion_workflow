"""Frozen, label-free seismic observation augmentation.

The canonical synthetic artifact remains clean.  This module owns the seam
between that artifact and an online dirty observation.  It deliberately
accepts only an explicit profile and returns the random identity used for one
sample, so a dirty pair can be reproduced and audited without materializing a
second seismic view.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


AUGMENTATION_PROFILE_SCHEMA = "bandlimited_evidence_observation_profile_v1"


def _finite(value: Any, *, label: str, minimum: float = 0.0) -> float:
    result = float(value)
    if not np.isfinite(result) or result < float(minimum):
        raise ValueError(f"{label} must be finite and >= {minimum}.")
    return result


@dataclass(frozen=True)
class ObservationAugmentationProfile:
    """Bounds frozen from real observation statistics, without underground labels."""

    phase_degrees: float = 8.0
    wavelet_shift_samples: int = 1
    vertical_static_samples: int = 1
    global_gain_log: float = 0.10
    trace_gain_log: float = 0.05
    lateral_attenuation_fraction: float = 0.10
    white_noise_fraction: float = 0.05
    colored_noise_fraction: float = 0.04
    coherent_noise_fraction: float = 0.03
    weak_reflection_fraction: float = 0.15
    colored_noise_kernel: int = 5
    source_summary: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        phase = _finite(self.phase_degrees, label="phase_degrees")
        if phase > 90.0:
            raise ValueError("phase_degrees must not exceed 90 degrees.")
        for name in (
            "global_gain_log",
            "trace_gain_log",
            "lateral_attenuation_fraction",
            "white_noise_fraction",
            "colored_noise_fraction",
            "coherent_noise_fraction",
            "weak_reflection_fraction",
        ):
            value = _finite(getattr(self, name), label=name)
            if name.endswith("fraction") and value > 1.0:
                raise ValueError(f"{name} must be <= 1.")
        for name in ("wavelet_shift_samples", "vertical_static_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        kernel = int(self.colored_noise_kernel)
        if kernel <= 0 or kernel % 2 != 1:
            raise ValueError("colored_noise_kernel must be a positive odd integer.")
        summary = {} if self.source_summary is None else dict(self.source_summary)
        try:
            json.dumps(summary, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("source_summary must be JSON serializable.") from exc
        object.__setattr__(self, "phase_degrees", phase)
        object.__setattr__(self, "wavelet_shift_samples", int(self.wavelet_shift_samples))
        object.__setattr__(
            self,
            "vertical_static_samples",
            int(self.vertical_static_samples),
        )
        object.__setattr__(self, "colored_noise_kernel", kernel)
        object.__setattr__(self, "source_summary", summary)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservationAugmentationProfile":
        names = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(names))
        if unknown:
            raise ValueError(f"unknown observation profile fields: {unknown}")
        return cls(**dict(value))

    @classmethod
    def from_observation_statistics(
        cls,
        seismic: np.ndarray,
        valid: np.ndarray,
        *,
        bounds: Mapping[str, Any] | None = None,
    ) -> "ObservationAugmentationProfile":
        """Create a profile from observed seismic statistics only.

        This helper never receives AI, state, object or zone labels.  The
        caller must still persist and review the returned profile before using
        it for calibration or training.
        """
        values = np.asarray(seismic, dtype=np.float64)
        mask = np.asarray(valid, dtype=bool)
        if values.shape != mask.shape or values.ndim not in (1, 2):
            raise ValueError("seismic and valid must have matching 1D or 2D shapes.")
        finite = mask & np.isfinite(values)
        if not np.any(finite):
            raise ValueError("observation statistics require finite valid samples.")
        scale = float(np.median(np.abs(values[finite])))
        spread = float(np.std(values[finite]))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("observed seismic median absolute amplitude is zero.")
        if not np.isfinite(spread) or spread <= 0.0:
            raise ValueError("observed seismic standard deviation is zero.")
        adjacent_correlation = 0.0
        if values.ndim == 2 and values.shape[0] > 1:
            left = values[:-1]
            right = values[1:]
            pair_mask = finite[:-1] & finite[1:]
            if np.count_nonzero(pair_mask) >= 2:
                left_values = left[pair_mask]
                right_values = right[pair_mask]
                if np.std(left_values) > 0.0 and np.std(right_values) > 0.0:
                    adjacent_correlation = float(np.corrcoef(left_values, right_values)[0, 1])
        payload: dict[str, Any] = {
            "median_abs_amplitude": scale,
            "standard_deviation": spread,
            "adjacent_trace_correlation": adjacent_correlation,
            "sample_count": int(np.count_nonzero(finite)),
        }
        if bounds is not None:
            payload.update(dict(bounds))
        return cls(source_summary=payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = AUGMENTATION_PROFILE_SCHEMA
        return payload


@dataclass(frozen=True)
class AugmentedSeismic:
    """One reproducible dirty seismic view and its frozen random identity."""

    seismic: np.ndarray
    observed_valid: np.ndarray
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.seismic, dtype=np.float64)
        valid = np.asarray(self.observed_valid, dtype=bool)
        if values.shape != valid.shape or values.ndim not in (1, 2):
            raise ValueError("augmented seismic and support shapes must match.")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError("augmented valid seismic contains non-finite values.")
        if not isinstance(self.identity, Mapping) or not self.identity:
            raise ValueError("augmentation identity must be explicit.")
        object.__setattr__(self, "seismic", values)
        object.__setattr__(self, "observed_valid", valid)
        object.__setattr__(self, "identity", dict(self.identity))


def stable_random_identity(*parts: object) -> int:
    """Derive a deterministic 64-bit identity from parent/patch/condition keys."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def load_observation_augmentation_profile(
    path: str | Path,
) -> ObservationAugmentationProfile:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("observation augmentation profile must be a JSON object.")
    if payload.get("schema") != AUGMENTATION_PROFILE_SCHEMA:
        raise ValueError("unsupported observation augmentation profile schema.")
    values = dict(payload)
    values.pop("schema")
    return ObservationAugmentationProfile.from_mapping(values)


def _analytic_signal(values: np.ndarray) -> np.ndarray:
    size = values.shape[-1]
    spectrum = np.fft.fft(values, axis=-1)
    multiplier = np.zeros(size, dtype=np.float64)
    multiplier[0] = 1.0
    if size % 2 == 0:
        multiplier[1 : size // 2] = 2.0
        multiplier[size // 2] = 1.0
    else:
        multiplier[1 : (size + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier, axis=-1)


def _phase_rotate(values: np.ndarray, degrees: float) -> np.ndarray:
    phase = np.deg2rad(float(degrees))
    return np.real(_analytic_signal(values) * np.exp(1j * phase))


def _fractional_shift(values: np.ndarray, shift: float) -> np.ndarray:
    axis = np.arange(values.shape[-1], dtype=np.float64)
    source = axis - float(shift)
    output = np.empty_like(values, dtype=np.float64)
    if values.ndim == 1:
        return np.interp(source, axis, values, left=0.0, right=0.0)
    for index, row in enumerate(values):
        output[index] = np.interp(source, axis, row, left=0.0, right=0.0)
    return output


def _smooth_noise(noise: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return noise
    taps = np.ones(int(kernel), dtype=np.float64) / float(kernel)
    if noise.ndim == 1:
        return np.convolve(noise, taps, mode="same")
    return np.stack(
        [np.convolve(row, taps, mode="same") for row in noise],
        axis=0,
    )


def apply_observation_augmentation(
    seismic: np.ndarray,
    observed_valid: np.ndarray,
    *,
    profile: ObservationAugmentationProfile,
    rng: np.random.Generator,
    relative_lateral_m: np.ndarray | None = None,
) -> AugmentedSeismic:
    """Apply bounded waveform perturbations while preserving the truth pair.

    The function accepts a 1D trace or a [lateral, vertical] patch.  All
    perturbations are applied only to finite valid samples; invalid support is
    returned unchanged and never filled by an edge trace.
    """
    values = np.asarray(seismic, dtype=np.float64)
    valid = np.asarray(observed_valid, dtype=bool)
    if values.shape != valid.shape or values.ndim not in (1, 2):
        raise ValueError("seismic and observed_valid must have matching 1D/2D shapes.")
    if np.any(valid & ~np.isfinite(values)):
        raise ValueError("input valid seismic contains non-finite values.")
    is_trace = values.ndim == 1
    work = values.copy()
    work[~valid] = 0.0
    if is_trace:
        work_2d = work[None, :]
        valid_2d = valid[None, :]
    else:
        work_2d = work
        valid_2d = valid
    finite_values = work_2d[valid_2d]
    scale = float(np.std(finite_values)) if finite_values.size else 0.0
    if not np.isfinite(scale):
        raise ValueError("input seismic scale is non-finite.")

    phase = float(rng.uniform(-profile.phase_degrees, profile.phase_degrees))
    work_2d = _phase_rotate(work_2d, phase)
    shift = float(
        rng.uniform(
            -profile.wavelet_shift_samples,
            profile.wavelet_shift_samples,
        )
    )
    work_2d = _fractional_shift(work_2d, shift)
    static = rng.uniform(
        -profile.vertical_static_samples,
        profile.vertical_static_samples,
        size=work_2d.shape[0],
    )
    if np.any(static != 0.0):
        work_2d = np.stack(
            [
                _fractional_shift(work_2d[index], float(static[index]))
                for index in range(work_2d.shape[0])
            ],
            axis=0,
        )

    global_gain = float(rng.uniform(-profile.global_gain_log, profile.global_gain_log))
    work_2d *= np.exp(global_gain)
    trace_gain = rng.uniform(
        -profile.trace_gain_log,
        profile.trace_gain_log,
        size=(work_2d.shape[0], 1),
    )
    work_2d *= np.exp(trace_gain)

    lateral_scale = None
    if relative_lateral_m is not None:
        distances = np.asarray(relative_lateral_m, dtype=np.float64).reshape(-1)
        if distances.size != work_2d.shape[0] or np.any(~np.isfinite(distances)):
            raise ValueError("relative_lateral_m must match the patch width and be finite.")
        maximum_distance = float(np.max(np.abs(distances)))
        lateral_scale = (
            np.abs(distances) / maximum_distance
            if maximum_distance > 0.0
            else np.zeros_like(distances)
        )
        attenuation = 1.0 - rng.uniform(
            0.0,
            profile.lateral_attenuation_fraction,
        ) * lateral_scale
        work_2d *= attenuation[:, None]

    white_std = float(rng.uniform(0.0, profile.white_noise_fraction)) * scale
    colored_std = float(rng.uniform(0.0, profile.colored_noise_fraction)) * scale
    coherent_std = float(rng.uniform(0.0, profile.coherent_noise_fraction)) * scale
    noise = rng.normal(0.0, white_std, size=work_2d.shape)
    colored = _smooth_noise(
        rng.normal(0.0, colored_std, size=work_2d.shape),
        profile.colored_noise_kernel,
    )
    coherent = rng.normal(0.0, coherent_std, size=(1, work_2d.shape[1]))
    coherent = np.repeat(coherent, work_2d.shape[0], axis=0)
    work_2d += noise + colored + coherent

    weak_fraction = float(
        rng.uniform(0.0, profile.weak_reflection_fraction)
    )
    threshold = weak_fraction * scale
    if threshold > 0.0:
        small = np.abs(work_2d) < threshold
        work_2d[small] *= 1.0 - weak_fraction

    work_2d[~valid_2d] = 0.0
    if np.any(~np.isfinite(work_2d)):
        raise FloatingPointError("observation augmentation produced non-finite seismic.")
    identity = {
        "schema": AUGMENTATION_PROFILE_SCHEMA,
        "seed": int(rng.bit_generator.state["state"]["state"]),
        "phase_degrees": phase,
        "wavelet_shift_samples": shift,
        "vertical_static_samples": static.tolist(),
        "global_gain_log": global_gain,
        "white_noise_std": white_std,
        "colored_noise_std": colored_std,
        "coherent_noise_std": coherent_std,
        "weak_reflection_fraction": weak_fraction,
    }
    output = work_2d[0] if is_trace else work_2d
    output_valid = valid_2d[0] if is_trace else valid_2d
    return AugmentedSeismic(output, output_valid, identity)


__all__ = [
    "AUGMENTATION_PROFILE_SCHEMA",
    "AugmentedSeismic",
    "ObservationAugmentationProfile",
    "apply_observation_augmentation",
    "load_observation_augmentation_profile",
    "stable_random_identity",
]
