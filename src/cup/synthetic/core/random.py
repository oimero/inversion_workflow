"""Order-independent named random streams for synthetic benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RandomNamespace:
    """Stable namespace prefix for all named scientific random streams."""

    benchmark_version: str
    science_revision: str
    random_stream_contract_version: str
    generator_family: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.benchmark_version,
                self.science_revision,
                self.random_stream_contract_version,
                self.generator_family,
            )
        ):
            raise ValueError("random namespace fields must be non-empty.")

    def keys(self) -> dict[str, str]:
        return {
            "benchmark_version": self.benchmark_version,
            "science_revision": self.science_revision,
            "random_stream_contract_version": self.random_stream_contract_version,
            "generator_family": self.generator_family,
        }


def named_seed(
    *,
    global_seed: int,
    benchmark_version: str,
    science_revision: str,
    random_stream_contract_version: str,
    generator_family: str,
    stream_purpose: str,
    realization_id: str = "",
    zone_id: str = "",
    object_id: str = "",
    coefficient_name: str = "",
    variant_id: str = "",
    operator_id: str = "",
    operator_spec_sha256: str = "",
) -> int:
    """Derive a stable 128-bit seed from the frozen naming contract."""
    payload = [
        str(int(global_seed)),
        str(benchmark_version),
        str(science_revision),
        str(random_stream_contract_version),
        str(generator_family),
        str(stream_purpose),
        str(realization_id),
        str(zone_id),
        str(object_id),
        str(coefficient_name),
        str(variant_id),
        str(operator_id),
        str(operator_spec_sha256),
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], byteorder="big", signed=False)


def named_rng(**keys: Any) -> np.random.Generator:
    """Return a PCG64DXSM generator for one named stream."""
    return np.random.Generator(np.random.PCG64DXSM(named_seed(**keys)))


def operator_rng(
    *,
    global_seed: int,
    generator_family: str,
    realization_id: str,
    operator_id: str,
    operator_spec_sha256: str,
    coefficient_name: str,
) -> np.random.Generator:
    """Return the v5 random stream for one operator coefficient.

    The view name and its position are intentionally absent. Reusing the same
    operator ID/spec in two views therefore reuses the same coefficient field.
    """

    return named_rng(
        global_seed=global_seed,
        benchmark_version="synthoseis_lite_v5",
        science_revision="synthoseis_lite_science_v3",
        random_stream_contract_version="synthoseis_random_v3",
        generator_family=generator_family,
        stream_purpose="seismic_view_operator",
        realization_id=realization_id,
        operator_id=operator_id,
        operator_spec_sha256=operator_spec_sha256,
        coefficient_name=coefficient_name,
    )


def ar1_irregular(
    distance_m: np.ndarray,
    *,
    correlation_length_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate, clip, center and RMS-normalize an irregularly sampled AR(1)."""
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    if distance.size < 2 or np.any(~np.isfinite(distance)) or np.any(np.diff(distance) <= 0.0):
        raise ValueError("distance_m must be finite and strictly increasing.")
    length = float(correlation_length_m)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("correlation_length_m must be positive.")
    raw = np.empty(distance.size, dtype=np.float64)
    raw[0] = rng.normal()
    for index in range(1, distance.size):
        rho = float(np.exp(-(distance[index] - distance[index - 1]) / length))
        raw[index] = rho * raw[index - 1] + np.sqrt(max(0.0, 1.0 - rho * rho)) * rng.normal()
    clipped = np.clip(raw, -3.0, 3.0)
    centered = clipped - float(np.mean(clipped))
    rms = float(np.sqrt(np.mean(centered * centered)))
    normalized = centered / rms if rms > 0.0 else np.zeros_like(centered)
    return normalized, {
        "raw_mean": float(np.mean(raw)),
        "raw_rms": float(np.sqrt(np.mean(raw * raw))),
        "clipped_mean": float(np.mean(clipped)),
        "clipped_rms": float(np.sqrt(np.mean(clipped * clipped))),
        "normalized_mean": float(np.mean(normalized)),
        "normalized_rms": float(np.sqrt(np.mean(normalized * normalized))),
        "empirical_correlation_length_m": empirical_correlation_length(distance, normalized),
    }


def empirical_correlation_length(distance_m: np.ndarray, values: np.ndarray) -> float:
    """Estimate the first e-folding lag using pairwise physical distances."""
    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if distance.size != signal.size or distance.size < 4 or np.std(signal) <= 0.0:
        return float("nan")
    spacing = float(np.median(np.diff(distance)))
    maximum_lag = max(1, signal.size // 2)
    target = np.exp(-1.0)
    for lag in range(1, maximum_lag + 1):
        left = signal[:-lag]
        right = signal[lag:]
        if left.size < 3 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if np.isfinite(corr) and corr <= target:
            return lag * spacing
    return float("nan")
