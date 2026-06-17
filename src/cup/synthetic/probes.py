"""Frequency-amplitude probes derived from the forward-observability gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey

from cup.seismic.observability import forward_log_ai
from cup.synthetic.forward import downsample_continuous
from cup.synthetic.generation import GeneratedSection
from cup.utils.statistics import centered_rms


@dataclass(frozen=True)
class ProbeFrequency:
    frequency_hz: float
    evidence_status: str
    operator_support: str
    experiment_class: str
    selection_reason: str
    reference_noise_equivalent_nominal: float
    reference_noise_equivalent_conservative: float
    conservative_to_nominal_ratio: float
    valid_nominal_cluster_count: int
    valid_conservative_cluster_count: int
    calibration_status: str


@dataclass(frozen=True)
class ProbeVariant:
    variant_id: str
    frequency_hz: float
    amplitude_multiplier: float
    phase: str
    lateral_shape: str
    paired_zero_variant_id: str


@dataclass(frozen=True)
class ProbeResult:
    variant: ProbeVariant
    probe_log_ai_highres: np.ndarray
    probe_log_ai_model_grid: np.ndarray
    model_target_log_ai: np.ndarray
    reflectivity_model: np.ndarray
    seismic_model_consistent: np.ndarray
    qc: dict[str, Any]


def _experiment_class(evidence: str, support: str) -> str:
    if support == "unsupported":
        return "unsupported_or_unresolved"
    if evidence == "robust_detectable" and support == "core":
        return "must_recover"
    if evidence in {"robust_detectable", "conditional"} and support in {
        "core",
        "weak",
    }:
        return "stress_test"
    return "unsupported_or_unresolved"


def _representatives(
    frequencies: Sequence[float],
    *,
    maximum: int,
) -> list[float]:
    values = np.asarray(sorted(set(float(value) for value in frequencies)))
    if values.size <= maximum:
        return values.tolist()
    indices = np.unique(
        np.rint(np.linspace(0, values.size - 1, maximum)).astype(np.int64)
    )
    return values[indices].tolist()


def _support_representatives(
    evidence: pd.DataFrame,
    *,
    support: str,
    maximum_per_band: int,
    excluded: set[float],
) -> list[float]:
    rows = evidence[
        evidence["conservative_operator_support"].astype(str).eq(support)
    ].sort_values("frequency_hz")
    frequencies = rows["frequency_hz"].to_numpy(dtype=np.float64)
    if frequencies.size == 0:
        return []
    all_frequency = np.sort(
        evidence["frequency_hz"].to_numpy(dtype=np.float64)
    )
    steps = np.diff(all_frequency)
    expected_step = float(np.median(steps)) if steps.size else float("nan")
    bands: list[list[float]] = []
    current: list[float] = []
    for frequency in frequencies:
        if (
            current
            and np.isfinite(expected_step)
            and not np.isclose(
                frequency - current[-1],
                expected_step,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            bands.append(current)
            current = []
        current.append(float(frequency))
    if current:
        bands.append(current)
    selected: list[float] = []
    for band in bands:
        remaining = [value for value in band if value not in excluded]
        selected.extend(
            _representatives(remaining, maximum=maximum_per_band)
        )
    return selected


def _cluster_median_then_global(
    rows: pd.DataFrame,
    *,
    value_column: str,
) -> tuple[float, int]:
    finite = rows[
        np.isfinite(pd.to_numeric(rows[value_column], errors="coerce"))
        & (pd.to_numeric(rows[value_column], errors="coerce") > 0.0)
    ].copy()
    if finite.empty:
        return float("nan"), 0
    finite[value_column] = pd.to_numeric(finite[value_column])
    cluster = finite.groupby("spatial_cluster_id", dropna=False)[
        value_column
    ].median()
    return float(np.median(cluster.to_numpy(dtype=np.float64))), int(len(cluster))


def _noise_equivalent_references(
    rows: pd.DataFrame,
    *,
    minimum_clusters: int,
) -> dict[float, dict[str, Any]]:
    valid = rows[
        rows["window_type"].astype(str).eq("whole_target")
        & rows["status"].astype(str).eq("ok")
    ].copy()
    valid["noise_equivalent_log_ai"] = pd.to_numeric(
        valid["noise_equivalent_log_ai"],
        errors="coerce",
    )
    records: dict[float, dict[str, Any]] = {}
    for frequency, group in valid.groupby("frequency_hz", dropna=False):
        nominal_rows = group[
            group["wavelet_scenario_kind"].astype(str).eq("nominal")
        ]
        nominal, nominal_clusters = _cluster_median_then_global(
            nominal_rows,
            value_column="noise_equivalent_log_ai",
        )
        conservative_wells = (
            group.groupby(
                ["well_name", "spatial_cluster_id"],
                dropna=False,
            )["noise_equivalent_log_ai"]
            .apply(
                lambda values: float(
                    np.quantile(
                        np.asarray(
                            [
                                value
                                for value in values
                                if np.isfinite(value) and value > 0.0
                            ],
                            dtype=np.float64,
                        ),
                        0.75,
                        method="inverted_cdf",
                    )
                )
                if any(np.isfinite(value) and value > 0.0 for value in values)
                else float("nan")
            )
            .reset_index(name="conservative_noise_equivalent")
        )
        conservative, conservative_clusters = _cluster_median_then_global(
            conservative_wells,
            value_column="conservative_noise_equivalent",
        )
        status = (
            "ok"
            if nominal_clusters >= minimum_clusters
            and conservative_clusters >= minimum_clusters
            and np.isfinite(nominal)
            and nominal > 0.0
            and np.isfinite(conservative)
            and conservative > 0.0
            else "insufficient_noise_equivalent_calibration"
        )
        records[float(frequency)] = {
            "nominal": nominal,
            "conservative": conservative,
            "nominal_clusters": nominal_clusters,
            "conservative_clusters": conservative_clusters,
            "status": status,
        }
    return records


def build_probe_frequency_catalog(
    evidence: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    weak_representatives_per_band: int,
    unsupported_representatives_per_band: int,
    minimum_clusters: int = 3,
) -> list[ProbeFrequency]:
    """Select frequencies and calibrate their field noise-equivalent amplitudes."""
    whole = evidence[
        evidence["window_type"].astype(str).eq("whole_target")
    ].copy()
    if whole.empty:
        raise ValueError("missing_whole_target_frequency_evidence")
    whole["frequency_hz"] = pd.to_numeric(whole["frequency_hz"])
    whole = whole.sort_values("frequency_hz")
    selected: dict[float, str] = {}
    for row in whole.to_dict(orient="records"):
        frequency = float(row["frequency_hz"])
        status = str(row["evidence_status"])
        support = str(row["conservative_operator_support"])
        if status in {"robust_detectable", "conditional"}:
            selected[frequency] = status
        elif status == "not_detectable" and support == "core":
            selected[frequency] = "core_but_not_detectable"
    excluded = set(selected)
    for support, maximum in (
        ("weak", int(weak_representatives_per_band)),
        ("unsupported", int(unsupported_representatives_per_band)),
    ):
        for frequency in _support_representatives(
            whole,
            support=support,
            maximum_per_band=maximum,
            excluded=excluded,
        ):
            selected[frequency] = f"{support}_control"
            excluded.add(frequency)
    references = _noise_equivalent_references(
        sensitivity,
        minimum_clusters=int(minimum_clusters),
    )
    catalog: list[ProbeFrequency] = []
    by_frequency = {
        float(row["frequency_hz"]): row
        for row in whole.to_dict(orient="records")
    }
    for frequency in sorted(selected):
        row = by_frequency[frequency]
        reference = references.get(
            frequency,
            {
                "nominal": float("nan"),
                "conservative": float("nan"),
                "nominal_clusters": 0,
                "conservative_clusters": 0,
                "status": "insufficient_noise_equivalent_calibration",
            },
        )
        nominal = float(reference["nominal"])
        conservative = float(reference["conservative"])
        ratio = (
            conservative / nominal
            if np.isfinite(nominal) and nominal > 0.0
            else float("nan")
        )
        support = str(row["conservative_operator_support"])
        status = str(row["evidence_status"])
        catalog.append(
            ProbeFrequency(
                frequency_hz=frequency,
                evidence_status=status,
                operator_support=support,
                experiment_class=_experiment_class(status, support),
                selection_reason=selected[frequency],
                reference_noise_equivalent_nominal=nominal,
                reference_noise_equivalent_conservative=conservative,
                conservative_to_nominal_ratio=ratio,
                valid_nominal_cluster_count=int(reference["nominal_clusters"]),
                valid_conservative_cluster_count=int(
                    reference["conservative_clusters"]
                ),
                calibration_status=str(reference["status"]),
            )
        )
    return catalog


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def probe_variants(
    frequency: ProbeFrequency,
    *,
    amplitude_multipliers: Sequence[float],
    phases: Sequence[str],
    lateral_shapes: Sequence[Mapping[str, Any]],
) -> list[ProbeVariant]:
    calibrated = frequency.calibration_status == "ok"
    multipliers = (
        [float(value) for value in amplitude_multipliers]
        if calibrated
        else [0.0]
    )
    variants: list[ProbeVariant] = []
    prefix = f"f{_number_token(frequency.frequency_hz)}"
    for phase in phases:
        zero_id = f"{prefix}__m0__{phase}__zero"
        variants.append(
            ProbeVariant(
                variant_id=zero_id,
                frequency_hz=frequency.frequency_hz,
                amplitude_multiplier=0.0,
                phase=str(phase),
                lateral_shape="zero",
                paired_zero_variant_id=zero_id,
            )
        )
        for multiplier in multipliers:
            if multiplier == 0.0:
                continue
            for shape in lateral_shapes:
                name = str(shape["name"])
                variants.append(
                    ProbeVariant(
                        variant_id=(
                            f"{prefix}__m{_number_token(multiplier)}"
                            f"__{phase}__{name}"
                        ),
                        frequency_hz=frequency.frequency_hz,
                        amplitude_multiplier=float(multiplier),
                        phase=str(phase),
                        lateral_shape=name,
                        paired_zero_variant_id=zero_id,
                    )
                )
    return variants


def _probe_support(section: GeneratedSection) -> np.ndarray:
    if str(section.qc.get("suite", "")) == "canonical":
        return (section.rgt_highres >= 0.0) & (section.rgt_highres <= 1.0)
    return section.zone_id_highres >= 0


def _lateral_weights(
    size: int,
    *,
    shape: str,
    shapes: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if shape == "zero":
        return np.ones(size, dtype=np.float64)
    if shape == "section_coherent":
        return np.ones(size, dtype=np.float64)
    config = next(
        (item for item in shapes if str(item["name"]) == shape),
        None,
    )
    if config is None or shape != "localized_tukey":
        raise ValueError(f"unsupported_probe_lateral_shape:{shape}")
    fraction = float(config["centered_fraction"])
    alpha = float(config["alpha"])
    count = max(2, int(round(size * fraction)))
    count = min(count, size)
    start = (size - count) // 2
    weights = np.zeros(size, dtype=np.float64)
    weights[start : start + count] = tukey(count, alpha=alpha, sym=True)
    return weights


def generate_probe(
    section: GeneratedSection,
    frequency: ProbeFrequency,
    variant: ProbeVariant,
    *,
    wavelet: np.ndarray,
    antialias_filter_taps: np.ndarray,
    vertical_tukey_alpha: float,
    lateral_shapes: Sequence[Mapping[str, Any]],
    low_probe_energy_warning_fraction: float,
) -> ProbeResult:
    """Add one paired deterministic probe to a frozen parent realization."""
    support = _probe_support(section)
    raw = np.zeros(section.truth_log_ai_highres.shape, dtype=np.float64)
    phase_function = np.sin if variant.phase == "sin" else np.cos
    for lateral_index in range(raw.shape[0]):
        indices = np.flatnonzero(support[lateral_index])
        if indices.size < 2:
            continue
        vertical = tukey(
            indices.size,
            alpha=float(vertical_tukey_alpha),
            sym=True,
        )
        values = phase_function(
            2.0
            * np.pi
            * frequency.frequency_hz
            * section.twt_highres_s[indices]
        )
        values = vertical * values
        positive = vertical > 0.0
        if np.any(positive):
            values[positive] -= float(np.mean(values[positive]))
        raw[lateral_index, indices] = values
    lateral = _lateral_weights(
        raw.shape[0],
        shape=variant.lateral_shape,
        shapes=lateral_shapes,
    )
    raw *= lateral[:, None]
    active = support & (lateral[:, None] > 0.0)
    requested = (
        variant.amplitude_multiplier
        * frequency.reference_noise_equivalent_nominal
        if frequency.calibration_status == "ok"
        else 0.0
    )
    if requested > 0.0:
        raw_rms = centered_rms(raw, active, min_count=2)
        if not np.isfinite(raw_rms) or raw_rms <= 0.0:
            raise ValueError("invalid_probe_basis_energy")
        probe_highres = raw * (requested / raw_rms)
    else:
        probe_highres = np.zeros_like(raw)
    factor = int(
        round(
            (section.twt_model_s[1] - section.twt_model_s[0])
            / (section.twt_highres_s[1] - section.twt_highres_s[0])
        )
    )
    probe_model = downsample_continuous(
        probe_highres,
        factor,
        antialias_filter_taps,
    )[..., : section.twt_model_s.size]
    target_model = section.model_target_log_ai + probe_model
    reflectivity_model = np.tanh(0.5 * np.diff(target_model, axis=-1))
    seismic = np.stack(
        [forward_log_ai(trace, wavelet) for trace in target_model],
        axis=0,
    )
    model_mask = section.valid_mask_model & np.isfinite(target_model)
    model_probe_rms = centered_rms(probe_model, model_mask, min_count=2)
    total_rms = centered_rms(target_model, model_mask, min_count=2)
    rms_fraction = (
        model_probe_rms / total_rms
        if np.isfinite(total_rms) and total_rms > 0.0
        else float("nan")
    )
    energy_fraction = (
        rms_fraction * rms_fraction
        if np.isfinite(rms_fraction)
        else float("nan")
    )
    low_energy = (
        np.isclose(variant.amplitude_multiplier, 4.0)
        and np.isfinite(energy_fraction)
        and energy_fraction < float(low_probe_energy_warning_fraction)
    )
    qc = {
        "probe_status": "ok",
        "probe_frequency_hz": frequency.frequency_hz,
        "probe_phase": variant.phase,
        "probe_lateral_shape": variant.lateral_shape,
        "probe_amplitude_multiplier": variant.amplitude_multiplier,
        "probe_reference_noise_equivalent_nominal": (
            frequency.reference_noise_equivalent_nominal
        ),
        "probe_reference_noise_equivalent_conservative": (
            frequency.reference_noise_equivalent_conservative
        ),
        "probe_conservative_to_nominal_ratio": (
            frequency.conservative_to_nominal_ratio
        ),
        "probe_rms_requested_highres": float(requested),
        "probe_rms_actual_highres": centered_rms(probe_highres, active, min_count=2),
        "probe_rms_after_antialias": model_probe_rms,
        "probe_rms_actual_model_grid": model_probe_rms,
        "probe_rms_fraction_of_total_model_grid": rms_fraction,
        "probe_energy_fraction_of_total_model_grid": energy_fraction,
        "low_probe_energy_warning": bool(low_energy),
        "probe_active_highres_samples": int(np.count_nonzero(active)),
        "probe_valid_model_samples": int(np.count_nonzero(model_mask)),
    }
    return ProbeResult(
        variant=variant,
        probe_log_ai_highres=probe_highres,
        probe_log_ai_model_grid=probe_model,
        model_target_log_ai=target_model,
        reflectivity_model=reflectivity_model,
        seismic_model_consistent=seismic,
        qc=qc,
    )


def frequency_catalog_rows(
    frequencies: Iterable[ProbeFrequency],
) -> list[dict[str, Any]]:
    return [
        {
            "frequency_hz": item.frequency_hz,
            "evidence_status": item.evidence_status,
            "operator_support": item.operator_support,
            "experiment_class": item.experiment_class,
            "selection_reason": item.selection_reason,
            "reference_noise_equivalent_nominal": (
                item.reference_noise_equivalent_nominal
            ),
            "reference_noise_equivalent_conservative": (
                item.reference_noise_equivalent_conservative
            ),
            "conservative_to_nominal_ratio": (
                item.conservative_to_nominal_ratio
            ),
            "valid_nominal_cluster_count": item.valid_nominal_cluster_count,
            "valid_conservative_cluster_count": (
                item.valid_conservative_cluster_count
            ),
            "calibration_status": item.calibration_status,
        }
        for item in frequencies
    ]
