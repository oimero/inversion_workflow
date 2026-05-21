"""Reusable LAS curve preprocessing helpers for the time-domain workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from wtie.processing import grid

STANDARD_MNEMONICS: dict[str, str] = {
    "p_sonic": "DT_USM",
    "s_sonic": "DTS_USM",
    "density": "RHO_GCC",
    "gamma_ray": "GR",
    "caliper": "CALI",
    "resistivity": "RT",
    "spontaneous_potential": "SP",
    "porosity": "POR",
    "permeability": "PERM",
    "water_saturation": "SW",
}

DEFAULT_MISSING_SENTINELS = (-999.0, -999.25, -9999.0, -99999.0)

SONIC_CATEGORIES = {"p_sonic", "s_sonic"}


@dataclass(frozen=True)
class UnitStandardization:
    values: np.ndarray
    original_unit: str
    standard_unit: str
    conversion_action: str
    hard_fail_reason: str = ""
    qc_flags: tuple[str, ...] = ()
    input_valid_count: int = 0
    output_valid_count: int = 0
    input_median: float | None = None
    output_median: float | None = None
    input_p01: float | None = None
    input_p99: float | None = None
    output_p01: float | None = None
    output_p99: float | None = None

    def report_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("values")
        row["qc_flags"] = ";".join(self.qc_flags)
        return row


@dataclass(frozen=True)
class ConstantRun:
    start_md: float
    end_md: float
    run_length: int
    constant_value: float
    action: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutlierRemoval:
    values: np.ndarray
    replaced_points: int
    lower: float | None
    upper: float | None
    threshold_source: str


@dataclass(frozen=True)
class CurveThreshold:
    standard_mnemonic: str
    lower: float | None
    upper: float | None
    source: str
    sample_count: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WellCurveSet:
    """Arbitrary MD-domain curves for one well.

    Unlike ``wtie.processing.grid.LogSet``, this object does not require Vp/Rho.
    It is intended for preprocessing stages that carry auxiliary LAS curves such
    as GR, caliper, porosity, and saturation before a rock-physics LogSet exists.
    """

    well_name: str
    logs: Mapping[str, grid.Log]
    source_las: str = ""

    def __post_init__(self) -> None:
        validate_shared_basis(self.logs)

    @property
    def basis(self) -> np.ndarray:
        first = next(iter(self.logs.values()))
        return np.asarray(first.basis, dtype=float)

    @property
    def curve_names(self) -> list[str]:
        return list(self.logs.keys())

    def get(self, name: str) -> grid.Log | None:
        return self.logs.get(name)

    def require(self, name: str) -> grid.Log:
        log = self.get(name)
        if log is None:
            raise KeyError(f"Curve {name!r} is not available in well {self.well_name!r}.")
        return log


def validate_shared_basis(logs: Mapping[str, grid.Log]) -> None:
    """Validate that all logs share one MD sampling axis."""
    if not logs:
        raise ValueError("WellCurveSet requires at least one curve.")
    first_name, first_log = next(iter(logs.items()))
    if not isinstance(first_log, grid.Log):
        raise TypeError(f"Curve {first_name!r} is not a grid.Log.")
    basis = np.asarray(first_log.basis, dtype=float)
    for name, log in logs.items():
        if not isinstance(log, grid.Log):
            raise TypeError(f"Curve {name!r} is not a grid.Log.")
        if not log.is_md:
            raise ValueError(f"Curve {name!r} must be in MD domain.")
        if log.basis.shape != basis.shape or not np.allclose(np.asarray(log.basis, dtype=float), basis, atol=1e-6):
            raise ValueError(f"Curve {name!r} does not share the same MD basis as {first_name!r}.")


def select_curves_by_category(
    curve_set: WellCurveSet,
    category_to_mnemonic: Mapping[str, str],
    categories: Sequence[str],
) -> dict[str, grid.Log]:
    """Select logs from a curve set according to a category->mnemonic mapping."""
    selected: dict[str, grid.Log] = {}
    for category in categories:
        mnemonic = category_to_mnemonic.get(category)
        if mnemonic is None:
            continue
        log = curve_set.get(mnemonic)
        if log is not None:
            selected[category] = log
    return selected


def standard_mnemonic_for_category(category: str) -> str:
    """Return the workflow standard mnemonic for a curve category."""
    try:
        return STANDARD_MNEMONICS[str(category)]
    except KeyError as exc:
        raise ValueError(f"No standard mnemonic configured for category {category!r}.") from exc


def normalize_unit(unit: object) -> str:
    """Normalize unit spelling for rule matching."""
    text = str(unit or "").strip().lower()
    text = text.replace(" ", "")
    text = text.replace("μ", "u").replace("µ", "u")
    return text


def values_to_nan(values: object, *, null_value: float | None = None, sentinels: Sequence[float] = DEFAULT_MISSING_SENTINELS) -> np.ndarray:
    """Convert numeric values to float and replace known missing sentinels with NaN."""
    out = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(out)
    for sentinel in sentinels:
        missing |= np.isclose(out, float(sentinel), equal_nan=False)
    if null_value is not None and np.isfinite(float(null_value)):
        missing |= np.isclose(out, float(null_value), equal_nan=False)
    out[missing] = np.nan
    return out


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    """Return robust finite-value stats for reports and QC rules."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"valid_count": 0, "median": None, "p01": None, "p99": None, "min": None, "max": None}
    return {
        "valid_count": int(finite.size),
        "median": float(np.nanmedian(finite)),
        "p01": float(np.nanquantile(finite, 0.01)),
        "p99": float(np.nanquantile(finite, 0.99)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
    }


def _with_stats(
    values: np.ndarray,
    converted: np.ndarray,
    *,
    original_unit: str,
    standard_unit: str,
    conversion_action: str,
    hard_fail_reason: str = "",
    qc_flags: Sequence[str] = (),
) -> UnitStandardization:
    before = finite_stats(values)
    after = finite_stats(converted)
    return UnitStandardization(
        values=converted,
        original_unit=original_unit,
        standard_unit=standard_unit,
        conversion_action=conversion_action,
        hard_fail_reason=hard_fail_reason,
        qc_flags=tuple(qc_flags),
        input_valid_count=int(before["valid_count"]),
        output_valid_count=int(after["valid_count"]),
        input_median=before["median"],
        output_median=after["median"],
        input_p01=before["p01"],
        input_p99=before["p99"],
        output_p01=after["p01"],
        output_p99=after["p99"],
    )


def standardize_curve_unit(values: np.ndarray, *, category: str, unit: str) -> UnitStandardization:
    """Standardize supported curve units and detect impossible unit mismatches."""
    original_unit = str(unit or "")
    unit_norm = normalize_unit(unit)
    clean = np.asarray(values, dtype=float)
    stats = finite_stats(clean)
    median = stats["median"]
    qc_flags: list[str] = []

    if category in SONIC_CATEGORIES:
        if stats["valid_count"] == 0:
            return _with_stats(
                clean,
                clean.copy(),
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="none",
                hard_fail_reason="no_finite_samples",
            )
        assert median is not None
        positive = clean.copy()
        positive[positive <= 0.0] = np.nan
        if unit_norm in {"us/ft", "usec/ft"}:
            if median > 1000.0:
                return _with_stats(
                    clean,
                    positive * 3.280839895,
                    original_unit=original_unit,
                    standard_unit="us/m",
                    conversion_action="us_ft_to_us_m",
                    hard_fail_reason="sonic_slowness_unit_impossible_median_gt_1000",
                )
            if median > 160.0:
                qc_flags.append("sonic_us_ft_median_looks_like_us_m")
            return _with_stats(
                clean,
                positive * 3.280839895,
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="us_ft_to_us_m",
                qc_flags=qc_flags,
            )
        if unit_norm in {"us/m", "usec/m"}:
            if median > 1000.0:
                return _with_stats(
                    clean,
                    positive,
                    original_unit=original_unit,
                    standard_unit="us/m",
                    conversion_action="keep_us_m",
                    hard_fail_reason="sonic_slowness_unit_impossible_median_gt_1000",
                )
            if 40.0 <= median <= 160.0:
                qc_flags.append("sonic_us_m_median_looks_like_us_ft")
            return _with_stats(
                clean,
                positive,
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="keep_us_m",
                qc_flags=qc_flags,
            )
        if unit_norm in {"m/s", "mps", "m/sec", "meter/s", "meters/s"}:
            if median < 1000.0:
                return _with_stats(
                    clean,
                    1.0e6 / positive,
                    original_unit=original_unit,
                    standard_unit="us/m",
                    conversion_action="mps_to_us_m",
                    hard_fail_reason="sonic_velocity_unit_impossible_median_lt_1000",
                )
            return _with_stats(
                clean,
                1.0e6 / positive,
                original_unit=original_unit,
                standard_unit="us/m",
                conversion_action="mps_to_us_m",
            )
        return _with_stats(
            clean,
            clean.copy(),
            original_unit=original_unit,
            standard_unit="us/m",
            conversion_action="unsupported_unit",
            hard_fail_reason="unsupported_sonic_unit",
        )

    if category == "density":
        if stats["valid_count"] == 0:
            return _with_stats(
                clean,
                clean.copy(),
                original_unit=original_unit,
                standard_unit="g/cm3",
                conversion_action="none",
                hard_fail_reason="no_finite_samples",
            )
        assert median is not None
        positive = clean.copy()
        positive[positive <= 0.0] = np.nan
        if unit_norm in {"g/cm3", "g/cc", "g/cm^3", "gcc"}:
            if median > 100.0:
                return _with_stats(
                    clean,
                    positive,
                    original_unit=original_unit,
                    standard_unit="g/cm3",
                    conversion_action="keep_g_cm3",
                    hard_fail_reason="density_g_cm3_unit_impossible_median_gt_100",
                )
            if median > 10.0:
                qc_flags.append("density_g_cm3_median_high")
            return _with_stats(
                clean,
                positive,
                original_unit=original_unit,
                standard_unit="g/cm3",
                conversion_action="keep_g_cm3",
                qc_flags=qc_flags,
            )
        if unit_norm in {"kg/m3", "kg/m^3"}:
            if median < 10.0:
                return _with_stats(
                    clean,
                    positive / 1000.0,
                    original_unit=original_unit,
                    standard_unit="g/cm3",
                    conversion_action="kg_m3_to_g_cm3",
                    hard_fail_reason="density_kg_m3_unit_impossible_median_lt_10",
                )
            return _with_stats(
                clean,
                positive / 1000.0,
                original_unit=original_unit,
                standard_unit="g/cm3",
                conversion_action="kg_m3_to_g_cm3",
            )
        return _with_stats(
            clean,
            clean.copy(),
            original_unit=original_unit,
            standard_unit="g/cm3",
            conversion_action="unsupported_unit",
            hard_fail_reason="unsupported_density_unit",
        )

    return _with_stats(
        clean,
        clean.copy(),
        original_unit=original_unit,
        standard_unit=original_unit,
        conversion_action="kept_original_unit",
    )


def replace_constant_runs(
    md: np.ndarray,
    values: np.ndarray,
    *,
    min_run_length: int,
    exclude: bool = False,
) -> tuple[np.ndarray, list[ConstantRun], int]:
    """Replace strict constant-value runs with NaN, optionally reporting skips."""
    if min_run_length < 1:
        raise ValueError("min_run_length must be >= 1.")
    basis = np.asarray(md, dtype=float)
    cleaned = np.asarray(values, dtype=float).copy()
    reports: list[ConstantRun] = []
    replaced = 0
    i = 0
    n = cleaned.size
    while i < n:
        if not np.isfinite(cleaned[i]):
            i += 1
            continue
        j = i + 1
        while j < n and np.isfinite(cleaned[j]) and cleaned[j] == cleaned[i]:
            j += 1
        run_len = j - i
        if run_len >= min_run_length:
            action = "skip_caliper" if exclude else "set_null"
            reports.append(
                ConstantRun(
                    start_md=float(basis[i]),
                    end_md=float(basis[j - 1]),
                    run_length=int(run_len),
                    constant_value=float(cleaned[i]),
                    action=action,
                )
            )
            if not exclude:
                cleaned[i:j] = np.nan
                replaced += run_len
        i = j
    return cleaned, reports, replaced


def compute_global_quantile_thresholds(
    curves_by_standard: Mapping[str, Sequence[np.ndarray]],
    *,
    lower_quantile: float,
    upper_quantile: float,
    min_samples: int,
) -> dict[str, CurveThreshold]:
    """Compute per-standard-curve global quantile thresholds."""
    thresholds: dict[str, CurveThreshold] = {}
    for standard, arrays in curves_by_standard.items():
        finite_parts = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in arrays]
        finite_parts = [values for values in finite_parts if values.size]
        if finite_parts:
            merged = np.concatenate(finite_parts)
        else:
            merged = np.asarray([], dtype=float)
        if merged.size < min_samples:
            thresholds[standard] = CurveThreshold(
                standard_mnemonic=standard,
                lower=None,
                upper=None,
                source="skipped_insufficient_samples",
                sample_count=int(merged.size),
            )
            continue
        thresholds[standard] = CurveThreshold(
            standard_mnemonic=standard,
            lower=float(np.nanquantile(merged, lower_quantile)),
            upper=float(np.nanquantile(merged, upper_quantile)),
            source="global_quantile",
            sample_count=int(merged.size),
        )
    return thresholds


def threshold_from_overrides(
    standard_mnemonic: str,
    *,
    well_name: str,
    overrides: Mapping[str, Any] | None,
    auto_thresholds: Mapping[str, CurveThreshold],
) -> CurveThreshold:
    """Resolve well-curve, global, then automatic threshold."""
    overrides = overrides or {}
    well_curve = overrides.get("well_curve", {})
    if isinstance(well_curve, Mapping):
        for candidate_well, curves in well_curve.items():
            if str(candidate_well).strip().casefold() != str(well_name).strip().casefold():
                continue
            if isinstance(curves, Mapping) and standard_mnemonic in curves and isinstance(curves[standard_mnemonic], Mapping):
                spec = curves[standard_mnemonic]
                return CurveThreshold(
                    standard_mnemonic=standard_mnemonic,
                    lower=_optional_float(spec.get("min")),
                    upper=_optional_float(spec.get("max")),
                    source="manual_well_curve",
                    sample_count=0,
                )
    global_spec = overrides.get("global", {})
    if isinstance(global_spec, Mapping) and standard_mnemonic in global_spec and isinstance(global_spec[standard_mnemonic], Mapping):
        spec = global_spec[standard_mnemonic]
        return CurveThreshold(
            standard_mnemonic=standard_mnemonic,
            lower=_optional_float(spec.get("min")),
            upper=_optional_float(spec.get("max")),
            source="manual_global",
            sample_count=0,
        )
    return auto_thresholds.get(
        standard_mnemonic,
        CurveThreshold(standard_mnemonic=standard_mnemonic, lower=None, upper=None, source="missing_threshold", sample_count=0),
    )


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def remove_outliers(values: np.ndarray, threshold: CurveThreshold) -> OutlierRemoval:
    """Set values outside a resolved threshold to NaN."""
    cleaned = np.asarray(values, dtype=float).copy()
    if threshold.lower is None and threshold.upper is None:
        return OutlierRemoval(values=cleaned, replaced_points=0, lower=None, upper=None, threshold_source=threshold.source)
    mask = np.zeros(cleaned.shape, dtype=bool)
    finite = np.isfinite(cleaned)
    if threshold.lower is not None:
        mask |= finite & (cleaned < float(threshold.lower))
    if threshold.upper is not None:
        mask |= finite & (cleaned > float(threshold.upper))
    replaced = int(mask.sum())
    cleaned[mask] = np.nan
    return OutlierRemoval(
        values=cleaned,
        replaced_points=replaced,
        lower=threshold.lower,
        upper=threshold.upper,
        threshold_source=threshold.source,
    )


def is_curve_usable(
    values: np.ndarray,
    *,
    initial_valid_count: int,
    min_valid_samples: int,
    min_valid_fraction_of_initial: float,
) -> tuple[bool, str, int, float]:
    """Evaluate whether a cleaned curve remains usable."""
    final_count = int(np.isfinite(np.asarray(values, dtype=float)).sum())
    if initial_valid_count <= 0:
        return False, "no_initial_valid_samples", final_count, 0.0
    fraction = float(final_count / initial_valid_count)
    if final_count < int(min_valid_samples):
        return False, "insufficient_valid_samples", final_count, fraction
    if fraction < float(min_valid_fraction_of_initial):
        return False, "valid_fraction_below_threshold", final_count, fraction
    return True, "", final_count, fraction
