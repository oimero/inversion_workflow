"""Preprocess selected LAS logs for the time-domain workflow.

This script consumes the second workflow step output. It standardizes curve
mnemonics, converts supported units, removes strict constant runs and outliers,
then exports traceable preprocessed LAS files.

Usage::

    python scripts/log_preprocess.py
    python scripts/log_preprocess.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from datetime import datetime
from typing import Any, Mapping, Sequence

import lasio
import numpy as np
import pandas as pd

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.well.assets import normalize_well_name
from cup.well.curves import exact_mnemonic, normalize_mnemonic
from cup.well.las import _header_value
from cup.well.preprocess import (
    DEFAULT_MISSING_SENTINELS,
    CurveThreshold,
    UnitStandardization,
    compute_global_quantile_thresholds,
    finite_stats,
    is_curve_usable,
    remove_outliers,
    replace_constant_runs,
    standard_mnemonic_for_category,
    standardize_curve_unit,
    threshold_from_overrides,
    values_to_nan,
)


# =============================================================================
# CLI and config
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common.yaml"),
        help="Time-domain common config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/log_preprocess_<timestamp>.",
    )
    return parser.parse_args()


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("log_preprocess") or {})
    script_cfg.setdefault("screen_file", None)
    script_cfg.setdefault("input_las_dir", None)
    script_cfg.setdefault("curve_inventory_file", None)
    script_cfg.setdefault("classification_dir", None)
    script_cfg.setdefault("output_las_dir", "preprocessed_las")
    script_cfg.setdefault("required_categories", ["p_sonic", "density"])
    script_cfg.setdefault(
        "selected_categories",
        [
            "caliper",
            "gamma_ray",
            "s_sonic",
            "p_sonic",
            "density",
            "resistivity",
            "spontaneous_potential",
            "porosity",
            "permeability",
            "water_saturation",
        ],
    )
    script_cfg.setdefault("mnemonic_standardization", {"enabled": True})
    script_cfg.setdefault("unit_standardization", {"enabled": True, "unit_mismatch_qc": True})
    script_cfg.setdefault(
        "constant_runs",
        {
            "enabled": True,
            "min_run_length": 8,
            "min_run_length_by_category": {
                "p_sonic": 16,
                "s_sonic": 16,
                "density": 16,
                "gamma_ray": 16,
                "resistivity": 16,
                "spontaneous_potential": 16,
                "porosity": 16,
                "permeability": 16,
                "water_saturation": 16,
            },
            "replacement": None,
            "exclude_categories": ["caliper"],
        },
    )
    script_cfg.setdefault(
        "outliers",
        {
            "enabled": True,
            "strategy": "global_quantile_with_override",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "replacement": None,
            "range_override_file": "experiments/log_preprocess_ranges.yaml",
            "min_samples_for_auto_threshold": 1000,
        },
    )
    script_cfg.setdefault(
        "usable_thresholds",
        {"min_valid_samples": 100, "min_valid_fraction_of_initial": 0.70},
    )
    script_cfg.setdefault("export", {"null_value": -999.25, "write_fmt": "%.6f"})
    script_cfg.setdefault("missing_sentinel_values", list(DEFAULT_MISSING_SENTINELS))
    return script_cfg


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"log_preprocess_{timestamp}"


def _discover_latest_screen_dir(cfg: dict[str, Any]) -> Path:
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    all_candidates = [path for path in output_root.glob("las_curve_screen_*") if path.is_dir()]
    timestamped = [
        path
        for path in all_candidates
        if re.fullmatch(r"las_curve_screen_\d{8}_\d{6}", path.name)
    ]
    candidates = sorted(timestamped or all_candidates, key=lambda path: path.name)
    if not candidates:
        raise FileNotFoundError("No las_curve_screen_* output directory found under output_root.")
    return candidates[-1]


def _resolve_inputs(cfg: dict[str, Any], script_cfg: dict[str, Any]) -> dict[str, Path]:
    latest_dir = _discover_latest_screen_dir(cfg)
    screen_file = (
        latest_dir / "well_curve_screen.csv"
        if script_cfg.get("screen_file") is None
        else _resolve_repo_path(script_cfg["screen_file"])
    )
    input_las_dir = (
        latest_dir / "selected_las"
        if script_cfg.get("input_las_dir") is None
        else _resolve_repo_path(script_cfg["input_las_dir"])
    )
    curve_inventory_file = (
        latest_dir / "las_curve_inventory.csv"
        if script_cfg.get("curve_inventory_file") is None
        else _resolve_repo_path(script_cfg["curve_inventory_file"])
    )
    classification_dir = (
        latest_dir / "curve_classification"
        if script_cfg.get("classification_dir") is None
        else _resolve_repo_path(script_cfg["classification_dir"])
    )
    return {
        "screen_file": screen_file,
        "input_las_dir": input_las_dir,
        "curve_inventory_file": curve_inventory_file,
        "classification_dir": classification_dir,
    }


def _load_range_overrides(script_cfg: dict[str, Any]) -> dict[str, Any]:
    outlier_cfg = dict(script_cfg.get("outliers") or {})
    path_value = outlier_cfg.get("range_override_file")
    if not path_value:
        return {}
    path = _resolve_repo_path(path_value)
    if not path.exists():
        return {}
    return load_yaml_config(path)


# =============================================================================
# LAS helpers
# =============================================================================


@dataclass
class RawCurve:
    well_name: str
    category: str
    original_mnemonic: str
    original_unit: str
    description: str
    index: int
    is_step2_primary: bool
    md: np.ndarray
    values: np.ndarray
    source_las: Path
    index_mnemonic: str
    index_unit: str
    las_null_value: float | None


@dataclass
class CandidateCurve:
    raw: RawCurve
    standard_mnemonic: str
    standard_unit: str
    conversion_action: str
    unit_hard_fail_reason: str
    unit_qc_flags: tuple[str, ...]
    initial_valid_count: int
    after_constant_values: np.ndarray
    constant_replaced_points: int
    final_values: np.ndarray | None = None
    outlier_replaced_points: int = 0
    threshold: CurveThreshold | None = None
    usable: bool = False
    unusable_reason: str = ""
    final_valid_count: int = 0
    final_valid_fraction: float = 0.0


def _optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _curve_index(las: lasio.LASFile) -> tuple[dict[str, int], dict[str, list[int]]]:
    by_exact = {exact_mnemonic(curve.mnemonic): index for index, curve in enumerate(las.curves)}
    by_norm: dict[str, list[int]] = {}
    for index, curve in enumerate(las.curves):
        by_norm.setdefault(normalize_mnemonic(curve.mnemonic), []).append(index)
    return by_exact, by_norm


def _resolve_curve_index(by_exact: Mapping[str, int], by_norm: Mapping[str, list[int]], mnemonic: str) -> int | None:
    exact = exact_mnemonic(mnemonic)
    if exact in by_exact:
        return int(by_exact[exact])
    candidates = by_norm.get(normalize_mnemonic(mnemonic), [])
    if candidates:
        return int(candidates[0])
    return None


def _load_raw_curves_for_well(
    *,
    well_name: str,
    las_file: Path,
    classification_file: Path,
    selected_categories: Sequence[str],
) -> list[RawCurve]:
    payload = json.loads(classification_file.read_text(encoding="utf-8"))
    las = lasio.read(str(las_file))
    data = np.asarray(las.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(las.curves):
        raise ValueError(f"LAS data shape does not match curve headers: {las_file}")
    by_exact, by_norm = _curve_index(las)
    index_curve = las.curves[0]
    md = np.asarray(data[:, 0], dtype=float)
    null_value = _optional_float(_header_value(las, "well", "NULL"))
    selected = set(str(item) for item in selected_categories)
    raw: list[RawCurve] = []
    for item in payload.get("curves", []):
        category = str(item.get("category", ""))
        if category not in selected:
            continue
        if category in {"unclassified", "ambiguous", "disabled"}:
            continue
        index = _resolve_curve_index(by_exact, by_norm, str(item.get("mnemonic", "")))
        # Curve index 0 is the LAS depth/index axis, not a measured log curve.
        if index is None or index == 0:
            continue
        curve = las.curves[index]
        raw.append(
            RawCurve(
                well_name=well_name,
                category=category,
                original_mnemonic=str(curve.mnemonic),
                original_unit=str(curve.unit or ""),
                description=str(curve.descr or ""),
                index=int(index),
                is_step2_primary=bool(item.get("is_primary", False)),
                md=md,
                values=np.asarray(data[:, index], dtype=float),
                source_las=las_file,
                index_mnemonic=str(index_curve.mnemonic),
                index_unit=str(index_curve.unit or ""),
                las_null_value=null_value,
            )
        )
    raw.sort(key=lambda curve: (curve.category, not curve.is_step2_primary, curve.index))
    return raw


def _write_preprocessed_las(
    *,
    output_las: Path,
    template_las: Path,
    md: np.ndarray,
    index_mnemonic: str,
    index_unit: str,
    curves: Sequence[CandidateCurve],
    null_value: float,
    write_fmt: str,
) -> None:
    source = lasio.read(str(template_las), ignore_data=True)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    out = lasio.LASFile()
    for item in source.well:
        out.well[item.mnemonic] = item
    out.well["NULL"].value = float(null_value)
    out.append_curve(index_mnemonic, np.asarray(md, dtype=float), unit=index_unit, descr="Measured depth")
    for curve in curves:
        if curve.final_values is None:
            continue
        values = np.asarray(curve.final_values, dtype=float).copy()
        values[~np.isfinite(values)] = float(null_value)
        out.append_curve(
            curve.standard_mnemonic,
            values,
            unit=curve.standard_unit,
            descr=f"{curve.raw.category} from {curve.raw.original_mnemonic}",
        )
    out.write(str(output_las), version=2.0, wrap=False, fmt=write_fmt)


# =============================================================================
# Preprocess orchestration
# =============================================================================


def _min_run_length(category: str, constant_cfg: Mapping[str, Any]) -> int:
    by_category = constant_cfg.get("min_run_length_by_category", {})
    if isinstance(by_category, Mapping) and category in by_category:
        return int(by_category[category])
    return int(constant_cfg.get("min_run_length", 8))


def _section_enabled(config: Mapping[str, Any], key: str, *, default: bool = True) -> bool:
    value = config.get(key, {})
    if isinstance(value, Mapping):
        return bool(value.get("enabled", default))
    return default


def _disabled_unit_standardization(values: np.ndarray, original_unit: str) -> UnitStandardization:
    stats = finite_stats(values)
    return UnitStandardization(
        values=np.asarray(values, dtype=float).copy(),
        original_unit=original_unit,
        standard_unit=original_unit,
        conversion_action="disabled",
        input_valid_count=int(stats["valid_count"]),
        output_valid_count=int(stats["valid_count"]),
        input_median=stats["median"],
        output_median=stats["median"],
        input_p01=stats["p01"],
        input_p99=stats["p99"],
        output_p01=stats["p01"],
        output_p99=stats["p99"],
    )


def _prepare_candidate(
    raw: RawCurve,
    *,
    script_cfg: dict[str, Any],
    constant_run_rows: list[dict[str, Any]],
    unit_report_rows: list[dict[str, Any]],
    unit_qc_rows: list[dict[str, Any]],
    skipped_curve_rows: list[dict[str, Any]],
) -> CandidateCurve:
    standard = (
        standard_mnemonic_for_category(raw.category)
        if _section_enabled(script_cfg, "mnemonic_standardization", default=True)
        else raw.original_mnemonic
    )
    sentinels = [float(item) for item in script_cfg.get("missing_sentinel_values", DEFAULT_MISSING_SENTINELS)]
    missing_clean = values_to_nan(raw.values, null_value=raw.las_null_value, sentinels=sentinels)
    unit_result = (
        standardize_curve_unit(missing_clean, category=raw.category, unit=raw.original_unit)
        if _section_enabled(script_cfg, "unit_standardization", default=True)
        else _disabled_unit_standardization(missing_clean, raw.original_unit)
    )
    unit_row = {
        "well_name": raw.well_name,
        "category": raw.category,
        "original_mnemonic": raw.original_mnemonic,
        "standard_mnemonic": standard,
        **unit_result.report_row(),
    }
    unit_report_rows.append(unit_row)
    if unit_result.qc_flags:
        unit_qc_rows.append(
            {
                "well_name": raw.well_name,
                "category": raw.category,
                "original_mnemonic": raw.original_mnemonic,
                "standard_mnemonic": standard,
                "original_unit": raw.original_unit,
                "qc_flags": ";".join(unit_result.qc_flags),
                "input_median": unit_result.input_median,
                "input_p01": unit_result.input_p01,
                "input_p99": unit_result.input_p99,
                "action": "report_only",
            }
        )

    converted = unit_result.values
    constant_cfg = dict(script_cfg.get("constant_runs") or {})
    exclude_categories = {str(item) for item in constant_cfg.get("exclude_categories", [])}
    constant_replaced = 0
    if bool(constant_cfg.get("enabled", True)) and not unit_result.hard_fail_reason:
        converted, runs, constant_replaced = replace_constant_runs(
            raw.md,
            converted,
            min_run_length=_min_run_length(raw.category, constant_cfg),
            exclude=raw.category in exclude_categories,
        )
        for run in runs:
            constant_run_rows.append(
                {
                    "well_name": raw.well_name,
                    "original_mnemonic": raw.original_mnemonic,
                    "standard_mnemonic": standard,
                    "category": raw.category,
                    **run.to_row(),
                }
            )

    candidate = CandidateCurve(
        raw=raw,
        standard_mnemonic=standard,
        standard_unit=unit_result.standard_unit,
        conversion_action=unit_result.conversion_action,
        unit_hard_fail_reason=unit_result.hard_fail_reason,
        unit_qc_flags=unit_result.qc_flags,
        initial_valid_count=int(np.isfinite(unit_result.values).sum()),
        after_constant_values=converted,
        constant_replaced_points=int(constant_replaced),
    )
    if unit_result.hard_fail_reason:
        skipped_curve_rows.append(
            {
                "well_name": raw.well_name,
                "original_mnemonic": raw.original_mnemonic,
                "standard_mnemonic": standard,
                "category": raw.category,
                "reason": unit_result.hard_fail_reason,
            }
        )
    return candidate


def _candidate_summary_row(candidate: CandidateCurve) -> dict[str, Any]:
    final_values = candidate.final_values if candidate.final_values is not None else candidate.after_constant_values
    stats = finite_stats(final_values)
    threshold = candidate.threshold
    return {
        "well_name": candidate.raw.well_name,
        "category": candidate.raw.category,
        "original_mnemonic": candidate.raw.original_mnemonic,
        "standard_mnemonic": candidate.standard_mnemonic,
        "is_step2_primary": candidate.raw.is_step2_primary,
        "is_final": candidate.usable,
        "original_unit": candidate.raw.original_unit,
        "standard_unit": candidate.standard_unit,
        "conversion_action": candidate.conversion_action,
        "unit_hard_fail_reason": candidate.unit_hard_fail_reason,
        "unit_qc_flags": ";".join(candidate.unit_qc_flags),
        "initial_valid_count": candidate.initial_valid_count,
        "constant_replaced_points": candidate.constant_replaced_points,
        "outlier_replaced_points": candidate.outlier_replaced_points,
        "final_valid_count": candidate.final_valid_count,
        "final_valid_fraction": candidate.final_valid_fraction,
        "usable": candidate.usable,
        "unusable_reason": candidate.unusable_reason,
        "threshold_lower": None if threshold is None else threshold.lower,
        "threshold_upper": None if threshold is None else threshold.upper,
        "threshold_source": "" if threshold is None else threshold.source,
        "final_median": stats["median"],
        "final_p01": stats["p01"],
        "final_p99": stats["p99"],
    }


def run_preprocess(
    *,
    screen_file: Path,
    input_las_dir: Path,
    curve_inventory_file: Path,
    classification_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    range_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    screen_df = pd.read_csv(screen_file)
    curve_inventory_df = pd.read_csv(curve_inventory_file)
    required_categories = [str(item) for item in config["required_categories"]]
    selected_categories = [str(item) for item in config["selected_categories"]]

    output_las_dir = output_dir / str(config.get("output_las_dir", "preprocessed_las"))
    output_las_dir.mkdir(parents=True, exist_ok=True)

    constant_run_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    preprocess_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    unit_report_rows: list[dict[str, Any]] = []
    unit_qc_rows: list[dict[str, Any]] = []
    reselection_rows: list[dict[str, Any]] = []
    skipped_well_rows: list[dict[str, Any]] = []
    skipped_curve_rows: list[dict[str, Any]] = []

    candidates_by_well: dict[str, list[CandidateCurve]] = {}
    auto_threshold_samples: dict[str, list[np.ndarray]] = {}

    for _, row in screen_df.sort_values("well_name").iterrows():
        well_name = str(row["well_name"])
        status = str(row.get("screen_status", ""))
        if status != "passed":
            skipped_well_rows.append({"well_name": well_name, "reason": f"upstream_screen_status_{status}"})
            continue
        exported = str(row.get("exported_las", "")).strip()
        if not exported:
            skipped_well_rows.append({"well_name": well_name, "reason": "upstream_exported_las_missing"})
            continue
        exported_path = _resolve_repo_path(exported)
        if not exported_path.exists() and (input_las_dir / Path(exported).name).exists():
            exported_path = input_las_dir / Path(exported).name
        if not exported_path.exists():
            skipped_well_rows.append({"well_name": well_name, "reason": "selected_las_missing"})
            continue
        raw_las_path = _resolve_repo_path(str(row.get("las_file", "")))
        class_file = classification_dir / f"{sanitize_filename(well_name)}.json"
        if not raw_las_path.exists() or not class_file.exists():
            skipped_well_rows.append({"well_name": well_name, "reason": "raw_las_or_classification_missing"})
            continue
        try:
            raw_curves = _load_raw_curves_for_well(
                well_name=well_name,
                las_file=raw_las_path,
                classification_file=class_file,
                selected_categories=selected_categories,
            )
            prepared: list[CandidateCurve] = []
            for raw in raw_curves:
                candidate = _prepare_candidate(
                    raw,
                    script_cfg=config,
                    constant_run_rows=constant_run_rows,
                    unit_report_rows=unit_report_rows,
                    unit_qc_rows=unit_qc_rows,
                    skipped_curve_rows=skipped_curve_rows,
                )
                prepared.append(candidate)
                mapping_rows.append(
                    {
                        "well_name": well_name,
                        "category": raw.category,
                        "original_mnemonic": raw.original_mnemonic,
                        "standard_mnemonic": candidate.standard_mnemonic,
                        "is_step2_primary": raw.is_step2_primary,
                        "source_las": repo_relative_path(raw.source_las, root=REPO_ROOT),
                    }
                )
                if raw.is_step2_primary and not candidate.unit_hard_fail_reason:
                    auto_threshold_samples.setdefault(candidate.standard_mnemonic, []).append(candidate.after_constant_values)
            candidates_by_well[well_name] = prepared
        except Exception as exc:
            skipped_well_rows.append({"well_name": well_name, "reason": str(exc)})

    outlier_cfg = dict(config.get("outliers") or {})
    if bool(outlier_cfg.get("enabled", True)):
        auto_thresholds = compute_global_quantile_thresholds(
            auto_threshold_samples,
            lower_quantile=float(outlier_cfg.get("lower_quantile", 0.01)),
            upper_quantile=float(outlier_cfg.get("upper_quantile", 0.99)),
            min_samples=int(outlier_cfg.get("min_samples_for_auto_threshold", 1000)),
        )
    else:
        auto_thresholds = {
            standard: CurveThreshold(standard_mnemonic=standard, lower=None, upper=None, source="disabled", sample_count=0)
            for standard in auto_threshold_samples
        }

    threshold_rows = [threshold.to_row() for threshold in sorted(auto_thresholds.values(), key=lambda item: item.standard_mnemonic)]
    usable_cfg = dict(config.get("usable_thresholds") or {})
    min_valid_samples = int(usable_cfg.get("min_valid_samples", 100))
    min_valid_fraction = float(usable_cfg.get("min_valid_fraction_of_initial", 0.70))

    for well_name, candidates in candidates_by_well.items():
        by_category: dict[str, list[CandidateCurve]] = {}
        for candidate in candidates:
            by_category.setdefault(candidate.raw.category, []).append(candidate)
        final_by_category: dict[str, CandidateCurve] = {}

        for category in selected_categories:
            ordered = sorted(by_category.get(category, []), key=lambda item: (not item.raw.is_step2_primary, item.raw.index))
            primary_name = ordered[0].raw.original_mnemonic if ordered else ""
            primary_failure = ""
            for candidate in ordered:
                if candidate.unit_hard_fail_reason:
                    candidate.unusable_reason = candidate.unit_hard_fail_reason
                    candidate.final_values = candidate.after_constant_values
                    continue
                threshold = threshold_from_overrides(
                    candidate.standard_mnemonic,
                    well_name=well_name,
                    overrides=range_overrides,
                    auto_thresholds=auto_thresholds,
                )
                outlier = (
                    remove_outliers(candidate.after_constant_values, threshold)
                    if bool(outlier_cfg.get("enabled", True))
                    else remove_outliers(
                        candidate.after_constant_values,
                        CurveThreshold(candidate.standard_mnemonic, None, None, "disabled", 0),
                    )
                )
                candidate.threshold = threshold
                candidate.final_values = outlier.values
                candidate.outlier_replaced_points = outlier.replaced_points
                usable, reason, final_count, fraction = is_curve_usable(
                    outlier.values,
                    initial_valid_count=candidate.initial_valid_count,
                    min_valid_samples=min_valid_samples,
                    min_valid_fraction_of_initial=min_valid_fraction,
                )
                candidate.usable = usable
                candidate.unusable_reason = reason
                candidate.final_valid_count = final_count
                candidate.final_valid_fraction = fraction
                outlier_rows.append(
                    {
                        "well_name": well_name,
                        "category": category,
                        "original_mnemonic": candidate.raw.original_mnemonic,
                        "standard_mnemonic": candidate.standard_mnemonic,
                        "lower": outlier.lower,
                        "upper": outlier.upper,
                        "threshold_source": outlier.threshold_source,
                        "replaced_points": outlier.replaced_points,
                    }
                )
                if usable:
                    final_by_category[category] = candidate
                    if not candidate.raw.is_step2_primary:
                        reselection_rows.append(
                            {
                                "well_name": well_name,
                                "category": category,
                                "failed_primary": primary_name,
                                "replacement": candidate.raw.original_mnemonic,
                                "standard_mnemonic": candidate.standard_mnemonic,
                                "reason": primary_failure or "primary_unusable",
                            }
                        )
                    break
                if candidate.raw.is_step2_primary:
                    primary_failure = reason or candidate.unit_hard_fail_reason
                skipped_curve_rows.append(
                    {
                        "well_name": well_name,
                        "original_mnemonic": candidate.raw.original_mnemonic,
                        "standard_mnemonic": candidate.standard_mnemonic,
                        "category": category,
                        "reason": reason or candidate.unit_hard_fail_reason,
                    }
                )

        for candidate in candidates:
            preprocess_rows.append(_candidate_summary_row(candidate))

        missing_required = [category for category in required_categories if category not in final_by_category]
        reasons = [f"missing_final_{category}" for category in missing_required]
        status = "passed" if not missing_required else "failed"
        preprocessed_las = ""
        final_curves = [final_by_category[category] for category in selected_categories if category in final_by_category]
        if status == "passed" and final_curves:
            first = final_curves[0].raw
            output_las = output_las_dir / f"{sanitize_filename(well_name)}.las"
            try:
                _write_preprocessed_las(
                    output_las=output_las,
                    template_las=first.source_las,
                    md=first.md,
                    index_mnemonic=first.index_mnemonic,
                    index_unit=first.index_unit,
                    curves=final_curves,
                    null_value=float(config.get("export", {}).get("null_value", -999.25)),
                    write_fmt=str(config.get("export", {}).get("write_fmt", "%.6f")),
                )
                preprocessed_las = repo_relative_path(output_las, root=REPO_ROOT)
            except Exception as exc:
                status = "failed"
                reasons.append(f"export_failed:{exc}")
                skipped_well_rows.append({"well_name": well_name, "reason": reasons[-1]})
        elif status != "passed":
            skipped_well_rows.append({"well_name": well_name, "reason": ";".join(reasons)})

        well_rows.append(
            {
                "well_name": well_name,
                "preprocess_status": status,
                "usable_p_sonic": "p_sonic" in final_by_category,
                "usable_density": "density" in final_by_category,
                "usable_caliper": "caliper" in final_by_category,
                "final_p_sonic": final_by_category.get("p_sonic").standard_mnemonic if "p_sonic" in final_by_category else "",
                "final_density": final_by_category.get("density").standard_mnemonic if "density" in final_by_category else "",
                "final_caliper": final_by_category.get("caliper").standard_mnemonic if "caliper" in final_by_category else "",
                "preprocessed_las": preprocessed_las,
                "reasons": ";".join(reasons),
            }
        )

    paths = {
        "preprocess_summary_csv": output_dir / "preprocess_summary.csv",
        "well_preprocess_status_csv": output_dir / "well_preprocess_status.csv",
        "mnemonic_mapping_csv": output_dir / "mnemonic_mapping.csv",
        "unit_conversion_report_csv": output_dir / "unit_conversion_report.csv",
        "unit_mismatch_qc_csv": output_dir / "unit_mismatch_qc.csv",
        "primary_reselection_report_csv": output_dir / "primary_reselection_report.csv",
        "constant_run_report_csv": output_dir / "constant_run_report.csv",
        "outlier_report_csv": output_dir / "outlier_report.csv",
        "range_thresholds_csv": output_dir / "range_thresholds.csv",
        "skipped_wells_csv": output_dir / "skipped_wells.csv",
        "skipped_curves_csv": output_dir / "skipped_curves.csv",
        "run_summary_json": output_dir / "run_summary.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(paths["preprocess_summary_csv"], preprocess_rows)
    _write_csv(paths["well_preprocess_status_csv"], well_rows)
    _write_csv(paths["mnemonic_mapping_csv"], mapping_rows)
    _write_csv(paths["unit_conversion_report_csv"], unit_report_rows)
    _write_csv(paths["unit_mismatch_qc_csv"], unit_qc_rows)
    _write_csv(paths["primary_reselection_report_csv"], reselection_rows)
    _write_csv(paths["constant_run_report_csv"], constant_run_rows)
    _write_csv(paths["outlier_report_csv"], outlier_rows)
    _write_csv(paths["range_thresholds_csv"], threshold_rows)
    _write_csv(paths["skipped_wells_csv"], skipped_well_rows, columns=["well_name", "reason"])
    _write_csv(
        paths["skipped_curves_csv"],
        skipped_curve_rows,
        columns=["well_name", "original_mnemonic", "standard_mnemonic", "category", "reason"],
    )

    status_df = pd.DataFrame.from_records(well_rows)
    status_counts = (
        status_df["preprocess_status"].value_counts(dropna=False).astype(int).to_dict()
        if not status_df.empty
        else {}
    )
    summary = {
        "script": "log_preprocess.py",
        "inputs": {
            "screen_file": repo_relative_path(screen_file, root=REPO_ROOT),
            "selected_las_dir": repo_relative_path(input_las_dir, root=REPO_ROOT),
            "curve_inventory_file": repo_relative_path(curve_inventory_file, root=REPO_ROOT),
            "classification_dir": repo_relative_path(classification_dir, root=REPO_ROOT),
            "raw_las_source": "well_curve_screen.las_file",
        },
        "input_contract": {
            "selected_las_dir": "checked for upstream exported LAS presence",
            "raw_las_source": "used to load primary and same-category fallback curves",
        },
        "curve_inventory_row_count": int(len(curve_inventory_df)),
        "mnemonic_standardization_enabled": _section_enabled(config, "mnemonic_standardization", default=True),
        "unit_standardization_enabled": _section_enabled(config, "unit_standardization", default=True),
        "selected_categories": selected_categories,
        "required_categories": required_categories,
        "candidate_well_count": int((screen_df["screen_status"].astype(str) == "passed").sum()),
        "preprocess_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "exported_las_count": int((status_df["preprocessed_las"].astype(str) != "").sum()) if not status_df.empty else 0,
        "unit_hard_fail_count": int(sum(1 for row in unit_report_rows if row.get("hard_fail_reason"))),
        "unit_qc_count": int(len(unit_qc_rows)),
        "constant_run_count": int(len(constant_run_rows)),
        "outlier_replaced_points": int(sum(int(row.get("replaced_points", 0) or 0) for row in outlier_rows)),
        "primary_reselection_count": int(len(reselection_rows)),
        "paths": {key: repo_relative_path(path, root=REPO_ROOT) for key, path in paths.items()},
    }
    write_json(paths["run_summary_json"], summary)
    return {"paths": paths, "summary": summary}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows, columns=columns).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _script_config(cfg)
    inputs = _resolve_inputs(cfg, script_cfg)
    output_dir = _resolve_output_dir(args, cfg)
    range_overrides = _load_range_overrides(script_cfg)
    result = run_preprocess(
        screen_file=inputs["screen_file"],
        input_las_dir=inputs["input_las_dir"],
        curve_inventory_file=inputs["curve_inventory_file"],
        classification_dir=inputs["classification_dir"],
        output_dir=output_dir,
        config=script_cfg,
        range_overrides=range_overrides,
    )
    summary = result["summary"]
    for label, path in result["paths"].items():
        print(f"Saved {label}: {path}")
    print(
        "Log preprocess summary: "
        f"{summary['candidate_well_count']} step2-passed wells, "
        f"{summary['preprocess_status_counts'].get('passed', 0)} passed, "
        f"{summary['preprocess_status_counts'].get('failed', 0)} failed, "
        f"{summary['exported_las_count']} LAS exported."
    )


if __name__ == "__main__":
    main()
