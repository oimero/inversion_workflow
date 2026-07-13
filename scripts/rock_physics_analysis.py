"""Run modular rock-physics analysis from Step 3 LAS outputs.

The input layer always audits every ``preprocess_status=passed`` LAS.  Analysis
modules are explicitly enabled in the common config.  With every module
disabled, the script performs a successful input-only run.

Usage::

    python scripts/rock_physics_analysis.py
    python scripts/rock_physics_analysis.py --config experiments/common/common.yaml
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Mapping

import lasio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.sources import resolve_source_run
from cup.config.workflow import WorkflowConfig
from cup.physics.numpy_backend import forward_time
from cup.physics.rock_physics import WellAiVpSamples, fit_equal_well_huber, well_fit_metrics
from cup.synthetic.schemas import FORWARD_MODEL_INPUTS_SCHEMA_VERSION, ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    load_yaml_config,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    resolve_timestamped_output_dir,
    write_json,
)


SCHEMA_VERSION = ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION
SCRIPT_VERSION = 1
RELATION_SCHEMA = "rock_physics_relation_v1"
FORWARD_INPUTS_SCHEMA = FORWARD_MODEL_INPUTS_SCHEMA_VERSION
DEFAULT_COMMON_CONFIG = Path("experiments/common/common.yaml")
KNOWN_MODULES = {"ai_vp_linear"}
EXPECTED_CURVES = {
    "DT_USM": "us/m",
    "RHO_GCC": "g/cm3",
    "AI": "m/s*g/cm3",
}
AI_CONSISTENCY_RTOL = 1e-5
AI_CONSISTENCY_ATOL = 1e-3


@dataclass(frozen=True)
class AiVpModuleConfig:
    enabled: bool
    min_valid_samples_per_well: int | None = None
    min_valid_wells: int | None = None
    huber_delta_sigma: float | None = None


@dataclass
class LoadedLas:
    well_id: str
    path: Path
    las: lasio.LASFile


class AiVpWellRejected(ValueError):
    """A module-level well rejection carrying the QC computed before rejection."""

    def __init__(self, reason: str, qc: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.qc = dict(qc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}.")


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive integer.") from exc
    if result <= 0 or result != float(value):
        raise ValueError(f"{path} must be a positive integer.")
    return result


def _positive_float(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive number.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be a positive number.")
    return result


def _required_text(mapping: Mapping[str, Any], key: str, *, path: str) -> str:
    value = mapping.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


def _parse_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(config.get("rock_physics_analysis"), path="rock_physics_analysis")
    _reject_unknown(root, {"source_runs", "modules", "forward_model"}, path="rock_physics_analysis")

    source_runs = _mapping(root.get("source_runs", {}), path="rock_physics_analysis.source_runs")
    _reject_unknown(source_runs, {"well_preprocess_dir"}, path="rock_physics_analysis.source_runs")
    explicit_source = str(source_runs.get("well_preprocess_dir") or "").strip()

    modules = _mapping(root.get("modules"), path="rock_physics_analysis.modules")
    unknown_modules = sorted(set(modules) - KNOWN_MODULES)
    if unknown_modules:
        raise ValueError(f"rock_physics_analysis.modules contains unknown modules: {unknown_modules}.")
    if "ai_vp_linear" not in modules:
        raise ValueError("rock_physics_analysis.modules.ai_vp_linear must be configured.")
    ai_cfg = _mapping(modules["ai_vp_linear"], path="rock_physics_analysis.modules.ai_vp_linear")
    _reject_unknown(
        ai_cfg,
        {"enabled", "min_valid_samples_per_well", "min_valid_wells", "huber_delta_sigma"},
        path="rock_physics_analysis.modules.ai_vp_linear",
    )
    if "enabled" not in ai_cfg or not isinstance(ai_cfg["enabled"], bool):
        raise ValueError("rock_physics_analysis.modules.ai_vp_linear.enabled must be explicitly true or false.")
    enabled = ai_cfg["enabled"]
    if enabled:
        required = {"min_valid_samples_per_well", "min_valid_wells", "huber_delta_sigma"}
        missing = sorted(required - set(ai_cfg))
        if missing:
            raise ValueError(f"rock_physics_analysis.modules.ai_vp_linear is missing keys: {missing}.")
        min_samples = _positive_int(
            ai_cfg["min_valid_samples_per_well"],
            path="rock_physics_analysis.modules.ai_vp_linear.min_valid_samples_per_well",
        )
        min_wells = _positive_int(
            ai_cfg["min_valid_wells"],
            path="rock_physics_analysis.modules.ai_vp_linear.min_valid_wells",
        )
        if min_samples < 100:
            raise ValueError("rock_physics_analysis.modules.ai_vp_linear.min_valid_samples_per_well must be >= 100.")
        if min_wells < 3:
            raise ValueError("rock_physics_analysis.modules.ai_vp_linear.min_valid_wells must be >= 3.")
        module_config = AiVpModuleConfig(
            enabled=True,
            min_valid_samples_per_well=min_samples,
            min_valid_wells=min_wells,
            huber_delta_sigma=_positive_float(
                ai_cfg["huber_delta_sigma"],
                path="rock_physics_analysis.modules.ai_vp_linear.huber_delta_sigma",
            ),
        )
    else:
        module_config = AiVpModuleConfig(enabled=False)

    forward_model_raw = root.get("forward_model")
    forward_model: dict[str, str] | None = None
    if forward_model_raw is not None:
        inspected_forward = _mapping(forward_model_raw, path="rock_physics_analysis.forward_model")
        _reject_unknown(
            inspected_forward,
            {"wavelet_file", "source_well"},
            path="rock_physics_analysis.forward_model",
        )
    if enabled:
        forward_cfg = _mapping(forward_model_raw, path="rock_physics_analysis.forward_model")
        forward_model = {
            "wavelet_file": _required_text(forward_cfg, "wavelet_file", path="rock_physics_analysis.forward_model"),
            "source_well": _required_text(forward_cfg, "source_well", path="rock_physics_analysis.forward_model"),
        }
    return {
        "explicit_source": explicit_source,
        "ai_vp_linear": module_config,
        "forward_model": forward_model,
    }


def _contract_business_config(script_config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert typed runtime config to strict, location-independent contract JSON."""
    module_config = script_config.get("ai_vp_linear")
    if not isinstance(module_config, AiVpModuleConfig):
        raise TypeError("script_config.ai_vp_linear must be AiVpModuleConfig.")
    forward_model = script_config.get("forward_model")
    if forward_model is not None and not isinstance(forward_model, Mapping):
        raise TypeError("script_config.forward_model must be a mapping or None.")
    return {
        "modules": {"ai_vp_linear": asdict(module_config)},
        "forward_model": None if forward_model is None else dict(forward_model),
    }


def _status_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"well_name", "preprocess_status", "preprocessed_las"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"well_preprocess_status.csv is missing columns: {missing}.")
    well_ids = table["well_name"].astype(str).str.strip()
    if (well_ids == "").any():
        raise ValueError("well_preprocess_status.csv contains an empty well_name.")
    folded = well_ids.str.casefold()
    duplicates = sorted(well_ids[folded.duplicated(keep=False)].unique())
    if duplicates:
        raise ValueError(f"well_preprocess_status.csv contains duplicate wells: {duplicates}.")
    table = table.copy()
    table["well_name"] = well_ids
    table["preprocess_status"] = table["preprocess_status"].astype(str).str.strip().str.casefold()
    return table


def _curve_inventory(las: lasio.LASFile) -> tuple[str, str]:
    names = [str(curve.mnemonic).strip() for curve in las.curves]
    units = [f"{str(curve.mnemonic).strip()}={str(curve.unit or '').strip()}" for curve in las.curves]
    return ";".join(names), ";".join(units)


def _load_inputs(status: pd.DataFrame, *, output_dir: Path) -> tuple[dict[str, LoadedLas], list[dict[str, Any]]]:
    loaded: dict[str, LoadedLas] = {}
    inventory: list[dict[str, Any]] = []
    seen_paths: dict[Path, str] = {}
    contract_errors: list[str] = []
    for row in status.to_dict(orient="records"):
        well_id = str(row["well_name"])
        preprocess_status = str(row["preprocess_status"])
        selected = preprocess_status == "passed"
        raw_path = str(row.get("preprocessed_las") or "").strip()
        item: dict[str, Any] = {
            "well_id": well_id,
            "preprocess_status": preprocess_status,
            "selected_for_input": selected,
            "las_path": raw_path,
            "read_status": "not_selected",
            "curve_names": "",
            "curve_units": "",
            "reasons": "",
        }
        if not selected:
            inventory.append(item)
            continue
        try:
            if not raw_path:
                raise ValueError("passed row has an empty preprocessed_las path")
            path = resolve_relative_path(raw_path, root=REPO_ROOT)
            if path in seen_paths:
                raise ValueError(f"duplicate LAS path already assigned to {seen_paths[path]}")
            seen_paths[path] = well_id
            if not path.is_file():
                raise FileNotFoundError(f"LAS file does not exist: {path}")
            las = lasio.read(path)
            curve_names, curve_units = _curve_inventory(las)
            item.update(
                {
                    "las_path": repo_relative_path(path, root=REPO_ROOT),
                    "read_status": "success",
                    "curve_names": curve_names,
                    "curve_units": curve_units,
                }
            )
            loaded[well_id] = LoadedLas(well_id=well_id, path=path, las=las)
        except Exception as exc:
            item["read_status"] = "failed"
            item["reasons"] = f"input_contract_error:{exc}"
            contract_errors.append(f"{well_id}: {exc}")
        inventory.append(item)
    pd.DataFrame(inventory).to_csv(output_dir / "well_input_inventory.csv", index=False)
    if contract_errors:
        raise ValueError("Step 3 passed LAS input contract failed: " + "; ".join(contract_errors))
    if not loaded:
        raise ValueError("well_preprocess_status.csv contains no preprocess_status=passed LAS inputs.")
    return loaded, inventory


def _exact_curve(las: lasio.LASFile, mnemonic: str) -> tuple[np.ndarray, str]:
    matches = [curve for curve in las.curves if str(curve.mnemonic).strip().upper() == mnemonic]
    if len(matches) != 1:
        raise ValueError(f"missing_or_duplicate_required_curve:{mnemonic}:matches={len(matches)}")
    curve = matches[0]
    unit = str(curve.unit or "").strip()
    expected = EXPECTED_CURVES[mnemonic]
    if unit != expected:
        raise ValueError(f"curve_unit_mismatch:{mnemonic}:expected={expected!r},actual={unit!r}")
    return np.asarray(curve.data, dtype=np.float64), unit


def _prepare_ai_vp_well(loaded: LoadedLas, *, min_samples: int) -> tuple[WellAiVpSamples, dict[str, Any]]:
    dt, _ = _exact_curve(loaded.las, "DT_USM")
    rho, _ = _exact_curve(loaded.las, "RHO_GCC")
    ai_las, _ = _exact_curve(loaded.las, "AI")
    if dt.shape != rho.shape or dt.shape != ai_las.shape or dt.ndim != 1:
        raise ValueError("curve_shape_mismatch:DT_USM,RHO_GCC,AI")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vp_all = 1_000_000.0 / dt
        ai_recomputed_all = vp_all * rho
    finite = (
        np.isfinite(dt)
        & np.isfinite(rho)
        & np.isfinite(ai_las)
        & np.isfinite(vp_all)
        & np.isfinite(ai_recomputed_all)
    )
    positive = (
        (dt > 0.0)
        & (rho > 0.0)
        & (ai_las > 0.0)
        & (vp_all > 0.0)
        & (ai_recomputed_all > 0.0)
    )
    valid = finite & positive
    n_valid = int(np.sum(valid))
    vp = vp_all[valid]
    ai_recomputed = ai_recomputed_all[valid]
    base_qc = {
        "n_total_samples": int(dt.size),
        "n_nonfinite_samples": int(np.sum(~finite)),
        "n_nonpositive_samples": int(np.sum(finite & ~positive)),
        "n_samples": n_valid,
        "vp_min_mps": float(np.min(vp)) if n_valid else np.nan,
        "vp_max_mps": float(np.max(vp)) if n_valid else np.nan,
        "ai_min_mps_gcc": float(np.min(ai_recomputed)) if n_valid else np.nan,
        "ai_max_mps_gcc": float(np.max(ai_recomputed)) if n_valid else np.nan,
        "ai_consistency_max_abs_mps_gcc": np.nan,
        "ai_consistency_max_rel": np.nan,
    }
    if n_valid < min_samples:
        raise AiVpWellRejected(f"insufficient_valid_samples:{n_valid}<{min_samples}", base_qc)
    absolute = np.abs(ai_recomputed - ai_las[valid])
    relative = absolute / np.abs(ai_las[valid])
    max_abs = float(np.max(absolute))
    max_rel = float(np.max(relative))
    base_qc["ai_consistency_max_abs_mps_gcc"] = max_abs
    base_qc["ai_consistency_max_rel"] = max_rel
    if not np.allclose(
        ai_recomputed,
        ai_las[valid],
        rtol=AI_CONSISTENCY_RTOL,
        atol=AI_CONSISTENCY_ATOL,
    ):
        raise AiVpWellRejected(
            f"ai_consistency_mismatch:max_abs={max_abs:.9g},max_rel={max_rel:.9g}",
            base_qc,
        )
    sample = WellAiVpSamples(loaded.well_id, vp, ai_recomputed)
    return sample, base_qc


def _failed_qc(well_id: str, reason: str) -> dict[str, Any]:
    return {
        "well_id": well_id,
        "module_status": "rejected",
        "reasons": reason,
        "n_total_samples": 0,
        "n_nonfinite_samples": 0,
        "n_nonpositive_samples": 0,
        "n_samples": 0,
        "vp_min_mps": np.nan,
        "vp_max_mps": np.nan,
        "ai_min_mps_gcc": np.nan,
        "ai_max_mps_gcc": np.nan,
        "ai_consistency_max_abs_mps_gcc": np.nan,
        "ai_consistency_max_rel": np.nan,
        "r2": np.nan,
        "rmse_mps_gcc": np.nan,
        "mae_mps_gcc": np.nan,
        "bias_mps_gcc": np.nan,
        "well_total_weight": 0.0,
        "well_effective_weight": 0.0,
    }


def _plot_fit(path: Path, samples: Mapping[str, WellAiVpSamples], a: float, b: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, well_id in enumerate(sorted(samples, key=str.casefold)):
        sample = samples[well_id]
        count = min(2500, sample.vp_mps.size)
        selected = np.linspace(0, sample.vp_mps.size - 1, count, dtype=int)
        color = colors(index % 10)
        axes[0].scatter(sample.vp_mps[selected], sample.ai_mps_gcc[selected], s=4, alpha=0.25, color=color, label=well_id)
        residual = sample.ai_mps_gcc[selected] - (a * sample.vp_mps[selected] + b)
        axes[1].scatter(sample.vp_mps[selected], residual, s=4, alpha=0.25, color=color, label=well_id)
    vp_min = min(float(np.min(sample.vp_mps)) for sample in samples.values())
    vp_max = max(float(np.max(sample.vp_mps)) for sample in samples.values())
    line_vp = np.linspace(vp_min, vp_max, 256)
    axes[0].plot(line_vp, a * line_vp + b, color="black", linewidth=2.0, label="equal-well Huber")
    axes[1].axhline(0.0, color="black", linewidth=1.2)
    axes[0].set(xlabel="Vp (m/s)", ylabel="AI (m/s·g/cm³)", title="Global AI–Vp relation")
    axes[1].set(xlabel="Vp (m/s)", ylabel="AI residual (m/s·g/cm³)", title="Residuals")
    axes[0].legend(markerscale=2.5, fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_wavelet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Configured wavelet file does not exist: {path}")
    table = pd.read_csv(path)
    if list(table.columns) != ["time_s", "amplitude"]:
        raise ValueError("Wavelet CSV columns must be exactly ['time_s', 'amplitude'].")
    time_s = table["time_s"].to_numpy(dtype=np.float64)
    amplitude = table["amplitude"].to_numpy(dtype=np.float64)
    forward_time(np.array([0.0, 0.0], dtype=np.float64), time_s, amplitude)
    return time_s, amplitude


def _relation_payload(
    fit: Any,
    *,
    module_config: AiVpModuleConfig,
    candidate_wells: list[str],
    accepted_wells: list[str],
    rejected: list[dict[str, str]],
    loaded: Mapping[str, LoadedLas],
) -> dict[str, Any]:
    return {
        "schema": RELATION_SCHEMA,
        "formula": "AI = a * Vp + b",
        "coefficients": {"a": fit.relation.a, "b": fit.relation.b},
        "units": {"ai": "m/s*g/cm3", "vp": "m/s", "a": "g/cm3", "b": "m/s*g/cm3"},
        "regression": {
            "method": "equal_well_weight_huber",
            "initial_estimator": "equal_well_weight_least_squares",
            "scale_estimator": "weighted_mad",
            "huber_delta_sigma": fit.huber_delta_sigma,
            "min_valid_samples_per_well": module_config.min_valid_samples_per_well,
            "min_valid_wells": module_config.min_valid_wells,
            "converged": fit.converged,
            "iterations": fit.iterations,
            "robust_scale_mps_gcc": fit.robust_scale_mps_gcc,
            "objective": fit.objective,
            "initial_coefficients": {"a": fit.initial_a, "b": fit.initial_b},
            "well_base_weights": dict(fit.well_base_weights),
            "well_effective_weights": dict(fit.well_effective_weights),
        },
        "ai_consistency": {"rtol": AI_CONSISTENCY_RTOL, "atol_mps_gcc": AI_CONSISTENCY_ATOL},
        "candidate_wells": candidate_wells,
        "accepted_wells": accepted_wells,
        "rejected_wells": rejected,
        "source_preprocessed_las": [
            {
                "well_id": well_id,
                "path": repo_relative_path(loaded[well_id].path, root=REPO_ROOT),
            }
            for well_id in candidate_wells
        ],
        "validation": {"a_positive": True, "inverse_velocity_finite_positive": True},
        "aggregate_qc": dict(fit.aggregate_qc),
    }


def run(config_path: Path, output_dir_arg: Path | None = None) -> Path:
    config_file = resolve_relative_path(config_path, root=REPO_ROOT)
    raw_config = load_yaml_config(config_file)
    workflow = WorkflowConfig.from_mapping(raw_config)
    script_config = _parse_config(raw_config)
    business_config = _contract_business_config(script_config)
    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    output_dir = (
        resolve_relative_path(output_dir_arg, root=REPO_ROOT)
        if output_dir_arg is not None
        else resolve_timestamped_output_dir(output_root, "rock_physics_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    source_dir = resolve_source_run(
        script_config["explicit_source"] or None,
        output_root=output_root,
        prefix="well_preprocess_",
        required_files=["well_preprocess_status.csv", "run_summary.json"],
        root=REPO_ROOT,
        label="Step 3 well_preprocess",
    )
    discovery_mode = "explicit" if script_config["explicit_source"] else "auto_discovered"
    status_path = source_dir / "well_preprocess_status.csv"
    inventory_path = output_dir / "well_input_inventory.csv"
    source_summary_path = source_dir / "run_summary.json"
    with source_summary_path.open("r", encoding="utf-8") as handle:
        source_run_summary = json.load(handle)
    input_contracts = {
        "well_preprocess": {
            "path": repo_relative_path(source_summary_path, root=REPO_ROOT),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                source_run_summary, label=f"well preprocess run {source_dir}"
            ),
        }
    }
    source_summary = {
        "well_preprocess_dir": repo_relative_path(source_dir, root=REPO_ROOT),
        "discovery_mode": discovery_mode,
        "well_preprocess_status_csv": repo_relative_path(status_path, root=REPO_ROOT),
    }
    try:
        status = _status_table(status_path)
        loaded, _ = _load_inputs(status, output_dir=output_dir)
    except Exception as exc:
        failure_summary = {
            "schema": SCHEMA_VERSION,
            "script": "rock_physics_analysis.py",
            "script_version": SCRIPT_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "failed",
            "source_run": source_summary,
            "input": {
                "inventory_path": (
                    repo_relative_path(inventory_path, root=REPO_ROOT)
                    if inventory_path.is_file()
                    else None
                ),
                "input_contract_error_count": 1,
                "reason": str(exc),
            },
            "modules": {},
            "artifacts": {},
        }
        write_json(output_dir / "run_summary.json", failure_summary)
        raise

    module_config: AiVpModuleConfig = script_config["ai_vp_linear"]
    base_summary: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "script": "rock_physics_analysis.py",
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "success",
        "input_contracts": input_contracts,
        "source_run": source_summary,
        "input": {
            "passed_wells": sorted(loaded, key=str.casefold),
            "successfully_read_wells": sorted(loaded, key=str.casefold),
            "inventory_path": repo_relative_path(inventory_path, root=REPO_ROOT),
            "input_contract_error_count": 0,
        },
        "modules": {},
        "artifacts": {},
    }

    if not module_config.enabled:
        base_summary["reason"] = "no_analysis_modules_enabled"
        base_summary["modules"]["ai_vp_linear"] = {"enabled": False, "status": "disabled"}
        base_summary["contract_fingerprint_schema"] = CONTRACT_FINGERPRINT_SCHEMA
        base_summary["contract_fingerprint_sha256"] = contract_fingerprint_sha256(
            contract_schema_version=SCHEMA_VERSION,
            semantics={
                "sample_domain": workflow.seismic.domain,
                "depth_basis": workflow.seismic.depth_basis,
                "modules": base_summary["modules"],
            },
            business_config=business_config,
            input_contracts=input_contracts,
            primary_artifacts={"well_input_inventory": inventory_path},
        )
        write_json(output_dir / "run_summary.json", base_summary)
        return output_dir

    module_dir = output_dir / "modules" / "ai_vp_linear"
    module_dir.mkdir(parents=True, exist_ok=False)
    candidate_wells = sorted(loaded, key=str.casefold)
    samples: dict[str, WellAiVpSamples] = {}
    qc_by_well: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for well_id in candidate_wells:
        try:
            sample, qc = _prepare_ai_vp_well(
                loaded[well_id],
                min_samples=int(module_config.min_valid_samples_per_well),
            )
            samples[well_id] = sample
            qc_by_well[well_id] = {
                "well_id": well_id,
                "module_status": "accepted",
                "reasons": "",
                **qc,
            }
        except AiVpWellRejected as exc:
            reason = str(exc)
            qc_by_well[well_id] = _failed_qc(well_id, reason)
            qc_by_well[well_id].update(exc.qc)
            rejected.append({"well_id": well_id, "reason": reason})
        except Exception as exc:
            reason = str(exc)
            qc_by_well[well_id] = _failed_qc(well_id, reason)
            rejected.append({"well_id": well_id, "reason": reason})

    accepted_wells = sorted(samples, key=str.casefold)
    if len(accepted_wells) < int(module_config.min_valid_wells):
        qc_path = module_dir / "well_fit_qc.csv"
        pd.DataFrame([qc_by_well[well_id] for well_id in candidate_wells]).to_csv(qc_path, index=False)
        base_summary["status"] = "failed"
        base_summary["modules"]["ai_vp_linear"] = {
            "enabled": True,
            "status": "failed",
            "candidate_wells": candidate_wells,
            "accepted_wells": accepted_wells,
            "rejected_wells": rejected,
            "reason": f"accepted wells {len(accepted_wells)} < min_valid_wells {module_config.min_valid_wells}",
        }
        write_json(output_dir / "run_summary.json", base_summary)
        raise RuntimeError(base_summary["modules"]["ai_vp_linear"]["reason"])

    try:
        fit = fit_equal_well_huber(samples, huber_delta_sigma=float(module_config.huber_delta_sigma))
    except Exception as exc:
        qc_path = module_dir / "well_fit_qc.csv"
        pd.DataFrame([qc_by_well[well_id] for well_id in candidate_wells]).to_csv(qc_path, index=False)
        base_summary["status"] = "failed"
        base_summary["modules"]["ai_vp_linear"] = {
            "enabled": True,
            "status": "failed",
            "candidate_wells": candidate_wells,
            "accepted_wells": accepted_wells,
            "rejected_wells": rejected,
            "reason": f"global_fit_failed:{exc}",
        }
        base_summary["artifacts"]["well_fit_qc"] = {
            "path": repo_relative_path(qc_path, root=REPO_ROOT),
        }
        write_json(output_dir / "run_summary.json", base_summary)
        raise
    for well_id in accepted_wells:
        qc_by_well[well_id].update(well_fit_metrics(samples[well_id], fit.relation))
        qc_by_well[well_id]["well_total_weight"] = fit.well_base_weights[well_id]
        qc_by_well[well_id]["well_effective_weight"] = fit.well_effective_weights[well_id]
    qc_path = module_dir / "well_fit_qc.csv"
    pd.DataFrame([qc_by_well[well_id] for well_id in candidate_wells]).to_csv(qc_path, index=False)

    relation_path = module_dir / "rock_physics_relation.json"
    relation_payload = _relation_payload(
        fit,
        module_config=module_config,
        candidate_wells=candidate_wells,
        accepted_wells=accepted_wells,
        rejected=rejected,
        loaded=loaded,
    )
    figure_path = module_dir / "figures" / "ai_vp_fit.png"

    rejection_counts = Counter(item["reason"].split(":", 1)[0] for item in rejected)
    base_summary["modules"]["ai_vp_linear"] = {
        "enabled": True,
        "status": "success",
        "candidate_wells": candidate_wells,
        "accepted_wells": accepted_wells,
        "rejected_wells": rejected,
        "rejection_counts": dict(rejection_counts),
        "sample_rejection_counts": {
            "nonfinite": int(sum(qc_by_well[well]["n_nonfinite_samples"] for well in candidate_wells)),
            "nonpositive": int(sum(qc_by_well[well]["n_nonpositive_samples"] for well in candidate_wells)),
        },
    }
    base_summary["artifacts"] = {
        "well_fit_qc": {"path": repo_relative_path(qc_path, root=REPO_ROOT)},
    }

    forward_model = script_config["forward_model"]
    wavelet_path = resolve_relative_path(forward_model["wavelet_file"], root=REPO_ROOT)
    try:
        wavelet_time, _ = _load_wavelet(wavelet_path)
    except Exception as exc:
        base_summary["status"] = "failed"
        base_summary["forward_model_inputs"] = {
            "status": "failed",
            "reason": f"wavelet_validation_failed:{exc}",
        }
        write_json(output_dir / "run_summary.json", base_summary)
        raise
    write_json(relation_path, relation_payload)
    _plot_fit(figure_path, samples, fit.relation.a, fit.relation.b)
    base_summary["artifacts"].update(
        {
            "rock_physics_relation": {
                "path": repo_relative_path(relation_path, root=REPO_ROOT),
            },
            "ai_vp_fit_figure": {
                "path": repo_relative_path(figure_path, root=REPO_ROOT),
            },
        }
    )
    source_files = [
        {
            "well_id": well_id,
            "path": repo_relative_path(loaded[well_id].path, root=REPO_ROOT),
        }
        for well_id in candidate_wells
    ]
    forward_inputs = {
        "schema": FORWARD_INPUTS_SCHEMA,
        "sample_domain": workflow.seismic.domain,
        "depth_basis": workflow.seismic.depth_basis,
        "wavelet": {
            "source_well": forward_model["source_well"],
            "path": repo_relative_path(wavelet_path, root=REPO_ROOT),
            "time_unit": "s",
            "sample_count": int(wavelet_time.size),
            "dt_s": float(wavelet_time[1] - wavelet_time[0]),
        },
        "ai_velocity_relation": {
            "path": repo_relative_path(relation_path, root=REPO_ROOT),
            "formula": "AI = a * Vp + b",
            "a": fit.relation.a,
            "b": fit.relation.b,
            "ai_unit": "m/s*g/cm3",
            "vp_unit": "m/s",
            "a_unit": "g/cm3",
            "b_unit": "m/s*g/cm3",
            "accepted_wells": accepted_wells,
        },
        "source_runs": {"well_preprocess_dir": repo_relative_path(source_dir, root=REPO_ROOT)},
        "source_preprocessed_las": source_files,
    }
    forward_inputs_path = output_dir / "forward_model_inputs.json"
    write_json(forward_inputs_path, forward_inputs)

    base_summary["artifacts"].update(
        {
            "wavelet": {"path": repo_relative_path(wavelet_path, root=REPO_ROOT)},
            "forward_model_inputs": {
                "path": repo_relative_path(forward_inputs_path, root=REPO_ROOT),
            },
        }
    )
    base_summary["contract_fingerprint_schema"] = CONTRACT_FINGERPRINT_SCHEMA
    base_summary["contract_fingerprint_sha256"] = contract_fingerprint_sha256(
        contract_schema_version=SCHEMA_VERSION,
        semantics={
            "sample_domain": workflow.seismic.domain,
            "depth_basis": workflow.seismic.depth_basis,
            "relation_schema": RELATION_SCHEMA,
            "forward_inputs_schema": FORWARD_INPUTS_SCHEMA,
        },
        business_config=business_config,
        input_contracts=input_contracts,
        primary_artifacts={
            "well_input_inventory": inventory_path,
            "rock_physics_relation": relation_path,
            "forward_model_inputs": forward_inputs_path,
            "external_wavelet": wavelet_path,
        },
    )
    write_json(output_dir / "run_summary.json", base_summary)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = run(args.config, args.output_dir)
    print(f"Wrote rock-physics analysis to {repo_relative_path(output_dir, root=REPO_ROOT)}")


if __name__ == "__main__":
    main()
