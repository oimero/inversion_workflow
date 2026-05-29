"""Generate one global time-domain wavelet and batch synthetic QC.

The script consumes fourth-step ``well_auto_tie`` outputs.  It evaluates all
usable per-well wavelets on the same fixed optimized TDT/seismic artifacts,
optimizes a PCA consensus wavelet, and exports one selected global wavelet.

Usage::

    python scripts/global_wavelet_generation.py
    python scripts/global_wavelet_generation.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

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

from cup.utils.config import merge_dict_defaults
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.utils.statistics import aggregate_cluster_then_global
from cup.well.assets import normalize_well_name
from cup.well.tie import (
    TieEvaluationWell,
    WaveletCandidate,
    WaveletWellMetric,
    build_well_spatial_clusters,
    evaluate_wavelet_on_well,
    load_tie_artifacts,
    prepare_well_for_evaluation,
)
from cup.well.wavelet import (
    infer_wavelet_dt,
    load_wavelet_csv,
    validate_wavelet_normalization,
    wavelet_roughness,
    wavelet_spectrum_features,
)
from cup.well.wavelet_consensus import ConsensusSearchPolicy, build_wavelet_pca_basis, optimize_consensus_wavelet
from wtie.modeling.modeling import ConvModeler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--well", type=str, default=None, help="Optional single evaluation well filter.")
    return parser.parse_args()


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("global_wavelet_generation") or {})
    merge_dict_defaults(
        script_cfg,
        "source_runs",
        {"mode": "latest", "well_auto_tie_dir": None},
    )
    merge_dict_defaults(
        script_cfg,
        "candidate_filter",
        {
            "min_source_tie_corr": 0.35,
            "max_source_tie_nmae": None,
            "exclude_source_wells": [],
            "include_source_wells": None,
        },
    )
    merge_dict_defaults(
        script_cfg,
        "evaluation_wells",
        {"status": "success", "exclude_wells": [], "include_wells": None},
    )
    merge_dict_defaults(
        script_cfg,
        "wavelet_qc",
        {
            "expected_l2_energy": 1.0,
            "l2_energy_tolerance": 1e-5,
            "max_center_abs_time_s": 1e-9,
            "allow_small_renormalization": True,
        },
    )
    merge_dict_defaults(
        script_cfg,
        "generation",
        {
            "mode": "optimize_consensus",
            "pca": {
                "n_components": 4,
                "coefficient_bounds": "quantile",
                "coefficient_quantiles": [0.05, 0.95],
                "include_mean_wavelet": True,
            },
            "optimizer": {
                "strategy": "random_then_powell",
                "random_trials": 512,
                "max_refine_iters": 120,
                "seed": 20260529,
            },
            "objective": {
                "name": "spatial_debiased_score",
                "corr_weight": 1.0,
                "p10_corr_weight": 0.5,
                "nmae_weight": 0.5,
                "deviation_from_mean_weight": 0.15,
                "roughness_weight": 0.05,
                "bandwidth_drift_weight": 0.05,
                "max_allowed_side_lobe_ratio": None,
            },
        },
    )
    merge_dict_defaults(
        script_cfg,
        "spatial_debias",
        {"enabled": True, "cluster_radius_m": 600.0},
    )
    merge_dict_defaults(
        script_cfg,
        "scoring",
        {"min_eval_well_count": 3, "on_insufficient_eval_wells": "select_best_source_tie"},
    )
    merge_dict_defaults(
        script_cfg,
        "export",
        {"selected_wavelet_name": "global_wavelet_201ms.csv", "write_unified_synthetics": True},
    )
    return script_cfg


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"global_wavelet_generation_{timestamp}"


def _discover_latest_dir(cfg: dict[str, Any], prefix: str) -> Path:
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    candidates = sorted([path for path in output_root.glob(f"{prefix}_*") if path.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_* output directory found under {output_root}.")
    return candidates[-1]


def _resolve_source_dirs(cfg: dict[str, Any], script_cfg: dict[str, Any]) -> dict[str, Path]:
    source_runs = dict(script_cfg.get("source_runs") or {})
    auto_tie_dir = (
        _resolve_repo_path(source_runs["well_auto_tie_dir"])
        if source_runs.get("well_auto_tie_dir") is not None
        else _discover_latest_dir(cfg, "well_auto_tie")
    )
    return {"auto_tie_dir": auto_tie_dir}


def _ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "synthetic_qc": output_dir / "synthetic_qc",
        "figures": output_dir / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _name_for_candidate(candidate: WaveletCandidate) -> str:
    return f"{sanitize_filename(candidate.source_well)}__{candidate.wavelet_file.stem}"


def _filter_by_well_names(wells: Sequence[TieEvaluationWell], cfg: dict[str, Any]) -> list[TieEvaluationWell]:
    include = cfg.get("include_wells")
    exclude = cfg.get("exclude_wells") or []
    include_keys = {normalize_well_name(name) for name in include} if include is not None else None
    exclude_keys = {normalize_well_name(name) for name in exclude}
    out = []
    for well in wells:
        key = normalize_well_name(well.well_name)
        if include_keys is not None and key not in include_keys:
            continue
        if key in exclude_keys:
            continue
        out.append(well)
    return out


def _load_candidate_wavelet_pool(
    candidates: Sequence[WaveletCandidate],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    qc_cfg = dict(cfg["wavelet_qc"])
    qc_rows: list[dict[str, Any]] = []
    pool: list[dict[str, Any]] = []
    reference_time: np.ndarray | None = None
    reference_dt: float | None = None

    for candidate in candidates:
        name = _name_for_candidate(candidate)
        reasons: list[str] = []
        try:
            time_s, amplitude = load_wavelet_csv(candidate.wavelet_file)
            dt_s = infer_wavelet_dt(time_s)
            normalized, qc = validate_wavelet_normalization(
                time_s,
                amplitude,
                expected_l2_energy=float(qc_cfg["expected_l2_energy"]),
                l2_energy_tolerance=float(qc_cfg["l2_energy_tolerance"]),
                max_center_abs_time_s=float(qc_cfg["max_center_abs_time_s"]),
                allow_small_renormalization=bool(qc_cfg["allow_small_renormalization"]),
            )
            if qc.status != "ok":
                reasons.append(qc.reasons)
            if time_s.size % 2 == 0:
                reasons.append("even_sample_count_expected_odd_for_centered_wavelet")
            if reference_time is None and not reasons:
                reference_time = time_s
                reference_dt = dt_s
            if reference_time is not None:
                if time_s.shape != reference_time.shape or not np.allclose(time_s, reference_time, rtol=0.0, atol=1e-9):
                    reasons.append("wavelet_axis_mismatch")
            if reference_dt is not None and not np.isclose(dt_s, reference_dt, rtol=1e-5, atol=1e-9):
                reasons.append("wavelet_dt_mismatch")
            features = wavelet_spectrum_features(time_s, normalized).to_row() if not reasons else {}
            row = {
                "candidate_wavelet": name,
                "source_well": candidate.source_well,
                "route": candidate.route,
                "wavelet_file": candidate.wavelet_file.as_posix(),
                "dt_s": dt_s,
                "n_samples": int(time_s.size),
                **qc.to_row(),
                **features,
                "status": "failed" if reasons else "ok",
                "reasons": ";".join(item for item in reasons if item),
            }
            qc_rows.append(row)
            if not reasons:
                pool.append(
                    {
                        "name": name,
                        "candidate": candidate,
                        "time_s": time_s,
                        "amplitude": normalized,
                        "roughness": wavelet_roughness(normalized),
                        "features": features,
                    }
                )
        except Exception as exc:
            qc_rows.append(
                {
                    "candidate_wavelet": name,
                    "source_well": candidate.source_well,
                    "route": candidate.route,
                    "wavelet_file": candidate.wavelet_file.as_posix(),
                    "status": "failed",
                    "reasons": f"{type(exc).__name__}:{exc}",
                }
            )

    if not pool or reference_time is None:
        raise ValueError("No candidate wavelets passed QC.")
    return pool, qc_rows, reference_time, np.vstack([item["amplitude"] for item in pool])


def _metric_rows(metrics: Sequence[WaveletWellMetric]) -> list[dict[str, Any]]:
    return [metric.to_row() for metric in metrics]


def _join_clusters(metrics_df: pd.DataFrame, clusters_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or clusters_df.empty:
        return metrics_df
    metrics_df = metrics_df.drop(
        columns=[col for col in ["spatial_cluster_id", "spatial_cluster_size"] if col in metrics_df.columns]
    )
    return metrics_df.merge(
        clusters_df[["well_name", "spatial_cluster_id", "spatial_cluster_size"]],
        left_on="eval_well",
        right_on="well_name",
        how="left",
    ).drop(columns=["well_name"])


def _aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ok = metrics_df.loc[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    return aggregate_cluster_then_global(
        ok,
        value_columns=["corr", "nmae"],
        cluster_column="spatial_cluster_id",
        group_columns=["candidate_wavelet", "source_well"],
        quantiles=[0.1],
    )


def _score_from_aggregate(row: pd.Series, cfg: dict[str, Any], *, regularization: dict[str, float] | None = None) -> float:
    objective = dict(cfg["generation"]["objective"])
    regularization = dict(regularization or {})
    corr = float(row.get("spatial_debiased_median_corr", np.nan))
    p10 = float(row.get("spatial_debiased_p10_corr", np.nan))
    nmae = float(row.get("spatial_debiased_median_nmae", np.nan))
    if not (np.isfinite(corr) and np.isfinite(p10) and np.isfinite(nmae)):
        return float("-inf")
    score = (
        float(objective["corr_weight"]) * corr
        + float(objective["p10_corr_weight"]) * p10
        - float(objective["nmae_weight"]) * nmae
    )
    score -= float(objective["deviation_from_mean_weight"]) * float(regularization.get("deviation_from_mean", 0.0))
    score -= float(objective["roughness_weight"]) * float(regularization.get("roughness", 0.0))
    score -= float(objective["bandwidth_drift_weight"]) * float(regularization.get("bandwidth_drift", 0.0))
    side_lobe_limit = objective.get("max_allowed_side_lobe_ratio")
    if side_lobe_limit is not None and regularization.get("side_lobe_ratio", 0.0) > float(side_lobe_limit):
        return float("-inf")
    return float(score)


def _evaluate_wavelet_on_all_wells(
    *,
    wavelet_name: str,
    source_well: str,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    wells: Sequence[TieEvaluationWell],
    clusters_df: pd.DataFrame,
    modeler: Any,
    output_qc_dir: Path | None = None,
    well_cache: dict[str, tuple[Any, Any]] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    metrics: list[WaveletWellMetric] = []
    qcs: list[pd.DataFrame] = []
    cache = dict(well_cache or {})
    for well in wells:
        try:
            precomputed = cache.get(normalize_well_name(well.well_name))
            metric, qc = evaluate_wavelet_on_well(
                wavelet_time_s=wavelet_time_s,
                wavelet_amplitude=wavelet_amplitude,
                well_artifact=well,
                candidate_wavelet=wavelet_name,
                source_well=source_well,
                modeler=modeler,
                seismic_match=precomputed[0] if precomputed is not None else None,
                reflectivity_match=precomputed[1] if precomputed is not None else None,
            )
            metrics.append(metric)
            qc = qc.copy()
            qc.insert(0, "well_name", well.well_name)
            qc.insert(1, "candidate_wavelet", wavelet_name)
            qcs.append(qc)
            if output_qc_dir is not None:
                safe = sanitize_filename(f"{wavelet_name}_{well.well_name}")
                qc.to_csv(output_qc_dir / f"synthetic_qc_{safe}.csv", index=False)
        except Exception as exc:
            metrics.append(
                WaveletWellMetric(
                    candidate_wavelet=wavelet_name,
                    source_well=source_well,
                    eval_well=well.well_name,
                    route=well.route,
                    corr=None,
                    nmae=None,
                    best_shift_ms=0.0,
                    status="failed",
                    reasons=f"{type(exc).__name__}:{exc}",
                    is_source_well=normalize_well_name(source_well) == normalize_well_name(well.well_name),
                )
            )
    metrics_df = _join_clusters(pd.DataFrame.from_records(_metric_rows(metrics)), clusters_df)
    return metrics_df, qcs


def _basis_dataframe(time_s: np.ndarray, basis: Any) -> pd.DataFrame:
    data: dict[str, Any] = {"time_s": time_s, "mean_wavelet": basis.mean_wavelet}
    for idx in range(basis.n_components):
        data[f"pc_{idx}"] = basis.components[idx]
    return pd.DataFrame(data)


def _regularization_for_wavelet(
    wavelet: np.ndarray,
    *,
    mean_wavelet: np.ndarray,
    mean_features: dict[str, float],
    time_s: np.ndarray,
) -> dict[str, float]:
    features = wavelet_spectrum_features(time_s, wavelet).to_row()
    centroid = float(features["spectral_centroid_hz"])
    mean_centroid = float(mean_features["spectral_centroid_hz"])
    denom = max(abs(mean_centroid), 1e-12)
    return {
        "deviation_from_mean": float(np.linalg.norm(wavelet - mean_wavelet)),
        "roughness": wavelet_roughness(wavelet),
        "bandwidth_drift": abs(centroid - mean_centroid) / denom,
        "side_lobe_ratio": float(features["side_lobe_ratio"]),
    }


def _add_candidate_regularization_and_score(
    candidate_aggregate: pd.DataFrame,
    *,
    pool: Sequence[dict[str, Any]],
    basis: Any,
    mean_features: dict[str, float],
    time_s: np.ndarray,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    regularization_by_name = {
        str(item["name"]): _regularization_for_wavelet(
            item["amplitude"],
            mean_wavelet=basis.mean_wavelet,
            mean_features=mean_features,
            time_s=time_s,
        )
        for item in pool
    }
    out = candidate_aggregate.copy()
    for key in ["deviation_from_mean", "roughness", "bandwidth_drift", "side_lobe_ratio"]:
        out[key] = out["candidate_wavelet"].map(lambda name: regularization_by_name.get(str(name), {}).get(key, np.nan))
    out["score"] = out.apply(
        lambda row: _score_from_aggregate(
            row,
            cfg,
            regularization=regularization_by_name.get(str(row["candidate_wavelet"]), {}),
        ),
        axis=1,
    )
    return out


def _plot_wavelets(time_s: np.ndarray, wavelets: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, values in wavelets.items():
        ax.plot(time_s * 1000.0, values, lw=1.0, label=name)
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Selected global wavelet")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _script_config(cfg)
    if args.well is not None:
        script_cfg["evaluation_wells"]["include_wells"] = [args.well]
        script_cfg["scoring"]["min_eval_well_count"] = 1
    source_dirs = _resolve_source_dirs(cfg, script_cfg)
    output_dir = _resolve_output_dir(args, cfg)
    output_dirs = _ensure_output_dirs(output_dir)

    index = load_tie_artifacts(source_dirs["auto_tie_dir"], repo_root=REPO_ROOT)
    candidate_filter = dict(script_cfg["candidate_filter"])
    candidates = index.candidate_wavelets(
        min_source_tie_corr=candidate_filter.get("min_source_tie_corr"),
        max_source_tie_nmae=candidate_filter.get("max_source_tie_nmae"),
        include_source_wells=set(candidate_filter["include_source_wells"]) if candidate_filter.get("include_source_wells") is not None else None,
        exclude_source_wells=set(candidate_filter.get("exclude_source_wells") or []),
    )
    pool, qc_rows, wavelet_time_s, wavelet_matrix = _load_candidate_wavelet_pool(candidates, script_cfg)

    wells = index.evaluation_wells(status=str(script_cfg["evaluation_wells"]["status"]))
    wells = _filter_by_well_names(wells, dict(script_cfg["evaluation_wells"]))
    min_eval_well_count = int(script_cfg["scoring"]["min_eval_well_count"])
    insufficient_eval_wells = len(wells) < min_eval_well_count
    insufficient_policy = str(script_cfg["scoring"].get("on_insufficient_eval_wells", "error"))
    if not wells:
        raise ValueError("No evaluation wells are available for global wavelet generation.")
    if insufficient_eval_wells and insufficient_policy != "select_best_source_tie":
        raise ValueError(f"Too few evaluation wells: {len(wells)} < {min_eval_well_count}")

    clusters_df = build_well_spatial_clusters(wells, radius_m=float(script_cfg["spatial_debias"]["cluster_radius_m"]))
    if not bool(script_cfg["spatial_debias"].get("enabled", True)):
        clusters_df["spatial_cluster_id"] = np.arange(len(clusters_df), dtype=np.int64)
        clusters_df["spatial_cluster_size"] = 1

    pd.DataFrame.from_records([item["candidate"].to_row() | {"candidate_wavelet": item["name"]} for item in pool]).to_csv(
        output_dir / "candidate_wavelets.csv",
        index=False,
    )
    pd.DataFrame.from_records(qc_rows).to_csv(output_dir / "wavelet_qc.csv", index=False)
    clusters_df.to_csv(output_dir / "evaluation_well_spatial_clusters.csv", index=False)

    modeler = ConvModeler()
    well_cache: dict[str, tuple[Any, Any]] = {
        normalize_well_name(well.well_name): prepare_well_for_evaluation(well)
        for well in wells
    }
    candidate_metric_frames = []
    for item in pool:
        metrics_df, _ = _evaluate_wavelet_on_all_wells(
            wavelet_name=item["name"],
            source_well=item["candidate"].source_well,
            wavelet_time_s=wavelet_time_s,
            wavelet_amplitude=item["amplitude"],
            wells=wells,
            clusters_df=clusters_df,
            modeler=modeler,
            well_cache=well_cache,
        )
        candidate_metric_frames.append(metrics_df)
    candidate_metrics_df = pd.concat(candidate_metric_frames, ignore_index=True) if candidate_metric_frames else pd.DataFrame()
    candidate_metrics_df.to_csv(output_dir / "wavelet_candidate_metrics.csv", index=False)
    candidate_aggregate = _aggregate_metrics(candidate_metrics_df)
    if candidate_aggregate.empty:
        raise ValueError("No finite candidate wavelet metrics were produced.")

    pca_cfg = dict(script_cfg["generation"]["pca"])
    basis = build_wavelet_pca_basis(
        wavelet_matrix,
        n_components=int(pca_cfg["n_components"]),
        coefficient_bounds=str(pca_cfg["coefficient_bounds"]),
        coefficient_quantiles=tuple(float(v) for v in pca_cfg["coefficient_quantiles"]),
    )
    _basis_dataframe(wavelet_time_s, basis).to_csv(output_dir / "wavelet_basis.csv", index=False)
    mean_features = wavelet_spectrum_features(wavelet_time_s, basis.mean_wavelet).to_row()
    candidate_aggregate = _add_candidate_regularization_and_score(
        candidate_aggregate,
        pool=pool,
        basis=basis,
        mean_features=mean_features,
        time_s=wavelet_time_s,
        cfg=script_cfg,
    )
    candidate_aggregate.to_csv(output_dir / "wavelet_candidate_aggregate.csv", index=False)

    best_candidate_row = candidate_aggregate.sort_values("score", ascending=False).iloc[0]
    best_candidate_name = str(best_candidate_row["candidate_wavelet"])
    best_candidate_item = next(item for item in pool if item["name"] == best_candidate_name)
    best_candidate_score = float(best_candidate_row["score"])
    if insufficient_eval_wells:
        best_candidate_item = max(
            pool,
            key=lambda item: (
                -np.inf
                if item["candidate"].tie_corr is None or not np.isfinite(float(item["candidate"].tie_corr))
                else float(item["candidate"].tie_corr),
                -np.inf
                if item["candidate"].tie_nmae is None or not np.isfinite(float(item["candidate"].tie_nmae))
                else -float(item["candidate"].tie_nmae),
            ),
        )
        best_candidate_name = str(best_candidate_item["name"])
        if best_candidate_item["candidate"].tie_corr is not None:
            best_candidate_score = float(best_candidate_item["candidate"].tie_corr)

    consensus_wavelet = basis.mean_wavelet
    consensus_score = float("nan")
    if insufficient_eval_wells:
        pd.DataFrame(columns=["trial_id", "score", "status", "reason", "selected"]).to_csv(
            output_dir / "consensus_search_trials.csv",
            index=False,
        )
        pd.DataFrame().to_csv(output_dir / "consensus_wavelet_metrics.csv", index=False)
        selected_name = best_candidate_name
        selected_source = best_candidate_item["candidate"].source_well
        selected_wavelet = best_candidate_item["amplitude"]
        selected_score = best_candidate_score
        selection_mode = "insufficient_eval_fallback"
    else:
        def consensus_evaluator(values: np.ndarray) -> dict[str, float]:
            reg = _regularization_for_wavelet(
                values,
                mean_wavelet=basis.mean_wavelet,
                mean_features=mean_features,
                time_s=wavelet_time_s,
            )
            metrics_df, _ = _evaluate_wavelet_on_all_wells(
                wavelet_name="optimized_consensus_trial",
                source_well="optimized_consensus",
                wavelet_time_s=wavelet_time_s,
                wavelet_amplitude=values,
                wells=wells,
                clusters_df=clusters_df,
                modeler=modeler,
                well_cache=well_cache,
            )
            aggregate = _aggregate_metrics(metrics_df)
            if aggregate.empty:
                return {"score": float("-inf")}
            row = aggregate.iloc[0]
            score = _score_from_aggregate(row, script_cfg, regularization=reg)
            out = {key: float(value) for key, value in row.items() if isinstance(value, (int, float, np.integer, np.floating))}
            out.update(reg)
            out["score"] = score
            return out

        optimizer_cfg = dict(script_cfg["generation"]["optimizer"])
        consensus_result = optimize_consensus_wavelet(
            basis,
            consensus_evaluator,
            policy=ConsensusSearchPolicy(
                random_trials=int(optimizer_cfg["random_trials"]),
                max_refine_iters=int(optimizer_cfg["max_refine_iters"]),
                seed=optimizer_cfg.get("seed"),
                score_key="score",
            ),
        )
        pd.DataFrame.from_records([trial.to_row() for trial in consensus_result.trials]).to_csv(
            output_dir / "consensus_search_trials.csv",
            index=False,
        )

        consensus_metrics_df, _ = _evaluate_wavelet_on_all_wells(
            wavelet_name="optimized_consensus",
            source_well="optimized_consensus",
            wavelet_time_s=wavelet_time_s,
            wavelet_amplitude=consensus_result.wavelet,
            wells=wells,
            clusters_df=clusters_df,
            modeler=modeler,
            well_cache=well_cache,
        )
        consensus_metrics_df.to_csv(output_dir / "consensus_wavelet_metrics.csv", index=False)
        consensus_score = float(consensus_result.score)
        consensus_wavelet = consensus_result.wavelet

        if np.isfinite(consensus_score) and consensus_score > best_candidate_score:
            selected_name = "optimized_consensus"
            selected_source = "optimized_consensus"
            selected_wavelet = consensus_result.wavelet
            selected_score = consensus_score
            selection_mode = "optimized_consensus"
        else:
            selected_name = best_candidate_name
            selected_source = best_candidate_item["candidate"].source_well
            selected_wavelet = best_candidate_item["amplitude"]
            selected_score = best_candidate_score
            selection_mode = "existing_candidate_wins"

    selected_wavelet_path = output_dir / str(script_cfg["export"]["selected_wavelet_name"])
    pd.DataFrame({"time_s": wavelet_time_s, "amplitude": selected_wavelet}).to_csv(selected_wavelet_path, index=False)
    batch_metrics_df, _ = _evaluate_wavelet_on_all_wells(
        wavelet_name=selected_name,
        source_well=selected_source,
        wavelet_time_s=wavelet_time_s,
        wavelet_amplitude=selected_wavelet,
        wells=wells,
        clusters_df=clusters_df,
        modeler=modeler,
        output_qc_dir=output_dirs["synthetic_qc"] if bool(script_cfg["export"]["write_unified_synthetics"]) else None,
        well_cache=well_cache,
    )
    batch_metrics_df.to_csv(output_dir / "batch_synthetic_metrics.csv", index=False)

    plot_wavelets = {"selected": selected_wavelet, "best_candidate": best_candidate_item["amplitude"]}
    plot_wavelets["mean_of_candidates" if selection_mode == "insufficient_eval_fallback" else "consensus"] = consensus_wavelet
    _plot_wavelets(wavelet_time_s, plot_wavelets, output_dirs["figures"] / "selected_wavelet.png")
    summary = {
        "selection_mode": selection_mode,
        "selected_wavelet": selected_name,
        "selected_source_well": selected_source,
        "selected_score": selected_score,
        "best_candidate_wavelet": best_candidate_name,
        "best_candidate_score": best_candidate_score,
        "consensus_score": consensus_score,
        "candidate_count": len(pool),
        "evaluation_well_count": len(wells),
        "selected_wavelet_file": repo_relative_path(selected_wavelet_path, root=REPO_ROOT),
        "source_auto_tie_dir": repo_relative_path(source_dirs["auto_tie_dir"], root=REPO_ROOT),
    }
    write_json(output_dir / "selected_wavelet_summary.json", summary)
    write_json(output_dir / "run_summary.json", {"config": script_cfg, **summary})

    print("=== Global Wavelet Generation ===")
    print(f"Output: {output_dir}")
    print(f"Selected: {selected_name} ({selection_mode}), score={selected_score:.4f}")


if __name__ == "__main__":
    main()
