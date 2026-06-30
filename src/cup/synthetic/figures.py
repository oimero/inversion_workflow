"""QC and report helper figures for synthoseis-lite artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STATE_COLORS = {
    "low_impedance": "#2b6cb0",
    "background": "#a0aec0",
    "high_impedance": "#c53030",
}
STATE_LABELS = {
    0: "low_impedance",
    1: "background",
    2: "high_impedance",
}


def _sanitize(value: Any) -> str:
    text = str(value)
    for char in "\\/:*?\"<>| ":
        text = text.replace(char, "_")
    return text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _finish(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _axis_label(axis_name: str) -> str:
    if axis_name.startswith("tvdss") or "depth" in axis_name:
        return "TVDSS (m)"
    return "TWT (s)"


def _sample_label(row: pd.Series) -> str:
    for name in ("realization_id", "parent_realization_id", "sample_id"):
        if name in row and pd.notna(row[name]) and str(row[name]):
            return str(row[name])
    return "sample"


def _write_figure_manifest(output_dir: Path, generated: list[str], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = figures_dir / "skipped_figures.csv"
    pd.DataFrame.from_records(skipped, columns=["figure", "reason"]).to_csv(skipped_path, index=False)
    manifest = {
        "generated": [str(Path(path).relative_to(output_dir)) for path in generated],
        "skipped": skipped,
    }
    manifest_path = figures_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return {
        "figure_manifest": str(manifest_path),
        "skipped_figures": str(skipped_path),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
    }


def _pick_well_zone(samples: pd.DataFrame, config: Mapping[str, Any], zone_id: str) -> tuple[str, str] | None:
    examples = dict(config.get("report_examples") or {})
    well_name = str(examples.get("well_name") or "").strip()
    requested_zone = str(examples.get("zone_id") or "").strip()
    if well_name and requested_zone:
        subset = samples[samples["well_name"].eq(well_name) & samples["zone_id"].eq(requested_zone)]
        if not subset.empty:
            return well_name, requested_zone
    zone = samples[samples["zone_id"].eq(zone_id)]
    if zone.empty:
        return None
    counts = zone.groupby(["well_name", "zone_id"], dropna=False).size().sort_values(ascending=False)
    if counts.empty:
        return None
    return tuple(str(value) for value in counts.index[0])


def _plot_background_and_residual(
    samples: pd.DataFrame,
    backgrounds: pd.DataFrame,
    *,
    well_name: str,
    zone_id: str,
    out_dir: Path,
) -> list[str]:
    axis_col = _first_existing_column(samples, ("twt_s", "tvdss_m"))
    observed_col = _first_existing_column(samples, ("filtered_log_ai", "observed_log_ai", "full_log_ai"))
    if axis_col is None or observed_col is None:
        return []
    subset = samples[samples["well_name"].eq(well_name) & samples["zone_id"].eq(zone_id)].sort_values(axis_col)
    if subset.empty:
        return []
    generated: list[str] = []
    safe = f"{_sanitize(well_name)}_{_sanitize(zone_id)}"
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(subset[axis_col], subset[observed_col], "--", color="#4a5568", label=observed_col)
    ax.plot(subset[axis_col], subset["background_log_ai"], color="#1a202c", label="background")
    ax.set_title(f"Background fit: {well_name} / {zone_id}")
    ax.set_xlabel(_axis_label(axis_col))
    ax.set_ylabel("log(AI)")
    fit = backgrounds[
        backgrounds["well_name"].eq(well_name) & backgrounds["zone_id"].eq(zone_id)
    ]
    if not fit.empty:
        ax.text(
            0.02,
            0.04,
            f"a={float(fit['background_a'].iloc[0]):.4f}\n"
            f"b={float(fit['background_b'].iloc[0]):.4f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    generated.append(_finish(fig, out_dir / "examples" / f"background_fit_{safe}.png"))

    center = float(subset["state_center"].iloc[0])
    sigma = float(subset["state_sigma"].iloc[0])
    residual = subset["residual"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = min(40, max(12, int(np.sqrt(residual.size))))
    for left, right, color, label in [
        (-np.inf, center - sigma, STATE_COLORS["low_impedance"], "low"),
        (center - sigma, center + sigma, STATE_COLORS["background"], "background"),
        (center + sigma, np.inf, STATE_COLORS["high_impedance"], "high"),
    ]:
        mask = (residual >= left) & (residual <= right)
        if np.any(mask):
            ax.hist(residual[mask], bins=bins, color=color, alpha=0.65, label=label)
    ax.axvline(center, color="black", linewidth=1.0, label="center")
    ax.axvline(center - sigma, color="black", linestyle="--", linewidth=1.0, label="+/- 1 sigma")
    ax.axvline(center + sigma, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"Residual threshold: {well_name} / {zone_id}")
    ax.set_xlabel("full log(AI) - background")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="best", fontsize=8)
    generated.append(_finish(fig, out_dir / "examples" / f"residual_threshold_{safe}.png"))
    return generated


def _plot_object_profile(profile_samples: pd.DataFrame, obj: pd.Series, out_dir: Path) -> str | None:
    subset = profile_samples[profile_samples["object_id"].eq(obj["object_id"])].sort_values("xi")
    if subset.empty:
        return None
    safe = _sanitize(obj["object_id"])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.scatter(subset["xi"], subset["residual"], s=16, color="#1a202c", label="residual samples")
    ax.plot(subset["xi"], subset["fitted_residual"], color="#2b6cb0", linewidth=1.8, label="profile fit")
    ax.set_title(f"Object profile: {obj['state']} / {obj['zone_id']}")
    ax.set_xlabel("object coordinate xi")
    ax.set_ylabel("residual log(AI)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return _finish(fig, out_dir / "examples" / f"object_profile_fit_{safe}.png")


def write_calibration_figures(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(config.get("enabled", True)):
        return _write_figure_manifest(
            Path(output_dir),
            [],
            [{"figure": "all", "reason": "figures disabled"}],
        )
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures" / "calibration"
    objects = _read_csv(output_dir / "well_object_catalog.csv")
    samples = _read_csv(output_dir / "well_calibration_samples.csv")
    backgrounds = _read_csv(output_dir / "well_background_fits.csv")
    profile_samples = _read_csv(output_dir / "well_object_profile_samples.csv")
    generated: list[str] = []
    skipped: list[dict[str, Any]] = []
    if objects.empty:
        skipped.append({"figure": "calibration", "reason": "empty well_object_catalog.csv"})
        return _write_figure_manifest(output_dir, generated, skipped)

    for zone_id, zone_objects in objects.groupby("zone_id", sort=False):
        safe_zone = _sanitize(zone_id)
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for state, group in zone_objects.groupby("state", sort=False):
            ax.hist(
                group["c0"].dropna().to_numpy(dtype=np.float64),
                bins=min(40, max(10, int(np.sqrt(len(group))))),
                alpha=0.55,
                color=STATE_COLORS.get(str(state), "#718096"),
                label=str(state),
            )
        ax.set_title(f"c0 distribution: {zone_id}")
        ax.set_xlabel("c0")
        ax.set_ylabel("object count")
        ax.grid(True, alpha=0.20)
        ax.legend(loc="best", fontsize=8)
        generated.append(_finish(fig, figures_dir / "summary" / f"c0_distribution_{safe_zone}.png"))

        size_col = _first_existing_column(zone_objects, ("duration_s", "thickness_m", "zone_thickness_m"))
        if size_col is None:
            skipped.append({"figure": f"duration_distribution_{zone_id}", "reason": "missing duration/thickness column"})
        else:
            states = [state for state in STATE_COLORS if state in set(zone_objects["state"])]
            if not states:
                skipped.append({"figure": f"duration_distribution_{zone_id}", "reason": "no known impedance state"})
            else:
                fig, axes = plt.subplots(len(states), 1, figsize=(6.0, max(2.4, 2.0 * len(states))), sharex=True)
                axes = np.atleast_1d(axes)
                for ax, state in zip(axes, states):
                    values = np.log(np.maximum(zone_objects.loc[zone_objects["state"].eq(state), size_col], 1e-9))
                    ax.hist(values, bins=min(35, max(8, int(np.sqrt(len(values))))), color=STATE_COLORS[state], alpha=0.70)
                    if len(values):
                        ax.axvline(float(np.median(values)), color="black", linestyle="--", linewidth=1.0)
                    ax.set_ylabel(state.replace("_impedance", ""))
                    ax.grid(True, alpha=0.20)
                axes[-1].set_xlabel(f"log({size_col})")
                fig.suptitle(f"Duration/thickness distribution: {zone_id}")
                generated.append(_finish(fig, figures_dir / "summary" / f"duration_distribution_{safe_zone}.png"))

    calibration_path = output_dir / "impedance_calibration.json"
    if calibration_path.is_file():
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        for zone_id, model in dict(calibration.get("zone_models") or {}).items():
            matrix = np.asarray(model.get("transition_matrix"), dtype=np.float64)
            if matrix.shape != (3, 3):
                skipped.append({"figure": f"transition_matrix_{zone_id}", "reason": "invalid matrix shape"})
                continue
            fig, ax = plt.subplots(figsize=(4.6, 4.0))
            image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(1e-9, float(np.nanmax(matrix))))
            ax.set_xticks(range(3), ["L", "B", "H"])
            ax.set_yticks(range(3), ["L", "B", "H"])
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"Transition matrix: {zone_id}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            generated.append(_finish(fig, figures_dir / "summary" / f"transition_matrix_{_sanitize(zone_id)}.png"))

    max_examples = int(config.get("max_example_objects_per_zone_state", 1))
    if not samples.empty:
        requested_zone = str(
            dict(config.get("report_examples") or {}).get("zone_id") or ""
        ).strip()
        zone_ids = (
            [requested_zone]
            if requested_zone and requested_zone in set(samples["zone_id"])
            else objects["zone_id"].drop_duplicates().tolist()
        )
        for zone_id in zone_ids:
            picked = _pick_well_zone(samples, config, str(zone_id))
            if picked is None:
                skipped.append({"figure": f"background/residual:{zone_id}", "reason": "no sample rows"})
                continue
            generated.extend(
                _plot_background_and_residual(
                    samples,
                    backgrounds,
                    well_name=picked[0],
                    zone_id=picked[1],
                    out_dir=figures_dir,
                )
            )
    else:
        skipped.append({"figure": "background/residual examples", "reason": "empty well_calibration_samples.csv"})

    if not profile_samples.empty:
        for (_, state), group in objects.groupby(["zone_id", "state"], sort=False):
            examples = group.sort_values("n_truth_samples", ascending=False).head(max_examples)
            for _, obj in examples.iterrows():
                path = _plot_object_profile(profile_samples, obj, figures_dir)
                if path is None:
                    skipped.append({"figure": f"object_profile:{obj['object_id']}", "reason": "no profile samples"})
                else:
                    generated.append(path)
    else:
        skipped.append({"figure": "object profile examples", "reason": "empty well_object_profile_samples.csv"})
    result = _write_figure_manifest(output_dir, generated, skipped)
    result["generated"] = generated
    result["skipped"] = skipped
    return result


def _plot_section_geometry(output_dir: Path, generated: list[str], skipped: list[dict[str, Any]]) -> None:
    frame = _read_csv(output_dir / "section_geometry_qc.csv")
    if frame.empty:
        skipped.append({"figure": "section_geometry_support", "reason": "missing or empty section_geometry_qc.csv"})
        return
    horizon_col = _first_existing_column(frame, ("horizon_twt_s", "horizon_tvdss_m"))
    if horizon_col is None:
        skipped.append({"figure": "section_geometry_support", "reason": "missing horizon_twt_s/horizon_tvdss_m column"})
        return
    figures_dir = output_dir / "figures" / "geometry"
    for section_id, section in frame.groupby("section_id", sort=False):
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        for horizon, group in section.groupby("horizon_name", sort=False):
            group = group.sort_values("lateral_m")
            ax.plot(group["lateral_m"], group[horizon_col], linewidth=1.5, label=str(horizon))
            filled = group[~group["trace_valid_control"].astype(bool)]
            if not filled.empty:
                ax.scatter(filled["lateral_m"], filled[horizon_col], s=10, marker="x", label=f"{horizon} filled")
        ax.invert_yaxis()
        ax.set_title(f"Section geometry support: {section_id}")
        ax.set_xlabel("lateral distance (m)")
        ax.set_ylabel(_axis_label(horizon_col))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7, ncol=2)
        generated.append(_finish(fig, figures_dir / f"section_geometry_support_{_sanitize(section_id)}.png"))


def _select_base_sample(index: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series | None:
    if index.empty:
        return None
    base = index[index["sample_kind"].eq("base") & index["status"].eq("ok")].copy()
    if base.empty:
        return None
    examples = dict(config.get("report_examples") or {})
    section_id = str(examples.get("section_id") or "").strip()
    geometry_family = str(examples.get("geometry_family") or "").strip()
    if section_id:
        selected = base[base["section_id"].eq(section_id)]
        if not selected.empty:
            base = selected
    if geometry_family:
        selected = base[base["geometry_family"].eq(geometry_family)]
        if not selected.empty:
            base = selected
    else:
        selected = base[base["geometry_family"].eq("none")]
        if not selected.empty:
            base = selected
    sort_cols = [name for name in ("section_id", "scenario_id", "attempt_id", "parent_realization_id", "sample_id") if name in base.columns]
    if sort_cols:
        base = base.sort_values(sort_cols)
    return base.iloc[0]


def _imshow_section(
    values: np.ndarray,
    *,
    lateral: np.ndarray,
    vertical: np.ndarray,
    vertical_label: str,
    title: str,
    cmap: str,
    path: Path,
    symmetric: bool = False,
) -> str:
    array = np.asarray(values, dtype=np.float64)
    extent = [float(lateral[0]), float(lateral[-1]), float(vertical[-1]), float(vertical[0])]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    if symmetric:
        limit = float(np.nanpercentile(np.abs(array), 99.0))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = np.nanpercentile(array, [1.0, 99.0])
    image = ax.imshow(array.T, aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("lateral distance (m)")
    ax.set_ylabel(vertical_label)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig, path)


def _plot_hdf5_examples(
    output_dir: Path,
    config: Mapping[str, Any],
    generated: list[str],
    skipped: list[dict[str, Any]],
) -> None:
    index = _read_csv(output_dir / "sample_index.csv")
    row = _select_base_sample(index, config)
    if row is None:
        skipped.append({"figure": "generation examples", "reason": "no accepted base sample"})
        return
    h5_path = output_dir / "synthetic_benchmark.h5"
    if not h5_path.is_file():
        skipped.append({"figure": "generation examples", "reason": "synthetic_benchmark.h5 not written"})
        return
    group_path = str(row["hdf5_group"])
    if not group_path:
        skipped.append({"figure": "generation examples", "reason": "selected sample has empty hdf5_group"})
        return
    sample_label = _sample_label(row)
    safe = _sanitize(sample_label)
    figures_dir = output_dir / "figures" / "generation"
    with h5py.File(h5_path, "r") as h5:
        if group_path not in h5:
            skipped.append({"figure": f"generation examples:{group_path}", "reason": "missing HDF5 group"})
            return
        group = h5[group_path]
        lateral = group["axes/lateral_m"][()]
        model_axis_name = "twt_model_s" if "axes/twt_model_s" in group else "tvdss_model_m"
        highres_axis_name = "twt_highres_s" if "axes/twt_highres_s" in group else "tvdss_highres_m"
        if f"axes/{model_axis_name}" not in group:
            skipped.append({"figure": f"generation examples:{group_path}", "reason": "missing model vertical axis"})
            return
        if f"axes/{highres_axis_name}" not in group:
            skipped.append({"figure": f"state_strip:{group_path}", "reason": "missing high-resolution vertical axis"})
            highres_axis_name = model_axis_name
        model_axis = group[f"axes/{model_axis_name}"][()]
        highres_axis = group[f"axes/{highres_axis_name}"][()]
        seismic_axis = group["axes/twt_forward_model_s"][()] if "axes/twt_forward_model_s" in group else model_axis
        model_axis_label = _axis_label(model_axis_name)
        seismic_axis_label = _axis_label("twt_model_s" if "axes/twt_forward_model_s" in group else model_axis_name)
        generated.append(
            _imshow_section(
                group["truth/model_target_log_ai"][()],
                lateral=lateral,
                vertical=model_axis,
                vertical_label=model_axis_label,
                title=f"log(AI) target: {sample_label}",
                cmap="viridis",
                path=figures_dir / f"log_ai_section_{safe}.png",
            )
        )
        generated.append(
            _imshow_section(
                group["seismic/seismic_model_consistent"][()],
                lateral=lateral,
                vertical=seismic_axis,
                vertical_label=seismic_axis_label,
                title=f"Model-consistent seismic: {sample_label}",
                cmap="seismic",
                path=figures_dir / f"seismic_section_{safe}.png",
                symmetric=True,
            )
        )
        state_path = "truth/state_id_highres" if "truth/state_id_highres" in group else "truth/categorical/state_id_highres"
        if state_path in group:
            state = group[state_path][()]
            center = state.shape[0] // 2
            color_values = np.asarray(state[center], dtype=float)
            color_values[color_values < 0] = np.nan
            fig, ax = plt.subplots(figsize=(8.0, 1.8))
            ax.imshow(
                color_values[np.newaxis, :],
                aspect="auto",
                extent=[highres_axis[0], highres_axis[-1], 0, 1],
                cmap="coolwarm",
                vmin=0,
                vmax=2,
            )
            valid_idx = np.where(color_values >= 0)[0]
            if len(valid_idx) > 0:
                ax.set_xlim(highres_axis[valid_idx[0]], highres_axis[valid_idx[-1]])
            ax.set_yticks([])
            ax.set_xlabel(_axis_label(highres_axis_name))
            ax.set_title(f"State strip, central trace: {sample_label}")
            generated.append(_finish(fig, figures_dir / f"state_strip_{safe}.png"))
        else:
            skipped.append({"figure": f"state_strip:{group_path}", "reason": "missing state_id_highres dataset"})
        if "priors/lfm_controlled_degraded" in group:
            model = group["truth/model_target_log_ai"][()]
            lfm = group["priors/lfm_controlled_degraded"][()]
            residual = (
                group["residuals/residual_vs_lfm_controlled_degraded"][()]
                if "residuals/residual_vs_lfm_controlled_degraded" in group
                else model - lfm
            )
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True, sharey=True)
            for ax, values, title, cmap in [
                (axes[0], model, "model target log(AI)", "viridis"),
                (axes[1], lfm, "controlled degraded LFM", "viridis"),
                (axes[2], residual, "target - controlled LFM", "seismic"),
            ]:
                symmetric = cmap == "seismic"
                if symmetric:
                    limit = float(np.nanpercentile(np.abs(values), 99.0))
                    vmin, vmax = -limit, limit
                else:
                    vmin, vmax = np.nanpercentile(values, [1.0, 99.0])
                image = ax.imshow(
                    values.T,
                    aspect="auto",
                    extent=[lateral[0], lateral[-1], model_axis[-1], model_axis[0]],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(title)
                ax.set_ylabel(model_axis_label)
                fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
            axes[-1].set_xlabel("lateral distance (m)")
            generated.append(_finish(fig, figures_dir / f"lfm_residual_overview_{safe}.png"))
        else:
            skipped.append({"figure": f"lfm_residual_overview:{group_path}", "reason": "missing controlled degraded LFM"})

    coeffs = _read_csv(output_dir / "object_lateral_coefficients.csv")
    if coeffs.empty:
        skipped.append({"figure": "c0_lateral", "reason": "empty object_lateral_coefficients.csv"})
        return
    realization_col = _first_existing_column(coeffs, ("realization_id", "parent_realization_id"))
    if realization_col is None:
        skipped.append({"figure": "c0_lateral", "reason": "missing realization_id/parent_realization_id column"})
        return
    subset = coeffs[coeffs[realization_col].eq(sample_label)]
    if subset.empty:
        skipped.append({"figure": "c0_lateral", "reason": "no coefficients for selected realization"})
        return
    candidates = subset[~subset["state"].eq("background")]
    if candidates.empty:
        candidates = subset
    object_id = candidates.groupby("object_id").size().sort_values(ascending=False).index[0]
    obj = subset[subset["object_id"].eq(object_id)].sort_values("lateral_m")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(obj["lateral_m"], obj["c0"], color="#2b6cb0", linewidth=1.8)
    ax.set_title(f"Actual c0(x): {sample_label} / object {object_id}")
    ax.set_xlabel("lateral distance (m)")
    ax.set_ylabel("c0")
    ax.grid(True, alpha=0.25)
    generated.append(_finish(fig, figures_dir / f"c0_lateral_{safe}_object_{_sanitize(object_id)}.png"))


def _plot_acceptance(output_dir: Path, generated: list[str], skipped: list[dict[str, Any]]) -> None:
    catalog = _read_csv(output_dir / "scenario_catalog.csv")
    if catalog.empty or "acceptance_fraction" not in catalog:
        skipped.append({"figure": "scenario_acceptance_summary", "reason": "empty scenario_catalog.csv"})
        return
    frame = catalog.copy()
    frame["label"] = frame["scenario_id"].astype(str)
    frame = frame.sort_values("acceptance_fraction")
    fig, ax = plt.subplots(figsize=(max(7.0, 0.18 * len(frame)), 4.2))
    ax.bar(np.arange(len(frame)), frame["acceptance_fraction"], color="#2b6cb0")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("acceptance fraction")
    ax.set_title("Scenario acceptance summary")
    ax.set_xticks(np.arange(len(frame)), frame["label"], rotation=90, fontsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    generated.append(_finish(fig, output_dir / "figures" / "qc" / "scenario_acceptance_summary.png"))


def write_generation_figures(
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    suite: str,
    qc_only: bool,
) -> dict[str, Any]:
    if not bool(config.get("enabled", True)):
        return _write_figure_manifest(
            Path(output_dir),
            [],
            [{"figure": "all", "reason": "figures disabled"}],
        )
    output_dir = Path(output_dir)
    generated: list[str] = []
    skipped: list[dict[str, Any]] = []
    if suite == "field_conditioned":
        _plot_section_geometry(output_dir, generated, skipped)
    _plot_acceptance(output_dir, generated, skipped)
    if qc_only:
        skipped.append({"figure": "hdf5 generation examples", "reason": "qc_only run"})
    else:
        _plot_hdf5_examples(output_dir, config, generated, skipped)
    result = _write_figure_manifest(output_dir, generated, skipped)
    result["generated"] = generated
    result["skipped"] = skipped
    return result
