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
    if "tvdss" in axis_name or "depth" in axis_name:
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


def write_figure_failure_manifest(
    output_dir: Path,
    *,
    scope: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Publish a diagnostic warning record when optional plotting fails."""
    return _write_figure_manifest(
        Path(output_dir),
        [],
        [{
            "figure": str(scope),
            "reason": f"{type(exc).__name__}: {exc}",
        }],
    )


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


def _plot_well_calibration_group(
    *,
    well_name: str,
    samples: pd.DataFrame,
    objects: pd.DataFrame,
    profile_samples: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    """Write a compact, repeatable calibration report for one source well."""
    well_samples = samples.loc[samples["well_name"].eq(well_name)].copy()
    well_objects = objects.loc[objects["well_name"].eq(well_name)].copy()
    well_profiles = profile_samples.loc[
        profile_samples["well_name"].eq(well_name)
    ].copy()
    if well_samples.empty or well_objects.empty:
        return []
    axis_col = _first_existing_column(well_samples, ("twt_s", "tvdss_m"))
    if axis_col is None:
        return []
    safe_well = _sanitize(well_name)
    well_dir = out_dir / "wells" / safe_well
    generated: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.4), sharey=True)
    for zone_id, zone in well_samples.groupby("zone_id", sort=False):
        zone = zone.sort_values(axis_col)
        axes[0].plot(
            zone["full_log_ai"],
            zone[axis_col],
            linewidth=1.0,
            alpha=0.75,
            label=f"{zone_id}: full",
        )
        axes[0].plot(
            zone["background_log_ai"],
            zone[axis_col],
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
            label=f"{zone_id}: background",
        )
        axes[1].plot(
            zone["residual"],
            zone[axis_col],
            linewidth=1.0,
            label=str(zone_id),
        )
    axes[0].set_xlabel("log(AI)")
    axes[0].set_ylabel(_axis_label(axis_col))
    axes[0].set_title("Well log and zone-linear background")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("full log(AI) - background")
    axes[1].set_title("Calibrated residual")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best", fontsize=6)
    fig.suptitle(f"Calibration overview: {well_name}")
    generated.append(_finish(fig, well_dir / "background_and_residual.png"))

    if not well_profiles.empty:
        valid = well_profiles[
            np.isfinite(well_profiles["residual"])
            & np.isfinite(well_profiles["fitted_residual"])
        ]
        if not valid.empty:
            fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
            for state, group in valid.groupby("state", sort=False):
                axes[0].scatter(
                    group["residual"],
                    group["fitted_residual"],
                    s=5,
                    alpha=0.35,
                    color=STATE_COLORS.get(str(state), "#718096"),
                    label=str(state),
                )
            bounds = np.nanpercentile(
                np.concatenate([
                    valid["residual"].to_numpy(dtype=np.float64),
                    valid["fitted_residual"].to_numpy(dtype=np.float64),
                ]),
                [0.5, 99.5],
            )
            axes[0].plot(bounds, bounds, color="black", linestyle="--", linewidth=1.0)
            axes[0].set_xlabel("calibration residual")
            axes[0].set_ylabel("three-parameter fitted residual")
            axes[0].set_title("Profile parity")
            axes[0].legend(loc="best", fontsize=7)
            fit_error = (
                valid["residual"].to_numpy(dtype=np.float64)
                - valid["fitted_residual"].to_numpy(dtype=np.float64)
            )
            axes[1].hist(
                fit_error,
                bins=min(50, max(15, int(np.sqrt(fit_error.size)))),
                color="#2b6cb0",
                alpha=0.75,
            )
            axes[1].axvline(0.0, color="black", linewidth=1.0)
            axes[1].set_xlabel("profile fit residual")
            axes[1].set_ylabel("sample count")
            axes[1].set_title("Profile fit error")
            for ax in axes:
                ax.grid(True, alpha=0.22)
            fig.suptitle(f"Three-parameter profile fit: {well_name}")
            generated.append(_finish(fig, well_dir / "profile_fit.png"))

    coordinate = (
        well_objects["zone_id"].astype("category").cat.codes.to_numpy(dtype=float)
        + 0.5
        * (
            well_objects["zeta_top"].to_numpy(dtype=np.float64)
            + well_objects["zeta_bottom"].to_numpy(dtype=np.float64)
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.0), sharey=True)
    for ax, coefficient in zip(axes, ("c0", "c1", "c2")):
        for state, color in STATE_COLORS.items():
            mask = well_objects["state"].eq(state).to_numpy()
            if np.any(mask):
                ax.scatter(
                    well_objects.loc[mask, coefficient],
                    coordinate[mask],
                    s=10,
                    alpha=0.65,
                    color=color,
                    label=state,
                )
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(coefficient)
        ax.set_title(f"{coefficient} by object position")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("zone order + normalized position")
    axes[0].invert_yaxis()
    axes[-1].legend(loc="best", fontsize=7)
    fig.suptitle(f"Object parameter calibration: {well_name}")
    generated.append(_finish(fig, well_dir / "object_parameters.png"))
    return generated


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

    well_names = sorted(
        set(objects["well_name"].dropna().astype(str))
        | set(samples.get("well_name", pd.Series(dtype=str)).dropna().astype(str))
    )
    for well_name in well_names:
        try:
            paths = _plot_well_calibration_group(
                well_name=well_name,
                samples=samples,
                objects=objects,
                profile_samples=profile_samples,
                out_dir=figures_dir,
            )
            if paths:
                generated.extend(paths)
            else:
                skipped.append({
                    "figure": f"well_calibration:{well_name}",
                    "reason": "well has no complete calibration rows",
                })
        except Exception as exc:
            plt.close("all")
            skipped.append({
                "figure": f"well_calibration:{well_name}",
                "reason": f"{type(exc).__name__}: {exc}",
            })

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
            if "trace_valid_control" in group:
                filled = group[~group["trace_valid_control"].astype(bool)]
            elif "support_status" in group:
                supported = (
                    group["support_status"]
                    .astype(str)
                    .str.casefold()
                    .isin({"ok", "supported"})
                )
                filled = group[~supported]
            else:
                filled = group.iloc[0:0]
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
    base = index.copy()
    if "sample_kind" in base:
        base = base[base["sample_kind"].eq("base")]
    if "status" in base:
        base = base[base["status"].eq("ok")]
    if base.empty:
        return None
    examples = dict(config.get("report_examples") or {})
    section_id = str(examples.get("section_id") or "").strip()
    geometry_family = str(examples.get("geometry_family") or "").strip()
    if section_id and "section_id" in base:
        selected = base[base["section_id"].eq(section_id)]
        if not selected.empty:
            base = selected
    if geometry_family and "geometry_family" in base:
        selected = base[base["geometry_family"].eq(geometry_family)]
        if not selected.empty:
            base = selected
    elif "geometry_family" in base:
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
    index_path = output_dir / "realization_index.csv"
    if not index_path.is_file():
        skipped.append({"figure": "generation examples", "reason": "missing realization_index.csv"})
        return
    index = _read_csv(index_path)
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
        # Canonical observed and model-consistent seismic share the model axis.
        seismic_axis = model_axis
        model_axis_label = _axis_label(model_axis_name)
        seismic_axis_label = model_axis_label
        generated.append(
            _imshow_section(
                group["truth/model_log_ai"][()],
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
                group["forward/model_consistent_seismic"][()],
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


def _plot_scenario_examples(
    output_dir: Path,
    generated: list[str],
    skipped: list[dict[str, Any]],
) -> None:
    """Write one compact visual summary for every accepted generation scenario."""
    index = _read_csv(output_dir / "realization_index.csv")
    if index.empty or "scenario_id" not in index:
        skipped.append({
            "figure": "scenario_examples",
            "reason": "realization index has no accepted scenarios",
        })
        return
    base = index.copy()
    if "status" in base:
        base = base.loc[base["status"].eq("ok")]
    if "sample_kind" in base:
        base = base.loc[base["sample_kind"].eq("base")]
    base = base.loc[base["hdf5_group"].fillna("").astype(str).ne("")]
    if base.empty:
        skipped.append({
            "figure": "scenario_examples",
            "reason": "no accepted HDF5-backed realization",
        })
        return
    sort_cols = [
        name
        for name in (
            "scenario_id",
            "corpus_role",
            "section_id",
            "attempt_id",
            "realization_id",
        )
        if name in base
    ]
    base = base.sort_values(sort_cols, kind="mergesort")
    selected = base.groupby("scenario_id", sort=True, as_index=False).head(1)
    h5_path = output_dir / "synthetic_benchmark.h5"
    if not h5_path.is_file():
        skipped.append({
            "figure": "scenario_examples",
            "reason": "synthetic_benchmark.h5 not written",
        })
        return
    figures_dir = output_dir / "figures" / "generation" / "scenarios"
    with h5py.File(h5_path, "r") as h5:
        for scenario_index, (_, row) in enumerate(
            selected.iterrows(),
            start=1,
        ):
            scenario_id = str(row["scenario_id"])
            group_path = str(row["hdf5_group"])
            try:
                if group_path not in h5:
                    raise KeyError(f"missing HDF5 group {group_path}")
                group = h5[group_path]
                lateral = np.asarray(group["axes/lateral_m"][()], dtype=np.float64)
                model_name = (
                    "twt_model_s"
                    if "axes/twt_model_s" in group
                    else "tvdss_model_m"
                )
                high_name = (
                    "twt_highres_s"
                    if "axes/twt_highres_s" in group
                    else "tvdss_highres_m"
                )
                model_axis = np.asarray(
                    group[f"axes/{model_name}"][()],
                    dtype=np.float64,
                )
                high_axis = np.asarray(
                    group[f"axes/{high_name}"][()],
                    dtype=np.float64,
                )
                log_ai = np.asarray(
                    group["truth/model_log_ai"][()],
                    dtype=np.float64,
                )
                seismic = np.asarray(
                    group["forward/model_consistent_seismic"][()],
                    dtype=np.float64,
                )
                state_path = (
                    "truth/state_id_highres"
                    if "truth/state_id_highres" in group
                    else "truth/categorical/state_id_highres"
                )
                if state_path not in group:
                    raise KeyError("missing high-resolution state dataset")
                state = np.asarray(group[state_path][()], dtype=np.float64)
                state[state < 0] = np.nan

                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(15.0, 5.0),
                    constrained_layout=True,
                )
                model_extent = [
                    float(lateral[0]),
                    float(lateral[-1]),
                    float(model_axis[-1]),
                    float(model_axis[0]),
                ]
                high_extent = [
                    float(lateral[0]),
                    float(lateral[-1]),
                    float(high_axis[-1]),
                    float(high_axis[0]),
                ]
                lo, hi = np.nanpercentile(log_ai, [1.0, 99.0])
                image = axes[0].imshow(
                    log_ai.T,
                    aspect="auto",
                    extent=model_extent,
                    cmap="viridis",
                    vmin=lo,
                    vmax=hi,
                )
                fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
                limit = max(
                    float(np.nanpercentile(np.abs(seismic), 99.0)),
                    np.finfo(np.float64).eps,
                )
                axes[1].imshow(
                    seismic.T,
                    aspect="auto",
                    extent=model_extent,
                    cmap="seismic",
                    vmin=-limit,
                    vmax=limit,
                )
                axes[2].imshow(
                    state.T,
                    aspect="auto",
                    extent=high_extent,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=2,
                    interpolation="nearest",
                )
                axes[0].set_title("Projected log(AI)")
                axes[1].set_title("Model-consistent seismic")
                axes[2].set_title("High-resolution state")
                for ax in axes:
                    ax.set_xlabel("lateral distance (m)")
                axes[0].set_ylabel(_axis_label(model_name))
                axes[1].set_ylabel(_axis_label(model_name))
                axes[2].set_ylabel(_axis_label(high_name))
                fig.suptitle(
                    f"Scenario {scenario_index:03d}: {scenario_id}\n"
                    f"example={_sample_label(row)}",
                    fontsize=10,
                )
                generated.append(_finish(
                    fig,
                    figures_dir / f"scenario_{scenario_index:03d}.png",
                ))
            except Exception as exc:
                plt.close("all")
                skipped.append({
                    "figure": f"scenario_example:{scenario_id}",
                    "reason": f"{type(exc).__name__}: {exc}",
                })


def _plot_acceptance(output_dir: Path, generated: list[str], skipped: list[dict[str, Any]]) -> None:
    catalog = _read_csv(output_dir / "scenario_catalog.csv")
    required = {"section_id", "scenario_id", "acceptance_fraction"}
    if catalog.empty or not required.issubset(catalog.columns):
        skipped.append({"figure": "scenario_acceptance_summary", "reason": "empty scenario_catalog.csv"})
        return
    frame = catalog.copy()
    sections = list(dict.fromkeys(frame["section_id"].astype(str)))
    scenarios = list(dict.fromkeys(frame["scenario_id"].astype(str)))
    matrix = frame.pivot(
        index="section_id", columns="scenario_id", values="acceptance_fraction"
    ).reindex(index=sections, columns=scenarios)
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=np.float64))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d1d5db")
    fig, ax = plt.subplots(
        figsize=(
            max(10.0, 0.34 * len(scenarios)),
            max(4.2, 0.55 * len(sections)),
        )
    )
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Scenario acceptance by section")
    ax.set_xlabel("scenario")
    ax.set_ylabel("section")
    ax.set_xticks(np.arange(len(scenarios)), scenarios, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(sections)), sections, fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("acceptance fraction")
    generated.append(_finish(
        fig,
        output_dir / "figures" / "generation" / "scenario_acceptance_summary.png",
    ))


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
        try:
            _plot_section_geometry(output_dir, generated, skipped)
        except Exception as exc:
            plt.close("all")
            skipped.append({
                "figure": "section_geometry",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    try:
        _plot_acceptance(output_dir, generated, skipped)
    except Exception as exc:
        plt.close("all")
        skipped.append({
            "figure": "scenario_acceptance",
            "reason": f"{type(exc).__name__}: {exc}",
        })
    if qc_only:
        skipped.append({"figure": "hdf5 generation examples", "reason": "qc_only run"})
    else:
        try:
            _plot_hdf5_examples(output_dir, config, generated, skipped)
        except Exception as exc:
            plt.close("all")
            skipped.append({
                "figure": "selected_generation_example",
                "reason": f"{type(exc).__name__}: {exc}",
            })
        try:
            _plot_scenario_examples(output_dir, generated, skipped)
        except Exception as exc:
            plt.close("all")
            skipped.append({
                "figure": "scenario_examples",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    result = _write_figure_manifest(output_dir, generated, skipped)
    result["generated"] = generated
    result["skipped"] = skipped
    return result
