"""Prepare a 2-D Marmousi2 experiment using the existing workflow artifacts."""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cup.physics.numpy_backend import forward_time
from cup.seismic.geometry import SampleAxis
from cup.seismic.wavelet import wavelet_l2_normalize
from cup.utils.io import repo_relative_path, write_json
from cup.well.controls import (
    MANIFEST_COLUMNS, NativeWellControl, WellControl, WellControlSet,
    write_well_control_set,
)
from cup.well.evaluation_support import derive_evaluation_support, write_evaluation_support_manifest
from marmousi2.raw import read_marmousi_models, resample_models_to_time
from marmousi2.seismic import read_processed_time_segy, resample_processed_seismic, estimate_training_wavelet
from wtie.processing import grid


@dataclass(frozen=True)
class PrepareSettings:
    trace_stride: int = 8
    depth_stride: int = 1
    # The published Madagascar/RSF model recipe declares 0.001249 km.
    dx_m: float = 1.249
    dz_m: float = 1.249
    dt_s: float = 0.004
    wavelet_frequency_hz: float = 30.0
    lfm_cutoff_hz: float = 5.0
    train_well_fractions: tuple[float, ...] = (0.180882, 0.325735, 0.687132)
    validation_well_fraction: float = 0.776471
    test_well_fraction: float = 0.841176
    target_top_s: float = 0.65
    target_bottom_buffer_s: float = 0.08
    seed: int = 20261004
    wavelet_duration_s: float = 0.16
    wavelet_ridge_fraction: float = 0.01
    wavelet_method: str = "workflow"

    def __post_init__(self):
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive.")
        fractions = (*self.train_well_fractions, self.validation_well_fraction, self.test_well_fraction)
        if not self.train_well_fractions or any(not 0 < f < 1 for f in fractions):
            raise ValueError("Pseudo-well fractions must lie strictly between zero and one.")
        if len(set(fractions)) != len(fractions):
            raise ValueError("Training, validation and test pseudo-wells must be distinct.")
        if not 0 < self.lfm_cutoff_hz < 0.5 / self.dt_s:
            raise ValueError("LFM cutoff must be positive and below Nyquist.")
        if not 0 < self.wavelet_frequency_hz < 0.5 / self.dt_s:
            raise ValueError("Wavelet frequency must be positive and below Nyquist.")
        if not np.isfinite(self.target_top_s) or self.target_top_s < 0:
            raise ValueError("target_top_s must be finite and non-negative.")
        if not np.isfinite(self.target_bottom_buffer_s) or self.target_bottom_buffer_s < 0:
            raise ValueError("target_bottom_buffer_s must be finite and non-negative.")


def _controls(log_ai, axis, x_m, roles):
    controls, rows = [], []
    for role in roles:
        if role["role"] != "train":
            continue
        index, name = role["profile_index"], role["well_name"]
        values = np.asarray(log_ai[index], dtype=float)
        mask = np.isfinite(values)
        provenance = {"source": "Marmousi2 synthetic pseudo-well", "role": "train", "extra_smoothing": False}
        native = NativeWellControl(name, axis.values, values, mask, "time", "s", None, provenance)
        n = len(values)
        controls.append(WellControl(
            name, axis, grid.Log(values, axis.values, "twt", name=name, unit="log(m/s*g/cm3)"),
            np.zeros(n), np.full(n, index), np.full(n, x_m[index]), np.zeros(n),
            mask, mask.copy(), "vertical", "synthetic_model_axis", "marmousi2_synthetic", provenance, native,
        ))
        row = dict.fromkeys(MANIFEST_COLUMNS, "")
        row.update(
            well_name=name, status="ok", source_run_type="marmousi2_synthetic", wellbore_class="vertical",
            sample_domain="time", sample_unit="s", sampling_mode="synthetic_model_axis", n_samples=n,
            n_valid_samples=n, n_observed_samples=n, n_interpolated_samples=0, n_native_samples=n,
            n_valid_native_samples=n, sample_min=float(axis.values[0]), sample_max=float(axis.values[-1]),
        )
        rows.append(row)
    return WellControlSet(axis, tuple(controls), "time", "s", None, "marmousi2_synthetic", {}), pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _assets(output_dir, models, time_model, roles):
    """Export only training pseudo-wells to upstream well assets."""
    import lasio

    assets = output_dir / "assets"
    for name in ("las", "well_trace", "time_depth"):
        (assets / name).mkdir(parents=True, exist_ok=True)
    heads, tops = [], []
    for role in roles:
        if role["role"] != "train":
            continue
        i, name = role["profile_index"], role["well_name"]
        las = lasio.LASFile()
        las.well.WELL = name
        las.append_curve("DEPT", models.depth_m, unit="m")
        las.append_curve("VP", models.vp_mps[i], unit="m/s")
        las.append_curve("RHOB", models.rho_gcc[i], unit="g/cm3")
        las.append_curve("AI", models.vp_mps[i] * models.rho_gcc[i], unit="m/s*g/cm3")
        las.write(str(assets / "las" / f"{name}.las"), version=2.0)
        pd.DataFrame({"depth_m": models.depth_m, "twt_s": time_model.native_twt_s[i]}).to_csv(
            assets / "time_depth" / f"{name}.csv", index=False,
        )
        pd.DataFrame({"depth_m": models.depth_m, "x_m": models.x_m[i], "y_m": 0.0}).to_csv(
            assets / "well_trace" / f"{name}.csv", index=False,
        )
        heads.append({"well_name": name, "x_m": models.x_m[i], "y_m": 0.0})
        tops.extend({"well_name": name, "horizon": label, "twt_s": value} for label, value in (
            ("experiment_top", time_model.twt_s[0]), ("experiment_bottom", time_model.twt_s[-1]),
        ))
    pd.DataFrame(heads).to_csv(assets / "well_heads.csv", index=False)
    pd.DataFrame(tops).to_csv(assets / "well_tops.csv", index=False)
    return assets


def prepare_marmousi2(vp_path: Path, density_path: Path, output_dir: Path, *, repo_root: Path, settings=None,
                     seismic_path: Path | None = None, wavelet_path: Path | None = None):
    """Prepare model labels and either observed AGL data or controlled convolution data."""
    settings = settings or PrepareSettings()
    output_dir, repo_root = output_dir.resolve(), repo_root.resolve()
    models = read_marmousi_models(
        vp_path, density_path, trace_stride=settings.trace_stride, depth_stride=settings.depth_stride,
        dx_m=settings.dx_m, dz_m=settings.dz_m,
    )
    time_model = resample_models_to_time(models, dt_s=settings.dt_s)
    truth = time_model.log_ai
    nx, nt = truth.shape
    if nx < 25:
        raise ValueError("The prepared profile must contain at least 25 traces for 17-trace patches and a spatial split.")
    names_roles = [(f"MARM_TRAIN_{i+1}", "train", f) for i, f in enumerate(settings.train_well_fractions)]
    names_roles += [("MARM_VALIDATION", "validation", settings.validation_well_fraction), ("MARM_TEST", "test", settings.test_well_fraction)]
    roles = [{"well_name": name, "role": role, "profile_index": round(f * (nx - 1)), "fraction": f} for name, role, f in names_roles]
    if len({r["profile_index"] for r in roles}) != len(roles):
        raise ValueError("Sampling maps different pseudo-wells onto the same trace; use a finer trace stride.")
    for role in roles:
        role["x_m"] = float(models.x_m[role["profile_index"]])
    if seismic_path is None:
        half = max(2, int(np.ceil(3.0 / settings.wavelet_frequency_hz / settings.dt_s)))
        wavelet_axis = np.arange(-half, half + 1, dtype=float) * settings.dt_s
        a = np.pi * settings.wavelet_frequency_hz * wavelet_axis
        wavelet, _ = wavelet_l2_normalize((1.0 - 2.0 * a * a) * np.exp(-a * a))
        observed = forward_time(truth, wavelet_axis, wavelet, sample_step_s=settings.dt_s).astype(np.float32)
        seismic_source = {"kind": "controlled_convolution", "description": "known Ricker wavelet; no noise"}
        wavelet_source = {"method": "known_ricker", "frequency_hz": settings.wavelet_frequency_hz}
    else:
        amplitudes, source_x, source_t, info = read_processed_time_segy(seismic_path)
        observed = resample_processed_seismic(amplitudes, source_x, source_t, models.x_m, time_model.twt_s)
        seismic_source = {"kind": "agl_processed_time", "description": Path(seismic_path).stem,
                          "file": str(Path(seismic_path).resolve()), "native": info,
                          "resampling": "linear onto model x/TWT; common supported model bottom"}
        if wavelet_path is None and settings.wavelet_method == "ridge":
            training_indices = [role["profile_index"] for role in roles if role["role"] == "train"]
            wavelet_axis, wavelet, wavelet_source = estimate_training_wavelet(
                truth, observed, time_model.twt_s, training_indices,
                duration_s=settings.wavelet_duration_s, ridge_fraction=settings.wavelet_ridge_fraction,
            )
            wavelet_source["training_well_names"] = [role["well_name"] for role in roles if role["role"] == "train"]
        elif wavelet_path is not None:
            from cup.seismic.wavelet import load_wavelet_csv
            wavelet_axis, wavelet = load_wavelet_csv(wavelet_path)
            wavelet, _ = wavelet_l2_normalize(wavelet)
            wavelet_source = {"method": "provided_csv", "file": str(Path(wavelet_path).resolve())}
        else:
            wavelet_axis = np.arange(-25, 26) * settings.dt_s
            a = np.pi * settings.wavelet_frequency_hz * wavelet_axis
            wavelet, _ = wavelet_l2_normalize((1.0 - 2.0 * a * a) * np.exp(-a * a))
            wavelet_source = {"method": "workflow_steps_4_5_pending", "initialization_frequency_hz": settings.wavelet_frequency_hz}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation").mkdir(exist_ok=True)
    # Full labels and held-out wells are never referenced by training config.
    np.savez_compressed(output_dir / "evaluation" / "truth.npz", log_ai=truth.astype(np.float32), x_m=models.x_m, twt_s=time_model.twt_s)
    pd.DataFrame(roles).to_csv(output_dir / "evaluation" / "well_roles.csv", index=False)
    # Keep original training pseudo-well curves as offline evaluation targets.
    training_roles = [role for role in roles if role["role"] == "train"]
    training_indices = np.asarray([role["profile_index"] for role in training_roles], dtype=int)
    np.savez_compressed(
        output_dir / "evaluation" / "well_targets.npz",
        profile_index=training_indices.astype(np.int64),
        fixed_log_ai=truth[training_indices].astype(np.float32),
        twt_s=time_model.twt_s.astype(np.float64),
    )
    spacing = float(models.x_m[1] - models.x_m[0])
    np.savez_compressed(
        output_dir / "seismic.npz", seismic=observed[None, :, :], sample_values=time_model.twt_s,
        sample_domain=np.asarray("time"), sample_unit=np.asarray("s"), depth_basis=np.asarray(""),
        ilines=np.asarray([0.0]), xlines=np.arange(nx, dtype=float), origin_xy_m=np.asarray([models.x_m[0], 0.0]),
        inline_step_xy_m=np.asarray([0.0, spacing]), xline_step_xy_m=np.asarray([spacing, 0.0]),
    )
    wavelet_dir = output_dir / "wavelet"
    wavelet_dir.mkdir(exist_ok=True)
    wavelet_filename = "initial_wavelet.csv" if wavelet_source["method"] == "workflow_steps_4_5_pending" else "selected_wavelet.csv"
    pd.DataFrame({"time_s": wavelet_axis, "amplitude": wavelet}).to_csv(wavelet_dir / wavelet_filename, index=False)
    write_json(wavelet_dir / "wavelet_estimation.json", wavelet_source)
    axis = SampleAxis(time_model.twt_s, "time", "s")
    controls, manifest = _controls(truth, axis, models.x_m, roles)
    controls_dir = output_dir / "well_controls"
    write_well_control_set(controls, manifest, output_dir=controls_dir, repo_root=repo_root, resolved_config={"adapter": "marmousi2", "pseudo_well_roles": roles})
    # Persist the explicit target used by the Step 6/8 well evaluation.
    target_interval = (settings.target_top_s, float(time_model.twt_s[-1] - settings.target_bottom_buffer_s))
    evaluation_support = {}
    for role in roles:
        index = int(role["profile_index"])
        evaluation_support[role["well_name"]] = derive_evaluation_support(
            well_name=role["well_name"],
            sample_axis=axis,
            target_interval=target_interval,
            observed_support=np.isfinite(observed[index]),
            well_curve_support=np.isfinite(truth[index]),
        )
    write_evaluation_support_manifest(
        controls_dir / "qc" / "evaluation_support.json",
        sample_axis=axis,
        supports=evaluation_support,
        source="marmousi2_prepare: fixed target interval",
    )
    _assets(output_dir, models, time_model, roles)
    relative = lambda p: repo_relative_path(p, root=repo_root)
    # Initial models belong to scripts/marmousi2_prepare_lfms.py.  This config
    # points at the training-well trend LFM that step publishes; the LFM step
    # rewrites the pointer once all three variants exist.
    lfm_dir = output_dir / "lfm_models" / "well_huber_trend"
    # Keep benchmark runs beside the public prepared inputs.  The generic
    # workflow resolves this path against the repository root, so an explicit
    # path here prevents a public Marmousi2 run from silently writing into the
    # private legacy ``scripts/output`` tree.
    config = {
        "data_root": relative(output_dir), "output_root": relative(output_dir / "runs"),
        "assets": {"well_heads_file": "assets/well_heads.csv", "las_dir": "assets/las", "well_trace_dir": "assets/well_trace", "well_tops_file": "assets/well_tops.csv", "time_depth_dir": "assets/time_depth"},
        "seismic": {"file": "seismic.npz", "type": "npz", "domain": "time"},
        "well_curves": {"required_categories": ["AI"], "selected_categories": ["AI", "VP", "RHOB"]},
        "spatial_debias": {"cluster_radius_m": spacing},
        "ginn_v3_body_inversion": {
            "inputs": {
                "lfm_run_dir": relative(lfm_dir),
                "variant_id": "well_huber_trend",
                "well_control_run_dir": relative(controls_dir),
                "wavelet_generation_run_dir": relative(
                    output_dir / "step5_wavelet_generation"
                    if wavelet_source["method"] == "workflow_steps_4_5_pending"
                    else wavelet_dir
                ),
            },
            "trusted_well_names": [r["well_name"] for r in roles if r["role"] == "train"],
            "network": {
                "tcn_channels": [16, 16, 16],
                "hidden_channels": 32,
                "kernel_size": 3,
                "dilation": 2,
                "gru_layers": 3,
                "dropout": 0.0,
            },
            "training": {
                "updates": 1000,
                "labeled_batch_size": 6,
                "unlabeled_batch_size": 32,
                "learning_rate": 0.004,
                "weight_decay": 0.01,
                "seed": settings.seed,
                "validate_every": 100,
                "log_every": 20,
                "max_train_traces": 4096,
                "validation_traces": 128,
                "validation_gap_m": max(300.0, 17 * spacing),
                "min_support_samples": 8,
                "device": "cpu",
                "loss_weights": {"independent": 1.0, "physics": 1.0, "cross": 1.0},
            },
            "inference": {"batch_size": 32, "min_support_samples": 8},
        },
    }
    config_path = output_dir / "ginn_v3.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {
        "schema_version": "marmousi2_prepared_v1", "status": "prepared", "settings": asdict(settings),
        "spatial_sampling_reference": "https://ahay.org/RSF/book/data/marmousi2/paper_html/node2.html; explicit dx_m/dz_m override allowed",
        "native_models": models.metadata, "time_resampling": time_model.metadata,
        "shape": [1, nx, nt], "orientation": "inline", "spatial_dimension": 2, "replicated_axis": False,
        "seismic_generation": seismic_source["description"], "seismic_source": seismic_source,
        "wavelet_source": wavelet_source,
        "evaluation_support_manifest": relative(controls_dir / "qc" / "evaluation_support.json"),
        "target_interval_s": list(target_interval),
        "lfm_source": "training pseudo-wells only; existing TrendBuilder", "training_labels": [r["well_name"] for r in roles if r["role"] == "train"],
        "held_out_labels": [r["well_name"] for r in roles if r["role"] != "train"],
        "config_path": relative(config_path),
    }
    write_json(output_dir / "preparation_summary.json", summary)
    return config_path
