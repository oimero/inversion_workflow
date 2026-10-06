"""Prepare a 2-D Marmousi2 experiment using the existing workflow artifacts."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from cup.config.workflow import WorkflowConfig
from cup.lfm.artifacts import load_lfm_input
from cup.lfm.math import parse_lowpass_spec
from cup.physics.numpy_backend import forward_time
from cup.seismic.geometry import SampleAxis
from cup.seismic.survey import open_survey
from cup.seismic.wavelet import wavelet_l2_normalize
from cup.utils.io import repo_relative_path, write_json
from cup.well.controls import (
    MANIFEST_COLUMNS, NativeWellControl, WellControl, WellControlSet,
    load_well_control_set, write_well_control_set,
)
from cup.well.evaluation_support import derive_evaluation_support, write_evaluation_support_manifest
from ginn_v2.data import (
    PatchReader, SurveyTraceSource, candidate_patch_keys, fit_lfm_normalization, seismic_profile_support,
)
from ginn_v2.model import CenterTraceBodyNet
from ginn_v2.train import BodyInversionConfig, BodyInversionTrainer, build_body_inversion_data
from ginn_v2.workflow import _load_domain_forward_inputs, load_config
from ginn_v2.physics import TimeDomainAdapter
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
    # PostM is the production-like default.  Keep the body-scale Gaussian
    # explicit in the benchmark configuration so the benchmark exercises the
    # same smooth-body semantics as the main workflow.
    body_smoothing_fwhm_s: float = 0.01
    waveform_qc_dynamic_window_s: float = 0.06
    seismic_support_relative_threshold: float = 0.25

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
        if not np.isfinite(self.body_smoothing_fwhm_s) or self.body_smoothing_fwhm_s <= 0:
            raise ValueError("body_smoothing_fwhm_s must be finite and positive.")
        if not np.isfinite(self.waveform_qc_dynamic_window_s) or self.waveform_qc_dynamic_window_s <= 0:
            raise ValueError("waveform_qc_dynamic_window_s must be finite and positive.")
        if not 0 < self.seismic_support_relative_threshold <= 1:
            raise ValueError("seismic_support_relative_threshold must lie in (0, 1].")
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
    # Keep both target semantics available to the evaluator: the original
    # fixed model curve and the body target after the configured physical
    # smoothing.  They are deliberately separate from the full truth file.
    from ginn_v2.model import BodySmoother
    training_roles = [role for role in roles if role["role"] == "train"]
    training_indices = np.asarray([role["profile_index"] for role in training_roles], dtype=int)
    smoother = BodySmoother(settings.body_smoothing_fwhm_s)
    support = np.ones(nt, dtype=bool)
    body_targets = np.stack([
        smoother.smooth_numpy(truth[index], time_model.twt_s, support)
        for index in training_indices
    ])
    np.savez_compressed(
        output_dir / "evaluation" / "well_targets.npz",
        profile_index=training_indices.astype(np.int64),
        fixed_log_ai=truth[training_indices].astype(np.float32),
        body_log_ai=body_targets.astype(np.float32),
        twt_s=time_model.twt_s.astype(np.float64),
        body_smoothing_fwhm_s=np.asarray(settings.body_smoothing_fwhm_s, dtype=np.float64),
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
        "ginn_v2_body_inversion": {
            "inputs": {"lfm_run_dir": relative(lfm_dir), "variant_id": "well_huber_trend", "well_control_run_dir": relative(controls_dir),
                       "evaluation_support_manifest": relative(controls_dir / "qc" / "evaluation_support.json"),
                       "wavelet_generation_run_dir": relative(output_dir / "step5_wavelet_generation" if wavelet_source["method"] == "workflow_steps_4_5_pending" else wavelet_dir)},
            "training": {"body_smoothing_fwhm_s": settings.body_smoothing_fwhm_s,
                "waveform_qc_dynamic_window_s": settings.waveform_qc_dynamic_window_s,
                "pretrain_epochs": 1, "finetune_epochs": 1, "patch_radius": 8, "batch_size": 8,
                "trusted_well_names": [r["well_name"] for r in roles if r["role"] == "train"],
                "orientations": ["inline"], "device": "cpu", "seed": settings.seed,
                "review_fraction": 0.1, "validation_gap_m": max(300.0, 17 * spacing), "validation_anchor": "maxmin",
                "selection_weights": {"well_rmse": 1.0, "amplitude_mapping": 0.5},
                "seismic_support_relative_threshold": settings.seismic_support_relative_threshold,
                "warnings": {"pretrain_masked_corr_improvement": 0.01, "pretrain_masked_shape_ratio": 0.99, "masked_corr_drop_tolerance": 0.01, "well_pooled_rmse_ratio_max": 1.0, "seismic_body_amplitude_spearman_max": 0.4},
            },
        },
        "ginn_v2_volume_inference": {"device": "cpu", "batch_size": 32, "orientations": ["inline"]},
    }
    config_path = output_dir / "ginn_v2.yaml"
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
        "config_path": relative(config_path), "training_schedule": "P1 then F1; explicit benchmark configs select the F schedule",
    }
    write_json(output_dir / "preparation_summary.json", summary)
    return config_path


def check_prepared_inputs(config_path: Path, *, repo_root: Path, report_dir: Path | None = None) -> dict[str, Any]:
    """Read standard inputs and check physics/gradients without optimizer updates."""
    raw = load_config(config_path)
    workflow = WorkflowConfig.from_mapping(raw)
    section = raw["ginn_v2_body_inversion"]
    inputs = section["inputs"]
    config = BodyInversionConfig.from_mapping(section["training"], sample_domain="time")
    torch.manual_seed(config.seed)
    controls = load_well_control_set(repo_root / inputs["well_control_run_dir"], repo_root=repo_root)
    lfm = load_lfm_input(inputs, repo_root=repo_root)
    data_root = repo_root / workflow.data_root
    report_dir = data_root if report_dir is None else Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    survey = open_survey(data_root / workflow.seismic.file, workflow.seismic.type)
    axis = survey.sample_axis("time")
    if not np.array_equal(axis.values, controls.sample_axis.values) or not np.array_equal(axis.values, lfm.sample_axis.values):
        raise ValueError("Seismic, wells and LFM must share the exact sample axis.")
    lowpass = parse_lowpass_spec(lfm.variant.variant_metadata["resolved_baseline_config"]["filter"], axis)
    times, wavelet, _relation, _payload = _load_domain_forward_inputs(repo_root / inputs["wavelet_generation_run_dir"], domain="time", depth_basis=None)
    adapter = TimeDomainAdapter(torch.as_tensor(times, dtype=torch.float32), torch.as_tensor(wavelet, dtype=torch.float32))
    support_mask = (
        seismic_profile_support(
            SurveyTraceSource(survey=survey, sample_axis=axis, geometry=survey.line_geometry),
            relative_threshold=config.seismic_support_relative_threshold,
        )
        if config.seismic_support_relative_threshold is not None else None
    )
    reader = PatchReader(
        SurveyTraceSource(survey=survey, sample_axis=axis, geometry=survey.line_geometry),
        lfm_log_ai=lfm.log_ai, lfm_valid_mask=lfm.valid_mask, ilines=lfm.ilines, xlines=lfm.xlines,
        sample_axis=axis, normalization=fit_lfm_normalization(lfm.log_ai, lfm.valid_mask, geometry=survey.line_geometry),
        patch_radius=config.patch_radius,
        seismic_support_mask=support_mask,
    )
    candidates = candidate_patch_keys(
        lfm.log_ai,
        lfm.valid_mask,
        patch_radius=config.patch_radius,
        orientations=config.orientations,
        seismic_support_mask=support_mask,
    )
    data = build_body_inversion_data(reader, controls, config=config, lfm_lowpass_spec=lowpass, candidate_keys=candidates, target_zone_mask=lfm.valid_mask)
    trainer = BodyInversionTrainer(data, adapter=adapter, config=config, lfm_lowpass_spec=lowpass, output_dir=data_root)
    batch = reader.batch(data.spatial_split.train_keys[:2], center_visible=False, device="cpu")
    model = CenterTraceBodyNet(config.network)
    body, synthetic, common, baseline = trainer._predict(model, batch)
    loss, _parts = trainer._loss(body, baseline, synthetic, common, weights=config.loss_weights)
    loss.backward()
    if not all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in model.parameters()):
        raise ValueError("Adapter gradient check failed.")
    gradient_norm = float(torch.sqrt(sum(torch.sum(p.grad.square()) for p in model.parameters())).item())
    if gradient_norm <= 0:
        raise ValueError("Adapter produced no learning signal.")
    with np.load(data_root / "evaluation" / "truth.npz", allow_pickle=False) as saved:
        truth = saved["log_ai"]
    forward_keys = (candidates[0], candidates[len(candidates) // 2], candidates[-1])
    forward_batch = reader.batch(forward_keys, center_visible=True, device="cpu")
    true_curves = torch.as_tensor(np.stack([truth[key.xline_index] for key in forward_keys]), dtype=torch.float32)
    reconstructed = adapter.close_body(true_curves, trainer._common(forward_batch)).synthetic_seismic
    numpy_forward = forward_time(true_curves.numpy().astype(float), times, wavelet, sample_step_s=float(axis.step))
    error = float(np.max(np.abs(reconstructed.detach().numpy() - numpy_forward)))
    if error > 1e-5:
        raise ValueError(f"NumPy/Torch forward closure differs by {error:g}.")
    result = {"status": "ok", "config_path": str(config_path.resolve()), "candidate_patches": len(candidates),
        "lfm_variant_id": inputs["variant_id"], "lfm_run_dir": inputs["lfm_run_dir"],
        "training_patches": len(data.spatial_split.train_keys), "validation_patches": len(data.spatial_split.validation_keys),
        "training_wells": [c.well_name for c in controls.controls], "tensor_shape": list(batch.features.shape),
        "initial_loss": float(loss.detach()), "gradients_finite": True, "gradient_norm": gradient_norm,
        "numpy_torch_forward_max_abs_error": error,
        "checked_forward_trace_indices": [key.xline_index for key in forward_keys],
        "optimizer_updates": 0, "single_orientation_disagreement": "not_applicable"}
    write_json(report_dir / "adapter_check.json", result)
    return result
