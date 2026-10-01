"""Run GINN with a trainable initial model and one final-curve smoothing."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from cup.config.workflow import WorkflowConfig
from cup.lfm.math import parse_lowpass_spec
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import write_json
from cup.utils.logging import configure_run_logger
from cup.utils.masks import true_runs
from cup.well.controls import load_well_control_set
from ginn_v2.data import (
    ArrayTraceSource, InputNormalization, PatchKey, PatchReader,
    candidate_patch_keys, fit_lfm_normalization,
)
from ginn_v2.model import CenterTraceBodyNet
from ginn_v2.infer import BodyInverter
from ginn_v2.physics import DepthDomainAdapter
from ginn_v2.train import BodyInversionConfig, BodyInversionTrainer, build_body_inversion_data, load_checkpoint
from ginn_v2.workflow import REPO_ROOT, load_config, load_forward_inputs

def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path

def _settings(config_path: Path):
    raw = load_config(config_path)
    return raw, WorkflowConfig.from_mapping(raw), BodyInversionConfig.from_mapping(
        raw["ginn_v2_body_inversion"]["training"]
    ), raw["final_curve_experiment"]

def _survey(workflow):
    return open_survey(
        _path(workflow.data_root) / workflow.seismic.file, workflow.seismic.type,
        segy_options=segy_options_from_config(workflow.seismic.as_dict()),
    )

def _comparison_arrays(trainer, model, controls, output_dir, settings):
    reader = trainer.data.reader
    inverter = BodyInverter(model, reader, trainer.adapter, smoother=trainer.smoother,
                            device=trainer.device, batch_size=trainer.config.batch_size)
    samples = reader.sample_axis.values
    names = trainer.config.trusted_well_names
    shape = (len(names), len(samples))
    prediction = np.zeros(shape)
    count = np.zeros(shape, dtype=int)
    prior = np.zeros(shape)
    initial_output = np.zeros(shape)
    reference = np.full(shape, np.nan)
    target = np.full(shape, np.nan)
    lookup = {c.well_name: c for c in controls.controls}
    for row, name in enumerate(names):
        well = lookup[name]
        info = trainer.data.well_targets[name]
        # The smoothed native target BEFORE the prior-dependent band projection.
        native = info.native_body_target
        coordinates = well.native.coordinates
        for start, stop in true_runs(np.isfinite(native)):
            inside = (samples >= coordinates[start]) & (samples <= coordinates[stop-1])
            reference[row, inside] = np.interp(samples[inside], coordinates[start:stop], native[start:stop])
        reference[row, ~info.valid_target_mask] = np.nan
        target[row] = info.model_axis_target
    items = trainer.data.trusted_well_patches
    with torch.no_grad():
        for start in range(0, len(items), trainer.config.batch_size):
            local = items[start:start+trainer.config.batch_size]
            batch = reader.batch(tuple(item.patch_key for item in local), center_visible=True, device=trainer.device)
            body, _, _ = trainer._predict(model, batch)
            inferred = inverter.predict_body(tuple(item.patch_key for item in local))
            support = inferred.valid_mask
            if not torch.allclose(body[support], inferred.body_log_ai[support], rtol=1e-5, atol=1e-6):
                raise ValueError("Training and inference predictions disagree.")
            initial, _, _ = trainer._predict(None, batch)
            for row, item in enumerate(local):
                dest = names.index(item.well_name)
                m = item.target_mask
                prediction[dest, m] += body[row].cpu().numpy()[m]
                prior[dest, m] += batch.lfm_log_ai[row].cpu().numpy()[m]
                initial_output[dest, m] += initial[row].cpu().numpy()[m]
                count[dest, m] += 1
    prediction = np.divide(prediction, count, out=np.full(shape, np.nan), where=count > 0)
    prior = np.divide(prior, count, out=np.full(shape, np.nan), where=count > 0)
    initial_output = np.divide(initial_output, count, out=np.full(shape, np.nan), where=count > 0)
    axes = np.load(_path(settings["prepared_inputs_dir"]) / "axes.npz")
    inline = int(settings["section_inline"])
    i = int(np.flatnonzero(axes["ilines"] == inline)[0])
    # Use both orientation predictions, as in the volume workflow.
    radius = trainer.config.patch_radius
    js = [j for j in range(radius, len(axes["xlines"])-radius)
          if np.count_nonzero(reader.lfm_valid_mask[i,j]) >= 8]
    keys = tuple(PatchKey(i,j,o) for j in js for o in trainer.config.orientations)
    section_prediction = np.full((len(js),len(samples)), np.nan)
    section_prior = section_prediction.copy()
    section_start = section_prediction.copy()
    section_seismic = section_prediction.copy()
    evaluated = inverter.predict_body(keys, center_visible=True)
    bodies = evaluated.body_log_ai.cpu().numpy()
    supports = evaluated.valid_mask.cpu().numpy()
    key_row = {key: row for row, key in enumerate(keys)}
    for row, j in enumerate(js):
        local = [PatchKey(i,j,o) for o in trainer.config.orientations]
        m = np.logical_and.reduce([supports[key_row[k]] for k in local])
        section_prediction[row,m] = np.mean([bodies[key_row[k]] for k in local],axis=0)[m]
        section_prior[row,m] = reader.lfm_log_ai[i,j,m]
        section_seismic[row] = reader.source.read_traces(((i,j),))[(i,j)]
    values = np.where(np.isfinite(section_prior), section_prior, 0.0)
    with torch.no_grad():
        for start in range(0, len(js), trainer.config.batch_size):
            stop = start + trainer.config.batch_size
            m = np.isfinite(section_prior[start:stop])
            smoothed = trainer.smoother.smooth(
                torch.as_tensor(values[start:stop],device=trainer.device,dtype=torch.float32),
                torch.as_tensor(samples,device=trainer.device,dtype=torch.float32),
                torch.as_tensor(m,device=trainer.device))
            section_start[start:stop] = np.where(m,smoothed.cpu().numpy(),np.nan)
    result = dict(samples=samples, well_names=np.asarray(names), well_reference_log_ai=reference,
                  well_target_log_ai=target, well_prediction_log_ai=prediction, well_prior_log_ai=prior,
                  section_prediction_log_ai=section_prediction, section_prior_log_ai=section_prior,
                  section_seismic=section_seismic, section_line_numbers=axes["xlines"][js], section_inline=inline,
                  well_start_log_ai=initial_output, section_start_log_ai=section_start)
    np.savez_compressed(output_dir / "comparison_traces.npz", **result)

def run_experiment(config_path: Path, output_dir: Path, *, review_only: bool = False) -> None:
    raw, workflow, config, settings = _settings(config_path)
    torch.set_num_threads(8)
    name = "final_curve_proportional"
    folder = output_dir
    folder.mkdir(parents=True, exist_ok=review_only)
    logger = configure_run_logger(folder, logger_name="final_curve_experiment", file_name="training.log")
    logger.info("Final-curve experiment %s", name)
    write_json(folder / "resolved_config.json", raw)
    cache = _path(settings["prepared_inputs_dir"])
    survey = _survey(workflow)
    axis = survey.sample_axis(workflow.seismic.domain)
    controls = load_well_control_set(_path(settings["well_control_run_dir"]), repo_root=REPO_ROOT)
    if not np.array_equal(controls.sample_axis.values, axis.values):
        raise ValueError("Well sample axis differs from seismic.")
    volume = np.load(cache / "proportional.npy", mmap_mode="r")
    mask = np.load(cache / "proportional_mask.npy", mmap_mode="r")
    velocity = np.load(cache / "velocity.npy", mmap_mode="r")
    axes = np.load(cache / "axes.npz")
    with (cache / "normalization.json").open() as handle:
        normalization = InputNormalization(**json.load(handle))
    with (cache / "filter.json").open() as handle:
        lowpass = parse_lowpass_spec(json.load(handle), axis)
    # This cache is the unchanged original seismic, prepared in the earlier full run.
    seismic = np.load(_path(settings["seismic_npy"]), mmap_mode="r")
    with np.load(_path(settings["seismic_axes_npz"])) as source_axes:
        if any(not np.array_equal(axes[k],source_axes[k]) for k in ("ilines","xlines","samples")):
            raise ValueError("Cached seismic axes differ from LFM axes.")
    source = ArrayTraceSource(seismic, axis, survey.line_geometry)
    reader = PatchReader(
        source, lfm_log_ai=volume, lfm_valid_mask=mask,
        ilines=axes["ilines"], xlines=axes["xlines"], sample_axis=axis,
        normalization=normalization, patch_radius=config.patch_radius,
        domain_extras={"velocity_mps":velocity}, cache_size=config.cache_size,
        seismic_feature_mode=config.seismic_feature_mode,
        seismic_balance_window_samples=config.seismic_balance_window_samples,
        seismic_balance_floor_fraction=config.seismic_balance_floor_fraction,
    )
    times, wavelet, _, _ = load_forward_inputs(_path(settings["forward_model_inputs"]), domain="depth",depth_basis="tvdss")
    adapter = DepthDomainAdapter(torch.as_tensor(times,dtype=torch.float32),torch.as_tensor(wavelet,dtype=torch.float32))
    data = build_body_inversion_data(reader, controls, config=config, lfm_lowpass_spec=lowpass,
        candidate_keys=candidate_patch_keys(volume,mask,patch_radius=config.patch_radius,orientations=config.orientations),
        target_zone_mask=mask)
    write_json(folder / "split.json", data.split_description())
    trainer = BodyInversionTrainer(data,adapter=adapter,config=config,lfm_lowpass_spec=lowpass,
                                  output_dir=folder,artifact_root=REPO_ROOT,logger=logger)
    logger.info("Data ready; baseline evaluation")
    baseline = trainer.evaluate(None)
    write_json(folder / "baseline_metrics.json", baseline.to_json_dict())
    if review_only:
        with (folder / "finetune_result.json").open() as handle:
            result = json.load(handle)
        payload = torch.load(folder / "pretraining" / "shared_checkpoint.pt",map_location="cpu",weights_only=False)
        pretrain_metrics = payload["metrics"]
    else:
        checkpoint, pretrain = trainer.run_pretraining()
        selected = trainer.run_finetuning(baseline=baseline,resume_checkpoint=checkpoint)
        shutil.copyfile(selected.selected_checkpoint,folder / "selected_checkpoint.pt")
        result = selected.to_json_dict()
        pretrain_metrics = pretrain.to_json_dict()
        write_json(folder / "finetune_result.json", result)
    model = CenterTraceBodyNet(config.network).to(trainer.device)
    load_checkpoint(folder / "selected_checkpoint.pt",model=model,expected_network_config=config.network,map_location=trainer.device)
    model.eval()
    logger.info("Writing common-reference wells and fixed inline section")
    _comparison_arrays(trainer,model,controls,folder,settings)
    write_json(folder / "experiment_summary.json", {
        "name":name,"selected_epoch":result["selected_epoch"],"baseline_metrics":baseline.to_json_dict(),
        "pretrain_metrics":pretrain_metrics,"selected_metrics":result["metrics"],
        "config":config.to_json_dict(),"fixed_normalization":asdict(normalization),
        "forward_velocity_source":"proportional", "training_target":"resample native filtered logAI then Gaussian smooth once",
        "comparison_reference":"common native 25 m smoothed filtered well for before-after comparison",
        "seismic_comparison":"waveform shape only; no amplitude loss, gain or vertical compensation",
        "prediction_definition":"Gaussian25(initial_log_ai + raw_network_correction)",
        "low_frequency_constraint":None,
        "source_inputs":settings,
        "checkpoint_selection":result["stop_reason"],
    })
    logger.info("Trial complete: %s", folder)
    gc.collect()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('scripts/ginn_v2_final_curve.yaml'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--review-only',action='store_true')
    args=parser.parse_args()
    run_experiment(_path(str(args.config)),_path(str(args.output_dir)),review_only=args.review_only)

if __name__=='__main__':
    main()
