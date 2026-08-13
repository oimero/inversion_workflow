"""唯一的真实工区 Enhance V2 训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import write_json
from enhance.config import EnhanceV2Config
from enhance.real_field import build_residual_library, load_runtime, write_residual_atlas
from enhance.stage2_trainer import EnhanceV2Trainer


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", nargs="?", default="stage2")
    parser.add_argument("--config", type=Path, default=Path("experiments/enhance_v2/enhance_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def _write_review_package(trainer: EnhanceV2Trainer, checkpoint: Path, review_dir: Path) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
    trainer.model.eval()
    well_dir = review_dir / "wells"
    well_dir.mkdir(parents=True, exist_ok=True)
    controls = {item.well_name: item for item in trainer.runtime.controls.controls}
    well_files: list[str] = []
    for record in trainer.library.records:
        control = controls[record.well_name]
        valid = np.flatnonzero(record.model_mask)
        if valid.size < 2:
            continue
        i_float, j_float = trainer.runtime.reader.geometry.line_to_index(
            float(control.inline_by_sample[valid[0]]), float(control.xline_by_sample[valid[0]])
        )
        body, predicted_residual, support = trainer.predict_trace(int(round(i_float)), int(round(j_float)))
        enhanced = body + predicted_residual
        axis = trainer.runtime.reader.sample_axis.values
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
        mask = support & np.isfinite(record.model_reference)
        axes[0].plot(record.model_reference[mask], axis[mask], color="black", label="filtered full")
        axes[0].plot(body[mask], axis[mask], color="tab:blue", label="GINN body")
        axes[0].plot(enhanced[mask], axis[mask], color="tab:red", label="enhanced")
        axes[0].legend(fontsize=8)
        axes[0].set_title(record.well_name)
        axes[1].plot(record.model_residual[mask], axis[mask], color="black", label="well residual")
        axes[1].plot(predicted_residual[mask], axis[mask], color="tab:red", label="predicted residual")
        axes[1].legend(fontsize=8)
        axes[1].set_title("residual anchor")
        axes[2].plot(body[mask], axis[mask], color="tab:blue", label="body")
        axes[2].plot(enhanced[mask], axis[mask], color="tab:red", label="enhanced")
        axes[2].set_title("body / enhanced")
        for current in axes:
            current.invert_yaxis()
            current.grid(alpha=0.2)
        fig.tight_layout()
        path = well_dir / f"{record.well_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        well_files.append(str(path))
    return {"well_figures": well_files, "checkpoint": str(checkpoint)}


def main() -> None:
    args = _args()
    output_root = (REPO_ROOT / "experiments/enhance_v2/results" / args.run_id).resolve()
    if args.output_dir is not None:
        output_root = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Enhance V2 output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config = EnhanceV2Config.from_yaml(args.config, repo_root=REPO_ROOT, result_dir=output_root)
    runtime = load_runtime(config, repo_root=REPO_ROOT)
    library = build_residual_library(runtime)
    library.save(output_root / "residual_library.npz")
    write_json(output_root / "residual_library_summary.json", library.metadata)
    atlas_paths = write_residual_atlas(library, output_root / "review_package" / "residual_atlas")
    trainer = EnhanceV2Trainer(config, runtime, library, output_root)
    smoke = trainer.smoke()
    write_json(output_root / "smoke.json", smoke)
    if args.smoke_only:
        write_json(output_root / "enhance_status.json", {"status": "smoke_passed", "smoke": smoke, "atlas": atlas_paths})
        return
    formal_result = trainer.train()
    run_items: list[tuple[str, EnhanceV2Trainer, dict[str, object]]] = [("formal", trainer, formal_result)]
    if formal_result["status"] != "accepted" and config.maximum_adjustment_runs > 0:
        diagnostic = formal_result.get("best_diagnostic_checkpoint")
        if diagnostic:
            adjustment_dir = output_root / "adjustment_01"
            adjusted_trainer = EnhanceV2Trainer(
                config,
                runtime,
                library,
                adjustment_dir,
                rms_weight_multiplier=config.adjustment_rms_weight_multiplier,
            )
            adjustment_result = adjusted_trainer.train(
                epochs=config.adjustment_epochs,
                start_checkpoint=Path(diagnostic),
            )
            run_items.append(("adjustment_01", adjusted_trainer, adjustment_result))
    accepted_items = [item for item in run_items if item[2]["status"] == "accepted"]
    if accepted_items:
        _, trainer, result = accepted_items[0]
    else:
        def monitor_score(item: tuple[str, EnhanceV2Trainer, dict[str, object]]) -> float:
            epochs = item[2].get("epochs") or []
            values = [float(epoch["monitor_rmse"]) for epoch in epochs]
            return min(values) if values else float("inf")

        _, trainer, result = min(run_items, key=monitor_score)
    runs = [{"name": name, **run_result} for name, _, run_result in run_items]
    selected = result.get("selected_checkpoint") or result.get("best_diagnostic_checkpoint")
    if selected:
        selected_target = output_root / ("selected_checkpoint.pt" if result["status"] == "accepted" else "best_diagnostic_checkpoint.pt")
        if Path(selected).resolve() != selected_target.resolve():
            shutil.copyfile(selected, selected_target)
        selected = selected_target
    review = {}
    if selected:
        review = _write_review_package(trainer, Path(selected), output_root / "review_package")
    result["review_package"] = review
    result["runs"] = runs
    result["residual_atlas"] = atlas_paths
    result["config"] = config.to_json_dict()
    result["runtime"] = {
        "device": str(trainer.device),
        "sample_domain": runtime.lfm.sample_axis.domain,
        "sample_unit": runtime.lfm.sample_axis.unit,
        "well_names": list(library.well_names),
    }
    write_json(output_root / "enhance_status.json", result)
    write_json(output_root / "resolved_config.json", config.to_json_dict())
    print(json.dumps({"status": result["status"], "output": str(output_root), "selected": str(selected) if selected else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
