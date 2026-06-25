"""Screen LAS curves and export slim LAS files for usable wells.

This script consumes ``well_inventory.csv`` from the first workflow step. It
uses local mnemonic rules plus optional human overrides to classify and select
curves before exporting the fixed slim-LAS contract.

Usage::

    python scripts/well_screen.py
    python scripts/well_screen.py --config experiments/common/common.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.config.workflow import TimeWorkflowConfig
from cup.utils.coerce import as_bool
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.well.assets import build_file_lookup, normalize_well_name
from cup.well.curves import (
    CurveSelection,
    classify_curves_by_rules,
    exact_mnemonic,
    normalize_mnemonic as normalize_curve_mnemonic,
    select_primary_curves,
)
from cup.well.las import export_selected_curves_to_las, scan_las_curves
from cup.well.mnemonics import CURVE_CATEGORY_MNEMONICS, CURVE_CATEGORY_PRIORITY


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common/common.yaml"),
        help="Time-domain common config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/well_screen_<timestamp>.",
    )
    return parser.parse_args()


# =============================================================================
# Config
# =============================================================================


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("well_screen") or {})
    source_runs = dict(script_cfg.get("source_runs") or {})
    source_runs.setdefault("well_inventory_dir", None)
    script_cfg["source_runs"] = source_runs
    candidate_filter = dict(script_cfg.get("candidate_filter") or {})
    candidate_filter.setdefault("include_survey_positions", ["inside", "near_outside"])
    script_cfg["candidate_filter"] = candidate_filter
    classification = dict(script_cfg.get("classification") or {})
    classification.setdefault("curve_schema_file", None)
    classification.setdefault("curve_override_file", "experiments/common/curve_alias_overrides.yaml")
    script_cfg["classification"] = classification
    return script_cfg


def _resolve_data_path(value: str | Path, *, data_root: Path) -> Path:
    return resolve_relative_path(value, root=data_root)


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"well_screen_{timestamp}"


def _discover_latest_inventory_file(cfg: dict[str, Any], script_cfg: dict[str, Any]) -> Path:
    source_runs = dict(script_cfg.get("source_runs") or {})
    if source_runs.get("well_inventory_dir") is not None:
        return _resolve_repo_path(source_runs["well_inventory_dir"]) / "well_inventory.csv"
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    candidates = sorted(output_root.glob("well_inventory_*/well_inventory.csv"))
    if not candidates:
        raise FileNotFoundError(
            "well_screen source run is not configured and no "
            "well_inventory_*/well_inventory.csv file was found under output_root."
        )
    return candidates[-1]


def _load_curve_schema(schema_file: str | Path | None) -> Mapping[str, Sequence[str]]:
    if schema_file is None:
        return CURVE_CATEGORY_MNEMONICS
    path = _resolve_repo_path(schema_file)
    if not path.exists():
        return CURVE_CATEGORY_MNEMONICS
    data = load_yaml_config(path)
    categories = data.get("categories", data)
    schema: dict[str, Sequence[str]] = {}
    if not isinstance(categories, Mapping):
        raise ValueError(f"Invalid curve schema file: {path}")
    for category, spec in categories.items():
        if isinstance(spec, Mapping):
            mnemonics = spec.get("mnemonics", [])
        else:
            mnemonics = spec
        if not isinstance(mnemonics, Sequence) or isinstance(mnemonics, (str, bytes)):
            raise ValueError(f"Invalid mnemonic list for category {category!r} in {path}")
        schema[str(category)] = [str(item) for item in mnemonics]
    return schema


def _load_overrides(override_file: str | Path | None) -> dict[str, Any]:
    if override_file is None:
        return {}
    path = _resolve_repo_path(override_file)
    if not path.exists():
        return {}
    return load_yaml_config(path)


# =============================================================================
# Inventory filtering
# =============================================================================


def _candidate_inventory_rows(
    inventory_df: pd.DataFrame,
    *,
    include_survey_positions: Sequence[str],
) -> pd.DataFrame:
    required_columns = {"well_name", "has_well_head", "has_las", "survey_position", "inventory_status"}
    missing = sorted(required_columns - set(inventory_df.columns))
    if missing:
        raise ValueError(f"well_inventory.csv is missing required columns: {missing}")

    include_positions = {str(item) for item in include_survey_positions}
    mask = (
        inventory_df["has_well_head"].map(as_bool)
        & inventory_df["has_las"].map(as_bool)
        & inventory_df["survey_position"].astype(str).isin(include_positions)
        & (inventory_df["inventory_status"].astype(str) == "usable_for_las_screen")
    )
    return inventory_df.loc[mask].copy()


def _well_screen_row(selection: CurveSelection) -> dict[str, Any]:
    exported_las = selection.exported_las or ""
    return {
        "well_name": selection.well_name,
        "las_file": selection.las_file,
        "screen_status": selection.screen_status,
        "has_p_sonic": selection.has_category("p_sonic"),
        "has_density": selection.has_category("density"),
        "has_caliper": selection.has_category("caliper"),
        "primary_p_sonic": selection.primary("p_sonic") or "",
        "primary_density": selection.primary("density") or "",
        "primary_caliper": selection.primary("caliper") or "",
        "selected_curve_count": len(selection.selected_mnemonics),
        "exported_las": exported_las,
        "reasons": ";".join(selection.reasons),
    }


def _write_classification_json(output_file: Path, *, header: Any, selection: CurveSelection) -> None:
    payload = {
        "well_name": selection.well_name,
        "las_file": selection.las_file,
        "header": header.to_row(),
        "screen_status": selection.screen_status,
        "primary_by_category": selection.primary_by_category,
        "selected_mnemonics": selection.selected_mnemonics,
        "reasons": selection.reasons,
        "curves": [
            {
                "mnemonic": item.mnemonic,
                "unit": item.unit,
                "description": item.description,
                "index": item.index,
                "category": item.category,
                "is_primary": item.is_primary,
                "classification_source": item.classification_source,
                "confidence": item.confidence,
                "notes": item.notes,
            }
            for item in selection.classifications
        ],
    }
    write_json(output_file, payload)


def _empty_skipped_curves_columns() -> list[str]:
    return ["well_name", "mnemonic", "category", "reason"]


def _exported_contains_required(
    *,
    selection: CurveSelection,
    exported_mnemonics: Sequence[str],
    required_categories: Sequence[str],
) -> tuple[bool, list[str]]:
    exported_exact = {exact_mnemonic(mnemonic) for mnemonic in exported_mnemonics}
    exported_base = {normalize_curve_mnemonic(mnemonic) for mnemonic in exported_mnemonics}
    missing: list[str] = []
    for category in required_categories:
        primary = selection.primary(category)
        if primary is None:
            continue
        if exact_mnemonic(primary) in exported_exact:
            continue
        if normalize_curve_mnemonic(primary) in exported_base:
            continue
        missing.append(category)
    return not missing, missing


def run_screening(
    *,
    inventory_file: Path,
    las_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    schema: Mapping[str, Sequence[str]],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    inventory_df = pd.read_csv(inventory_file)
    candidates = _candidate_inventory_rows(
        inventory_df,
        include_survey_positions=config["candidate_filter"]["include_survey_positions"],
    )
    las_lookup = build_file_lookup(las_dir.glob("*.las"), asset_label=str(las_dir))

    selected_las_dir = output_dir / "selected_las"
    classification_dir = output_dir / "curve_classification"
    selected_las_dir.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    curve_inventory_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    skipped_well_rows: list[dict[str, Any]] = []
    skipped_curve_rows: list[dict[str, Any]] = []
    classification_source_counts: dict[str, int] = {}

    for _, inv_row in candidates.sort_values("well_name").iterrows():
        well_name = str(inv_row["well_name"])
        well_key = normalize_well_name(well_name)
        las_file = las_lookup.get(well_key)
        if las_file is None:
            well_rows.append(
                {
                    "well_name": well_name,
                    "las_file": "",
                    "screen_status": "failed",
                    "has_p_sonic": False,
                    "has_density": False,
                    "has_caliper": False,
                    "primary_p_sonic": "",
                    "primary_density": "",
                    "primary_caliper": "",
                    "selected_curve_count": 0,
                    "exported_las": "",
                    "reasons": "las_file_missing",
                }
            )
            skipped_well_rows.append({"well_name": well_name, "reason": "las_file_missing"})
            continue

        try:
            header, curves = scan_las_curves(las_file)
            classifications = classify_curves_by_rules(
                curves,
                schema=schema,
                well_name=well_name,
                overrides=overrides,
            )
            selection = select_primary_curves(
                classifications,
                well_name=well_name,
                las_file=repo_relative_path(las_file, root=REPO_ROOT),
                selected_categories=config["curve_selection"]["selected_categories"],
                required_categories=config["curve_selection"]["required_categories"],
                overrides=overrides,
                category_priority=CURVE_CATEGORY_PRIORITY,
            )

            if selection.screen_status == "passed":
                output_las = selected_las_dir / f"{sanitize_filename(well_name)}.las"
                _, skipped, exported_mnemonics = export_selected_curves_to_las(
                    las_file,
                    output_las,
                    selection.selected_mnemonics,
                )
                for item in skipped:
                    skipped_curve_rows.append(
                        {
                            "well_name": well_name,
                            "mnemonic": item.get("curve", ""),
                            "category": "",
                            "reason": item.get("reason", "export_failed"),
                        }
                    )
                required_exported, missing_exported_categories = _exported_contains_required(
                    selection=selection,
                    exported_mnemonics=exported_mnemonics,
                    required_categories=config["curve_selection"]["required_categories"],
                )
                if required_exported:
                    selection.exported_las = repo_relative_path(output_las, root=REPO_ROOT)
                else:
                    selection.screen_status = "failed"
                    selection.exported_las = None
                    for category in missing_exported_categories:
                        reason = f"export_missing_required_{category}"
                        if reason not in selection.reasons:
                            selection.reasons.append(reason)
                    skipped_well_rows.append({"well_name": well_name, "reason": ";".join(selection.reasons)})
            else:
                skipped_well_rows.append({"well_name": well_name, "reason": ";".join(selection.reasons)})

            for item in selection.classifications:
                curve_inventory_rows.append(item.to_inventory_row(well_name=well_name))
                classification_source_counts[item.classification_source] = (
                    classification_source_counts.get(item.classification_source, 0) + 1
                )
                if item.category == "ambiguous" or item.disabled:
                    skipped_curve_rows.append(
                        {
                            "well_name": well_name,
                            "mnemonic": item.mnemonic,
                            "category": item.category,
                            "reason": item.notes or item.category,
                        }
                    )

            _write_classification_json(
                classification_dir / f"{sanitize_filename(well_name)}.json",
                header=header,
                selection=selection,
            )
            well_rows.append(_well_screen_row(selection))

        except Exception as exc:
            reason = str(exc)
            well_rows.append(
                {
                    "well_name": well_name,
                    "las_file": repo_relative_path(las_file, root=REPO_ROOT),
                    "screen_status": "failed",
                    "has_p_sonic": False,
                    "has_density": False,
                    "has_caliper": False,
                    "primary_p_sonic": "",
                    "primary_density": "",
                    "primary_caliper": "",
                    "selected_curve_count": 0,
                    "exported_las": "",
                    "reasons": reason,
                }
            )
            skipped_well_rows.append({"well_name": well_name, "reason": reason})

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "las_curve_inventory_csv": output_dir / "las_curve_inventory.csv",
        "well_screen_csv": output_dir / "well_screen.csv",
        "skipped_wells_csv": output_dir / "skipped_wells.csv",
        "skipped_curves_csv": output_dir / "skipped_curves.csv",
        "run_summary_json": output_dir / "run_summary.json",
    }
    pd.DataFrame.from_records(curve_inventory_rows).to_csv(paths["las_curve_inventory_csv"], index=False, encoding="utf-8")
    pd.DataFrame.from_records(well_rows).to_csv(paths["well_screen_csv"], index=False, encoding="utf-8")
    pd.DataFrame.from_records(skipped_well_rows, columns=["well_name", "reason"]).to_csv(
        paths["skipped_wells_csv"],
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame.from_records(skipped_curve_rows, columns=_empty_skipped_curves_columns()).to_csv(
        paths["skipped_curves_csv"],
        index=False,
        encoding="utf-8",
    )

    well_screen_df = pd.DataFrame.from_records(well_rows)
    status_counts = (
        well_screen_df["screen_status"].value_counts(dropna=False).astype(int).to_dict()
        if not well_screen_df.empty
        else {}
    )
    summary = {
        "script": "well_screen.py",
        "inputs": {
            "inventory_file": repo_relative_path(inventory_file, root=REPO_ROOT),
            "las_dir": repo_relative_path(las_dir, root=REPO_ROOT),
        },
        "candidate_filter": {
            "has_well_head": True,
            "has_las": True,
            "survey_position": list(config["candidate_filter"]["include_survey_positions"]),
            "inventory_status": "usable_for_las_screen",
        },
        "required_categories": list(config["curve_selection"]["required_categories"]),
        "selected_categories": list(config["curve_selection"]["selected_categories"]),
        "candidate_well_count": int(len(candidates)),
        "screen_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "exported_las_count": int((well_screen_df["exported_las"].astype(str) != "").sum()) if not well_screen_df.empty else 0,
        "classification_source_counts": classification_source_counts,
        "has_caliper_count": int(well_screen_df["has_caliper"].sum()) if not well_screen_df.empty else 0,
        "paths": {key: repo_relative_path(path, root=REPO_ROOT) for key, path in paths.items()},
    }
    write_json(paths["run_summary_json"], summary)
    return {"paths": paths, "summary": summary}


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    workflow = TimeWorkflowConfig.from_mapping(cfg)
    script_cfg = _script_config(cfg)
    script_cfg["curve_selection"] = {
        "required_categories": list(workflow.well_curves.required_categories),
        "selected_categories": list(workflow.well_curves.selected_categories),
    }

    data_root = _resolve_repo_path(workflow.data_root)
    inventory_file = _discover_latest_inventory_file(cfg, script_cfg)
    las_dir = _resolve_data_path(workflow.assets.las_dir, data_root=data_root)
    output_dir = _resolve_output_dir(args, cfg)
    schema = _load_curve_schema(script_cfg["classification"].get("curve_schema_file"))
    overrides = _load_overrides(script_cfg["classification"].get("curve_override_file"))

    result = run_screening(
        inventory_file=inventory_file,
        las_dir=las_dir,
        output_dir=output_dir,
        config=script_cfg,
        schema=schema,
        overrides=overrides,
    )
    summary = result["summary"]
    print(f"Saved LAS curve inventory: {result['paths']['las_curve_inventory_csv']}")
    print(f"Saved well screen: {result['paths']['well_screen_csv']}")
    print(f"Saved skipped wells: {result['paths']['skipped_wells_csv']}")
    print(f"Saved skipped curves: {result['paths']['skipped_curves_csv']}")
    print(f"Saved run summary: {result['paths']['run_summary_json']}")
    print(
        "LAS curve screen summary: "
        f"{summary['candidate_well_count']} candidates, "
        f"{summary['screen_status_counts'].get('passed', 0)} passed, "
        f"{summary['screen_status_counts'].get('partial', 0)} partial, "
        f"{summary['screen_status_counts'].get('failed', 0)} failed, "
        f"{summary['exported_las_count']} LAS exported."
    )


if __name__ == "__main__":
    main()
