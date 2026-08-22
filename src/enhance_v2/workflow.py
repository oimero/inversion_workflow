"""Workflow adapters shared by Enhance V2 prototype and volume entrypoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from cup.well.real_field_controls import WellControl, WellControlSet


def select_controls(controls: WellControlSet, names: tuple[str, ...]) -> WellControlSet:
    """Select the frozen residual-dictionary controls in caller order."""

    by_name = {item.well_name.casefold(): item for item in controls.controls}
    selected: list[WellControl] = []
    for name in names:
        control = by_name.get(name.casefold())
        if control is None:
            raise ValueError(f"Trusted GINN well is absent from WellControlSet: {name}")
        selected.append(control)
    return WellControlSet(
        sample_axis=controls.sample_axis,
        controls=tuple(selected),
        sample_domain=controls.sample_domain,
        sample_unit=controls.sample_unit,
        depth_basis=controls.depth_basis,
        source_run_type=controls.source_run_type,
        provenance=controls.provenance,
    )


def well_zone_intervals(
    controls: WellControlSet,
    target_zone: Any,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Resolve interpreted zone intervals along each selected well path."""

    intervals_by_well: dict[str, dict[str, tuple[float, float]]] = {}
    for control in controls.controls:
        axis = np.asarray(control.sample_axis.values, dtype=np.float64)
        position_valid = np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
        markers: list[tuple[str, float]] = []
        for horizon_name in target_zone.horizon_names:
            local_horizon = np.full(axis.shape, np.nan, dtype=np.float64)
            for index in np.flatnonzero(position_valid):
                local_horizon[index] = target_zone.get_horizon_interpretation_at_location(
                    horizon_name,
                    float(control.inline_by_sample[index]),
                    float(control.xline_by_sample[index]),
                )
            candidates = np.flatnonzero(np.isfinite(local_horizon))
            if candidates.size == 0:
                raise ValueError(f"{control.well_name}: horizon {horizon_name!r} has no well-path support.")
            selected = int(candidates[np.argmin(np.abs(axis[candidates] - local_horizon[candidates]))])
            markers.append((horizon_name, float(axis[selected])))
        if any(markers[index + 1][1] <= markers[index][1] for index in range(len(markers) - 1)):
            raise ValueError(f"{control.well_name}: target horizons are not ordered along the well path.")
        intervals_by_well[control.well_name] = {
            f"{top_name}__to__{bottom_name}": (top, bottom)
            for (top_name, top), (bottom_name, bottom) in zip(markers[:-1], markers[1:])
        }
    return intervals_by_well


__all__ = ["select_controls", "well_zone_intervals"]
