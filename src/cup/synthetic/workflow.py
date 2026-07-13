"""Primary time-domain Synthoseis-lite calibration and generation workflow.

The depth-domain extension is dispatched explicitly by the command-line
entrypoint and is not required to mirror this workflow module.
"""

from __future__ import annotations

from cup.synthetic.core.calibration import load_calibration
from cup.synthetic.time.calibration_pipeline import build_calibration_inputs, run_calibration
from cup.synthetic.time.config import parse_synthoseis_config, resolve_sources
from cup.synthetic.time.pipeline import generation_scenarios, run_generation
from cup.synthetic.time.geometry import SectionGeometry, build_section_geometries

__all__ = [
    "SectionGeometry",
    "build_calibration_inputs",
    "build_section_geometries",
    "generation_scenarios",
    "load_calibration",
    "parse_synthoseis_config",
    "resolve_sources",
    "run_calibration",
    "run_generation",
]
