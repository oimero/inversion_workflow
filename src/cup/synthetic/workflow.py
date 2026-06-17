"""Facade for synthoseis-lite calibration and generation workflows."""

from __future__ import annotations

from cup.synthetic.calibration_pipeline import build_calibration_inputs, run_calibration
from cup.synthetic.config import parse_synthoseis_config
from cup.synthetic.generation_pipeline import generation_scenarios, run_generation
from cup.synthetic.geometry import SectionGeometry, build_section_geometries
from cup.synthetic.sources import load_calibration, resolve_sources

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
