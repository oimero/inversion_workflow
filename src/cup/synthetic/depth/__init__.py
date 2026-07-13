"""Depth-domain Synthoseis-lite implementation."""

from cup.synthetic.depth.config import (
    CALIBRATION_SCHEMA,
    GENERATOR_FAMILY,
    SCHEMA_VERSION,
    load_composed_config,
    parse_depth_v2_config,
    resolve_depth_v2_sources,
)
from cup.synthetic.depth.calibration import run_depth_calibration
from cup.synthetic.depth.generation import run_depth_generation
from cup.synthetic.depth.model import DepthGeneratedSection, DepthSectionGeometry

__all__ = [
    "CALIBRATION_SCHEMA",
    "GENERATOR_FAMILY",
    "SCHEMA_VERSION",
    "load_composed_config",
    "parse_depth_v2_config",
    "resolve_depth_v2_sources",
    "run_depth_calibration",
    "run_depth_generation",
    "DepthGeneratedSection",
    "DepthSectionGeometry",
]
