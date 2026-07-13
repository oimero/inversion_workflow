"""Time-domain Synthoseis-lite generation modules."""

from cup.synthetic.time.config import parse_synthoseis_config, resolve_sources
from cup.synthetic.time.pipeline import generation_scenarios, run_generation

__all__ = ["generation_scenarios", "parse_synthoseis_config", "resolve_sources", "run_generation"]
