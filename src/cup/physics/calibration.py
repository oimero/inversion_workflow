"""Domain-independent frozen AI--Vp relation values.

Robust fitting lives in :mod:`cup.physics.rock_physics`; workflow discovery
and artifact writing live in ``scripts/rock_physics_analysis.py``.  This module
defines the strict relation value consumed by forward callers without file
discovery or unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from cup.physics.numpy_backend import ai_from_velocity as _ai_from_velocity
from cup.physics.numpy_backend import velocity_from_ai as _velocity_from_ai


AI_UNIT = "m/s*g/cm3"
VP_UNIT = "m/s"
A_UNIT = "g/cm3"
B_UNIT = AI_UNIT
FORMULA = "AI = a * Vp + b"


@dataclass(frozen=True)
class AIVelocityRelation:
    """One frozen, unit-explicit linear acoustic-impedance relation."""

    a: float
    b: float
    ai_unit: str = AI_UNIT
    vp_unit: str = VP_UNIT
    a_unit: str = A_UNIT
    b_unit: str = B_UNIT

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.a)) or float(self.a) <= 0.0:
            raise ValueError("AI--Vp coefficient a must be finite and positive.")
        if not math.isfinite(float(self.b)):
            raise ValueError("AI--Vp coefficient b must be finite.")
        expected = {
            "ai_unit": AI_UNIT,
            "vp_unit": VP_UNIT,
            "a_unit": A_UNIT,
            "b_unit": B_UNIT,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise ValueError(
                    f"{field_name} must be {expected_value!r}, got {actual!r}."
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AIVelocityRelation":
        """Read the relation from an explicit mapping without schema guessing."""
        required = {"formula", "a", "b", "ai_unit", "vp_unit", "a_unit", "b_unit"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(f"AI--Vp relation is missing required fields: {missing}.")
        if payload["formula"] != FORMULA:
            raise ValueError(
                f"AI--Vp formula must be {FORMULA!r}, got {payload['formula']!r}."
            )
        return cls(
            a=float(payload["a"]),
            b=float(payload["b"]),
            ai_unit=str(payload["ai_unit"]),
            vp_unit=str(payload["vp_unit"]),
            a_unit=str(payload["a_unit"]),
            b_unit=str(payload["b_unit"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical unit-explicit serialized relation."""
        return {
            "formula": FORMULA,
            "a": float(self.a),
            "b": float(self.b),
            "ai_unit": self.ai_unit,
            "vp_unit": self.vp_unit,
            "a_unit": self.a_unit,
            "b_unit": self.b_unit,
        }

    def ai_from_velocity(self, velocity_mps: Any):
        """Apply this relation through the strict NumPy backend."""
        return _ai_from_velocity(velocity_mps, a=self.a, b=self.b)

    def velocity_from_ai(self, ai: Any):
        """Apply the inverse relation through the strict NumPy backend."""
        return _velocity_from_ai(ai, a=self.a, b=self.b)


__all__ = [
    "AI_UNIT",
    "AIVelocityRelation",
    "A_UNIT",
    "B_UNIT",
    "FORMULA",
    "VP_UNIT",
]
