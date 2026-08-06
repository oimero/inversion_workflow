"""Ordered EventTrack invariants used behind the generator seam."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cup.synthetic.core.prior import ProducerPrior
from ginn_v2.contracts import EventTrack, InputContractError


def validate_event_track_order(
    tracks: Sequence[EventTrack],
    *,
    width: int,
    tolerance: float = 1.0e-6,
) -> None:
    """Validate the topology-free part of an ordered section realization."""

    if width <= 0 or not tracks:
        raise InputContractError("an event section requires a positive width and tracks.")
    identities = [track.event_id for track in tracks]
    if len(set(identities)) != len(identities):
        raise InputContractError("event identities must be unique within a zone.")
    for track in tracks:
        if track.presence.size != width:
            raise InputContractError("event track width differs from section width.")
    duration = np.stack([track.duration_fraction for track in tracks], axis=0)
    active = np.stack([track.presence for track in tracks], axis=0)
    if np.any(duration[~active] > tolerance):
        raise InputContractError("inactive events must have zero duration.")
    totals = np.sum(duration, axis=0)
    if np.any(~np.isclose(totals, 1.0, rtol=0.0, atol=tolerance)):
        raise InputContractError("active event durations must fill each valid zone trace.")


__all__ = [
    "EventTrack",
    "ProducerPrior",
    "validate_event_track_order",
]
