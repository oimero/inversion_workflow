"""Time-domain Synthoseis-lite calibration facade.

The current time-v1 implementation still lives in the historical top-level
modules for compatibility.  New code should import through this domain package
so the implementation can move behind the seam without touching callers.
"""

from cup.synthetic.calibration import *  # noqa: F403
from cup.synthetic.calibration_pipeline import *  # noqa: F403

