"""Read-only figures and metric aggregation for synthetic artifacts."""

from cup.synthetic.reporting.metrics import (
    aggregate_metric_rows,
    energy_rms,
    finite_mask,
    metric_row,
    regression_metrics,
)

__all__ = [
    "aggregate_metric_rows",
    "energy_rms",
    "finite_mask",
    "metric_row",
    "regression_metrics",
]
