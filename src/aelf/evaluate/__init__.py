"""Reference-based quality metrics.

Exports are intentionally narrow: the metrics that need a clean reference, and
an explicit explanation of what is missing when there is not one.
"""

from __future__ import annotations

from .compare import (
    ComparisonReport,
    best_lag_samples,
    compare_to_reference,
    describe_missing_reference,
    sdr_db,
    sisdr_db,
)

__all__ = [
    "ComparisonReport",
    "best_lag_samples",
    "compare_to_reference",
    "describe_missing_reference",
    "sdr_db",
    "sisdr_db",
]
