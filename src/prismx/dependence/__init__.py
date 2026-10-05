"""Correlation and dependence analysis."""

from .correlation import (
    correlation_pairs,
    pairwise_correlation,
    rolling_pairwise_correlation,
    stress_period_returns,
)

__all__ = [
    "correlation_pairs",
    "pairwise_correlation",
    "rolling_pairwise_correlation",
    "stress_period_returns",
]
