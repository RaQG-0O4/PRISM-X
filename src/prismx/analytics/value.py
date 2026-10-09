"""Investor-centred comparisons for portfolio recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd


RISK_UTILITY_WEIGHTS: dict[str, tuple[float, float]] = {
    "conservative": (1.50, 0.75),
    "moderate": (1.00, 0.50),
    "aggressive": (0.50, 0.25),
}


def add_investor_utility_score(
    candidate_metrics: pd.DataFrame,
    risk_appetite: str,
) -> pd.DataFrame:
    """Add a transparent, profile-specific utility proxy to candidate metrics.

    This is deliberately labelled a proxy: it is a comparison aid, not a
    forecast of future wealth.  Return, volatility and drawdown are measured
    on the same historical sample for every candidate.
    """

    if candidate_metrics.empty:
        return candidate_metrics.copy()
    appetite = str(risk_appetite).casefold()
    volatility_penalty, drawdown_penalty = RISK_UTILITY_WEIGHTS.get(
        appetite, RISK_UTILITY_WEIGHTS["moderate"]
    )
    result = candidate_metrics.copy()
    result["utility_score"] = (
        result["CAGR"]
        - volatility_penalty * result["volatility"]
        - drawdown_penalty * result["max_drawdown"].abs()
    )
    best = result["utility_score"].max()
    result["utility_rank"] = result["utility_score"].rank(ascending=False, method="min").astype(int)
    result["utility_gap_to_best"] = result["utility_score"] - best
    return result


def compare_to_benchmark(
    candidate_metrics: pd.DataFrame,
    benchmark_name: str = "market_benchmark",
) -> pd.DataFrame:
    """Add historical differences versus a benchmark when available."""

    result = candidate_metrics.copy()
    if benchmark_name not in result.index:
        result["CAGR_vs_benchmark"] = np.nan
        result["volatility_vs_benchmark"] = np.nan
        result["drawdown_vs_benchmark"] = np.nan
        return result
    benchmark = result.loc[benchmark_name]
    result["CAGR_vs_benchmark"] = result["CAGR"] - float(benchmark["CAGR"])
    result["volatility_vs_benchmark"] = result["volatility"] - float(benchmark["volatility"])
    result["drawdown_vs_benchmark"] = result["max_drawdown"] - float(benchmark["max_drawdown"])
    return result
