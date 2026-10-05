"""Transparent market and portfolio factor exposures."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd


def calculate_factor_snapshot(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate recent momentum, volatility, beta and drawdown by asset."""

    benchmark = pd.Series(benchmark_returns, dtype=float).rename("benchmark")
    rows = []
    for ticker in asset_returns.columns:
        series = pd.concat([asset_returns[ticker], benchmark], axis=1).dropna()
        returns = series.iloc[:, 0]
        market = series["benchmark"]
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        market_variance = float(market.var(ddof=1))
        beta = float(returns.cov(market) / market_variance) if market_variance > 0 else np.nan
        rows.append(
            {
                "ticker": ticker,
                "momentum_1m": float((1.0 + returns.tail(21)).prod() - 1.0),
                "momentum_3m": float((1.0 + returns.tail(63)).prod() - 1.0),
                "momentum_12m": float((1.0 + returns.tail(252)).prod() - 1.0),
                "volatility_1m": float(returns.tail(21).std(ddof=1) * sqrt(periods_per_year)),
                "beta_to_market": beta,
                "correlation_to_market": float(returns.corr(market)),
                "maximum_drawdown": float(drawdown.min()),
                "negative_days_3m": float(returns.tail(63).lt(0).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
