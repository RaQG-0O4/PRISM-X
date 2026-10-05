"""Simple transparent walk-forward portfolio comparison."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from prismx.analytics import calculate_metrics, calculate_portfolio_returns


def backtest_fixed_weights(
    asset_returns: pd.DataFrame,
    portfolios: Mapping[str, Mapping[str, float]],
    transaction_cost_bps: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare buy-and-hold fixed-weight portfolios with initial costs.

    This helper does not model target-weight drift or monthly trades. It
    therefore charges only an initial deployment cost. Strategies that are
    actually re-optimised and rebalanced should use walk-forward evaluation,
    which calculates turnover from the old and new weights.
    """

    series = {}
    metrics = {}
    for name, weights in portfolios.items():
        returns = calculate_portfolio_returns(asset_returns, weights).copy()
        turnover = pd.Series(0.0, index=returns.index)
        turnover.iloc[0] = 1.0
        costs = turnover * transaction_cost_bps / 10_000
        net_returns = returns - costs
        series[name] = net_returns
        metrics[name] = calculate_metrics(net_returns).as_dict()

    return pd.DataFrame(series), pd.DataFrame(metrics).T
