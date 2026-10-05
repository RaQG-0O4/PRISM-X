"""Return calculations and baseline portfolio performance metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, sqrt
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from prismx.data import resolve_portfolio_tickers, resolve_ticker
from prismx.portfolio import PortfolioInput


@dataclass(frozen=True)
class PortfolioMetrics:
    """Summary metrics for one portfolio return series."""

    observations: int
    total_return: float
    cagr: float | None
    annualised_volatility: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    beta: float | None
    maximum_drawdown: float

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable dictionary representation."""

        return asdict(self)


def load_price_history(path: str | Path) -> pd.DataFrame:
    """Load the saved adjusted-close table."""

    prices = pd.read_csv(path, parse_dates=["date"], index_col="date")
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices.index.name = "date"
    return prices.sort_index()


def calculate_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily returns without filling missing observations."""

    if prices.empty:
        raise ValueError("Price history cannot be empty")
    return prices.sort_index().pct_change(fill_method=None).dropna(how="all")


def portfolio_weights_from_input(portfolio: PortfolioInput) -> dict[str, float]:
    """Convert user holding weights into the resolved ticker representation."""

    ticker_by_name = resolve_portfolio_tickers(portfolio)
    return {
        ticker_by_name[holding.name]: holding.weight
        for holding in portfolio.holdings
    }


def calculate_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Calculate a daily weighted portfolio return series.

    Rows with missing returns for any portfolio holding are excluded rather
    than filled, avoiding the invention of a price observation.
    """

    if len(weights) == 0:
        raise ValueError("At least one portfolio weight is required")

    weight_series = pd.Series(weights, dtype=float)
    if (weight_series < 0).any():
        raise ValueError("Portfolio weights cannot be negative")
    if float(weight_series.sum()) <= 0:
        raise ValueError("At least one portfolio weight must be positive")
    if not isclose(float(weight_series.sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Portfolio weights must sum to 1.0")

    missing_columns = [ticker for ticker in weight_series.index if ticker not in asset_returns]
    if missing_columns:
        raise ValueError(f"Missing asset returns for: {', '.join(missing_columns)}")

    selected = asset_returns.loc[:, list(weight_series.index)].dropna(how="any")
    if selected.empty:
        raise ValueError("No complete return observations are available")

    portfolio_returns = selected.mul(weight_series, axis="columns").sum(axis=1)
    portfolio_returns.name = "portfolio"
    return portfolio_returns


def _ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    return float(numerator / denominator)


def calculate_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_annual: float = 0.0,
    target_return_annual: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioMetrics:
    """Calculate transparent baseline portfolio performance metrics.

    The initial risk-free rate is zero unless supplied explicitly. A sourced
    risk-free series will be added in a later data module.
    """

    returns = pd.Series(portfolio_returns, dtype=float).dropna().sort_index()
    if returns.empty:
        raise ValueError("Portfolio returns cannot be empty")

    wealth = (1.0 + returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = len(returns) / periods_per_year
    cagr = None
    if years > 0 and wealth.iloc[-1] > 0:
        cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)

    annualised_volatility = float(returns.std(ddof=1) * sqrt(periods_per_year))
    annualised_mean = float(returns.mean() * periods_per_year)
    sharpe_ratio = _ratio(annualised_mean - risk_free_annual, annualised_volatility)

    daily_target = target_return_annual / periods_per_year
    downside = np.minimum(returns.to_numpy() - daily_target, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * sqrt(periods_per_year))
    sortino_ratio = _ratio(annualised_mean - target_return_annual, downside_deviation)

    beta = None
    if benchmark_returns is not None:
        benchmark = pd.Series(benchmark_returns, dtype=float).rename("benchmark")
        paired = pd.concat([returns.rename("portfolio"), benchmark], axis=1).dropna()
        if len(paired) > 1:
            benchmark_variance = float(paired["benchmark"].var(ddof=1))
            covariance = float(paired["portfolio"].cov(paired["benchmark"]))
            beta = _ratio(covariance, benchmark_variance)

    drawdowns = wealth / wealth.cummax() - 1.0
    maximum_drawdown = float(drawdowns.min())

    return PortfolioMetrics(
        observations=len(returns),
        total_return=total_return,
        cagr=cagr,
        annualised_volatility=annualised_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        beta=beta,
        maximum_drawdown=maximum_drawdown,
    )


def benchmark_returns_from_frame(returns: pd.DataFrame) -> pd.Series:
    """Extract the resolved NIFTY 50 benchmark return series."""

    benchmark_ticker = resolve_ticker("NIFTY 50")
    if benchmark_ticker not in returns:
        raise ValueError(f"Benchmark returns are missing: {benchmark_ticker}")
    return returns[benchmark_ticker]
