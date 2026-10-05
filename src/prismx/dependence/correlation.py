"""Pairwise, rolling and stress-period correlation analysis."""

from __future__ import annotations

from math import isclose
from typing import Mapping

import pandas as pd


def _selected_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if weights is None:
        selected = asset_returns.copy()
    else:
        missing = [ticker for ticker in weights if ticker not in asset_returns]
        if missing:
            raise ValueError(f"Missing asset returns for: {', '.join(missing)}")
        selected = asset_returns.loc[:, list(weights)]

    selected = selected.apply(pd.to_numeric, errors="coerce")
    if selected.shape[1] < 2:
        raise ValueError("At least two assets are required for correlation analysis")
    return selected


def pairwise_correlation(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate the normal-period pairwise correlation matrix."""

    selected = _selected_returns(asset_returns, weights)
    return selected.corr(method=method)


def stress_period_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    tail_quantile: float = 0.05,
) -> pd.DataFrame:
    """Return holding returns during the portfolio's worst historical days."""

    if not 0 < tail_quantile < 1:
        raise ValueError("tail_quantile must be between 0 and 1")
    if len(weights) == 0:
        raise ValueError("Weights are required to identify portfolio stress days")

    selected = _selected_returns(asset_returns, weights).dropna(how="any")
    weight_series = pd.Series(weights, dtype=float)
    if (weight_series < 0).any() or not isclose(float(weight_series.sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Weights must be non-negative and sum to 1.0")

    portfolio_returns = selected.mul(weight_series, axis="columns").sum(axis=1)
    threshold = float(portfolio_returns.quantile(tail_quantile))
    stress_returns = selected.loc[portfolio_returns <= threshold]
    if len(stress_returns) < 2:
        raise ValueError("At least two stress observations are required")
    return stress_returns


def rolling_pairwise_correlation(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return rolling correlations in long format for charting and analysis."""

    if window < 2:
        raise ValueError("window must be at least 2")
    selected = _selected_returns(asset_returns, weights)
    min_periods = min_periods or window
    if min_periods < 2 or min_periods > window:
        raise ValueError("min_periods must be between 2 and window")

    frames: list[pd.DataFrame] = []
    columns = list(selected.columns)
    for first_position, first_asset in enumerate(columns):
        for second_asset in columns[first_position + 1 :]:
            correlation = (
                selected[first_asset]
                .rolling(window=window, min_periods=min_periods)
                .corr(selected[second_asset])
                .dropna()
                .rename("correlation")
                .to_frame()
            )
            if correlation.empty:
                continue
            correlation["asset_1"] = first_asset
            correlation["asset_2"] = second_asset
            frames.append(correlation.reset_index())

    if not frames:
        raise ValueError("Not enough observations for rolling correlations")

    result = pd.concat(frames, ignore_index=True)
    date_column = result.columns[0]
    return result.rename(columns={date_column: "date"})[
        ["date", "asset_1", "asset_2", "correlation"]
    ]


def correlation_pairs(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a symmetric correlation matrix into an ordered pair table."""

    frames: list[dict[str, object]] = []
    columns = list(correlation_matrix.columns)
    for first_position, first_asset in enumerate(columns):
        for second_asset in columns[first_position + 1 :]:
            value = float(correlation_matrix.loc[first_asset, second_asset])
            frames.append(
                {
                    "asset_1": first_asset,
                    "asset_2": second_asset,
                    "correlation": value,
                    "absolute_correlation": abs(value),
                }
            )

    return pd.DataFrame(frames).sort_values(
        "absolute_correlation",
        ascending=False,
        ignore_index=True,
    )
