"""Constrained portfolio optimisation baselines and resilience objective."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _optimise(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    objective,
    maximum_weight: float,
) -> pd.Series:
    assets = list(mean_returns.index)
    n_assets = len(assets)
    x0 = np.full(n_assets, 1.0 / n_assets)
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, maximum_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise ValueError(f"Portfolio optimisation failed: {result.message}")
    cleaned_weights = np.clip(result.x, 0.0, None)
    cleaned_weights = cleaned_weights / cleaned_weights.sum()
    return pd.Series(cleaned_weights, index=assets, name="weight")


def optimise_portfolios(
    asset_returns: pd.DataFrame,
    maximum_weight: float = 0.30,
    risk_free_annual: float = 0.0,
    stress_penalty: float = 1.0,
) -> dict[str, pd.Series]:
    """Return minimum-volatility, maximum-Sharpe, risk-parity and resilient weights."""

    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("Asset returns must be a non-empty DataFrame")
    if not np.isfinite(maximum_weight) or not 0.0 < maximum_weight <= 1.0:
        raise ValueError("maximum_weight must be between zero and one")
    if not np.isfinite(risk_free_annual) or not np.isfinite(stress_penalty):
        raise ValueError("risk_free_annual and stress_penalty must be finite")
    selected = asset_returns.dropna(how="any")
    if selected.shape[1] < 2:
        raise ValueError("At least two assets are required for diversification")
    if len(selected) < 2:
        raise ValueError("At least two complete return observations are required")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Asset returns must contain only finite values")
    if maximum_weight * selected.shape[1] < 1.0 - 1e-12:
        raise ValueError("maximum_weight is too small to allocate the full portfolio")
    mean_returns = selected.mean() * 252
    covariance = selected.cov() * 252
    cov = covariance.to_numpy()

    def volatility(weights: np.ndarray) -> float:
        return float(np.sqrt(max(weights.T @ cov @ weights, 0.0)))

    def negative_sharpe(weights: np.ndarray) -> float:
        vol = volatility(weights)
        return 1e6 if vol <= 0 else -float((weights @ mean_returns.to_numpy() - risk_free_annual) / vol)

    def risk_parity(weights: np.ndarray) -> float:
        vol = volatility(weights)
        if vol <= 0:
            return 1e6
        component = weights * (cov @ weights) / vol
        target = vol / len(weights)
        return float(np.sum((component - target) ** 2))

    def resilient(weights: np.ndarray) -> float:
        portfolio_returns = selected.to_numpy() @ weights
        tail = portfolio_returns[portfolio_returns <= np.quantile(portfolio_returns, 0.05)]
        expected_tail_loss = max(0.0, -float(tail.mean())) if len(tail) else 0.0
        return volatility(weights) + stress_penalty * expected_tail_loss - 0.10 * float(
            weights @ mean_returns.to_numpy()
        )

    return {
        "minimum_volatility": _optimise(mean_returns, covariance, volatility, maximum_weight),
        "maximum_sharpe": _optimise(mean_returns, covariance, negative_sharpe, maximum_weight),
        "risk_parity": _optimise(mean_returns, covariance, risk_parity, maximum_weight),
        "resilient": _optimise(mean_returns, covariance, resilient, maximum_weight),
    }
