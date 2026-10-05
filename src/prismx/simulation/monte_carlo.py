"""Parametric Monte Carlo simulation using historical moments."""

from __future__ import annotations

from typing import Mapping
from math import isclose

import numpy as np
import pandas as pd


def simulate_portfolio(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    horizon_days: int = 252,
    simulations: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate cumulative portfolio returns from a multivariate normal model."""

    if horizon_days < 1 or simulations < 1:
        raise ValueError("horizon_days and simulations must be positive")
    if len(weights) == 0:
        raise ValueError("At least one portfolio weight is required")
    weight_series = pd.Series(weights, dtype=float)
    if (weight_series < 0).any() or not isclose(float(weight_series.sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Portfolio weights must be non-negative and sum to 1.0")
    selected = asset_returns.loc[:, list(weights)].dropna(how="any")
    if len(selected) < 2:
        raise ValueError("At least two complete return observations are required")
    mean = selected.mean().to_numpy()
    covariance = selected.cov().to_numpy()
    vector = weight_series.reindex(selected.columns).to_numpy()
    rng = np.random.default_rng(seed)
    covariance = covariance + np.eye(len(covariance)) * 1e-10
    draws = rng.multivariate_normal(mean, covariance, size=(simulations, horizon_days))
    daily_portfolio_returns = draws @ vector
    return np.prod(1.0 + daily_portfolio_returns, axis=1) - 1.0


def simulation_summary(
    outcomes: np.ndarray,
    loss_threshold: float = -0.20,
    horizon_days: int = 252,
) -> dict[str, float]:
    """Summarise simulated cumulative outcomes."""

    values = np.asarray(outcomes, dtype=float)
    if values.size == 0:
        raise ValueError("Simulation outcomes cannot be empty")
    if horizon_days < 1 or loss_threshold >= 0:
        raise ValueError("horizon_days must be positive and loss_threshold must be negative")
    return {
        "simulations": float(values.size),
        "horizon_days": float(horizon_days),
        "loss_threshold": float(loss_threshold),
        "mean_return": float(values.mean()),
        "median_return": float(np.median(values)),
        "p01_return": float(np.quantile(values, 0.01)),
        "p05_return": float(np.quantile(values, 0.05)),
        "p95_return": float(np.quantile(values, 0.95)),
        "probability_loss": float(np.mean(values < 0.0)),
        "probability_beyond_loss_threshold": float(np.mean(values <= loss_threshold)),
    }
