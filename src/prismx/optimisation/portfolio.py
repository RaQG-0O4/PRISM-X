"""Constrained portfolio optimisation baselines and resilience objective."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _stabilise_covariance(covariance: pd.DataFrame) -> pd.DataFrame:
    """Return a symmetric, positive-definite covariance estimate.

    Historical covariance matrices can become nearly singular when assets are
    highly correlated or a rolling window is short. A small eigenvalue floor
    keeps the optimiser numerically stable without materially changing the
    observed risk structure.
    """

    matrix = covariance.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    diagonal_scale = max(float(np.nanmedian(np.diag(matrix))), 1.0e-8)
    eigenvalue_floor = diagonal_scale * 1.0e-8
    clipped_eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    stable = (eigenvectors * clipped_eigenvalues) @ eigenvectors.T
    return pd.DataFrame(stable, index=covariance.index, columns=covariance.columns)


def _optimise(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    objective,
    maximum_weight: float,
) -> pd.Series:
    assets = list(mean_returns.index)
    n_assets = len(assets)
    x0 = np.full(n_assets, 1.0 / n_assets)
    last_message = "unknown solver failure"
    # SLSQP can be sensitive to finite-difference steps when an objective is
    # close to a boundary. Retry with a slightly different numerical scale
    # before reporting a genuine convergence problem to the caller.
    for options in (
        {"maxiter": 2_000, "ftol": 1e-8, "eps": 1e-7},
        {"maxiter": 4_000, "ftol": 1e-7, "eps": 1e-5},
    ):
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=[(0.0, maximum_weight)] * n_assets,
            constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            options=options,
        )
        last_message = str(result.message)
        candidate = np.asarray(result.x, dtype=float)
        feasible = (
            candidate.shape == (n_assets,)
            and np.isfinite(candidate).all()
            and abs(float(candidate.sum()) - 1.0) <= 1e-5
            and candidate.min() >= -1e-6
            and candidate.max() <= maximum_weight + 1e-6
        )
        if result.success and feasible:
            cleaned_weights = np.clip(candidate, 0.0, None)
            cleaned_weights = cleaned_weights / cleaned_weights.sum()
            return pd.Series(cleaned_weights, index=assets, name="weight")
    raise ValueError(f"Portfolio optimisation failed: {last_message}")


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
    covariance = _stabilise_covariance(selected.cov() * 252)
    cov = covariance.to_numpy()

    # A shrinkage covariance estimate is included as a robustness diagnostic.
    # It reduces the sensitivity of the covariance matrix to noisy historical
    # correlations; both covariance estimates are numerically stabilised.
    try:
        from sklearn.covariance import LedoitWolf

        shrinkage_covariance = pd.DataFrame(
            LedoitWolf().fit(selected.to_numpy()).covariance_ * 252,
            index=selected.columns,
            columns=selected.columns,
        )
    except Exception:  # pragma: no cover - scikit-learn is a project dependency
        shrinkage_covariance = covariance.copy()
    shrinkage_covariance = _stabilise_covariance(shrinkage_covariance)

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
        # Scale the small squared-error objective so SLSQP's finite-difference
        # gradient is not lost to machine precision. The minimiser is unchanged.
        return float(np.sum((component - target) ** 2) * 1.0e4)

    def resilient(weights: np.ndarray) -> float:
        portfolio_returns = selected.to_numpy() @ weights
        tail = portfolio_returns[portfolio_returns <= np.quantile(portfolio_returns, 0.05)]
        expected_tail_loss = max(0.0, -float(tail.mean())) if len(tail) else 0.0
        return volatility(weights) + stress_penalty * expected_tail_loss - 0.10 * float(
            weights @ mean_returns.to_numpy()
        )

    robust_cov = shrinkage_covariance.to_numpy()

    def robust_volatility(weights: np.ndarray) -> float:
        return float(np.sqrt(max(weights.T @ robust_cov @ weights, 0.0)))

    return {
        "minimum_volatility": _optimise(mean_returns, covariance, volatility, maximum_weight),
        "robust_minimum_volatility": _optimise(
            mean_returns, shrinkage_covariance, robust_volatility, maximum_weight
        ),
        "maximum_sharpe": _optimise(mean_returns, covariance, negative_sharpe, maximum_weight),
        "risk_parity": _optimise(mean_returns, covariance, risk_parity, maximum_weight),
        "resilient": _optimise(mean_returns, covariance, resilient, maximum_weight),
    }
