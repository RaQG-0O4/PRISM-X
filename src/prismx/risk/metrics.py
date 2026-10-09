"""Value-at-Risk, Expected Shortfall and risk-contribution calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, sqrt
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class RiskMetrics:
    """One-day portfolio risk metrics expressed as positive loss magnitudes."""

    confidence: float
    observations: int
    historical_var: float
    historical_expected_shortfall: float
    parametric_var: float | None
    parametric_expected_shortfall: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")


def _clean_returns(returns: pd.Series) -> pd.Series:
    cleaned = pd.Series(returns, dtype=float).dropna()
    if cleaned.empty:
        raise ValueError("Returns cannot be empty")
    return cleaned


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate historical one-day VaR as a positive loss magnitude."""

    _validate_confidence(confidence)
    cleaned = _clean_returns(returns)
    quantile = float(cleaned.quantile(1.0 - confidence))
    return max(0.0, -quantile)


def historical_expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate average loss beyond the historical VaR threshold."""

    _validate_confidence(confidence)
    cleaned = _clean_returns(returns)
    threshold = float(cleaned.quantile(1.0 - confidence))
    tail = cleaned[cleaned <= threshold]
    return max(0.0, -float(tail.mean()))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float | None:
    """Calculate normal-distribution VaR for comparison with historical VaR."""

    _validate_confidence(confidence)
    cleaned = _clean_returns(returns)
    standard_deviation = float(cleaned.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return None
    lower_quantile = float(
        cleaned.mean() + standard_deviation * norm.ppf(1.0 - confidence)
    )
    return max(0.0, -lower_quantile)


def parametric_expected_shortfall(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float | None:
    """Calculate normal-distribution Expected Shortfall."""

    _validate_confidence(confidence)
    cleaned = _clean_returns(returns)
    standard_deviation = float(cleaned.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return None

    tail_probability = 1.0 - confidence
    expected_tail_return = float(
        cleaned.mean()
        - standard_deviation * norm.pdf(norm.ppf(tail_probability)) / tail_probability
    )
    return max(0.0, -expected_tail_return)


def calculate_risk_metrics(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
) -> RiskMetrics:
    """Calculate historical and normal-model risk metrics."""

    cleaned = _clean_returns(portfolio_returns)
    return RiskMetrics(
        confidence=confidence,
        observations=len(cleaned),
        historical_var=historical_var(cleaned, confidence),
        historical_expected_shortfall=historical_expected_shortfall(cleaned, confidence),
        parametric_var=parametric_var(cleaned, confidence),
        parametric_expected_shortfall=parametric_expected_shortfall(cleaned, confidence),
    )


def _validate_weights(weights: Mapping[str, float]) -> pd.Series:
    if len(weights) == 0:
        raise ValueError("At least one portfolio weight is required")
    weight_series = pd.Series(weights, dtype=float)
    if (weight_series < 0).any():
        raise ValueError("Portfolio weights cannot be negative")
    if float(weight_series.sum()) <= 0:
        raise ValueError("At least one portfolio weight must be positive")
    if not isclose(float(weight_series.sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Portfolio weights must sum to 1.0")
    return weight_series


def calculate_concentration_metrics(weights: Mapping[str, float]) -> dict[str, float]:
    """Measure how concentrated a long-only portfolio is.

    HHI is the sum of squared portfolio weights.  A lower value indicates
    broader diversification; the effective number of holdings is its inverse.
    These are descriptive diagnostics, not a claim that diversification alone
    removes investment risk.
    """

    weight_series = _validate_weights(weights)
    hhi = float((weight_series**2).sum())
    return {
        "hhi": hhi,
        "effective_number_of_holdings": float(1.0 / hhi) if hhi > 0 else 0.0,
        "largest_holding_weight": float(weight_series.max()),
        "top_three_weight": float(weight_series.nlargest(3).sum()),
        "number_of_holdings": float(weight_series.size),
    }


def calculate_volatility_risk_contribution(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate each holding's contribution to annualised portfolio volatility."""

    weight_series = _validate_weights(weights)
    missing_columns = [ticker for ticker in weight_series.index if ticker not in asset_returns]
    if missing_columns:
        raise ValueError(f"Missing asset returns for: {', '.join(missing_columns)}")

    selected = asset_returns.loc[:, list(weight_series.index)].dropna(how="any")
    if len(selected) < 2:
        raise ValueError("At least two complete observations are required")

    covariance = selected.cov() * periods_per_year
    vector = weight_series.to_numpy()
    portfolio_variance = float(vector.T @ covariance.to_numpy() @ vector)
    portfolio_volatility = sqrt(max(portfolio_variance, 0.0))
    if portfolio_volatility <= 0:
        raise ValueError("Portfolio volatility must be greater than zero")

    marginal = covariance.to_numpy() @ vector / portfolio_volatility
    component = vector * marginal
    contribution = component / portfolio_volatility

    return pd.DataFrame(
        {
            "weight": vector,
            "marginal_volatility": marginal,
            "component_volatility": component,
            "contribution_pct": contribution,
        },
        index=weight_series.index,
    ).sort_values("contribution_pct", ascending=False)


def calculate_tail_risk_contribution(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Estimate holding contributions to historical Expected Shortfall losses."""

    _validate_confidence(confidence)
    weight_series = _validate_weights(weights)
    missing_columns = [ticker for ticker in weight_series.index if ticker not in asset_returns]
    if missing_columns:
        raise ValueError(f"Missing asset returns for: {', '.join(missing_columns)}")

    selected = asset_returns.loc[:, list(weight_series.index)].dropna(how="any")
    weighted_contributions = selected.mul(weight_series, axis="columns")
    portfolio_returns = weighted_contributions.sum(axis=1)
    threshold = float(portfolio_returns.quantile(1.0 - confidence))
    tail_contributions = weighted_contributions.loc[portfolio_returns <= threshold]
    if tail_contributions.empty:
        raise ValueError("No tail observations are available")

    loss_contribution = -tail_contributions.mean()
    total_loss = float(loss_contribution.sum())
    if total_loss <= 0:
        contribution_pct = pd.Series(0.0, index=loss_contribution.index)
    else:
        contribution_pct = loss_contribution / total_loss

    return pd.DataFrame(
        {
            "weight": weight_series,
            "average_tail_loss_contribution": loss_contribution,
            "contribution_pct": contribution_pct,
        }
    ).sort_values("contribution_pct", ascending=False)
