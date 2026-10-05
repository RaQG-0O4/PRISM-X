"""Portfolio stress-testing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class StressScenario:
    name: str
    shocks: Mapping[str, float]


def apply_scenario(weights: Mapping[str, float], shocks: Mapping[str, float]) -> float:
    """Return the portfolio loss/return produced by a shock scenario."""

    return float(sum(weights.get(asset, 0.0) * shocks.get(asset, 0.0) for asset in weights))


def run_scenarios(
    weights: Mapping[str, float],
    scenarios: list[StressScenario],
) -> pd.DataFrame:
    """Evaluate named hypothetical scenarios."""

    rows = []
    for scenario in scenarios:
        portfolio_return = apply_scenario(weights, scenario.shocks)
        rows.append(
            {
                "scenario": scenario.name,
                "portfolio_return": portfolio_return,
                "portfolio_loss": -portfolio_return,
                **{f"shock_{asset}": scenario.shocks.get(asset, 0.0) for asset in weights},
            }
        )
    return pd.DataFrame(rows)


def historical_stress_scenarios(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    number_of_scenarios: int = 5,
) -> pd.DataFrame:
    """Return the worst historical portfolio days and their asset shocks."""

    selected = asset_returns.loc[:, list(weights)].dropna(how="any")
    portfolio_returns = selected.mul(pd.Series(weights), axis="columns").sum(axis=1)
    worst_dates = portfolio_returns.nsmallest(number_of_scenarios).index
    shocks = selected.loc[worst_dates].copy()
    shocks.insert(0, "portfolio_return", portfolio_returns.loc[worst_dates])
    return shocks.sort_values("portfolio_return")


def reverse_stress_scenario(
    weights: Mapping[str, float],
    target_loss: float = -0.20,
    minimum_shock: float = -1.0,
    maximum_shock: float = 0.25,
) -> pd.DataFrame:
    """Find a minimum-size shock vector capable of reaching a target loss."""

    if target_loss >= 0:
        raise ValueError("target_loss must be negative")
    assets = list(weights)
    vector = np.array([weights[asset] for asset in assets], dtype=float)

    def objective(shocks: np.ndarray) -> float:
        return float(np.sum(shocks**2))

    # The weighted shock must be at or below the requested loss.  The previous
    # sign allowed zero shocks because it enforced weighted_shock >= target_loss.
    constraints = [{"type": "ineq", "fun": lambda shocks: float(target_loss - vector @ shocks)}]
    result = minimize(
        objective,
        x0=np.full(len(assets), target_loss),
        method="SLSQP",
        bounds=[(minimum_shock, maximum_shock)] * len(assets),
        constraints=constraints,
    )
    if not result.success:
        raise ValueError(f"Reverse stress optimisation failed: {result.message}")

    shocks = pd.DataFrame({"asset": assets, "shock": result.x})
    shocks["weighted_shock"] = shocks["shock"] * shocks["asset"].map(weights)
    return shocks
