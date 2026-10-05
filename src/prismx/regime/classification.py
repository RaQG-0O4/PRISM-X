"""Transparent market-regime classification using a Gaussian mixture model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def classify_regimes(
    portfolio_returns: pd.Series,
    number_of_regimes: int = 3,
    random_state: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Classify observations as calm, normal or stress-like regimes.

    Labels are assigned after fitting according to each state's average
    volatility, making the economic interpretation explicit rather than
    assuming that arbitrary cluster numbers have meaning.
    """

    returns = pd.Series(portfolio_returns, dtype=float).dropna()
    if len(returns) < 100:
        raise ValueError("At least 100 observations are recommended for regime classification")
    features = pd.DataFrame(index=returns.index)
    features["return"] = returns
    features["volatility_20d"] = returns.rolling(20).std()
    features["drawdown"] = (1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1
    features = features.dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = GaussianMixture(n_components=number_of_regimes, random_state=random_state)
    raw_labels = model.fit_predict(scaled)
    result = features.copy()
    result["raw_state"] = raw_labels
    state_volatility = result.groupby("raw_state")["volatility_20d"].mean().sort_values()
    labels = ["calm", "normal", "stress", "crisis"][:number_of_regimes]
    state_to_label = {
        state: labels[position]
        for position, state in enumerate(state_volatility.index)
    }
    result["regime"] = result["raw_state"].map(state_to_label)
    summary = (
        result.groupby("regime")[["return", "volatility_20d", "drawdown"]]
        .agg(["mean", "count"])
        .sort_values(("volatility_20d", "mean"))
    )
    return result["regime"], summary
