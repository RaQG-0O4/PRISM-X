"""SHAP explanations with a lightweight fallback."""

from __future__ import annotations

from typing import Any

import pandas as pd


def explain_model(model: Any, features: pd.DataFrame, max_samples: int = 200) -> pd.DataFrame:
    """Return feature-attribution values for a fitted risk model."""

    sample = features.tail(max_samples)
    try:
        import shap

        explainer = shap.Explainer(model, sample)
        values = explainer(sample).values
        if values.ndim == 3:
            values = values[:, :, -1]
        return pd.DataFrame(values, index=sample.index, columns=sample.columns)
    except ImportError:
        estimator = getattr(model, "named_steps", {}).get("logisticregression")
        if estimator is None or not hasattr(estimator, "coef_"):
            raise RuntimeError("Install SHAP for explanations of this model")
        coefficients = estimator.coef_[0]
        return pd.DataFrame(
            [sample.to_numpy() * coefficients],
            columns=sample.columns,
            index=[sample.index[-1]],
        )
