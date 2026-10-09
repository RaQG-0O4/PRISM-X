"""Optional model signals used to make the interactive recommendation dynamic.

The models are deliberately kept behind an optional boundary.  The portfolio
can still be analysed with historical risk and optimisation when a heavy
dependency (TensorFlow or Transformers) is not installed.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from prismx.ml.lstm import build_volatility_sequences, train_lstm_volatility
from prismx.ml.risk_prediction import build_risk_dataset, train_risk_model
from prismx.sentiment.finbert import aggregate_sentiment, score_finbert


def _run_xgboost_signal(
    portfolio_returns: pd.Series,
    external_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    features, target = build_risk_dataset(portfolio_returns, external_features=external_features)
    result = train_risk_model(features, target)
    latest_probability = float(result.model.predict_proba(features.tail(1))[:, 1][0])
    explanation_method = "feature importance"
    feature_importance: dict[str, float] = {}
    try:
        from prismx.explainability import explain_model

        attribution = explain_model(result.model, features)
        feature_importance = {
            str(name): float(value)
            for name, value in attribution.abs().mean().sort_values(ascending=False).items()
        }
        explanation_method = "SHAP"
    except Exception:  # noqa: BLE001 - the model remains usable without SHAP
        raw_importance = getattr(result.model, "feature_importances_", None)
        if raw_importance is None:
            estimator = getattr(result.model, "named_steps", {}).get("logisticregression")
            raw_importance = getattr(estimator, "coef_", [[0.0] * len(features.columns)])[0]
            explanation_method = "coefficient magnitude"
        feature_importance = {
            name: float(value)
            for name, value in zip(features.columns, raw_importance, strict=False)
        }
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
        )
    return {
        "model_name": result.model_name,
        "severe_loss_probability": latest_probability,
        "metrics": result.metrics,
        "feature_names": result.feature_names,
        "explanation_method": explanation_method,
        "feature_importance": feature_importance,
    }


def _run_lstm_signal(portfolio_returns: pd.Series) -> dict[str, Any]:
    x, y = build_volatility_sequences(portfolio_returns)
    model, test_metrics = train_lstm_volatility(x, y, return_metrics=True)
    forecast = float(np.asarray(model.predict(x[-1:], verbose=0)).reshape(-1)[0])
    current_volatility = float(np.std(x[-1, :, 0]))
    return {
        "model_name": "lstm",
        "forecast_volatility": max(forecast, 0.0),
        "current_sequence_volatility": max(current_volatility, 1e-8),
        "training_sequences": int(len(x)),
        "test_metrics": test_metrics,
    }


def _run_finbert_signal(news: pd.DataFrame) -> dict[str, Any]:
    if news.empty:
        collection_errors = news.attrs.get("collection_errors", {})
        if collection_errors:
            details = "; ".join(
                f"{ticker}: {reason}" for ticker, reason in collection_errors.items()
            )
            raise ValueError(f"No recent financial-news articles were returned; provider errors: {details}")
        raise ValueError("No recent financial-news articles were returned")
    scored = score_finbert(news)
    daily = aggregate_sentiment(scored)
    by_ticker = (
        scored.groupby("ticker")["sentiment_signal"].mean().clip(-1.0, 1.0).to_dict()
        if "ticker" in scored
        else {}
    )
    labelled_accuracy = None
    if "label" in scored:
        expected = scored["label"].astype(str).str.casefold().str.strip()
        predicted = scored["sentiment_label"].astype(str).str.casefold().str.strip()
        labelled_accuracy = float(expected.eq(predicted).mean())
    result = {
        "model_name": "finbert",
        "articles_scored": int(len(scored)),
        "overall_sentiment": float(scored["sentiment_signal"].mean()),
        "average_confidence": float(scored["sentiment_score"].mean()),
        "by_ticker": {str(key): float(value) for key, value in by_ticker.items()},
        "daily_observations": int(len(daily)),
    }
    if labelled_accuracy is not None:
        result["labelled_accuracy"] = labelled_accuracy
    return result


def run_optional_models(
    portfolio_returns: pd.Series,
    tickers: Iterable[str],
    *,
    run_xgboost: bool = False,
    run_finbert: bool = False,
    run_lstm: bool = False,
    external_features: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Run selected optional models and return signals plus user-facing errors."""

    signals: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    if run_xgboost:
        try:
            signals["xgboost"] = _run_xgboost_signal(portfolio_returns, external_features)
        except Exception as error:  # noqa: BLE001 - keep one model from blocking the others
            errors["xgboost"] = str(error)

    if run_lstm:
        try:
            signals["lstm"] = _run_lstm_signal(portfolio_returns)
        except Exception as error:  # noqa: BLE001 - keep one model from blocking the others
            errors["lstm"] = str(error)

    if run_finbert:
        try:
            from prismx.data.news import collect_yfinance_news

            news_frame = news
            if news_frame is None:
                news_frame = collect_yfinance_news(tickers, limit_per_ticker=10)
            signals["finbert"] = _run_finbert_signal(news_frame)
        except Exception as error:  # noqa: BLE001 - keep one model from blocking the others
            errors["finbert"] = str(error)

    overall = build_overall_validation_score(signals, errors)
    if overall:
        signals["overall_validation"] = overall
    return signals, errors


def build_overall_validation_score(
    signals: dict[str, dict[str, Any]],
    errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a transparent 0-100 score from comparable validation metrics.

    This is a model-quality summary, not a promise of portfolio returns.  The
    score only averages models with a measured out-of-sample metric.
    """

    errors = errors or {}
    components: dict[str, float] = {}

    if "xgboost" in signals:
        metrics = signals["xgboost"].get("metrics", {})
        roc_auc = metrics.get("roc_auc")
        components["xgboost"] = float(roc_auc if roc_auc is not None else metrics.get("accuracy", 0.0))

    if "lstm" in signals:
        lstm_metrics = signals["lstm"].get("test_metrics", {})
        components["lstm"] = float(lstm_metrics.get("test_skill", 0.0))

    if "finbert" in signals and signals["finbert"].get("labelled_accuracy") is not None:
        components["finbert"] = float(signals["finbert"]["labelled_accuracy"])

    if not components:
        return {}

    requested = set(signals).difference({"overall_validation"}) | set(errors)
    unmeasured = sorted(requested.difference(components))
    return {
        "score_pct": float(np.mean(list(components.values())) * 100.0),
        "components": {name: value * 100.0 for name, value in components.items()},
        "measured_models": sorted(components),
        "unmeasured_models": unmeasured,
        "measurement_coverage_pct": float(len(components) / max(len(requested), 1) * 100.0),
    }


def _cap_and_normalise(weights: pd.Series, maximum_weight: float) -> pd.Series:
    """Normalise weights while respecting the same per-security cap as the optimiser."""

    result = pd.Series(weights, dtype=float).clip(lower=0.0)
    if result.sum() <= 0:
        result[:] = 1.0 / len(result)
    result = result / result.sum()

    # Redistribute excess weight until the cap is satisfied or no further
    # redistribution is possible because every asset is at its cap.
    for _ in range(len(result) + 2):
        excess = (result - maximum_weight).clip(lower=0.0).sum()
        if excess <= 1e-10:
            break
        over_cap = result > maximum_weight
        result[over_cap] = maximum_weight
        available = ~over_cap
        if not available.any():
            break
        room = (maximum_weight - result[available]).clip(lower=0.0)
        if room.sum() <= 0:
            break
        result.loc[available] += excess * room / room.sum()
    return result / result.sum()


def adjust_weights_with_signals_trace(
    base_weights: pd.Series,
    minimum_volatility_weights: pd.Series,
    signals: dict[str, dict[str, Any]],
    risk_appetite: str,
    maximum_weight: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply small, explainable model adjustments to the optimiser output.

    Signals do not replace the optimiser.  They tilt the result toward the
    minimum-volatility portfolio when predicted risk rises and apply only a
    small capped sentiment tilt.
    """

    weights = pd.Series(base_weights, dtype=float).copy()
    low_vol = pd.Series(minimum_volatility_weights, dtype=float).reindex(weights.index).fillna(0.0)
    trace = pd.DataFrame({"base_optimizer": weights.copy()})

    if "xgboost" in signals:
        probability = float(signals["xgboost"].get("severe_loss_probability", 0.0))
        appetite_blend = {"conservative": 0.30, "moderate": 0.20, "aggressive": 0.10}.get(
            risk_appetite, 0.20
        )
        blend = float(np.clip(probability * appetite_blend, 0.0, 0.30))
        weights = (1.0 - blend) * weights + blend * low_vol
    trace["after_xgboost"] = weights.copy()

    if "lstm" in signals:
        lstm_signal = signals["lstm"]
        forecast = float(lstm_signal.get("forecast_volatility", 0.0))
        current = max(float(lstm_signal.get("current_sequence_volatility", 1e-8)), 1e-8)
        rising_risk = max(forecast / current - 1.0, 0.0)
        blend = float(np.clip(rising_risk * 0.20, 0.0, 0.20))
        weights = (1.0 - blend) * weights + blend * low_vol
    trace["after_lstm"] = weights.copy()

    if "finbert" in signals:
        sentiment = signals["finbert"].get("by_ticker", {})
        tilt_size = {"conservative": 0.02, "moderate": 0.04, "aggressive": 0.05}.get(
            risk_appetite, 0.04
        )
        for ticker in weights.index:
            signal = float(sentiment.get(ticker, 0.0))
            weights.loc[ticker] *= 1.0 + tilt_size * float(np.clip(signal, -1.0, 1.0))

    trace["after_finbert"] = weights.copy()
    final_weights = _cap_and_normalise(weights, maximum_weight)
    trace["final_constrained"] = final_weights.copy()
    trace.index.name = "ticker"
    return final_weights, trace.reset_index()


def adjust_weights_with_signals(
    base_weights: pd.Series,
    minimum_volatility_weights: pd.Series,
    signals: dict[str, dict[str, Any]],
    risk_appetite: str,
    maximum_weight: float,
) -> pd.Series:
    """Apply model tilts while preserving the original public API."""

    weights, _ = adjust_weights_with_signals_trace(
        base_weights,
        minimum_volatility_weights,
        signals,
        risk_appetite,
        maximum_weight,
    )
    return weights
