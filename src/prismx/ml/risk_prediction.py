"""Time-ordered high-risk event prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RiskModelResult:
    model: Any
    model_name: str
    feature_names: list[str]
    metrics: dict[str, float | None]
    test_predictions: pd.Series


def build_risk_dataset(
    portfolio_returns: pd.Series,
    horizon_days: int = 20,
    loss_threshold: float = -0.10,
    external_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build leakage-aware features and a future severe-loss target."""

    returns = pd.Series(portfolio_returns, dtype=float).dropna().sort_index()
    features = pd.DataFrame(index=returns.index)
    features["return_1d"] = returns
    features["return_5d"] = returns.rolling(5).sum()
    features["return_20d"] = returns.rolling(20).sum()
    features["volatility_20d"] = returns.rolling(20).std()
    features["negative_days_20d"] = returns.lt(0).rolling(20).sum()
    wealth = (1 + returns).cumprod()
    features["drawdown"] = wealth / wealth.cummax() - 1
    if external_features is not None and not external_features.empty:
        external = external_features.sort_index().reindex(features.index).ffill()
        external = external.add_prefix("macro_")
        features = features.join(external)

    future_return = (1 + returns).rolling(horizon_days).apply(np.prod, raw=True).shift(-horizon_days) - 1
    target = future_return.le(loss_threshold).astype(float).rename("high_risk_event")
    dataset = features.join(target).dropna()
    return dataset.drop(columns="high_risk_event"), dataset["high_risk_event"].astype(int)


def train_risk_model(
    features: pd.DataFrame,
    target: pd.Series,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
    random_state: int = 42,
) -> RiskModelResult:
    """Train XGBoost when installed, otherwise a documented logistic baseline."""

    if len(features) < 100:
        raise ValueError("At least 100 observations are recommended for risk modelling")
    split_test = int(len(features) * (1 - test_fraction))
    split_validation = int(split_test * (1 - validation_fraction))
    x_train = features.iloc[:split_validation]
    y_train = target.iloc[:split_validation]
    x_validation, x_test = features.iloc[split_validation:split_test], features.iloc[split_test:]
    y_validation, y_test = target.iloc[split_validation:split_test], target.iloc[split_test:]
    if y_train.nunique() < 2:
        raise ValueError("Training period contains only one target class")

    positive_count = int(y_train.sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        )
        model_name = "xgboost"
    except ImportError:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=random_state),
        )
        model_name = "logistic_baseline"

    model.fit(x_train, y_train)
    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold = 0.5
    if y_validation.nunique() > 1:
        candidate_thresholds = np.linspace(0.10, 0.90, 17)
        threshold = float(
            max(
                candidate_thresholds,
                key=lambda value: balanced_accuracy_score(
                    y_validation,
                    (validation_probabilities >= value).astype(int),
                ),
            )
        )
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "decision_threshold": threshold,
        "majority_baseline_accuracy": float(max(y_test.mean(), 1.0 - y_test.mean())),
        "train_observations": float(len(x_train)),
        "validation_observations": float(len(x_validation)),
        "test_observations": float(len(x_test)),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_test, probabilities))
    return RiskModelResult(
        model=model,
        model_name=model_name,
        feature_names=list(features.columns),
        metrics=metrics,
        test_predictions=pd.Series(probabilities, index=x_test.index, name="risk_probability"),
    )
