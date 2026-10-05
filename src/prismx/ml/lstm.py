"""Optional LSTM forecasting of future realised volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_volatility_sequences(
    portfolio_returns: pd.Series,
    sequence_length: int = 30,
    forecast_horizon: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences whose target is future realised volatility."""

    returns = pd.Series(portfolio_returns, dtype=float).dropna().to_numpy()
    volatility = pd.Series(returns).rolling(forecast_horizon).std().to_numpy()
    x_values, y_values = [], []
    for end in range(sequence_length, len(returns) - forecast_horizon + 1):
        target = volatility[end : end + forecast_horizon]
        if np.isnan(target).any():
            continue
        x_values.append(returns[end - sequence_length : end])
        y_values.append(float(np.std(target)))
    x = np.asarray(x_values, dtype=float)[..., np.newaxis]
    y = np.asarray(y_values, dtype=float)
    return x, y


def train_lstm_volatility(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
    purge_gap: int = 5,
    return_metrics: bool = False,
) -> object | tuple[object, dict[str, float]]:
    """Train a small LSTM model; TensorFlow remains an optional dependency.

    Set ``return_metrics=True`` to also return metrics from a chronological,
    purged test period.  The purge gap prevents overlapping sequence targets
    from crossing the train/validation/test boundaries.
    """

    try:
        from tensorflow import keras
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "Install LSTM dependencies with: "
            "python -m pip install -e \".[deep-learning]\""
        ) from error
    if len(x) < 100:
        raise ValueError("At least 100 sequences are recommended for LSTM training")
    if len(x) != len(y):
        raise ValueError("Sequence features and targets must have the same length")
    if not 0.0 < test_fraction < 0.5 or not 0.0 < validation_fraction < 0.5:
        raise ValueError("Test and validation fractions must be between zero and 0.5")
    if purge_gap < 0:
        raise ValueError("Purge gap cannot be negative")

    split_test = int(len(x) * (1.0 - test_fraction))
    split_validation = int(split_test * (1.0 - validation_fraction))
    train_end = split_validation
    validation_start = split_validation + purge_gap
    validation_end = split_test
    test_start = split_test + purge_gap
    x_train, y_train = x[:train_end], y[:train_end]
    x_validation, y_validation = x[validation_start:validation_end], y[validation_start:validation_end]
    x_test, y_test = x[test_start:], y[test_start:]
    if min(len(x_train), len(x_validation), len(x_test)) == 0:
        raise ValueError("Not enough sequences for purged train, validation and test periods")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=x.shape[1:]),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_validation, y_validation),
        shuffle=False,
        verbose=0,
    )
    if not return_metrics:
        return model

    test_predictions = np.asarray(model.predict(x_test, verbose=0)).reshape(-1)
    test_mae = float(np.mean(np.abs(y_test - test_predictions)))
    test_rmse = float(np.sqrt(np.mean((y_test - test_predictions) ** 2)))
    baseline_value = float(np.mean(y_train))
    baseline_rmse = float(np.sqrt(np.mean((y_test - baseline_value) ** 2)))
    test_skill = (
        float(np.clip(1.0 - test_rmse / baseline_rmse, 0.0, 1.0))
        if baseline_rmse > 0
        else 0.0
    )
    return model, {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "baseline_rmse": baseline_rmse,
        "test_skill": test_skill,
        "train_sequences": float(len(x_train)),
        "validation_sequences": float(len(x_validation)),
        "test_sequences": float(len(x_test)),
        "purge_gap": float(purge_gap),
    }
