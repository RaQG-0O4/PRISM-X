import numpy as np
import pandas as pd

from prismx.analytics import calculate_metrics, calculate_portfolio_returns


def test_portfolio_returns_are_weighted_correctly():
    asset_returns = pd.DataFrame(
        {
            "AAA": [0.10, 0.00],
            "BBB": [0.00, 0.10],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )

    result = calculate_portfolio_returns(asset_returns, {"AAA": 0.60, "BBB": 0.40})

    assert np.isclose(result.iloc[0], 0.06)
    assert np.isclose(result.iloc[1], 0.04)


def test_metrics_capture_total_return_and_drawdown():
    returns = pd.Series(
        [0.10, -0.20, 0.05],
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )

    metrics = calculate_metrics(returns, periods_per_year=3)

    assert np.isclose(metrics.total_return, (1.10 * 0.80 * 1.05) - 1.0)
    assert np.isclose(metrics.maximum_drawdown, (1.10 * 0.80 / 1.10) - 1.0)
