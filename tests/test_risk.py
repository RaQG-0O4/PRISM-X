import numpy as np
import pandas as pd

from prismx.risk import (
    calculate_risk_metrics,
    calculate_volatility_risk_contribution,
)


def test_expected_shortfall_is_at_least_var_for_loss_data():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.02, 0.03])

    metrics = calculate_risk_metrics(returns, confidence=0.80)

    assert metrics.historical_expected_shortfall >= metrics.historical_var


def test_volatility_contribution_percentages_sum_to_one():
    returns = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.03],
            "BBB": [-0.01, 0.00, 0.02, -0.02],
        }
    )

    contribution = calculate_volatility_risk_contribution(
        returns,
        {"AAA": 0.60, "BBB": 0.40},
    )

    assert np.isclose(contribution["contribution_pct"].sum(), 1.0)
