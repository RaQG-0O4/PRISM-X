import numpy as np
import pandas as pd

from prismx.backtesting import backtest_fixed_weights


def test_fixed_weight_transaction_costs_are_charged_once_for_buy_and_hold():
    index = pd.date_range("2024-01-02", periods=40, freq="B")
    returns = pd.DataFrame({"AAA": np.zeros(len(index)), "BBB": np.zeros(len(index))}, index=index)

    series, _ = backtest_fixed_weights(
        returns,
        {"portfolio": {"AAA": 0.5, "BBB": 0.5}},
        transaction_cost_bps=25.0,
    )

    rebalance_costs = -series["portfolio"]
    assert np.isclose((rebalance_costs > 0).sum(), 1)
    assert np.isclose(rebalance_costs.sum(), 0.0025)
