import numpy as np
import pandas as pd

from prismx.ml.risk_prediction import build_risk_dataset
from prismx.simulation import simulation_summary
from prismx.stress import apply_scenario, reverse_stress_scenario


def test_scenario_applies_weighted_shocks():
    assert np.isclose(
        apply_scenario({"AAA": 0.6, "BBB": 0.4}, {"AAA": -0.1, "BBB": -0.2}),
        -0.14,
    )


def test_reverse_stress_reaches_target_loss():
    result = reverse_stress_scenario({"AAA": 0.5, "BBB": 0.5}, target_loss=-0.20)
    assert np.isclose((result["weighted_shock"]).sum(), -0.20, atol=1e-5)


def test_simulation_summary_reports_loss_probability():
    summary = simulation_summary(
        np.array([-0.3, -0.1, 0.05, 0.2]), horizon_days=63, loss_threshold=-0.20
    )
    assert summary["probability_loss"] == 0.5
    assert summary["horizon_days"] == 63
    assert summary["loss_threshold"] == -0.20


def test_risk_dataset_uses_future_loss_as_target():
    index = pd.date_range("2020-01-01", periods=150, freq="B")
    returns = pd.Series(np.sin(np.arange(150)) / 100, index=index)
    features, target = build_risk_dataset(returns)
    assert len(features) == len(target)
    assert "volatility_20d" in features.columns
