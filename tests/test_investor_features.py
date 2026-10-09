import numpy as np
import pandas as pd

from prismx.analytics import add_investor_utility_score, compare_to_benchmark
from prismx.ml import adjust_weights_with_signals_trace
from prismx.risk import calculate_concentration_metrics


def test_concentration_metrics_are_consistent():
    metrics = calculate_concentration_metrics({"AAA": 0.6, "BBB": 0.4})

    assert np.isclose(metrics["hhi"], 0.52)
    assert np.isclose(metrics["top_three_weight"], 1.0)
    assert np.isclose(metrics["effective_number_of_holdings"], 1.0 / 0.52)


def test_investor_value_comparison_adds_benchmark_gaps():
    candidates = pd.DataFrame(
        {
            "CAGR": [0.10, 0.08],
            "volatility": [0.15, 0.12],
            "Sharpe": [0.6, 0.5],
            "max_drawdown": [-0.25, -0.18],
        },
        index=["resilient", "market_benchmark"],
    )

    compared = compare_to_benchmark(add_investor_utility_score(candidates, "moderate"))

    assert np.isclose(compared.loc["resilient", "CAGR_vs_benchmark"], 0.02)
    assert compared.loc["resilient", "utility_rank"] >= 1


def test_weight_trace_preserves_constraints_and_exposes_stages():
    base = pd.Series({"AAA": 0.6, "BBB": 0.4})
    low_vol = pd.Series({"AAA": 0.4, "BBB": 0.6})
    final, trace = adjust_weights_with_signals_trace(
        base,
        low_vol,
        {"xgboost": {"severe_loss_probability": 0.5}},
        "moderate",
        0.7,
    )

    assert np.isclose(final.sum(), 1.0)
    assert (final <= 0.7 + 1e-9).all()
    assert {"base_optimizer", "after_xgboost", "final_constrained"}.issubset(trace.columns)
