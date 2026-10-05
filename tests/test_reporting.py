from types import SimpleNamespace

import pandas as pd

from prismx.analytics import PortfolioMetrics
from prismx.reporting import build_html_report
from prismx.risk import RiskMetrics


def test_complete_html_report_contains_interpretation_and_allocation():
    result = SimpleNamespace(
        amount_inr=100_000.0,
        risk_appetite="moderate",
        benchmark_ticker="^NSEI",
        data_start="2022-01-01",
        data_end="2026-01-01",
        recommended_method="resilient",
        name_to_ticker={"Test Co": "TEST.NS", "Other Co": "OTHER.NS"},
        allocation=pd.DataFrame(
            {
                "stock": ["Test Co", "Other Co"],
                "ticker": ["TEST.NS", "OTHER.NS"],
                "weight": [0.6, 0.4],
                "amount_inr": [60_000.0, 40_000.0],
            }
        ),
        candidate_metrics=pd.DataFrame(
            {"CAGR": [0.08], "volatility": [0.12], "Sharpe": [0.6], "max_drawdown": [-0.2]},
            index=["resilient"],
        ),
        performance_metrics=PortfolioMetrics(10, 0.08, 0.08, 0.12, 0.6, 0.8, 1.0, -0.2),
        risk_metrics=RiskMetrics(0.95, 10, 0.02, 0.03, 0.02, 0.03),
        stress_results=pd.DataFrame(),
        reverse_stress=pd.DataFrame(),
        factor_snapshot=pd.DataFrame(),
        walk_forward_metrics=pd.DataFrame(),
        model_signals={},
        model_errors={},
    )

    report = build_html_report(result)

    assert "Recommended allocation" in report
    assert "Test Co" in report
    assert "How the recommendation was produced" in report
    assert "historical observations, not forecasts" in report
