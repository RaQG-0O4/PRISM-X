"""Streamlit dashboard for saved PRISM-X results."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

from prismx.reporting import build_html_report
from prismx.workflow import analyse_user_portfolio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports" / "tables"


def load_json(name: str) -> dict:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_csv(name: str, index_col: int | None = None) -> pd.DataFrame:
    path = REPORTS / name
    return pd.read_csv(path, index_col=index_col) if path.exists() else pd.DataFrame()


def run_pipeline() -> None:
    with st.spinner("Running PRISM-X analysis..."):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_prismx.py")],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        st.success("Analysis completed. Refreshing the dashboard...")
        st.rerun()
    else:
        st.error("The analysis could not complete.")
        if result.stdout:
            st.code(result.stdout)
        if result.stderr:
            st.code(result.stderr)


st.set_page_config(page_title="PRISM-X", layout="wide")
st.title("PRISM-X — Portfolio Risk & Resilience Engine")
st.caption("Historical and model-based analysis. Results are not investment guarantees.")

st.sidebar.header("Analyse a portfolio")
amount_inr = st.sidebar.number_input(
    "Portfolio amount (INR)",
    min_value=1_000.0,
    value=3_700_000.0,
    step=10_000.0,
)
stock_text = st.sidebar.text_area(
    "Stock names or tickers",
    value="HDFC Bank\nICICI Bank\nReliance Industries\nTCS\nInfosys\nLarsen & Toubro",
    help="Enter one supported stock name or exchange ticker per line.",
)
risk_appetite = st.sidebar.selectbox(
    "Risk appetite",
    ["Conservative", "Moderate", "Aggressive"],
    index=1,
)
history_years = st.sidebar.slider("History (years)", min_value=3, max_value=10, value=5)
st.sidebar.subheader("Optional AI models")
run_xgboost = st.sidebar.checkbox(
    "XGBoost loss-risk model",
    help="Estimates the probability of a severe loss over the next 20 trading days.",
)
run_finbert = st.sidebar.checkbox(
    "FinBERT news sentiment",
    help="Downloads recent financial-news headlines and scores their sentiment.",
)
historical_news_file = st.sidebar.file_uploader(
    "Optional labelled/historical news CSV",
    type=["csv"],
    help="Required columns: title, published_at. Add ticker for per-stock sentiment and label for accuracy measurement.",
)
run_lstm = st.sidebar.checkbox(
    "LSTM volatility forecast",
    help="Forecasts near-term realised volatility. TensorFlow is required.",
)
run_walk_forward = st.sidebar.checkbox(
    "Walk-forward benchmark",
    help="Tests the strategies on unseen rolling periods and compares them with equal-weight and NIFTY 50.",
)
transaction_cost_bps = st.sidebar.slider(
    "Transaction cost (basis points)",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=5.0,
    help="Used in walk-forward validation. 25 basis points equals 0.25% per rebalance turnover.",
)
run_macro = st.sidebar.checkbox(
    "Macro factors",
    help="Downloads India VIX, USD/INR, crude oil, gold and US 10Y yield data for context and XGBoost features.",
)

if st.sidebar.button("Analyse Portfolio", type="primary"):
    entered_stocks = [value.strip() for value in stock_text.replace(",", "\n").splitlines()]
    with st.spinner("Downloading data and optimising the portfolio..."):
        try:
            uploaded_news = None
            if historical_news_file is not None:
                uploaded_news = pd.read_csv(historical_news_file)
                required_news_columns = {"title", "published_at"}
                missing_news_columns = required_news_columns.difference(uploaded_news.columns)
                if missing_news_columns:
                    raise ValueError(
                        "News CSV is missing: " + ", ".join(sorted(missing_news_columns))
                    )
            st.session_state["interactive_analysis"] = analyse_user_portfolio(
                amount_inr=amount_inr,
                stock_names=entered_stocks,
                risk_appetite=risk_appetite,
                history_years=history_years,
                run_xgboost=run_xgboost,
                run_finbert=run_finbert,
                run_lstm=run_lstm,
                run_walk_forward=run_walk_forward,
                run_macro=run_macro,
                news=uploaded_news,
                transaction_cost_bps=transaction_cost_bps,
            )
            st.session_state.pop("interactive_error", None)
        except Exception as error:  # noqa: BLE001 - show a user-readable dashboard error
            st.session_state["interactive_error"] = str(error)

interactive_error = st.session_state.get("interactive_error")
interactive_analysis = st.session_state.get("interactive_analysis")
if interactive_error:
    st.error(interactive_error)

if interactive_analysis is not None:
    result = interactive_analysis
    st.subheader("Recommended portfolio")
    st.write(
        f"Risk appetite: **{result.risk_appetite.title()}** · "
        f"Selected objective: **{result.recommended_method.replace('_', ' ').title()}**"
    )
    st.download_button(
        "Download complete analysis report (HTML)",
        data=build_html_report(result),
        file_name="prismx_complete_analysis_report.html",
        mime="text/html",
        help="Includes inputs, ticker resolution, recommended weights, ratios, risk, stress tests, model evidence, methodology and interpretation. Open it in a browser and use Print > Save as PDF if needed.",
    )
    metrics = st.columns(5)
    metrics[0].metric("CAGR", f"{result.performance_metrics.cagr:.2%}")
    metrics[1].metric("Volatility", f"{result.performance_metrics.annualised_volatility:.2%}")
    metrics[2].metric("Sharpe", f"{result.performance_metrics.sharpe_ratio:.2f}")
    metrics[3].metric("Max drawdown", f"{result.performance_metrics.maximum_drawdown:.2%}")
    metrics[4].metric("95% Expected Shortfall", f"{result.risk_metrics.historical_expected_shortfall:.2%}")

    st.subheader("Resolved securities")
    resolved = pd.DataFrame(
        [{"stock_name": name, "resolved_ticker": ticker} for name, ticker in result.name_to_ticker.items()]
    )
    st.dataframe(resolved, width="stretch", hide_index=True)

    st.subheader("Suggested allocations")
    st.dataframe(
        result.allocation.style.format({"weight": "{:.2%}", "amount_inr": "INR {:,.0f}"}),
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(
        result.allocation.set_index("stock")["weight"],
        y_label="Weight",
        alt="Recommended portfolio weights by stock",
    )
    st.subheader("Optimisation method comparison")
    st.dataframe(
        result.candidate_metrics.style.format(
            {"CAGR": "{:.2%}", "volatility": "{:.2%}", "Sharpe": "{:.2f}", "max_drawdown": "{:.2%}"}
        ),
        width="stretch",
    )

    if not result.walk_forward_metrics.empty:
        st.subheader("Out-of-sample walk-forward comparison")
        st.caption(
            "Weights are learned from earlier data and tested on the next unseen period. "
            "Transaction costs are included at each rebalance."
        )
        st.dataframe(
            result.walk_forward_metrics.style.format(
                {
                    "total_return": "{:.2%}",
                    "cagr": "{:.2%}",
                    "annualised_volatility": "{:.2%}",
                    "sharpe_ratio": "{:.2f}",
                    "sortino_ratio": "{:.2f}",
                    "maximum_drawdown": "{:.2%}",
                }
            ),
            width="stretch",
        )
        wealth = (1.0 + result.walk_forward_returns).cumprod()
        st.line_chart(
            wealth,
            y_label="Growth of INR 1",
            alt="Walk-forward growth of one invested rupee",
        )
    if result.walk_forward_error:
        st.warning(f"Walk-forward validation was not completed: {result.walk_forward_error}")

    st.subheader("Stress testing")
    st.caption("Hypothetical shocks and the worst observed portfolio days show how the recommendation can fail.")
    stress_columns = st.columns(2)
    with stress_columns[0]:
        st.write("Hypothetical scenarios")
        st.dataframe(
            result.stress_results.style.format(
                {"portfolio_return": "{:.2%}", "portfolio_loss": "{:.2%}"}
            ),
            width="stretch",
            hide_index=True,
        )
    with stress_columns[1]:
        st.write("Reverse stress: shocks required to reach -20%")
        st.dataframe(
            result.reverse_stress.style.format(
                {"shock": "{:.2%}", "weighted_shock": "{:.2%}"}
            ),
            width="stretch",
            hide_index=True,
        )
    st.write("Worst observed portfolio days")
    st.dataframe(result.historical_stress, width="stretch", hide_index=True)

    st.subheader("Factor and market exposure snapshot")
    st.caption("Recent momentum, volatility, beta, correlation and drawdown are calculated from the selected history.")
    st.dataframe(
        result.factor_snapshot.style.format(
            {
                "momentum_1m": "{:.2%}",
                "momentum_3m": "{:.2%}",
                "momentum_12m": "{:.2%}",
                "volatility_1m": "{:.2%}",
                "beta_to_market": "{:.2f}",
                "correlation_to_market": "{:.2f}",
                "maximum_drawdown": "{:.2%}",
                "negative_days_3m": "{:.2%}",
            }
        ),
        width="stretch",
    )
    if not result.macro_returns.empty:
        st.write("Macro-factor snapshot")
        macro_summary = pd.DataFrame(
            {
                "1M return": (1.0 + result.macro_returns.tail(21)).prod() - 1.0,
                "3M return": (1.0 + result.macro_returns.tail(63)).prod() - 1.0,
            }
        )
        st.dataframe(
            macro_summary.style.format("{:.2%}"),
            width="stretch",
        )
    for macro_name, macro_error in result.macro_errors.items():
        st.warning(f"Macro factor unavailable — {macro_name}: {macro_error}")

    if result.model_signals or result.model_errors:
        st.subheader("AI model signals used")
        overall_validation = result.model_signals.get("overall_validation")
        if overall_validation:
            overall_columns = st.columns(2)
            overall_columns[0].metric(
                "Overall model validation score",
                f"{float(overall_validation['score_pct']):.1f}/100",
            )
            overall_columns[1].metric(
                "Validation coverage",
                f"{float(overall_validation['measurement_coverage_pct']):.0f}%",
            )
            st.caption(
                "This is a model-quality score from time-ordered validation, not a guarantee of future returns."
            )
        signal_rows = []
        if "xgboost" in result.model_signals:
            signal = result.model_signals["xgboost"]
            xgb_metrics = signal.get("metrics", {})
            balanced_accuracy = xgb_metrics.get("balanced_accuracy", xgb_metrics.get("accuracy", 0.0))
            pr_auc = xgb_metrics.get("pr_auc")
            roc_auc = xgb_metrics.get("roc_auc")
            signal_rows.append(
                {
                    "model": "XGBoost",
                    "signal": f"{float(signal['severe_loss_probability']):.2%} severe-loss probability",
                    "details": (
                        f"{signal['model_name']} · test balanced accuracy: "
                        f"{float(balanced_accuracy):.2%} · "
                        f"PR-AUC: {float(pr_auc):.3f} · "
                        f"test accuracy: "
                        f"{float(xgb_metrics.get('accuracy', 0.0)):.2%} · "
                        f"ROC-AUC: {float(roc_auc):.3f}"
                        if roc_auc is not None and pr_auc is not None
                        else (
                            f"{signal['model_name']} · test balanced accuracy: "
                            f"{float(balanced_accuracy):.2%} · "
                            f"PR-AUC unavailable · test accuracy: "
                            f"{float(xgb_metrics.get('accuracy', 0.0)):.2%} · "
                            + (f"ROC-AUC: {float(roc_auc):.3f}" if roc_auc is not None else "ROC-AUC unavailable")
                        )
                    ),
                }
            )
        if "finbert" in result.model_signals:
            signal = result.model_signals["finbert"]
            signal_rows.append(
                {
                    "model": "FinBERT",
                    "signal": f"{float(signal['overall_sentiment']):+.2f} average news sentiment",
                    "details": (
                        f"{signal['articles_scored']} articles scored · "
                        f"average confidence: {float(signal['average_confidence']):.2%} · "
                        + (
                            f"labelled accuracy: {float(signal['labelled_accuracy']):.2%}"
                            if signal.get("labelled_accuracy") is not None
                            else "live accuracy unavailable without labelled news"
                        )
                    ),
                }
            )
        if "lstm" in result.model_signals:
            signal = result.model_signals["lstm"]
            lstm_metrics = signal.get("test_metrics") or signal.get("validation_metrics") or {}
            is_legacy_validation = "test_metrics" not in signal
            mae = lstm_metrics.get("test_mae", lstm_metrics.get("validation_mae"))
            rmse = lstm_metrics.get("test_rmse", lstm_metrics.get("validation_rmse"))
            skill = lstm_metrics.get("test_skill", lstm_metrics.get("validation_skill"))
            if mae is None or rmse is None or skill is None:
                lstm_details = f"{signal.get('training_sequences', 0)} sequences · evaluation metrics unavailable"
            else:
                metric_label = "legacy validation" if is_legacy_validation else "untouched test"
                lstm_details = (
                    f"{signal.get('training_sequences', 0)} sequences · {metric_label} MAE: "
                    f"{float(mae):.4f} · RMSE: {float(rmse):.4f} · "
                    f"skill vs baseline: {float(skill):.2%}"
                )
            signal_rows.append(
                {
                    "model": "LSTM",
                    "signal": f"{float(signal.get('forecast_volatility', 0.0)):.2%} forecast volatility",
                    "details": lstm_details,
                }
            )
        if signal_rows:
            st.dataframe(pd.DataFrame(signal_rows), width="stretch", hide_index=True)
        if "xgboost" in result.model_signals:
            feature_importance = result.model_signals["xgboost"].get("feature_importance", {})
            if feature_importance:
                st.write(
                    "XGBoost explanation — "
                    f"{result.model_signals['xgboost'].get('explanation_method', 'model importance')}"
                )
                explanation = pd.DataFrame(
                    list(feature_importance.items()), columns=["feature", "importance"]
                )
                st.dataframe(
                    explanation.head(10).style.format({"importance": "{:.4f}"}),
                    width="stretch",
                    hide_index=True,
                )
        for model_name, error in result.model_errors.items():
            st.warning(f"{model_name.title()} was not used: {error}")
    st.divider()

summary = load_json("pipeline_summary.json")
# Older pipeline runs did not include performance metrics in this file. Use the
# existing analytics report until the pipeline is refreshed.
risk = summary.get("performance_metrics") or load_json("portfolio_metrics.json")
simulation = summary.get("simulation", {})

with st.sidebar:
    st.header("Controls")
    if st.button("Run / refresh analysis", type="primary"):
        run_pipeline()

if not summary:
    st.warning("No pipeline results are available yet.")
    st.write("Run the core analysis to download or load prices, calculate risk, optimise weights and create the dashboard tables.")
    if st.button("Run analysis now", type="primary"):
        run_pipeline()
else:
    st.subheader("Portfolio overview")
    columns = st.columns(6)
    columns[0].metric("Historical CAGR", f"{risk.get('cagr', 0):.2%}")
    columns[1].metric("Volatility", f"{risk.get('annualised_volatility', 0):.2%}")
    columns[2].metric("Sharpe", f"{risk.get('sharpe_ratio', 0):.2f}")
    columns[3].metric("Max drawdown", f"{risk.get('maximum_drawdown', 0):.2%}")
    simulation_horizon = int(simulation.get("horizon_days", 252))
    columns[4].metric(
        f"Probability of any loss over {simulation_horizon} days",
        f"{simulation.get('probability_loss', 0):.2%}",
    )
    columns[5].metric(
        f"Probability of loss beyond {abs(float(simulation.get('loss_threshold', -0.20))):.0%}",
        f"{simulation.get('probability_beyond_loss_threshold', 0):.2%}",
    )

    st.subheader("Risk contribution")
    contribution = load_csv("pipeline_risk_contribution.csv", index_col=0)
    if not contribution.empty:
        st.dataframe(contribution, width="stretch")

    st.subheader("Backtest comparison")
    backtest = load_csv("pipeline_backtest_metrics.csv", index_col=0)
    if not backtest.empty:
        st.dataframe(backtest, width="stretch")

    st.subheader("Interconnectedness")
    network = load_json("pipeline_network_summary.json")
    if network:
        network_rows = pd.DataFrame(
            [{"period": period, **values} for period, values in network.items()]
        )
        st.dataframe(
            network_rows.style.format(
                {
                    "nodes": "{:.0f}",
                    "edges": "{:.0f}",
                    "density": "{:.2%}",
                    "connected_components": "{:.0f}",
                    "largest_component_share": "{:.2%}",
                    "average_clustering": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption("Normal and stress-period correlation-network summary. Higher density or larger connected components indicate stronger interconnectedness.")

    st.subheader("Regime history")
    regimes = load_csv("pipeline_regimes.csv", index_col=0)
    if not regimes.empty:
        st.dataframe(regimes.tail(100), width="stretch")
