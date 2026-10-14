"""Streamlit dashboard for saved PRISM-X results."""

from __future__ import annotations

import json
import importlib
import inspect
from datetime import date
from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st

# Prefer the source tree from the current GitHub checkout over any cached
# package installation that Streamlit Cloud may still have in its environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# A long-lived Streamlit process may retain modules imported from a previous
# deployment. Remove only PRISM-X modules so the current checkout is loaded.
for module_name in list(sys.modules):
    if module_name == "prismx" or module_name.startswith("prismx."):
        del sys.modules[module_name]
importlib.invalidate_caches()

from prismx.reporting import build_html_report
from prismx.workflow import analyse_user_portfolio


REPORTS = PROJECT_ROOT / "reports" / "tables"

# The picker starts empty by design.  This is only a searchable convenience
# list; users can also type a company name or ticker that is not listed here.
STOCK_PICKER_OPTIONS = [
    "HDFC Bank",
    "ICICI Bank",
    "Reliance Industries",
    "TCS",
    "Infosys",
    "Larsen & Toubro",
    "State Bank of India",
    "JSW Energy",
    "Tata Power",
    "Tata Technologies",
    "HDFC Life",
    "Sun Pharma",
    "ITC",
    "Bharti Airtel",
    "Axis Bank",
    "Kotak Mahindra Bank",
]
PRESET_PORTFOLIO = [
    "HDFC Bank",
    "ICICI Bank",
    "Reliance Industries",
    "TCS",
    "Infosys",
    "Larsen & Toubro",
]
STOCK_TICKER_HINTS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Larsen & Toubro": "LT.NS",
    "State Bank of India": "SBIN.NS",
    "JSW Energy": "JSWENERGY.NS",
    "Tata Power": "TATAPOWER.NS",
    "Tata Technologies": "TATATECH.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "ITC": "ITC.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
}


def stock_picker_label(value: object) -> str:
    """Show a friendly company name and ticker in the searchable picker."""

    text = str(value)
    ticker = STOCK_TICKER_HINTS.get(text)
    return f"{text} · {ticker}" if ticker else f"{text} · search online"


def human_label(value: object) -> str:
    """Convert technical snake_case labels into readable dashboard labels."""

    return str(value).replace("_", " ").strip().title()


METHOD_LABELS = {
    "minimum_volatility": "Minimum volatility",
    "robust_minimum_volatility": "Robust minimum volatility",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk parity",
    "resilient": "Resilient",
    "market_benchmark": "NIFTY 50 benchmark",
    "model_adjusted_in_sample": "Model-adjusted (in-sample)",
    "equal_weight": "Equal weight",
}


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


st.set_page_config(
    page_title="PRISM-X | Portfolio Risk & Resilience",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("PRISM-X — Portfolio Risk & Resilience Engine", icon=":material/monitoring:")
st.caption(
    "Decision-support analytics for portfolio construction, risk diagnostics and model evidence. "
    "Historical and model-based analysis; results are not investment guarantees."
)

st.sidebar.header("Build a portfolio", icon=":material/account_balance_wallet:")
st.sidebar.caption("Choose your starting point, search for stocks, then run one reproducible analysis.")
amount_inr = st.sidebar.number_input(
    "Portfolio amount (INR)",
    min_value=1_000.0,
    value=3_700_000.0,
    step=10_000.0,
)
portfolio_mode = st.sidebar.segmented_control(
    "Starting portfolio",
    ["Start from scratch", "Use preset portfolio"],
    default="Start from scratch",
    key="portfolio_mode",
    width="stretch",
    help="Start from an empty portfolio or load the six-stock teaching preset.",
)
if portfolio_mode == "Use preset portfolio":
    selected_stocks = st.sidebar.multiselect(
        "Search and select stocks",
        options=STOCK_PICKER_OPTIONS,
        default=PRESET_PORTFOLIO,
        format_func=stock_picker_label,
        key="preset_stock_picker",
        max_selections=20,
        placeholder="Search by company name or ticker",
        help="Search the list, remove any preset holding, or type a new name/ticker and press Enter.",
        accept_new_options=True,
        filter_mode="fuzzy",
    )
else:
    selected_stocks = st.sidebar.multiselect(
        "Search and select stocks",
        options=STOCK_PICKER_OPTIONS,
        default=[],
        format_func=stock_picker_label,
        key="custom_stock_picker",
        max_selections=20,
        placeholder="Type a company name or ticker and press Enter",
        help="The portfolio starts empty. Search the list or add any company name/ticker supported by the market-data provider.",
        accept_new_options=True,
        filter_mode="fuzzy",
    )

entered_stocks = [str(value).strip() for value in selected_stocks if str(value).strip()]
if entered_stocks:
    st.sidebar.badge(
        f"{len(entered_stocks)} stock{'s' if len(entered_stocks) != 1 else ''} selected",
        icon=":material/check_circle:",
        color="blue",
    )
else:
    st.sidebar.badge(
        "No stocks selected yet",
        icon=":material/search:",
        color="orange",
    )
st.sidebar.caption(
    "Tip: type a company name or ticker, press Enter, and select at least two holdings. "
    "Names are resolved to exchange tickers before analysis."
)
risk_appetite = st.sidebar.selectbox(
    "Risk appetite",
    ["Conservative", "Moderate", "Aggressive"],
    index=1,
)
history_years = st.sidebar.slider("History (years)", min_value=3, max_value=10, value=5)
st.sidebar.subheader("AI signals", icon=":material/psychology:")
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
run_monte_carlo = st.sidebar.checkbox(
    "Monte Carlo simulation",
    help="Simulates portfolio outcomes using historical means and correlations.",
)
monte_carlo_horizon_days = st.sidebar.slider(
    "Simulation horizon (days)", min_value=21, max_value=756, value=252, step=21
)
monte_carlo_simulations = st.sidebar.select_slider(
    "Number of simulations",
    options=[1_000, 2_500, 5_000, 10_000],
    value=5_000,
)
monte_carlo_loss_threshold = st.sidebar.slider(
    "Simulation loss threshold",
    min_value=-0.50,
    max_value=-0.05,
    value=-0.20,
    step=0.05,
    format="%.0f%%",
)

if st.sidebar.button(
    "Analyse portfolio",
    type="primary",
    icon=":material/play_arrow:",
    disabled=len(entered_stocks) < 2,
):
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
            analysis_options = {
                "amount_inr": amount_inr,
                "stock_names": entered_stocks,
                "risk_appetite": risk_appetite,
                "history_years": history_years,
                "run_xgboost": run_xgboost,
                "run_finbert": run_finbert,
                "run_lstm": run_lstm,
                "run_walk_forward": run_walk_forward,
                "run_macro": run_macro,
                "news": uploaded_news,
                "transaction_cost_bps": transaction_cost_bps,
            }
            # Streamlit can briefly reuse an older installed package while a
            # new GitHub commit is redeploying.  Pass optional arguments only
            # when the loaded workflow exposes them, so that the dashboard
            # remains usable during that short transition.
            workflow_parameters = inspect.signature(analyse_user_portfolio).parameters
            if "run_monte_carlo" in workflow_parameters:
                analysis_options.update(
                    {
                        "run_monte_carlo": run_monte_carlo,
                        "monte_carlo_simulations": int(monte_carlo_simulations),
                        "monte_carlo_horizon_days": monte_carlo_horizon_days,
                        "monte_carlo_loss_threshold": monte_carlo_loss_threshold,
                    }
                )
            else:
                st.info(
                    "The deployment is refreshing its analysis engine. Monte Carlo options will become available after the latest version finishes loading."
                )
            st.session_state["interactive_analysis"] = analyse_user_portfolio(
                **analysis_options,
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
    st.header("Recommended portfolio", icon=":material/verified:")
    selected_method_label = METHOD_LABELS.get(
        result.recommended_method,
        human_label(result.recommended_method),
    )
    st.write(
        f"Risk appetite: **{result.risk_appetite.title()}** · "
        f"Selected objective: **{selected_method_label}**"
    )
    with st.container(horizontal=True, gap="small"):
        st.badge(
            f"{len(result.name_to_ticker)} holdings",
            icon=":material/account_balance:",
            color="blue",
        )
        st.badge(
            f"{result.data_start} to {result.data_end}",
            icon=":material/date_range:",
            color="violet",
        )
        st.badge(
            "Historical and model-based",
            icon=":material/insights:",
            color="green",
        )
    st.download_button(
        "Download analysis report",
        data=build_html_report(result),
        file_name="prismx_complete_analysis_report.html",
        mime="text/html",
        icon=":material/download:",
        help="Includes inputs, ticker resolution, recommended weights, ratios, risk, stress tests, model evidence, methodology and interpretation. Open it in a browser and use Print > Save as PDF if needed.",
    )
    with st.container(border=True):
        metrics = st.columns(5)
        metrics[0].metric("CAGR", f"{result.performance_metrics.cagr:.2%}")
        metrics[1].metric("Volatility", f"{result.performance_metrics.annualised_volatility:.2%}")
        metrics[2].metric("Sharpe", f"{result.performance_metrics.sharpe_ratio:.2f}")
        metrics[3].metric("Max drawdown", f"{result.performance_metrics.maximum_drawdown:.2%}")
        metrics[4].metric("95% Expected Shortfall", f"{result.risk_metrics.historical_expected_shortfall:.2%}")

    with st.container(border=True):
        st.subheader("At a glance", icon=":material/summarize:")
        summary_columns = st.columns(3)
        with summary_columns[0]:
            st.markdown("**Decision**")
            st.write(
                f"The **{selected_method_label.lower()}** objective was selected for a "
                f"**{result.risk_appetite}** risk profile."
            )
        with summary_columns[1]:
            concentration = getattr(result, "concentration_metrics", {})
            st.markdown("**Risk posture**")
            st.write(
                f"The historical worst drawdown was **{result.performance_metrics.maximum_drawdown:.2%}**. "
                f"The largest position is **{concentration.get('largest_holding_weight', 0.0):.2%}**."
            )
        with summary_columns[2]:
            overall_validation = result.model_signals.get("overall_validation")
            evidence_text = (
                f"Evidence score **{float(overall_validation['score_pct']):.1f}/100**"
                if overall_validation
                else "Historical risk evidence"
            )
            validation_text = (
                "Walk-forward results available."
                if not result.walk_forward_metrics.empty
                else "Walk-forward was not run."
            )
            st.markdown("**Evidence**")
            st.write(f"{evidence_text}. {validation_text}")

    st.subheader("Resolved securities", icon=":material/link:")
    resolved = pd.DataFrame(
        [{"Entered name": name, "Resolved ticker": ticker} for name, ticker in result.name_to_ticker.items()]
    )
    st.dataframe(
        resolved,
        width="stretch",
        hide_index=True,
        alt="Entered stock names and their resolved market tickers",
    )
    st.subheader("Data health", icon=":material/health_and_safety:")
    with st.container(border=True):
        health_columns = st.columns(4)
        health_columns[0].metric("Price observations", f"{result.performance_metrics.observations:,}")
        health_columns[1].metric("Data start", result.data_start)
        health_columns[2].metric("Data end", result.data_end)
        health_columns[3].metric("Benchmark", result.benchmark_ticker)
    st.caption(
        "All model and optimisation results use the resolved securities and the displayed historical period. Provider failures are shown explicitly rather than replaced with invented values."
    )
    data_age_days = max(0, (date.today() - pd.Timestamp(result.data_end).date()).days)
    if data_age_days > 7:
        st.warning(
            f"Market data ends on {result.data_end}, approximately {data_age_days} days old. "
            "Refresh the analysis before using the allocation for a current decision.",
            icon=":material/update:",
        )

    st.subheader("Suggested allocations", icon=":material/pie_chart:")
    allocation_display = result.allocation.rename(
        columns={
            "stock": "Stock",
            "ticker": "Ticker",
            "weight": "Weight",
            "amount_inr": "Amount (INR)",
            "latest_price": "Latest price",
            "estimated_shares": "Estimated shares",
            "estimated_cost_inr": "Estimated cost (INR)",
        }
    )
    st.dataframe(
        allocation_display.style.format(
            {
                "Weight": "{:.2%}",
                "Amount (INR)": "INR {:,.0f}",
                "Latest price": "INR {:,.2f}",
                "Estimated shares": "{:,.0f}",
                "Estimated cost (INR)": "INR {:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
        alt="Recommended stock allocations with weights and estimated investment amounts",
    )
    st.bar_chart(
        result.allocation.set_index("stock")["weight"],
        y_label="Weight",
        alt="Recommended portfolio weights by stock",
    )
    with st.expander("Optimisation method comparison", icon=":material/compare_arrows:"):
        candidate_display = result.candidate_metrics.copy()
        candidate_display.index = [
            METHOD_LABELS.get(value, human_label(value)) for value in candidate_display.index
        ]
        candidate_display.index.name = "Method"
        candidate_display = candidate_display.reset_index()
        candidate_display = candidate_display.rename(
            columns={
                "CAGR": "CAGR",
                "volatility": "Volatility",
                "Sharpe": "Sharpe",
                "max_drawdown": "Max drawdown",
                "utility_score": "Utility score",
                "CAGR_vs_benchmark": "CAGR vs benchmark",
                "volatility_vs_benchmark": "Volatility vs benchmark",
                "drawdown_vs_benchmark": "Drawdown vs benchmark",
            }
        )
        st.dataframe(
            candidate_display.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Max drawdown": "{:.2%}",
                    "Utility score": "{:.4f}",
                    "CAGR vs benchmark": "{:.2%}",
                    "Volatility vs benchmark": "{:.2%}",
                    "Drawdown vs benchmark": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
            alt="Comparison of portfolio optimisation methods",
        )

    with st.expander("Investor value comparison", icon=":material/insights:"):
        st.caption(
            "The utility score is a transparent historical comparison proxy that penalises volatility and drawdown according to the selected risk appetite. It is not a return guarantee."
        )
        investor_display = getattr(result, "investor_value", pd.DataFrame()).copy()
        if not investor_display.empty:
            investor_display.index = [
                METHOD_LABELS.get(value, human_label(value)) for value in investor_display.index
            ]
            investor_display.index.name = "Method"
            investor_display = investor_display.reset_index()
            investor_display = investor_display.rename(
                columns={
                    "utility_score": "Utility score",
                    "utility_rank": "Utility rank",
                    "utility_gap_to_best": "Gap to best",
                    "CAGR_vs_benchmark": "CAGR vs benchmark",
                    "volatility_vs_benchmark": "Volatility vs benchmark",
                    "drawdown_vs_benchmark": "Drawdown vs benchmark",
                }
            )
        st.dataframe(
            investor_display.style.format(
                {
                    "Utility score": "{:.4f}",
                    "Gap to best": "{:.4f}",
                    "CAGR vs benchmark": "{:.2%}",
                    "Volatility vs benchmark": "{:.2%}",
                    "Drawdown vs benchmark": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
            alt="Historical investor utility comparison by optimisation method",
        )

    with st.expander("Why these weights?", icon=":material/auto_graph:"):
        st.caption(
            "Each column shows how the constrained optimiser was adjusted by the available model signals. An unchanged column means that model was not selected or produced no usable signal."
        )
        weight_display = getattr(result, "weight_trace", pd.DataFrame()).copy()
        if not weight_display.empty:
            weight_display.index.name = "Ticker"
            weight_display = weight_display.reset_index().rename(
                columns={
                    "base_optimizer": "Base optimiser",
                    "after_xgboost": "After XGBoost",
                    "after_lstm": "After LSTM",
                    "after_finbert": "After FinBERT",
                    "final_constrained": "Final constrained",
                }
            )
        st.dataframe(
            weight_display.style.format(
                {
                    "Base optimiser": "{:.2%}",
                    "After XGBoost": "{:.2%}",
                    "After LSTM": "{:.2%}",
                    "After FinBERT": "{:.2%}",
                    "Final constrained": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
            alt="Step-by-step changes from base optimiser weights to final constrained weights",
        )

    st.subheader("Portfolio concentration", icon=":material/donut_large:")
    concentration = getattr(result, "concentration_metrics", {})
    concentration_columns = st.columns(4)
    concentration_columns[0].metric("HHI", f"{concentration.get('hhi', 0.0):.3f}")
    concentration_columns[1].metric(
        "Effective holdings", f"{concentration.get('effective_number_of_holdings', 0.0):.1f}"
    )
    concentration_columns[2].metric(
        "Largest holding", f"{concentration.get('largest_holding_weight', 0.0):.2%}"
    )
    concentration_columns[3].metric(
        "Top 3 concentration", f"{concentration.get('top_three_weight', 0.0):.2%}"
    )

    if not result.walk_forward_metrics.empty:
        st.subheader("Out-of-sample walk-forward comparison", icon=":material/timeline:")
        st.caption(
            "Weights are learned from earlier data and tested on the next unseen period. "
            "Transaction costs are included at each rebalance."
        )
        walk_forward_display = result.walk_forward_metrics.copy()
        if walk_forward_display.index.name:
            walk_forward_display.index.name = human_label(walk_forward_display.index.name)
        st.dataframe(
            walk_forward_display.style.format(
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
            alt="Out-of-sample walk-forward portfolio comparison",
        )
        wealth = (1.0 + result.walk_forward_returns).cumprod()
        st.line_chart(
            wealth,
            y_label="Growth of INR 1",
            alt="Walk-forward growth of one invested rupee",
        )
        if not getattr(result, "walk_forward_turnover", pd.DataFrame()).empty:
            st.write("Turnover at each rebalance")
            st.line_chart(
                getattr(result, "walk_forward_turnover", pd.DataFrame()),
                y_label="Turnover",
                alt="Walk-forward portfolio turnover",
            )
        if getattr(result, "walk_forward_model_errors", []):
            st.warning(
                "Some walk-forward windows or optional model signals were unavailable; "
                "the affected diagnostics were retained: "
                + "; ".join(getattr(result, "walk_forward_model_errors", [])[:3]),
                icon=":material/warning:",
            )
    if result.walk_forward_error:
        st.warning(f"Walk-forward validation was not completed: {result.walk_forward_error}")

    st.subheader("Stress testing", icon=":material/bolt:")
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
            alt="Hypothetical portfolio stress scenarios",
        )
    with stress_columns[1]:
        st.write("Reverse stress: shocks required to reach -20%")
        st.dataframe(
            result.reverse_stress.style.format(
                {"shock": "{:.2%}", "weighted_shock": "{:.2%}"}
            ),
            width="stretch",
            hide_index=True,
            alt="Reverse stress shocks required to reach a twenty percent loss",
        )
    with st.expander("Worst observed portfolio days", icon=":material/event_busy:"):
        st.dataframe(
            result.historical_stress,
            width="stretch",
            hide_index=True,
            alt="Worst historical portfolio days and contributing security returns",
        )

    st.subheader("Factor and market exposure snapshot", icon=":material/monitoring:")
    st.caption("Recent momentum, volatility, beta, correlation and drawdown are calculated from the selected history.")
    factor_display = result.factor_snapshot.copy()
    factor_display.index.name = "Ticker"
    factor_display = factor_display.reset_index().rename(
        columns={
            "momentum_1m": "Momentum (1 month)",
            "momentum_3m": "Momentum (3 months)",
            "momentum_12m": "Momentum (12 months)",
            "volatility_1m": "Volatility (1 month)",
            "beta_to_market": "Beta to market",
            "correlation_to_market": "Correlation to market",
            "maximum_drawdown": "Maximum drawdown",
            "negative_days_3m": "Negative days (3 months)",
        }
    )
    st.dataframe(
        factor_display.style.format(
            {
                "Momentum (1 month)": "{:.2%}",
                "Momentum (3 months)": "{:.2%}",
                "Momentum (12 months)": "{:.2%}",
                "Volatility (1 month)": "{:.2%}",
                "Beta to market": "{:.2f}",
                "Correlation to market": "{:.2f}",
                "Maximum drawdown": "{:.2%}",
                "Negative days (3 months)": "{:.2%}",
            }
        ),
        width="stretch",
        hide_index=True,
        alt="Stock momentum, volatility and market exposure factors",
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

    if getattr(result, "monte_carlo_summary", {}):
        st.subheader("Monte Carlo simulation", icon=":material/casino:")
        st.caption(
            "A reproducible parametric simulation based on historical moments. It is a scenario tool, not a forecast."
        )
        simulation_columns = st.columns(4)
        simulation_columns[0].metric(
            "Probability of loss", f"{result.monte_carlo_summary['probability_loss']:.2%}"
        )
        simulation_columns[1].metric(
            "Probability beyond threshold",
            f"{result.monte_carlo_summary['probability_beyond_loss_threshold']:.2%}",
        )
        simulation_columns[2].metric(
            "5th percentile outcome", f"{result.monte_carlo_summary['p05_return']:.2%}"
        )
        simulation_columns[3].metric(
            "Median outcome", f"{result.monte_carlo_summary['median_return']:.2%}"
        )
        histogram_counts = pd.cut(
            result.monte_carlo_outcomes, bins=20, include_lowest=True
        ).value_counts().sort_index()
        histogram = pd.DataFrame(
            {
                "lower": [interval.left for interval in histogram_counts.index],
                "upper": [interval.right for interval in histogram_counts.index],
                "Simulations": histogram_counts.to_numpy(dtype=int),
            }
        )
        histogram["Outcome range"] = histogram.apply(
            lambda row: f"{row['lower']:.1%} to {row['upper']:.1%}", axis=1
        )
        try:
            import altair as alt

            chart = (
                alt.Chart(histogram)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "lower:Q",
                        title="Cumulative simulated return",
                        axis=alt.Axis(format=".0%"),
                    ),
                    x2="upper:Q",
                    y=alt.Y("Simulations:Q", title="Simulations"),
                    tooltip=[
                        alt.Tooltip("Outcome range:N", title="Outcome range"),
                        alt.Tooltip("Simulations:Q", title="Simulations", format=",d"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(
                chart,
                width="stretch",
                theme="streamlit",
            )
        except ImportError:  # pragma: no cover - Streamlit installs Altair
            st.bar_chart(
                histogram.set_index("lower")["Simulations"],
                y_label="Simulations",
                alt="Monte Carlo outcome distribution",
            )

    if result.model_signals or result.model_errors:
        st.subheader("Model evidence centre", icon=":material/science:")
        overall_validation = result.model_signals.get("overall_validation")
        if overall_validation:
            overall_columns = st.columns(2)
            overall_columns[0].metric(
                "Composite evidence score (not accuracy)",
                f"{float(overall_validation['score_pct']):.1f}/100",
            )
            overall_columns[1].metric(
                "Validation coverage",
                f"{float(overall_validation['measurement_coverage_pct']):.0f}%",
            )
            st.caption(
                "Different models predict different targets. This composite is only a coverage-weighted research summary, not total portfolio accuracy or a guarantee of future returns."
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
                        f"ROC-AUC: {float(roc_auc):.3f} · "
                        f"test n: {int(xgb_metrics.get('test_observations', 0))}"
                        if roc_auc is not None and pr_auc is not None
                        else (
                            f"{signal['model_name']} · test balanced accuracy: "
                            f"{float(balanced_accuracy):.2%} · "
                            f"PR-AUC unavailable · test accuracy: "
                            f"{float(xgb_metrics.get('accuracy', 0.0)):.2%} · "
                            f"test n: {int(xgb_metrics.get('test_observations', 0))} · "
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
        if "lstm" in result.model_signals:
            lstm_metrics = (
                result.model_signals["lstm"].get("test_metrics")
                or result.model_signals["lstm"].get("validation_metrics")
                or {}
            )
            lstm_skill = lstm_metrics.get("test_skill", lstm_metrics.get("validation_skill"))
            if lstm_skill is not None and float(lstm_skill) <= 0.0:
                st.info(
                    "LSTM did not outperform its baseline on the untouched test period, "
                    "so its forecast is shown for diagnostics but does not tilt the recommended weights.",
                    icon=":material/info:",
                )
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
    st.space("medium")

summary = load_json("pipeline_summary.json")

with st.sidebar:
    st.header("Application controls", icon=":material/tune:")
    st.caption("Manage the current interactive run or refresh the saved teaching snapshot.")
    if st.button(
        "Clear current analysis",
        icon=":material/delete_sweep:",
        disabled=interactive_analysis is None and not interactive_error,
        help="Remove the current recommendation and return to the empty start screen.",
    ):
        st.session_state.pop("interactive_analysis", None)
        st.session_state.pop("interactive_error", None)
        st.rerun()
    if st.button(
        "Refresh saved preset run",
        icon=":material/refresh:",
        help="Refreshes the separate reproducible pipeline snapshot. It does not choose holdings for your current analysis.",
    ):
        run_pipeline()

if interactive_analysis is None:
    st.header("Start a new analysis", icon=":material/add_chart:")
    with st.container(border=True):
        st.subheader("Your portfolio starts empty", icon=":material/search:")
        st.write(
            "Use the sidebar to search for company names or exchange tickers. "
            "Choose at least two holdings, set your risk appetite, and run the analysis."
        )
        start_columns = st.columns(3)
        with start_columns[0]:
            st.markdown("**1 · Choose a starting point**")
            st.caption("Build from scratch or load the preset six-stock teaching portfolio.")
        with start_columns[1]:
            st.markdown("**2 · Search and review**")
            st.caption("Type a company name or ticker and press Enter. The resolved exchange ticker is shown before the results.")
        with start_columns[2]:
            st.markdown("**3 · Run the evidence-led analysis**")
            st.caption("Review allocation, risk, stress tests, validation and model evidence in one report.")
    if summary:
        st.caption(
            "A separate saved teaching snapshot is available in the project, but it is not loaded as your current portfolio. "
            "Run a new analysis above to create a fresh recommendation."
        )
