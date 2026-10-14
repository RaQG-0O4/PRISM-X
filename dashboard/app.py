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


DEFAULT_ANALYSIS_CONFIG = {
    "amount_inr": 3_700_000.0,
    "risk_appetite": "Moderate",
    "history_years": 10,
    "run_xgboost": True,
    "run_finbert": True,
    "run_lstm": True,
    "run_walk_forward": True,
    "run_macro": True,
    "run_monte_carlo": True,
    "transaction_cost_bps": 25.0,
    "monte_carlo_horizon_days": 252,
    "monte_carlo_simulations": 5_000,
    "monte_carlo_loss_threshold": -0.20,
    "news": None,
    "news_file_name": None,
}


def _clean_stock_values(values: object) -> list[str]:
    """Normalise values returned by searchable Streamlit inputs."""

    if values is None:
        return []
    cleaned = []
    for value in values if isinstance(values, (list, tuple, set)) else [values]:
        text = str(value).strip().strip("\"'").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _set_widget_default(key: str, value: object) -> None:
    """Set a widget's initial value without conflicting with Streamlit state."""

    if key not in st.session_state:
        st.session_state[key] = value


def _render_stepper(active_step: int) -> None:
    """Render the guided analysis stages as a compact visual stepper."""

    steps = [
        (1, "Portfolio", "account_balance_wallet"),
        (2, "Models", "psychology"),
        (3, "Review", "fact_check"),
        (4, "Results", "insights"),
    ]
    step_columns = st.columns(len(steps))
    for column, (step_number, label, icon) in zip(step_columns, steps):
        with column:
            if step_number < active_step:
                st.badge(
                    f"{step_number}  {label}",
                    icon=":material/check_circle:",
                    color="green",
                )
            elif step_number == active_step:
                st.badge(
                    f"{step_number}  {label}",
                    icon=f":material/{icon}:",
                    color="blue",
                )
            else:
                st.badge(
                    f"{step_number}  {label}",
                    icon=f":material/{icon}:",
                    color="gray",
                )


def _read_uploaded_news(uploaded_file: object) -> tuple[pd.DataFrame | None, str | None]:
    """Validate an optional historical news file before analysis starts."""

    if uploaded_file is None:
        return None, None
    uploaded_news = pd.read_csv(uploaded_file)
    required_news_columns = {"title", "published_at"}
    missing_news_columns = required_news_columns.difference(uploaded_news.columns)
    if missing_news_columns:
        raise ValueError(
            "News CSV is missing: " + ", ".join(sorted(missing_news_columns))
        )
    return uploaded_news, getattr(uploaded_file, "name", "uploaded_news.csv")


def _analysis_options(config: dict[str, object]) -> dict[str, object]:
    """Build workflow arguments while remaining compatible with older deploys."""

    options = {
        "amount_inr": float(config["amount_inr"]),
        "stock_names": list(config["stock_names"]),
        "risk_appetite": str(config["risk_appetite"]),
        "history_years": int(config["history_years"]),
        "run_xgboost": bool(config["run_xgboost"]),
        "run_finbert": bool(config["run_finbert"]),
        "run_lstm": bool(config["run_lstm"]),
        "run_walk_forward": bool(config["run_walk_forward"]),
        "run_macro": bool(config["run_macro"]),
        "news": config.get("news"),
        "transaction_cost_bps": float(config["transaction_cost_bps"]),
    }
    workflow_parameters = inspect.signature(analyse_user_portfolio).parameters
    if "run_monte_carlo" in workflow_parameters:
        options.update(
            {
                "run_monte_carlo": bool(config["run_monte_carlo"]),
                "monte_carlo_simulations": int(config["monte_carlo_simulations"]),
                "monte_carlo_horizon_days": int(config["monte_carlo_horizon_days"]),
                "monte_carlo_loss_threshold": float(config["monte_carlo_loss_threshold"]),
            }
        )
    return options


def _execute_analysis(config: dict[str, object]) -> None:
    """Run one analysis with visible, honest progress stages."""

    status = None
    try:
        with st.status(
            "PRISM-X is preparing your recommendation",
            expanded=True,
        ) as status:
            status.write("Validating the investment brief")
            status.write("Resolving company names to exchange tickers")
            status.write("Downloading price history and aligning currencies to INR")
            result = analyse_user_portfolio(**_analysis_options(config))
            status.write("Calculating risk, optimisation, scenarios and evidence")
            status.update(
                label="Analysis complete — loading the evidence dashboard",
                state="complete",
                expanded=False,
            )
        st.session_state["interactive_analysis"] = result
        st.session_state["analysis_config"] = config
        st.session_state.pop("interactive_error", None)
        st.session_state["wizard_step"] = 1
        st.rerun()
    except Exception as error:  # noqa: BLE001 - show a user-readable dashboard error
        if status is not None:
            status.update(
                label="Analysis needs attention",
                state="error",
                expanded=True,
            )
        st.session_state["interactive_error"] = str(error)


def _initialise_setup_state() -> None:
    """Initialise all guided-flow state in one predictable place."""

    st.session_state.setdefault("wizard_step", 1)
    st.session_state.setdefault("draft_config", {})
    _set_widget_default("setup_amount", DEFAULT_ANALYSIS_CONFIG["amount_inr"])
    _set_widget_default("setup_mode", "Start from scratch")
    _set_widget_default("setup_scratch_stocks", [])
    _set_widget_default("setup_preset_stocks", PRESET_PORTFOLIO)
    _set_widget_default("setup_risk", DEFAULT_ANALYSIS_CONFIG["risk_appetite"])
    _set_widget_default("setup_history", DEFAULT_ANALYSIS_CONFIG["history_years"])
    _set_widget_default("setup_xgboost", DEFAULT_ANALYSIS_CONFIG["run_xgboost"])
    _set_widget_default("setup_finbert", DEFAULT_ANALYSIS_CONFIG["run_finbert"])
    _set_widget_default("setup_lstm", DEFAULT_ANALYSIS_CONFIG["run_lstm"])
    _set_widget_default("setup_walk_forward", DEFAULT_ANALYSIS_CONFIG["run_walk_forward"])
    _set_widget_default("setup_macro", DEFAULT_ANALYSIS_CONFIG["run_macro"])
    _set_widget_default("setup_monte_carlo", DEFAULT_ANALYSIS_CONFIG["run_monte_carlo"])
    _set_widget_default("setup_transaction_cost", DEFAULT_ANALYSIS_CONFIG["transaction_cost_bps"])
    _set_widget_default("setup_mc_horizon", DEFAULT_ANALYSIS_CONFIG["monte_carlo_horizon_days"])
    _set_widget_default("setup_mc_simulations", DEFAULT_ANALYSIS_CONFIG["monte_carlo_simulations"])
    _set_widget_default("setup_mc_threshold", DEFAULT_ANALYSIS_CONFIG["monte_carlo_loss_threshold"])


def _render_portfolio_step() -> None:
    st.header("Welcome to PRISM-X", icon=":material/rocket_launch:")
    st.write(
        "Build a portfolio brief first. PRISM-X will then resolve your securities, "
        "convert foreign prices to INR where needed, and explain how the final weights were produced."
    )
    with st.container(border=True):
        st.subheader("Step 1 · Define your portfolio", icon=":material/account_balance_wallet:")
        st.caption("Start empty for a real custom portfolio, or use the preset for a quick teaching/demo run.")
        mode = st.segmented_control(
            "Portfolio starting point",
            ["Start from scratch", "Use preset portfolio"],
            key="setup_mode",
            width="stretch",
            help="The preset is optional; it is not silently included in a custom analysis.",
        )
        with st.form("portfolio_setup_form", border=False):
            amount = st.number_input(
                "Amount to invest (INR)",
                min_value=1_000.0,
                step=10_000.0,
                key="setup_amount",
                help="All portfolio allocations and foreign-currency conversions are reported in INR.",
            )
            if mode == "Use preset portfolio":
                selected = st.multiselect(
                    "Search and select securities",
                    options=STOCK_PICKER_OPTIONS,
                    format_func=stock_picker_label,
                    key="setup_preset_stocks",
                    max_selections=20,
                    placeholder="Search a company name or ticker",
                    accept_new_options=True,
                    filter_mode="fuzzy",
                    help="You can remove preset holdings or add more securities by typing their name or ticker.",
                )
            else:
                selected = st.multiselect(
                    "Search and select securities",
                    options=STOCK_PICKER_OPTIONS,
                    format_func=stock_picker_label,
                    key="setup_scratch_stocks",
                    max_selections=20,
                    placeholder="Type a company name or ticker and press Enter",
                    accept_new_options=True,
                    filter_mode="fuzzy",
                    help="The portfolio is empty until you add at least two securities.",
                )
            risk = st.selectbox(
                "Risk appetite",
                ["Conservative", "Moderate", "Aggressive"],
                key="setup_risk",
                help="This controls the optimisation objective and maximum holding concentration.",
            )
            st.caption("Names are resolved to exchange tickers before analysis. Non-Indian securities are converted to INR using aligned FX data.")
            continue_setup = st.form_submit_button(
                "Continue to model selection",
                type="primary",
                icon=":material/arrow_forward:",
                width="stretch",
            )
        if continue_setup:
            stocks = _clean_stock_values(selected)
            if len(stocks) < 2:
                st.error("Add at least two securities before continuing.", icon=":material/error:")
            else:
                st.session_state["draft_config"] = {
                    "amount_inr": float(amount),
                    "stock_names": stocks,
                    "risk_appetite": risk,
                }
                st.session_state["wizard_step"] = 2
                st.rerun()


def _render_model_step() -> None:
    draft = st.session_state["draft_config"]
    st.header("Choose the evidence layers", icon=":material/psychology:")
    st.write("Select the diagnostics you want PRISM-X to run. Recommended defaults are already selected for a complete demonstration.")
    with st.container(border=True):
        st.subheader("Step 2 · Models and history", icon=":material/tune:")
        with st.form("model_setup_form", border=False):
            history = st.slider(
                "Historical data window (years)",
                min_value=3,
                max_value=10,
                key="setup_history",
                help="Longer histories give more observations, but may include market regimes that are less relevant today.",
            )
            st.caption("The longer window is the recommended starting point; all results still remain historical estimates.")
            model_columns = st.columns(3)
            with model_columns[0].container(border=True):
                xgboost = st.checkbox(
                    "XGBoost loss-risk model",
                    key="setup_xgboost",
                    help="Estimates severe-loss probability using market and macro features.",
                )
                st.caption("Classification signal")
            with model_columns[1].container(border=True):
                lstm = st.checkbox(
                    "LSTM volatility forecast",
                    key="setup_lstm",
                    help="Forecasts near-term realised volatility when the deep-learning dependency is available.",
                )
                st.caption("Time-series signal")
            with model_columns[2].container(border=True):
                finbert = st.checkbox(
                    "FinBERT news sentiment",
                    key="setup_finbert",
                    help="Scores recent financial-news text. Upload labelled history for accuracy measurement.",
                )
                st.caption("Text sentiment signal")
            st.subheader("Validation and market context", icon=":material/fact_check:")
            validation_columns = st.columns(3)
            with validation_columns[0]:
                walk_forward = st.checkbox(
                    "Walk-forward benchmark",
                    key="setup_walk_forward",
                    help="Tests strategies on unseen rolling periods and includes transaction costs.",
                )
            with validation_columns[1]:
                macro = st.checkbox(
                    "Macro factors",
                    key="setup_macro",
                    help="Adds VIX, USD/INR, crude, gold and US 10Y context where available.",
                )
            with validation_columns[2]:
                monte_carlo = st.checkbox(
                    "Monte Carlo scenarios",
                    key="setup_monte_carlo",
                    help="Simulates outcome distributions from historical means and correlations.",
                )
            uploaded_news = st.file_uploader(
                "Optional labelled or historical news CSV",
                type=["csv"],
                help="Required columns: title, published_at. Add ticker and label to measure labelled performance.",
            )
            st.caption("This upload is optional. FinBERT can still use recent available headlines when the live news source returns data.")
            with st.expander("Advanced simulation and trading assumptions", icon=":material/settings:"):
                transaction_cost = st.slider(
                    "Transaction cost (basis points)",
                    min_value=0.0,
                    max_value=100.0,
                    step=5.0,
                    key="setup_transaction_cost",
                    help="25 basis points equals 0.25% per rebalance turnover.",
                )
                advanced_columns = st.columns(3)
                with advanced_columns[0]:
                    horizon = st.slider(
                        "Simulation horizon (days)",
                        min_value=21,
                        max_value=756,
                        step=21,
                        key="setup_mc_horizon",
                    )
                with advanced_columns[1]:
                    simulations = st.select_slider(
                        "Number of simulations",
                        options=[1_000, 2_500, 5_000, 10_000],
                        key="setup_mc_simulations",
                    )
                with advanced_columns[2]:
                    threshold = st.slider(
                        "Loss threshold",
                        min_value=-0.50,
                        max_value=-0.05,
                        step=0.05,
                        format="%.0f%%",
                        key="setup_mc_threshold",
                    )
            review_setup = st.form_submit_button(
                "Review analysis",
                type="primary",
                icon=":material/fact_check:",
                width="stretch",
            )
            back_to_portfolio = st.form_submit_button(
                "Back to portfolio setup",
                icon=":material/arrow_back:",
                width="stretch",
            )
        if back_to_portfolio:
            st.session_state["wizard_step"] = 1
            st.rerun()
        if review_setup:
            try:
                news, news_file_name = _read_uploaded_news(uploaded_news)
                st.session_state["draft_config"] = {
                    **draft,
                    "history_years": int(history),
                    "run_xgboost": bool(xgboost),
                    "run_finbert": bool(finbert),
                    "run_lstm": bool(lstm),
                    "run_walk_forward": bool(walk_forward),
                    "run_macro": bool(macro),
                    "run_monte_carlo": bool(monte_carlo),
                    "transaction_cost_bps": float(transaction_cost),
                    "monte_carlo_horizon_days": int(horizon),
                    "monte_carlo_simulations": int(simulations),
                    "monte_carlo_loss_threshold": float(threshold),
                    "news": news,
                    "news_file_name": news_file_name,
                }
                st.session_state["wizard_step"] = 3
                st.rerun()
            except Exception as error:  # noqa: BLE001 - validation message for uploaded data
                st.error(str(error), icon=":material/error:")


def _render_review_step() -> None:
    config = st.session_state["draft_config"]
    st.header("Review before running", icon=":material/fact_check:")
    st.write("This is the final checkpoint. Nothing is downloaded or calculated until you run the analysis.")
    with st.container(border=True):
        st.subheader("Step 3 · Analysis brief", icon=":material/assignment:")
        overview_columns = st.columns(4)
        overview_columns[0].metric("Investment", f"INR {float(config['amount_inr']):,.0f}")
        overview_columns[1].metric("Holdings", len(config["stock_names"]))
        overview_columns[2].metric("Risk appetite", str(config["risk_appetite"]))
        overview_columns[3].metric("History", f"{int(config.get('history_years', 10))} years")
        st.markdown("**Selected securities**")
        with st.container(horizontal=True, gap="small", wrap=True):
            for stock in config["stock_names"]:
                st.badge(str(stock), icon=":material/candlestick_chart:", color="blue")
        selected_models = []
        for key, label in (
            ("run_xgboost", "XGBoost"),
            ("run_lstm", "LSTM"),
            ("run_finbert", "FinBERT"),
            ("run_walk_forward", "Walk-forward"),
            ("run_macro", "Macro factors"),
            ("run_monte_carlo", "Monte Carlo"),
        ):
            if config.get(key):
                selected_models.append(label)
        st.markdown("**Evidence layers**")
        st.write(" · ".join(selected_models) if selected_models else "Historical optimisation only")
        if config.get("news_file_name"):
            st.caption(f"Historical news file: {config['news_file_name']}")
        else:
            st.caption("No historical news file uploaded. FinBERT will use live headlines only when available.")
    action_columns = st.columns(2)
    with action_columns[0]:
        if st.button("Edit model and history settings", icon=":material/arrow_back:", width="stretch"):
            st.session_state["wizard_step"] = 2
            st.rerun()
    with action_columns[1]:
        if st.button("Run PRISM-X analysis", type="primary", icon=":material/play_arrow:", width="stretch"):
            _execute_analysis(config)


def _render_results_sidebar(current_result: object) -> None:
    """Keep the full control surface available after results are loaded."""

    saved_config = dict(st.session_state.get("analysis_config", {}))
    saved_config.setdefault("stock_names", list(getattr(current_result, "name_to_ticker", {}).keys()))
    for key, value in DEFAULT_ANALYSIS_CONFIG.items():
        saved_config.setdefault(key, value)
    options = list(dict.fromkeys(STOCK_PICKER_OPTIONS + list(saved_config["stock_names"])))
    _set_widget_default("rerun_amount", saved_config["amount_inr"])
    _set_widget_default("rerun_stocks", saved_config["stock_names"])
    _set_widget_default("rerun_risk", saved_config["risk_appetite"])
    _set_widget_default("rerun_history", saved_config["history_years"])
    for source_key, widget_key in (
        ("run_xgboost", "rerun_xgboost"),
        ("run_finbert", "rerun_finbert"),
        ("run_lstm", "rerun_lstm"),
        ("run_walk_forward", "rerun_walk_forward"),
        ("run_macro", "rerun_macro"),
        ("run_monte_carlo", "rerun_monte_carlo"),
        ("transaction_cost_bps", "rerun_transaction_cost"),
        ("monte_carlo_horizon_days", "rerun_mc_horizon"),
        ("monte_carlo_simulations", "rerun_mc_simulations"),
        ("monte_carlo_loss_threshold", "rerun_mc_threshold"),
    ):
        _set_widget_default(widget_key, saved_config[source_key])
    with st.sidebar:
        st.header("Analysis controls", icon=":material/tune:")
        st.caption("The current result stays visible until a replacement analysis succeeds.")
        if st.button("Build another portfolio", icon=":material/add:", width="stretch"):
            st.session_state.pop("interactive_analysis", None)
            st.session_state.pop("interactive_error", None)
            st.session_state["draft_config"] = {}
            st.session_state["wizard_step"] = 1
            st.rerun()
        with st.expander("Adjust and rerun", icon=":material/refresh:"):
            with st.form("rerun_analysis_form", border=False):
                amount = st.number_input("Amount to invest (INR)", min_value=1_000.0, step=10_000.0, key="rerun_amount")
                stocks = st.multiselect(
                    "Search and select securities",
                    options=options,
                    format_func=stock_picker_label,
                    key="rerun_stocks",
                    max_selections=20,
                    accept_new_options=True,
                    filter_mode="fuzzy",
                    placeholder="Search or type a company",
                )
                risk = st.selectbox("Risk appetite", ["Conservative", "Moderate", "Aggressive"], key="rerun_risk")
                history = st.slider("History (years)", min_value=3, max_value=10, key="rerun_history")
                st.caption("Model switches")
                xgboost = st.checkbox("XGBoost loss-risk model", key="rerun_xgboost")
                finbert = st.checkbox("FinBERT news sentiment", key="rerun_finbert")
                lstm = st.checkbox("LSTM volatility forecast", key="rerun_lstm")
                walk_forward = st.checkbox("Walk-forward benchmark", key="rerun_walk_forward")
                macro = st.checkbox("Macro factors", key="rerun_macro")
                monte_carlo = st.checkbox("Monte Carlo simulation", key="rerun_monte_carlo")
                news_file = st.file_uploader("Optional historical news CSV", type=["csv"], key="rerun_news_file")
                with st.expander("Advanced assumptions"):
                    transaction_cost = st.slider("Transaction cost (basis points)", min_value=0.0, max_value=100.0, step=5.0, key="rerun_transaction_cost")
                    horizon = st.slider("Simulation horizon (days)", min_value=21, max_value=756, step=21, key="rerun_mc_horizon")
                    simulations = st.select_slider("Number of simulations", options=[1_000, 2_500, 5_000, 10_000], key="rerun_mc_simulations")
                    threshold = st.slider("Loss threshold", min_value=-0.50, max_value=-0.05, step=0.05, format="%.0f%%", key="rerun_mc_threshold")
                rerun = st.form_submit_button("Run replacement analysis", type="primary", icon=":material/play_arrow:", width="stretch")
            if rerun:
                try:
                    news, news_file_name = _read_uploaded_news(news_file)
                    cleaned_stocks = _clean_stock_values(stocks)
                    if len(cleaned_stocks) < 2:
                        raise ValueError("Select at least two securities before rerunning.")
                    _execute_analysis(
                        {
                            "amount_inr": float(amount),
                            "stock_names": cleaned_stocks,
                            "risk_appetite": risk,
                            "history_years": int(history),
                            "run_xgboost": bool(xgboost),
                            "run_finbert": bool(finbert),
                            "run_lstm": bool(lstm),
                            "run_walk_forward": bool(walk_forward),
                            "run_macro": bool(macro),
                            "run_monte_carlo": bool(monte_carlo),
                            "transaction_cost_bps": float(transaction_cost),
                            "monte_carlo_horizon_days": int(horizon),
                            "monte_carlo_simulations": int(simulations),
                            "monte_carlo_loss_threshold": float(threshold),
                            "news": news,
                            "news_file_name": news_file_name,
                        }
                    )
                except Exception as error:  # noqa: BLE001 - validation message for uploaded data
                    st.session_state["interactive_error"] = str(error)
        if st.button("Refresh saved teaching snapshot", icon=":material/refresh:", width="stretch"):
            run_pipeline()


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
_initialise_setup_state()
interactive_error = st.session_state.get("interactive_error")
interactive_analysis = st.session_state.get("interactive_analysis")
if interactive_analysis is not None:
    _render_results_sidebar(interactive_analysis)
else:
    with st.sidebar:
        st.header("Guided setup", icon=":material/route:")
        st.caption("Complete the three steps in the main workspace. The portfolio starts empty unless you choose the preset.")
        st.badge(
            f"Step {min(int(st.session_state.get('wizard_step', 1)), 3)} of 3",
            icon=":material/pending:",
            color="blue",
        )
        if st.button("Reset setup", icon=":material/restart_alt:", width="stretch"):
            st.session_state["wizard_step"] = 1
            st.session_state["draft_config"] = {}
            for key in ("setup_scratch_stocks", "setup_preset_stocks"):
                st.session_state[key] = [] if key.endswith("scratch_stocks") else PRESET_PORTFOLIO
            st.rerun()
        if summary := load_json("pipeline_summary.json"):
            st.caption("A separate saved teaching snapshot is available, but it is not silently used for custom analyses.")
            if st.button("Refresh teaching snapshot", icon=":material/refresh:", width="stretch"):
                run_pipeline()

if interactive_analysis is None:
    active_step = min(int(st.session_state.get("wizard_step", 1)), 3)
    _render_stepper(active_step)
    if interactive_error:
        st.error(interactive_error, icon=":material/error:")
    if active_step == 1:
        _render_portfolio_step()
    elif active_step == 2:
        _render_model_step()
    else:
        _render_review_step()
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
            "latest_price": "Latest price (INR)",
            "native_price": "Native price",
            "currency": "Currency",
            "fx_rate_to_inr": "FX rate to INR",
            "estimated_shares": "Estimated shares",
            "estimated_cost_inr": "Estimated cost (INR)",
        }
    )
    st.dataframe(
        allocation_display.style.format(
            {
                "Weight": "{:.2%}",
                "Amount (INR)": "INR {:,.0f}",
                "Latest price (INR)": "INR {:,.2f}",
                "Native price": "{:,.2f}",
                "FX rate to INR": "{:,.4f}",
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
    currency_summary = getattr(result, "currency_summary", pd.DataFrame()).copy()
    if not currency_summary.empty:
        currency_display = currency_summary.rename(
            columns={
                "ticker": "Ticker",
                "native_currency": "Native currency",
                "native_price": "Native price",
                "fx_rate_to_inr": "FX rate to INR",
                "price_inr": "Price in INR",
            }
        )
        with st.expander("Currency conversion details", icon=":material/currency_exchange:"):
            if set(currency_display["Native currency"]) == {"INR"}:
                st.caption("All selected securities are INR-denominated; no foreign-exchange conversion was required.")
            else:
                st.caption(
                    "Foreign prices were converted to INR using aligned daily exchange rates before returns, risk and share quantities were calculated."
                )
            st.dataframe(
                currency_display.style.format(
                    {
                        "Native price": "{:,.2f}",
                        "FX rate to INR": "{:,.4f}",
                        "Price in INR": "INR {:,.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
                alt="Native prices, exchange rates and INR-converted prices",
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
