"""Interactive portfolio-analysis workflow used by the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from prismx.analytics import (
    PortfolioMetrics,
    calculate_asset_returns,
    calculate_factor_snapshot,
    calculate_metrics,
    calculate_portfolio_returns,
)
from prismx.data import (
    collect_macro_returns,
    download_adjusted_close,
    resolve_ticker,
    search_ticker_online,
)
from prismx.evaluation import run_walk_forward_evaluation
from prismx.ml import adjust_weights_with_signals, run_optional_models
from prismx.optimisation import optimise_portfolios
from prismx.risk import RiskMetrics, calculate_risk_metrics
from prismx.stress import (
    StressScenario,
    historical_stress_scenarios,
    reverse_stress_scenario,
    run_scenarios,
)


RISK_APPETITE_SETTINGS: dict[str, dict[str, float | str]] = {
    "conservative": {
        "maximum_weight": 0.25,
        "method": "minimum_volatility",
        "stress_penalty": 2.0,
    },
    "moderate": {
        "maximum_weight": 0.30,
        "method": "resilient",
        "stress_penalty": 1.0,
    },
    "aggressive": {
        "maximum_weight": 0.40,
        "method": "maximum_sharpe",
        "stress_penalty": 0.35,
    },
}


@dataclass
class InteractiveAnalysis:
    amount_inr: float
    risk_appetite: str
    benchmark_ticker: str
    data_start: str
    data_end: str
    name_to_ticker: dict[str, str]
    recommended_method: str
    recommended_weights: pd.Series
    allocation: pd.DataFrame
    candidate_metrics: pd.DataFrame
    performance_metrics: PortfolioMetrics
    risk_metrics: RiskMetrics
    model_signals: dict[str, dict[str, object]]
    model_errors: dict[str, str]
    walk_forward_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    walk_forward_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    walk_forward_error: str | None = None
    stress_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_stress: pd.DataFrame = field(default_factory=pd.DataFrame)
    reverse_stress: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_snapshot: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    macro_errors: dict[str, str] = field(default_factory=dict)


def analyse_user_portfolio(
    amount_inr: float,
    stock_names: list[str],
    risk_appetite: str,
    history_years: int = 5,
    run_xgboost: bool = False,
    run_finbert: bool = False,
    run_lstm: bool = False,
    run_walk_forward: bool = False,
    run_macro: bool = False,
    news: pd.DataFrame | None = None,
    transaction_cost_bps: float = 25.0,
) -> InteractiveAnalysis:
    """Download data and calculate a risk-appetite-specific recommendation."""

    if amount_inr <= 0:
        raise ValueError("Portfolio amount must be greater than zero")
    names = [name.strip() for name in stock_names if name.strip()]
    if len(names) < 2:
        raise ValueError("Enter at least two stocks")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Enter each stock only once")
    risk_appetite = risk_appetite.casefold()
    if risk_appetite not in RISK_APPETITE_SETTINGS:
        raise ValueError("Risk appetite must be conservative, moderate or aggressive")

    name_to_ticker = {}
    for name in names:
        try:
            name_to_ticker[name] = resolve_ticker(name)
        except Exception:
            name_to_ticker[name] = search_ticker_online(name)
    if len(set(name_to_ticker.values())) != len(name_to_ticker):
        raise ValueError("Each entered stock must resolve to a different ticker")
    benchmark_ticker = resolve_ticker("NIFTY 50")
    tickers = list(name_to_ticker.values()) + [benchmark_ticker]
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=history_years)).date()
    prices = download_adjusted_close(tickers, start=start, end=end)
    returns = calculate_asset_returns(prices)
    holding_tickers = list(name_to_ticker.values())
    holding_returns = returns.loc[:, holding_tickers]
    walk_forward_metrics = pd.DataFrame()
    walk_forward_returns = pd.DataFrame()
    walk_forward_error = None
    stress_results = pd.DataFrame()
    historical_stress = pd.DataFrame()
    reverse_stress = pd.DataFrame()
    macro_returns = pd.DataFrame()
    macro_errors: dict[str, str] = {}

    if run_macro:
        macro_returns, macro_errors = collect_macro_returns(start=start, end=end)
    factor_snapshot = calculate_factor_snapshot(holding_returns, returns[benchmark_ticker])

    settings = RISK_APPETITE_SETTINGS[risk_appetite]
    maximum_weight = max(float(settings["maximum_weight"]), 1.0 / len(holding_tickers))
    candidates = optimise_portfolios(
        holding_returns,
        maximum_weight=maximum_weight,
        stress_penalty=float(settings["stress_penalty"]),
    )
    method = str(settings["method"])
    recommended_weights = candidates[method]

    model_signals, model_errors = run_optional_models(
        calculate_portfolio_returns(holding_returns, recommended_weights),
        holding_tickers,
        run_xgboost=run_xgboost,
        run_finbert=run_finbert,
        run_lstm=run_lstm,
        external_features=macro_returns,
        news=news,
    )
    if model_signals:
        recommended_weights = adjust_weights_with_signals(
            recommended_weights,
            candidates["minimum_volatility"],
            model_signals,
            risk_appetite,
            maximum_weight,
        )
        method = f"{method} + model signals"

    if run_walk_forward:
        try:
            walk_forward = run_walk_forward_evaluation(
                returns.loc[:, holding_tickers + [benchmark_ticker]],
                maximum_weight=maximum_weight,
                stress_penalty=float(settings["stress_penalty"]),
                benchmark_ticker=benchmark_ticker,
                transaction_cost_bps=transaction_cost_bps,
            )
            walk_forward_metrics = walk_forward.metrics
            walk_forward_returns = walk_forward.net_returns
        except Exception as error:  # noqa: BLE001 - keep the recommendation usable
            walk_forward_error = str(error)

    candidate_rows = []
    for candidate_name, candidate_weights in candidates.items():
        candidate_returns = calculate_portfolio_returns(holding_returns, candidate_weights)
        metrics = calculate_metrics(candidate_returns)
        candidate_rows.append(
            {
                "method": candidate_name,
                "CAGR": metrics.cagr,
                "volatility": metrics.annualised_volatility,
                "Sharpe": metrics.sharpe_ratio,
                "max_drawdown": metrics.maximum_drawdown,
            }
        )
    candidate_metrics = pd.DataFrame(candidate_rows).set_index("method")

    recommended_returns = calculate_portfolio_returns(holding_returns, recommended_weights)
    if model_signals:
        adjusted_metrics = calculate_metrics(recommended_returns)
        candidate_metrics.loc["model_adjusted"] = {
            "CAGR": adjusted_metrics.cagr,
            "volatility": adjusted_metrics.annualised_volatility,
            "Sharpe": adjusted_metrics.sharpe_ratio,
            "max_drawdown": adjusted_metrics.maximum_drawdown,
        }

    stress_weights = recommended_weights.to_dict()
    top_holding = str(recommended_weights.idxmax())
    stress_results = run_scenarios(
        stress_weights,
        [
            StressScenario("Broad market shock (-20%)", {ticker: -0.20 for ticker in holding_tickers}),
            StressScenario("Severe market shock (-35%)", {ticker: -0.35 for ticker in holding_tickers}),
            StressScenario("Largest holding shock (-40%)", {top_holding: -0.40}),
        ],
    )
    historical_stress = historical_stress_scenarios(
        holding_returns,
        stress_weights,
        number_of_scenarios=5,
    ).reset_index(names="date")
    reverse_stress = reverse_stress_scenario(stress_weights, target_loss=-0.20)

    performance_metrics = calculate_metrics(recommended_returns)
    risk_metrics = calculate_risk_metrics(recommended_returns)
    display_names = {ticker: name for name, ticker in name_to_ticker.items()}
    allocation = pd.DataFrame(
        {
            "stock": [display_names[ticker] for ticker in recommended_weights.index],
            "ticker": recommended_weights.index,
            "weight": recommended_weights.to_numpy(),
            "amount_inr": recommended_weights.to_numpy() * amount_inr,
        }
    ).sort_values("weight", ascending=False, ignore_index=True)
    return InteractiveAnalysis(
        amount_inr=amount_inr,
        risk_appetite=risk_appetite,
        benchmark_ticker=benchmark_ticker,
        data_start=pd.Timestamp(returns.index.min()).date().isoformat(),
        data_end=pd.Timestamp(returns.index.max()).date().isoformat(),
        name_to_ticker=name_to_ticker,
        recommended_method=method,
        recommended_weights=recommended_weights,
        allocation=allocation,
        candidate_metrics=candidate_metrics,
        performance_metrics=performance_metrics,
        risk_metrics=risk_metrics,
        model_signals=model_signals,
        model_errors=model_errors,
        walk_forward_metrics=walk_forward_metrics,
        walk_forward_returns=walk_forward_returns,
        walk_forward_error=walk_forward_error,
        stress_results=stress_results,
        historical_stress=historical_stress,
        reverse_stress=reverse_stress,
        factor_snapshot=factor_snapshot,
        macro_returns=macro_returns,
        macro_errors=macro_errors,
    )
