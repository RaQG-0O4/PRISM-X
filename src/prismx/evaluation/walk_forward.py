"""Walk-forward validation for portfolio strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from prismx.analytics import calculate_metrics, calculate_portfolio_returns
from prismx.ml import adjust_weights_with_signals, run_optional_models
from prismx.optimisation import optimise_portfolios


@dataclass
class WalkForwardResult:
    """Out-of-sample returns and metrics for each comparison strategy."""

    net_returns: pd.DataFrame
    metrics: pd.DataFrame
    turnover: pd.DataFrame = field(default_factory=pd.DataFrame)
    transaction_costs: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_errors: list[str] = field(default_factory=list)


def _equal_weights(columns: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(columns), index=columns, name="weight")


def _turnover(previous: pd.Series | None, current: pd.Series) -> float:
    if previous is None:
        return 1.0
    aligned = current.align(previous, join="outer", fill_value=0.0)
    return float(0.5 * (aligned[0] - aligned[1]).abs().sum())


def run_walk_forward_evaluation(
    asset_returns: pd.DataFrame,
    maximum_weight: float = 0.30,
    stress_penalty: float = 1.0,
    benchmark_ticker: str | None = None,
    train_window: int = 504,
    test_window: int = 63,
    transaction_cost_bps: float = 25.0,
    include_ai_model: bool = False,
) -> WalkForwardResult:
    """Evaluate strategies using rolling training and unseen test windows.

    Each rebalance learns weights only from the preceding training window. The
    following test window is then held out, with transaction costs charged at
    the rebalance date. This prevents future prices from influencing earlier
    recommendations.
    """

    data = asset_returns.sort_index().dropna(how="all")
    if benchmark_ticker and benchmark_ticker not in data:
        raise ValueError(f"Benchmark returns are missing: {benchmark_ticker}")
    holding_columns = [column for column in data.columns if column != benchmark_ticker]
    if len(holding_columns) < 2:
        raise ValueError("At least two investable assets are required for walk-forward evaluation")
    if len(data) < train_window + test_window:
        raise ValueError(
            f"At least {train_window + test_window} return observations are required "
            "for walk-forward evaluation"
        )

    strategy_names = [
        "equal_weight",
        "minimum_volatility",
        "robust_minimum_volatility",
        "maximum_sharpe",
        "risk_parity",
        "resilient",
    ]
    return_parts: dict[str, list[pd.Series]] = {name: [] for name in strategy_names}
    if include_ai_model:
        return_parts["ai_xgboost"] = []
    if benchmark_ticker:
        return_parts["market_benchmark"] = []
    previous_weights: dict[str, pd.Series] = {}
    turnover_events: list[dict[str, object]] = []
    cost_events: list[dict[str, object]] = []
    model_errors: list[str] = []

    for test_start in range(train_window, len(data) - test_window + 1, test_window):
        train = data.iloc[test_start - train_window : test_start]
        test = data.iloc[test_start : test_start + test_window]
        train_holdings = train.loc[:, holding_columns].dropna(how="any")
        test_holdings = test.loc[:, holding_columns].dropna(how="any")
        if train_holdings.empty or test_holdings.empty:
            continue

        candidates = optimise_portfolios(
            train_holdings,
            maximum_weight=max(maximum_weight, 1.0 / len(holding_columns)),
            stress_penalty=stress_penalty,
        )
        candidates = {"equal_weight": _equal_weights(holding_columns), **candidates}

        if include_ai_model:
            try:
                base_weights = candidates["resilient"]
                training_portfolio = calculate_portfolio_returns(train_holdings, base_weights)
                signals, errors = run_optional_models(
                    training_portfolio,
                    holding_columns,
                    run_xgboost=True,
                )
                if "xgboost" in signals:
                    candidates["ai_xgboost"] = adjust_weights_with_signals(
                        base_weights,
                        candidates["minimum_volatility"],
                        signals,
                        "moderate",
                        max(maximum_weight, 1.0 / len(holding_columns)),
                    )
                model_errors.extend(str(error) for error in errors.values())
            except Exception as error:  # noqa: BLE001 - preserve other strategies
                model_errors.append(f"AI walk-forward window {test.index[0].date()}: {error}")

        for name, weights in candidates.items():
            gross_returns = calculate_portfolio_returns(test_holdings, weights)
            if gross_returns.empty:
                continue
            turnover = _turnover(previous_weights.get(name), weights)
            net_returns = gross_returns.copy()
            net_returns.iloc[0] -= turnover * transaction_cost_bps / 10_000.0
            return_parts[name].append(net_returns)
            previous_weights[name] = weights
            turnover_events.append(
                {"date": net_returns.index[0], "strategy": name, "turnover": turnover}
            )
            cost_events.append(
                {
                    "date": net_returns.index[0],
                    "strategy": name,
                    "transaction_cost": turnover * transaction_cost_bps / 10_000.0,
                }
            )

        if benchmark_ticker:
            benchmark = test[benchmark_ticker].dropna()
            if not benchmark.empty:
                return_parts["market_benchmark"].append(benchmark.rename("market_benchmark"))

    combined = {}
    for name, parts in return_parts.items():
        if parts:
            combined[name] = pd.concat(parts).sort_index()
    if not combined:
        raise ValueError("No complete out-of-sample windows were available")

    net_returns = pd.DataFrame(combined).sort_index()
    turnover = (
        pd.DataFrame(turnover_events).pivot(index="date", columns="strategy", values="turnover").sort_index()
        if turnover_events
        else pd.DataFrame()
    )
    transaction_costs = (
        pd.DataFrame(cost_events)
        .pivot(index="date", columns="strategy", values="transaction_cost")
        .sort_index()
        if cost_events
        else pd.DataFrame()
    )
    metrics_rows = []
    for name in net_returns:
        metrics = calculate_metrics(net_returns[name])
        metrics_rows.append(
            {
                "strategy": name,
                "observations": metrics.observations,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "annualised_volatility": metrics.annualised_volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "beta": metrics.beta,
                "maximum_drawdown": metrics.maximum_drawdown,
            }
        )
    return WalkForwardResult(
        net_returns=net_returns,
        metrics=pd.DataFrame(metrics_rows).set_index("strategy"),
        turnover=turnover,
        transaction_costs=transaction_costs,
        model_errors=model_errors,
    )
