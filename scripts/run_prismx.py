"""Run the PRISM-X core analysis pipeline from one command."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from prismx.analytics import (
    calculate_asset_returns,
    calculate_metrics,
    calculate_portfolio_returns,
    load_price_history,
    portfolio_weights_from_input,
)
from prismx.backtesting import backtest_fixed_weights
from prismx.config import load_project_config
from prismx.data import (
    download_adjusted_close,
    resolve_portfolio_tickers,
    resolve_ticker,
    save_price_frame,
)
from prismx.dependence import pairwise_correlation, stress_period_returns
from prismx.evaluation import run_walk_forward_evaluation
from prismx.network import (
    build_correlation_network,
    network_node_metrics,
    network_summary,
)
from prismx.optimisation import optimise_portfolios
from prismx.regime import classify_regimes
from prismx.risk import calculate_risk_metrics, calculate_volatility_risk_contribution
from prismx.simulation import simulate_portfolio, simulation_summary
from prismx.stress import historical_stress_scenarios, reverse_stress_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/portfolio.example.yaml")
    parser.add_argument("--prices", default="data/raw/prices/portfolio_adjusted_close.csv")
    parser.add_argument("--history-years", type=int, default=None)
    parser.add_argument("--network-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def ensure_prices(
    args: argparse.Namespace,
    project_config,
    history_years: int,
) -> tuple[Path, str]:
    path = Path(args.prices)
    benchmark_ticker = resolve_ticker(project_config.market.benchmark)
    required_tickers = list(resolve_portfolio_tickers(project_config.portfolio).values()) + [
        benchmark_ticker
    ]
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=history_years)).date()
    if not args.download and path.exists():
        try:
            cached = load_price_history(path)
            has_tickers = set(required_tickers).issubset(cached.columns)
            has_start = cached.index.min().date() <= start
            has_recent_end = cached.index.max().date() >= (
                pd.Timestamp(end) - pd.Timedelta(7, unit="D")
            ).date()
            if has_tickers and has_start and has_recent_end:
                return path, "cached"
            print("Cached prices do not cover the requested universe or date range; downloading fresh data.")
        except Exception as error:  # noqa: BLE001 - fall back to a fresh download
            print(f"Cached prices could not be loaded ({error}); downloading fresh data.")
    prices = download_adjusted_close(required_tickers, start=start, end=end)
    save_price_frame(prices, path)
    return path, "downloaded"


def main() -> int:
    args = parse_args()
    project_config = load_project_config(args.config)
    portfolio = project_config.portfolio
    history_years = args.history_years or project_config.market.history_years
    price_path, price_source = ensure_prices(args, project_config, history_years)
    prices = load_price_history(price_path)
    asset_returns = calculate_asset_returns(prices)
    weights = portfolio_weights_from_input(portfolio)
    benchmark_ticker = resolve_ticker(project_config.market.benchmark)
    holding_returns = asset_returns.loc[:, list(weights)]
    portfolio_returns = calculate_portfolio_returns(holding_returns, weights)
    benchmark_returns = asset_returns[benchmark_ticker]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    performance_metrics = calculate_metrics(portfolio_returns, benchmark_returns)
    risk_metrics = calculate_risk_metrics(portfolio_returns)
    risk_contribution = calculate_volatility_risk_contribution(holding_returns, weights)
    risk_contribution.to_csv(output_dir / "pipeline_risk_contribution.csv")

    stress = historical_stress_scenarios(holding_returns, weights)
    reverse_stress = reverse_stress_scenario(
        weights,
        target_loss=project_config.risk_targets.reverse_stress_target,
    )
    stress.to_csv(output_dir / "pipeline_historical_stress.csv")
    reverse_stress.to_csv(output_dir / "pipeline_reverse_stress.csv", index=False)

    outcomes = simulate_portfolio(holding_returns, weights, seed=42)
    simulation = simulation_summary(outcomes, horizon_days=252, loss_threshold=-0.20)
    (output_dir / "pipeline_monte_carlo.json").write_text(
        json.dumps(simulation, indent=2), encoding="utf-8"
    )

    regimes, regime_summary = classify_regimes(portfolio_returns)
    regimes.rename("regime").to_csv(output_dir / "pipeline_regimes.csv")
    regime_summary.to_csv(output_dir / "pipeline_regime_summary.csv")

    optimised = optimise_portfolios(
        holding_returns,
        maximum_weight=project_config.constraints.maximum_single_holding,
    )
    optimised_table = pd.DataFrame(optimised)
    optimised_table["original"] = pd.Series(weights)
    optimised_table.to_csv(output_dir / "pipeline_optimised_weights.csv")

    original_series, original_metrics = backtest_fixed_weights(
        holding_returns,
        {"original": weights},
        transaction_cost_bps=project_config.constraints.transaction_cost_bps,
    )
    walk_forward = run_walk_forward_evaluation(
        asset_returns,
        maximum_weight=project_config.constraints.maximum_single_holding,
        benchmark_ticker=benchmark_ticker,
        transaction_cost_bps=project_config.constraints.transaction_cost_bps,
    )
    backtest_series = pd.concat([original_series, walk_forward.net_returns], axis=1)
    backtest_metrics = pd.concat([original_metrics, walk_forward.metrics], axis=0)
    backtest_series.to_csv(output_dir / "pipeline_backtest_returns.csv")
    backtest_metrics.to_csv(output_dir / "pipeline_backtest_metrics.csv")

    normal_corr = pairwise_correlation(asset_returns, weights)
    stress_corr = stress_period_returns(asset_returns, weights).corr()
    normal_network = build_correlation_network(normal_corr, args.network_threshold)
    stress_network = build_correlation_network(stress_corr, args.network_threshold)
    network_node_metrics(normal_network, weights).to_csv(output_dir / "pipeline_network_nodes.csv")
    network_node_metrics(stress_network, weights).to_csv(
        output_dir / "pipeline_stress_network_nodes.csv"
    )
    network_results = {
        "normal": network_summary(normal_network, args.network_threshold).as_dict(),
        "stress": network_summary(stress_network, args.network_threshold).as_dict(),
    }
    (output_dir / "pipeline_network_summary.json").write_text(
        json.dumps(network_results, indent=2), encoding="utf-8"
    )

    summary = {
        "performance_metrics": performance_metrics.as_dict(),
        "risk_metrics": risk_metrics.as_dict(),
        "simulation": simulation,
        "regime_observations": int(len(regimes)),
        "optimisation_methods": list(optimised),
        "price_file": str(price_path),
        "data_provenance": {
            "price_source": price_source,
            "price_start": prices.index.min().date().isoformat(),
            "price_end": prices.index.max().date().isoformat(),
            "benchmark": benchmark_ticker,
        },
        "backtest_methodology": {
            "type": "walk_forward_for_optimised_strategies",
            "train_window_days": 504,
            "test_window_days": 63,
            "transaction_cost_bps": project_config.constraints.transaction_cost_bps,
        },
        "gross_performance_note": "performance_metrics are historical fixed-weight gross metrics; original backtest includes one initial deployment cost, while walk-forward strategies include turnover costs at rebalances.",
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("PRISM-X core pipeline completed.")
    print(f"Portfolio observations: {len(portfolio_returns)}")
    print(f"Historical maximum drawdown: {performance_metrics.maximum_drawdown:.2%}")
    print(f"Monte Carlo probability of loss: {simulation['probability_loss']:.2%}")
    print(f"Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
