"""Calculate baseline portfolio analytics from the saved price history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismx.analytics import (
    calculate_asset_returns,
    calculate_metrics,
    calculate_portfolio_returns,
    load_price_history,
    portfolio_weights_from_input,
)
from prismx.analytics.returns import benchmark_returns_from_frame
from prismx.portfolio import load_portfolio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/portfolio.example.yaml")
    parser.add_argument(
        "--prices",
        default="data/raw/prices/portfolio_adjusted_close.csv",
    )
    parser.add_argument(
        "--metrics-output",
        default="reports/tables/portfolio_metrics.json",
    )
    parser.add_argument(
        "--returns-output",
        default="data/processed/portfolio_returns.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    portfolio = load_portfolio_config(args.config)
    prices = load_price_history(args.prices)
    returns = calculate_asset_returns(prices)
    weights = portfolio_weights_from_input(portfolio)
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    benchmark_returns = benchmark_returns_from_frame(returns)
    metrics = calculate_metrics(portfolio_returns, benchmark_returns)

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics.as_dict(), indent=2), encoding="utf-8")

    returns_path = Path(args.returns_output)
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_returns.to_csv(returns_path, header=True)

    print("Portfolio analytics completed.")
    print(f"Observations: {metrics.observations}")
    print(f"Total return: {metrics.total_return:.2%}")
    print(f"CAGR: {metrics.cagr:.2%}" if metrics.cagr is not None else "CAGR: n/a")
    print(
        f"Annualised volatility: {metrics.annualised_volatility:.2%}"
        if metrics.annualised_volatility is not None
        else "Annualised volatility: n/a"
    )
    print(
        f"Sharpe ratio: {metrics.sharpe_ratio:.3f}"
        if metrics.sharpe_ratio is not None
        else "Sharpe ratio: n/a"
    )
    print(
        f"Sortino ratio: {metrics.sortino_ratio:.3f}"
        if metrics.sortino_ratio is not None
        else "Sortino ratio: n/a"
    )
    print(f"Beta: {metrics.beta:.3f}" if metrics.beta is not None else "Beta: n/a")
    print(f"Maximum drawdown: {metrics.maximum_drawdown:.2%}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Portfolio returns saved to: {returns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
