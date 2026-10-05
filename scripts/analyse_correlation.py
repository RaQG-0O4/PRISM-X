"""Analyse normal, rolling and stress-period portfolio correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

from prismx.analytics import calculate_asset_returns, load_price_history, portfolio_weights_from_input
from prismx.dependence import (
    correlation_pairs,
    pairwise_correlation,
    rolling_pairwise_correlation,
    stress_period_returns,
)
from prismx.portfolio import load_portfolio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/portfolio.example.yaml")
    parser.add_argument("--prices", default="data/raw/prices/portfolio_adjusted_close.csv")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--tail-quantile", type=float, default=0.05)
    parser.add_argument("--output-dir", default="reports/tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    portfolio = load_portfolio_config(args.config)
    prices = load_price_history(args.prices)
    asset_returns = calculate_asset_returns(prices)
    weights = portfolio_weights_from_input(portfolio)

    normal_matrix = pairwise_correlation(asset_returns, weights)
    stress_returns = stress_period_returns(
        asset_returns,
        weights,
        tail_quantile=args.tail_quantile,
    )
    stress_matrix = stress_returns.corr()
    rolling = rolling_pairwise_correlation(
        asset_returns,
        weights,
        window=args.window,
    )
    normal_pairs = correlation_pairs(normal_matrix)
    stress_pairs = correlation_pairs(stress_matrix)
    stress_change = stress_matrix - normal_matrix

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_matrix.to_csv(output_dir / "correlation_matrix.csv")
    stress_matrix.to_csv(output_dir / "stress_correlation_matrix.csv")
    stress_change.to_csv(output_dir / "stress_correlation_change.csv")
    normal_pairs.to_csv(output_dir / "correlation_pairs.csv", index=False)
    stress_pairs.to_csv(output_dir / "stress_correlation_pairs.csv", index=False)
    rolling.to_csv(output_dir / "rolling_correlations.csv", index=False)

    print("Correlation analysis completed.")
    print(f"Stress observations: {len(stress_returns)}")
    print(f"Rolling window: {args.window} trading days")
    print("Strongest normal-period relationship:")
    strongest = normal_pairs.iloc[0]
    print(
        f"  - {strongest['asset_1']} / {strongest['asset_2']}: "
        f"{strongest['correlation']:.3f}"
    )
    print("Strongest stress-period relationship:")
    strongest_stress = stress_pairs.iloc[0]
    print(
        f"  - {strongest_stress['asset_1']} / {strongest_stress['asset_2']}: "
        f"{strongest_stress['correlation']:.3f}"
    )
    print(f"Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
