"""Calculate portfolio VaR, Expected Shortfall and risk contribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prismx.analytics import (
    calculate_asset_returns,
    calculate_portfolio_returns,
    load_price_history,
    portfolio_weights_from_input,
)
from prismx.portfolio import load_portfolio_config
from prismx.risk import (
    calculate_risk_metrics,
    calculate_tail_risk_contribution,
    calculate_volatility_risk_contribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/portfolio.example.yaml")
    parser.add_argument(
        "--prices",
        default="data/raw/prices/portfolio_adjusted_close.csv",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-dir", default="reports/tables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    portfolio = load_portfolio_config(args.config)
    prices = load_price_history(args.prices)
    asset_returns = calculate_asset_returns(prices)
    weights = portfolio_weights_from_input(portfolio)
    portfolio_returns = calculate_portfolio_returns(asset_returns, weights)

    risk_metrics = calculate_risk_metrics(portfolio_returns, confidence=args.confidence)
    volatility_contribution = calculate_volatility_risk_contribution(asset_returns, weights)
    tail_contribution = calculate_tail_risk_contribution(
        asset_returns,
        weights,
        confidence=args.confidence,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "risk_metrics.json").write_text(
        json.dumps(risk_metrics.as_dict(), indent=2),
        encoding="utf-8",
    )
    volatility_contribution.to_csv(output_dir / "volatility_risk_contribution.csv")
    tail_contribution.to_csv(output_dir / "tail_risk_contribution.csv")

    print("Risk analysis completed.")
    print(f"Confidence level: {args.confidence:.0%}")
    print(f"Historical VaR: {risk_metrics.historical_var:.2%}")
    print(
        f"Historical Expected Shortfall: "
        f"{risk_metrics.historical_expected_shortfall:.2%}"
    )
    print(
        f"Parametric VaR: {risk_metrics.parametric_var:.2%}"
        if risk_metrics.parametric_var is not None
        else "Parametric VaR: n/a"
    )
    print("Top volatility risk contributors:")
    for ticker, row in volatility_contribution.head(3).iterrows():
        print(f"  - {ticker}: {row['contribution_pct']:.2%}")
    print(f"Reports saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
