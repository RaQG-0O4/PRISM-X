"""Download the example portfolio's adjusted daily close prices."""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from prismx.data import (
    download_adjusted_close,
    resolve_portfolio_tickers,
    resolve_ticker,
    save_price_frame,
    validate_price_frame,
)
from prismx.portfolio import load_portfolio_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/portfolio.example.yaml",
        help="Path to a PRISM-X portfolio YAML file",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of daily history to request",
    )
    parser.add_argument(
        "--output",
        default="data/raw/prices/portfolio_adjusted_close.csv",
        help="CSV path for the downloaded price table",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    portfolio = load_portfolio_config(args.config)
    holding_tickers = resolve_portfolio_tickers(portfolio)
    benchmark_ticker = resolve_ticker("NIFTY 50")
    tickers = list(holding_tickers.values()) + [benchmark_ticker]

    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(years=args.years)).date()

    print("Resolved tickers:")
    for name, ticker in holding_tickers.items():
        print(f"  - {name}: {ticker}")
    print(f"  - NIFTY 50: {benchmark_ticker}")
    print(f"Requesting daily adjusted prices from {start} to {end}...")

    prices = download_adjusted_close(tickers, start=start, end=end)
    quality = validate_price_frame(prices)
    output_path = save_price_frame(prices, args.output)

    print("Market data downloaded successfully.")
    print(f"Rows: {quality['rows']}")
    print(f"Date range: {quality['start']} to {quality['end']}")
    print(f"Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
