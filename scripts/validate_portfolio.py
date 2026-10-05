"""Validate a PRISM-X portfolio configuration from the command line."""

from __future__ import annotations

import sys
from pathlib import Path

from prismx.portfolio import holding_values, load_portfolio_config


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/portfolio.example.yaml")

    try:
        portfolio = load_portfolio_config(config_path)
    except Exception as error:  # noqa: BLE001 - present a simple CLI error to the learner
        print("Portfolio validation failed:")
        print(error)
        return 1

    print("Portfolio is valid.")
    print(f"Portfolio value: INR {portfolio.value_inr:,.2f}")
    print(f"Risk appetite: {portfolio.risk_appetite.value}")
    print("Holdings:")
    for name, value in holding_values(portfolio).items():
        print(f"  - {name}: INR {value:,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
