# Module 3: Portfolio Returns and Core Analytics

## Problem being solved

The raw price table is not yet a portfolio analysis. This module converts prices into asset returns, combines them using the user's weights and creates transparent baseline performance metrics.

## Metrics included

- Total return
- CAGR
- Annualised volatility
- Sharpe ratio
- Sortino ratio
- Beta relative to NIFTY 50
- Maximum drawdown

The initial implementation uses 252 trading periods per year. The risk-free rate defaults to zero until a sourced Indian risk-free-rate series is added. This assumption is reported rather than hidden.

## Methodological choices

- Simple percentage returns are used initially.
- Missing asset returns are not filled with invented values.
- Portfolio returns use complete observations for all holdings.
- Volatility uses sample standard deviation annualised by the square root of 252.
- Beta is covariance with the benchmark divided by benchmark variance.
- Maximum drawdown is calculated from the cumulative wealth curve.

## Run the module

From the `PRISM-X` folder with `(.venv)` active:

```powershell
python scripts/calculate_analytics.py
```

Outputs are saved to:

- `reports/tables/portfolio_metrics.json`
- `data/processed/portfolio_returns.csv`

These metrics are descriptive historical results, not forecasts or guarantees.
