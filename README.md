# PRISM-X

PRISM-X (Portfolio Risk & Stress Testing Engine) is an explainable portfolio analytics and resilience platform.

The intended user workflow is:

1. Enter stock names, portfolio value and risk appetite.
2. Resolve and validate the securities.
3. Collect historical market data and timestamped financial news.
4. Calculate portfolio risk and return characteristics.
5. Run stress tests, simulations, regime analysis and predictive models.
6. Recommend portfolio weights subject to the investor's constraints.
7. Explain the recommendation and compare it with the original portfolio.

## Current status

The repository contains a working research prototype with:

- yfinance historical adjusted-close data and optional macro/news collection;
- name-to-ticker resolution for the configured Indian-equity universe;
- long-only minimum-volatility, maximum-Sharpe, risk-parity and resilient optimisers;
- shrinkage-covariance robust minimum-volatility diagnostics;
- historical risk, stress testing, Monte Carlo simulation and correlation-network analysis;
- chronological walk-forward comparison with turnover costs;
- investor utility comparisons, benchmark gaps, concentration diagnostics and a
  transparent optimiser-to-model weight trace;
- optional XGBoost, LSTM and FinBERT signals with model-specific validation metrics;
- a Streamlit dashboard and printable HTML faculty report.

The machine-learning and news components remain optional and should be described as
research prototypes: live FinBERT sentiment is not historical out-of-sample evidence,
and the LSTM/XGBoost scores do not guarantee portfolio returns. When XGBoost is
selected for walk-forward evaluation, PRISM-X retrains it inside each historical
training window before applying the signal to the next unseen period.

## Initial scope

- Indian equities
- INR reporting
- Daily observations
- NIFTY 50 as the initial benchmark
- Long-only portfolios
- Monthly rebalancing
- No leverage
- Five to ten years of history where available
- Explainable outputs suitable for a finance-student viva

The saved core-pipeline comparison uses a buy-and-hold original portfolio with one
initial deployment cost and separately evaluates optimised strategies using rolling
walk-forward windows. It does not claim to simulate every real-world trade, tax or
slippage effect.

See [`docs/project_charter.md`](docs/project_charter.md) and [`config/portfolio.example.yaml`](config/portfolio.example.yaml) for the initial assumptions.

## Important limitation

PRISM-X will produce a risk-aware recommendation, not a guaranteed or universally perfect portfolio. Results depend on data quality, assumptions, model uncertainty and the selected constraints.
