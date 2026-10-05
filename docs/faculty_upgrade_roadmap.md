# PRISM-X faculty-ready upgrade roadmap

## Research framing

PRISM-X is an explainable, risk-aware portfolio recommendation system. It is
not presented as a perfect-return predictor. Every recommendation is evaluated
against historical baselines, unseen periods, transaction costs and stress
scenarios.

## Implemented in the interactive dashboard

- Name-to-ticker resolution and automatic market-data collection.
- Risk-appetite-specific constrained optimisation.
- XGBoost severe-loss probability with holdout metrics.
- FinBERT sentiment with optional uploaded historical or labelled news.
- LSTM volatility forecast with purged chronological test error and baseline skill.
- Walk-forward out-of-sample comparison against equal-weight and NIFTY 50.
- Hypothetical, historical and reverse stress tests.
- Momentum, volatility, beta, correlation and drawdown factor snapshot.
- Optional macro features: India VIX, USD/INR, crude, gold and US 10Y yield.
- SHAP or feature-importance explanations for the risk model.
- Printable HTML report that can be saved as PDF from the browser.

## Recommended empirical study

Compare the following strategies on the same out-of-sample windows:

1. Equal-weight portfolio.
2. Minimum-volatility portfolio.
3. Maximum-Sharpe portfolio.
4. Resilient portfolio.
5. Resilient portfolio plus XGBoost.
6. Resilient portfolio plus FinBERT.
7. Full PRISM-X model.

Report CAGR, volatility, Sharpe, Sortino, maximum drawdown, Expected Shortfall,
turnover and transaction-cost-adjusted returns. Include an ablation discussion
of which component actually improved results.

## Known limitations to disclose

- Public market-data providers may have missing observations or delayed news.
- Live FinBERT headlines do not provide ground-truth labels, so live accuracy
  cannot be claimed without a labelled news dataset.
- LSTM forecasts volatility rather than exact prices.
- Historical backtests do not guarantee future performance.
- Fundamental quality, sector classifications and licensed historical news are
  natural next extensions for a production version.
- The configured sector cap is currently documented but not enforced because
  sector metadata is not yet part of the data contract.
