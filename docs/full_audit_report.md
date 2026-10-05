# PRISM-X full verification, error audit and correction report

**Audit date:** 5 October 2026  
**Scope:** data, analytics, risk, optimisation, backtesting, simulation, stress testing, machine learning, dashboard, configuration, reporting and tests.

## 1. Executive conclusion

PRISM-X is a working finance-research prototype, not a guaranteed-return
prediction engine. The main analytics pipeline runs successfully on the
available cached five-year dataset, the Streamlit dashboard renders without
exceptions, and 24 automated tests pass.

The audit found and corrected several material issues. The most important was
an accounting error in the old backtest that charged transaction costs almost
every day. That error produced an artificial -94.75% total return and -45.11%
CAGR for the original portfolio. The corrected buy-and-hold comparison is
approximately +16.10% total return and +3.09% CAGR after one initial
deployment cost, consistent with the +16.40% gross historical result.

The project is suitable for a strong academic demonstration when the results
are described as historical, model-based and assumption-dependent. It is not
yet a production investment advisory system.

## 2. Critical errors found and corrected

| Area | Finding | Correction | Status |
|---|---|---|---|
| Reverse stress | The constraint allowed a weighted shock greater than the target, so zero shock could satisfy a -20% loss request. | Reversed the inequality so the weighted shock must be at or below the target. Added a regression test. | Corrected |
| Fixed-weight backtest | `PeriodIndex.shift()` shifted calendar labels rather than row positions. Almost every observation was treated as a rebalance and charged 25 bps. | The fixed-weight helper is explicitly buy-and-hold and charges one initial deployment cost. Actual re-optimisation uses walk-forward turnover. | Corrected |
| Benchmark contamination | NIFTY 50 was included in the core optimisation universe as though it were an investable holding. | Core optimisation, risk and stress calculations now use holdings only; NIFTY is used separately as benchmark. | Corrected |
| Look-ahead risk | A full-history optimiser output could be mistaken for an out-of-sample performance claim. | The core comparison now includes rolling train/test walk-forward evaluation with 504 training days, 63 test days and turnover costs. Full-history weights are labelled descriptive. | Corrected and documented |
| Benchmark beta | The core summary did not pass benchmark returns into the beta calculation. | Benchmark returns are now passed explicitly. The regenerated result gives beta approximately 1.0194. | Corrected |
| XGBoost validation | Plain accuracy could be inflated by the imbalanced severe-loss target. | Added class balancing, chronological train/validation/test separation, validation threshold selection, balanced accuracy, precision, recall, PR-AUC, Brier score and majority baseline. | Corrected |
| LSTM validation | The model used Keras `validation_split` without an untouched test period or purge gap. | Added chronological train/validation/test periods with a five-sequence purge gap and test MAE/RMSE/skill versus a training mean baseline. | Corrected |

## 3. Before-and-after performance evidence

The regenerated core pipeline used the cached data from **5 October 2021 to 1
October 2026**, 1,238 portfolio observations, and NIFTY 50 as the benchmark.

| Measure | Old gross historical report | Old faulty cost backtest | Corrected original backtest |
|---|---:|---:|---:|
| Total return | +16.40% | -94.75% | +16.10% |
| CAGR | +3.14% | -45.11% | +3.09% |
| Annualised volatility | 15.22% | — | 15.23% |
| Maximum drawdown | -22.79% | -94.84% | -22.79% |

The small difference between +16.40% and +16.10% is the single 25-basis-point
initial deployment cost. It is not evidence that the strategy earns a
guaranteed return.

The corrected walk-forward comparison is also saved in
`reports/tables/pipeline_backtest_metrics.csv`. Its test windows contain 693
observations because they begin only after the 504-day training window. The
benchmark has 685 usable observations because of its own missing dates.

## 4. What is currently working

- Name-to-ticker resolution for known Indian securities, with a yfinance search
  fallback for other names.
- Historical adjusted-close collection, cache validation and date/universe
  provenance reporting.
- Daily returns, CAGR, volatility, Sharpe, Sortino, beta and maximum drawdown.
- Historical VaR, Expected Shortfall and volatility risk contribution.
- Long-only, no-leverage portfolio optimisation with per-security weight caps.
- Minimum-volatility, maximum-Sharpe, risk-parity and resilient objectives.
- Hypothetical, historical and reverse stress tests.
- Monte Carlo loss probabilities with an explicit horizon and loss threshold.
- Correlation networks for normal and stress periods, including node metrics.
- Descriptive Gaussian-mixture regime classification.
- Rolling walk-forward strategy comparison with transaction costs.
- Optional XGBoost severe-loss classification, LSTM volatility forecasting and
  FinBERT sentiment scoring.
- Optional macro-factor collection and SHAP/feature-importance explanation.
- Dashboard entry for amount, stock names/tickers and risk appetite. The
  recommendation assigns weights automatically, so no redundant starting
  weights are required.
- Printable HTML faculty report.

## 5. Methodology that must be explained in a viva

### Portfolio analytics

Returns are simple daily percentage returns from adjusted close prices. CAGR
uses the trading-day convention of 252 periods per year. Sharpe uses a default
annual risk-free rate of zero unless a different rate is supplied. This is a
transparent baseline assumption, not a claim that the real risk-free rate is
zero.

### Optimisation

The optimiser is long-only, fully invested and capped per security. It does not
short, borrow or include taxes, bid/ask spreads, market impact or lot sizes.
The resilient objective combines volatility, empirical lower-tail loss and an
expected-return term; it is a research objective, not a universal definition
of resilience.

### Backtesting

The original portfolio comparison is buy-and-hold with one initial deployment
cost. Optimised strategies are evaluated through rolling walk-forward windows:
weights are learned from the previous 504 observations and tested on the next
63 observations. Turnover costs are charged at each strategy rebalance.

### Machine learning

XGBoost predicts a future severe-loss event over a 20-trading-day horizon from
lagged portfolio and optional macro features. LSTM predicts realised volatility,
not price or return. FinBERT scores news language. These model metrics measure
the model-specific target, not the accuracy of the entire portfolio or future
investment returns.

## 6. Remaining major limitations and prototype classifications

1. **No single “total accuracy” exists.** XGBoost classification, LSTM
   regression and FinBERT sentiment have different targets. The dashboard’s
   overall validation score is a transparent composite research score, not a
   probability that the complete portfolio recommendation is correct.
2. **FinBERT live news is not historical out-of-sample evidence.** Current
   headlines have no ground-truth labels and are not timestamp-aligned into the
   walk-forward backtest. Accuracy is shown only when labelled news is supplied.
3. **Optional model signals are not yet included in the walk-forward strategy
   comparison.** They can adjust the interactive recommendation, but their
   economic value still requires a dedicated time-aligned ablation study.
4. **Monte Carlo is parametric.** It uses a multivariate normal distribution
   estimated from historical moments and therefore does not reproduce all
   fat-tail, autocorrelation or regime-switching behaviour.
5. **Sector exposure is not enforced.** The YAML contains a reserved sector-cap
   field, but sector metadata and a sector constraint engine are not implemented.
6. **The configured monthly rebalancing field is documentation for the intended
   policy.** The current walk-forward implementation uses 63-trading-day test
   windows; it is not an order-execution engine.
7. **Data-provider limitations remain.** yfinance can have missing, delayed or
   revised data and is not a substitute for a licensed institutional feed.
8. **No fundamentals are currently used.** Balance-sheet quality, valuation,
   earnings revisions, analyst estimates, liquidity, taxes and investor-specific
   suitability are outside the current model.
9. **Regime classification is descriptive.** The GMM labels historical states by
   volatility after fitting; it is not yet a validated forward regime forecast.

## 7. Tests and verification performed

- `python -m pytest -q`: **24 passed**.
- Python compilation of `src`, `dashboard`, `scripts` and `tests`: passed.
- Streamlit `AppTest` headless render of `dashboard/app.py`: **0 exceptions**.
- Corrected reverse-stress regression: target weighted shock reached.
- Corrected buy-and-hold transaction-cost regression: one initial cost only.
- Monte Carlo metadata and loss-threshold regression.
- Optimiser invalid-input regressions for one asset, infeasible caps and
  non-finite returns.
- Synthetic LSTM smoke test: purged test metrics returned successfully.
- Synthetic XGBoost smoke test: balanced metrics and PR-AUC returned; its
  85.0% accuracy was below the 86.7% majority baseline, showing why accuracy
  alone is unsafe for this target.
- Core pipeline rerun successfully with 1,238 portfolio observations.

## 8. Files changed during the verification/correction work

Core corrections and integration:

- `src/prismx/stress/scenarios.py`
- `src/prismx/backtesting/simple.py`
- `src/prismx/evaluation/walk_forward.py`
- `src/prismx/config/__init__.py`
- `src/prismx/portfolio/input.py`
- `src/prismx/simulation/monte_carlo.py`
- `src/prismx/optimisation/portfolio.py`
- `src/prismx/ml/risk_prediction.py`
- `src/prismx/ml/lstm.py`
- `src/prismx/ml/integration.py`
- `src/prismx/data/news.py`
- `src/prismx/workflow.py`
- `src/prismx/reporting/html_report.py`
- `scripts/run_prismx.py`
- `dashboard/app.py`

Documentation, configuration and tests:

- `README.md`
- `config/portfolio.example.yaml`
- `docs/faculty_upgrade_roadmap.md`
- `tests/test_backtesting.py`
- `tests/test_accelerated_modules.py`
- `tests/test_portfolio_validation.py`
- `tests/test_optimisation_validation.py`
- `docs/full_audit_report.md`

## 9. Recommended next stage

The next credible research stage is not adding more algorithms. It is a
reproducible empirical study: freeze the data cut-off, create timestamp-aligned
historical news and fundamental features, run identical walk-forward windows
for every model combination, include sector and turnover constraints, and
report confidence intervals plus an ablation table. That would show whether
XGBoost, LSTM or FinBERT improves the portfolio rather than merely making the
dashboard look more sophisticated.

## 10. Likely faculty viva questions

1. Why is the optimiser not a guarantee of the best future portfolio?
2. Why must the benchmark be excluded from investable holdings?
3. What caused the original -94.75% backtest result?
4. What is the difference between gross historical performance and net
   transaction-cost-adjusted performance?
5. Why is a chronological split safer than a random train/test split?
6. Why can accuracy be misleading for severe-loss classification?
7. What do balanced accuracy, PR-AUC and Brier score add?
8. Why does the LSTM forecast volatility rather than exact price?
9. Why cannot live FinBERT sentiment be called accurate without labels?
10. What assumptions does the Gaussian Monte Carlo model make?
11. What does beta of approximately 1.02 mean relative to NIFTY 50?
12. What does maximum drawdown measure, and why is it different from VaR?
13. How does risk parity differ from minimum volatility?
14. Why is sector exposure not currently enforced?
15. What would make this suitable for production rather than an academic
    prototype?
