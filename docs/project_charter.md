# PRISM-X Project Charter

## Purpose

PRISM-X will accept a user-defined portfolio and risk appetite, analyse its historical and forward-looking risks, and recommend portfolio weights that improve the trade-off between return, risk, concentration and stress resilience.

The output is an analytical recommendation for education and research. It is not a guarantee of performance or a substitute for regulated investment advice.

## Version 0.1 assumptions

| Area | Initial decision |
|---|---|
| Asset universe | Indian listed equities |
| Currency | INR |
| Frequency | Daily observations |
| Benchmark | NIFTY 50 |
| History | Up to ten years where available |
| Portfolio | Long-only, no leverage |
| Rebalancing | Monthly |
| Holding limit | 30% per security |
| Sector limit | 50% per sector |
| Severe-risk target | At least 10% loss over the next 20 trading days |
| Reverse-stress target | Find shocks capable of producing a 20% portfolio loss |
| Transaction costs | 25 basis points per one-way turnover, provisional |

These are starting assumptions and will be revised when data availability and initial results are known.

## Model purposes

### Historical analytics

Measure portfolio return, volatility, Sharpe ratio, Sortino ratio, beta, drawdown and concentration.

### Risk contribution

Identify holdings that contribute disproportionately to portfolio volatility or expected shortfall, rather than ranking holdings only by their weights.

### Dependence and network analysis

Study pairwise, rolling and stress-period relationships. A network will be created only after the edge definition and threshold are economically justified.

### Stress testing and simulation

Combine historical episodes, user-defined shocks, parametric or bootstrap Monte Carlo simulation and reverse stress testing.

### Predictive modelling

- XGBoost: estimate the probability of a defined high-risk event.
- LSTM: investigate forecasting of portfolio volatility or risk, not automatically stock prices.
- FinBERT: convert timestamped financial news into sentiment features and test whether they add predictive value.

Every advanced model must be compared with a transparent baseline and evaluated using time-ordered validation.

### Optimisation

The recommendation engine will balance expected return, volatility, expected shortfall, concentration, stress losses and turnover, subject to user constraints and risk appetite.

## Data contract

Every data record must retain:

- Security identifier
- Observation or publication timestamp
- Source and retrieval timestamp
- Currency and unit
- Raw value where possible
- Cleaned value used by the model
- Data-quality status

News features must be aligned using publication time. Information published after an investment decision date must never be used for that decision.

## Success criteria

The first release is successful when it can:

1. Accept a valid portfolio and risk appetite.
2. Reject invalid weights, unknown securities and inconsistent inputs.
3. Reproduce portfolio metrics from stored data.
4. Show risk contribution and stress losses.
5. Produce constrained alternative weights.
6. Compare the original and recommended portfolios through walk-forward backtesting.
7. Explain important assumptions and limitations in plain language.
