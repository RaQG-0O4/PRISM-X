# Module 4: Risk Measurement and Risk Contribution

## Problem being solved

Portfolio weights alone do not show where risk comes from. Two holdings with equal weights can contribute very different amounts of risk because of their volatility and relationship with other holdings.

## Risk measures included

- Historical one-day Value at Risk at a selected confidence level
- Historical Expected Shortfall
- Parametric normal VaR for comparison
- Parametric normal Expected Shortfall for comparison
- Volatility risk contribution
- Tail-loss contribution during the worst historical observations

VaR and Expected Shortfall are reported as positive loss magnitudes. For example, `0.025` means a 2.5% loss magnitude.

## Methodological choices

Historical methods use the observed portfolio-return distribution and make fewer distributional assumptions. Parametric methods assume normally distributed returns and are included as a transparent comparison, not as the final answer.

Volatility contribution uses the covariance matrix:

```text
Portfolio volatility = sqrt(weights' × covariance × weights)
```

Each holding's component contribution is calculated using marginal risk multiplied by its portfolio weight. Tail contribution uses the weighted average loss contribution during observations beyond the historical Expected Shortfall threshold.

## Run the module

```powershell
python scripts/calculate_risk.py
```

Outputs:

- `reports/tables/risk_metrics.json`
- `reports/tables/volatility_risk_contribution.csv`
- `reports/tables/tail_risk_contribution.csv`

These are backward-looking risk estimates. They are not guarantees of future losses.
