# Module 5: Correlation and Dependence Analysis

## Problem being solved

Diversification depends on relationships between holdings, not only on the number of holdings. Correlations can also increase during market stress, making a portfolio less diversified exactly when losses are severe.

## Views included

- Normal-period pairwise correlation matrix
- Rolling 60-trading-day pairwise correlations
- Worst-5%-day stress-period correlation matrix
- Stress correlation change relative to the normal-period matrix

## Methodological choices

Pearson correlation is the initial transparent baseline. Rolling correlations show whether relationships change over time. Stress days are identified from the portfolio's own worst historical daily returns rather than chosen arbitrarily.

The worst 5% threshold and 60-day window are provisional parameters that can later be tested for sensitivity.

## Run the module

```powershell
python scripts/analyse_correlation.py
```

Outputs:

- `reports/tables/correlation_matrix.csv`
- `reports/tables/stress_correlation_matrix.csv`
- `reports/tables/stress_correlation_change.csv`
- `reports/tables/correlation_pairs.csv`
- `reports/tables/stress_correlation_pairs.csv`
- `reports/tables/rolling_correlations.csv`

These outputs will later feed the network analysis, stress testing and dashboard modules.
