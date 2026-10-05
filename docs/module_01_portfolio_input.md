# Module 1: Portfolio Input and Validation

## Problem being solved

Before PRISM-X can collect market data or run models, it must know that the portfolio itself is valid. Invalid weights, duplicate holdings or an unknown risk appetite would otherwise contaminate every downstream result.

## Method

The module uses typed Pydantic models to validate the portfolio configuration. The first version checks:

- Positive portfolio value
- Supported risk appetite: conservative, moderate or aggressive
- Non-empty holding names
- Holding weights between 0% and 100%
- No duplicate holding names, ignoring capitalisation
- Weights sum to 100%

## Required input

The input follows the `portfolio` section of `config/portfolio.example.yaml`. Weights are decimals: `0.20` means 20%.

## Expected output

A validated `PortfolioInput` object that later modules can safely use. The command-line check also prints the INR value assigned to every holding.

## Assumptions and limitations

- A user-entered name is accepted at this stage but is not yet resolved to an exchange ticker.
- Short selling and leverage are not supported in the first version.
- A 100% weight total is required.
- Security-name and ticker resolution will be added in the data-ingestion module.

## Run the module

From the `PRISM-X` folder, with the virtual environment active:

```powershell
python scripts/validate_portfolio.py
```

To validate another YAML file:

```powershell
python scripts/validate_portfolio.py path\to\your_portfolio.yaml
```
