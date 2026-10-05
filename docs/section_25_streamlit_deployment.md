# 25. Streamlit Deployment and Live Application

## 25.1 Application framework

PRISM-X is delivered through a Streamlit web application. Streamlit was selected because it allows the Python analytics workflow to be exposed through an interactive browser interface without requiring a separate front-end framework. The deployed entry point is `dashboard/app.py`.

The application presents the project as a decision-support tool rather than as a collection of independent scripts. A user can enter a portfolio, request an analysis, review the calculated risk and allocation outputs, and download a complete HTML analysis report from the same interface.

## 25.2 Connection between the interface and the Python backend

The Streamlit interface imports the PRISM-X backend package from `src/prismx`. The backend contains the portfolio input validation, market-data acquisition, return and factor calculations, risk metrics, stress testing, simulation, optimisation, walk-forward evaluation, and optional machine-learning integrations.

When the user selects **Analyse Portfolio**, `dashboard/app.py` converts the entries from the sidebar into Python values and calls the PRISM-X workflow. The workflow then resolves stock names to exchange tickers, downloads or loads historical data, calculates portfolio statistics, evaluates candidate strategies, selects the risk-appetite-specific recommendation, and returns a structured analysis object. Streamlit renders the returned data as metrics, tables, charts, warnings, and downloadable output.

The repository is installed as a local package during deployment using `-e .` in `requirements.txt`. This makes the same `src/prismx` backend available to the dashboard in the cloud environment.

## 25.3 Main user interaction flow

The normal interaction flow is:

1. Open the live PRISM-X application.
2. Expand the left sidebar and enter the portfolio amount in INR.
3. Enter one supported stock name or exchange ticker per line.
4. Select Conservative, Moderate, or Aggressive risk appetite.
5. Select the historical lookback period.
6. Choose any optional analysis modules, such as XGBoost, FinBERT, LSTM, walk-forward validation, transaction-cost modelling, or macro factors.
7. Select **Analyse Portfolio**.
8. Review the resolved securities, recommended weights, rupee allocations, risk measures, optimisation comparison, stress results, model signals, and validation evidence.
9. Download the complete HTML report for submission or further conversion to PDF.

The application displays a disclaimer that results are historical and model-based and are not investment guarantees.

## 25.4 Inputs available to the user

The current dashboard provides the following inputs:

- Portfolio amount in INR.
- Stock names or exchange tickers.
- Risk appetite: Conservative, Moderate, or Aggressive.
- Historical data lookback between three and ten years.
- Optional XGBoost severe-loss-risk model.
- Optional FinBERT financial-news sentiment analysis.
- Optional labelled or historical news CSV upload for sentiment evaluation.
- Optional LSTM volatility forecast.
- Optional walk-forward benchmark comparison.
- Transaction cost assumption in basis points.
- Optional macro-factor collection, including India VIX, USD/INR, crude oil, gold, and the US 10-year yield when available.

The input workflow validates the portfolio amount, risk appetite, ticker resolution, and uploaded-news columns before starting the analysis.

## 25.5 Outputs displayed by the application

After an analysis completes, PRISM-X displays:

- Resolved stock names and exchange tickers.
- Recommended portfolio weights and INR allocations.
- Historical CAGR, volatility, Sharpe ratio, maximum drawdown, and expected shortfall.
- A comparison of minimum-volatility, maximum-Sharpe, risk-parity, resilient, and model-adjusted strategies when available.
- Risk-contribution tables showing marginal volatility, component volatility, and percentage contribution to portfolio risk.
- Out-of-sample walk-forward results and growth charts when selected.
- Hypothetical stress scenarios, historical worst days, and reverse-stress results.
- Factor and market-exposure information, including momentum, volatility, beta, correlation, drawdown, and negative-day measures.
- Macro-factor summaries when macro data is successfully collected.
- XGBoost, FinBERT, and LSTM signals, together with validation evidence and model warnings where an optional model could not run.
- A downloadable HTML report containing the inputs, ticker resolution, allocations, ratios, risk analysis, stress testing, model evidence, methodology, limitations, and interpretation.

## 25.6 Deployment configuration

The application is deployed from the `main` branch of the GitHub repository. The Streamlit Community Cloud configuration uses:

- Repository: `RaQG-0O4/PRISM-X`
- Main file: `dashboard/app.py`
- Dependency file: `requirements.txt`
- Python project definition: `pyproject.toml`
- Backend package: `src/prismx`

The public dependency file includes the Streamlit runtime and the market-data, optimisation, explainability, and network-analysis dependencies required by the deployed application. Heavy AI frameworks remain optional so that the public application can start reliably; when an optional framework or data source is unavailable, PRISM-X reports that limitation in the dashboard rather than silently presenting a model result.

## 25.7 GitHub repository integration

The complete project source code, configuration, documentation, tests, dashboard, and backend modules are maintained in the following GitHub repository:

[PRISM-X GitHub repository](https://github.com/RaQG-0O4/PRISM-X)

The repository provides version control and acts as the source for the Streamlit deployment. Future commits to the configured branch can be used to update the deployed application.

## 25.8 Live application access

**Live Application:** [https://prism-x-vybrrng4ovuuukhxkmubvg.streamlit.app/](https://prism-x-vybrrng4ovuuukhxkmubvg.streamlit.app/)

The link provides direct access to the deployed PRISM-X application through Streamlit Community Cloud. It can be shared with faculty members or evaluators for demonstration and evaluation. No local Python installation is required to open the public application; the user only needs a web browser. The first request after inactivity may take longer because the cloud application may need to wake up before processing the analysis.

