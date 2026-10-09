"""Self-contained downloadable HTML reports for interactive analyses."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

import pandas as pd


def _table(frame: pd.DataFrame) -> str:
    """Render a safe, readable HTML table."""

    if frame.empty:
        return "<p>No data available.</p>"
    return frame.to_html(index=True, classes="report-table", border=0, escape=True)


def _percent(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):.{decimals}%}"


def _number(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Not available"
    return f"{float(value):.{decimals}f}"


def _model_rows(result: Any) -> pd.DataFrame:
    """Create a compact evidence table without exposing nested model objects."""

    rows: list[dict[str, object]] = []
    signals = getattr(result, "model_signals", {}) or {}
    if "xgboost" in signals:
        signal = signals["xgboost"]
        metrics = signal.get("metrics", {})
        rows.append(
            {
                "model": "XGBoost",
                "signal": f"Severe-loss probability: {_percent(signal.get('severe_loss_probability'))}",
                "validation evidence": (
                    f"Balanced accuracy {_percent(metrics.get('balanced_accuracy'))}; "
                    f"ROC-AUC {_number(metrics.get('roc_auc'), 3)}; "
                    f"PR-AUC {_number(metrics.get('pr_auc'), 3)}"
                ),
            }
        )
    if "lstm" in signals:
        signal = signals["lstm"]
        metrics = signal.get("test_metrics") or signal.get("validation_metrics") or {}
        rows.append(
            {
                "model": "LSTM",
                "signal": f"Forecast volatility: {_percent(signal.get('forecast_volatility'))}",
                "validation evidence": (
                    f"Test MAE {_number(metrics.get('test_mae', metrics.get('validation_mae')), 4)}; "
                    f"test RMSE {_number(metrics.get('test_rmse', metrics.get('validation_rmse')), 4)}; "
                    f"skill versus baseline {_percent(metrics.get('test_skill', metrics.get('validation_skill')))}"
                ),
            }
        )
    if "finbert" in signals:
        signal = signals["finbert"]
        labelled_accuracy = signal.get("labelled_accuracy")
        rows.append(
            {
                "model": "FinBERT",
                "signal": f"Average news sentiment: {_number(signal.get('overall_sentiment'), 2)}",
                "validation evidence": (
                    f"{signal.get('articles_scored', 0)} articles; "
                    f"confidence {_percent(signal.get('average_confidence'))}; "
                    f"labelled accuracy {_percent(labelled_accuracy) if labelled_accuracy is not None else 'not available'}"
                ),
            }
        )
    return pd.DataFrame(rows, columns=["model", "signal", "validation evidence"])


def _format_frame(frame: pd.DataFrame, percent_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Prepare report tables with human-readable numeric values."""

    formatted = frame.copy()
    for column in percent_columns:
        if column in formatted:
            formatted[column] = formatted[column].map(_percent)
    return formatted


def build_html_report(result: Any) -> str:
    """Build a complete faculty-ready report from an interactive analysis."""

    metrics = result.performance_metrics
    risk = result.risk_metrics
    title = "PRISM-X Portfolio Risk & Resilience Report"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    risk_appetite = str(result.risk_appetite).title()
    method = str(result.recommended_method).replace("_", " ").title()
    benchmark = getattr(result, "benchmark_ticker", "^NSEI")
    data_start = getattr(result, "data_start", "Not recorded")
    data_end = getattr(result, "data_end", "Not recorded")
    resolution = pd.DataFrame(
        [
            {"entered name": name, "resolved ticker": ticker}
            for name, ticker in result.name_to_ticker.items()
        ]
    )

    allocation = _format_frame(result.allocation, ("weight",))
    if "amount_inr" in allocation:
        allocation["amount_inr"] = allocation["amount_inr"].map(
            lambda value: f"INR {float(value):,.0f}"
        )
    candidates = _format_frame(
        result.candidate_metrics,
        ("CAGR", "volatility", "max_drawdown"),
    )
    weight_trace = _format_frame(
        getattr(result, "weight_trace", pd.DataFrame()),
        ("base_optimizer", "after_xgboost", "after_lstm", "after_finbert", "final_constrained"),
    )
    investor_value = _format_frame(
        getattr(result, "investor_value", pd.DataFrame()),
        (
            "CAGR",
            "volatility",
            "max_drawdown",
            "utility_score",
            "utility_gap_to_best",
            "CAGR_vs_benchmark",
            "volatility_vs_benchmark",
            "drawdown_vs_benchmark",
        ),
    )
    risk_table = pd.DataFrame(
        [
            {
                "measure": "Historical one-day VaR",
                "value": _percent(risk.historical_var),
                "interpretation": "Loss magnitude exceeded on approximately 5% of historical days.",
            },
            {
                "measure": "Historical Expected Shortfall",
                "value": _percent(risk.historical_expected_shortfall),
                "interpretation": "Average loss among observations beyond the 95% VaR threshold.",
            },
            {
                "measure": "Parametric one-day VaR",
                "value": _percent(risk.parametric_var),
                "interpretation": "Normal-distribution comparison estimate.",
            },
            {
                "measure": "Beta to benchmark",
                "value": _number(metrics.beta),
                "interpretation": f"Sensitivity relative to {benchmark}.",
            },
        ]
    )

    allocation_top = result.allocation.iloc[0] if not result.allocation.empty else None
    top_holding_text = (
        f"The largest recommended allocation is {allocation_top['stock']} "
        f"({_percent(allocation_top['weight'])})."
        if allocation_top is not None
        else "No allocation was available."
    )
    appetite_explanation = {
        "Conservative": "The recommendation prioritises lower volatility and limits concentration.",
        "Moderate": "The recommendation balances return, volatility and lower-tail resilience.",
        "Aggressive": "The recommendation gives greater emphasis to historical risk-adjusted return.",
    }.get(risk_appetite, "The recommendation follows the selected risk profile.")
    outcome_explanation = (
        f"PRISM-X compared several constrained portfolio objectives and selected the {method} result "
        f"for the {risk_appetite.lower()} risk profile. {appetite_explanation} {top_holding_text} "
        f"The historical CAGR was {_percent(metrics.cagr)}, with annualised volatility "
        f"of {_percent(metrics.annualised_volatility)} and maximum drawdown of "
        f"{_percent(metrics.maximum_drawdown)}. These are historical observations, not forecasts."
    )

    model_table = _model_rows(result)
    model_errors = getattr(result, "model_errors", {}) or {}
    error_text = "".join(
        f"<li>{escape(str(name).title())}: {escape(str(error))}</li>"
        for name, error in model_errors.items()
    )
    model_error_section = (
        f"<p>Models not used or unavailable:</p><ul>{error_text}</ul>" if error_text else ""
    )
    walk_forward_metrics = getattr(result, "walk_forward_metrics", pd.DataFrame())
    walk_forward_error = getattr(result, "walk_forward_error", None)
    walk_forward_section = _table(walk_forward_metrics)
    if walk_forward_error:
        walk_forward_section += f"<p><b>Walk-forward note:</b> {escape(str(walk_forward_error))}</p>"
    concentration = getattr(result, "concentration_metrics", {}) or {}
    concentration_table = pd.DataFrame(
        [
            {"measure": "HHI", "value": _number(concentration.get("hhi"))},
            {
                "measure": "Effective number of holdings",
                "value": _number(concentration.get("effective_number_of_holdings")),
            },
            {
                "measure": "Largest holding",
                "value": _percent(concentration.get("largest_holding_weight")),
            },
            {
                "measure": "Top-three concentration",
                "value": _percent(concentration.get("top_three_weight")),
            },
        ]
    )
    monte_carlo_summary = getattr(result, "monte_carlo_summary", {}) or {}

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; color: #172033; margin: 40px; line-height: 1.45; }}
h1, h2 {{ color: #173b67; }}
h1 {{ border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
.notice {{ background: #eef4fb; padding: 14px; border-left: 4px solid #2b6cb0; }}
.explanation {{ background: #f7f9fc; padding: 16px; border: 1px solid #d7dee8; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
.metric {{ border: 1px solid #d7dee8; padding: 12px; }}
.label {{ color: #56657a; font-size: 12px; }}
.value {{ font-size: 22px; font-weight: bold; }}
.report-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
.report-table th, .report-table td {{ border: 1px solid #d7dee8; padding: 7px; text-align: right; }}
.report-table th:first-child, .report-table td:first-child {{ text-align: left; }}
li {{ margin: 6px 0; }}
footer {{ margin-top: 35px; color: #667085; font-size: 12px; }}
@media print {{ body {{ margin: 15mm; }} }}
</style></head><body>
<h1>{escape(title)}</h1>
<div class="notice">Historical and model-based analysis. This report is not investment advice or a guarantee of returns.</div>
<p><b>Generated:</b> {escape(generated_at)}<br>
<b>Risk appetite:</b> {escape(risk_appetite)}<br>
<b>Selected objective:</b> {escape(method)}<br>
<b>Portfolio amount:</b> INR {result.amount_inr:,.0f}<br>
<b>Benchmark:</b> {escape(str(benchmark))}<br>
<b>Price-data period:</b> {escape(str(data_start))} to {escape(str(data_end))}</p>

<h2>Executive explanation</h2>
<div class="explanation"><p>{escape(outcome_explanation)}</p></div>

<h2>How the recommendation was produced</h2>
<ol>
<li>The entered company names were resolved to market tickers shown below.</li>
<li>Adjusted daily prices were collected and converted into daily returns.</li>
<li>Several long-only, fully invested portfolio objectives were optimised subject to per-security caps.</li>
<li>The selected risk appetite determined which objective was preferred.</li>
<li>Portfolio performance, drawdown, VaR, Expected Shortfall, factor exposures and stress scenarios were calculated.</li>
<li>Optional AI signals were shown only when the selected model and its dependencies/data were available.</li>
</ol>

<h2>Resolved securities</h2>{_table(resolution)}
<h2>Recommended allocation</h2>{_table(allocation)}
<h2>Why these weights?</h2>
<p>The table traces the recommendation from the selected optimiser through each available model adjustment and the final per-security constraint.</p>
{_table(weight_trace)}

<h2>Portfolio ratios and outcome</h2>
<div class="metrics">
<div class="metric"><div class="label">Historical CAGR</div><div class="value">{_percent(metrics.cagr)}</div></div>
<div class="metric"><div class="label">Total return</div><div class="value">{_percent(metrics.total_return)}</div></div>
<div class="metric"><div class="label">Annualised volatility</div><div class="value">{_percent(metrics.annualised_volatility)}</div></div>
<div class="metric"><div class="label">Sharpe ratio</div><div class="value">{_number(metrics.sharpe_ratio)}</div></div>
<div class="metric"><div class="label">Sortino ratio</div><div class="value">{_number(metrics.sortino_ratio)}</div></div>
<div class="metric"><div class="label">Maximum drawdown</div><div class="value">{_percent(metrics.maximum_drawdown)}</div></div>
</div>
<p>{escape(outcome_explanation)}</p>

<h2>Risk analysis</h2>{_table(risk_table)}
<h2>Concentration diagnostics</h2>{_table(concentration_table)}
<h2>Optimisation comparison</h2>{_table(candidates)}
<h2>Investor value comparison</h2>{_table(investor_value)}
<h2>Stress scenarios</h2>{_table(getattr(result, "stress_results", pd.DataFrame()))}
<h2>Reverse stress test</h2>{_table(getattr(result, "reverse_stress", pd.DataFrame()))}
<h2>Factor and market exposure</h2>{_table(getattr(result, "factor_snapshot", pd.DataFrame()))}
<h2>Walk-forward validation</h2>{walk_forward_section}
<h2>AI model evidence</h2>{_table(model_table)}{model_error_section}
<h2>Monte Carlo simulation</h2>
{_table(pd.DataFrame([monte_carlo_summary]) if monte_carlo_summary else pd.DataFrame())}

<h2>Interpretation and limitations</h2>
<ul>
<li>A positive historical ratio does not guarantee a positive future return.</li>
<li>Sharpe and Sortino use a zero annual risk-free assumption unless configured otherwise.</li>
<li>Monte Carlo results use a parametric historical-moment model and may understate fat-tail or regime risk.</li>
<li>FinBERT live sentiment has no ground-truth accuracy unless labelled news is supplied.</li>
<li>XGBoost and LSTM metrics describe their own prediction targets, not total portfolio accuracy.</li>
<li>Taxes, liquidity, market impact, sector constraints and execution slippage are not fully modelled.</li>
</ul>
<footer>Generated by PRISM-X. Review assumptions, data quality, costs and model limitations before using this analysis.</footer>
</body></html>"""
