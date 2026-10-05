"""Optional macro-market factor collection using the existing market provider."""

from __future__ import annotations

from datetime import date

import pandas as pd

from prismx.data.market import download_adjusted_close


DEFAULT_MACRO_TICKERS: dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "India VIX": "^INDIAVIX",
    "USD/INR": "INR=X",
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "US 10Y Yield": "^TNX",
}


def collect_macro_returns(
    start: str | date,
    end: str | date | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download available macro series and return daily percentage changes."""

    series = {}
    errors: dict[str, str] = {}
    for label, ticker in DEFAULT_MACRO_TICKERS.items():
        try:
            prices = download_adjusted_close([ticker], start=start, end=end)
            returns = prices.sort_index().pct_change(fill_method=None).dropna(how="all").iloc[:, 0]
            series[label] = returns
        except Exception as error:  # noqa: BLE001 - one unavailable factor should not block others
            errors[label] = str(error)
    if not series:
        return pd.DataFrame(), errors
    return pd.DataFrame(series).sort_index(), errors
