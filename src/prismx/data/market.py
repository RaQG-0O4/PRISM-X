"""Market-data resolution, download and quality checks."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from prismx.portfolio import PortfolioInput


KNOWN_INDIAN_TICKERS: dict[str, str] = {
    "hdfc bank": "HDFCBANK.NS",
    "hdfc bank limited": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "icici bank limited": "ICICIBANK.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "reliance industries limited": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS",
    "infosys limited": "INFY.NS",
    "l&t": "LT.NS",
    "larsen & toubro": "LT.NS",
    "larsen and toubro": "LT.NS",
    "nifty 50": "^NSEI",
    "nifty": "^NSEI",
}


class MarketDataError(RuntimeError):
    """Raised when requested market data cannot be downloaded safely."""


def _normalise_name(name: str) -> str:
    value = name.casefold().strip()
    value = re.sub(r"[^a-z0-9&]+", " ", value)
    return " ".join(value.split())


def resolve_ticker(name: str, explicit_ticker: str | None = None) -> str:
    """Resolve a user-entered name to a Yahoo Finance ticker.

    Explicit tickers take precedence. The transparent local map is used first;
    the interactive dashboard can opt into provider-backed name search.
    """

    if explicit_ticker:
        return explicit_ticker.strip().upper()

    stripped_name = name.strip().upper()
    if stripped_name.startswith("^") or stripped_name.endswith((".NS", ".BO")):
        return stripped_name

    normalised_name = _normalise_name(name)
    try:
        return KNOWN_INDIAN_TICKERS[normalised_name]
    except KeyError as error:
        raise MarketDataError(
            f"No ticker mapping found for '{name}'. Add an explicit ticker to the holding."
        ) from error


def resolve_portfolio_tickers(portfolio: PortfolioInput) -> dict[str, str]:
    """Resolve every holding name in a validated portfolio."""

    return {
        holding.name: resolve_ticker(holding.name, holding.ticker)
        for holding in portfolio.holdings
    }


def search_ticker_online(name: str) -> str:
    """Resolve a company name using yfinance search results.

    Indian NSE/BSE results are preferred for this project. If no Indian quote
    is returned, the first equity quote is used so global ticker names can also
    be analysed when the data provider supports them.
    """

    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover - depends on environment
        raise MarketDataError("Install yfinance before searching company names") from error

    try:
        search = yf.Search(
            name,
            max_results=10,
            news_count=0,
            lists_count=0,
            enable_fuzzy_query=True,
        )
        quotes = list(getattr(search, "quotes", []) or [])
    except Exception as error:  # noqa: BLE001 - provider-specific failure
        raise MarketDataError(f"Company search failed for '{name}': {error}") from error

    equities = [
        quote
        for quote in quotes
        if str(quote.get("quoteType", "")).upper() in {"EQUITY", "ETF"}
        and quote.get("symbol")
    ]
    if not equities:
        raise MarketDataError(f"No searchable equity was found for '{name}'")

    indian = [
        quote
        for quote in equities
        if str(quote["symbol"]).upper().endswith((".NS", ".BO"))
    ]
    selected = (indian or equities)[0]
    return str(selected["symbol"]).upper()


def clean_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Standardise a downloaded adjusted-close table without inventing prices."""

    if prices.empty:
        raise MarketDataError("The downloaded price table is empty")

    cleaned = prices.copy()
    cleaned.index = pd.to_datetime(cleaned.index).tz_localize(None)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.sort_index()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(how="all")

    if cleaned.empty or cleaned.shape[1] == 0:
        raise MarketDataError("No usable numeric price columns were downloaded")

    cleaned.index.name = "date"
    return cleaned


def validate_price_frame(prices: pd.DataFrame, minimum_observations: int = 30) -> dict[str, object]:
    """Return basic quality information and reject unusable securities."""

    cleaned = clean_price_frame(prices)
    observations = cleaned.notna().sum()
    insufficient = observations[observations < minimum_observations]
    if not insufficient.empty:
        details = ", ".join(f"{ticker}: {count}" for ticker, count in insufficient.items())
        raise MarketDataError(
            f"Insufficient observations (minimum {minimum_observations}): {details}"
        )

    return {
        "rows": len(cleaned),
        "columns": list(cleaned.columns),
        "start": cleaned.index.min().date().isoformat(),
        "end": cleaned.index.max().date().isoformat(),
        "missing_observations": {
            str(ticker): int(count) for ticker, count in cleaned.isna().sum().items()
        },
    }


def download_adjusted_close(
    tickers: Iterable[str],
    start: str | date,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close prices using yfinance.

    The provider is isolated in this function so it can later be replaced by
    a licensed or institutional data source without changing analytics code.
    """

    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover - depends on environment
        raise MarketDataError(
            "yfinance is not installed. Run: python -m pip install -e \".[market-data]\""
        ) from error

    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        raise MarketDataError("At least one ticker is required")

    series_by_ticker: dict[str, pd.Series] = {}
    failures: dict[str, str] = {}

    for ticker in unique_tickers:
        try:
            history = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
            )
            if history.empty or "Close" not in history:
                failures[ticker] = "no adjusted-close observations returned"
                continue

            series = history["Close"].rename(ticker)
            series.index = pd.to_datetime(series.index).tz_localize(None)
            series_by_ticker[ticker] = series
        except Exception as error:  # noqa: BLE001 - report provider failures together
            failures[ticker] = str(error)

    if failures:
        details = "; ".join(f"{ticker}: {reason}" for ticker, reason in failures.items())
        raise MarketDataError(f"Market-data download failed: {details}")

    return clean_price_frame(pd.concat(series_by_ticker.values(), axis=1))


def save_price_frame(prices: pd.DataFrame, path: str | Path) -> Path:
    """Validate and save a price table as CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_prices = clean_price_frame(prices)
    clean_prices.to_csv(output_path)
    return output_path
