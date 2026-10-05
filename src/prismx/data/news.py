"""Financial-news ingestion with a replaceable provider boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


def collect_yfinance_news(tickers: Iterable[str], limit_per_ticker: int = 20) -> pd.DataFrame:
    """Collect available news metadata from yfinance.

    This is a prototype provider. It records the retrieval timestamp and keeps
    the provider-specific parsing isolated for later replacement by a licensed
    news API.
    """

    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Install yfinance before collecting news") from error

    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows = []
    failures: dict[str, str] = {}
    for ticker in dict.fromkeys(tickers):
        try:
            articles = yf.Ticker(ticker).news or []
        except Exception as error:  # noqa: BLE001 - one provider failure should not corrupt other rows
            failures[ticker] = str(error)
            articles = []
        for article in articles[:limit_per_ticker]:
            content = article.get("content", article)
            title = content.get("title") or article.get("title")
            if not title:
                continue
            published = content.get("pubDate") or article.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
            rows.append(
                {
                    "ticker": ticker,
                    "title": str(title),
                    "published_at": published,
                    "url": content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else content.get("link"),
                    "source": content.get("provider", {}).get("displayName")
                    if isinstance(content.get("provider"), dict)
                    else None,
                    "retrieved_at": retrieved_at,
                }
            )
    result = pd.DataFrame(rows)
    result.attrs["collection_errors"] = failures
    return result


def load_news_csv(path: str) -> pd.DataFrame:
    """Load a user-provided news file for reproducible sentiment analysis."""

    news = pd.read_csv(path)
    required = {"title", "published_at"}
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"News file is missing columns: {', '.join(sorted(missing))}")
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    return news.dropna(subset=["published_at", "title"]).reset_index(drop=True)
