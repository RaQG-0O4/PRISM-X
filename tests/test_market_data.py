import pandas as pd
import pytest

from prismx.data import MarketDataError, resolve_ticker, validate_price_frame


def test_known_indian_names_resolve_to_tickers():
    assert resolve_ticker("HDFC Bank") == "HDFCBANK.NS"
    assert resolve_ticker("Larsen & Toubro") == "LT.NS"
    assert resolve_ticker("NIFTY 50") == "^NSEI"


def test_explicit_ticker_takes_precedence():
    assert resolve_ticker("Some Company", "abc.ns") == "ABC.NS"


def test_unknown_name_requires_explicit_ticker():
    with pytest.raises(MarketDataError, match="No ticker mapping"):
        resolve_ticker("Unknown Company")


def test_price_quality_report_handles_missing_observations():
    index = pd.date_range("2024-01-01", periods=35, freq="B")
    prices = pd.DataFrame(
        {"AAA.NS": range(100, 135), "BBB.NS": [None] * 5 + list(range(200, 230))},
        index=index,
    )

    quality = validate_price_frame(prices)

    assert quality["rows"] == 35
    assert quality["missing_observations"]["BBB.NS"] == 5
