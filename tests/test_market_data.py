import pandas as pd
import pytest

from prismx.data import (
    MarketDataError,
    convert_prices_to_inr,
    infer_currency,
    normalise_currency,
    resolve_ticker,
    validate_price_frame,
)


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


def test_currency_metadata_normalises_and_infers_common_markets():
    assert normalise_currency("GBp") == "GBP"
    assert infer_currency("AAPL") == "USD"
    assert infer_currency("VOD.L") == "GBP"
    assert infer_currency("TCS.NS") == "INR"


def test_foreign_prices_are_converted_to_inr_using_aligned_fx(monkeypatch):
    import prismx.data.market as market

    index = pd.date_range("2025-01-01", periods=3, freq="B")
    prices = pd.DataFrame(
        {"AAPL": [100.0, 110.0, 120.0], "TCS.NS": [4000.0, 4100.0, 4200.0]},
        index=index,
    )

    def fake_download(tickers, start, end):
        assert tickers == ["INR=X"]
        return pd.DataFrame({"INR=X": [80.0, 82.0, 81.0]}, index=index)

    monkeypatch.setattr(market, "download_adjusted_close", fake_download)
    converted, latest_rates = convert_prices_to_inr(
        prices,
        {"AAPL": "USD", "TCS.NS": "INR"},
        start=index[0].date(),
        end=index[-1].date(),
    )

    assert converted["AAPL"].tolist() == [8000.0, 9020.0, 9720.0]
    assert converted["TCS.NS"].tolist() == [4000.0, 4100.0, 4200.0]
    assert latest_rates == {"AAPL": 81.0, "TCS.NS": 1.0}
