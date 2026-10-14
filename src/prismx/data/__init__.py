"""Market, macroeconomic and financial-news data interfaces."""

from .market import (
    KNOWN_INDIAN_TICKERS,
    FX_TICKERS_TO_INR,
    MarketDataError,
    clean_price_frame,
    convert_prices_to_inr,
    download_adjusted_close,
    download_adjusted_close_with_currency,
    infer_currency,
    normalise_currency,
    resolve_portfolio_tickers,
    resolve_ticker,
    search_ticker_online,
    save_price_frame,
    validate_price_frame,
)
from .macro import DEFAULT_MACRO_TICKERS, collect_macro_returns

__all__ = [
    "KNOWN_INDIAN_TICKERS",
    "FX_TICKERS_TO_INR",
    "DEFAULT_MACRO_TICKERS",
    "MarketDataError",
    "clean_price_frame",
    "collect_macro_returns",
    "convert_prices_to_inr",
    "download_adjusted_close",
    "download_adjusted_close_with_currency",
    "infer_currency",
    "normalise_currency",
    "resolve_portfolio_tickers",
    "resolve_ticker",
    "search_ticker_online",
    "save_price_frame",
    "validate_price_frame",
]
