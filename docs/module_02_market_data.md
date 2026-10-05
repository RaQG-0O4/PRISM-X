# Module 2: Historical Market Data

## Problem being solved

Portfolio analytics require consistently identified securities and a common historical price table. This module converts user-entered names into exchange tickers, downloads adjusted daily prices and records basic data-quality information.

## Method

The first prototype uses `yfinance` as a replaceable market-data provider. Adjusted daily closing prices are requested for each holding and the NIFTY 50 benchmark. Adjusted prices are used because stock splits and distributions can otherwise create artificial return jumps.

The provider is isolated in `prismx.data.market`, so a licensed or institutional data source can later be substituted without rewriting the analytics modules.

## Expected output

The downloader saves a CSV price table with:

- One row per trading date
- One column per security
- One benchmark column: `^NSEI`
- Adjusted closing prices
- A quality summary containing date range, row count and missing observations

## Important limitations

- The initial name-to-ticker map is intentionally transparent and small.
- Unknown securities must temporarily be supplied with an explicit ticker.
- `yfinance` is suitable for a prototype and academic exploration, not a guarantee of institutional-grade data availability.
- Market holidays are normal missing calendar dates and are not filled with invented prices.
- The downloaded raw file must be preserved so analysis can be reproduced.

## Run the downloader

From the `PRISM-X` folder, with `(.venv)` active:

```powershell
python scripts/download_market_data.py
```

To request a different history length:

```powershell
python scripts/download_market_data.py --years 10
```
