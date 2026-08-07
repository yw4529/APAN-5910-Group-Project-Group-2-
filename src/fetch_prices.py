"""
Pulls historical daily price data for a small set of tickers via yfinance.
"""

import os
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]
START_DATE = "2023-01-01"
END_DATE = "2026-08-07"
OUT_DIR = "data/prices"


def fetch_price_data():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ticker in TICKERS:
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
        df.to_csv(out_path)
        print(f"Saved {ticker}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    fetch_price_data()
