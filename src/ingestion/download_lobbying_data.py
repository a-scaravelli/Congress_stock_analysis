"""src/ingestion/download_lobbying_data.py

Download lobbying data from QuiverQuant for all tickers in the trade universe.
Uses concurrent requests to minimise wall-clock time.

Endpoint: GET /historical/lobbying/{ticker}
Response fields: Date, Amount, Client, Issue (LDA codes, newline-separated),
                 Specific_Issue, Registrant, Ticker

Output: data/lobbying/lobbying_data.parquet
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

TOKEN_FILE = "config/quiverquant.json"
OUTPUT_PATH = "data/lobbying/lobbying_data.parquet"
TICKERS_SOURCE = "data/output/politician_trades_enriched.parquet"
BASE_URL = "https://api.quiverquant.com/beta/historical/lobbying"
MAX_WORKERS = 20   # concurrent threads — stays well within typical rate limits


def load_api_token(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
    with open(filepath, "r") as f:
        return json.load(f).get("Authorization Token")


def get_unique_tickers() -> list:
    df = pd.read_parquet(TICKERS_SOURCE)
    return sorted(df["Ticker"].dropna().unique().tolist())


def fetch_one(ticker: str, session: requests.Session) -> list:
    try:
        resp = session.get(f"{BASE_URL}/{ticker}", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                for rec in data:
                    rec["Ticker"] = ticker
                return data
        return []
    except Exception:
        return []


def main():
    Path("data/lobbying").mkdir(parents=True, exist_ok=True)

    token = load_api_token(TOKEN_FILE)
    tickers = get_unique_tickers()
    n = len(tickers)
    print(f"Fetching lobbying data for {n} tickers using {MAX_WORKERS} parallel workers...")

    all_records = []
    done = 0

    with requests.Session() as session:
        session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(fetch_one, t, session): t for t in tickers}
            for fut in as_completed(futures):
                records = fut.result()
                all_records.extend(records)
                done += 1
                if done % 200 == 0 or done == n:
                    print(f"  [{done}/{n}] {len(all_records):,} records so far")

    if not all_records:
        print("No lobbying data retrieved. Check API key and tier access.")
        return

    df = pd.DataFrame(all_records)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH, index=False)
    n_tickers = df["Ticker"].nunique()
    pct = n_tickers / n * 100
    print(f"\nSaved {len(df):,} lobbying records for {n_tickers:,}/{n} tickers ({pct:.1f}%) → {OUTPUT_PATH}")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
