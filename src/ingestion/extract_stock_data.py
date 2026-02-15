from logging import warning
import warnings
import os
import json
import csv
from datetime import datetime
import warnings
from dateutil.relativedelta import relativedelta, MO
import pandas as pd
import io
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

# --- Suppression of yfinance/Pandas 4.0 Warnings ---
warnings.filterwarnings("ignore")

import polars as pl
import yfinance as yf

# --- Configuration ---
BASE_DIR = "data/stocks"
CSV_DIR = os.path.join(BASE_DIR, "csv_files")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")
META_FILE = os.path.join(BASE_DIR, "metadata.json")
LOG_FILE = os.path.join("logs", "execution_log.csv")
SOURCE_FILE = "data/trades/congress_trades_full.parquet"
TICKER_MAPPING_FILE = "data/ticker_mapping.csv"
PROFESSOR_DELISTED_FILE = "data/congress_trades_professor/delisted_tickers_research.csv"
JARVIS_TICKERS_FILE = "data/congress_trades_professor/jarvis_available_tickers.csv"

YFINANCE_COMPATIBLE_SUFFIXES = {
    ".AX", ".L", ".TO", ".V", ".NS", ".PA", ".MI", ".F",
    ".BK", ".AS", ".ST", ".TA", ".JK", ".KL",
}


def resolve_ticker(ticker, mapping):
    """Resolve a ticker to its Yahoo Finance downloadable symbol.

    Priority: 1) explicit mapping (our CSV + professor's research)
              2) preferred share fix ($ -> -P)
              3) original ticker as-is
    Returns (download_ticker, was_remapped).
    """
    if ticker in mapping:
        return mapping[ticker], True

    # Preferred shares: Yahoo Finance uses -P instead of $
    if "$" in ticker:
        return ticker.replace("$", "-P"), True

    # Share classes: Yahoo Finance uses - instead of . (e.g. BRK.A -> BRK-A)
    if "." in ticker:
        return ticker.replace(".", "-"), True

    return ticker, False


def load_ticker_mapping():
    """Build unified ticker mapping from all sources."""
    mapping = {}

    # 1. Professor's delisted research (load first, lower priority)
    if os.path.exists(PROFESSOR_DELISTED_FILE):
        with open(PROFESSOR_DELISTED_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig = row.get("original_ticker", "").strip()
                new = row.get("new_ticker", "").strip()
                status = row.get("status", "").strip()
                if not orig or not new:
                    continue
                skip_values = ("none", "n/a", "otcmkts", "private", "unknown", "")
                if new.lower() in skip_values:
                    continue
                if new.endswith("Q") and "BANKRUPT" in status.upper():
                    continue
                mapping[orig] = new

    # 2. Our curated mapping (overrides professor's if both exist)
    if os.path.exists(TICKER_MAPPING_FILE):
        with open(TICKER_MAPPING_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                orig = row.get("original_ticker", "").strip()
                new = row.get("mapped_ticker", "").strip()
                if orig and new:
                    mapping[orig] = new

    return mapping


def load_jarvis_fallbacks():
    """Load international exchange alternatives from JARVIS dataset.

    Returns dict: ticker -> [list of yfinance-compatible tickers], sorted by
    data_points descending (most liquid first).
    """
    fallbacks = {}
    if not os.path.exists(JARVIS_TICKERS_FILE):
        return fallbacks

    with open(JARVIS_TICKERS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get("TICKER", "").strip()
            ric = row.get("RIC", "").strip()
            activity = row.get("ACTIVITY", "").strip()
            data_points = int(row.get("DATA_POINTS", 0) or 0)

            if not ticker or not ric or activity != "Active":
                continue

            # Extract the suffix from the RIC (e.g., ".AX" from "ABB.AX")
            dot_pos = ric.rfind(".")
            if dot_pos == -1:
                continue
            suffix = ric[dot_pos:]
            prefix = ric[:dot_pos]

            # Skip .O (NASDAQ) and .N (NYSE) - already tried as bare US ticker
            if suffix in (".O", ".N"):
                continue

            # Only keep yfinance-compatible exchange suffixes
            if suffix not in YFINANCE_COMPATIBLE_SUFFIXES:
                continue

            # Prefix must match the ticker (avoid different companies with same symbol)
            if prefix != ticker:
                continue

            # Build yfinance ticker: use the RIC as-is (e.g., ABB.AX)
            yf_ticker = ric
            fallbacks.setdefault(ticker, []).append((yf_ticker, data_points))

    # Sort each ticker's fallbacks by data_points descending
    for ticker in fallbacks:
        fallbacks[ticker] = [t for t, _ in sorted(fallbacks[ticker], key=lambda x: -x[1])]

    return fallbacks


def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            return json.load(f)
    return {"tickers": {}}

def save_metadata(metadata):
    metadata["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(META_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def append_to_log(ticker, start_date, end_date, status, error=""):
    """Appends exactly one row per API attempt to the CSV log."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Start_Date", "End_Date", "Status", "Error"])
        # We strip newlines from the error to ensure the CSV row integrity
        clean_error = str(error).replace('\n', ' ').replace('\r', '')
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            start_date,
            end_date,
            status,
            clean_error
        ])

def process_ticker_data(max_calls=0):
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Load data and sort alphabetically for deterministic selection
    df = pl.read_parquet(SOURCE_FILE)
    df = df.with_columns([
        pl.col("Ticker").cast(pl.Utf8).fill_null("N/A"),
        pl.col("Traded").str.to_date(strict=False),
        pl.col("Filed").str.to_date(strict=False)
    ]).filter(
    (pl.col("Ticker") != "N/A") &
    ((pl.col("TickerType") == "Stock") | (pl.col("TickerType") == "ST")) #&
    #(pl.col("Ticker") == "AAIGF")
    
    
    )
    

    ticker_ranges = df.group_by("Ticker").agg([
        pl.col("Traded").min().alias("min_traded"),
        pl.col("Filed").max().alias("max_filed")
    ]).sort("Ticker")

    metadata = load_metadata()
    ticker_mapping = load_ticker_mapping()
    jarvis_fallbacks = load_jarvis_fallbacks()
    attempts = 0

    print(f"Processing {len(ticker_ranges)} unique tickers...")
    print(f"Ticker mappings loaded: {len(ticker_mapping)} remappings available")
    print(f"JARVIS fallbacks loaded: {len(jarvis_fallbacks)} tickers with international alternatives")
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for row in tqdm(ticker_ranges.iter_rows(named=True), total=len(ticker_ranges), desc="Downloading stock data"):

        today = pd.Timestamp("today").normalize().date()
        ticker = row["Ticker"]
        download_ticker, _ = resolve_ticker(ticker, ticker_mapping)
        target_start = row["min_traded"] - relativedelta(years=3)  # Go back 3 years from first trade
        target_end = row["max_filed"] + relativedelta(months=12)

        if target_end.weekday() >= 5:
            target_end += relativedelta(weekday=MO)

        target_end = min(target_end, today)
        s_str, e_str = str(target_start), str(target_end)

        # Skip logic
        if ticker in metadata["tickers"]:
            m_min = datetime.strptime(metadata["tickers"][ticker]["min_date"], "%Y-%m-%d").date()
            m_max = datetime.strptime(metadata["tickers"][ticker]["max_date"], "%Y-%m-%d").date()
            if m_min <= target_start and m_max >= target_end:
                attempts += 1
                skipped_count += 1
                continue

        # Limit logic
        if max_calls > 0 and attempts >= max_calls:
            break

        attempts += 1
        
        # Build list of tickers to try: primary first, then JARVIS fallbacks
        tickers_to_try = [download_ticker]
        if ticker in jarvis_fallbacks:
            tickers_to_try.extend(jarvis_fallbacks[ticker])

        downloaded = False
        last_error = ""

        for try_ticker in tickers_to_try:
            try:
                f = io.StringIO()

                with redirect_stdout(f), redirect_stderr(f):
                    data = yf.download(
                        try_ticker,
                        start=target_start,
                        end=target_end + relativedelta(days=1),
                        progress=False,
                        multi_level_index=False
                    )

                yfinance_output = f.getvalue()

                if data.empty:
                    last_error = yfinance_output.strip()
                    continue

                data = data.asfreq('D', method='bfill')

                if data.empty:
                    continue

                stock_pl = pl.from_pandas(data.reset_index()).select([
                    pl.col("Date").cast(pl.Date),
                    pl.col("Close").alias("Close_Price"),
                    pl.col("Open").alias("Open_Price"),
                    pl.col("High").alias("High_Price"),
                    pl.col("Low").alias("Low_Price"),
                ]).with_columns([
                    pl.lit(ticker).alias("OG_Ticker"),
                    pl.lit(try_ticker).alias("Download_Ticker"),
                ])

                # Save under the ORIGINAL ticker name
                stock_pl.write_parquet(os.path.join(PARQUET_DIR, f"{ticker}.parquet"))
                stock_pl.write_csv(os.path.join(CSV_DIR, f"{ticker}.csv"))

                metadata["tickers"][ticker] = {
                    "min_date": str(stock_pl["Date"].min()),
                    "max_date": str(stock_pl["Date"].max())
                }

                append_to_log(ticker, s_str, e_str, "SUCCESS")
                success_count += 1
                downloaded = True
                break

            except Exception as e:
                last_error = str(e)
                continue

        if not downloaded:
            warning = """/Users/ascaravelli/Documents/GitHub/Congress_stock_analysis/venv/lib/python3.14/site-packages/yfinance/scrapers/history.py:201: Pandas4Warning: Timestamp.utcnow is deprecated and will be removed in a future version. Use Timestamp.now('UTC') instead.
  dt_now = pd.Timestamp.utcnow()"""
            append_to_log(ticker, s_str, e_str, "FAILED", last_error.replace(warning, ''))
            failed_count += 1
    
    # Print summary recap
    print("\n" + "="*60)
    print("DOWNLOAD RECAP")
    print("="*60)
    print(f"Total Attempted: {attempts}")
    print(f"Successful:      {success_count} ({success_count/attempts*100:.1f}%)" if attempts > 0 else "Successful:      0")
    print(f"Failed:          {failed_count} ({failed_count/attempts*100:.1f}%)" if attempts > 0 else "Failed:          0")
    print(f"Skipped:         {skipped_count}")
    print("="*60)

    save_metadata(metadata)

if __name__ == "__main__":
    process_ticker_data(max_calls=0)