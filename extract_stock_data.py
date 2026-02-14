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

# --- Suppression of yfinance/Pandas 4.0 Warnings ---
warnings.filterwarnings("ignore")

import polars as pl
import yfinance as yf

# --- Configuration ---
BASE_DIR = "stock_data"
CSV_DIR = os.path.join(BASE_DIR, "csv_files")
PARQUET_DIR = os.path.join(BASE_DIR, "parquet_files")
META_FILE = os.path.join(BASE_DIR, "metadata.json")
LOG_FILE = os.path.join(BASE_DIR, "execution_log.csv")
SOURCE_FILE = "politician_trades_data/congress_trades_full.parquet"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

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
    attempts = 0

    print(f"{'TICKER':<10} | {'STATUS':<10} | {'LOG'}")
    print("-" * 50)
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for row in ticker_ranges.iter_rows(named=True):

        today = pd.Timestamp("today").normalize().date()
        ticker = row["Ticker"]
        target_start = row["min_traded"]
        target_end = row["max_filed"] + relativedelta(months=12)

        if target_end.weekday() >= 5:
            target_end += relativedelta(weekday=MO)

        target_end = min(target_end, today)
        s_str, e_str = str(target_start), str(target_end)
        print(f's_tr: {s_str}, e_str: {e_str}, target_start: {target_start}, target_end: {target_end}')

        # Skip logic
        if ticker in metadata["tickers"]:
            print(f"Checking existing data for {ticker}...")
            m_min = datetime.strptime(metadata["tickers"][ticker]["min_date"], "%Y-%m-%d").date()
            m_max = datetime.strptime(metadata["tickers"][ticker]["max_date"], "%Y-%m-%d").date()
            #print(f"Existing data range: {m_min} to {m_max}")
            #print(f"Target data range:   {target_start} to {target_end}")
            if m_min <= target_start and m_max >= target_end:
                print(f" =====> {ticker:<10} | {'SKIPPED':<10} | Data already up-to-date")
                attempts += 1
                skipped_count += 1
                continue

        # Limit logic
        if max_calls > 0 and attempts >= max_calls:
            break

        attempts += 1
        
        try:
            # The download call
            print('################# START ################')
            print(f"Downloading data for {ticker} from {s_str} to {e_str}")
            
            f = io.StringIO()

            with redirect_stdout(f), redirect_stderr(f):
                data = yf.download(
                    ticker,
                    start=target_start,
                    end=target_end + relativedelta(days=1),
                    progress=False,
                    multi_level_index=False
                )

            yfinance_output = f.getvalue()

            if data.empty:
                warning_message = yfinance_output.strip()
            else:
                warning_message = None
            #data = yf.download(ticker, start=target_start, end=target_end+relativedelta(days=1), multi_level_index=False, progress=False)
            #print(f'data', data)
            data = data.asfreq('D', method='bfill')
            print('################ END #################')
            if not data.empty:
                stock_pl = pl.from_pandas(data.reset_index()).select([
                    pl.col("Date").cast(pl.Date),
                    pl.col("Close").alias("Close_Price"),
                    pl.col("Open").alias("Open_Price"),
                    pl.col("High").alias("High_Price"),
                    pl.col("Low").alias("Low_Price"),
                ])
                
                # Save data
                stock_pl.write_parquet(os.path.join(PARQUET_DIR, f"{ticker}.parquet"))
                stock_pl.write_csv(os.path.join(CSV_DIR, f"{ticker}.csv"))

                # Update metadata
                metadata["tickers"][ticker] = {
                    "min_date": str(stock_pl["Date"].min()),
                    "max_date": str(stock_pl["Date"].max())
                }

                append_to_log(ticker, s_str, e_str, "SUCCESS")
                print(f"{ticker:<10} | {'SUCCESS':<10} | Data saved")
                success_count += 1
            else:
                # API returned nothing (symbol might be delisted or invalid)
                warning = """/Users/ascaravelli/Documents/GitHub/Congress_stock_analysis/venv/lib/python3.14/site-packages/yfinance/scrapers/history.py:201: Pandas4Warning: Timestamp.utcnow is deprecated and will be removed in a future version. Use Timestamp.now('UTC') instead.
  dt_now = pd.Timestamp.utcnow()"""
                append_to_log(ticker, s_str, e_str, "FAILED", warning_message.replace(warning,''))
                print(f"{ticker:<10} | {'FAILED':<10} | {warning_message.replace(warning,'')}")
                failed_count += 1

        except Exception as e:
            # Capture the exact exception from the API/Python
            exact_error = str(e)
            append_to_log(ticker, s_str, e_str, "FAILED", exact_error)
            print(f"{ticker:<10} | {'ERROR':<10} | {exact_error[:30]}...")
            failed_count += 1
    print("-" * 50)
    print(f"Total Attempts: {attempts}, Successes: {success_count}, Skipped: {skipped_count}, Failures: {failed_count}")

    save_metadata(metadata)

if __name__ == "__main__":
    process_ticker_data(max_calls=0)