import requests
import polars as pl
import json
import os

# --- Configuration ---
TOKEN_FILE = "quiverquant.json"

def load_api_token(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f).get("Authorization Token")

def fetch_and_save_single_file():
    try:
        api_token = load_api_token(TOKEN_FILE)
        
        # Use the BULK endpoint for all historical data
        url = "https://api.quiverquant.com/beta/bulk/congresstrading"
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
        
        print(f"Fetching data from: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("No data returned.")
            return
        
        # Convert to Polars DataFrame
        df = pl.DataFrame(data)
        print(f"Raw data fetched: {df.height} records.")
        
        # Check what columns we have
        print(f"\nColumns: {df.columns}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Parse Traded date column (just for analysis, not filtering)
        df = df.with_columns([
            pl.col("Traded").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("TradeDate")
        ])
        
        # Check data quality
        null_dates = df.filter(pl.col("TradeDate").is_null()).height
        print(f"\nRows with unparseable/null dates: {null_dates}")
        print(f"Rows with valid dates: {df.filter(pl.col("TradeDate").is_not_null()).height}")
        
        # Show year distribution (for valid dates only)
        if df.filter(pl.col("TradeDate").is_not_null()).height > 0:
            year_counts = df.filter(pl.col("TradeDate").is_not_null()).group_by(
                pl.col("TradeDate").dt.year().alias("Year")
            ).agg(
                pl.count().alias("Count")
            ).sort("Year")
            print(f"\nRecords by year:")
            print(year_counts)
            
            print(f"\nDate range: {df.filter(pl.col('TradeDate').is_not_null())['TradeDate'].min()} to {df.filter(pl.col('TradeDate').is_not_null())['TradeDate'].max()}")
        
        # Show summary stats
        print(f"\n--- SUMMARY ---")
        print(f"- Total records: {df.height}")
        print(f"- Unique politicians: {df['Name'].n_unique()}")
        print(f"- Unique tickers: {df['Ticker'].n_unique()}")
        print(f"\nTop 10 most active traders:")
        print(df.group_by("Name").agg(pl.count().alias("Trades")).sort("Trades", descending=True).head(10))
        
        # Save to Files
        output_dir = "politician_trades_data"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_filename = os.path.join(output_dir, "congress_trades_full.csv")
        parquet_filename = os.path.join(output_dir, "congress_trades_full.parquet")
        
        df.write_csv(csv_filename)
        df.write_parquet(parquet_filename)
        
        print(f"\nSaved ALL {df.height} records to:")
        print(f" - {csv_filename}")
        print(f" - {parquet_filename}")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_and_save_single_file()