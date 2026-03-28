import requests
import polars as pl
import json
import os

# --- Configuration ---
TOKEN_FILE = "config/quiverquant.json"

def load_api_token(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f).get("Authorization Token")

def fetch_and_save_single_file():
    try:
        api_token = load_api_token(TOKEN_FILE)
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        }
        
        # Quick API health check
        print("Checking API connectivity...")
        health_url = "https://api.quiverquant.com/beta/bulk/congresstrading"
        try:
            response = requests.head(health_url, headers=headers, timeout=15)
            if response.status_code == 200:
                print("✓ API is accessible")
            elif response.status_code == 401:
                print("✗ API returned 401 Unauthorized - your API key is invalid or expired")
                raise Exception("Invalid or expired API key. Check config/quiverquant.json")
            elif response.status_code == 403:
                print("✗ API returned 403 Forbidden - you don't have permission to access this endpoint")
                raise Exception("API key doesn't have permission for this endpoint.")
            else:
                print(f"⚠ API returned status {response.status_code}")
        except requests.exceptions.Timeout:
            print("⚠ API health check timed out (15s) - proceeding with full request anyway...")
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to API")
            raise Exception("Cannot connect to API. Check your internet connection.")
        
        # Use the BULK endpoint for all historical data
        url = "https://api.quiverquant.com/beta/bulk/congresstrading"
        
        print(f"\nFetching data from: {url}")
        print("Sending API request... (this may take a moment)")
        response = requests.get(url, headers=headers)
        print("✓ API response received")
        response.raise_for_status()
        
        print("Parsing JSON response...")
        data = response.json()
        
        if not data:
            print("No data returned.")
            return
        
        # Convert to Polars DataFrame
        print(f"Converting {len(data)} records to DataFrame...")
        df = pl.DataFrame(data, infer_schema_length=None)  # Scan all rows for schema
        print(f"✓ Raw data fetched: {df.height} records.")
        
        # Check what columns we have
        print(f"\nColumns: {df.columns}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Parse Traded date column (just for analysis, not filtering)
        print("\nParsing date columns...")
        df = df.with_columns([
            pl.col("Traded").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("TradeDate")
        ])
        print("✓ Dates parsed")
        
        # Check data quality
        null_dates = df.filter(pl.col("TradeDate").is_null()).height
        print(f"\nRows with unparseable/null dates: {null_dates}")
        print(f"Rows with valid dates: {df.filter(pl.col("TradeDate").is_not_null()).height}")
        
        # Show year distribution (for valid dates only)
        if df.filter(pl.col("TradeDate").is_not_null()).height > 0:
            print("\nCalculating year distribution...")
            year_counts = df.filter(pl.col("TradeDate").is_not_null()).group_by(
                pl.col("TradeDate").dt.year().alias("Year")
            ).agg(
                pl.count().alias("Count")
            ).sort("Year")
            print(f"✓ Records by year:")
            print(year_counts)
            
            print(f"\n✓ Date range: {df.filter(pl.col('TradeDate').is_not_null())['TradeDate'].min()} to {df.filter(pl.col('TradeDate').is_not_null())['TradeDate'].max()}")
        
        # Show summary stats
        print(f"\n--- SUMMARY ---")
        print(f"- Total records: {df.height}")
        print(f"- Unique politicians: {df['Name'].n_unique()}")
        print(f"- Unique tickers: {df['Ticker'].n_unique()}")
        print(f"\nTop 10 most active traders:")
        print(df.group_by("Name").agg(pl.count().alias("Trades")).sort("Trades", descending=True).head(10))
        
        # Save to Files
        print("\nSaving data to files...")
        output_dir = "data/trades"
        os.makedirs(output_dir, exist_ok=True)
        
        xlsx_filename = os.path.join(output_dir, "congress_trades_full.xlsx")
        parquet_filename = os.path.join(output_dir, "congress_trades_full.parquet")
        
        print(f"  Writing XLSX ({xlsx_filename})...")
        df.write_excel(xlsx_filename)
        print(f"  ✓ XLSX written")
        
        print(f"  Writing Parquet ({parquet_filename})...")
        df.write_parquet(parquet_filename)
        print(f"  ✓ Parquet written")
        
        print(f"\n✓ SUCCESS: Saved ALL {df.height} records to:")
        print(f" - {xlsx_filename}")
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