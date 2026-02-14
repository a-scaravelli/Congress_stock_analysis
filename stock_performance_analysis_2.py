import polars as pl
import yfinance as yf
from datetime import timedelta, datetime
import time

# --- Configuration ---
INPUT_FILE = "politician_trades_data/congress_trades_full.parquet"
OUTPUT_FILE = "politician_trades_data/stock_performance_analysis.parquet"
FILTER_YEAR = 2023
MAX_ROWS = 50

def get_stock_data_bulk(ticker, start_date, end_date):
    """
    Get all historical data for a ticker in one API call.
    Returns a dataframe with dates and closing prices.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return None
        
        # Convert to simple dataframe with date and close price
        hist_reset = hist.reset_index()
        hist_reset['Date'] = hist_reset['Date'].dt.date
        return hist_reset[['Date', 'Close']].copy()
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def find_closest_price(price_df, target_date):
    """
    Find the closing price on or after the target date.
    """
    if price_df is None:
        return None
    
    # Convert target_date to date if it's datetime
    if isinstance(target_date, datetime):
        target_date = target_date.date()
    
    # Get dates >= target_date
    valid_rows = price_df[price_df['Date'] >= target_date]
    
    if len(valid_rows) == 0:
        return None
    
    return valid_rows.iloc[0]['Close']

def fetch_price_data(ticker, trade_date, filed_date):
    """
    Fetch trade price, filed price, percentage returns for all intervals.
    Returns dict with trade_price, filed_price, traded_to_filed_pct, and return_Xm_pct.
    """
    print(f"  API call for {ticker}...")
    
    # Convert dates to date if needed
    if isinstance(trade_date, datetime):
        trade_date = trade_date.date()
    if isinstance(filed_date, datetime):
        filed_date = filed_date.date()
    
    # Calculate the date range we need (from trade date to filed date + 12 months + buffer)
    start_date = trade_date - timedelta(days=5)
    end_date = filed_date + timedelta(days=365 + 10)
    
    # ONE API CALL to get all the data we need
    price_df = get_stock_data_bulk(ticker, start_date, end_date)
    
    if price_df is None:
        return {
            'trade_price': None,
            'filed_price': None,
            'traded_to_filed_pct': None,
            'return_1m_pct': None,
            'return_3m_pct': None,
            'return_6m_pct': None,
            'return_9m_pct': None,
            'return_12m_pct': None,
        }
    
    # Get price on trade date
    trade_price = find_closest_price(price_df, trade_date)
    
    # Get price on filed date
    filed_price = find_closest_price(price_df, filed_date)
    
    if trade_price is None or filed_price is None:
        return {
            'trade_price': trade_price,
            'filed_price': filed_price,
            'traded_to_filed_pct': None,
            'return_1m_pct': None,
            'return_3m_pct': None,
            'return_6m_pct': None,
            'return_9m_pct': None,
            'return_12m_pct': None,
        }
    
    # Calculate % change from Traded to Filed
    traded_to_filed_pct = ((filed_price - trade_price) / trade_price) * 100
    
    # Calculate all intervals FROM FILED DATE
    intervals = {
        '1m': filed_date + timedelta(days=30),
        '3m': filed_date + timedelta(days=90),
        '6m': filed_date + timedelta(days=180),
        '9m': filed_date + timedelta(days=270),
        '12m': filed_date + timedelta(days=365),
    }
    
    results = {
        'trade_price': trade_price,
        'filed_price': filed_price,
        'traded_to_filed_pct': traded_to_filed_pct
    }
    
    for interval_name, future_date in intervals.items():
        future_price = find_closest_price(price_df, future_date)
        
        if future_price is not None:
            # Calculate return from FILED price
            pct_return = ((future_price - filed_price) / filed_price) * 100
            results[f'return_{interval_name}_pct'] = pct_return
        else:
            results[f'return_{interval_name}_pct'] = None
    
    time.sleep(0.3)  # Rate limiting
    return results

def main():
    total_start = time.time()
    
    print("=" * 70)
    print("STOCK PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    print("\n[1/4] Loading congressional trading data...")
    load_start = time.time()
    df = pl.read_parquet(INPUT_FILE)
    print(f"      ✓ Loaded {df.height:,} records in {time.time() - load_start:.2f}s")
    
    print(f"\n[2/4] Filtering for TickerType='Stock' and year {FILTER_YEAR}...")
    filter_start = time.time()
    
    df_filtered = df.filter(
        (pl.col("TickerType") == "Stock") &
        (pl.col("Traded").str.strptime(pl.Date, "%Y-%m-%d", strict=False).dt.year() == FILTER_YEAR)
    )
    
    print(f"      ✓ Filtered to {df_filtered.height:,} records in {time.time() - filter_start:.2f}s")
    
    # Take first 50 rows
    df_sample = df_filtered.head(MAX_ROWS)
    print(f"      ✓ Processing first {df_sample.height} rows")
    
    # Parse the Traded and Filed dates
    df_sample = df_sample.with_columns([
        pl.col("Traded").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("TradeDate"),
        pl.col("Filed").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("FiledDate")
    ])
    
    print(f"\n[3/4] Fetching stock prices from Yahoo Finance API...")
    api_start = time.time()
    
    # Fetch all price data via API calls
    price_results = []
    for i, row in enumerate(df_sample.iter_rows(named=True)):
        ticker = row['Ticker']
        trade_date = row['TradeDate']
        filed_date = row['FiledDate']
        
        print(f"      [{i+1}/{df_sample.height}] {ticker} - Traded: {trade_date}, Filed: {filed_date}")
        
        if trade_date is None or filed_date is None:
            print(f"          ⚠ Invalid dates, skipping...")
            price_results.append({
                'trade_price': None,
                'filed_price': None,
                'traded_to_filed_pct': None,
                'return_1m_pct': None,
                'return_3m_pct': None,
                'return_6m_pct': None,
                'return_9m_pct': None,
                'return_12m_pct': None,
            })
            continue
        
        perf = fetch_price_data(ticker, trade_date, filed_date)
        price_results.append(perf)
    
    api_time = time.time() - api_start
    print(f"      ✓ Completed {len(price_results)} API calls in {api_time:.2f}s")
    print(f"      ✓ Average time per API call: {api_time/len(price_results):.2f}s")
    
    print(f"\n[4/4] Calculating dollar returns using Polars...")
    calc_start = time.time()
    
    # Convert to Polars DataFrame
    df_prices = pl.DataFrame(price_results)
    
    # Combine original data with price data
    df_combined = pl.concat([df_sample, df_prices], how="horizontal")
    
    # Calculate dollar returns using Polars
    # First, clean Trade_Size_USD and convert to float
    df_combined = df_combined.with_columns([
        pl.col("Trade_Size_USD").cast(pl.Float64, strict=False).alias("Trade_Size_USD_clean")
    ])
    
    # Calculate shares (Trade_Size_USD / filed_price)
    df_combined = df_combined.with_columns([
        (pl.col("Trade_Size_USD_clean") / pl.col("filed_price")).alias("estimated_shares")
    ])
    
    # Calculate dollar returns for each interval (based on filed price)
    # Delta = shares * filed_price * (return_pct / 100)
    for interval in ['1m', '3m', '6m', '9m', '12m']:
        df_combined = df_combined.with_columns([
            (pl.col("estimated_shares") * pl.col("filed_price") * (pl.col(f"return_{interval}_pct") / 100))
            .alias(f"delta_{interval}_usd")
        ])
    
    # Select only the columns we want in the output
    output_cols = [
        # Original columns
        'Ticker', 'TradeDate', 'Traded', 'FiledDate', 'Filed', 'Name', 'Transaction', 'Trade_Size_USD', 
        'Party', 'Chamber', 'Company', 'Description',
        # Prices at trade and filed
        'trade_price', 'filed_price', 'traded_to_filed_pct',
        # 1 month (from filed date)
        'return_1m_pct', 'delta_1m_usd',
        # 3 months
        'return_3m_pct', 'delta_3m_usd',
        # 6 months
        'return_6m_pct', 'delta_6m_usd',
        # 9 months
        'return_9m_pct', 'delta_9m_usd',
        # 12 months
        'return_12m_pct', 'delta_12m_usd',
    ]
    
    # Filter to only existing columns
    existing_cols = [col for col in output_cols if col in df_combined.columns]
    df_results = df_combined.select(existing_cols)
    
    calc_time = time.time() - calc_start
    print(f"      ✓ Calculated dollar returns in {calc_time:.2f}s")
    
    # Save results
    print(f"\n[5/5] Saving results...")
    save_start = time.time()
    
    df_results.write_parquet(OUTPUT_FILE)
    csv_file = OUTPUT_FILE.replace('.parquet', '.csv')
    df_results.write_csv(csv_file)
    
    print(f"      ✓ Saved to:")
    print(f"        - {OUTPUT_FILE}")
    print(f"        - {csv_file}")
    print(f"      ✓ Save time: {time.time() - save_start:.2f}s")
    
    # Show summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    successful = df_results.filter(pl.col('trade_price').is_not_null()).height
    print(f"Total trades analyzed: {df_results.height}")
    print(f"Successful price lookups: {successful}")
    
    for interval in ['1m', '3m', '6m', '9m', '12m']:
        returns_pct = df_results.filter(pl.col(f'return_{interval}_pct').is_not_null())[f'return_{interval}_pct']
        returns_usd = df_results.filter(pl.col(f'delta_{interval}_usd').is_not_null())[f'delta_{interval}_usd']
        
        if len(returns_pct) > 0:
            avg_return_pct = returns_pct.mean()
            positive = (returns_pct > 0).sum()
            print(f"\n{interval.upper()} Performance:")
            print(f"  Avg % return: {avg_return_pct:+.2f}%")
            print(f"  Win rate: {positive}/{len(returns_pct)} ({100*positive/len(returns_pct):.1f}%)")
            print(f"  Best: {returns_pct.max():+.2f}%  |  Worst: {returns_pct.min():+.2f}%")
            
            if len(returns_usd) > 0:
                avg_usd = returns_usd.mean()
                total_usd = returns_usd.sum()
                print(f"  Avg $ return: ${avg_usd:+,.2f}")
                print(f"  Total $ return: ${total_usd:+,.2f}")
    
    # Performance summary
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"  - Data loading: {time.time() - load_start:.2f}s")
    print(f"  - API calls: {api_time:.2f}s ({api_time/total_time*100:.1f}%)")
    print(f"  - Calculations: {calc_time:.2f}s ({calc_time/total_time*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()