import polars as pl
import json
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime
import sys
import yfinance as yf

def download_sp500_data():
    """Download S&P 500 data from Yahoo Finance."""
    print("Downloading S&P 500 data from Yahoo Finance...")
    sp500 = yf.download('^GSPC', start='2010-01-01', progress=False)
    
    # Convert dates to list of date objects
    dates = [d.date() for d in sp500.index]
    
    # Convert to polars dataframe
    sp500_df = pl.DataFrame({
        'Date': dates,
        'Close_Price': sp500['Close'].values.flatten().tolist(),
        'Open_Price': sp500['Open'].values.flatten().tolist()
    }).sort('Date')
    
    print(f"Downloaded S&P 500 data: {sp500_df.height} rows")
    return sp500_df

def get_nearest_previous_price(stock_df, target_date, column_type='Close_Price'):
    """Get the close price for the nearest previous trading day."""
    # Filter dates that are on or before target date
    previous_dates = stock_df.filter(pl.col('Date') <= target_date)
    
    if previous_dates.height == 0:
        return None
    
    # Get the most recent date and its price
    column = 'Close_Price' if column_type == 'Close_Price' else 'Open_Price'
    price = previous_dates.sort('Date', descending=True).select(column).row(0)[0]
    
    return price

def calculate_percentage_increase(old_price, new_price):
    """Calculate percentage increase."""
    if old_price is None or new_price is None or old_price == 0:
        return None
    return ((new_price - old_price) / old_price) * 100

def calculate_alpha(stock_pct, sp500_pct):
    """Calculate alpha by subtracting S&P 500 performance."""
    if stock_pct is None or sp500_pct is None:
        return None
    return stock_pct - sp500_pct

def process_trades(limit=0):
    """
    Process politician trades and calculate stock performance metrics.
    
    Args:
        limit: Number of unique tickers to process. If 0, process all tickers.
    """
    # Download S&P 500 data once
    sp500_df = download_sp500_data()
    
    print("\nLoading congress trades data...")
    df = pl.read_parquet('politician_trades_data/congress_trades_full.parquet')
    
    # Filter by TickerType
    print("Filtering by TickerType (Stock or ST)...")
    df = df.filter(pl.col('TickerType').is_in(['Stock', 'ST']))
    
    # Load metadata
    print("Loading metadata...")
    with open('stock_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    valid_tickers = set(metadata['tickers'].keys())
    
    # Filter by tickers in metadata
    print("Filtering by tickers in metadata...")
    df = df.filter(pl.col('Ticker').is_in(valid_tickers))
    
    # Sort by Ticker
    print("Sorting by Ticker...")
    df = df.sort('Ticker')
    
    # Get unique tickers
    unique_tickers = df.select('Ticker').unique().sort('Ticker').to_series().to_list()
    
    # Limit tickers if specified
    if limit > 0:
        unique_tickers = unique_tickers[:limit]
        print(f"Processing first {limit} tickers...")
        df = df.filter(pl.col('Ticker').is_in(unique_tickers))
    else:
        print(f"Processing all {len(unique_tickers)} tickers...")
    
    # Initialize new columns
    new_columns = {
        'price_at_filed': [],
        'alpha_traded_to_filed': [],
        'alpha_filed_to_1m': [],
        'alpha_filed_to_3m': [],
        'alpha_filed_to_6m': [],
        'alpha_filed_to_9m': [],
        'alpha_filed_to_12m': []
    }
    
    stock_data_path = Path('stock_data/parquet_files')
    current_ticker = None
    current_stock_df = None
    
    print("\nProcessing trades...")
    total_rows = df.height
    processed = 0
    
    for row in df.iter_rows(named=True):
        ticker = row['Ticker']
        
        # Load new stock data if ticker changed
        if ticker != current_ticker:
            stock_file = stock_data_path / f"{ticker}.parquet"
            
            if not stock_file.exists():
                print(f"Warning: Stock file not found for {ticker}, skipping...")
                # Add None values for this row
                for col in new_columns:
                    new_columns[col].append(None)
                processed += 1
                continue
            
            current_ticker = ticker
            current_stock_df = pl.read_parquet(stock_file)
            current_stock_df = current_stock_df.with_columns(
                pl.col('Date').cast(pl.Date)
            )
            print(f"Loaded stock data for {ticker}")
        
        # Get dates
        traded_date = row['Traded']
        filed_date = row['Filed']
        
        if traded_date is None or filed_date is None:
            for col in new_columns:
                new_columns[col].append(None)
            processed += 1
            continue
        
        # Convert to datetime.date if needed
        if isinstance(traded_date, str):
            traded_date = datetime.strptime(traded_date, '%Y-%m-%d').date()
        if isinstance(filed_date, str):
            filed_date = datetime.strptime(filed_date, '%Y-%m-%d').date()
        
        # Calculate target dates
        filed_1m = filed_date + relativedelta(months=1)
        filed_3m = filed_date + relativedelta(months=3)
        filed_6m = filed_date + relativedelta(months=6)
        filed_9m = filed_date + relativedelta(months=9)
        filed_12m = filed_date + relativedelta(months=12)
        
        # Get stock prices
        price_traded = get_nearest_previous_price(current_stock_df, traded_date)
        price_filed = get_nearest_previous_price(current_stock_df, filed_date + relativedelta(days=1), 'Open_Price')
        price_1m = get_nearest_previous_price(current_stock_df, filed_1m, 'Open_Price')
        price_3m = get_nearest_previous_price(current_stock_df, filed_3m, 'Open_Price')
        price_6m = get_nearest_previous_price(current_stock_df, filed_6m, 'Open_Price')
        price_9m = get_nearest_previous_price(current_stock_df, filed_9m, 'Open_Price')
        price_12m = get_nearest_previous_price(current_stock_df, filed_12m, 'Open_Price')
        
        # Get S&P 500 prices
        sp500_traded = get_nearest_previous_price(sp500_df, traded_date)
        sp500_filed = get_nearest_previous_price(sp500_df, filed_date + relativedelta(days=1), 'Open_Price')
        sp500_1m = get_nearest_previous_price(sp500_df, filed_1m, 'Open_Price')
        sp500_3m = get_nearest_previous_price(sp500_df, filed_3m, 'Open_Price')
        sp500_6m = get_nearest_previous_price(sp500_df, filed_6m, 'Open_Price')
        sp500_9m = get_nearest_previous_price(sp500_df, filed_9m, 'Open_Price')
        sp500_12m = get_nearest_previous_price(sp500_df, filed_12m, 'Open_Price')
        
        # Calculate stock percentage increases
        stock_pct_traded_to_filed = calculate_percentage_increase(price_traded, price_filed)
        stock_pct_filed_to_1m = calculate_percentage_increase(price_filed, price_1m)
        stock_pct_filed_to_3m = calculate_percentage_increase(price_filed, price_3m)
        stock_pct_filed_to_6m = calculate_percentage_increase(price_filed, price_6m)
        stock_pct_filed_to_9m = calculate_percentage_increase(price_filed, price_9m)
        stock_pct_filed_to_12m = calculate_percentage_increase(price_filed, price_12m)
        
        # Calculate S&P 500 percentage increases
        sp500_pct_traded_to_filed = calculate_percentage_increase(sp500_traded, sp500_filed)
        sp500_pct_filed_to_1m = calculate_percentage_increase(sp500_filed, sp500_1m)
        sp500_pct_filed_to_3m = calculate_percentage_increase(sp500_filed, sp500_3m)
        sp500_pct_filed_to_6m = calculate_percentage_increase(sp500_filed, sp500_6m)
        sp500_pct_filed_to_9m = calculate_percentage_increase(sp500_filed, sp500_9m)
        sp500_pct_filed_to_12m = calculate_percentage_increase(sp500_filed, sp500_12m)
        
        # Calculate alpha (stock performance - S&P 500 performance)
        new_columns['price_at_filed'].append(price_filed)
        new_columns['alpha_traded_to_filed'].append(
            calculate_alpha(stock_pct_traded_to_filed, sp500_pct_traded_to_filed)
        )
        new_columns['alpha_filed_to_1m'].append(
            calculate_alpha(stock_pct_filed_to_1m, sp500_pct_filed_to_1m)
        )
        new_columns['alpha_filed_to_3m'].append(
            calculate_alpha(stock_pct_filed_to_3m, sp500_pct_filed_to_3m)
        )
        new_columns['alpha_filed_to_6m'].append(
            calculate_alpha(stock_pct_filed_to_6m, sp500_pct_filed_to_6m)
        )
        new_columns['alpha_filed_to_9m'].append(
            calculate_alpha(stock_pct_filed_to_9m, sp500_pct_filed_to_9m)
        )
        new_columns['alpha_filed_to_12m'].append(
            calculate_alpha(stock_pct_filed_to_12m, sp500_pct_filed_to_12m)
        )
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total_rows} rows ({processed/total_rows*100:.1f}%)")
    
    # Add new columns to dataframe
    print("\nAdding calculated columns to dataframe...")
    for col_name, col_data in new_columns.items():
        df = df.with_columns(pl.Series(name=col_name, values=col_data))
    
    # Save to CSV and Parquet
    output_path = 'politician_trades_data/stock_performance_analysis.csv'
    output_path_parquet = 'politician_trades_data/stock_performance_analysis.parquet'
    print(f"\nSaving results to {output_path}...")
    df.write_csv(output_path, separator=';', decimal_comma=True)
    df.write_parquet(output_path_parquet)

    print(f"Done! Processed {processed} rows across {len(unique_tickers)} tickers.")

if __name__ == "__main__":
    # Get limit from command line argument, default to 0 (all tickers)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    process_trades(limit)