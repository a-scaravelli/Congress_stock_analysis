import polars as pl
import json
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime
import sys
import yfinance as yf
import numpy as np

def download_sp500_data(start_date='2010-01-01'):
    """Download S&P 500 data from Yahoo Finance."""
    print(f"Downloading S&P 500 data from {start_date} to today...")
    sp500 = yf.download('^GSPC', start=start_date, progress=False)
    
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

def calculate_beta(stock_df, sp500_df, calc_date, lookback_days=756):
    """
    Calculate beta using up to 756 trading days (3 years) before calc_date.
    If insufficient data, uses whatever is available (minimum 60 days).
    Beta = Covariance(Stock_returns, S&P500_returns) / Variance(S&P500_returns)
    """
    # Get data lookback_days before calc_date (756 days ≈ 3 years of trading)
    lookback_start = calc_date - relativedelta(days=lookback_days*1.5)  # Get extra data to account for weekends
    
    # Filter stock data
    stock_data = stock_df.filter(
        (pl.col('Date') >= lookback_start) & 
        (pl.col('Date') < calc_date)
    ).sort('Date')
    
    # Filter S&P 500 data
    sp500_data = sp500_df.filter(
        (pl.col('Date') >= lookback_start) & 
        (pl.col('Date') < calc_date)
    ).sort('Date')
    
    if stock_data.height < 60 or sp500_data.height < 60:
        return None  # Minimum 60 days required
    
    # Find common dates between stock and S&P 500
    stock_dates = set(stock_data['Date'].to_list())
    sp500_dates = set(sp500_data['Date'].to_list())
    common_dates = sorted(stock_dates & sp500_dates)
    
    if len(common_dates) < 60:
        return None  # Need at least 60 common dates
    
    # Filter both datasets to common dates
    stock_data = stock_data.filter(pl.col('Date').is_in(common_dates)).sort('Date')
    sp500_data = sp500_data.filter(pl.col('Date').is_in(common_dates)).sort('Date')
    
    # Extract prices in order of common dates
    stock_prices = stock_data['Close_Price'].to_numpy()
    sp500_prices = sp500_data['Close_Price'].to_numpy()
    
    # Calculate daily log returns (additive, consistent with CAR summation)
    stock_returns = np.log(stock_prices[1:] / stock_prices[:-1])
    sp500_returns = np.log(sp500_prices[1:] / sp500_prices[:-1])

    # Use up to 756 days / 3 years (this now respects date alignment)
    if len(stock_returns) > 756:
        stock_returns = stock_returns[-756:]
        sp500_returns = sp500_returns[-756:]

    # Calculate beta using covariance (sample-based, ddof=1)
    covariance_matrix = np.cov(stock_returns, sp500_returns, ddof=1)
    covariance = covariance_matrix[0, 1]
    variance_market = np.var(sp500_returns, ddof=1)
    
    if variance_market == 0:
        return None
    
    beta = covariance / variance_market
    return beta


def calculate_stock_features(stock_df, sp500_df, target_date):
    """Calculate momentum and volatility features at a given date.

    Filters stock_df to S&P500 trading dates before computing features,
    eliminating both weekend rows and weekday market holidays added by
    the bfill resampling in extract_stock_data.py.

    Returns: (momentum_30d, momentum_90d, volatility_30d)
    """
    from dateutil.relativedelta import relativedelta

    # Filter to actual S&P500 trading dates to remove weekend/holiday rows
    sp500_trading_dates = set(sp500_df['Date'].to_list())
    stock_df = stock_df.filter(pl.col('Date').is_in(sp500_trading_dates))

    date_30d_ago = target_date - relativedelta(days=30)
    date_90d_ago = target_date - relativedelta(days=90)

    data_30d = stock_df.filter(
        (pl.col('Date') >= date_30d_ago) & (pl.col('Date') < target_date)
    ).sort('Date')

    data_90d = stock_df.filter(
        (pl.col('Date') >= date_90d_ago) & (pl.col('Date') < target_date)
    ).sort('Date')

    momentum_30d = None
    if data_30d.height >= 2:
        p_start = data_30d['Open_Price'][0]
        p_end = data_30d['Open_Price'][-1]
        if p_start > 0 and p_end > 0:
            momentum_30d = float(np.log(p_end / p_start))

    momentum_90d = None
    if data_90d.height >= 2:
        p_start = data_90d['Open_Price'][0]
        p_end = data_90d['Open_Price'][-1]
        if p_start > 0 and p_end > 0:
            momentum_90d = float(np.log(p_end / p_start))

    volatility_30d = None
    if data_30d.height >= 5:
        prices = data_30d['Open_Price'].to_numpy()
        log_returns = np.log(prices[1:] / prices[:-1])
        volatility_30d = float(np.std(log_returns, ddof=1))

    return momentum_30d, momentum_90d, volatility_30d


def calculate_car(stock_df, sp500_df, start_date, end_date, beta):
    """
    Calculate Cumulative Abnormal Returns (CAR).
    CAR = Sum of (Stock_return - Beta * S&P500_return) for each day
    Returns None if insufficient data for the period.
    """
    if beta is None:
        return None
    
    # Filter data for the period
    stock_period = stock_df.filter(
        (pl.col('Date') > start_date) & 
        (pl.col('Date') <= end_date)
    ).sort('Date')
    
    sp500_period = sp500_df.filter(
        (pl.col('Date') > start_date) & 
        (pl.col('Date') <= end_date)
    ).sort('Date')
    
    if stock_period.height < 2 or sp500_period.height < 2:
        return None  # Need at least 2 data points to calculate returns
    
    # Get dates and prices
    stock_dates = set(stock_period['Date'].to_list())
    sp500_dates = set(sp500_period['Date'].to_list())
    
    # Find common dates
    common_dates = sorted(stock_dates & sp500_dates)
    
    if len(common_dates) < 2:
        return None  # Need at least 2 common dates
    
    # Get indices for common dates
    stock_date_list = stock_period['Date'].to_list()
    sp500_date_list = sp500_period['Date'].to_list()
    
    stock_prices = stock_period['Open_Price'].to_numpy()
    sp500_prices = sp500_period['Open_Price'].to_numpy()
    
    # Map common dates to indices
    stock_indices = [stock_date_list.index(d) for d in common_dates if d in stock_date_list]
    sp500_indices = [sp500_date_list.index(d) for d in common_dates if d in sp500_date_list]
    
    if len(stock_indices) < 2 or len(sp500_indices) < 2:
        return None
    
    stock_prices_aligned = stock_prices[stock_indices]
    sp500_prices_aligned = sp500_prices[sp500_indices]
    
    # Calculate daily log returns (additive, so CAR = sum of log ARs is correct)
    stock_returns = np.log(stock_prices_aligned[1:] / stock_prices_aligned[:-1])
    sp500_returns = np.log(sp500_prices_aligned[1:] / sp500_prices_aligned[:-1])
    
    if len(stock_returns) == 0 or len(sp500_returns) == 0:
        return None
    
    # Make sure they're the same length
    min_len = min(len(stock_returns), len(sp500_returns))
    stock_returns = stock_returns[:min_len]
    sp500_returns = sp500_returns[:min_len]
    
    if min_len == 0:
        return None
    
    # Calculate abnormal returns: AR = Stock_return - Beta * Market_return
    abnormal_returns = stock_returns - (beta * sp500_returns)
    
    # Cumulative abnormal returns
    car = np.sum(abnormal_returns)
    
    return car

def find_matching_sell(df, ticker, bioguide_id, filed_date):
    """
    Find the first sell transaction for the same ticker by the same politician
    within 12 months after the purchase filed date.

    Args:
        df: DataFrame with all trades
        ticker: Stock ticker
        bioguide_id: Politician's BioGuide ID
        filed_date: Filed date of the purchase

    Returns:
        dict with sell_date and transaction_type, or None if no match found
    """
    # Calculate 12-month window
    max_sell_date = filed_date + relativedelta(months=12)

    # Filter for potential sell transactions
    # Same ticker, same politician, after purchase, within 12 months
    # Transaction types: Sale, Sale (Full), Sale (Partial)
    potential_sells = df.filter(
        (pl.col('Ticker') == ticker) &
        (pl.col('BioGuideID') == bioguide_id) &
        (pl.col('Filed') > filed_date) &
        (pl.col('Filed') <= max_sell_date) &
        (pl.col('Transaction').str.contains('Sale'))
    ).sort('Filed')

    if potential_sells.height > 0:
        # Return the first sell (earliest date)
        first_sell = potential_sells.row(0, named=True)
        return {
            'sell_date': first_sell['Filed'],
            'transaction_type': first_sell['Transaction']
        }

    return None

def process_trades(limit=0):
    """
    Process politician trades and calculate stock performance metrics.
    
    Args:
        limit: Number of unique tickers to process. If 0, process all tickers.
    """
    print("Loading congress trades data...")
    df = pl.read_parquet('data/trades/congress_trades_full.parquet')
    
    # Convert date columns to proper date type
    df = df.with_columns([
        pl.col("Traded").str.to_date(strict=False),
        pl.col("Filed").str.to_date(strict=False)
    ])

    # Deduplicate same-person / same-ticker / same-day trades (STOCK Act tranche splits
    # and duplicate filings). Group by (BioGuideID, Ticker, Traded, Transaction), keep
    # earliest Filed date, and sum Trade_Size_USD across tranches.
    rows_before = df.height
    dedup_key = ["BioGuideID", "Ticker", "Traded", "Transaction"]
    other_cols = [c for c in df.columns if c not in dedup_key + ["Filed", "Trade_Size_USD"]]
    df = (
        df
        .sort("Filed")
        .group_by(dedup_key)
        .agg(
            [pl.col("Filed").first()]
            + [pl.col("Trade_Size_USD").cast(pl.Float64, strict=False).sum().cast(pl.Utf8)]
            + [pl.col(c).first() for c in other_cols]
        )
    )
    rows_after = df.height
    print(f"Deduplication: {rows_before} → {rows_after} rows (removed {rows_before - rows_after})")

    # Find minimum trade date and go back 3 years
    min_trade_date = df.select('Traded').min().item()
    sp500_start_date = (min_trade_date - relativedelta(years=3)).strftime('%Y-%m-%d')
    print(f"Minimum trade date: {min_trade_date}")
    print(f"S&P 500 start date: {sp500_start_date}")
    
    # Download S&P 500 data with parametric start date
    sp500_df = download_sp500_data(sp500_start_date)
    
    # Filter by TickerType
    print("Filtering by TickerType (Stock, ST, CS, PS)...")
    df = df.filter(pl.col('TickerType').is_in(['Stock', 'ST', 'CS', 'PS']))
    
    # Load metadata
    print("Loading metadata...")
    with open('data/stocks/metadata.json', 'r') as f:
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
    
    # Initialize new columns for CAR metrics
    new_columns = {
        'beta': [],
        'car_traded_to_filed': [],
        'car_filed_to_1m': [],
        'car_filed_to_3m': [],
        'car_filed_to_6m': [],
        'car_filed_to_9m': [],
        'car_filed_to_12m': [],
        'stock_momentum_30d': [],
        'stock_momentum_90d': [],
        'stock_volatility_30d': [],
        'realized_car': [],
        'holding_period_days': [],
        'position_closed': [],
    }
    
    stock_data_path = Path('data/stocks/parquet_files')
    current_ticker = None
    current_stock_df = None
    
    print("\nProcessing trades and calculating CAR...")
    total_rows = df.height
    processed = 0
    
    for row_idx, row in enumerate(df.iter_rows(named=True)):
        ticker = row['Ticker']
        
        # Load new stock data if ticker changed
        if ticker != current_ticker:
            stock_file = stock_data_path / f"{ticker}.parquet"
            
            if not stock_file.exists():
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
        
        try:
            # Calculate beta using 252 days before traded_date
            beta = calculate_beta(current_stock_df, sp500_df, traded_date)
            
            # Calculate target dates
            filed_1m = filed_date + relativedelta(months=1)
            filed_3m = filed_date + relativedelta(months=3)
            filed_6m = filed_date + relativedelta(months=6)
            filed_9m = filed_date + relativedelta(months=9)
            filed_12m = filed_date + relativedelta(months=12)
            
            # Calculate CAR for each period
            car_traded_filed = calculate_car(current_stock_df, sp500_df, traded_date, filed_date, beta)
            car_filed_1m = calculate_car(current_stock_df, sp500_df, filed_date, filed_1m, beta)
            car_filed_3m = calculate_car(current_stock_df, sp500_df, filed_date, filed_3m, beta)
            car_filed_6m = calculate_car(current_stock_df, sp500_df, filed_date, filed_6m, beta)
            car_filed_9m = calculate_car(current_stock_df, sp500_df, filed_date, filed_9m, beta)
            car_filed_12m = calculate_car(current_stock_df, sp500_df, filed_date, filed_12m, beta)
            
            # Calculate stock-level features at traded date (information available at decision time)
            mom_30d, mom_90d, vol_30d = calculate_stock_features(current_stock_df, sp500_df, traded_date)

            # Calculate realized CAR if this is a Purchase
            realized_car = None
            holding_days = None
            position_closed = False

            if row['Transaction'] == 'Purchase':
                matching_sell = find_matching_sell(df, ticker, row['BioGuideID'], filed_date)

                if matching_sell:
                    sell_date = matching_sell['sell_date']
                    # Convert to datetime.date if needed
                    if isinstance(sell_date, str):
                        sell_date = datetime.strptime(sell_date, '%Y-%m-%d').date()

                    # Calculate CAR from filed to sell date
                    realized_car = calculate_car(
                        current_stock_df, sp500_df, filed_date, sell_date, beta
                    )

                    # Calculate holding period
                    if realized_car is not None:
                        holding_days = (sell_date - filed_date).days
                        position_closed = True

            # Add to columns
            new_columns['beta'].append(beta)
            new_columns['car_traded_to_filed'].append(car_traded_filed)
            new_columns['car_filed_to_1m'].append(car_filed_1m)
            new_columns['car_filed_to_3m'].append(car_filed_3m)
            new_columns['car_filed_to_6m'].append(car_filed_6m)
            new_columns['car_filed_to_9m'].append(car_filed_9m)
            new_columns['car_filed_to_12m'].append(car_filed_12m)
            new_columns['stock_momentum_30d'].append(mom_30d)
            new_columns['stock_momentum_90d'].append(mom_90d)
            new_columns['stock_volatility_30d'].append(vol_30d)
            new_columns['realized_car'].append(realized_car)
            new_columns['holding_period_days'].append(holding_days)
            new_columns['position_closed'].append(position_closed)
            
        except Exception as e:
            print(f"    ERROR at row {row_idx} ticker {ticker}: {e}")
            # Add None values on error
            for col in new_columns:
                new_columns[col].append(None)
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total_rows} rows ({processed/total_rows*100:.1f}%)")
    
    # Add new columns to dataframe
    print("\nAdding calculated columns to dataframe...")
    print(f"DataFrame shape: {df.height} rows")
    for col_name, col_data in new_columns.items():
        print(f"  Processing column {col_name} with {len(col_data)} values...")
        # Ensure the length matches
        if len(col_data) != df.height:
            print(f"    WARNING: Column {col_name} has {len(col_data)} rows but dataframe has {df.height} rows")
            # Pad with None values if needed
            while len(col_data) < df.height:
                col_data.append(None)
            col_data = col_data[:df.height]
        
        try:
            df = df.with_columns(pl.Series(name=col_name, values=col_data))
        except Exception as e:
            print(f"    ERROR adding column {col_name}: {e}")
            raise
    
    # Save to CSV and Parquet
    output_path = 'data/trades/stock_performance_analysis.csv'
    output_path_parquet = 'data/trades/stock_performance_analysis.parquet'
    print(f"\nSaving results to {output_path}...")
    df.write_csv(output_path, separator=';', decimal_comma=True)
    df.write_parquet(output_path_parquet)

    print(f"Done! Processed {processed} rows across {len(unique_tickers)} tickers.")

if __name__ == "__main__":
    # Get limit from command line argument, default to 0 (all tickers)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    process_trades(limit)