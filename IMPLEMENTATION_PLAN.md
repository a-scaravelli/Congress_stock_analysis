# Implementation Plan: Realized CAR Feature

## Overview
Add "realized_car" column to track actual returns from buy to sell, providing a more realistic target variable than fixed time windows.

## Current State
- File: `src/analysis/stock_performance_analysis.py` (404 lines)
- Function: `process_trades()` calculates CAR at fixed intervals (1m, 3m, 6m, 9m, 12m)
- Output: `data/trades/stock_performance_analysis.parquet` 
- Enriched output: `data/output/politician_trades_enriched.parquet` (39,326 rows, 104 columns)

## Requirements

### New Columns
1. **realized_car** (float, nullable):
   - CAR from buy date to actual sell date
   - NULL if no matching sell within 12 months
   
2. **holding_period_days** (float, nullable):
   - Days between buy and sell
   - NULL if not sold
   
3. **position_closed** (boolean):
   - True if matching sell found within 12 months
   - False otherwise

### Matching Logic
- Match buy/sell by: same `Ticker` + same `BioGuideID` (politician)
- Sell must occur AFTER the purchase
- Sell must be within 12 months of purchase (Filed date)
- Transaction types to match:
  - Purchase: "Purchase"
  - Sell: "Sale", "Sale (Full)", "Sale (Partial)"

## Implementation Steps

### Step 1: Add helper function to match buy/sell pairs
**File**: `src/analysis/stock_performance_analysis.py`
**Location**: After `calculate_car()` function (around line 217)

```python
def find_matching_sell(df, row_idx, ticker, bioguide_id, filed_date):
    """
    Find the first sell transaction for the same ticker by the same politician
    within 12 months after the purchase filed date.
    
    Args:
        df: DataFrame with all trades
        row_idx: Current row index to avoid matching with itself
        ticker: Stock ticker
        bioguide_id: Politician's BioGuide ID
        filed_date: Filed date of the purchase
    
    Returns:
        dict with sell_date and sell_idx, or None if no match found
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
```

### Step 2: Modify process_trades() function
**File**: `src/analysis/stock_performance_analysis.py`
**Changes**:

1. Add new columns to `new_columns` dict (around line 277):
```python
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
    'realized_car': [],              # NEW
    'holding_period_days': [],       # NEW
    'position_closed': [],           # NEW
}
```

2. In the main processing loop (around line 330), add realized CAR logic:
```python
# After calculating the fixed-window CARs...

# Calculate realized CAR if this is a Purchase
if row['Transaction'] == 'Purchase':
    matching_sell = find_matching_sell(
        df, row_idx, ticker, row['BioGuideID'], filed_date
    )
    
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
        holding_days = (sell_date - filed_date).days
        
        new_columns['realized_car'].append(realized_car)
        new_columns['holding_period_days'].append(float(holding_days))
        new_columns['position_closed'].append(True)
    else:
        # No matching sell found
        new_columns['realized_car'].append(None)
        new_columns['holding_period_days'].append(None)
        new_columns['position_closed'].append(False)
else:
    # Not a Purchase transaction (it's a Sale or Exchange)
    # These don't have "realized" returns in the buy-to-sell sense
    new_columns['realized_car'].append(None)
    new_columns['holding_period_days'].append(None)
    new_columns['position_closed'].append(None)
```

### Step 3: Update downstream files (if needed)

**File**: `src/model/model.py`
- May want to add `realized_car` as an alternative target variable
- Could create separate models for realized vs fixed-window returns
- This is OPTIONAL for this iteration

## Testing Plan

1. **Unit test the matching function**:
   - Create test cases with known buy/sell pairs
   - Verify 12-month window enforcement
   - Test edge cases (same-day sells, multiple sells, no sells)

2. **Integration test**:
   - Run on a small subset (e.g., 100 tickers)
   - Verify columns are added correctly
   - Check distribution of position_closed (expect ~30-40% true)

3. **Data validation**:
   - Check that realized_car is only populated for Purchases
   - Verify holding_period_days ≤ 365
   - Compare realized_car vs car_filed_to_12m for closed positions

## Expected Outcomes

- **Output files**:
  - `data/trades/stock_performance_analysis.parquet` (updated)
  - `data/output/politician_trades_enriched.parquet` (updated in step 7)
  
- **Performance**:
  - Processing time may increase by ~10-20% due to sell matching
  - Could be optimized with indexing if needed

- **Data quality**:
  - ~60-70% of purchases will have position_closed=False (no sell found)
  - ~30-40% will have realized returns
  - Holding periods should range from 1-365 days

## File Changes Summary

| File | Changes | Lines Modified |
|------|---------|----------------|
| `src/analysis/stock_performance_analysis.py` | Add find_matching_sell() + update process_trades() | ~50 new lines |

## Execution Command

```bash
# Run step 3 only (stock performance analysis)
python src/main.py --steps 3

# Then run step 7 to create enriched file
python src/main.py --steps 7
```

## Rollback Plan

- Original files remain unchanged
- Can revert by checking out previous commit
- Backup current output files before running:
  ```bash
  cp data/trades/stock_performance_analysis.parquet data/trades/stock_performance_analysis.parquet.bak
  ```

## Notes

- The existing `politician_trades_with_realized_car.parquet` file appears to be incorrectly implemented
- This plan will create the proper implementation
- Future enhancement: Could track partial position sizes (buy 100 shares, sell 50 shares)
