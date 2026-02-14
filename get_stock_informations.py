import json
import math
from yahooquery import Ticker
import time

# 1. Load your existing metadata file
input_file = 'stock_data/metadata.json'
output_file = 'stock_data/metadata_enriched.json' # Save to new file to be safe

print(f"Reading {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract all ticker symbols (the keys)
all_tickers = list(data.get("tickers", {}).keys())
total_tickers = len(all_tickers)
print(f"Found {total_tickers} tickers to process.")

# 2. Configuration for batching
BATCH_SIZE = 500  # Yahooquery handles internal batching, but this keeps memory usage safe
batches = math.ceil(total_tickers / BATCH_SIZE)

# 3. Loop through batches
for i in range(batches):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, total_tickers)
    batch_symbols = all_tickers[start_idx:end_idx]
    
    print(f"Processing batch {i+1}/{batches} ({len(batch_symbols)} tickers)...")
    
    try:
        # Initialize Ticker object for this batch
        # asynchronous=True makes it much faster
        t = Ticker(batch_symbols, asynchronous=True)
        
        # Fetch only the 'summaryProfile' module (contains sector, industry, description)
        response = t.get_modules('summaryProfile')
        
        # 4. Update the main data dictionary
        for symbol in batch_symbols:
            # Check if we got a valid dictionary response for this symbol
            # (Sometimes APIs return string errors for delisted/bad tickers)
            if symbol in response and isinstance(response[symbol], dict):
                profile = response[symbol]
                
                # Only add if we actually found the profile data
                if profile:
                    # Update the specific ticker's entry in your structure
                    data['tickers'][symbol]['summaryProfile'] = profile
                    
            else:
                # Optional: Log missing data
                # print(f"No data found for {symbol}")
                pass
                
    except Exception as e:
        print(f"Error processing batch {i+1}: {e}")
    
    # Optional: Small sleep to be nice to the API if you have massive lists
    time.sleep(1)

# 5. Save the updated data
print(f"Saving updated data to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print("Done!")
