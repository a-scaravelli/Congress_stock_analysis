import polars as pl
import yaml
import os
import glob
import json
from datetime import datetime

# --- Configuration ---
TRANSACTIONS_FILE = 'politician_trades_data/stock_performance_analysis.csv'
METADATA_FILE = 'stock_data/metadata_enriched.json'
COMMITTEE_DIR = 'committee_history'
MAP_FILE = 'commette_industry_map.yaml'
OUTPUT_FILE = 'politician_trades_enriched.csv'

# --- Helper Functions ---

def parse_file_date(filename):
    try:
        base_name = os.path.basename(filename)
        timestamp_str = base_name.split('_')[0]
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%SZ")
    except (ValueError, IndexError):
        return None

def load_membership_data(file_paths):
    """
    Reads YAML files, truncates committee codes to 4 chars, and returns unique memberships.
    """
    records = []
    unique_paths = set(file_paths)
    
    print(f"Parsing {len(unique_paths)} unique committee files...")
    
    for f_path in unique_paths:
        if not f_path: continue
        
        try:
            with open(f_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if not data: continue

            for committee_code, members in data.items():
                if members:
                    # --- CHANGE 1: Truncate to first 4 chars ---
                    short_code = committee_code[:4]
                    
                    for member in members:
                        if isinstance(member, dict):
                            bid = member.get('bioguide')
                            if bid:
                                records.append({
                                    'mapped_file_path': f_path,
                                    'BioGuideID': bid,
                                    'Committee': short_code, # Use short code
                                    'IsMember': 1
                                })
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    if not records:
        return pl.DataFrame(
            schema={'mapped_file_path': pl.Utf8, 'BioGuideID': pl.Utf8, 'Committee': pl.Utf8, 'IsMember': pl.Int8}
        )
    
    # --- CHANGE 2: Deduplicate ---
    # Since HSAG14 and HSAG03 both became HSAG, a person might appear twice for HSAG.
    # We call .unique() to keep just one row per person per committee code.
    return pl.DataFrame(records).unique()

def load_rules_data(map_file):
    with open(map_file, 'r') as f:
        data = yaml.safe_load(f)
        
    sec_records = []
    ind_records = []
    
    for comm, rules in data.items():
        # Ensure rules map also uses keys (which are usually 4 chars already)
        # If the map key is longer, we might need to slice it too, 
        # but typically the map defines the parent codes (HSIF, HLIG).
        # We use the key as provided in the YAML.
        
        if 'sectors' in rules and rules['sectors']:
            for sec, score in rules['sectors'].items():
                sec_records.append({'Committee': comm, 'Ticker_Sector': sec, 'Sector_Score': score})
                
        if 'industries' in rules and rules['industries']:
            for ind, score in rules['industries'].items():
                ind_records.append({'Committee': comm, 'Ticker_Industry': ind, 'Industry_Score': score})
                
    s_df = pl.DataFrame(sec_records, schema={'Committee': pl.Utf8, 'Ticker_Sector': pl.Utf8, 'Sector_Score': pl.Int64})
    i_df = pl.DataFrame(ind_records, schema={'Committee': pl.Utf8, 'Ticker_Industry': pl.Utf8, 'Industry_Score': pl.Int64})
    
    return s_df, i_df

# --- Main Execution ---

def main():
    print("Loading Transactions...")
    
    df = pl.read_csv(TRANSACTIONS_FILE, separator=';', try_parse_dates=False).with_row_index("row_id")
    
    df = df.with_columns(
        pl.col("Traded").str.to_datetime(strict=False).alias("Traded")
    )
    df = df.filter(pl.col("Traded").is_not_null())

    print("Loading Metadata...")
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    meta_records = []
    for ticker, data in metadata.get('tickers', {}).items():
        profile = data.get('summaryProfile', {})
        meta_records.append({
            'Ticker': ticker,
            'Ticker_Industry': profile.get('industry'),
            'Ticker_Sector': profile.get('sector')
        })
    
    meta_df = pl.DataFrame(meta_records)
    df = df.join(meta_df, on='Ticker', how='left')

    print("Matching Transaction Dates to Committee Files...")
    all_files = glob.glob(os.path.join(COMMITTEE_DIR, "*.yaml"))
    file_records = []
    for f in all_files:
        d = parse_file_date(f)
        if d:
            file_records.append({'file_date': d, 'mapped_file_path': f})
            
    files_df = pl.DataFrame(file_records).sort('file_date')
    df_sorted = df.sort('Traded')
    
    df_with_files = df_sorted.join_asof(
        files_df, 
        left_on='Traded', 
        right_on='file_date', 
        strategy='backward'
    )

    print("Loading Committee Memberships (Merged by 4-char code)...")
    needed_files = df_with_files.filter(pl.col('mapped_file_path').is_not_null())['mapped_file_path'].unique().to_list()
    membership_df = load_membership_data(needed_files)
    
    # --- Pivot Committee Columns (Now using short codes) ---
    membership_pivoted = membership_df.pivot(
        values="IsMember",
        index=["mapped_file_path", "BioGuideID"],
        on="Committee",
        aggregate_function="first"
    ).fill_null(0)

    # Rename columns to Committee_XXXX
    comm_cols = [c for c in membership_pivoted.columns if c not in ["mapped_file_path", "BioGuideID"]]
    rename_map = {c: f"Committee_{c}" for c in comm_cols}
    membership_pivoted = membership_pivoted.rename(rename_map)

    # Join Flags
    df_with_flags = df_with_files.join(
        membership_pivoted,
        on=['mapped_file_path', 'BioGuideID'],
        how='left'
    )
    
    # Fill nulls for flags
    flag_cols = list(rename_map.values())
    df_with_flags = df_with_flags.with_columns([
        pl.col(c).fill_null(0) for c in flag_cols
    ])

    print("Applying Conflict Logic...")
    # Join membership again (long format) for scoring
    # Since membership_df already has short codes, this works automatically
    df_long = df_with_files.join(
        membership_df, 
        on=['mapped_file_path', 'BioGuideID'], 
        how='left'
    )

    sec_rules_df, ind_rules_df = load_rules_data(MAP_FILE)

    df_scored = df_long.join(sec_rules_df, on=['Committee', 'Ticker_Sector'], how='left')
    df_scored = df_scored.join(ind_rules_df, on=['Committee', 'Ticker_Industry'], how='left')

    df_scored = df_scored.with_columns([
        pl.col('Sector_Score').fill_null(0),
        pl.col('Industry_Score').fill_null(0)
    ])

    print("Aggregating Results...")
    final_aggs = df_scored.group_by('row_id').agg([
        (pl.col('Industry_Score') == 1).any().cast(pl.Int8).alias('Industry match 1'),
        (pl.col('Industry_Score') == 2).any().cast(pl.Int8).alias('Industry match 2'),
        (pl.col('Industry_Score') == 3).any().cast(pl.Int8).alias('Industry match 3'),
        
        (pl.col('Sector_Score') == 1).any().cast(pl.Int8).alias('Sector match 1'),
        (pl.col('Sector_Score') == 2).any().cast(pl.Int8).alias('Sector match 2'),
        (pl.col('Sector_Score') == 3).any().cast(pl.Int8).alias('Sector match 3'),
        
        pl.col('Ticker_Industry').first(),
        pl.col('Ticker_Sector').first()
    ])

    result = df_with_flags.join(final_aggs, on='row_id', how='left').drop('row_id')

    match_cols = [
        'Industry match 1', 'Industry match 2', 'Industry match 3',
        'Sector match 1', 'Sector match 2', 'Sector match 3'
    ]
    result = result.with_columns([pl.col(c).fill_null(0) for c in match_cols])

    print(f"Saving to {OUTPUT_FILE}...")
    result.write_csv(OUTPUT_FILE, separator=';')
    print("Done.")

if __name__ == "__main__":
    main()
