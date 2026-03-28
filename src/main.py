"""
Main pipeline runner for the Congress Stock Analysis project.
Executes steps from config/execution_order.yaml in sequence.

Usage:
    python src/main.py                  # Run all steps
    python src/main.py --steps 1 2 3    # Run specific steps
    python src/main.py --start 3        # Run from step 3 onwards
    python src/main.py --steps 7 8      # Run only steps 7 and 8
"""

import sys
import os
import time

# Ensure project root is on sys.path so package imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.download_quant_files import fetch_and_save_single_file
from src.ingestion.extract_stock_data import process_ticker_data
from src.analysis.stock_performance_analysis import process_trades
from src.ingestion.committes_data_extractor import download_history
from src.ingestion.get_stock_informations import main as enrich_metadata
from src.analysis.industry_matching import main as match_industries
from src.ingestion.download_lobbying_data import main as download_lobbying
from src.model.model_realized import main as train_model

STEPS = {
    1: {
        "function": fetch_and_save_single_file,
        "description": "Download congress trades from QuiverQuant API",
    },
    2: {
        "function": process_ticker_data,
        "description": "Download historical stock prices from Yahoo Finance",
    },
    3: {
        "function": process_trades,
        "description": "Calculate stock performance (alpha vs S&P 500)",
    },
    4: {
        "function": download_history,
        "description": "Scrape committee membership history from GitHub",
    },
    5: {
        "function": enrich_metadata,
        "description": "Enrich stock metadata with sector/industry info",
    },
    6: {
        "function": None,
        "description": "MANUAL STEP: Create config/commette_industry_map.yaml",
    },
    7: {
        "function": match_industries,
        "description": "Flag trades within committee jurisdiction",
    },
    8: {
        "function": download_lobbying,
        "description": "Download lobbying disclosures from QuiverQuant (per ticker)",
    },
    9: {
        "function": train_model,
        "description": "Train and evaluate XGBoost and LightGBM models",
    },
}


def run_step(step_num):
    step = STEPS[step_num]
    func = step["function"]
    desc = step["description"]

    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {desc}")
    print(f"{'='*60}")

    if func is None:
        print(f"Skipping - this is a manual step.")
        print(f"Ensure config/commette_industry_map.yaml exists before running step 7.")
        return True

    start = time.time()

    try:
        func()
    except Exception as e:
        elapsed = time.time() - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"Step {step_num} FAILED after {minutes}m {seconds:.1f}s: {e}")
        return False

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"Step {step_num} completed in {minutes}m {seconds:.1f}s")
    return True


def main():
    import argparse

    # Ensure CWD is project root so relative data paths work
    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Congress Stock Analysis Pipeline")
    parser.add_argument("--steps", nargs="+", type=int, help="Run specific steps (e.g. --steps 1 2 3)")
    parser.add_argument("--start", type=int, help="Start from this step (e.g. --start 3)")
    args = parser.parse_args()

    if args.steps:
        steps_to_run = sorted(args.steps)
    elif args.start:
        steps_to_run = [s for s in sorted(STEPS.keys()) if s >= args.start]
    else:
        steps_to_run = sorted(STEPS.keys())

    invalid = [s for s in steps_to_run if s not in STEPS]
    if invalid:
        print(f"ERROR: Invalid step numbers: {invalid}")
        print(f"Valid steps: {sorted(STEPS.keys())}")
        sys.exit(1)

    print("Congress Stock Analysis Pipeline")
    print(f"Steps to run: {steps_to_run}")

    for step_num in steps_to_run:
        success = run_step(step_num)
        if not success and STEPS[step_num]["function"] is not None:
            print(f"\nPipeline stopped at step {step_num}.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
