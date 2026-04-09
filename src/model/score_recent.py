"""src/model/score_recent.py

Score post-cutoff trades using the saved XGBoost classifier.

These are purchases filed AFTER the cutoff date (max(Traded) - 12 months) that
were excluded from training/testing because their 12-month forward return is not
yet observable. The model only needs features — no target required.

Usage (run from project root):
    python src/model/score_recent.py

Output: data/output/recent_trades_scored.csv
"""

import os
import sys

# Ensure CWD is project root
_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_here, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
import xgboost as xgb

from model.model_realized import PoliticianTradeModel, Config


def main():
    cfg = Config()

    # --- Load data and run sells-pressure on full trades (before purchase filter) ---
    print(f"Loading {cfg.data_path}...")
    df = pd.read_parquet(cfg.data_path)

    print("Calculating sells pressure (full dataset)...")
    df = PoliticianTradeModel._calculate_sells_pressure(df, "Filed")
    df = PoliticianTradeModel._calculate_all_pol_sells_same_ticker(df, "Filed", window_days=30)

    # Filter to purchases
    purchases = df[df["Transaction"] == "Purchase"].copy()
    print(f"Total purchases: {len(purchases)}")

    # --- Compute cutoff (same formula as main()) ---
    purchases["Traded"] = pd.to_datetime(purchases["Traded"], errors="coerce")
    max_traded = purchases["Traded"].max()
    cutoff = (max_traded - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    cutoff_ts = pd.Timestamp(cutoff)
    print(f"Max traded date: {max_traded.date()}")
    print(f"Cutoff date (12m lookback): {cutoff}")

    purchases["Filed"] = pd.to_datetime(purchases["Filed"], errors="coerce")
    n_post = (purchases["Filed"] > cutoff_ts).sum()
    print(f"Post-cutoff purchases to score: {n_post}")

    if n_post == 0:
        print("No post-cutoff trades found. Nothing to score.")
        return

    # --- Run normal pipeline on within-cutoff data to fit encoders ---
    # preprocess() applies the cutoff filter internally, builds the hybrid target,
    # and runs all feature engineering on within-cutoff rows only.
    model = PoliticianTradeModel(cfg, cutoff_date=cutoff, horizon_months=12)
    df_within = model.preprocess(purchases)

    train_df, test_df = model.time_split(df_within)
    # Fit label encoders and fill values on training data
    model.prepare_features(train_df, is_training=True)

    # Load saved model weights
    clf = xgb.XGBClassifier(enable_categorical=True)
    clf.load_model("data/output/xgboost_model_realized.json")
    model.xgb_model = clf
    print("Loaded saved model.")

    # --- Run full feature engineering on ALL purchases (within + post cutoff) ---
    # This ensures lookback features for post-cutoff rows (politician hit rate, etc.)
    # see the complete historical record.
    #
    # For the hybrid target: within-cutoff rows get their real value; post-cutoff rows
    # get 0.0 (neutral placeholder). Since the horizon gap is 12 months and all
    # post-cutoff rows are within the last 12 months, none of them will appear as
    # "eligible history" in another post-cutoff row's lookback — so the 0.0 fill
    # never contaminates feature values.
    purchases_all = purchases.copy()

    # Parse decimal-comma columns (same as preprocess())
    for col in ["car_traded_to_filed", "beta", "stock_momentum_30d",
                "stock_momentum_90d", "stock_volatility_30d",
                "max_committee_rank", "holding_period_days"]:
        if col in purchases_all.columns:
            purchases_all[col] = model._parse_decimal_comma(purchases_all[col])

    # Build hybrid target
    car_col = f"car_filed_to_{model.horizon_months}m"
    purchases_all["realized_car"] = model._parse_decimal_comma(purchases_all["realized_car"])
    purchases_all[car_col] = model._parse_decimal_comma(purchases_all[car_col])
    purchases_all["holding_period_days"] = pd.to_numeric(
        purchases_all["holding_period_days"], errors="coerce"
    )
    mask_use_realized = (
        (purchases_all["position_closed"] == True)
        & (purchases_all["holding_period_days"] <= model.horizon_days)
        & (purchases_all["realized_car"].notna())
    )
    purchases_all[model.target_continuous] = purchases_all[car_col]
    purchases_all.loc[mask_use_realized, model.target_continuous] = purchases_all.loc[
        mask_use_realized, "realized_car"
    ]
    # Post-cutoff rows have no target yet — fill with 0.0 so feature engineering can run
    purchases_all[model.target_continuous] = purchases_all[model.target_continuous].fillna(0.0)

    # Run feature engineering pipeline on full purchases dataset
    purchases_all = model._add_committee_category_flags(purchases_all)
    purchases_all = model._add_lobbying_features(purchases_all)
    purchases_all = model._add_market_regime_features(purchases_all)
    purchases_all = model._add_engineered_features(purchases_all)

    # --- Extract post-cutoff rows and score ---
    purchases_all["Filed"] = pd.to_datetime(purchases_all["Filed"], errors="coerce")
    post_cutoff = purchases_all[purchases_all["Filed"] > cutoff_ts].copy()
    print(f"\nPost-cutoff rows after feature engineering: {len(post_cutoff)}")

    X_recent = model.prepare_features(post_cutoff, is_training=False)
    probs = model.xgb_model.predict_proba(X_recent)[:, 1]
    post_cutoff = post_cutoff.copy()
    post_cutoff["clf_prob"] = probs

    # --- Output ---
    post_cutoff["predicted_positive"] = (post_cutoff["clf_prob"] >= 0.70).astype(int)
    result = post_cutoff.sort_values("clf_prob", ascending=False).reset_index(drop=True)

    # Move prediction columns to the front for readability
    front_cols = ["Traded", "Filed", "BioGuideID", "Ticker", "clf_prob", "predicted_positive"]
    other_cols = [c for c in result.columns if c not in front_cols]
    result = result[front_cols + other_cols]

    print(f"\nTop 30 recent trades by predicted probability:")
    print(result[["Traded", "Filed", "BioGuideID", "Ticker", "clf_prob", "predicted_positive",
                  "Ticker_Sector", "Ticker_Industry"]].head(30).to_string(index=False))

    result.to_parquet("data/output/recent_trades_scored.parquet", index=False)
    result.to_excel("data/output/recent_trades_scored.xlsx", index=False)
    print(f"\nSaved {len(result)} scored trades:")
    print(f"  data/output/recent_trades_scored.parquet")
    print(f"  data/output/recent_trades_scored.xlsx")
    print(f"Predicted positive (>= 0.70): {result['predicted_positive'].sum()}")
    print(f"Above thr=0.84: {(result['clf_prob'] >= 0.84).sum()}")
    print(f"Above thr=0.86: {(result['clf_prob'] >= 0.86).sum()}")


if __name__ == "__main__":
    main()
