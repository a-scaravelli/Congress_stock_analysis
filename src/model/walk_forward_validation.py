"""src/model/walk_forward_validation.py

Walk-forward validation of the XGBoost classifier across 3 macro regimes,
as suggested by Prof. Ben's diagnostic framework.

Each fold trains on all data up to the regime start, then evaluates on the
regime window — no data from the test regime bleeds into training.

Folds:
  Fold 1 — COVID crash + recovery  : train <2020-01-01, test 2020 Q1-Q2
  Fold 2 — Rate hike drawdown      : train <2022-10-01, test 2022 Q4
  Fold 3 — Sideways/softer market  : train <2024-01-01, test 2024 Q1-Q3

Run from project root:
    python src/model/walk_forward_validation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.model.model_realized import Config, PoliticianTradeModel  # noqa: E402

# ── Fold definitions ─────────────────────────────────────────────────────────
FOLDS = [
    {
        "name": "Fold 1 — COVID crash + recovery (2020 Q1-Q2)",
        "train_end":  "2019-12-31",
        "test_start": "2020-01-01",
        "test_end":   "2020-06-30",
        "regime":     "COVID crash",
    },
    {
        "name": "Fold 2 — Rate hike drawdown (2022 Q4)",
        "train_end":  "2022-09-30",
        "test_start": "2022-10-01",
        "test_end":   "2022-12-31",
        "regime":     "Rate hike",
    },
    {
        "name": "Fold 3 — Sideways/softer market (2024 Q1-Q3)",
        "train_end":  "2023-12-31",
        "test_start": "2024-01-01",
        "test_end":   "2024-09-30",
        "regime":     "Sideways",
    },
]

# Thresholds to report in summary (mirroring main model sweet spot)
REPORT_THRESHOLDS = [0.60, 0.70, 0.80, 0.84, 0.86, 0.87]


def load_and_preprocess(cfg: Config) -> pd.DataFrame:
    """Load raw data, run sell-pressure calculations, filter to purchases,
    then run the full preprocess pipeline once with a liberal cutoff so that
    all available historical data (with valid 12m targets) is retained.
    """
    print(f"Loading {cfg.data_path} ...")
    raw = pd.read_parquet(cfg.data_path)

    # Sell-pressure features must be computed on the full dataset (buys + sells)
    # before filtering to purchases — same logic as main().
    raw = PoliticianTradeModel._calculate_sells_pressure(raw, "Filed")
    raw = PoliticianTradeModel._calculate_all_pol_sells_same_ticker(
        raw, "Filed", window_days=30
    )

    purchases = raw[raw["Transaction"] == "Purchase"].copy()
    print(f"Total purchases (pre-cutoff): {len(purchases)}")

    # Determine cutoff: max(Traded) - 12m, same logic as main model.
    purchases["Traded"] = pd.to_datetime(purchases["Traded"], errors="coerce")
    max_traded = purchases["Traded"].max()
    cutoff = (max_traded - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
    print(f"Max Traded: {max_traded.date()} → Cutoff (Filed ≤): {cutoff}")

    # Preprocess ONCE on a master model; all engineered features are computed
    # with strict no-lookahead rolling logic, so slicing later is safe.
    master = PoliticianTradeModel(cfg, cutoff_date=cutoff, horizon_months=12)
    preprocessed = master.preprocess(purchases)
    preprocessed["Filed"] = pd.to_datetime(preprocessed["Filed"], errors="coerce")
    return preprocessed


def run_fold(
    fold: dict,
    preprocessed: pd.DataFrame,
    cfg: Config,
) -> list[dict]:
    """Train on data before test_start, evaluate on [test_start, test_end]."""
    name       = fold["name"]
    train_end  = pd.Timestamp(fold["train_end"])
    test_start = pd.Timestamp(fold["test_start"])
    test_end   = pd.Timestamp(fold["test_end"])

    print(f"\n{'#'*80}")
    print(f"  {name}")
    print(f"  Train: Filed ≤ {train_end.date()}  |  Test: {test_start.date()} – {test_end.date()}")
    print(f"{'#'*80}")

    filed = preprocessed["Filed"]
    train_df = preprocessed[filed <= train_end].copy().reset_index(drop=True)
    test_df  = preprocessed[(filed >= test_start) & (filed <= test_end)].copy().reset_index(drop=True)

    print(f"  Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")

    if len(train_df) < 200:
        print("  ⚠ Skipping — training set too small (<200 rows).")
        return []
    if len(test_df) < 10:
        print("  ⚠ Skipping — test set too small (<10 rows). "
              "This regime may fall outside the data cutoff window.")
        return []

    # Fresh model per fold so label-encoders / fill-values don't leak across folds
    fold_model = PoliticianTradeModel(cfg, cutoff_date=fold["test_end"], horizon_months=12)

    y_train      = train_df[fold_model.target_binary]
    y_test       = test_df[fold_model.target_binary]
    y_test_cont  = test_df[fold_model.target_continuous]

    X_train = fold_model.prepare_features(train_df, is_training=True)
    X_test  = fold_model.prepare_features(test_df,  is_training=False)

    fold_model.train_xgboost(X_train, y_train)

    if fold_model.xgb_model is None:
        print("  ⚠ XGBoost training failed.")
        return []

    # Detailed threshold grid (same output format as main model)
    fold_model.evaluate_threshold_grid(
        fold_model.xgb_model, X_test, y_test, y_test_cont, name
    )

    # Feature importance
    fold_model.print_feature_importance(fold_model.xgb_model, f"XGBoost [{fold['regime']}]")

    # Collect summary rows
    y_proba = fold_model.xgb_model.predict_proba(X_test)[:, 1]
    rows = []
    for thr in REPORT_THRESHOLDS:
        mask = y_proba >= thr
        n    = int(mask.sum())
        mean_car = float(y_test_cont[mask].mean()) if n > 0 else float("nan")
        rows.append(
            {
                "fold":          fold["regime"],
                "threshold":     thr,
                "n_selected":    n,
                "mean_car":      round(mean_car, 4),
                "pct_selected":  round(n / len(y_test) * 100, 1),
                "test_size":     len(test_df),
                "base_hit_rate": round(float((y_test == 1).mean()), 4),
                "base_mean_car": round(float(y_test_cont.mean()), 4),
            }
        )
    return rows


def main():
    cfg = Config(save_plots=False)  # no plot files during validation

    preprocessed = load_and_preprocess(cfg)

    print(f"\nPreprocessed dataset spans "
          f"{preprocessed['Filed'].min().date()} – {preprocessed['Filed'].max().date()} "
          f"({len(preprocessed)} rows)")

    all_rows: list[dict] = []
    for fold in FOLDS:
        rows = run_fold(fold, preprocessed, cfg)
        all_rows.extend(rows)

    # ── Final summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("WALK-FORWARD SUMMARY — XGBoost classifier across 3 macro regimes")
    print(f"{'='*90}")

    if not all_rows:
        print("No results — all folds were skipped.")
        return

    summary = pd.DataFrame(all_rows)

    # Pivot for readability: rows = threshold, cols = fold
    for col in ["n_selected", "mean_car", "base_mean_car", "base_hit_rate"]:
        pivot = summary.pivot(index="threshold", columns="fold", values=col)
        print(f"\n── {col} ──")
        print(pivot.to_string())

    print(f"\n── test_size per fold ──")
    sizes = summary.groupby("fold")[["test_size", "base_hit_rate", "base_mean_car"]].first()
    print(sizes.to_string())

    print(f"\n{'='*90}")
    print("Done.")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
