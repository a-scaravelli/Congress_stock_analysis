# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

All steps are orchestrated through `src/main.py`. Always run from the **project root** (the working directory is set automatically by main.py via `os.chdir(PROJECT_ROOT)`).

```bash
source venv/bin/activate

# Run all steps
python src/main.py

# Run specific steps
python src/main.py --steps 3
python src/main.py --steps 7 9

# Run from a specific step onwards
python src/main.py --start 3
```

Run a model directly (step 9 equivalent):
```bash
python src/model/model_realized.py   # realized CAR model (primary)
python src/model/model.py            # fixed-window CAR models (1m/3m/6m/9m/12m)
```

Execute the EDA notebook:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/eda_features_vs_target.ipynb
```

There are no automated tests or linters configured.

## Architecture

### Pipeline steps (defined in `config/execution_order.yaml`, wired in `src/main.py`)

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `ingestion/download_quant_files.py` | Downloads congressional trades from QuiverQuant → `data/trades/` |
| 2 | `ingestion/extract_stock_data.py` | Downloads historical OHLC from Yahoo Finance per ticker → `data/stocks/` |
| 3 | `analysis/stock_performance_analysis.py` | Calculates CAR (1m/3m/6m/9m/12m), beta, momentum, volatility, **realized_car** (buy-to-sell) → `data/trades/stock_performance_analysis.parquet` |
| 4 | `ingestion/committes_data_extractor.py` | Scrapes committee membership YAML snapshots from GitHub → `data/committees/` |
| 5 | `ingestion/get_stock_informations.py` | Enriches stock metadata with sector/industry via yahooquery → `data/stocks/metadata_enriched.json` |
| 6 | **MANUAL** | Edit `config/commette_industry_map.yaml` to map committees to industries/tiers |
| 7 | `analysis/industry_matching.py` | Joins trades + committee membership + industry map → `data/output/politician_trades_enriched.parquet` (main dataset, ~39k rows, 107 cols) |
| 8 | `ingestion/download_lobbying_data.py` | Downloads lobbying disclosures per ticker from QuiverQuant → `data/lobbying/lobbying_data.parquet` |
| 9 | `model/model_realized.py` | Trains XGBoost + LightGBM on realized_car_hybrid target → `data/output/xgboost_model_realized.json` |

### Key data files

- `data/output/politician_trades_enriched.parquet` — the central dataset produced by step 7; input to all models and notebooks
- `data/model/eda_features.parquet` — preprocessed model-ready dataset saved by `notebooks/eda_features_vs_target.ipynb`
- `config/quiverquant.json` — must contain `{"Authorization Token": "<token>"}` for steps 1 and 8
- `config/commette_industry_map.yaml` — manual mapping of committee codes → industry sectors + tier (1/2/3)
- `data/ticker_mapping.csv` — M&A/rename mapping for tickers that changed symbols
- `data/failed_tickers_map.csv` — categorised map of tickers that failed to download

### Model architecture (`src/model/model_realized.py`)

The primary model uses a **time-based train/test split (80/20)** — never random split, always chronological. The target variable `realized_car_hybrid` is:
- `realized_car` (actual buy-to-sell CAR) when the position was closed within 12 months
- `car_filed_to_12m` (fixed-window CAR) as fallback for positions held longer or not sold

A cutoff date is enforced: `max(Traded) - 12 months` — trades after this are excluded to prevent lookahead bias on the 12m window.

Feature engineering runs on the **full dataset before the purchase filter** for sell-pressure signals (`all_pol_sells_same_ticker_30d`, `politician_recent_sells_15d`), then filters to purchases only. Politician skill metrics (`politician_hit_rate_past`, `politician_mean_car_past`) use strict no-lookahead rolling logic.

`model.py` is the older multi-horizon variant (runs all 5 fixed windows). `model_realized.py` is the current primary model.

### CAR calculation (`src/analysis/stock_performance_analysis.py`)

Written in **Polars** (not pandas). Key design decisions:
- CAR windows use `Open_Price` (not Close) — accounts for the disclosure lag
- Beta is calculated from `Close_Price` (756 trading days lookback, min 60 days)
- Stock features are filtered to S&P 500 trading dates to exclude weekend/holiday artefacts from bfill resampling
- CAR windows start the day **after** the reference date (exclusive)
- `realized_car` tracks actual buy-to-sell return matched by `Ticker + BioGuideID`, sell must be within 12 months

### Industry matching (`src/analysis/industry_matching.py`)

Also written in **Polars**. Committee membership is reconstructed from timestamped YAML snapshots in `data/committees/`. The snapshot closest in time to each trade's filed date is used to determine which committees the politician belonged to at trade time. Committee codes are truncated to 4 characters to normalise across snapshot formats.

### Notebooks

- `notebooks/eda_features_vs_target.ipynb` — comprehensive EDA of all features vs the model target; imports `PoliticianTradeModel` and `Config` directly from `model_realized.py` to reproduce the exact preprocessing pipeline
- `notebooks/EDA.ipynb`, `notebooks/performance_analysis.ipynb` — older exploratory notebooks

## Model findings & experiments

### Deduplication (step 3)
The raw trades data (`data/trades/congress_trades_full.parquet`) contains duplicate rows — same person, ticker, trade date, and filed date — due to STOCK Act tranche splits and duplicate filings. **~18k rows removed** (109k → 91k) after dedup was added at the top of `stock_performance_analysis.py`. Logic: group by `(BioGuideID, Ticker, Traded, Transaction)`, sum `Trade_Size_USD`, keep earliest `Filed` date. This reduced purchases from ~19k to ~17.8k and the test set from 3,025 to 2,806 rows.

### Classifier performance (post-dedup, 12m horizon, stable final sort)
The XGBoost classifier (`model_realized.py`) only generates reliable positive alpha at **high thresholds**:
- Sweet spot: **thr=0.86–0.87** (mean_CAR +0.010–+0.018, N=74–87)
- thr=0.84: N=118, mean_CAR=-0.007 (below threshold, avoid)
- thr=0.88–0.90: signal weakens (N too small)
- Below 0.86 mean_CAR is negative

**Hardcoded params** (from GridSearchCV with stable final preprocess sort, no new features):
`colsample_bytree=0.9, learning_rate=0.1, max_depth=7, min_child_weight=1, n_estimators=300, subsample=0.7`

**Sort stability note**: The final sort in `preprocess` is stable (`[Filed, BioGuideID, Ticker]`) to lock the train/test split boundary. The internal sorts inside `_add_engineered_features` (lines 497, 648) remain unstable (`sort_values(date_column)` only) to preserve the feature value distribution the params were tuned against. Do NOT make internal sorts stable without re-running GridSearchCV.

### Two-layer architecture (classifier + regressor)
A second-layer `XGBRegressor` (absoluteerror objective, Spearman-r optimized) was added. Standalone regressor performance is weak (R²≈-0.08, Spearman ρ≈0.03, p>0.10 — not significant). It adds value **only when AND-gated with clf≥0.84**:
- Best AND gate: clf=0.75 + reg>0.10 → N=37, mean_CAR=+0.074
- Section B ranker (top 10% of clf=0.84) → N=10, mean_CAR≈+0.12 (very small N)
- Do NOT use regressor ranking below clf=0.80 — it degrades results

### Feature reduction experiment (failed)
Tried dropping 6 features with importance < 0.025: `log_trade_size`, `car_traded_to_filed`, `stock_momentum_90d`, `ticker_prior_buys`, `Party`, `lag_bucket`, `Industry match 1`, `Industry match 2`. Result: **worse performance** — N at thr=0.84 collapsed from 100 → 47, mean_CAR from +0.014 → +0.002. The uniform importance distribution (0.025–0.050 across all features) reflects a genuinely flat signal landscape, not noise. **Do not drop features based on importance cutoff.** Commented-out features are preserved in `self.dropped_low_importance_features` for reference.

### Raw committee columns already excluded
The ~40 raw `Committee_XXXX` binary columns are **not** in the model feature lists. Only the 9 semantic groups (`committee_finance_housing`, `committee_oversight`, etc.) are used. This was already the case before any experiments.

### Next ideas to try
- Target encoding for `Ticker`, `Ticker_Sector`, `Ticker_Industry` (instead of label encoding)
- Better politician skill metrics (longer window, decay weighting)
- Sector/time filters on top of clf threshold (Communication Services and Energy sectors outperform at clf=0.70+)

### Regressor improvement ideas
Current regressor weakness: Spearman ρ≈0.03 (p=0.14, not significant). Ideas to improve:
1. ~~**Winsorize target at ±2 std**~~ — **tested, made things worse**. Spearman ρ dropped from 0.030 → 0.015 (p=0.44). The extreme positive values are actual signal the regressor learns from — clipping them destroyed ranking ability. The problem is weak features, not outliers.
2. ~~Train only on classifier positives~~ — **rejected**, regressor must train on full dataset.
3. **Better magnitude features** — stock momentum at 3m/6m, volatility-adjusted CAR, time since last trade on same ticker, sector momentum at trade time.
4. **Change objective to ranking** — LightGBM `lambdarank` or pairwise ranking directly optimises the rank problem rather than point prediction.
5. **Two separate regressors** — one trained on winners only (positive CAR), one on losers; combine to rank up winners and penalise predicted big losers.

## Important conventions

- **All scripts assume CWD = project root.** Relative paths like `data/output/...` are used throughout. Running scripts from any other directory will break file resolution.
- **Decimal-comma parsing**: several columns in the enriched parquet are stored as strings with comma as decimal separator (e.g. `"0,013..."`). The `_parse_decimal_comma()` method in `model_realized.py` handles this and must be applied before numeric operations.
- **Polars vs pandas**: ingestion and analysis steps use Polars; model and notebook code uses pandas. Do not mix within a single processing chain without explicit conversion.
- The `config/commette_industry_map.yaml` key is intentionally misspelled (`commette` not `committee`) — this is consistent across the codebase.
