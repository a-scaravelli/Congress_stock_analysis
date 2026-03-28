"""src/model/model_realized.py

Train XGBoost + LightGBM classifiers on politician trades (Purchases only)
with time-based split and multi-threshold evaluation.

Uses REALIZED CAR as target: actual returns from buy to sell (or 12m if not sold).
Cutoff date: max(Traded) - 12 months to ensure sufficient forward-looking data.

Target variable logic:
- Primary: realized_car (actual CAR from buy to sell date, if sold ≤12 months)
- Fallback: car_filed_to_12m (if no sell within 12 months or held longer)

Key features:
- Politician hit rate (past trades only)
- Politician mean CAR (past, horizon-aware)
- Politician sells same ticker 50D
- Stock momentum and volatility
- Committee rank and majority status
- Holding period and position closure signals

Error analysis with continuous alpha stats for FPs/TPs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Mapping from LDA issue name substrings to the 9 committee groups.
# Keys are lowercase substrings matched against the full issue description from QuiverQuant.
# Order matters: first match wins per issue string.
LDA_ISSUE_TO_COMMITTEE_GROUP: Dict[str, str] = {
    # committee_defense_security
    "defense":                  "committee_defense_security",
    "foreign relations":        "committee_defense_security",
    "intelligence":             "committee_defense_security",
    "homeland security":        "committee_defense_security",
    "aerospace":                "committee_defense_security",
    # committee_finance_housing
    "banking":                  "committee_finance_housing",
    "financial institutions":   "committee_finance_housing",
    "housing":                  "committee_finance_housing",
    "insurance":                "committee_finance_housing",
    "real estate":              "committee_finance_housing",
    "bankruptcy":               "committee_finance_housing",
    "minting/money":            "committee_finance_housing",
    # committee_fiscal_policy
    "taxation":                 "committee_fiscal_policy",
    "budget/appropriations":    "committee_fiscal_policy",
    "economics":                "committee_fiscal_policy",
    "trade (domestic":          "committee_fiscal_policy",
    "tariff":                   "committee_fiscal_policy",
    "commodities":              "committee_fiscal_policy",
    # committee_energy_environment
    "energy/nuclear":           "committee_energy_environment",
    "environment/superfund":    "committee_energy_environment",
    "fuel/gas/oil":             "committee_energy_environment",
    "clean air":                "committee_energy_environment",
    "natural resources":        "committee_energy_environment",
    "utilities":                "committee_energy_environment",
    "waste (hazardous":         "committee_energy_environment",
    # committee_health_labor
    "health issues":            "committee_health_labor",
    "medicare/medicaid":        "committee_health_labor",
    "medical/disease":          "committee_health_labor",
    "pharmacy":                 "committee_health_labor",
    "labor issues":             "committee_health_labor",
    "alcohol and drug":         "committee_health_labor",
    "retirement":               "committee_health_labor",
    "unemployment":             "committee_health_labor",
    "veterans":                 "committee_health_labor",
    "torts":                    "committee_health_labor",
    "family issues":            "committee_health_labor",
    "manufacturing":            "committee_health_labor",
    # committee_commerce_technology
    "communications":           "committee_commerce_technology",
    "telecommunications":       "committee_commerce_technology",
    "computer industry":        "committee_commerce_technology",
    "science/technology":       "committee_commerce_technology",
    "copyright/patent":         "committee_commerce_technology",
    "consumer issues":          "committee_commerce_technology",
    "advertising":              "committee_commerce_technology",
    "media (information":       "committee_commerce_technology",
    "sports/athletics":         "committee_commerce_technology",
    "automotive industry":      "committee_commerce_technology",
    "chemicals":                "committee_commerce_technology",
    "beverage industry":        "committee_commerce_technology",
    "tobacco":                  "committee_commerce_technology",
    "small business":           "committee_commerce_technology",
    "food industry":            "committee_commerce_technology",
    # committee_agriculture
    "agriculture":              "committee_agriculture",
    "animals":                  "committee_agriculture",
    "apparel/clothing":         "committee_agriculture",
    # committee_infrastructure
    "transportation":           "committee_infrastructure",
    "roads/highway":            "committee_infrastructure",
    "aviation/airlines":        "committee_infrastructure",
    "marine/maritime":          "committee_infrastructure",
    "railroads":                "committee_infrastructure",
    "urban development":        "committee_infrastructure",
    "trucking/shipping":        "committee_infrastructure",
    "postal":                   "committee_infrastructure",
    "travel/tourism":           "committee_infrastructure",
    "disaster planning":        "committee_infrastructure",
    # committee_oversight
    "government issues":        "committee_oversight",
    "accounting":               "committee_oversight",
    "law enforcement":          "committee_oversight",
    "civil rights":             "committee_oversight",
    "constitution":             "committee_oversight",
    "immigration":              "committee_oversight",
    "firearms":                 "committee_oversight",
    "gaming/gambling":          "committee_oversight",
    "indian/native":            "committee_oversight",
    "arts/entertainment":       "committee_oversight",
}


@dataclass
class Config:
    data_path: str = "data/output/politician_trades_enriched.parquet"
    train_ratio: float = 0.8
    alpha_threshold: float = 0.0
    prob_thresholds: List[float] = field(default_factory=lambda: [round(t, 2) for t in __import__('numpy').arange(0.60, 0.91, 0.02)])
    reg_car_thresholds: List[float] = field(default_factory=lambda: [round(t, 2) for t in __import__('numpy').arange(0.00, 0.21, 0.02)])
    do_grid_search: bool = False  # use best_known_params; set True to re-run GridSearchCV
    tscv_splits: int = 3
    grid_scoring: str = "precision"
    save_plots: bool = False


class PoliticianTradeModel:
    def __init__(self, cfg: Config, cutoff_date: str, horizon_months: int = 12):
        self.cfg = cfg
        self.cutoff_date = cutoff_date
        self.horizon_months = horizon_months
        self.horizon_days = int(horizon_months * 30.5)  # Approximate days
        self.xgb_model = None
        self.lgbm_model = None
        self.xgb_regressor = None
        self.lgbm_ranker = None
        self.label_encoders: Dict[str, object] = {}
        self.feature_names: Optional[List[str]] = None
        self.numeric_fill_values: Dict[str, float] = {}

        # Features removed (importance < 0.025) — re-add to lists above to restore
        self.dropped_low_importance_features = [
            "log_trade_size",       # 0.0255 — borderline
            "car_traded_to_filed",  # 0.0234
            "stock_momentum_90d",   # 0.0247 — borderline
            "ticker_prior_buys",    # 0.0246 — borderline
        ]

        self.numerical_features = [
            "log_trade_size", "car_traded_to_filed",
            "politician_trades_last_year", "politician_hit_rate_past",
            "politician_mean_car_past", "politician_mean_realized_car_past",
            "stock_momentum_90d", "stock_volatility_30d",
            "beta", "max_committee_rank", "all_pol_sells_same_ticker_30d",
            "n_committees", "ticker_prior_buys",
            # dropped_low_importance_features (restored): see self.dropped_low_importance_features
        ]

        # Extra features used ONLY by the regressor (magnitude-focused signals)
        self.regressor_extra_features = [
            "politician_mean_realized_car_past_positive",
            "politician_win_loss_magnitude_ratio",
            "politician_realized_car_std",
            "sp500_momentum_90d",
            "sp500_volatility_30d",
            "relative_trade_size",           # this trade vs politician's own median past size
            "sector_momentum_90d",           # sector ETF 90d momentum at filed date
            "all_pol_buys_same_ticker_30d",  # buy herding: other politicians same ticker 30d
        ]

        self.categorical_features = [
            "Party", "Chamber", "Ticker_Sector",
            "Ticker_Industry", "ticker_filed_density_50",
            "lag_bucket",
            # "Party",       # low importance: 0.0253
            # "lag_bucket",  # low importance: 0.0233
        ]

        self.binary_features = [
            "Industry match 1", "Industry match 2", "Industry match 3",
            # "Industry match 1",  # low importance: 0.0236
            # "Industry match 2",  # low importance: 0.0180
            "is_committee_majority", "is_committee_chair",
            # Committee category flags (9 semantic groups from config)
            "committee_defense_security",
            "committee_finance_housing",
            "committee_fiscal_policy",
            "committee_energy_environment",
            "committee_health_labor",
            "committee_commerce_technology",
            "committee_agriculture",
            "committee_infrastructure",
            "committee_oversight",
            # Lobbying activity signal (any lobbying by this company in 90d before trade)
            "lobbied_any_90d",
        ]

        # Committee category → constituent committee codes (from commette_industry_map.yaml)
        self.committee_categories = {
            "committee_defense_security":    ["Committee_HSAS", "Committee_SSAS", "Committee_HLIG", "Committee_SSFR"],
            "committee_finance_housing":     ["Committee_HSBA", "Committee_SSBK"],
            "committee_fiscal_policy":       ["Committee_HSWM", "Committee_SSFI"],
            "committee_energy_environment":  ["Committee_HSIF", "Committee_SSEG", "Committee_SSEV"],
            "committee_health_labor":        ["Committee_SSHR", "Committee_HSED"],
            "committee_commerce_technology": ["Committee_SSCM", "Committee_HSJU"],
            "committee_agriculture":         ["Committee_HSAG", "Committee_SSAF"],
            "committee_infrastructure":      ["Committee_HSPW"],
            "committee_oversight":           ["Committee_HSGO"],
        }

        self.date_column = "Filed"
        self.target_continuous = "realized_car_hybrid"
        self.target_binary = "alpha_above_threshold"

    @staticmethod
    def _parse_decimal_comma(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        s = s.replace({"None": np.nan, "nan": np.nan, "NaN": np.nan, "<NA>": np.nan})

        s = (s.str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace(" ", "", regex=False))

        mask_both = s.str.contains(r"\.") & s.str.contains(",")
        s.loc[mask_both] = s.loc[mask_both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        mask_comma_only = s.str.contains(",") & (~s.str.contains(r"\."))
        s.loc[mask_comma_only] = s.loc[mask_comma_only].str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _calculate_sells_pressure(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Calculate politician sell pressure using full dataset. Gap-independent."""
        print("Calculating politician sell pressure (using full dataset)...")

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df["Traded"] = pd.to_datetime(df["Traded"], errors="coerce")

        df = df.sort_values(date_column).reset_index(drop=True)
        df["_temp_id"] = range(len(df))

        sells = df[df["Transaction"].str.contains("Sale", na=False)].copy()
        if len(sells) == 0:
            df["politician_recent_sells_15d"] = 0.0
            print("No sell transactions found.")
            return df.drop(columns=["_temp_id"])

        sells_idx = sells.set_index([date_column, "BioGuideID"]).sort_index()

        def count_prior_sells(row):
            try:
                bio = row["BioGuideID"]
                filed_date = row[date_column]
                prior_start = filed_date - pd.Timedelta(days=15)
                prior_slice = sells_idx.loc[(slice(prior_start, filed_date), bio), :]
                return len(prior_slice)
            except (KeyError, ValueError):
                return 0

        purchase_mask = df["Transaction"] == "Purchase"
        df["politician_recent_sells_15d"] = 0.0

        if purchase_mask.sum() > 0:
            sells_counts = df.loc[purchase_mask].apply(count_prior_sells, axis=1)
            df.loc[purchase_mask, "politician_recent_sells_15d"] = sells_counts.values

        print(f"Sell pressure stats: {df['politician_recent_sells_15d'].describe()}")
        return df.drop(columns=["_temp_id"])

    @staticmethod
    def _calculate_all_pol_sells_same_ticker(df: pd.DataFrame, date_column: str, window_days: int = 30) -> pd.DataFrame:
        """Count how many Congress members sold the same ticker in the N days before each purchase.
        Must be called on the full dataset (sells + purchases) before filtering to purchases."""
        print(f"Calculating all-politician sells same ticker ({window_days}d window)...")

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        col = f"all_pol_sells_same_ticker_{window_days}d"

        sells = df[df["Transaction"].str.contains("Sale", na=False)].copy()
        if len(sells) == 0:
            df[col] = 0.0
            print("No sell transactions found.")
            return df

        sells_idx = sells.set_index([date_column, "Ticker"]).sort_index()

        def count_prior_sells(row):
            try:
                ticker = row["Ticker"]
                d = row[date_column]
                start = d - pd.Timedelta(days=window_days)
                sl = sells_idx.loc[(slice(start, d - pd.Timedelta(seconds=1)), ticker), :]
                return len(sl)
            except (KeyError, ValueError):
                return 0

        purchase_mask = df["Transaction"] == "Purchase"
        df[col] = 0.0
        if purchase_mask.sum() > 0:
            counts = df.loc[purchase_mask].apply(count_prior_sells, axis=1)
            df.loc[purchase_mask, col] = counts.values

        print(f"All-pol sells same ticker stats: {df[col].describe()}")
        return df

    def _add_committee_category_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse 58 sparse Committee_XXXX binary columns into 9 semantic category flags."""
        print("Adding committee category flags (9 semantic groups)...")
        for flag_name, cols in self.committee_categories.items():
            present = [c for c in cols if c in df.columns]
            if present:
                df[flag_name] = df[present].max(axis=1).fillna(0).astype(int)
            else:
                df[flag_name] = 0
        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add S&P 500 market regime features: 90d momentum and 30d volatility.

        Fetched directly via yfinance — no pipeline step needed.
        Both features are point-in-time at the Filed date to avoid lookahead bias.
        """
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed — skipping market regime features.")
            df["sp500_momentum_90d"] = 0.0
            df["sp500_volatility_30d"] = 0.0
            return df

        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        min_date = df[self.date_column].min() - pd.Timedelta(days=120)
        max_date = df[self.date_column].max() + pd.Timedelta(days=1)

        print(f"Fetching S&P 500 data ({min_date.date()} to {max_date.date()})...")
        sp = yf.download("^GSPC", start=min_date.strftime("%Y-%m-%d"),
                         end=max_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

        if sp.empty:
            print("S&P 500 data unavailable — skipping market regime features.")
            df["sp500_momentum_90d"] = 0.0
            df["sp500_volatility_30d"] = 0.0
            return df

        close = sp["Close"].squeeze()
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()

        daily_returns = close.pct_change()

        # Point-in-time features indexed by calendar date (forward-filled for weekends)
        date_index = pd.date_range(min_date, max_date, freq="D")
        close_daily = close.reindex(date_index).ffill()
        returns_daily = daily_returns.reindex(date_index).ffill()

        sp500_mom = np.log(close_daily / close_daily.shift(90))
        sp500_vol = returns_daily.rolling(30).std() * np.sqrt(252)

        df["sp500_momentum_90d"] = df[self.date_column].map(sp500_mom).astype(float)
        df["sp500_volatility_30d"] = df[self.date_column].map(sp500_vol).astype(float)

        n_valid = df["sp500_momentum_90d"].notna().sum()
        print(f"Market regime features added ({n_valid}/{len(df)} valid rows).")
        return df

    def _add_sector_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sector-level 90d momentum at filed date using SPDR sector ETFs.

        Maps Ticker_Sector → ETF ticker, computes log return over 90 days prior
        to Filed date. Captures sector trend as a magnitude signal for the regressor.
        """
        SECTOR_ETF_MAP = {
            "Technology":             "XLK",
            "Financial Services":     "XLF",
            "Healthcare":             "XLV",
            "Consumer Defensive":     "XLP",
            "Consumer Cyclical":      "XLY",
            "Communication Services": "XLC",
            "Utilities":              "XLU",
            "Industrials":            "XLI",
            "Basic Materials":        "XLB",
            "Energy":                 "XLE",
            "Real Estate":            "XLRE",
        }

        try:
            import yfinance as yf
        except ImportError:
            print("yfinance not installed — skipping sector momentum features.")
            df["sector_momentum_90d"] = 0.0
            return df

        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        min_date = df[self.date_column].min() - pd.Timedelta(days=120)
        max_date = df[self.date_column].max() + pd.Timedelta(days=1)

        sectors_in_data = df["Ticker_Sector"].dropna().unique() if "Ticker_Sector" in df.columns else []
        etfs_needed = list({SECTOR_ETF_MAP[s] for s in sectors_in_data if s in SECTOR_ETF_MAP})

        if not etfs_needed:
            df["sector_momentum_90d"] = 0.0
            return df

        print(f"Fetching sector ETF data ({len(etfs_needed)} ETFs)...")
        raw = yf.download(etfs_needed, start=min_date.strftime("%Y-%m-%d"),
                          end=max_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

        if raw.empty:
            df["sector_momentum_90d"] = 0.0
            return df

        close = raw["Close"] if len(etfs_needed) > 1 else raw["Close"].to_frame(etfs_needed[0])
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()

        date_index = pd.date_range(min_date, max_date, freq="D")
        sector_mom = {}
        for sector, etf in SECTOR_ETF_MAP.items():
            if etf not in close.columns:
                continue
            c = close[etf].reindex(date_index).ffill()
            sector_mom[sector] = np.log(c / c.shift(90))

        df["sector_momentum_90d"] = 0.0
        for sector, mom_series in sector_mom.items():
            mask = df["Ticker_Sector"] == sector
            if mask.any():
                df.loc[mask, "sector_momentum_90d"] = df.loc[mask, self.date_column].map(mom_series).fillna(0.0)

        n_valid = (df["sector_momentum_90d"] != 0.0).sum()
        print(f"Sector momentum features added ({n_valid}/{len(df)} valid rows).")
        return df

    def _add_regressor_only_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regressor-only features using isolated copies.

        Deliberately does NOT modify df sort order — all computations happen on
        separate copies and are mapped back via the original RangeIndex.
        Must be called AFTER _add_engineered_features (which assigns RangeIndex).
        """
        print("Computing regressor-only features (isolated from classifier pipeline)...")
        df = df.copy()

        # --- 1. relative_trade_size ---
        # This trade's size vs the politician's own expanding median past size (no lookahead)
        work = df[[self.date_column, "BioGuideID", "Trade_Size_USD"]].copy()
        work = work.sort_values([self.date_column, "BioGuideID"]).reset_index(drop=False)
        # 'index' column now holds original row positions
        work["_ts"] = pd.to_numeric(work["Trade_Size_USD"], errors="coerce").fillna(0)

        def get_relative_size(grp):
            g = grp.sort_values(self.date_column).reset_index(drop=True)
            sizes = g["_ts"].values
            orig_ids = g["index"].values
            result = np.ones(len(g))
            for i in range(1, len(g)):
                past = sizes[:i][sizes[:i] > 0]
                if len(past) > 0:
                    med = np.median(past)
                    result[i] = sizes[i] / med if med > 0 else 1.0
            return pd.Series(result, index=orig_ids)

        rts = work.groupby("BioGuideID", group_keys=False).apply(get_relative_size)
        df["relative_trade_size"] = df.index.map(rts).fillna(1.0)

        # --- 2. sector_momentum_90d ---
        # ETF-based 90d momentum for the stock's sector at filed date (no df sorting)
        df = self._add_sector_momentum_features(df)

        # --- 3. all_pol_buys_same_ticker_30d ---
        # How many OTHER politicians bought the same ticker in the 30d before this trade
        print("Calculating buy herding (regressor feature, isolated copy)...")
        work2 = df[[self.date_column, "Ticker", "BioGuideID"]].copy()
        work2_sorted = work2.sort_values(
            [self.date_column, "BioGuideID", "Ticker"]
        ).reset_index(drop=False)
        buys_idx = work2_sorted.set_index([self.date_column, "Ticker"]).sort_index()

        def count_prior_buys(row):
            try:
                ticker = row["Ticker"]
                bio = row["BioGuideID"]
                d = row[self.date_column]
                start = d - pd.Timedelta(days=30)
                sl = buys_idx.loc[(slice(start, d - pd.Timedelta(seconds=1)), ticker), :]
                return len(sl[sl["BioGuideID"] != bio])
            except (KeyError, ValueError):
                return 0

        work2_sorted["_herding"] = work2_sorted.apply(count_prior_buys, axis=1)
        herding = work2_sorted.set_index("index")["_herding"]
        df["all_pol_buys_same_ticker_30d"] = df.index.map(herding).fillna(0).astype(int)

        print(f"  relative_trade_size: mean={df['relative_trade_size'].mean():.3f}")
        print(f"  sector_momentum_90d: valid={(df['sector_momentum_90d'] != 0).sum()}/{len(df)}")
        print(f"  all_pol_buys_same_ticker_30d: mean={df['all_pol_buys_same_ticker_30d'].mean():.2f}")
        return df

    def _add_lobbying_features(self, df: pd.DataFrame, lobbying_path: str = "data/lobbying/lobbying_data.parquet") -> pd.DataFrame:
        """Add lobbying activity feature.

        lobbied_any_90d (binary): 1 if the company filed any lobbying disclosure
        in the 90 days before the trade date, regardless of issue or committee.
        Companies that actively lobby tend to be large-cap, politically connected
        names with higher positive CAR on politician purchases (+57% vs +48%).
        """
        import os
        if not os.path.exists(lobbying_path):
            raise FileNotFoundError(f"Lobbying data not found at {lobbying_path}. Run step 8 first.")

        print("Adding lobbying activity feature (90d window, any issue)...")
        lob = pd.read_parquet(lobbying_path)
        lob["Date"] = pd.to_datetime(lob["Date"], errors="coerce")
        lob = lob.dropna(subset=["Date", "Ticker"])

        # Precompute per-ticker sorted list of lobbying dates
        ticker_dates: dict = {}
        for ticker, grp in lob.groupby("Ticker"):
            ticker_dates[ticker] = sorted(grp["Date"].tolist())

        df["Filed"] = pd.to_datetime(df["Filed"], errors="coerce")

        def lobbied_90d(row):
            ticker = row["Ticker"]
            filed_date = row["Filed"]
            if pd.isna(filed_date) or ticker not in ticker_dates:
                return 0
            window_start = filed_date - pd.Timedelta(days=90)
            return int(any(window_start <= d <= filed_date for d in ticker_dates[ticker]))

        df["lobbied_any_90d"] = df.apply(lobbied_90d, axis=1)
        pos_rate = df["lobbied_any_90d"].mean()
        print(f"Lobbying activity rate: {pos_rate:.1%} of trades have lobbying in 90d window")
        return df

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-purchase-filter features"""
        print("Adding purchase-only engineered features...")

        df["Traded"] = pd.to_datetime(df["Traded"], errors="coerce")
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        df["delta_traded_filed"] = (df[self.date_column] - df["Traded"]).dt.days

        # Filing lag bucket and STOCK Act compliance flag
        df["lag_bucket"] = pd.cut(
            df["delta_traded_filed"],
            bins=[-999, 15, 30, 45, 90, 999999],
            labels=["0-15d", "15-30d", "30-45d", "45-90d", ">90d"]
        ).astype(str)
        df["filed_within_45d"] = (df["delta_traded_filed"] <= 45).astype(int)

        df = df.sort_values(self.date_column).reset_index(drop=True)
        df["_temp_id"] = range(len(df))

        def get_rolling_count(grp, window):
            g = grp.set_index(self.date_column).sort_index()
            rc = g["_temp_id"].rolling(window, closed="left").count()
            return pd.Series(rc.values, index=g["_temp_id"].values)

        t_res = df.groupby("Ticker", group_keys=False).apply(lambda x: get_rolling_count(x, "90D"))
        df["ticker_filed_density_50"] = df["_temp_id"].map(t_res).fillna(0)

        p_res = df.groupby("BioGuideID", group_keys=False).apply(lambda x: get_rolling_count(x, "365D"))
        df["politician_trades_last_year"] = df["_temp_id"].map(p_res).fillna(0)

        # gap_offset shared by hit rate and mean CAR: a past trade's outcome is only
        # known once its filing date + horizon months has elapsed.
        gap_offset = pd.DateOffset(months=self.horizon_months)

        print("Calculating politician hit rates (horizon-aware, no lookahead)...")
        def get_hit_rate(grp):
            if len(grp) < 2:
                return pd.Series(0.0, index=grp["_temp_id"])

            g = grp.sort_values(self.date_column).reset_index(drop=True)
            filed_dates = g[self.date_column].values
            hits = (g[self.target_continuous] > self.cfg.alpha_threshold).astype(int).values
            temp_ids = g["_temp_id"].values
            result = np.zeros(len(g))

            for i in range(1, len(g)):
                current_date = pd.Timestamp(filed_dates[i])
                eligible = [hits[j] for j in range(i)
                            if pd.Timestamp(filed_dates[j]) + gap_offset <= current_date]
                if eligible:
                    result[i] = np.mean(eligible)

            return pd.Series(result, index=temp_ids)

        p_hit = df.groupby("BioGuideID", group_keys=False).apply(get_hit_rate)
        df["politician_hit_rate_past"] = df["_temp_id"].map(p_hit).fillna(0)

        # Rolling mean CAR from past trades where the outcome is actually known
        # For realized CAR model, use {horizon}-month horizon for conservative estimation
        # A past trade's CAR is only known if its filing date + {horizon} months <= current trade's filing date
        print(f"Calculating politician mean CAR (past, horizon-aware, {self.horizon_months}m horizon)...")

        def get_mean_car_past(grp):
            if len(grp) < 2:
                return pd.Series(0.0, index=grp["_temp_id"])

            g = grp.sort_values(self.date_column).reset_index(drop=True)
            filed_dates = g[self.date_column].values
            cars = g[self.target_continuous].values
            temp_ids = g["_temp_id"].values
            result = np.zeros(len(g))

            for i in range(1, len(g)):
                current_date = pd.Timestamp(filed_dates[i])
                eligible_cars = []
                for j in range(i):
                    past_date = pd.Timestamp(filed_dates[j])
                    if past_date + gap_offset <= current_date:
                        eligible_cars.append(cars[j])
                if len(eligible_cars) > 0:
                    result[i] = np.nanmean(eligible_cars)

            return pd.Series(result, index=temp_ids)

        p_mean_car = df.groupby("BioGuideID", group_keys=False).apply(get_mean_car_past)
        df["politician_mean_car_past"] = df["_temp_id"].map(p_mean_car).fillna(0)

        # Rolling realized CAR stats from past CLOSED positions only (single pass).
        # Computes 4 magnitude-focused features simultaneously:
        #   mean         — overall skill signal (existing)
        #   mean_positive — average magnitude of winning trades
        #   win_loss_ratio — how much bigger are wins than losses?
        #   std           — consistency of returns
        print("Calculating politician realized CAR stats (past closed positions, single pass)...")
        realized_car_col = self._parse_decimal_comma(df['realized_car']) if 'realized_car' in df.columns else None

        if realized_car_col is not None:
            df['_realized_car_parsed'] = realized_car_col
            df['_position_closed_bool'] = df['position_closed'].fillna(False).astype(bool)
            df['_hold_days'] = pd.to_numeric(df['holding_period_days'], errors='coerce')

            def get_realized_car_stats(grp):
                if len(grp) < 2:
                    empty = pd.Series(0.0, index=grp["_temp_id"])
                    return pd.DataFrame({
                        'mean': empty, 'mean_pos': empty,
                        'wl_ratio': empty, 'std': empty,
                    })
                g = grp.sort_values(self.date_column).reset_index(drop=True)
                realized_cars = g['_realized_car_parsed'].values
                closed = g['_position_closed_bool'].values
                filed_dates = g[self.date_column].values
                hold_days = g['_hold_days'].values
                temp_ids = g["_temp_id"].values
                r_mean = np.zeros(len(g))           # classifier feature: keep 0 default
                r_mean_pos = np.full(len(g), np.nan)  # regressor-only: NaN = no history
                r_wl_ratio = np.full(len(g), np.nan)  # regressor-only: NaN = no history
                r_std = np.full(len(g), np.nan)        # regressor-only: NaN = no history

                for i in range(1, len(g)):
                    current_date = pd.Timestamp(filed_dates[i])
                    eligible = []
                    for j in range(i):
                        if not closed[j] or np.isnan(realized_cars[j]) or np.isnan(hold_days[j]):
                            continue
                        sell_date_j = pd.Timestamp(filed_dates[j]) + pd.Timedelta(days=hold_days[j])
                        if sell_date_j < current_date:
                            eligible.append(realized_cars[j])
                    if eligible:
                        arr = np.array(eligible)
                        r_mean[i] = np.nanmean(arr)
                        wins = arr[arr > 0]
                        losses = arr[arr < 0]
                        r_mean_pos[i] = np.nanmean(wins) if len(wins) > 0 else 0.0
                        mean_win = np.nanmean(np.abs(wins)) if len(wins) > 0 else 0.0
                        mean_loss = np.nanmean(np.abs(losses)) if len(losses) > 0 else np.nan
                        r_wl_ratio[i] = mean_win / mean_loss if (mean_loss and mean_loss > 0) else 1.0
                        r_std[i] = np.nanstd(arr) if len(arr) >= 3 else 0.0

                idx = temp_ids
                return pd.DataFrame({
                    'mean':     pd.Series(r_mean, index=idx),
                    'mean_pos': pd.Series(r_mean_pos, index=idx),
                    'wl_ratio': pd.Series(r_wl_ratio, index=idx),
                    'std':      pd.Series(r_std, index=idx),
                })

            stats = df.groupby("BioGuideID", group_keys=False).apply(get_realized_car_stats)
            df["politician_mean_realized_car_past"]          = df["_temp_id"].map(stats['mean']).fillna(0)  # classifier: keep 0
            df["politician_mean_realized_car_past_positive"] = df["_temp_id"].map(stats['mean_pos'])          # regressor: NaN = no history
            df["politician_win_loss_magnitude_ratio"]        = df["_temp_id"].map(stats['wl_ratio'])          # regressor: NaN = no history
            df["politician_realized_car_std"]                = df["_temp_id"].map(stats['std'])               # regressor: NaN = no history
            df = df.drop(columns=["_realized_car_parsed", "_position_closed_bool", "_hold_days"])
        else:
            df["politician_mean_realized_car_past"] = 0.0          # classifier: keep 0
            df["politician_mean_realized_car_past_positive"] = np.nan  # regressor: NaN = no history
            df["politician_win_loss_magnitude_ratio"] = np.nan         # regressor: NaN = no history
            df["politician_realized_car_std"] = np.nan                 # regressor: NaN = no history

        # log(Trade_Size_USD) — captures non-linear conviction signal
        df["log_trade_size"] = np.log1p(pd.to_numeric(df["Trade_Size_USD"], errors="coerce").fillna(0))

        # Number of committees the politician sits on
        committee_cols = [c for c in df.columns if c.startswith("Committee_")]
        df["n_committees"] = df[committee_cols].clip(upper=1).sum(axis=1)

        # Prior buys of this exact ticker by this politician (no lookahead)
        df = df.sort_values(self.date_column).reset_index(drop=True)
        df["_temp_id2"] = range(len(df))

        def get_ticker_prior_buys(grp):
            g = grp.sort_values(self.date_column).reset_index(drop=True)
            counts = np.arange(len(g))  # 0 for first buy, 1 for second, etc.
            return pd.Series(counts, index=g["_temp_id2"].values)

        pb = df.groupby(["BioGuideID", "Ticker"], group_keys=False).apply(get_ticker_prior_buys)
        df["ticker_prior_buys"] = df["_temp_id2"].map(pb).fillna(0)

        return df.drop(columns=["_temp_id", "_temp_id2"])

    def preprocess(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Take pre-loaded purchase data and create hybrid realized CAR target."""
        df = base_df.copy()

        # Apply cutoff
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        df = df.dropna(subset=[self.date_column])

        cutoff = pd.Timestamp(self.cutoff_date)
        df = df[df[self.date_column] <= cutoff].copy()

        # Parse realized_car and car_filed_to_{horizon}m
        car_col = f'car_filed_to_{self.horizon_months}m'
        df['realized_car'] = self._parse_decimal_comma(df['realized_car'])
        df[car_col] = self._parse_decimal_comma(df[car_col])
        df['holding_period_days'] = pd.to_numeric(df['holding_period_days'], errors='coerce')

        # Create hybrid target: use realized_car ONLY if sold within horizon
        # Otherwise use car_filed_to_{horizon}m
        mask_use_realized = (
            (df['position_closed'] == True) &
            (df['holding_period_days'] <= self.horizon_days) &
            (df['realized_car'].notna())
        )

        # Initialize with car_filed_to_{horizon}m as default
        df[self.target_continuous] = df[car_col]

        # Override with realized_car where applicable
        df.loc[mask_use_realized, self.target_continuous] = df.loc[mask_use_realized, 'realized_car']

        # Remove rows where target is still null
        df = df.dropna(subset=[self.target_continuous]).copy()

        # Parse decimal-comma columns that flow through CSV
        for col in ["car_traded_to_filed", "beta",
                     "stock_momentum_30d", "stock_momentum_90d", "stock_volatility_30d",
                     "max_committee_rank", "holding_period_days"]:
            if col in df.columns:
                df[col] = self._parse_decimal_comma(df[col])

        # Add committee category flags (9 semantic groups, collapsing 58 sparse columns)
        df = self._add_committee_category_flags(df)

        # Add lobbying-based committee overlap features (requires committee flags to be set)
        df = self._add_lobbying_features(df)

        # Add market regime features (S&P 500 momentum + volatility at trade date)
        df = self._add_market_regime_features(df)

        # Sector momentum features reverted — caused classifier instability via sort order side effects
        # df = self._add_sector_momentum_features(df)

        # Add engineered features (hit rate depends on target)
        df = self._add_engineered_features(df)

        # Add regressor-only features (isolated copies — does not affect classifier sort order)
        df = self._add_regressor_only_features(df)

        # Create binary target
        df[self.target_binary] = (df[self.target_continuous] > self.cfg.alpha_threshold).astype(int)

        df = df.sort_values([self.date_column, "BioGuideID", "Ticker"]).reset_index(drop=True)

        # Calculate which target was used
        using_realized = mask_use_realized.sum()
        using_12m = len(df) - using_realized

        print(f"Final dataset: {len(df)} purchases")
        print(f"   Target: {self.target_continuous} (realized_car if sold ≤{self.horizon_days}d, else car_filed_to_{self.horizon_months}m)")
        print(f"   Using realized_car (sold ≤{self.horizon_months}m): {using_realized} ({using_realized/len(df)*100:.1f}%)")
        print(f"   Using car_filed_to_{self.horizon_months}m (held >{self.horizon_months}m or not sold): {using_12m} ({using_12m/len(df)*100:.1f}%)")
        print(f"   Alpha stats: {df[self.target_continuous].describe().round(3)}")
        print(f"   Positive rate: {(df[self.target_binary]==1).mean():.1%}")

        return df

    def prepare_features(self, df: pd.DataFrame, *, is_training: bool) -> pd.DataFrame:
        """Prepare features for XGBoost (LabelEncoded categoricals)."""
        from sklearn.preprocessing import LabelEncoder
        X = pd.DataFrame(index=df.index)

        for col in self.numerical_features:
            if col not in df.columns:
                X[col] = 0.0
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            if is_training:
                fill = float(series.median() if not series.median() != series.median() else 0.0)
                self.numeric_fill_values[col] = fill
            else:
                fill = float(self.numeric_fill_values.get(col, 0.0))
            X[col] = series.fillna(fill)

        for col in self.categorical_features:
            if col not in df.columns:
                X[col] = pd.Categorical([-1])
                continue
            s = df[col].fillna("Unknown").astype(str)
            if is_training:
                le = LabelEncoder()
                encoded = le.fit_transform(s)
                self.label_encoders[col] = le
                X[col] = pd.Categorical(encoded)
            else:
                le = self.label_encoders[col]
                encoded = s.map(lambda v: int(le.transform([v])[0]) if v in le.classes_ else -1)
                X[col] = pd.Categorical(encoded)

        for col in self.binary_features:
            if col not in df.columns:
                X[col] = 0
                continue
            X[col] = df[col].fillna(0).astype(int)

        if is_training:
            self.feature_names = X.columns.tolist()
        return X

    def prepare_features_regressor(self, df: pd.DataFrame, *, is_training: bool) -> pd.DataFrame:
        """Prepare features for the regressor: classifier features + regressor_extra_features."""
        X = self.prepare_features(df, is_training=is_training)

        for col in self.regressor_extra_features:
            if col not in df.columns:
                X[col] = 0.0
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            if is_training:
                fill = float(series.median()) if series.notna().any() else 0.0
                self.numeric_fill_values[col] = fill
            else:
                fill = float(self.numeric_fill_values.get(col, 0.0))
            X[col] = series.fillna(fill)

        return X

    def prepare_features_native_cat(self, df: pd.DataFrame, *, is_training: bool):
        """Prepare features for LightGBM (native categorical support).

        Returns (X, cat_feature_names).
        """
        X = pd.DataFrame(index=df.index)

        for col in self.numerical_features:
            if col not in df.columns:
                X[col] = 0.0
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            fill = float(self.numeric_fill_values.get(col, 0.0))
            X[col] = series.fillna(fill)

        for col in self.categorical_features:
            if col not in df.columns:
                X[col] = pd.Categorical(["Unknown"])
                continue
            X[col] = pd.Categorical(df[col].fillna("Unknown").astype(str))

        for col in self.binary_features:
            if col not in df.columns:
                X[col] = 0
                continue
            X[col] = df[col].fillna(0).astype(int)

        cat_names = [c for c in self.categorical_features if c in X.columns]
        return X, cat_names

    def time_split(self, df: pd.DataFrame):
        split_idx = int(len(df) * self.cfg.train_ratio)
        split_dt = df[self.date_column].iloc[split_idx]
        train_df = df[df[self.date_column] < split_dt].copy()
        test_df = df[df[self.date_column] >= split_dt].copy()
        print(f"\nTime split at {split_dt.date()} | Train: {len(train_df)} | Test: {len(test_df)}")
        return train_df, test_df

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series):
        try:
            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        except ImportError:
            print("XGBoost not installed.")
            return

        y_train = y_train.astype(int)
        pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        base_score = float(np.clip(pos / len(y_train), 0.01, 0.99))

        fixed_params = dict(
            objective="binary:logistic", eval_metric="logloss", random_state=42,
            n_jobs=-1, tree_method="hist", scale_pos_weight=scale_pos_weight, base_score=base_score,
            enable_categorical=True
        )

        # Best params from GridSearchCV run (stable final sort, no new features)
        # Sweet spot: thr=0.86–0.87, N≈74–87, mean_CAR≈+0.010–+0.018
        best_known_params = dict(
            colsample_bytree=0.9, learning_rate=0.1, max_depth=7,
            min_child_weight=1, n_estimators=300, subsample=0.7
        )
        if not self.cfg.do_grid_search:
            self.xgb_model = xgb.XGBClassifier(**fixed_params, **best_known_params)
            self.xgb_model.fit(X_train, y_train)
            return

        param_grid = {
            "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [300, 600], "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9], "min_child_weight": [1, 5]
        }

        base = xgb.XGBClassifier(**fixed_params)
        tscv = TimeSeriesSplit(n_splits=self.cfg.tscv_splits)

        print(f"\nXGBoost GridSearchCV ({self.cfg.grid_scoring})...")
        grid = GridSearchCV(base, param_grid, scoring=self.cfg.grid_scoring, cv=tscv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        self.xgb_model = grid.best_estimator_

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, cat_feature_names: list):
        try:
            import lightgbm as lgb
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        except ImportError:
            print("LightGBM not installed.")
            return

        y_train = y_train.astype(int)
        pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        # LightGBM auto-detects 'category' dtype columns
        X_train_lgb = X_train.copy()
        for col in cat_feature_names:
            if col in X_train_lgb.columns:
                X_train_lgb[col] = X_train_lgb[col].astype('category')

        fixed_params = dict(
            objective='binary',
            metric='binary_logloss',
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbose=-1,
        )

        if not self.cfg.do_grid_search:
            self.lgbm_model = lgb.LGBMClassifier(**fixed_params, n_estimators=500, max_depth=6, learning_rate=0.05)
            self.lgbm_model.fit(X_train_lgb, y_train)
            return

        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [300, 600],
            "min_child_samples": [5, 20],
        }

        base = lgb.LGBMClassifier(**fixed_params)
        tscv = TimeSeriesSplit(n_splits=self.cfg.tscv_splits)

        print(f"\nLightGBM GridSearchCV ({self.cfg.grid_scoring})...")
        grid = GridSearchCV(base, param_grid, scoring=self.cfg.grid_scoring, cv=tscv, n_jobs=-1, verbose=1)
        grid.fit(X_train_lgb, y_train)
        print(f"Best params: {grid.best_params_}")
        self.lgbm_model = grid.best_estimator_

    def train_xgboost_regressor(self, X_train: pd.DataFrame, y_train_cont: pd.Series):
        try:
            import xgboost as xgb
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        except ImportError:
            print("XGBoost not installed.")
            return

        from sklearn.metrics import make_scorer
        from scipy.stats import spearmanr as _spearmanr
        spearman_scorer = make_scorer(lambda y, y_pred: _spearmanr(y, y_pred)[0], greater_is_better=True)

        fixed_params = dict(
            objective="reg:absoluteerror", random_state=42, n_jobs=-1,
            tree_method="hist", enable_categorical=True,
        )

        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [300, 600],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 5],
        }

        # Winsorize target — TESTED, made Spearman ρ worse (0.030 → 0.015), reverted
        # _std = y_train_cont.std()
        # y_train_cont = y_train_cont.clip(-2 * _std, 2 * _std)

        if not self.cfg.do_grid_search:
            self.xgb_regressor = xgb.XGBRegressor(**fixed_params, n_estimators=500, max_depth=6, learning_rate=0.05)
            self.xgb_regressor.fit(X_train, y_train_cont)
            return

        tscv = TimeSeriesSplit(n_splits=self.cfg.tscv_splits)
        base = xgb.XGBRegressor(**fixed_params)
        print(f"\nXGBoost Regressor [absoluteerror] GridSearchCV (spearman_r)...")
        grid = GridSearchCV(base, param_grid, scoring=spearman_scorer, cv=tscv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train_cont)
        print(f"Best params: {grid.best_params_}")
        self.xgb_regressor = grid.best_estimator_

    def train_lightgbm_ranker(self, X_train: pd.DataFrame, y_train_cont: pd.Series, date_train: pd.Series):
        """Train LightGBM lambdarank model. Groups defined as filing quarter."""
        try:
            import lightgbm as lgb
        except ImportError:
            print("LightGBM not installed — skipping ranker.")
            self.lgbm_ranker = None
            return

        # Sort data by quarter — LightGBM lambdarank requires data sorted by group
        quarters = pd.to_datetime(date_train).dt.to_period("Q")
        sort_idx = np.argsort(quarters.values.astype(str), kind="stable")
        X_sorted = X_train.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y_train_cont.iloc[sort_idx].reset_index(drop=True)
        q_sorted = quarters.iloc[sort_idx]

        # Group sizes: number of trades per quarter in order
        group_sizes = q_sorted.value_counts(sort=False).sort_index().values

        # Relevance labels: global quintiles on train set mapped to 0-4
        quintile_breaks = np.nanpercentile(y_sorted.values, [20, 40, 60, 80])
        self._ranker_quintile_breaks = quintile_breaks
        relevance = np.zeros(len(y_sorted), dtype=int)
        for i, brk in enumerate(quintile_breaks):
            relevance[y_sorted.values > brk] = i + 1

        print(f"\nLightGBM Ranker (lambdarank) — {len(group_sizes)} quarters, "
              f"train_size={len(X_sorted)}, relevance={np.bincount(relevance).tolist()}")

        self.lgbm_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self.lgbm_ranker.fit(X_sorted, relevance, group=group_sizes)
        print("LightGBM Ranker trained.")

    def evaluate_lgbm_ranker(self, X_test: pd.DataFrame, y_test_cont: pd.Series) -> np.ndarray:
        """Evaluate lambdarank model. Returns ranker scores for downstream Section C analysis."""
        from scipy.stats import spearmanr

        ranker_scores = self.lgbm_ranker.predict(X_test)
        y_true = y_test_cont.to_numpy()
        rho, p_val = spearmanr(y_true, ranker_scores)

        print(f"\n{'='*80}")
        print(f"LightGBM Ranker [lambdarank] EVALUATION")
        print(f"{'='*80}")
        print(f"  Spearman rho:  {rho:.4f}  (p={p_val:.4f})")
        print(f"  Test size:     {len(y_true)}")
        return ranker_scores

    def evaluate_combined_ranker(
        self,
        y_proba_clf: np.ndarray,
        ranker_scores: np.ndarray,
        y_test_cont: pd.Series,
        clf_thresholds: list = None,
        top_pcts: list = None,
    ):
        """Section C: rank classifier positives by lambdarank score (top-K% analysis)."""
        if clf_thresholds is None:
            clf_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87]
        if top_pcts is None:
            top_pcts = [1.0, 0.75, 0.5, 0.25, 0.1]

        y_true = y_test_cont.to_numpy()

        print(f"\n{'='*80}")
        print(f"Section C — LightGBM lambdarank: rank clf positives by ranker score")
        print(f"  For each clf_thr, take top-K% sorted by ranker score desc")
        print(f"{'='*80}")

        for clf_thr in clf_thresholds:
            clf_mask = y_proba_clf >= clf_thr
            n_clf = clf_mask.sum()
            if n_clf == 0:
                continue

            clf_indices = np.where(clf_mask)[0]
            rank_scores_clf = ranker_scores[clf_indices]
            true_cars_clf = y_true[clf_indices]

            sorted_order = np.argsort(-rank_scores_clf)
            sorted_true = true_cars_clf[sorted_order]

            print(f"\n  clf_thr={clf_thr:.2f} | N_clf={n_clf}")
            print(f"  {'top_pct':<10}{'n_sel':>8}{'mean_actual':>14}{'sum_actual':>14}")
            print(f"  {'-'*48}")
            for pct in top_pcts:
                k = max(1, int(np.ceil(n_clf * pct)))
                top_true = sorted_true[:k]
                mean_act = top_true.mean()
                sum_act = top_true.sum()
                print(f"  {pct*100:>6.0f}%   {k:>8}{mean_act:>14.4f}{sum_act:>14.2f}")

    def evaluate_ranker_standalone(
        self,
        ranker_scores: np.ndarray,
        y_test_cont: pd.Series,
        top_pcts: list = None,
    ):
        """Rank ALL test trades by lambdarank score (no classifier gate). Top-K% analysis."""
        if top_pcts is None:
            top_pcts = [1.0, 0.50, 0.25, 0.15, 0.10, 0.05]

        y_true = y_test_cont.to_numpy()
        sorted_order = np.argsort(-ranker_scores)
        sorted_true = y_true[sorted_order]

        print(f"\n{'='*80}")
        print(f"Section D — LightGBM Ranker STANDALONE (no classifier gate)")
        print(f"  Rank all {len(y_true)} test trades by ranker score, take top-K%")
        print(f"{'='*80}")
        print(f"  {'top_pct':<10}{'n_sel':>8}{'mean_actual':>14}{'sum_actual':>14}")
        print(f"  {'-'*48}")
        for pct in top_pcts:
            k = max(1, int(np.ceil(len(y_true) * pct)))
            top_true = sorted_true[:k]
            print(f"  {pct*100:>6.1f}%   {k:>8}{top_true.mean():>14.4f}{top_true.sum():>14.2f}")

    def evaluate_regressor(self, X_test: pd.DataFrame, y_test_cont: pd.Series):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from scipy.stats import spearmanr

        y_pred = self.xgb_regressor.predict(X_test)
        y_true = y_test_cont.to_numpy()

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

        print(f"\n{'='*80}")
        print(f"XGBoost Regressor [absoluteerror] EVALUATION — Realized CAR ({self.horizon_months}m)")
        print(f"{'='*80}")
        print(f"  R²:              {r2:.4f}")
        print(f"  MAE:             {mae:.4f}")
        print(f"  RMSE:            {rmse:.4f}")
        print(f"  Spearman ρ:      {spearman_r:.4f}  (p={spearman_p:.4f})")
        print(f"  Baseline mean:   {np.mean(y_true):.4f}")
        print(f"  Baseline median: {np.median(y_true):.4f}")

        print(f"\n{'pred_thr':<10}{'n_selected':>12}{'mean_actual':>14}{'sum_actual':>14}{'pct_sel':>10}")
        print("-" * 60)
        for thr in self.cfg.reg_car_thresholds:
            sel_mask = y_pred >= thr
            n_sel = sel_mask.sum()
            if n_sel > 0:
                mean_act = y_true[sel_mask].mean()
                sum_act = y_true[sel_mask].sum()
            else:
                mean_act = sum_act = 0.0
            pct_sel = n_sel / len(y_true) * 100
            print(f"{thr:<10.2f}{n_sel:>12}{mean_act:>14.4f}{sum_act:>14.2f}{pct_sel:>9.1f}%")

        self.print_feature_importance(self.xgb_regressor, "XGBoost Regressor [absoluteerror]",
                                      feature_names=X_test.columns.tolist())

    def _metrics(self, y_true, y_pred, y_proba=None):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        out = {
            "acc": accuracy_score(y_true, y_pred),
            "prec": precision_score(y_true, y_pred, zero_division=0),
            "rec": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if y_proba is not None:
            out["auc"] = roc_auc_score(y_true, y_proba)
        return out

    def evaluate_threshold_grid(self, model, X_test, y_test, y_test_cont, model_name: str):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_true = y_test.astype(int).to_numpy()
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*80}")
        print(f"{model_name} EVALUATION — Realized CAR ({self.horizon_months}m)")
        print(f"{'='*80}")

        def print_prediction_summary(y_proba, y_true, y_test_cont, threshold=0.6):
            y_pred = (y_proba >= threshold).astype(int)
            stats_cols = ['min', 'max', 'mean', '50%', 'std']

            def get_stats(series):
                if len(series) == 0:
                    return pd.Series(['N/A']*7, index=['sum', 'min', 'max', 'mean', '50%', 'std', 'count'])
                s = series.describe()[stats_cols]
                s['sum'] = series.sum()
                s['count'] = len(series)
                return s.round(4)

            all_alphas = get_stats(y_test_cont)
            pp_alphas = get_stats(y_test_cont[y_pred == 1])
            fp_alphas = get_stats(y_test_cont[(y_pred == 1) & (y_true == 0)])
            tp_alphas = get_stats(y_test_cont[(y_pred == 1) & (y_true == 1)])

            summary_df = pd.DataFrame({
                'All Test Alphas': all_alphas,
                'Predicted Positives': pp_alphas,
                'False Positives': fp_alphas,
                'True Positives': tp_alphas
            })

            ordered_cols = ['sum', 'min', 'max', 'mean', '50%', 'std', 'count']
            summary_df = summary_df.reindex(ordered_cols)

            print(f"\nPrediction Summary @ threshold {threshold}")
            print("=" * 60)
            print(summary_df.to_string())

        def find_optimal_thresholds(y_proba, y_true, y_test_cont, thresholds=np.linspace(0.1, 0.9, 81), min_n=100):
            results = []

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                tp_mask = (y_pred == 1)
                tp_alphas = y_test_cont[tp_mask]

                if len(tp_alphas) > 0:
                    results.append({
                        'threshold': round(thresh, 3),
                        'n_tp': len(tp_alphas),
                        'sum_tp': tp_alphas.sum(),
                        'mean_tp': tp_alphas.mean(),
                        'median_tp': tp_alphas.median()
                    })
                else:
                    results.append({
                        'threshold': round(thresh, 3),
                        'n_tp': 0,
                        'sum_tp': np.nan,
                        'mean_tp': np.nan,
                        'median_tp': np.nan
                    })

            df_results = pd.DataFrame(results)

            opt_sum = df_results.loc[df_results['sum_tp'].idxmax()]

            # Mean: enforce minimum N to avoid noise from tiny samples
            mean_candidates = df_results[df_results['n_tp'] >= min_n]
            if len(mean_candidates) > 0:
                opt_mean = mean_candidates.loc[mean_candidates['mean_tp'].idxmax()]
            else:
                # Fallback: relax to N >= 20
                mean_candidates = df_results[df_results['n_tp'] >= 20]
                if len(mean_candidates) > 0:
                    opt_mean = mean_candidates.loc[mean_candidates['mean_tp'].idxmax()]
                else:
                    opt_mean = df_results.loc[df_results['mean_tp'].idxmax()]

            print(f"Optimal Thresholds (Predicted Positive Alphas, min_n={min_n})")
            print("=" * 60)
            print(f"Max Sum:     thr={opt_sum['threshold']}, sum={opt_sum['sum_tp']:.4f} (N={int(opt_sum['n_tp'])})")
            print(f"Max Mean:    thr={opt_mean['threshold']}, mean={opt_mean['mean_tp']:.4f} (N={int(opt_mean['n_tp'])})")

            return opt_sum, opt_mean, df_results

        opt_sum, opt_mean, full_scan = find_optimal_thresholds(y_proba, y_true, y_test_cont)

        print_prediction_summary(y_proba, y_true, y_test_cont, threshold=opt_sum['threshold'])
        print_prediction_summary(y_proba, y_true, y_test_cont, threshold=opt_mean['threshold'])

        print(f"\n{'thr':<6}{'acc':>8}{'prec':>10}{'rec':>10}{'f1':>10}{'n_pred':>10}{'mean_alpha':>12}{'sum_alpha':>12}")
        print("-" * 88)

        safe_name = model_name.lower().replace(' ', '_')
        for thr in self.cfg.prob_thresholds:
            y_pred = (y_proba >= thr).astype(int)
            m = self._metrics(y_true, y_pred)

            # Calculate alpha stats for predicted positives
            pred_pos_mask = (y_pred == 1)
            if pred_pos_mask.sum() > 0:
                mean_alpha = y_test_cont[pred_pos_mask].mean()
                sum_alpha = y_test_cont[pred_pos_mask].sum()
            else:
                mean_alpha = 0.0
                sum_alpha = 0.0

            print(f"{thr:<6.2f}{m['acc']:>8.4f}{m['prec']:>10.4f}{m['rec']:>10.4f}{m['f1']:>10.4f}{(y_pred.sum()):>10}{mean_alpha:>12.4f}{sum_alpha:>12.2f}")

            if self.cfg.save_plots:
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                           xticklabels=["0", "1"], yticklabels=["0", "1"])
                plt.title(f"{model_name} CM — Realized CAR (12m) (thr={thr:.2f})")
                plt.tight_layout()
                plt.savefig(f"data/output/{safe_name}_cm_realized_thr_{thr:.2f}.png", dpi=100)
                plt.close()

    def print_feature_importance(self, model, model_name: str = "XGBoost", top_n: int = 20, feature_names: list = None):
        names = feature_names if feature_names is not None else self.feature_names
        fi = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
        print("\n" + "-" * 60)
        print(f"{model_name} Top Features — Realized CAR (12m)")
        print("-" * 60)
        print(fi.head(top_n).to_string())

        if self.cfg.save_plots:
            import matplotlib.pyplot as plt
            safe_name = model_name.lower().replace(' ', '_')
            plt.figure(figsize=(10, 8))
            fi.head(top_n).plot(kind='barh')
            plt.title(f"{model_name} Feature Importance — Realized CAR (12m)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"data/output/{safe_name}_feature_imp_realized.png", dpi=100)
            plt.close()

    def analyze_predictions(self, test_df: pd.DataFrame, y_proba: np.ndarray, threshold: float = 0.86):
        """Analyze characteristics of high-confidence predictions."""
        y_pred = (y_proba >= threshold).astype(int)
        pred_mask = (y_pred == 1)

        if pred_mask.sum() == 0:
            print(f"No predictions at threshold {threshold}")
            return

        high_conf_trades = test_df[pred_mask].copy()
        high_conf_trades['predicted_proba'] = y_proba[pred_mask]

        print(f"\n{'='*80}")
        print(f"HIGH-CONFIDENCE TRADES ANALYSIS (threshold={threshold}, N={len(high_conf_trades)})")
        print(f"{'='*80}")

        # Politician breakdown
        print(f"\n{'─'*80}")
        print("TOP POLITICIANS (by number of high-confidence trades)")
        print(f"{'─'*80}")
        pol_counts = high_conf_trades['Name'].value_counts().head(15)
        for i, (name, count) in enumerate(pol_counts.items(), 1):
            pct = count / len(high_conf_trades) * 100
            print(f"{i:2d}. {name:<40s} {count:3d} trades ({pct:5.1f}%)")

        # Party breakdown
        print(f"\n{'─'*80}")
        print("PARTY BREAKDOWN")
        print(f"{'─'*80}")
        if 'Party' in high_conf_trades.columns:
            party_counts = high_conf_trades['Party'].value_counts()
            for party, count in party_counts.items():
                pct = count / len(high_conf_trades) * 100
                print(f"{party:<20s} {count:3d} trades ({pct:5.1f}%)")

        # Chamber breakdown
        print(f"\n{'─'*80}")
        print("CHAMBER BREAKDOWN")
        print(f"{'─'*80}")
        if 'Chamber' in high_conf_trades.columns:
            chamber_counts = high_conf_trades['Chamber'].value_counts()
            for chamber, count in chamber_counts.items():
                pct = count / len(high_conf_trades) * 100
                print(f"{chamber:<20s} {count:3d} trades ({pct:5.1f}%)")

        # Sector breakdown
        print(f"\n{'─'*80}")
        print("TOP SECTORS")
        print(f"{'─'*80}")
        if 'Ticker_Sector' in high_conf_trades.columns:
            sector_counts = high_conf_trades['Ticker_Sector'].value_counts().head(10)
            for sector, count in sector_counts.items():
                pct = count / len(high_conf_trades) * 100
                print(f"{str(sector):<40s} {count:3d} trades ({pct:5.1f}%)")

        # Industry breakdown
        print(f"\n{'─'*80}")
        print("TOP INDUSTRIES")
        print(f"{'─'*80}")
        if 'Ticker_Industry' in high_conf_trades.columns:
            industry_counts = high_conf_trades['Ticker_Industry'].value_counts().head(10)
            for industry, count in industry_counts.items():
                pct = count / len(high_conf_trades) * 100
                print(f"{str(industry):<50s} {count:3d} ({pct:4.1f}%)")

        # Committee characteristics
        print(f"\n{'─'*80}")
        print("COMMITTEE CHARACTERISTICS")
        print(f"{'─'*80}")
        if 'is_committee_chair' in high_conf_trades.columns:
            chair_pct = (high_conf_trades['is_committee_chair'] == 1).mean() * 100
            print(f"Committee Chair: {chair_pct:.1f}%")
        if 'is_committee_majority' in high_conf_trades.columns:
            majority_pct = (high_conf_trades['is_committee_majority'] == 1).mean() * 100
            print(f"Committee Majority: {majority_pct:.1f}%")

        # Position closure stats
        print(f"\n{'─'*80}")
        print("POSITION CLOSURE")
        print(f"{'─'*80}")
        if 'position_closed' in high_conf_trades.columns:
            closed_pct = (high_conf_trades['position_closed'] == True).mean() * 100
            print(f"Positions closed within 12m: {closed_pct:.1f}%")

            if 'holding_period_days' in high_conf_trades.columns:
                closed_trades = high_conf_trades[high_conf_trades['position_closed'] == True]
                if len(closed_trades) > 0:
                    holding_stats = closed_trades['holding_period_days'].describe()
                    print(f"\nHolding period (days) for closed positions:")
                    print(f"  Mean: {holding_stats['mean']:.1f}")
                    print(f"  Median: {holding_stats['50%']:.1f}")
                    print(f"  Min: {holding_stats['min']:.1f}")
                    print(f"  Max: {holding_stats['max']:.1f}")

        # Key numerical features
        print(f"\n{'─'*80}")
        print("KEY FEATURE STATISTICS")
        print(f"{'─'*80}")

        numeric_features = ['politician_hit_rate_past', 'politician_mean_car_past',
                           'politician_trades_last_year', 'stock_momentum_30d',
                           'stock_volatility_30d', 'beta']

        for feat in numeric_features:
            if feat in high_conf_trades.columns:
                feat_series = pd.to_numeric(high_conf_trades[feat], errors='coerce')
                if feat_series.notna().sum() > 0:
                    print(f"{feat:<35s} mean={feat_series.mean():>8.4f}  median={feat_series.median():>8.4f}")

        print(f"\n{'='*80}\n")

    def evaluate_combined(
        self,
        y_proba_clf: np.ndarray,
        y_pred_reg: np.ndarray,
        y_test_cont: pd.Series,
        clf_thresholds: list = None,
        reg_thresholds: list = None,
        top_pcts: list = None,
    ):
        if clf_thresholds is None:
            clf_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.85]
        if reg_thresholds is None:
            reg_thresholds = self.cfg.reg_car_thresholds
        if top_pcts is None:
            top_pcts = [1.0, 0.75, 0.5, 0.25, 0.1]

        y_true = y_test_cont.to_numpy()

        print(f"\n{'#'*80}")
        print("  COMBINED CLASSIFIER + REGRESSOR EVALUATION")
        print(f"{'#'*80}")

        # --- Section A: AND gate grid ---
        print(f"\n{'='*80}")
        print(f"Section A — AND Gate: Classifier AND Regressor [absoluteerror]")
        print(f"  Rows = classifier prob threshold | Columns = regressor CAR threshold")
        print(f"  Cell format: N | mean_CAR")
        print(f"{'='*80}")

        reg_thr_cols = [t for t in reg_thresholds if t <= 0.10]
        header = f"{'clf_thr':<10}" + "".join(f"  reg>{t:.2f}(N/mean)" for t in reg_thr_cols)
        print(header)
        print("-" * len(header))

        for clf_thr in clf_thresholds:
            clf_mask = y_proba_clf >= clf_thr
            n_clf = clf_mask.sum()
            mean_clf = y_true[clf_mask].mean() if n_clf > 0 else 0.0
            row = f"{clf_thr:<10.2f}"
            for reg_thr in reg_thr_cols:
                combined_mask = clf_mask & (y_pred_reg >= reg_thr)
                n = combined_mask.sum()
                mean_act = y_true[combined_mask].mean() if n > 0 else float("nan")
                row += f"  {n:4d}/{mean_act:+.3f}    "
            row += f"  [clf-only: N={n_clf}, mean={mean_clf:+.4f}]"
            print(row)

        # --- Section B: Secondary ranker ---
        print(f"\n{'='*80}")
        print(f"Section B — Secondary Ranker: rank classifier positives by Regressor [absoluteerror]")
        print(f"  For each clf_thr, take top-K% of classifier positives sorted by predicted CAR desc")
        print(f"{'='*80}")

        for clf_thr in clf_thresholds:
            clf_mask = y_proba_clf >= clf_thr
            n_clf = clf_mask.sum()
            if n_clf == 0:
                continue

            clf_indices = np.where(clf_mask)[0]
            reg_preds_clf = y_pred_reg[clf_indices]
            true_cars_clf = y_true[clf_indices]

            sorted_order = np.argsort(-reg_preds_clf)
            sorted_true = true_cars_clf[sorted_order]

            print(f"\n  clf_thr={clf_thr:.2f} | N_clf={n_clf}")
            print(f"  {'top_pct':<10}{'n_sel':>8}{'mean_actual':>14}{'sum_actual':>14}")
            print(f"  {'-'*48}")
            for pct in top_pcts:
                k = max(1, int(np.ceil(n_clf * pct)))
                top_true = sorted_true[:k]
                mean_act = top_true.mean()
                sum_act = top_true.sum()
                print(f"  {pct*100:>6.0f}%   {k:>8}{mean_act:>14.4f}{sum_act:>14.2f}")

    def run(self, base_purchases: pd.DataFrame):
        print(f"\n{'#'*80}")
        print(f"  REALIZED CAR MODEL ({self.horizon_months}m) | Cutoff: {self.cutoff_date}")
        print(f"{'#'*80}")

        df = self.preprocess(base_purchases)

        if len(df) < 100:
            print(f"Skipping — only {len(df)} rows after cutoff/filtering.")
            return

        train_df, test_df = self.time_split(df)
        y_train = train_df[self.target_binary]
        y_test = test_df[self.target_binary]
        y_test_cont = test_df[self.target_continuous]

        # --- XGBoost (LabelEncoded features) ---
        X_train_xgb = self.prepare_features(train_df, is_training=True)
        X_test_xgb = self.prepare_features(test_df, is_training=False)

        self.train_xgboost(X_train_xgb, y_train)
        if self.xgb_model:
            self.evaluate_threshold_grid(self.xgb_model, X_test_xgb, y_test, y_test_cont, "XGBoost")
            self.print_feature_importance(self.xgb_model, "XGBoost")
            try:
                self.xgb_model.save_model(f"data/output/xgboost_model_realized.json")
                print(f"\nXGBoost model saved.")
            except Exception as e:
                print(f"Save failed: {e}")

        # --- Two-layer architecture (DEACTIVATED) ---
        # Regressor + LightGBM ranker are kept for reference but not called in production.
        # Rationale: model is used to score individual trades at disclosure time (single-stage clf only).
        # To re-enable: call _run_second_layer(train_df, test_df, X_test_xgb, y_test_cont) below.
        pass

    def _run_second_layer(self, train_df, test_df, X_test_xgb, y_test_cont):
        """Deactivated second-layer: XGBoost regressor + LightGBM ranker + combined evaluation.

        Not called from run(). Kept for reference / future experiments.
        See CLAUDE.md § Two-layer architecture for performance notes.
        """
        # --- XGBoost Regressor (absoluteerror, spearman-optimized, extra features) ---
        X_train_reg = self.prepare_features_regressor(train_df, is_training=True)
        X_test_reg = self.prepare_features_regressor(test_df, is_training=False)

        y_train_cont = train_df[self.target_continuous]
        self.train_xgboost_regressor(X_train_reg, y_train_cont)
        if self.xgb_regressor:
            self.evaluate_regressor(X_test_reg, y_test_cont)

        # --- Combined Classifier + Regressor Evaluation ---
        y_proba_clf = self.xgb_model.predict_proba(X_test_xgb)[:, 1] if self.xgb_model else None

        if self.xgb_model and self.xgb_regressor:
            y_pred_reg = self.xgb_regressor.predict(X_test_reg)
            self.evaluate_combined(y_proba_clf, y_pred_reg, y_test_cont)

        # --- LightGBM Ranker (lambdarank) ---
        self.train_lightgbm_ranker(X_train_reg, y_train_cont, train_df[self.date_column])
        if self.lgbm_ranker:
            ranker_scores = self.evaluate_lgbm_ranker(X_test_reg, y_test_cont)
            if y_proba_clf is not None:
                self.evaluate_combined_ranker(y_proba_clf, ranker_scores, y_test_cont)



def main():
    cfg = Config()

    # --- Load data once ---
    print(f"Loading {cfg.data_path}...")
    df = pd.read_parquet(cfg.data_path)

    # Compute sells pressure once (expensive, gap-independent)
    df = PoliticianTradeModel._calculate_sells_pressure(df, "Filed")

    # Compute all-politician sells same ticker (must run before purchase filter)
    df = PoliticianTradeModel._calculate_all_pol_sells_same_ticker(df, "Filed", window_days=30)

    # Filter to purchases
    df = df[df["Transaction"] == "Purchase"].copy()
    print(f"Total purchases: {len(df)}")

    # 12-month horizon
    df["Traded"] = pd.to_datetime(df["Traded"], errors="coerce")
    max_traded = df["Traded"].max()
    print(f"Max Traded date: {max_traded.date()}")

    cutoff = (max_traded - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    print(f"Cutoff date (12m lookback): {cutoff}")

    model = PoliticianTradeModel(cfg, cutoff_date=cutoff, horizon_months=12)
    model.run(base_purchases=df)

    print(f"\n{'='*80}")
    print("Model training complete.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
