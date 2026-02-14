"""src/model/model.py

Train XGBoost classifier on politician trades (Purchases only)
with time-based split and multi-threshold evaluation.

Key features:
- Politician hit rate (past trades only) 
- Politician sells same ticker 50D (calculated from FULL dataset before purchase filter!)
- All other engineered features

Error analysis with continuous alpha stats for FPs/TPs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

@dataclass
class Config:
    data_path: str = "politician_trades_enriched.parquet"
    cutoff_date: str = "2025-01-05"
    prediction_gap: str = '12'
    train_ratio: float = 0.8
    alpha_threshold: float = 0.0
    prob_thresholds: List[float] = (0.5, 0.6, 0.7, 0.8)
    do_xgb_grid_search: bool = True
    tscv_splits: int = 3
    grid_scoring: str = "precision"
    save_plots: bool = True
    

class PoliticianTradeModel:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.xgb_model = None
        self.label_encoders: Dict[str, object] = {}
        self.feature_names: Optional[List[str]] = None
        self.numeric_fill_values: Dict[str, float] = {}

        self.numerical_features = [
            "Trade_Size_USD", "alpha_traded_to_filed", "delta_traded_filed",
            "politician_trades_last_year", "politician_hit_rate_past"  # Now real values!
        ]

        self.categorical_features = [
            "BioGuideID", "Party", "Chamber", "Ticker_Sector", 
            "Ticker_Industry", "ticker_filed_density_50",
            "politician_sells_same_ticker_50d"
        ]

        self.binary_features = [
            #"Committee_SSBK", "Committee_HSAS", "Committee_SSCM", "Committee_HSFA", "Committee_HSAP",
            #"Committee_HSED", "Committee_HSWM", "Committee_SLIN", "Committee_HSSY", "Committee_HSZS",
            #"Committee_SSAS", "Committee_HSAG", "Committee_SSGA", "Committee_HLIG", "Committee_SLIA",
            #"Committee_SSBU", "Committee_SSFI", "Committee_SSFR", "Committee_HSIF", "Committee_SSEG",
            #"Committee_HSHM", "Committee_JSPR", "Committee_SPAG", "Committee_HSVR", "Committee_SSAP",
            #"Committee_SSAF", "Committee_SSVA", "Committee_HSSO", "Committee_HSII", "Committee_SSRA",
            #"Committee_HSJU", "Committee_SSEV", "Committee_JSEC", "Committee_SCNC", "Committee_HSPW",
            #"Committee_HSGO", "Committee_HSBA", "Committee_HSHA", "Committee_HSBU", "Committee_SSHR",
            #"Committee_HSIJ", "Committee_HSCN", "Committee_SSJU", "Committee_HSSM", "Committee_HSFD",
            #"Committee_SSSB", "Committee_JCSE", "Committee_JSLC", "Committee_SLET", "Committee_HSMH",
            #"Committee_HSRU", "Committee_JSTX", "Committee_HSVC", "Committee_HLZI", "Committee_HSEF",
            #"Committee_JSDF", "Committee_HSZT", "Committee_HSQJ",
            "Industry match 1", "Industry match 2", "Industry match 3",
            "Sector match 1", "Sector match 2", "Sector match 3"
        ]

        self.date_column = "Filed"
        self.target_continuous = "alpha_filed_to_"+self.cfg.prediction_gap+"m"
        self.target_binary = "alpha_above_5"

    @staticmethod
    def _parse_decimal_comma(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        s = s.replace({"None": np.nan, "nan": np.nan, "NaN": np.nan, "<NA>": np.nan})
        
        # FIXED: Chain individual replaces
        s = (s.str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace(" ", "", regex=False))
        
        mask_both = s.str.contains(r"\.") & s.str.contains(",")
        s.loc[mask_both] = s.loc[mask_both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        mask_comma_only = s.str.contains(",") & (~s.str.contains(r"\."))
        s.loc[mask_comma_only] = s.loc[mask_comma_only].str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")


    def _calculate_sells_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixed version - handles missing BioGuideIDs safely"""
        print("Calculating politician sell pressure (using full dataset)...")
        
        df[self.date_column] = pd.to_datetime(df[self.date_column], format="%Y-%m-%d", errors="coerce")
        df["Traded"] = pd.to_datetime(df["Traded"], format="%Y-%m-%d", errors="coerce")
        
        df = df.sort_values(self.date_column).reset_index(drop=True)
        df["_temp_id"] = range(len(df))
        
        # Get sells
        sells = df[df["Transaction"].str.contains("Sale", na=False)].copy()
        if len(sells) == 0:
            df["politician_sells_same_ticker_50d"] = 0.0
            print("No sell transactions found.")
            return df.drop(columns=["_temp_id"])
        
        # FIXED: Use loc with try/except for safe slicing
        sells_idx = sells.set_index([self.date_column, "BioGuideID"]).sort_index()
        
        def count_prior_sells(row):
            try:
                bio = row["BioGuideID"]
                filed_date = row[self.date_column]
                prior_start = filed_date - pd.Timedelta(days=15)
                
                # Safe MultiIndex slicing
                prior_slice = sells_idx.loc[(slice(prior_start, filed_date), bio), :]
                return len(prior_slice)
            except (KeyError, ValueError):
                return 0  # BioGuideID not in sells or date range empty
        
        # Calculate only for purchases
        purchase_mask = df["Transaction"] == "Purchase"
        df["politician_sells_same_ticker_50d"] = 0.0
        
        if purchase_mask.sum() > 0:
            sells_counts = df.loc[purchase_mask].apply(count_prior_sells, axis=1)
            df.loc[purchase_mask, "politician_sells_same_ticker_50d"] = sells_counts.values
        
        print(f"Sell pressure stats: {df['politician_sells_same_ticker_50d'].describe()}")
        return df.drop(columns=["_temp_id"])

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-purchase-filter features"""
        print("Adding purchase-only engineered features...")
        
        df["Traded"] = pd.to_datetime(df["Traded"], format="%Y-%m-%d", errors="coerce")
        df[self.date_column] = pd.to_datetime(df[self.date_column], format="%Y-%m-%d", errors="coerce")
        df["delta_traded_filed"] = (df[self.date_column] - df["Traded"]).dt.days
        
        df = df.sort_values(self.date_column).reset_index(drop=True)
        df["_temp_id"] = range(len(df))
        
        def get_rolling_count(grp, window):
            g = grp.set_index(self.date_column).sort_index()
            rc = g["_temp_id"].rolling(window, closed="left").count()
            return pd.Series(rc.values, index=g["_temp_id"].values)
        
        # Ticker density
        t_res = df.groupby("Ticker", group_keys=False).apply(lambda x: get_rolling_count(x, "10D"))
        df["ticker_filed_density_50"] = df["_temp_id"].map(t_res).fillna(0)
        
        # Politician activity
        p_res = df.groupby("BioGuideID", group_keys=False).apply(lambda x: get_rolling_count(x, "365D"))
        df["politician_trades_last_year"] = df["_temp_id"].map(p_res).fillna(0)
        
        # Politician hit rate (previous trades only)
        print("Calculating politician hit rates...")
        def get_hit_rate(grp):
            if len(grp) < 2:  # Need at least 2 trades
                return pd.Series(0.0, index=grp["_temp_id"])
            
            g = grp.sort_values(self.date_column).reset_index(drop=True)
            # Use the TEMPORARY column name we created
            g['temp_target'] = (g[self.target_continuous] > self.cfg.alpha_threshold).astype(int)
            
            cum_hits = g['temp_target'].shift(1).fillna(0).cumsum()
            cum_total = pd.Series(np.arange(len(g)))
            
            prev_total = cum_total - 1
            hit_rate = cum_hits / prev_total.replace(0, np.nan)
            return pd.Series(hit_rate.fillna(0.0).values, index=g["_temp_id"].values)
        
        p_hit = df.groupby("BioGuideID", group_keys=False).apply(get_hit_rate)
        df["politician_hit_rate_past"] = df["_temp_id"].map(p_hit).fillna(0)
        
        return df.drop(columns=["_temp_id"])


    def load_and_preprocess_data(self) -> pd.DataFrame:
        print(f"Loading {self.cfg.data_path}...")
        df = pd.read_parquet(self.cfg.data_path)
        
        # 1. Calculate sells pressure FIRST (full dataset)
        df = self._calculate_sells_pressure(df)
        
        # 2. Filter purchases
        df = df[df["Transaction"] == "Purchase"].copy()
        
        # 3. Dates and cutoff
        df[self.date_column] = pd.to_datetime(df[self.date_column], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=[self.date_column])
        
        cutoff = pd.Timestamp(self.cfg.cutoff_date)
        df = df[df[self.date_column] <= cutoff].copy()
        
        # 4. Parse continuous target
        df[self.target_continuous] = self._parse_decimal_comma(df[self.target_continuous])
        df = df.dropna(subset=[self.target_continuous]).copy()
        
        # 5. Add ALL engineered features (including hit rate which needs target)
        df = self._add_engineered_features(df)
        
        # 6. Create binary target LAST (after all features use continuous target)
        df[self.target_binary] = (df[self.target_continuous] > self.cfg.alpha_threshold).astype(int)
        
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        print(f"✅ Final dataset: {len(df)} purchases")
        print(f"   Alpha stats: {df[self.target_continuous].describe().round(3)}")
        print(f"   Positive rate: {(df[self.target_binary]==1).mean():.1%}")
        
        return df

    def prepare_features(self, df: pd.DataFrame, *, is_training: bool) -> pd.DataFrame:
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
                X[col] = -1
                continue
            s = df[col].fillna("Unknown").astype(str)
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(s)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                X[col] = s.map(lambda v: int(le.transform([v])[0]) if v in le.classes_ else -1)

        for col in self.binary_features:
            if col not in df.columns:
                X[col] = 0
                continue
            X[col] = df[col].fillna(0).astype(int)

        if is_training:
            self.feature_names = X.columns.tolist()
        return X

    def time_split(self, df: pd.DataFrame):
        split_idx = int(len(df) * self.cfg.train_ratio)
        split_dt = df[self.date_column].iloc[split_idx]
        train_df = df[df[self.date_column] < split_dt].copy()
        test_df = df[df[self.date_column] >= split_dt].copy()
        print(f"\nTime split at {split_dt} | Train: {len(train_df)} | Test: {len(test_df)}")
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
            n_jobs=-1, tree_method="hist", scale_pos_weight=scale_pos_weight, base_score=base_score
        )

        if not self.cfg.do_xgb_grid_search:
            self.xgb_model = xgb.XGBClassifier(**fixed_params, n_estimators=500, max_depth=6, learning_rate=0.05)
            self.xgb_model.fit(X_train, y_train)
            return

        param_grid = {
            "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [300, 600], "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9], "min_child_weight": [1, 5]
        }

        base = xgb.XGBClassifier(**fixed_params)
        tscv = TimeSeriesSplit(n_splits=self.cfg.tscv_splits)
        
        print("\nXGBoost GridSearchCV (precision)...")
        grid = GridSearchCV(base, param_grid, scoring=self.cfg.grid_scoring, cv=tscv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        self.xgb_model = grid.best_estimator_

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
        print(f"{model_name} EVALUATION")
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
            
            # Get stats
            all_alphas = get_stats(y_test_cont)
            pp_alphas = get_stats(y_test_cont[y_pred == 1])
            fp_alphas = get_stats(y_test_cont[(y_pred == 1) & (y_true == 0)])
            tp_alphas = get_stats(y_test_cont[(y_pred == 1) & (y_true == 1)])
            
            # Build DataFrame, then reorder columns safely
            summary_df = pd.DataFrame({
                'All Test Alphas': all_alphas,
                'Predicted Positives': pp_alphas,
                'False Positives': fp_alphas,
                'True Positives': tp_alphas
            })
            
            # Reorder to put sum first (safe even with N/A)
            ordered_cols = ['sum', 'min', 'max', 'mean', '50%', 'std', 'count']
            summary_df = summary_df.reindex(ordered_cols)
            
            print(f"\nPrediction Summary @ threshold {threshold}")
            print("=" * 60)
            print(summary_df.to_string())

        #print_prediction_summary(y_proba, y_true, y_test_cont, threshold=0.5)
        #print_prediction_summary(y_proba, y_true, y_test_cont, threshold=0.6)
        #print_prediction_summary(y_proba, y_true, y_test_cont, threshold=0.7)
        #print_prediction_summary(y_proba, y_true, y_test_cont, threshold=0.8)

        def find_optimal_thresholds(y_proba, y_true, y_test_cont, thresholds=np.linspace(0.1, 0.9, 81)):
            """
            Finds thresholds maximizing: sum(TP alphas), mean(TP alphas), median(TP alphas).
            Returns optimal results + full scan DataFrame.
            """
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
            
            # Optimal (maximizes metric, tiebreak by higher N_TP)
            opt_sum = df_results.loc[df_results['sum_tp'].idxmax()]
            opt_mean = df_results.loc[df_results['mean_tp'].idxmax()]
            opt_median = df_results.loc[df_results['median_tp'].idxmax()]
            
            print("Optimal Thresholds (True Positive Alphas)")
            print("=" * 50)
            print(f"Max Sum:     {opt_sum['threshold']}, sum={opt_sum['sum_tp']:.4f} (N={opt_sum['n_tp']})")
            print(f"Max Mean:    {opt_mean['threshold']}, mean={opt_mean['mean_tp']:.4f} (N={opt_mean['n_tp']})")
            print(f"Max Median:  {opt_median['threshold']}, median={opt_median['median_tp']:.4f} (N={opt_median['n_tp']})")
            
            return opt_sum, opt_mean, opt_median, df_results
        
        opt_sum, opt_mean, opt_median, full_scan = find_optimal_thresholds(y_proba, y_true, y_test_cont)

        print_prediction_summary(y_proba, y_true, y_test_cont, threshold=opt_sum['threshold'])
        print_prediction_summary(y_proba, y_true, y_test_cont, threshold=opt_mean['threshold'])
        print_prediction_summary(y_proba, y_true, y_test_cont, threshold=opt_median['threshold'])

        print(f"\n{'thr':<6}{'acc':>8}{'prec':>10}{'rec':>10}{'f1':>10}{'n_pred1':>10}")
        print("-" * 54)

        for thr in self.cfg.prob_thresholds:
            y_pred = (y_proba >= thr).astype(int)
            m = self._metrics(y_true, y_pred)
            print(f"{thr:<6.2f}{m['acc']:>8.4f}{m['prec']:>10.4f}{m['rec']:>10.4f}{m['f1']:>10.4f}{(y_pred.sum()):>10}")

            if self.cfg.save_plots:
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                           xticklabels=["0", "1"], yticklabels=["0", "1"])
                plt.title(f"{model_name} CM (thr={thr:.2f})")
                plt.tight_layout()
                plt.savefig(f"xgboost_cm_thr_{thr:.2f}.png", dpi=100)
                plt.close()

    def print_feature_importance(self, model, top_n: int = 20):
        fi = pd.Series(model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        print("\n" + "-" * 60)
        print("XGBoost Top Features")
        print("-" * 60)
        print(fi.head(top_n).to_string())
        
        if self.cfg.save_plots:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            fi.head(top_n).plot(kind='barh')
            plt.title("XGBoost Feature Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("xgboost_feature_imp.png", dpi=100)
            plt.close()

    def run(self):
        df = self.load_and_preprocess_data()
        train_df, test_df = self.time_split(df)

        X_train = self.prepare_features(train_df, is_training=True)
        y_train = train_df[self.target_binary]
        X_test = self.prepare_features(test_df, is_training=False)
        y_test = test_df[self.target_binary]
        y_test_cont = test_df[self.target_continuous]

        self.train_xgboost(X_train, y_train)
        if self.xgb_model:
            self.evaluate_threshold_grid(self.xgb_model, X_test, y_test, y_test_cont, "XGBoost")
            self.print_feature_importance(self.xgb_model)
            try:
                self.xgb_model.save_model("xgboost_model.json")
                print("\n✅ Model saved: xgboost_model.json")
            except Exception as e:
                print(f"Save failed: {e}")

def main():
    model = PoliticianTradeModel(Config())
    model.run()

if __name__ == "__main__":
    main()
