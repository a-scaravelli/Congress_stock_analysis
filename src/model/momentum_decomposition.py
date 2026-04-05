"""src/model/momentum_decomposition.py

Diagnostic 2 — Momentum Decomposition (Prof. Ben's framework).

Regresses realized CAR on the stock's prior momentum to separate:
  - Mean reversion: β < 0 on prior return → buying beaten-down stocks
                   explains positive CAR mechanically
  - Information:   α > 0 after controlling for momentum → real edge

Regressions:
  Model 1  (simple):   realized_car ~ momentum_90d
  Model 2  (extended): realized_car ~ momentum_90d + momentum_30d + beta
                                    + stock_volatility_30d + log_trade_size
  Model 3  (interacted): adds momentum_90d² to test non-linear reversion
  Model 4  (subgroups): split by momentum quintile — does alpha persist across
                        all quintiles or only in the beaten-down bucket?

Also reports binned mean CAR by prior-return quintile so the pattern is
immediately interpretable.

Run from project root:
    python src/model/momentum_decomposition.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.model.model_realized import PoliticianTradeModel, Config  # noqa: E402


def parse_decimal_comma(series: pd.Series) -> pd.Series:
    return PoliticianTradeModel._parse_decimal_comma(series)


def build_dataset(cfg: Config) -> pd.DataFrame:
    """Load parquet, filter to purchases, build realized_car_hybrid target."""
    print("Loading data...")
    df = pd.read_parquet(cfg.data_path)
    df = df[df["Transaction"] == "Purchase"].copy()

    # Parse numeric columns stored as decimal-comma strings
    for col in ["realized_car", "car_filed_to_12m", "car_traded_to_filed",
                "stock_momentum_30d", "stock_momentum_90d",
                "stock_volatility_30d", "beta", "holding_period_days"]:
        if col in df.columns:
            df[col] = parse_decimal_comma(df[col])

    df["Traded"] = pd.to_datetime(df["Traded"], errors="coerce")
    df["Filed"]  = pd.to_datetime(df["Filed"],  errors="coerce")

    # Determine cutoff (max_traded - 12m), same as main model
    max_traded = df["Traded"].max()
    cutoff = max_traded - pd.DateOffset(months=12)
    df = df[df["Filed"] <= cutoff].copy()
    print(f"After cutoff ({cutoff.date()}): {len(df)} purchases")

    # Build hybrid target (same logic as model_realized.preprocess)
    horizon_days = 366
    mask_realized = (
        (df["position_closed"] == True) &
        (df["holding_period_days"] <= horizon_days) &
        (df["realized_car"].notna())
    )
    df["realized_car_hybrid"] = df["car_filed_to_12m"]
    df.loc[mask_realized, "realized_car_hybrid"] = df.loc[mask_realized, "realized_car"]
    df = df.dropna(subset=["realized_car_hybrid", "stock_momentum_90d"]).copy()

    df["log_trade_size"] = np.log1p(
        pd.to_numeric(df["Trade_Size_USD"], errors="coerce").fillna(0)
    )
    df["momentum_90d_sq"] = df["stock_momentum_90d"] ** 2

    print(f"Analysis dataset: {len(df)} rows")
    print(f"  realized_car_hybrid  mean={df['realized_car_hybrid'].mean():.4f} "
          f" median={df['realized_car_hybrid'].median():.4f}")
    print(f"  stock_momentum_90d   mean={df['stock_momentum_90d'].mean():.4f} "
          f" median={df['stock_momentum_90d'].median():.4f}")
    return df


def run_ols(y: pd.Series, X: pd.DataFrame, model_name: str) -> pd.Series:
    """Run OLS via statsmodels and print a clean summary."""
    import statsmodels.api as sm

    X_const = sm.add_constant(X.copy())
    # Drop rows with any NaN in X or y
    mask = X_const.notna().all(axis=1) & y.notna()
    X_fit = X_const[mask]
    y_fit = y[mask]

    res = sm.OLS(y_fit, X_fit).fit(cov_type="HC3")  # heteroskedasticity-robust SEs

    print(f"\n{'='*70}")
    print(f"{model_name}  (N={len(y_fit):,})")
    print(f"{'='*70}")
    print(f"  R²={res.rsquared:.4f}   Adj-R²={res.rsquared_adj:.4f}   "
          f"F-stat={res.fvalue:.2f} (p={res.f_pvalue:.4f})")
    print(f"\n  {'Variable':<35}{'coef':>10}{'std err':>10}{'t':>8}{'p>|t|':>9}{'sig':>5}")
    print(f"  {'-'*77}")
    for var in res.params.index:
        coef = res.params[var]
        se   = res.bse[var]
        t    = res.tvalues[var]
        p    = res.pvalues[var]
        sig  = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"  {var:<35}{coef:>10.4f}{se:>10.4f}{t:>8.2f}{p:>9.4f}{sig:>5}")

    return res.params


def quintile_analysis(df: pd.DataFrame) -> None:
    """Show mean CAR by prior-return quintile (binned decomposition)."""
    print(f"\n{'='*70}")
    print("BINNED ANALYSIS — Mean CAR by prior-return (momentum_90d) quintile")
    print(f"{'='*70}")

    df = df.copy()
    df["mom_quintile"] = pd.qcut(
        df["stock_momentum_90d"], q=5,
        labels=["Q1\n(most\nneg)", "Q2", "Q3", "Q4", "Q5\n(most\npos)"]
    )

    tbl = df.groupby("mom_quintile", observed=True).agg(
        n=("realized_car_hybrid", "count"),
        mean_mom=("stock_momentum_90d", "mean"),
        mean_car=("realized_car_hybrid", "mean"),
        median_car=("realized_car_hybrid", "median"),
        pct_pos=("realized_car_hybrid", lambda x: (x > 0).mean()),
    ).reset_index()

    print(f"\n  {'Quintile':<10}{'N':>6}{'mean_mom':>12}{'mean_CAR':>12}"
          f"{'median_CAR':>12}{'pct_pos':>10}")
    print(f"  {'-'*62}")
    for _, row in tbl.iterrows():
        label = str(row["mom_quintile"]).replace("\n", " ")
        print(f"  {label:<10}{int(row['n']):>6}{row['mean_mom']:>12.4f}"
              f"{row['mean_car']:>12.4f}{row['median_car']:>12.4f}"
              f"{row['pct_pos']:>9.1%}")

    print(f"\n  Interpretation:")
    q1_car = tbl[tbl["mom_quintile"] == "Q1\n(most\nneg)"]["mean_car"].values[0]
    q5_car = tbl[tbl["mom_quintile"] == "Q5\n(most\npos)"]["mean_car"].values[0]
    spread = q1_car - q5_car
    print(f"  Q1 (beaten-down) mean CAR = {q1_car:+.4f}")
    print(f"  Q5 (momentum)   mean CAR = {q5_car:+.4f}")
    print(f"  Spread Q1-Q5 = {spread:+.4f}  "
          f"({'mean reversion dominant' if spread > 0.02 else 'weak/no mean reversion'})")


def subgroup_regressions(df: pd.DataFrame) -> None:
    """Run Model 1 (simple) separately per momentum quintile.

    If α is significant even in Q3 (neutral momentum), there is an
    information component beyond pure mean reversion.
    """
    print(f"\n{'='*70}")
    print("SUBGROUP REGRESSIONS — α per momentum quintile")
    print("(Does alpha persist even in Q3/Q4/Q5, or only in beaten-down stocks?)")
    print(f"{'='*70}")

    import statsmodels.api as sm

    df = df.copy()
    df["mom_quintile"] = pd.qcut(df["stock_momentum_90d"], q=5, labels=[1, 2, 3, 4, 5])

    print(f"\n  {'Quintile':<12}{'N':>6}{'α (intercept)':>16}{'p(α)':>10}"
          f"{'β (mom)':>12}{'p(β)':>10}{'R²':>8}")
    print(f"  {'-'*74}")

    for q in [1, 2, 3, 4, 5]:
        grp = df[df["mom_quintile"] == q].copy()
        y = grp["realized_car_hybrid"]
        X = sm.add_constant(grp[["stock_momentum_90d"]])
        mask = X.notna().all(axis=1) & y.notna()
        if mask.sum() < 30:
            print(f"  Q{q:<11}{mask.sum():>6}  {'(too few rows)':>16}")
            continue
        res = sm.OLS(y[mask], X[mask]).fit(cov_type="HC3")
        alpha = res.params.get("const", np.nan)
        p_a   = res.pvalues.get("const", np.nan)
        beta  = res.params.get("stock_momentum_90d", np.nan)
        p_b   = res.pvalues.get("stock_momentum_90d", np.nan)
        sig_a = "***" if p_a < 0.01 else ("**" if p_a < 0.05 else ("*" if p_a < 0.10 else ""))
        label = f"Q{q} ({'neg' if q == 1 else ('pos' if q == 5 else 'mid')})"
        print(f"  {label:<12}{mask.sum():>6}{alpha:>14.4f}{sig_a:<2}{p_a:>10.4f}"
              f"{beta:>12.4f}{p_b:>10.4f}{res.rsquared:>8.4f}")


def regime_check(df: pd.DataFrame) -> None:
    """Show baseline mean CAR and momentum by year — context for regime effects."""
    print(f"\n{'='*70}")
    print("ANNUAL BASELINE — Mean CAR and mean prior-momentum by year")
    print(f"{'='*70}")

    df = df.copy()
    df["year"] = df["Filed"].dt.year
    tbl = df.groupby("year").agg(
        n=("realized_car_hybrid", "count"),
        mean_car=("realized_car_hybrid", "mean"),
        mean_mom=("stock_momentum_90d", "mean"),
        pct_pos=("realized_car_hybrid", lambda x: (x > 0).mean()),
    ).reset_index()

    print(f"\n  {'Year':<8}{'N':>6}{'mean_CAR':>12}{'mean_mom':>12}{'pct_pos':>10}")
    print(f"  {'-'*48}")
    for _, row in tbl.iterrows():
        print(f"  {int(row['year']):<8}{int(row['n']):>6}{row['mean_car']:>12.4f}"
              f"{row['mean_mom']:>12.4f}{row['pct_pos']:>9.1%}")


def main():
    cfg = Config()
    df = build_dataset(cfg)

    # ── 1. Simple regression ─────────────────────────────────────────────────
    run_ols(
        y=df["realized_car_hybrid"],
        X=df[["stock_momentum_90d"]],
        model_name="Model 1 — Simple: realized_car ~ momentum_90d",
    )

    # ── 2. Extended regression ───────────────────────────────────────────────
    ext_cols = ["stock_momentum_90d", "stock_momentum_30d",
                "beta", "stock_volatility_30d", "log_trade_size"]
    ext_cols = [c for c in ext_cols if c in df.columns]
    run_ols(
        y=df["realized_car_hybrid"],
        X=df[ext_cols],
        model_name="Model 2 — Extended: + momentum_30d + beta + volatility + trade_size",
    )

    # ── 3. Non-linear (quadratic momentum) ───────────────────────────────────
    nl_cols = ["stock_momentum_90d", "momentum_90d_sq"]
    run_ols(
        y=df["realized_car_hybrid"],
        X=df[nl_cols],
        model_name="Model 3 — Non-linear: + momentum_90d² (tests asymmetric reversion)",
    )

    # ── 4. Binned quintile analysis ──────────────────────────────────────────
    quintile_analysis(df)

    # ── 5. Subgroup α per quintile ────────────────────────────────────────────
    subgroup_regressions(df)

    # ── 6. Annual baseline ───────────────────────────────────────────────────
    regime_check(df)

    # ── Final interpretation ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("HOW TO INTERPRET RESULTS")
    print(f"{'='*70}")
    print("""
  β on momentum_90d:
    < 0 and significant → mean reversion: Congress buys beaten-down stocks
                          and earns positive CAR mechanically
    = 0 or non-sig      → prior momentum does NOT explain congressional returns

  α (intercept) after controlling for momentum:
    > 0 and significant → information: excess return persists even after
                          removing the mechanical mean-reversion component
    ≈ 0 or non-sig      → all the return is explained by momentum / reversion

  Q1 vs Q5 spread (quintile table):
    Large Q1-Q5 spread  → most of the "edge" is concentrated in beaten-down
                          stocks (consistent with buy-the-dip, not information)
    Small/uniform spread → returns are similar across momentum quintiles
                          (consistent with genuine information advantage)
""")


if __name__ == "__main__":
    main()
