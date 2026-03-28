"""
Exploratory Data Analysis — Cumulative Abnormal Returns (CAR)
Analyses CAR at multiple horizons: traded→filed, filed→1m/3m/6m/9m/12m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, os

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

OUT_DIR = "output/car_eda"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_parquet("data/output/politician_trades_enriched.parquet")

car_cols = [
    "car_traded_to_filed",
    "car_filed_to_1m",
    "car_filed_to_3m",
    "car_filed_to_6m",
    "car_filed_to_9m",
    "car_filed_to_12m",
]
for c in car_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

df["Traded"] = pd.to_datetime(df["Traded"], errors="coerce")
df["Filed"]  = pd.to_datetime(df["Filed"].astype(str), errors="coerce")
df["Year"]   = df["Traded"].dt.year

# Friendly labels
label_map = {
    "car_traded_to_filed": "Trade → Filing",
    "car_filed_to_1m":  "Filing + 1m",
    "car_filed_to_3m":  "Filing + 3m",
    "car_filed_to_6m":  "Filing + 6m",
    "car_filed_to_9m":  "Filing + 9m",
    "car_filed_to_12m": "Filing + 12m",
}

purchases = df[df["Transaction"] == "Purchase"].copy()
sales     = df[df["Transaction"].isin(["Sale", "Sale (Full)", "Sale (Partial)"])].copy()

print(f"Total rows: {len(df):,}")
print(f"Purchases:  {len(purchases):,}")
print(f"Sales:      {len(sales):,}")
print()

# ── 2. Descriptive statistics ─────────────────────────────────────────────────
print("=" * 90)
print("DESCRIPTIVE STATISTICS — ALL TRADES")
print("=" * 90)

stats_rows = []
for c in car_cols:
    s = df[c].dropna()
    stats_rows.append({
        "Horizon": label_map[c],
        "N": len(s),
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(),
        "Skew": s.skew(),
        "P5": s.quantile(0.05),
        "P25": s.quantile(0.25),
        "P75": s.quantile(0.75),
        "P95": s.quantile(0.95),
        "% > 0": (s > 0).mean() * 100,
    })

stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False, float_format="{:.4f}".format))
print()

print("=" * 90)
print("DESCRIPTIVE STATISTICS — PURCHASES ONLY")
print("=" * 90)

purch_stats = []
for c in car_cols:
    s = purchases[c].dropna()
    purch_stats.append({
        "Horizon": label_map[c],
        "N": len(s),
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(),
        "% > 0": (s > 0).mean() * 100,
    })
purch_df = pd.DataFrame(purch_stats)
print(purch_df.to_string(index=False, float_format="{:.4f}".format))
print()

print("=" * 90)
print("DESCRIPTIVE STATISTICS — SALES ONLY")
print("=" * 90)

sales_stats = []
for c in car_cols:
    s = sales[c].dropna()
    sales_stats.append({
        "Horizon": label_map[c],
        "N": len(s),
        "Mean": s.mean(),
        "Median": s.median(),
        "Std": s.std(),
        "% > 0": (s > 0).mean() * 100,
    })
sales_df = pd.DataFrame(sales_stats)
print(sales_df.to_string(index=False, float_format="{:.4f}".format))
print()

# ── 3. Distribution plots ────────────────────────────────────────────────────
# 3a. Histograms — all filed horizons
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, c in enumerate(car_cols):
    ax = axes[i]
    data = purchases[c].dropna()
    # Clip for visualization
    clipped = data.clip(-1, 1)
    ax.hist(clipped, bins=80, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(data.median(), color="orange", linestyle="-", linewidth=1.5, label=f"median={data.median():.3f}")
    ax.axvline(data.mean(), color="green", linestyle="-", linewidth=1.5, label=f"mean={data.mean():.3f}")
    ax.set_title(label_map[c], fontsize=13, fontweight="bold")
    ax.set_xlabel("CAR")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
fig.suptitle("Distribution of CAR — Purchases (clipped to ±100%)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_car_histograms.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_car_histograms.png")

# 3b. Box plots side by side
filed_cols = [c for c in car_cols if "filed_to" in c]
box_data = [purchases[c].dropna().clip(-1, 1) for c in filed_cols]
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(box_data, labels=[label_map[c] for c in filed_cols], patch_artist=True,
                showfliers=False, medianprops=dict(color="red", linewidth=2))
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_ylabel("CAR")
ax.set_title("CAR Distribution by Horizon — Purchases (no outliers)", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_car_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_car_boxplots.png")

# ── 4. Mean CAR by horizon (bar chart) ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Purchases
means_p = [purchases[c].dropna().mean() for c in filed_cols]
medians_p = [purchases[c].dropna().median() for c in filed_cols]
x = range(len(filed_cols))
labels = [label_map[c] for c in filed_cols]

axes[0].bar(x, means_p, width=0.4, label="Mean", color="#4C72B0", alpha=0.8, align="center")
axes[0].bar([i + 0.4 for i in x], medians_p, width=0.4, label="Median", color="#55A868", alpha=0.8, align="center")
axes[0].set_xticks([i + 0.2 for i in x])
axes[0].set_xticklabels(labels, rotation=15)
axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
axes[0].set_title("Purchases — Mean vs Median CAR", fontsize=13, fontweight="bold")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[0].legend()

# Sales
means_s = [sales[c].dropna().mean() for c in filed_cols]
medians_s = [sales[c].dropna().median() for c in filed_cols]

axes[1].bar(x, means_s, width=0.4, label="Mean", color="#C44E52", alpha=0.8, align="center")
axes[1].bar([i + 0.4 for i in x], medians_s, width=0.4, label="Median", color="#CCB974", alpha=0.8, align="center")
axes[1].set_xticks([i + 0.2 for i in x])
axes[1].set_xticklabels(labels, rotation=15)
axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
axes[1].set_title("Sales — Mean vs Median CAR", fontsize=13, fontweight="bold")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_car_mean_median_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_car_mean_median_bars.png")

# ── 5. % Positive by horizon ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pct_pos_p = [(purchases[c].dropna() > 0).mean() * 100 for c in filed_cols]
pct_pos_s = [(sales[c].dropna() > 0).mean() * 100 for c in filed_cols]

ax.plot(labels, pct_pos_p, "o-", color="#4C72B0", linewidth=2, markersize=8, label="Purchases")
ax.plot(labels, pct_pos_s, "s--", color="#C44E52", linewidth=2, markersize=8, label="Sales")
ax.axhline(50, color="gray", linestyle=":", linewidth=1)
ax.set_ylabel("% of Trades with CAR > 0")
ax.set_title("Win Rate by Horizon", fontsize=14, fontweight="bold")
ax.set_ylim(40, 65)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_win_rate_by_horizon.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_win_rate_by_horizon.png")

# ── 6. CAR over time (yearly mean) ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
yearly = purchases.groupby("Year")[filed_cols].mean()
for c in filed_cols:
    ax.plot(yearly.index, yearly[c], "o-", label=label_map[c], linewidth=1.5, markersize=5)
ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
ax.set_xlabel("Year")
ax.set_ylabel("Mean CAR")
ax.set_title("Mean CAR per Year — Purchases", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_car_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_car_yearly_trend.png")

# ── 7. Correlation matrix between horizons ────────────────────────────────────
corr = purchases[car_cols].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-0.3, vmax=1)
ax.set_xticks(range(len(car_cols)))
ax.set_yticks(range(len(car_cols)))
ax.set_xticklabels([label_map[c] for c in car_cols], rotation=45, ha="right")
ax.set_yticklabels([label_map[c] for c in car_cols])
for i in range(len(car_cols)):
    for j in range(len(car_cols)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(corr.values[i, j]) > 0.6 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlation Matrix — CAR Horizons (Purchases)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_car_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_car_correlation.png")

# ── 8. Party analysis ────────────────────────────────────────────────────────
if "Party" in df.columns:
    print()
    print("=" * 90)
    print("CAR BY PARTY — PURCHASES")
    print("=" * 90)
    party_stats = []
    for party in purchases["Party"].dropna().unique():
        subset = purchases[purchases["Party"] == party]
        row = {"Party": party, "N": len(subset)}
        for c in filed_cols:
            s = subset[c].dropna()
            row[label_map[c] + " mean"] = s.mean()
            row[label_map[c] + " median"] = s.median()
            row[label_map[c] + " %>0"] = (s > 0).mean() * 100
        party_stats.append(row)
    party_df = pd.DataFrame(party_stats).sort_values("N", ascending=False)
    print(party_df.to_string(index=False, float_format="{:.4f}".format))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    parties = party_df["Party"].tolist()[:3]  # Top 3
    party_colors = {"Republican": "#C44E52", "Democrat": "#4C72B0", "Independent": "#55A868"}

    for party in parties:
        subset = purchases[purchases["Party"] == party]
        means = [subset[c].dropna().mean() for c in filed_cols]
        color = party_colors.get(party, "gray")
        axes[0].plot(labels, means, "o-", label=party, color=color, linewidth=2, markersize=7)
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_title("Mean CAR by Party", fontsize=13, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[0].legend()

    for party in parties:
        subset = purchases[purchases["Party"] == party]
        pct = [(subset[c].dropna() > 0).mean() * 100 for c in filed_cols]
        color = party_colors.get(party, "gray")
        axes[1].plot(labels, pct, "o-", label=party, color=color, linewidth=2, markersize=7)
    axes[1].axhline(50, color="gray", linestyle=":")
    axes[1].set_title("Win Rate by Party", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("% CAR > 0")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_car_by_party.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_car_by_party.png")

# ── 9. Committee conflict analysis ───────────────────────────────────────────
print()
print("=" * 90)
print("CAR BY COMMITTEE CONFLICT MATCH — PURCHASES")
print("=" * 90)

match_cols_industry = ["Industry match 1", "Industry match 2", "Industry match 3"]
match_cols_sector   = ["Sector match 1", "Sector match 2", "Sector match 3"]

for mc_col in match_cols_industry + match_cols_sector:
    if mc_col not in purchases.columns:
        continue
    purchases[mc_col] = pd.to_numeric(purchases[mc_col], errors="coerce").fillna(0)

# Any industry match
purchases["has_industry_match"] = (
    purchases[match_cols_industry].max(axis=1) > 0
).astype(int)

# Any sector match
purchases["has_sector_match"] = (
    purchases[match_cols_sector].max(axis=1) > 0
).astype(int)

for flag_col, flag_name in [("has_industry_match", "Industry Match"), ("has_sector_match", "Sector Match")]:
    print(f"\n--- {flag_name} ---")
    for val, vname in [(1, "Yes"), (0, "No")]:
        subset = purchases[purchases[flag_col] == val]
        parts = [f"{vname} (N={len(subset):,}):"]
        for c in filed_cols:
            s = subset[c].dropna()
            parts.append(f"  {label_map[c]}: mean={s.mean():.4f}, median={s.median():.4f}, %>0={((s>0).mean()*100):.1f}%")
        print("\n".join(parts))

# Plot industry match vs no match
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for flag_col, flag_name, ax in [("has_industry_match", "Industry Match", axes[0]),
                                  ("has_sector_match", "Sector Match", axes[1])]:
    for val, vname, color in [(1, "With match", "#C44E52"), (0, "No match", "#4C72B0")]:
        subset = purchases[purchases[flag_col] == val]
        means = [subset[c].dropna().mean() for c in filed_cols]
        ax.plot(labels, means, "o-", label=f"{vname} (N={len(subset):,})", color=color, linewidth=2, markersize=7)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title(f"Mean CAR — {flag_name}", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_car_by_committee_match.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 08_car_by_committee_match.png")

# ── 10. Top/bottom politicians ────────────────────────────────────────────────
print()
print("=" * 90)
print("TOP 15 POLITICIANS BY MEAN CAR (6m) — Purchases, min 20 trades")
print("=" * 90)

pol_group = purchases.groupby("Name").agg(
    n_trades=("car_filed_to_6m", "count"),
    mean_1m=("car_filed_to_1m", "mean"),
    mean_3m=("car_filed_to_3m", "mean"),
    mean_6m=("car_filed_to_6m", "mean"),
    mean_12m=("car_filed_to_12m", "mean"),
    pct_pos_6m=("car_filed_to_6m", lambda x: (x.dropna() > 0).mean() * 100),
).reset_index()

pol_active = pol_group[pol_group["n_trades"] >= 20].sort_values("mean_6m", ascending=False)
print("\nTop 15:")
print(pol_active.head(15).to_string(index=False, float_format="{:.4f}".format))
print("\nBottom 15:")
print(pol_active.tail(15).to_string(index=False, float_format="{:.4f}".format))

# ── 11. Trade size buckets ────────────────────────────────────────────────────
if "Trade_Size_USD" in purchases.columns:
    purchases["Trade_Size_USD_num"] = pd.to_numeric(purchases["Trade_Size_USD"], errors="coerce")
    size_bins = [0, 15_000, 50_000, 100_000, 250_000, 500_000, float("inf")]
    size_labels_b = ["<15K", "15-50K", "50-100K", "100-250K", "250-500K", ">500K"]
    purchases["size_bucket"] = pd.cut(purchases["Trade_Size_USD_num"], bins=size_bins, labels=size_labels_b)

    print()
    print("=" * 90)
    print("CAR BY TRADE SIZE — PURCHASES")
    print("=" * 90)
    for bucket in size_labels_b:
        subset = purchases[purchases["size_bucket"] == bucket]
        parts = [f"{bucket} (N={len(subset):,}):"]
        for c in filed_cols:
            s = subset[c].dropna()
            if len(s) > 0:
                parts.append(f"  {label_map[c]}: mean={s.mean():.4f}, median={s.median():.4f}")
        print("\n".join(parts))

    fig, ax = plt.subplots(figsize=(12, 6))
    for bucket in size_labels_b:
        subset = purchases[purchases["size_bucket"] == bucket]
        if len(subset) >= 10:
            means = [subset[c].dropna().mean() for c in filed_cols]
            ax.plot(labels, means, "o-", label=f"{bucket} (N={len(subset):,})", linewidth=1.5, markersize=6)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Mean CAR by Trade Size — Purchases", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/09_car_by_trade_size.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 09_car_by_trade_size.png")

# ── 12. Filing lag analysis ──────────────────────────────────────────────────
# Compute filing lag from Traded and Filed dates
purchases["lag_days"] = (purchases["Filed"] - purchases["Traded"]).dt.days
if purchases["lag_days"].notna().any():
    lag_bins = [0, 15, 30, 45, 60, 90, float("inf")]
    lag_labels_b = ["0-15d", "15-30d", "30-45d", "45-60d", "60-90d", ">90d"]
    purchases["lag_bucket"] = pd.cut(purchases["lag_days"], bins=lag_bins, labels=lag_labels_b)

    print()
    print("=" * 90)
    print("CAR BY FILING LAG — PURCHASES")
    print("=" * 90)
    for bucket in lag_labels_b:
        subset = purchases[purchases["lag_bucket"] == bucket]
        parts = [f"{bucket} (N={len(subset):,}):"]
        for c in filed_cols:
            s = subset[c].dropna()
            if len(s) > 0:
                parts.append(f"  {label_map[c]}: mean={s.mean():.4f}, median={s.median():.4f}")
        print("\n".join(parts))

    fig, ax = plt.subplots(figsize=(12, 6))
    for bucket in lag_labels_b:
        subset = purchases[purchases["lag_bucket"] == bucket]
        if len(subset) >= 10:
            means = [subset[c].dropna().mean() for c in filed_cols]
            ax.plot(labels, means, "o-", label=f"{bucket} (N={len(subset):,})", linewidth=1.5, markersize=6)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_title("Mean CAR by Filing Lag — Purchases", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/10_car_by_filing_lag.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 10_car_by_filing_lag.png")

# ── 13. Summary ──────────────────────────────────────────────────────────────
print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Total trades analyzed: {len(df):,}")
print(f"Purchases: {len(purchases):,}")
print(f"Sales: {len(sales):,}")
print(f"Date range: {df['Traded'].min().date()} to {df['Traded'].max().date()}")
print(f"Unique politicians: {df['Name'].nunique()}")
print(f"Unique tickers: {df['Ticker'].nunique()}")
print()
print("Key findings (Purchases):")
for c in filed_cols:
    s = purchases[c].dropna()
    direction = "outperform" if s.mean() > 0 else "underperform"
    print(f"  {label_map[c]:15s}: mean={s.mean():+.4f} ({direction}), win rate={(s>0).mean()*100:.1f}%")
print()
print(f"All plots saved to {OUT_DIR}/")
