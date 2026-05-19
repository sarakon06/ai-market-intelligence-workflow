"""
plot_feature_importance.py
Run this from the root of your repo after running ai_market_intel.py.
Saves feature_importance.png to yellowwood_outputs/ for embedding in README.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = Path("yellowwood_outputs/feature_importance.csv")
OUTPUT_PNG = Path("yellowwood_outputs/feature_importance.png")
TOP_N      = 10

# Cleaner display names for the top features
RENAME = {
    "GSPC_ret_6m":                    "S&P 500 — 6M Return",
    "rel_perf_GSPC_vs_MSFT_3m":       "S&P 500 vs MSFT — 3M Rel. Perf.",
    "GSPC_ma_gap_6m":                  "S&P 500 — 6M MA Gap",
    "GSPC_vol_3m":                     "S&P 500 — 3M Volatility",
    "rel_perf_GSPC_vs_GOOGL_6m":      "S&P 500 vs GOOGL — 6M Rel. Perf.",
    "GSPC_ret_3m":                     "S&P 500 — 3M Return",
    "rel_perf_GSPC_vs_AMZN_6m":       "S&P 500 vs AMZN — 6M Rel. Perf.",
    "GSPC_ret_1m":                     "S&P 500 — 1M Return",
    "GSPC_ma_gap_3m":                  "S&P 500 — 3M MA Gap",
    "GOOGL_ret_1m":                    "GOOGL — 1M Return",
    "FedFundsRate_chg_1m":             "Fed Funds Rate — 1M Change",
    "CPI_chg_1m":                      "CPI — 1M Change",
    "UnemploymentRate_lag_1":          "Unemployment Rate — 1M Lag",
    "ConsumerSentiment_lag_1":         "Consumer Sentiment — 1M Lag",
    "RetailSales_chg_3m":              "Retail Sales — 3M Change",
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, index_col=0)
df.columns = ["importance"]
df = df.sort_values("importance", ascending=False).head(TOP_N)
df.index = [RENAME.get(f, f) for f in df.index]
df = df.sort_values("importance", ascending=True)   # ascending for horizontal bar

# ── Plot ──────────────────────────────────────────────────────────────────────
NAVY   = "#1F3864"
TEAL   = "#2E8B7A"
LIGHT  = "#F7F9FC"

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(LIGHT)
ax.set_facecolor(LIGHT)

bars = ax.barh(
    df.index,
    df["importance"],
    color=TEAL,
    edgecolor="white",
    linewidth=0.6,
    height=0.62,
)

# Value labels on bars
for bar in bars:
    w = bar.get_width()
    ax.text(
        w + 0.0005,
        bar.get_y() + bar.get_height() / 2,
        f"{w:.4f}",
        va="center",
        ha="left",
        fontsize=8.5,
        color="#444444",
    )

# Gridlines
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#CCCCCC")
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
ax.tick_params(axis="y", labelsize=9.5, colors="#222222")
ax.tick_params(axis="x", labelsize=8.5, colors="#666666")

# Titles
ax.set_title(
    "Top 10 Feature Importances — Random Forest Walk-Forward Model",
    fontsize=12,
    fontweight="bold",
    color=NAVY,
    pad=14,
    loc="left",
)
ax.set_xlabel("Mean Feature Importance (avg. across 24-month walk-forward window)", fontsize=8.5, color="#666666")

# Footnote
fig.text(
    0.01, -0.04,
    "Model: Random Forest (300 estimators, max_depth=6)  ·  "
    "Target: S&P 500 monthly direction  ·  "
    "Test window: Jan 2023 – Dec 2024 (walk-forward, out-of-sample)",
    fontsize=7.5,
    color="#888888",
)

plt.tight_layout()
OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight", facecolor=LIGHT)
plt.close()
print(f"Saved → {OUTPUT_PNG.resolve()}")
