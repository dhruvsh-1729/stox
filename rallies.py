"""Biggest 2-month rallies among real, liquid mainboard stocks.

Cleanups vs the naive version:
  - TRUE date window: last 60 calendar days, not "last 42 rows" (fixes the
    multi-year gaps from stocks with missing data).
  - Drops -ST / -SM / surveillance & SME names.
  - Liquidity floor: median daily turnover must clear a threshold, so
    thinly-traded micro-caps (the pump-and-dump zone) are excluded.
  - Requires enough trading days in the window (no sparse-data artifacts).

This still ranks by PAST gain = past volatility. It is a 'what happened to
real companies' report, NOT a buy list.
"""
import duckdb
import pandas as pd
from datetime import timedelta
from engine import config

WINDOW_DAYS = 60          # real calendar window
MIN_TRADING_DAYS = 30     # must have actually traded most of the window
MIN_MEDIAN_TURNOVER = 5_00_00_000   # ₹5 crore median daily turnover floor

con = duckdb.connect(config.DUCKDB_PATH)

# most recent date in the cache, and the window start
max_date = con.execute("SELECT max(date) FROM prices").fetchone()[0]
start = max_date - timedelta(days=WINDOW_DAYS)
print(f"Window: {start} to {max_date}\n")

symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]

rows = []
for sym in symbols:
    # exclude SME / surveillance / trade-for-trade by suffix
    if sym.endswith("-ST") or sym.endswith("-SM"):
        continue

    df = con.execute(
        "SELECT date, low, high, close, volume FROM prices "
        "WHERE symbol = ? AND date >= ? ORDER BY date",
        [sym, start],
    ).fetchdf()

    if len(df) < MIN_TRADING_DAYS:
        continue

    # liquidity floor: median daily turnover over the window
    turnover = (df["close"] * df["volume"]).median()
    if turnover < MIN_MEDIAN_TURNOVER:
        continue

    df = df.reset_index(drop=True)
    low_idx = df["low"].idxmin()
    after = df.iloc[low_idx:]
    high_idx = after["high"].idxmax()

    buy = float(df.loc[low_idx, "low"])
    sell = float(df.loc[high_idx, "high"])
    if buy <= 0:
        continue

    rows.append({
        "ticker": sym,
        "buy_date": df.loc[low_idx, "date"],
        "buy_price": round(buy, 2),
        "sell_date": df.loc[high_idx, "date"],
        "sell_price": round(sell, 2),
        "gain_pct": round((sell / buy - 1) * 100, 1),
        "med_turnover_cr": round(turnover / 1e7, 1),
    })

con.close()

result = pd.DataFrame(rows).sort_values("gain_pct", ascending=False)
print(f"{len(result)} liquid mainboard stocks passed the filters.\n")
print("Top 100 rallies among real, liquid companies:\n")
print(result.head(100).to_string(index=False))
result.to_csv("top_rallies_clean.csv", index=False)
print("\nFull results saved to top_rallies_clean.csv")