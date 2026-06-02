"""Technical portion of a screener.in-style query, applied to Kite OHLCV.

Filters applied (the subset of your query that doesn't need fundamentals):
  * Current price > 50-day SMA
  * 50-day SMA > 200-day SMA
  * 6-month return > 15%

Plus a liquidity floor (₹5 cr median daily turnover) so we don't surface
illiquid noise. NOT the same as your full screener.in query — Market Cap,
P/E, ROE, D/E are not derivable from Kite data.
"""
import duckdb
import numpy as np
import pandas as pd
from engine import config
from engine import indicators as ta

MIN_TURNOVER = 5_00_00_000   # ₹5 cr/day
LOOKBACK = 200               # need enough history for 200 DMA

con = duckdb.connect(config.DUCKDB_PATH, read_only=True)
asof = con.execute("SELECT max(date) FROM prices").fetchone()[0]
symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
print(f"Screening as of {asof}, {len(symbols)} cached symbols\n")

rows = []
for sym in symbols:
    if sym.endswith("-ST") or sym.endswith("-SM"):
        continue
    df = con.execute(
        "SELECT date, close, volume FROM prices "
        "WHERE symbol = ? ORDER BY date", [sym]
    ).fetchdf()
    if len(df) < LOOKBACK + 5:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if (df["close"] * df["volume"]).tail(60).median() < MIN_TURNOVER:
        continue

    close = df["close"]
    dma50 = ta.sma(close, 50).iloc[-1]
    dma200 = ta.sma(close, 200).iloc[-1]
    price = close.iloc[-1]
    if len(close) < 126:
        continue
    ret_6m = price / close.iloc[-126] - 1

    if not (price > dma50 and dma50 > dma200 and ret_6m > 0.15):
        continue

    rows.append({
        "ticker": sym,
        "price": round(float(price), 2),
        "dma50": round(float(dma50), 2),
        "dma200": round(float(dma200), 2),
        "pct_above_50dma": round((price / dma50 - 1) * 100, 1),
        "pct_above_200dma": round((price / dma200 - 1) * 100, 1),
        "ret_6m_pct": round(float(ret_6m) * 100, 1),
        "med_turnover_cr": round(
            float((df["close"] * df["volume"]).tail(60).median()) / 1e7, 1
        ),
    })

con.close()
res = pd.DataFrame(rows).sort_values("ret_6m_pct", ascending=False)
print(f"{len(res)} liquid mainboard names pass the technical filter.\n")
print("Top 30 by 6-month return:\n")
print(res.head(30).to_string(index=False))
res.to_csv("tech_filter.csv", index=False)
print("\nFull list saved to tech_filter.csv")
