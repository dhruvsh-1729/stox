"""Technical portion of a small-cap breakout screen, against the cache.

Filters applied (the only 3 of 12 criteria derivable from Kite OHLCV):
  * Current price < ₹400               -- small-cap / cheap-share band
  * Current price > 200 DMA            -- in established uptrend
  * Last day's volume > 20d avg volume -- volume picking up

Plus a liquidity floor (₹2 cr median daily turnover — looser than the momentum
screen since this is meant to catch small-caps).

Caveats:
  * Mcap, P/E, Industry P/E, OPM, CCC, ROCE, sales/profit growth, D/E, CFO are
    NOT available from Kite. This is the technical sleeve only.
  * Cache is 3 trading days stale; the volume signal in particular benefits
    from a fresh fetch.
"""
import duckdb
import numpy as np
import pandas as pd
from engine import config
from engine import indicators as ta

MIN_TURNOVER = 2_00_00_000   # ₹2 cr/day
PRICE_CEIL   = 400.0
VOL_MA_DAYS  = 20

con = duckdb.connect(config.DUCKDB_PATH, read_only=True)
asof = con.execute("SELECT max(date) FROM prices").fetchone()[0]
symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
print(f"Screening as of {asof} ({len(symbols)} cached symbols)\n")

# Drop the obvious commodity-tracker leaks the universe filter missed.
SILVER_LIKE = ("SILVER", "HDFCSILVER", "AXISILVER")

rows = []
for sym in symbols:
    if sym.endswith("-ST") or sym.endswith("-SM"):
        continue
    if sym.startswith("SILVER") or sym in SILVER_LIKE:
        continue

    df = con.execute(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE symbol = ? ORDER BY date", [sym]
    ).fetchdf()
    if len(df) < 220:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if (df["close"] * df["volume"]).tail(60).median() < MIN_TURNOVER:
        continue

    close = df["close"]
    vol = df["volume"]
    price = close.iloc[-1]

    if price >= PRICE_CEIL:
        continue

    dma200 = ta.sma(close, 200).iloc[-1]
    if np.isnan(dma200) or price <= dma200:
        continue

    vol_today = vol.iloc[-1]
    vol_ma = vol.iloc[-VOL_MA_DAYS-1:-1].mean()   # exclude today, average prior 20d
    if vol_today <= vol_ma:
        continue

    dma50 = ta.sma(close, 50).iloc[-1]
    rsi14 = ta.rsi(close, 14).iloc[-1]
    ret_1m = price / close.iloc[-21] - 1 if len(close) > 21 else np.nan
    ret_3m = price / close.iloc[-63] - 1 if len(close) > 63 else np.nan
    high_52w = close.tail(252).max()
    drawdown = price / high_52w - 1
    vol_surge = vol_today / vol_ma

    rows.append({
        "ticker": sym,
        "price": round(float(price), 2),
        "dma50": round(float(dma50), 2) if not np.isnan(dma50) else None,
        "dma200": round(float(dma200), 2),
        "pct_above_200dma": round((price / dma200 - 1) * 100, 1),
        "pct_above_50dma":  round((price / dma50  - 1) * 100, 1) if not np.isnan(dma50) else None,
        "vol_surge_x": round(float(vol_surge), 2),
        "rsi14": round(float(rsi14), 1),
        "ret_1m_pct": round(float(ret_1m) * 100, 1) if not np.isnan(ret_1m) else None,
        "ret_3m_pct": round(float(ret_3m) * 100, 1) if not np.isnan(ret_3m) else None,
        "pct_below_52w_high": round(float(drawdown) * 100, 1),
        "med_turnover_cr": round(
            float((close * vol).tail(60).median()) / 1e7, 2
        ),
    })

con.close()
res = pd.DataFrame(rows).sort_values("vol_surge_x", ascending=False)
print(f"{len(res)} liquid mainboard names pass the technical filter.\n")

# Sanity-check sub-segments.
under_100 = res[res["price"] < 100]
band_100_200 = res[(res["price"] >= 100) & (res["price"] < 200)]
band_200_400 = res[res["price"] >= 200]
print("Price-band distribution:")
print(f"  < ₹100      : {len(under_100):>3}")
print(f"  ₹100 – 200  : {len(band_100_200):>3}")
print(f"  ₹200 – 400  : {len(band_200_400):>3}\n")

# Volume surge bands.
print("Volume surge distribution (today / 20d avg):")
for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 100)]:
    n = ((res["vol_surge_x"] >= lo) & (res["vol_surge_x"] < hi)).sum()
    label = f"  {lo:.1f}x – {hi:.1f}x" if hi < 100 else f"  > {lo:.1f}x"
    print(f"{label:<16}: {n:>3}")
print()

print("Top 30 by volume surge:\n")
print(res.head(30).to_string(index=False))
res.to_csv("breakout_filter.csv", index=False)
print("\nFull list saved to breakout_filter.csv")
