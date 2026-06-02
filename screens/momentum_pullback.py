"""Screener.in-style momentum/pullback query, applied to cached Kite OHLCV.

Conditions:
  * Close > DMA 20
  * Close > DMA 50
  * DMA 20 > DMA 50
  * DMA 20 < DMA 50 * 1.05            (20-DMA not yet stretched vs 50-DMA)
  * Volume(today) > Volume_1mo_avg * 1.5
  * 50 < RSI(14) < 65

The original query also has `Price to Earning < Industry PE`. PE / industry-PE
are not in the Kite-sourced cache (`fundamentals` table is empty), so that leg
is reported but NOT applied — final list is the technical subset.

Run:
  python -m screens.momentum_pullback
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd

from engine import config
from engine import indicators as ta

MIN_TURNOVER = 5_00_00_000   # ₹5 cr/day liquidity floor
LOOKBACK = 80                # need >= 50 for DMA50, plus RSI warm-up


def screen() -> pd.DataFrame:
    con = duckdb.connect(config.DUCKDB_PATH, read_only=True)
    asof = con.execute("SELECT max(date) FROM prices").fetchone()[0]
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM prices"
    ).fetchall()]
    print(f"Screening as of {asof}, {len(symbols)} cached symbols\n")

    rows = []
    for sym in symbols:
        if sym.endswith("-ST") or sym.endswith("-SM"):
            continue
        df = con.execute(
            "SELECT date, close, volume FROM prices "
            "WHERE symbol = ? ORDER BY date", [sym]
        ).fetchdf()
        if len(df) < LOOKBACK:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        close = df["close"]
        vol = df["volume"]

        if (close * vol).tail(60).median() < MIN_TURNOVER:
            continue

        dma20 = ta.sma(close, 20).iloc[-1]
        dma50 = ta.sma(close, 50).iloc[-1]
        price = float(close.iloc[-1])
        if np.isnan(dma20) or np.isnan(dma50):
            continue

        vol_today = float(vol.iloc[-1])
        vol_avg_1m = float(vol.tail(21).mean())
        if vol_avg_1m <= 0:
            continue

        rsi14 = float(ta.rsi(close, 14).iloc[-1])
        if np.isnan(rsi14):
            continue

        if not (price > dma20
                and price > dma50
                and dma20 > dma50
                and dma20 < dma50 * 1.05
                and vol_today > vol_avg_1m * 1.5
                and 50 < rsi14 < 65):
            continue

        rows.append({
            "ticker": sym,
            "price": round(price, 2),
            "dma20": round(float(dma20), 2),
            "dma50": round(float(dma50), 2),
            "dma20_over_dma50_pct": round((dma20 / dma50 - 1) * 100, 2),
            "vol_x_avg": round(vol_today / vol_avg_1m, 2),
            "rsi14": round(rsi14, 1),
            "med_turnover_cr": round(
                float((close * vol).tail(60).median()) / 1e7, 1
            ),
        })

    con.close()
    return pd.DataFrame(rows)


def main():
    res = screen()
    if res.empty:
        print("No names pass the technical filter today.")
        return
    res = res.sort_values("vol_x_avg", ascending=False)
    print(f"{len(res)} names pass the technical filter "
          f"(PE < Industry PE not applied — fundamentals cache is empty).\n")
    print(res.to_string(index=False))
    out = "momentum_pullback.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
