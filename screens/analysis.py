"""Compare pre-rally indicators: rally winners vs everything else.

The point: a pattern only matters if winners differ from non-winners. This
script computes indicators as of the rally-start date for BOTH groups and
shows the gap. If the groups look the same, the 'pattern' is selection bias.

NOT a predictive model. A diagnostic to teach you what real vs fake signals
look like.
"""
import duckdb
import numpy as np
import pandas as pd
from datetime import timedelta
from engine import config
from engine import indicators as ta

RALLY_START = pd.Timestamp("2026-03-30")   # the common bottom you spotted
WINNER_GAIN = 50.0      # define "winner" as >50% in the window
WINDOW_DAYS = 60
LOOKBACK = 90           # days of history before rally start to compute indicators

con = duckdb.connect(config.DUCKDB_PATH)
symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]

records = []
for sym in symbols:
    if sym.endswith("-ST") or sym.endswith("-SM"):
        continue
    df = con.execute(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE symbol = ? ORDER BY date", [sym]
    ).fetchdf()
    if len(df) < LOOKBACK + 5:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # split: history up to rally start (to compute indicators) and the window after
    pre = df[df.index < RALLY_START]
    post = df[(df.index >= RALLY_START) & (df.index <= RALLY_START + timedelta(days=WINDOW_DAYS))]
    if len(pre) < LOOKBACK or len(post) < 10:
        continue

    # liquidity filter (same ₹5cr floor)
    if (pre["close"] * pre["volume"]).tail(60).median() < 5_00_00_000:
        continue

    # indicators as of the last day BEFORE the rally (no look-ahead)
    rsi14 = ta.rsi(pre["close"], 14).iloc[-1]
    above_50dma = pre["close"].iloc[-1] / ta.sma(pre["close"], 50).iloc[-1] - 1
    above_200dma = pre["close"].iloc[-1] / ta.sma(pre["close"], 200).iloc[-1] - 1 \
        if len(pre) >= 200 else np.nan
    ret_3m = pre["close"].iloc[-1] / pre["close"].iloc[-63] - 1 if len(pre) >= 63 else np.nan
    vol_3m = pre["close"].pct_change().tail(63).std() * np.sqrt(252)
    dist_from_high = pre["close"].iloc[-1] / pre["close"].tail(252).max() - 1

    # outcome
    gain = post["high"].max() / post["low"].min() - 1

    records.append({
        "ticker": sym,
        "gain_pct": round(gain * 100, 1),
        "is_winner": gain * 100 >= WINNER_GAIN,
        "rsi14": round(rsi14, 1),
        "pct_vs_50dma": round(above_50dma * 100, 1),
        "pct_vs_200dma": round(above_200dma * 100, 1) if not np.isnan(above_200dma) else None,
        "ret_prior_3m": round(ret_3m * 100, 1) if not np.isnan(ret_3m) else None,
        "ann_vol_pct": round(vol_3m * 100, 1),
        "pct_below_52w_high": round(dist_from_high * 100, 1),
    })

con.close()
data = pd.DataFrame(records)
print(f"Analyzed {len(data)} liquid stocks. "
      f"{data['is_winner'].sum()} winners (>{WINNER_GAIN}%), "
      f"{(~data['is_winner']).sum()} non-winners.\n")

cols = ["rsi14", "pct_vs_50dma", "pct_vs_200dma", "ret_prior_3m",
        "ann_vol_pct", "pct_below_52w_high"]
comp = data.groupby("is_winner")[cols].median().T
comp.columns = ["non_winners (median)", "winners (median)"]
comp["difference"] = comp["winners (median)"] - comp["non_winners (median)"]
print("Pre-rally indicators: winners vs non-winners (medians)\n")
print(comp.round(2).to_string())

data.to_csv("pre_rally_analysis.csv", index=False)
print("\nFull per-stock data saved to pre_rally_analysis.csv")
print("\nHOW TO READ THIS: a 'difference' near zero means that indicator did NOT")
print("distinguish winners from losers — i.e. it was NOT a useful pre-rally signal,")
print("even if every winner happened to have it. Only sizeable, consistent gaps")
print("are even candidates for a real pattern (and even those need further testing).")