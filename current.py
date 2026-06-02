"""Stocks whose CURRENT profile matches the pre-rally winners:
beaten-down, discounted, high-volatility liquid names.

READ THIS: the winners-vs-losers analysis showed this profile predicts
AMPLITUDE, not DIRECTION. These are the stocks most likely to move a lot —
up OR down — depending on what the market does next, which is unknown. This
is a RISK screen, not a buy list. The names at the top are the most volatile,
most distressed, and therefore most dangerous, not the most promising.
"""
import duckdb
import numpy as np
import pandas as pd
from engine import config
from engine import indicators as ta

LOOKBACK = 200
MIN_TURNOVER = 5_00_00_000   # ₹5 cr/day liquidity floor

con = duckdb.connect(config.DUCKDB_PATH)
asof = con.execute("SELECT max(date) FROM prices").fetchone()[0]
symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices").fetchall()]
print(f"Screening as of {asof}\n")

rows = []
for sym in symbols:
    if sym.endswith("-ST") or sym.endswith("-SM"):
        continue
    df = con.execute(
        "SELECT date, high, low, close, volume FROM prices "
        "WHERE symbol = ? ORDER BY date", [sym]
    ).fetchdf()
    if len(df) < LOOKBACK:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if (df["close"] * df["volume"]).tail(60).median() < MIN_TURNOVER:
        continue

    rsi14 = ta.rsi(df["close"], 14).iloc[-1]
    vs_50 = df["close"].iloc[-1] / ta.sma(df["close"], 50).iloc[-1] - 1
    vs_200 = df["close"].iloc[-1] / ta.sma(df["close"], 200).iloc[-1] - 1
    ret_3m = df["close"].iloc[-1] / df["close"].iloc[-63] - 1
    ann_vol = df["close"].pct_change().tail(63).std() * np.sqrt(252)
    below_high = df["close"].iloc[-1] / df["close"].tail(252).max() - 1

    # "match the winner profile" score: more negative + more volatile = higher.
    # built to mirror the medians we found (low RSI, far below MAs, beaten down,
    # high vol). This is a similarity score, NOT a quality or return score.
    score = (
        (40 - rsi14) / 40
        + (-vs_200)
        + (-ret_3m)
        + (-below_high)
        + ann_vol            # the dominant term — this is a volatility screen
    )

    rows.append({
        "ticker": sym,
        "rsi14": round(rsi14, 1),
        "pct_vs_50dma": round(vs_50 * 100, 1),
        "pct_vs_200dma": round(vs_200 * 100, 1),
        "ret_prior_3m_pct": round(ret_3m * 100, 1),
        "ann_vol_pct": round(ann_vol * 100, 1),
        "pct_below_52w_high": round(below_high * 100, 1),
        "risk_match_score": round(score, 3),
    })

con.close()
res = pd.DataFrame(rows).sort_values("risk_match_score", ascending=False)
print(f"{len(res)} liquid stocks screened.\n")
print("Top 30 matching the high-risk pre-rally profile "
      "(MOST volatile/distressed = top):\n")
print(res.head(30).to_string(index=False))
res.to_csv("current_screen.csv", index=False)
print("\nFull list saved to current_screen.csv")
print("\nThese rank by SIMILARITY to the winner profile = by RISK. A top-ranked")
print("name is the most likely to make a big move in EITHER direction. Nothing")
print("here estimates which way it goes. Treat as a watchlist-of-volatile-names,")
print("not a buy list. Anything you act on, paper-trade first and risk only what")
print("you can lose entirely.")