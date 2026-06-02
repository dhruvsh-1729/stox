"""High-conviction multi-factor technical screen with edge-over-benchmark.

Builds a 10-of-10 composite over Kite OHLCV. For every qualifying name today,
scans its full history for prior instances of the same signature and measures
forward 60-trading-day returns. Compares each name's signal base rate against
the unconditional benchmark (random liquid-Indian-stock 60d return over the
same dates), so the bull-market tailwind is subtracted out.

The 10 conditions (all must be true today):
  1.  50-DMA > 200-DMA                       (long-term trend up)
  2.  price > 50-DMA                         (above near-term trend)
  3.  6-month return > 0                     (positive medium-term momentum)
  4.  12-month return > 0                    (positive long-term momentum)
  5.  20d avg volume > 60d avg volume * 0.9  (interest not fading)
  6.  RSI(14) between 45 and 72              (room to run, not overbought)
  7.  price <= 130% of 200-DMA               (not parabolic)
  8.  price within -20% of 52w high          (not in deep correction)
  9.  60d median turnover >= ₹5 cr           (tradeable)
  10. annualized 60d realized vol < 60%      (not a casino stock)

These are NOT fundamentals. Mcap/PE/ROE/D/E/CFO are still not in Kite.
"""
import duckdb
import numpy as np
import pandas as pd
from engine import config
from engine import indicators as ta

FWD_DAYS = 60                   # ~3-month forward window
MIN_HIST_TRIGGERS = 8           # require enough prior instances for a meaningful base rate
MIN_TURNOVER = 5_00_00_000      # ₹5 cr/day liquidity floor

print("Loading cache…")
con = duckdb.connect(config.DUCKDB_PATH, read_only=True)
asof = con.execute("SELECT max(date) FROM prices").fetchone()[0]
symbols = [r[0] for r in con.execute(
    "SELECT DISTINCT symbol FROM prices ORDER BY symbol"
).fetchall()]
SILVER_LIKE = ("SILVER", "HDFCSILVER", "AXISILVER", "SILVER1",
               "SILVERADD", "SILVERCASE")


def per_symbol(df: pd.DataFrame) -> dict | None:
    """Build all panel indicators once per symbol; return None if too short."""
    if len(df) < 260 + FWD_DAYS:
        return None
    close = df["close"]
    vol = df["volume"]
    dma50 = ta.sma(close, 50)
    dma200 = ta.sma(close, 200)
    vol20 = vol.shift(1).rolling(20).mean()
    vol60 = vol.shift(1).rolling(60).mean()
    rsi = ta.rsi(close, 14)
    ret_6m = close / close.shift(126) - 1
    ret_12m = close / close.shift(252) - 1
    high_52w = close.shift(1).rolling(252).max()
    turnover60 = (close * vol).shift(1).rolling(60).median()
    rvol60 = close.pct_change().rolling(60).std() * np.sqrt(252)

    # composite mask: all 10 conditions
    mask = (
        (dma50 > dma200)
        & (close > dma50)
        & (ret_6m > 0)
        & (ret_12m > 0)
        & (vol20 > vol60 * 0.9)
        & (rsi.between(45, 72))
        & (close <= 1.30 * dma200)
        & (close >= 0.80 * high_52w)
        & (turnover60 >= MIN_TURNOVER)
        & (rvol60 < 0.60)
    )
    return {
        "close": close,
        "rsi": rsi,
        "dma50": dma50,
        "dma200": dma200,
        "ret_6m": ret_6m,
        "ret_12m": ret_12m,
        "vol20": vol20,
        "vol60": vol60,
        "turnover60": turnover60,
        "rvol60": rvol60,
        "high_52w": high_52w,
        "mask": mask,
    }


# ---- pass 1: score everyone, build per-name panels, find current qualifiers ----
print("Building panels & scoring composite for each symbol…")
panels: dict[str, dict] = {}
current_pass = []
for sym in symbols:
    if sym.endswith("-ST") or sym.endswith("-SM") or sym in SILVER_LIKE:
        continue
    df = con.execute(
        "SELECT date, close, volume FROM prices WHERE symbol = ? ORDER BY date",
        [sym],
    ).fetchdf()
    if len(df) < 260 + FWD_DAYS:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    p = per_symbol(df)
    if p is None:
        continue
    panels[sym] = p
    if bool(p["mask"].iloc[-1]):
        current_pass.append(sym)

print(f"\n{len(panels)} symbols panelized, {len(current_pass)} pass the 10/10 today.\n")


# ---- pass 2: build the unconditional benchmark ----
# For every (symbol, date) where the name was LIQUID enough, what was the 60-day
# forward return? This is the random-buy-and-hold reference distribution.
print("Building unconditional benchmark distribution…")
bench_returns = []
for sym, p in panels.items():
    close = p["close"]
    liq = p["turnover60"] >= MIN_TURNOVER
    idx = close.index[210:-FWD_DAYS]
    valid = liq.loc[idx] & close.loc[idx].notna()
    valid_idx = idx[valid]
    if len(valid_idx) == 0:
        continue
    # subsample to keep memory bounded
    sample = valid_idx if len(valid_idx) <= 200 else valid_idx[::max(len(valid_idx)//200, 1)]
    for d in sample:
        loc = close.index.get_loc(d)
        if loc + FWD_DAYS < len(close):
            bench_returns.append(close.iloc[loc + FWD_DAYS] / close.iloc[loc] - 1)
bench = np.array(bench_returns)
b_med = float(np.median(bench)) * 100
b_mean = float(np.mean(bench)) * 100
b_win = float((bench > 0).mean()) * 100
print(f"Unconditional 60d benchmark: median {b_med:+.1f}%, mean {b_mean:+.1f}%, "
      f"win rate {b_win:.1f}% (n={len(bench):,})\n")


# ---- pass 3: per-name historical base rate vs benchmark ----
print("Computing per-name historical base rate for today's qualifiers…")
rows = []
for sym in current_pass:
    p = panels[sym]
    close = p["close"]
    mask = p["mask"]
    # all prior trigger dates (excluding the forward window)
    valid_idx = close.index[210:-FWD_DAYS]
    trig_dates = valid_idx[mask.loc[valid_idx]]
    # de-duplicate clustered triggers — 30-day min spacing
    keep, last = [], None
    for d in trig_dates:
        if last is None or (d - last).days >= 30:
            keep.append(d)
            last = d
    if len(keep) < MIN_HIST_TRIGGERS:
        continue
    fwd = []
    for d in keep:
        loc = close.index.get_loc(d)
        if loc + FWD_DAYS < len(close):
            fwd.append(close.iloc[loc + FWD_DAYS] / close.iloc[loc] - 1)
    if len(fwd) < MIN_HIST_TRIGGERS:
        continue
    fwd = np.array(fwd)
    win = float((fwd > 0).mean()) * 100
    med = float(np.median(fwd)) * 100
    mean = float(np.mean(fwd)) * 100
    p25 = float(np.percentile(fwd, 25)) * 100
    p75 = float(np.percentile(fwd, 75)) * 100
    edge_median = med - b_med
    edge_win = win - b_win

    price = float(close.iloc[-1])
    rows.append({
        "ticker": sym,
        "price": round(price, 2),
        "pct_above_200dma": round((price / p["dma200"].iloc[-1] - 1) * 100, 1),
        "rsi14": round(float(p["rsi"].iloc[-1]), 1),
        "ret_6m_pct": round(float(p["ret_6m"].iloc[-1]) * 100, 1),
        "drawdown_pct": round((price / p["high_52w"].iloc[-1] - 1) * 100, 1),
        "rvol60_pct": round(float(p["rvol60"].iloc[-1]) * 100, 1),
        "turnover_cr": round(float(p["turnover60"].iloc[-1]) / 1e7, 1),
        "n_triggers": len(fwd),
        "win_rate_pct": round(win, 1),
        "median_fwd_pct": round(med, 1),
        "mean_fwd_pct": round(mean, 1),
        "p25_fwd_pct": round(p25, 1),
        "p75_fwd_pct": round(p75, 1),
        "edge_win_pp": round(edge_win, 1),       # vs benchmark, in percentage points
        "edge_median_pp": round(edge_median, 1),
    })
con.close()


if not rows:
    print("\nNo names cleared all 10 criteria AND have enough prior history.")
    raise SystemExit(0)

df = pd.DataFrame(rows)
# composite ranking: edge in win rate + edge in median return (both vs benchmark)
df["score"] = df["edge_win_pp"] + df["edge_median_pp"]
df = df.sort_values("score", ascending=False).reset_index(drop=True)

print(f"\n{len(df)} names pass the 10/10 today AND have ≥{MIN_HIST_TRIGGERS} "
      "historical triggers for a meaningful base rate.\n")

# Show the top picks with the cleanest evidence.
show_cols = [
    "ticker", "price", "pct_above_200dma", "drawdown_pct", "rsi14",
    "ret_6m_pct", "rvol60_pct", "turnover_cr",
    "n_triggers", "win_rate_pct", "median_fwd_pct", "p25_fwd_pct", "p75_fwd_pct",
    "edge_win_pp", "edge_median_pp", "score",
]
print("=== TOP 15 — highest evidence-of-edge over the unconditional benchmark ===\n")
print(df.head(15)[show_cols].to_string(index=False))
print()

# Bottom of the list — names that pass the technical screen but have HISTORICALLY
# underperformed the benchmark. Useful to flag what NOT to buy.
print("=== Bottom 10 — pass the screen but historically WORSE than benchmark ===\n")
print(df.tail(10)[show_cols].to_string(index=False))

df.to_csv("conviction.csv", index=False)
print(f"\nFull ranked table ({len(df)} names) saved to conviction.csv")
print(f"\nBenchmark reference: unconditional 60d median +{b_med:.1f}%, "
      f"win rate {b_win:.1f}%. 'edge_win_pp' / 'edge_median_pp' are this signal "
      f"vs that benchmark.")
