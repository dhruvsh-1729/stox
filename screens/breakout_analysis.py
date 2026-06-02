"""Analyze the breakout_filter.csv output.

Three things:
  1. Cross-reference with the momentum screen (fresh_tech_screen.csv).
     Names in BOTH lists = established uptrend AND volume picking up = best
     continuation candidates.
  2. Stratify the 88 names by signal quality so we don't treat noisy
     large-cap block-deal blips the same as clean small-cap breakouts.
  3. Forward-return base rate: for each name on the list, scan its full
     history for every prior instance where the SAME filter triggered, then
     measure the 60-trading-day forward return. This is the non-circular
     test of "does this pattern actually predict anything for this name."
"""
import duckdb
import numpy as np
import pandas as pd
from engine import config
from engine import indicators as ta

FWD_DAYS = 60          # ~3 months forward
PRICE_CEIL = 400.0
VOL_MA_DAYS = 20

cur = pd.read_csv("breakout_filter.csv")
print(f"Loaded {len(cur)} names from breakout_filter.csv\n")

# --- 1) overlap with momentum screen ---
try:
    mom = pd.read_csv("fresh_tech_screen.csv")
    overlap = sorted(set(cur["ticker"]) & set(mom["ticker"]))
    print(f"=== Overlap with the momentum screen (192 names) ===")
    print(f"{len(overlap)} names appear in BOTH (uptrend + vol picking up):\n")
    cross = cur[cur["ticker"].isin(overlap)].merge(
        mom[["ticker", "ret_6m_pct"]].rename(columns={"ret_6m_pct": "mom_6m_pct"}),
        on="ticker", how="left"
    ).sort_values("vol_surge_x", ascending=False)
    print(cross.to_string(index=False))
    print()
except FileNotFoundError:
    print("(no fresh_tech_screen.csv; skipping overlap)\n")

# --- 2) stratify by signal quality ---
def categorize(r):
    # noisy event-driven blip: huge vol surge on a name with negligible recent move
    if r["vol_surge_x"] >= 5 and abs(r.get("ret_1m_pct") or 0) < 5 and r["med_turnover_cr"] > 100:
        return "event_blip"
    # overheated: high RSI + already up a lot in last 3m
    if (r.get("rsi14") or 0) >= 75 or (r.get("ret_3m_pct") or 0) > 50:
        return "overheated"
    # clean continuation: in uptrend, modest-to-strong vol confirm, RSI 55-72
    if 55 <= (r.get("rsi14") or 0) <= 72 and r["vol_surge_x"] >= 1.5 \
       and r["pct_above_200dma"] > 5 and r["pct_below_52w_high"] > -25:
        return "clean_continuation"
    # early stage: above 200DMA but only just, modest signals
    if r["pct_above_200dma"] < 10 and 1.0 <= r["vol_surge_x"] < 2.0:
        return "early_stage"
    return "other"

cur["bucket"] = cur.apply(categorize, axis=1)
print("=== Signal-quality buckets ===")
print(cur.groupby("bucket").size().sort_values(ascending=False).to_string())
print()
print("Clean continuation candidates (best technical setups):\n")
clean = cur[cur["bucket"] == "clean_continuation"].sort_values("vol_surge_x", ascending=False)
print(clean[["ticker", "price", "pct_above_200dma", "vol_surge_x", "rsi14",
             "ret_1m_pct", "ret_3m_pct", "pct_below_52w_high", "med_turnover_cr"]]
      .to_string(index=False))
print()

# --- 3) forward-return base rate from history ---
print("=== Forward 60d return base rate (per-name) ===")
print("Scanning each ticker's full history for prior instances where the\n"
      "SAME filter triggered, then measuring the next 60 trading days.\n")

con = duckdb.connect(config.DUCKDB_PATH, read_only=True)
focus = clean["ticker"].tolist()[:20] if len(clean) >= 20 else cur.head(20)["ticker"].tolist()

br_rows = []
for sym in focus:
    df = con.execute(
        "SELECT date, close, volume FROM prices WHERE symbol = ? ORDER BY date", [sym]
    ).fetchdf()
    if len(df) < 260 + FWD_DAYS:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    close = df["close"]
    vol = df["volume"]
    dma200 = ta.sma(close, 200)
    vol_ma = vol.shift(1).rolling(VOL_MA_DAYS).mean()

    # candidates: not the most recent FWD_DAYS bars (we need forward data),
    # and indicator warmup behind us
    valid_idx = close.index[210:-FWD_DAYS]
    triggers = (
        (close.loc[valid_idx] < PRICE_CEIL)
        & (close.loc[valid_idx] > dma200.loc[valid_idx])
        & (vol.loc[valid_idx] > vol_ma.loc[valid_idx])
    )
    trig_dates = valid_idx[triggers]

    # de-duplicate clustered triggers (a single uptrend produces a cluster of
    # consecutive triggers — only count one per 60-day window)
    keep = []
    last = None
    for d in trig_dates:
        if last is None or (d - last).days >= 30:
            keep.append(d)
            last = d
    if not keep:
        continue
    fwd = []
    for d in keep:
        loc = close.index.get_loc(d)
        if loc + FWD_DAYS < len(close):
            fwd.append(close.iloc[loc + FWD_DAYS] / close.iloc[loc] - 1)
    if not fwd:
        continue
    fwd = np.array(fwd)
    br_rows.append({
        "ticker": sym,
        "n_triggers": len(fwd),
        "median_fwd_60d_pct": round(float(np.median(fwd)) * 100, 1),
        "mean_fwd_60d_pct": round(float(np.mean(fwd)) * 100, 1),
        "win_rate_pct": round(float((fwd > 0).mean()) * 100, 1),
        "p25_fwd_pct": round(float(np.percentile(fwd, 25)) * 100, 1),
        "p75_fwd_pct": round(float(np.percentile(fwd, 75)) * 100, 1),
    })
con.close()

base = pd.DataFrame(br_rows).sort_values("median_fwd_60d_pct", ascending=False)
print(f"Per-name historical base rates over {FWD_DAYS} trading days "
      f"(~3 months), {len(base)} names with enough history:\n")
print(base.to_string(index=False))
print()
print("Reading this table:")
print("  win_rate_pct >> 55% AND median > 0 = pattern has historically paid for this name")
print("  win_rate_pct ≈ 50% AND median ≈ 0  = pattern is noise for this name")
print("  p25 vs p75 spread = how dispersed forward returns are (bigger = riskier setup)")
print()
print("This is per-name, not a portfolio test. It tells you whether this technical")
print("signature has predictive value for *this specific stock* historically — not")
print("whether the strategy generalizes across all names (that's a different test).")

base.to_csv("breakout_base_rates.csv", index=False)
print("\nFull base-rate table saved to breakout_base_rates.csv")
