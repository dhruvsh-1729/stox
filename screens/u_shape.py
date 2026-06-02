"""U-shape (drawdown-recovery) pattern screen.

Three stages:

  1. Scan all of cached history (~10y, ~2k symbols) for "potential U-bottom
     events": a peak in the last 30-120 days, current price 15-40% below the
     peak, current price within 5% of the recent 30-day low. This is the
     bottom-of-the-U state irrespective of what happens next.

  2. Label each historical event as RECOVERED (price made it back to >=95% of
     the prior peak within 60 trading days) or FAILED. Compute features at
     the bottom day for every event in both classes.

  3. Compare feature distributions for RECOVERED vs FAILED. Build a simple
     composite score from features whose distributions actually differ.
     Apply to today's U-bottom candidates and rank.

The key point: both classes are sampled the same way (any name at a U-bottom
shape today qualifies); the test asks what historically distinguished the
recoverers from the non-recoverers. That is non-circular by construction.
"""
from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from engine import config
from engine import indicators as ta

# ------------------------------ parameters ------------------------------ #
LOOKBACK_PEAK     = 120        # peak must be within this many trading days
MIN_DAYS_SINCE_PEAK = 30       # peak must be at least this old (so we have a real drawdown)
DD_MIN, DD_MAX    = 0.15, 0.40 # qualifying drawdown band
RECOVERY_THRESH   = 0.95       # recovery = within 5% of peak
RECOVERY_DAYS     = 60         # within 60 trading days
NEAR_LOW_THRESH   = 0.05       # current price within 5% of trailing-30d low
MIN_TURNOVER      = 5_00_00_000  # ₹5 cr liquidity floor
EVENT_DEDUP_DAYS  = 45         # de-cluster historical events
SILVER_LIKE = {"SILVER", "HDFCSILVER", "AXISILVER", "SILVER1",
               "SILVERADD", "SILVERCASE"}


# ---------------------- feature & event extraction ---------------------- #
def make_panels(close: pd.Series, vol: pd.Series):
    """Indicators we'll need at every candidate event."""
    n = len(close)
    arr = close.values
    rsi = ta.rsi(close, 14).values
    dma50 = ta.sma(close, 50).values
    dma200 = ta.sma(close, 200).values
    vol5 = vol.rolling(5).mean().values
    vol60 = vol.rolling(60).mean().values
    turnover60 = (close * vol).rolling(60).median().values
    # rolling 252d high (for "peak was 52w high?")
    roll252_max = close.rolling(252, min_periods=200).max().values
    # 5-day low pattern: rolling min of 5d window — used to detect higher-lows
    roll5_min = close.rolling(5).min().values
    return {
        "arr": arr,
        "rsi": rsi,
        "dma50": dma50,
        "dma200": dma200,
        "vol5": vol5,
        "vol60": vol60,
        "turnover60": turnover60,
        "roll252_max": roll252_max,
        "roll5_min": roll5_min,
    }


def scan_events(close: pd.Series, vol: pd.Series, label_forward: bool):
    """For one symbol, walk every day & emit U-bottom events.

    If label_forward, also compute the recovered/failed label (requires
    RECOVERY_DAYS of future data, so we won't have a label for the most
    recent RECOVERY_DAYS bars).
    """
    n = len(close)
    if n < LOOKBACK_PEAK + 252 + (RECOVERY_DAYS if label_forward else 0):
        return []
    P = make_panels(close, vol)
    arr = P["arr"]
    events = []
    last_event_idx = -10**9

    start = max(LOOKBACK_PEAK + 252, 250)
    stop = n - (RECOVERY_DAYS if label_forward else 0)

    for t in range(start, stop):
        # peak inside [t - LOOKBACK_PEAK, t - MIN_DAYS_SINCE_PEAK)
        lo = t - LOOKBACK_PEAK
        hi = t - MIN_DAYS_SINCE_PEAK
        if hi <= lo:
            continue
        peak_window = arr[lo:hi]
        peak_pos = int(np.argmax(peak_window))
        peak_val = peak_window[peak_pos]
        peak_idx = lo + peak_pos
        if peak_val <= 0 or arr[t] <= 0:
            continue

        # drawdown band
        dd = (peak_val - arr[t]) / peak_val
        if dd < DD_MIN or dd > DD_MAX:
            continue

        # current price within 5% of 30-day low
        recent_low = arr[max(0, t - 30): t + 1].min()
        if arr[t] > recent_low * (1 + NEAR_LOW_THRESH):
            continue

        # liquidity gate
        if not np.isfinite(P["turnover60"][t]) or P["turnover60"][t] < MIN_TURNOVER:
            continue

        # dedup: at least EVENT_DEDUP_DAYS since last event for this symbol
        if t - last_event_idx < EVENT_DEDUP_DAYS:
            continue
        last_event_idx = t

        # ---- features at the bottom ----
        peak_was_52w_high = int(peak_val >= 0.98 * P["roll252_max"][peak_idx]
                                if np.isfinite(P["roll252_max"][peak_idx]) else 0)
        above_200dma_at_peak = (
            1 if (np.isfinite(P["dma200"][peak_idx])
                  and arr[peak_idx] > P["dma200"][peak_idx]) else 0
        )
        below_200dma_now = (
            1 if (np.isfinite(P["dma200"][t]) and arr[t] < P["dma200"][t]) else 0
        )
        dist_from_200dma_pct = float(
            (arr[t] / P["dma200"][t] - 1) * 100
            if np.isfinite(P["dma200"][t]) and P["dma200"][t] > 0 else 0.0
        )
        rsi_now = float(P["rsi"][t]) if np.isfinite(P["rsi"][t]) else 50.0
        rel_vol_5_60 = float(P["vol5"][t] / P["vol60"][t]
                             if np.isfinite(P["vol60"][t]) and P["vol60"][t] > 0 else 1.0)
        # higher-lows micro-pattern: 5d-min today > 5d-min 5 days ago
        higher_low_5d = int(
            np.isfinite(P["roll5_min"][t])
            and np.isfinite(P["roll5_min"][t - 5])
            and P["roll5_min"][t] > P["roll5_min"][t - 5]
        )

        ev = {
            "trough_idx": t,
            "peak_idx": peak_idx,
            "drawdown_pct": float(dd * 100),
            "days_since_peak": int(t - peak_idx),
            "rsi": rsi_now,
            "above_200dma_at_peak": above_200dma_at_peak,
            "below_200dma_now": below_200dma_now,
            "dist_from_200dma_pct": dist_from_200dma_pct,
            "peak_was_52w_high": peak_was_52w_high,
            "rel_vol_5_60": rel_vol_5_60,
            "higher_low_5d": higher_low_5d,
        }
        if label_forward:
            future = arr[t + 1: t + 1 + RECOVERY_DAYS]
            recovered = int((future >= RECOVERY_THRESH * peak_val).any())
            ev["recovered"] = recovered
        events.append(ev)
    return events


# ---------------------- main loop over symbols ---------------------- #
print("Loading cache…")
con = duckdb.connect(config.DUCKDB_PATH, read_only=True)
asof = con.execute("SELECT max(date) FROM prices").fetchone()[0]
symbols = [r[0] for r in con.execute(
    "SELECT DISTINCT symbol FROM prices ORDER BY symbol"
).fetchall()]
print(f"As of {asof}; scanning {len(symbols)} symbols.\n")

historical_events: list[dict] = []
today_candidates: list[dict] = []
per_symbol_data: dict[str, pd.DataFrame] = {}

for sym in symbols:
    if sym.endswith(("-ST", "-SM")) or sym in SILVER_LIKE:
        continue
    df = con.execute(
        "SELECT date, close, volume FROM prices WHERE symbol = ? ORDER BY date",
        [sym],
    ).fetchdf()
    if len(df) < LOOKBACK_PEAK + 252 + RECOVERY_DAYS + 5:
        continue
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    close = df["close"]
    vol = df["volume"]
    per_symbol_data[sym] = df

    # historical events (with forward labels)
    evs = scan_events(close, vol, label_forward=True)
    for e in evs:
        e["ticker"] = sym
        e["trough_date"] = close.index[e["trough_idx"]].date()
        e["peak_date"] = close.index[e["peak_idx"]].date()
        historical_events.append(e)

con.close()
hist = pd.DataFrame(historical_events)
print(f"Found {len(hist):,} historical U-bottom events across "
      f"{hist['ticker'].nunique() if len(hist) else 0} symbols.")
if len(hist) == 0:
    raise SystemExit("No events; can't proceed.")

recovered_rate = hist["recovered"].mean() * 100
print(f"Of those, {hist['recovered'].sum():,} ({recovered_rate:.1f}%) "
      "recovered to ≥95% of peak within 60 trading days.\n")


# ---------------------- recent (last 6 months) Us ---------------------- #
six_months_ago = pd.Timestamp(asof) - pd.Timedelta(days=183)
recent_completed = hist[
    (pd.to_datetime(hist["trough_date"]) >= six_months_ago)
    & (hist["recovered"] == 1)
].copy()
print(f"=== Completed Us with troughs in the last 6 months: "
      f"{len(recent_completed)} events ===")
if len(recent_completed):
    show = recent_completed[["ticker", "peak_date", "trough_date",
                             "drawdown_pct", "days_since_peak",
                             "rsi", "rel_vol_5_60", "peak_was_52w_high"]]
    show = show.sort_values("trough_date", ascending=False).head(20)
    print(show.to_string(index=False))
print()


# ---------------------- feature comparison ---------------------- #
FEATS = ["drawdown_pct", "days_since_peak", "rsi", "above_200dma_at_peak",
         "below_200dma_now", "dist_from_200dma_pct", "peak_was_52w_high",
         "rel_vol_5_60", "higher_low_5d"]

print("=== Feature comparison: RECOVERED vs FAILED bottoms ===\n")
comp = hist.groupby("recovered")[FEATS].median().T
comp.columns = ["failed (median)", "recovered (median)"]
comp["delta"] = comp["recovered (median)"] - comp["failed (median)"]
# also win-rate of "recovered" sliced by a binary threshold for each feature
print(comp.round(2).to_string())
print("\n(Positive delta = recovered side is higher than failed side.)\n")

# Recovery rate as a function of each binary feature
print("Recovery rate conditioned on binary features:")
for f in ["above_200dma_at_peak", "below_200dma_now", "peak_was_52w_high",
          "higher_low_5d"]:
    g = hist.groupby(f)["recovered"].agg(["mean", "size"])
    g["mean"] = (g["mean"] * 100).round(1)
    g.columns = ["recovery_rate_pct", "n"]
    print(f"\n  {f}:")
    print(g.to_string())
print()


# ---------------------- learn a simple linear score ---------------------- #
# Logistic regression on standardized features, no sklearn dep.
print("=== Fitting a simple logistic model to combine features ===")
X_df = hist[FEATS].astype(float)
mu = X_df.mean()
sd = X_df.std().replace(0, 1)
Xz = (X_df - mu) / sd
X = np.c_[np.ones(len(Xz)), Xz.values]
y = hist["recovered"].astype(int).values


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


# tiny ridge-regularized logistic regression by Newton-Raphson
beta = np.zeros(X.shape[1])
lam = 1.0
for _ in range(50):
    p = sigmoid(X @ beta)
    W = p * (1 - p)
    grad = X.T @ (p - y) + lam * np.r_[0, beta[1:]]
    H = X.T @ (X * W[:, None]) + lam * np.eye(X.shape[1])
    H[0, 0] -= lam  # don't regularize intercept
    try:
        step = np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        break
    beta -= step
    if np.linalg.norm(step) < 1e-6:
        break

# in-sample AUC-ish: probability assigned to recovered class
p_hat = sigmoid(X @ beta)
auc_proxy = ((p_hat[y == 1].mean()) - (p_hat[y == 0].mean()))
print(f"Model in-sample lift (mean P[recovered|y=1] − mean P[recovered|y=0]): "
      f"{auc_proxy:.3f}\n")

coef = pd.Series(beta[1:], index=FEATS, name="coef").sort_values(key=abs, ascending=False)
print("Standardized coefficients (positive = pushes toward recovery):")
print(coef.round(3).to_string())
print()


# ---------------------- today's U-bottom candidates ---------------------- #
print("=== Scoring today's U-bottom candidates ===\n")
today_rows = []
for sym, df in per_symbol_data.items():
    close = df["close"]
    vol = df["volume"]
    # scan but only keep the most recent event at index n-1 if it qualifies
    n = len(close)
    if n < LOOKBACK_PEAK + 252 + 5:
        continue
    P = make_panels(close, vol)
    arr = P["arr"]
    t = n - 1
    lo = t - LOOKBACK_PEAK
    hi = t - MIN_DAYS_SINCE_PEAK
    if hi <= lo:
        continue
    peak_window = arr[lo:hi]
    peak_pos = int(np.argmax(peak_window))
    peak_val = peak_window[peak_pos]
    peak_idx = lo + peak_pos
    if peak_val <= 0 or arr[t] <= 0:
        continue
    dd = (peak_val - arr[t]) / peak_val
    if dd < DD_MIN or dd > DD_MAX:
        continue
    recent_low = arr[max(0, t - 30): t + 1].min()
    if arr[t] > recent_low * (1 + NEAR_LOW_THRESH):
        continue
    if not np.isfinite(P["turnover60"][t]) or P["turnover60"][t] < MIN_TURNOVER:
        continue

    peak_was_52w_high = int(peak_val >= 0.98 * P["roll252_max"][peak_idx]
                            if np.isfinite(P["roll252_max"][peak_idx]) else 0)
    above_200dma_at_peak = int(
        np.isfinite(P["dma200"][peak_idx]) and arr[peak_idx] > P["dma200"][peak_idx]
    )
    below_200dma_now = int(
        np.isfinite(P["dma200"][t]) and arr[t] < P["dma200"][t]
    )
    dist_from_200dma_pct = float(
        (arr[t] / P["dma200"][t] - 1) * 100
        if np.isfinite(P["dma200"][t]) and P["dma200"][t] > 0 else 0.0
    )
    rsi_now = float(P["rsi"][t]) if np.isfinite(P["rsi"][t]) else 50.0
    rel_vol_5_60 = float(
        P["vol5"][t] / P["vol60"][t]
        if np.isfinite(P["vol60"][t]) and P["vol60"][t] > 0 else 1.0
    )
    higher_low_5d = int(
        np.isfinite(P["roll5_min"][t])
        and np.isfinite(P["roll5_min"][t - 5])
        and P["roll5_min"][t] > P["roll5_min"][t - 5]
    )

    fvec = pd.Series({
        "drawdown_pct": dd * 100,
        "days_since_peak": t - peak_idx,
        "rsi": rsi_now,
        "above_200dma_at_peak": above_200dma_at_peak,
        "below_200dma_now": below_200dma_now,
        "dist_from_200dma_pct": dist_from_200dma_pct,
        "peak_was_52w_high": peak_was_52w_high,
        "rel_vol_5_60": rel_vol_5_60,
        "higher_low_5d": higher_low_5d,
    })
    z = ((fvec - mu) / sd).values
    p = float(sigmoid(beta[0] + z @ beta[1:]))
    today_rows.append({
        "ticker": sym,
        "price": round(float(arr[t]), 2),
        "peak_date": close.index[peak_idx].date(),
        "peak_price": round(float(peak_val), 2),
        "drawdown_pct": round(float(dd * 100), 1),
        "days_since_peak": int(t - peak_idx),
        "rsi": round(rsi_now, 1),
        "dist_from_200dma_pct": round(dist_from_200dma_pct, 1),
        "peak_was_52w_high": peak_was_52w_high,
        "rel_vol_5_60": round(rel_vol_5_60, 2),
        "higher_low_5d": higher_low_5d,
        "turnover_cr": round(float(P["turnover60"][t]) / 1e7, 1),
        "recover_prob": round(p, 3),
    })

today = pd.DataFrame(today_rows).sort_values("recover_prob", ascending=False)
print(f"{len(today)} symbols sit at a U-bottom shape today "
      f"(peak in last 30-120d, drawdown 15-40%, near recent low).\n")
print("Baseline recovery rate across all historical events: "
      f"{recovered_rate:.1f}%. Anything notably above this is a positive signal.\n")
print("=== Top 25 by modeled recovery probability ===")
print(today.head(25).to_string(index=False))
print("\n=== Bottom 10 (looks like a U-bottom but historical features suggest no) ===")
print(today.tail(10).to_string(index=False))

today.to_csv("u_shape_candidates.csv", index=False)
hist.to_csv("u_shape_history.csv", index=False)
print("\nFiles written: u_shape_candidates.csv, u_shape_history.csv")
