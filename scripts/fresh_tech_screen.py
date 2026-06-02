"""Fresh-from-Kite technical screen, no local cache.

Pipeline:
  1. Pull live NSE instrument list, filter to clean mainboard equities.
  2. Concurrently fetch last ~320 days of daily OHLCV per symbol, rate-limited
     to 3 req/s (Kite's historical-data ceiling).
  3. Apply the technical portion of the screen:
       current price > 50 DMA
       50 DMA > 200 DMA
       6-month return > 15%
       ₹5 cr median daily turnover floor
  4. Write the passing list to fresh_tech_screen.csv.

Does NOT write to cache/ — every run pulls fresh data.

Usage:
  python -m engine.kite_data --login         # refresh access token first
  python -m scripts.fresh_tech_screen
  python -m scripts.fresh_tech_screen --top-n 500    # cap universe for speed
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from engine import indicators as ta
from engine.kite_data import get_kite


# ---------- universe filter (same rules as build_universe.py) ---------- #
SUFFIX_JUNK_RE = r"-[A-Z]?\d"
SUFFIX_SERIES_RE = r"-(?:TB|RE|BE|BZ|GB|GS|SG|IV|NV|NA|NB|NC|ND|YW|YY|NR|NX)$"
SME_RE = r"-SM$"
FUND_RE = r"ETF|LIQUID|GOLD|BEES|SDL|GSEC|GILT|TBILL"


def clean_mainboard_equities(kite) -> list[dict]:
    df = pd.DataFrame(kite.instruments("NSE"))
    eq = df[(df["instrument_type"] == "EQ") & (df["segment"] == "NSE")].copy()
    ts = eq["tradingsymbol"].astype(str)
    bad = (
        ts.str.contains(SUFFIX_JUNK_RE, regex=True, na=False)
        | ts.str.contains(SUFFIX_SERIES_RE, regex=True, na=False)
        | ts.str.contains(SME_RE, regex=True, na=False)
        | ts.str.contains(FUND_RE, case=False, na=False)
        | ts.str.match(r"^\d", na=False)
    )
    eq = eq[~bad].copy()
    return eq[["tradingsymbol", "instrument_token"]].to_dict("records")


# ---------- token-bucket rate limiter ---------- #
class RateLimiter:
    """Allow at most `rate` calls per second across all threads."""

    def __init__(self, rate: float):
        self.min_interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
            self.next_ok = max(now, self.next_ok) + self.min_interval


# ---------- historical fetch ---------- #
def fetch_one(kite, sym: str, tok: int, start: dt.date, end: dt.date,
              limiter: RateLimiter, retries: int = 1) -> pd.DataFrame | None:
    for attempt in range(retries + 1):
        limiter.wait()
        try:
            candles = kite.historical_data(tok, start, end, "day")
            if not candles:
                return None
            df = pd.DataFrame(candles)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["symbol"] = sym
            return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                print(f"  ! {sym}: {e}", file=sys.stderr)
                return None
            time.sleep(1)
    return None


def fetch_all(kite, instruments: list[dict], days: int,
              workers: int, rate: float) -> dict[str, pd.DataFrame]:
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    limiter = RateLimiter(rate)
    out: dict[str, pd.DataFrame] = {}
    n = len(instruments)
    done = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(fetch_one, kite, i["tradingsymbol"], i["instrument_token"],
                      start, end, limiter): i["tradingsymbol"]
            for i in instruments
        }
        for fut in as_completed(futs):
            sym = futs[fut]
            df = fut.result()
            done += 1
            if df is not None and not df.empty:
                out[sym] = df
            if done % 50 == 0 or done == n:
                elapsed = time.monotonic() - t0
                rate_actual = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate_actual if rate_actual > 0 else 0
                print(f"  [{done}/{n}] {rate_actual:.1f} req/s, eta {eta:.0f}s")
    return out


# ---------- the screen ---------- #
def screen(prices: dict[str, pd.DataFrame], min_turnover: float) -> pd.DataFrame:
    rows = []
    for sym, df in prices.items():
        if len(df) < 210:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        close = df["close"]
        vol = df["volume"]
        if (close * vol).tail(60).median() < min_turnover:
            continue
        dma50 = ta.sma(close, 50).iloc[-1]
        dma200 = ta.sma(close, 200).iloc[-1]
        price = close.iloc[-1]
        if np.isnan(dma50) or np.isnan(dma200):
            continue
        if len(close) < 126:
            continue
        ret_6m = price / close.iloc[-126] - 1
        if not (price > dma50 > dma200 and ret_6m > 0.15):
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
                float((close * vol).tail(60).median()) / 1e7, 1
            ),
        })
    return pd.DataFrame(rows).sort_values("ret_6m_pct", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=None,
                    help="cap universe to N tickers (alphabetical); default: all clean mainboard")
    ap.add_argument("--days", type=int, default=320,
                    help="calendar days of history to pull per symbol")
    ap.add_argument("--rate", type=float, default=3.0,
                    help="global req/s ceiling")
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent fetch threads")
    ap.add_argument("--min-turnover-cr", type=float, default=5.0,
                    help="₹ cr median daily turnover floor")
    args = ap.parse_args()

    kite = get_kite()
    kite.profile()  # fast credential check

    print("Pulling NSE instrument list…")
    inst = clean_mainboard_equities(kite)
    if args.top_n:
        inst = sorted(inst, key=lambda x: x["tradingsymbol"])[:args.top_n]
    print(f"{len(inst)} mainboard equities to fetch "
          f"(~{len(inst)/args.rate:.0f}s at {args.rate:.0f} req/s)\n")

    t0 = time.monotonic()
    prices = fetch_all(kite, inst, days=args.days,
                       workers=args.workers, rate=args.rate)
    elapsed = time.monotonic() - t0
    print(f"\nFetched {len(prices)}/{len(inst)} symbols in {elapsed:.0f}s\n")

    res = screen(prices, min_turnover=args.min_turnover_cr * 1e7)
    print(f"{len(res)} names pass the technical filter.\n")
    print("Top 30 by 6-month return:\n")
    print(res.head(30).to_string(index=False))
    res.to_csv("fresh_tech_screen.csv", index=False)
    print("\nFull list saved to fresh_tech_screen.csv")


if __name__ == "__main__":
    main()
