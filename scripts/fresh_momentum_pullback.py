"""Fresh-from-Kite momentum/pullback screen, no local cache.

Conditions:
  * Close > DMA 20
  * Close > DMA 50
  * DMA 20 > DMA 50
  * DMA 20 < DMA 50 * 1.05
  * Volume(today) > Volume_1mo_avg * 1.5
  * 50 < RSI(14) < 65

The original query also has `Price to Earning < Industry PE`. Kite doesn't
expose PE / industry PE, so that leg is NOT applied — final list is the
technical subset.

Run:
  python -m engine.kite_data --login            # refresh access token first
  python -m scripts.fresh_momentum_pullback
  python -m scripts.fresh_momentum_pullback --top-n 500   # cap for speed
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


class RateLimiter:
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


def fetch_one(kite, sym, tok, start, end, limiter, retries=1):
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


def fetch_all(kite, instruments, days, workers, rate):
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
            if done % 100 == 0 or done == n:
                elapsed = time.monotonic() - t0
                r_actual = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / r_actual if r_actual > 0 else 0
                print(f"  [{done}/{n}] {r_actual:.1f} req/s, eta {eta:.0f}s")
    return out


def screen(prices, min_turnover):
    rows = []
    for sym, df in prices.items():
        if len(df) < 60:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        close = df["close"]
        vol = df["volume"]

        if (close * vol).tail(60).median() < min_turnover:
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
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=None,
                    help="cap universe to N tickers (alphabetical)")
    ap.add_argument("--days", type=int, default=120,
                    help="calendar days of history to pull per symbol")
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-turnover-cr", type=float, default=5.0)
    args = ap.parse_args()

    kite = get_kite()
    kite.profile()

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

    asof = max(df["date"].max() for df in prices.values())
    print(f"Latest candle date across universe: {asof}\n")

    res = screen(prices, min_turnover=args.min_turnover_cr * 1e7)
    if res.empty:
        print("No names pass the technical filter today.")
        return
    res = res.sort_values("vol_x_avg", ascending=False)
    print(f"{len(res)} names pass the technical filter "
          f"(PE < Industry PE not applied — fundamentals not in Kite).\n")
    print(res.to_string(index=False))
    out = "fresh_momentum_pullback.csv"
    res.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
