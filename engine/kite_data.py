"""Kite Connect data layer.

Responsibilities:
  * Daily OAuth login (request_token -> access_token), cached to disk.
  * Fetch daily historical OHLCV per instrument.
  * Cache everything into DuckDB so screens run fully offline.

Kite's access token expires every morning (~6am IST). Backfill once, then read
from DuckDB — screens never touch the network.
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
import time

import duckdb
import pandas as pd

from . import config
from .universe import load_universe


def _con():
    con = duckdb.connect(config.DUCKDB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            symbol   VARCHAR,
            date     DATE,
            open     DOUBLE,
            high     DOUBLE,
            low      DOUBLE,
            close    DOUBLE,
            volume   BIGINT,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    return con


def write_prices(df: pd.DataFrame):
    if df.empty:
        return
    con = _con()
    con.register("incoming", df)
    con.execute("INSERT OR REPLACE INTO prices SELECT * FROM incoming")
    con.unregister("incoming")
    con.close()


def read_prices(symbol: str) -> pd.DataFrame:
    con = _con()
    df = con.execute(
        "SELECT * FROM prices WHERE symbol = ? ORDER BY date", [symbol]
    ).fetchdf()
    con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def cached_symbols() -> list[str]:
    con = _con()
    rows = con.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchall()
    con.close()
    return [r[0] for r in rows]


def get_kite():
    from kiteconnect import KiteConnect

    if not config.KITE_API_KEY:
        raise RuntimeError("KITE_API_KEY not set. Fill in .env")
    kite = KiteConnect(api_key=config.KITE_API_KEY)
    if config.TOKEN_FILE.exists():
        kite.set_access_token(config.TOKEN_FILE.read_text().strip())
    return kite


def login_flow():
    """Interactive daily login. Prints the login URL, you paste request_token."""
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=config.KITE_API_KEY)
    print("\n1. Open this URL, log in, and copy the request_token from the redirect URL:\n")
    print("   " + kite.login_url() + "\n")
    request_token = input("2. Paste request_token here: ").strip()
    data = kite.generate_session(request_token, api_secret=config.KITE_API_SECRET)
    token = data["access_token"]
    config.TOKEN_FILE.write_text(token)
    print(f"\nAccess token saved to {config.TOKEN_FILE}")
    print("  (valid until ~6am IST tomorrow; re-run --login then)")


def _instrument_token_map(kite) -> dict[str, int]:
    """Map tradingsymbol -> instrument_token for NSE equities."""
    instruments = kite.instruments("NSE")
    return {
        i["tradingsymbol"]: i["instrument_token"]
        for i in instruments
        if i["instrument_type"] == "EQ"
    }


def backfill(years: int = 10, throttle: float = 0.34):
    """Fetch daily OHLCV for the full universe into DuckDB.

    Kite historical limits ~3 req/s; throttle keeps us safe. Daily candles
    allow a 2000-day window per call, so we chunk by date range.
    """
    kite = get_kite()
    token_map = _instrument_token_map(kite)
    symbols = load_universe()
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)

    have = set(cached_symbols())
    for i, sym in enumerate(symbols, 1):
        if sym in have:
            print(f"[{i}/{len(symbols)}] {sym} cached, skipping")
            continue
        tok = token_map.get(sym)
        if tok is None:
            print(f"[{i}/{len(symbols)}] {sym} not found on NSE EQ, skipping")
            continue
        try:
            frames = []
            cur = start
            while cur < end:
                chunk_end = min(cur + dt.timedelta(days=1999), end)
                candles = kite.historical_data(tok, cur, chunk_end, "day")
                if candles:
                    frames.append(pd.DataFrame(candles))
                cur = chunk_end + dt.timedelta(days=1)
                time.sleep(throttle)
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["symbol"] = sym
            df = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
            write_prices(df)
            print(f"[{i}/{len(symbols)}] {sym}: {len(df)} rows")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(symbols)}] {sym} ERROR: {e}", file=sys.stderr)
            time.sleep(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()
    if args.login:
        login_flow()
    elif args.backfill:
        backfill(years=args.years)
    else:
        ap.print_help()
