# stox — Zerodha Kite screening scripts

Small set of Python scripts that pull NSE OHLCV via Zerodha Kite Connect,
cache it locally in DuckDB, and run simple screens that shortlist names for
follow-up research elsewhere.

Not a backtesting engine, not a dashboard — just data + screens.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add KITE_API_KEY and KITE_API_SECRET
```

You need Zerodha Kite Connect + the Historical Data add-on (~₹2,000/mo) —
sign up at https://kite.trade.

## Daily flow

```bash
# 1. Refresh access token (Kite tokens expire ~6am IST daily)
python -m engine.kite_data --login

# 2. Pull / top-up the price cache
python -m engine.kite_data --backfill
```

All screens read from `cache/market.duckdb` and need no live token.

## Building the universe

```bash
python build_universe.py        # full clean NSE-EQ list -> data/nifty500.csv
python make_liquid_universe.py  # top 500 most-liquid names (needs cache)
```

## Screens

Each script writes a CSV you can hand off to other research tools.

| Script | What it does | Output |
|---|---|---|
| `rallies.py`  | Biggest 2-month rallies among liquid mainboard names | `top_rallies_clean.csv` |
| `current.py`  | Stocks whose current profile matches the pre-rally winner profile (volatility/discount screen) | `current_screen.csv` |
| `analysis.py` | Compares pre-rally indicators of past winners vs non-winners (diagnostic, not predictive) | `pre_rally_analysis.csv` |

Run them like:

```bash
python rallies.py
python current.py
python analysis.py
```

## Layout

```
stox/
  engine/
    config.py       # paths, env loading
    kite_data.py    # Kite login + historical fetch + DuckDB cache
    universe.py     # universe loading
    indicators.py   # RSI / SMA / EMA / ATR / Bollinger / MACD / zscore
  build_universe.py
  make_liquid_universe.py
  rallies.py
  current.py
  analysis.py
  data/nifty500.csv
  requirements.txt
```

These screens rank by past volatility / drawdown profile. They surface
candidates worth investigating — they are not buy signals. Research further
before risking capital.
