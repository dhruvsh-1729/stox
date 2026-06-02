"""Central configuration: paths, env, constants."""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
DATA_DIR = ROOT / "data"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

DUCKDB_PATH = str(CACHE_DIR / "market.duckdb")
TOKEN_FILE = CACHE_DIR / "access_token.txt"

KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")

# Universe file (csv with column 'Symbol'); falls back to a built-in sample.
NIFTY500_CSV = DATA_DIR / "nifty500.csv"
