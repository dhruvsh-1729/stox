"""Universe loading. Reads data/nifty500.csv if present, else a built-in list.

To get the real list: download the official constituents CSV from NSE
(niftyindices.com -> Nifty 500 -> Download) and save as data/nifty500.csv with
a 'Symbol' column. The fallback below is a representative subset so the engine
runs out of the box.
"""
from __future__ import annotations
import pandas as pd
from . import config

_FALLBACK = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "ONGC", "NTPC", "POWERGRID", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "COALINDIA", "GRASIM", "HINDALCO", "DRREDDY", "CIPLA", "DIVISLAB",
    "BRITANNIA", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "M&M", "INDUSINDBK",
    "TECHM", "BPCL", "IOC", "GAIL", "DABUR", "GODREJCP", "PIDILITIND", "DLF",
    "SBILIFE", "HDFCLIFE", "ICICIPRULI", "BAJAJFINSV", "SHREECEM", "AMBUJACEM",
    "VEDL", "TATACONSUM", "BERGEPAINT", "MARICO", "COLPAL", "MUTHOOTFIN",
    "PEL", "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM", "MOTHERSON", "BOSCHLTD",
    "HAVELLS", "SIEMENS", "ABB", "PAGEIND", "PGHH", "UBL", "MCDOWELL-N", "TRENT",
    "NAUKRI", "PERSISTENT", "LTIM", "MPHASIS", "COFORGE", "INDIGO", "JUBLFOOD",
    "PIIND", "SRF", "DEEPAKNTR", "AARTIIND", "BALKRISIND", "APOLLOTYRE", "ESCORTS",
    "TVSMOTOR", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB", "BANKBARODA",
    "CANBK", "CHOLAFIN", "RECLTD", "PFC", "IRCTC", "CONCOR",
]


def load_universe() -> list[str]:
    if config.NIFTY500_CSV.exists():
        df = pd.read_csv(config.NIFTY500_CSV)
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        return df[col].astype(str).str.strip().str.upper().tolist()
    return _FALLBACK
