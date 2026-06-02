"""Build a clean equity universe from the Kite NSE instrument dump."""
import pandas as pd
from engine.kite_data import get_kite

kite = get_kite()
df = pd.DataFrame(kite.instruments("NSE"))
eq = df[(df["instrument_type"] == "EQ") & (df["segment"] == "NSE")].copy()
ts = eq["tradingsymbol"].astype(str)

suffix_junk = ts.str.contains(r"-[A-Z]?\d", regex=True, na=False) \
    | ts.str.contains(r"-(?:TB|RE|BE|BZ|GB|GS|SG|IV|NV|NA|NB|NC|ND|YW|YY|NR|NX)$",
                      regex=True, na=False)
sme = ts.str.contains(r"-SM$", regex=True, na=False)
funds = ts.str.contains("ETF|LIQUID|GOLD|BEES|SDL|GSEC|GILT|TBILL", case=False, na=False)
numeric = ts.str.match(r"^\d", na=False)

bad = suffix_junk | sme | funds | numeric
syms = sorted(ts[~bad].unique())
pd.DataFrame({"Symbol": syms}).to_csv("data/nifty500.csv", index=False)
print(f"{len(syms)} mainboard equity symbols written")
print("sample:", syms[:10])