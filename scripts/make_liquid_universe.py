import duckdb, pandas as pd
from engine import config

con = duckdb.connect(config.DUCKDB_PATH)
# turnover = close * volume; use the last ~250 trading days, take the median
q = """
WITH recent AS (
  SELECT symbol, close*volume AS turnover,
         ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) rn
  FROM prices
)
SELECT symbol, median(turnover) AS med_turnover, count(*) AS days
FROM recent WHERE rn <= 250
GROUP BY symbol
HAVING days >= 200
ORDER BY med_turnover DESC
LIMIT 500
"""
top = con.execute(q).fetchdf()
con.close()
top[['symbol']].rename(columns={'symbol':'Symbol'}).to_csv('data/nifty500.csv', index=False)
print(f"Wrote {len(top)} most-liquid names. Top 5:")
print(top.head(5).to_string(index=False))