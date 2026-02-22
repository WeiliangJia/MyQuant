import ast
import pandas as pd
from pathlib import Path

files = list(Path("data/raw").glob("*.parquet"))
dfs = []

for f in files:
    df = pd.read_parquet(f)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    else:
        # Handle tuple-like column names serialized as strings: "('Date', '')"
        normalized_cols = []
        for c in df.columns:
            if isinstance(c, str) and c.startswith("(") and c.endswith(")"):
                try:
                    parsed = ast.literal_eval(c)
                except (SyntaxError, ValueError):
                    parsed = c
                if isinstance(parsed, tuple) and parsed:
                    normalized_cols.append(parsed[0])
                else:
                    normalized_cols.append(c)
            else:
                normalized_cols.append(c)
        df.columns = normalized_cols
    # Some parquet files may store Date as index instead of column.
    if "Date" not in df.columns and df.index.name in ("Date", "Datetime", "date", "DATE"):
        df = df.reset_index()
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
        elif "DATE" in df.columns:
            df = df.rename(columns={"DATE": "Date"})
        else:
            raise ValueError(f"Missing Date column in {f.name}. Columns: {list(df.columns)}")
    if "ticker" not in df.columns and "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    dfs.append(df)

panel = pd.concat(dfs, ignore_index=True)
if "Date" not in panel.columns and panel.index.name in ("Date", "Datetime", "date", "DATE"):
    panel = panel.reset_index()

panel = panel.sort_values(["Date", "ticker"])
Path("data/processed").mkdir(parents=True, exist_ok=True)
panel.to_parquet("data/processed/panel.parquet", index=False)

print(panel.head())
