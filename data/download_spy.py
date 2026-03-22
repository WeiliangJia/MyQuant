"""下载 SPY 基准数据，与个股时间范围对齐。"""
import yfinance as yf
import pandas as pd
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "data" / "raw"
out_dir.mkdir(parents=True, exist_ok=True)

df = yf.download("SPY", start="2015-01-01", end="2026-01-01",
                 interval="1d", auto_adjust=True, progress=False)
df = df.reset_index()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

df["ticker"] = "SPY"
df.to_parquet(out_dir / "SPY.parquet", index=False)
print(f"SPY saved: {len(df)} rows, {df['Date'].min()} ~ {df['Date'].max()}")
