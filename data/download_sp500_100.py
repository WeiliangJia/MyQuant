import yfinance as yf
import pandas as pd
from pathlib import Path

# 读取 100 只股票
tickers = pd.read_csv("sp500_100.csv", header=None)[0].tolist()
import yfinance as yf
import pandas as pd
from pathlib import Path

# 读取 100 只股票
tickers = pd.read_csv("sp500_100.csv", header=None)[0].tolist()

start_date = "2015-01-01"
end_date   = "2026-01-01"

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

for t in tickers:
    print(f"Downloading {t} ...")
    df = yf.download(
        t,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,   # 自动复权，强烈推荐
        progress=False
    )

    if df.empty:
        print(f"⚠️ No data for {t}")
        continue

    df = df.reset_index()
    df["ticker"] = t
    df.to_parquet(out_dir / f"{t}.parquet", index=False)

print("✅ All done")