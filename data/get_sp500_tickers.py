from io import StringIO

import pandas as pd
import requests

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Wikipedia often blocks default Python user agents; set a browser-like UA.
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

tables = pd.read_html(StringIO(response.text))
df = tables[0]
df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
tickers = df["Symbol"].tolist()

print(f"Total tickers: {len(tickers)}")
print(tickers[:10])

# save all & top100 tickers
pd.Series(tickers).to_csv("sp500_all.csv", index=False, header=False)
pd.Series(tickers[:100]).to_csv("sp500_100.csv", index=False, header=False)
