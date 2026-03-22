"""
第一轮因子工程 — 6 个经典因子
=============================
基于 panel_tradable.parquet 计算以下因子:
  1. momentum_20d    — 20日动量（过去20日收益率）
  2. volatility_20d  — 20日收益波动率
  3. volume_ratio_20d— 量比（当日成交量 / 20日均量）
  4. rsi_14          — 14日RSI
  5. ma_deviation_20d— 均线偏离度（Close / MA20 - 1）
  6. return_1d       — 1日反转（已有，直接使用）

所有因子最后做截面 z-score 标准化。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "data" / "processed"
INPUT_PATH = PROCESSED_DIR / "panel_tradable.parquet"
OUTPUT_PATH = SCRIPT_DIR / "factors_v1.parquet"
REPORT_PATH = SCRIPT_DIR / "factors_v1_report.json"

# ── 因子参数 ──────────────────────────────────────────
MOMENTUM_WINDOW = 20
VOLATILITY_WINDOW = 20
VOLUME_MA_WINDOW = 20
RSI_WINDOW = 14
MA_WINDOW = 20

RAW_FACTOR_COLS = [
    "return_1d",
    "momentum_20d",
    "volatility_20d",
    "volume_ratio_20d",
    "rsi_14",
    "ma_deviation_20d",
]


# ── 因子计算（每只股票时序） ──────────────────────────
def _compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    close = df["Close"]
    volume = df["Volume"]

    # 1. return_1d: 已有，直接用
    # （保留原始列，后续统一处理）

    # 2. momentum_20d: 过去20日累计收益
    df["momentum_20d"] = close.pct_change(MOMENTUM_WINDOW, fill_method=None)

    # 3. volatility_20d: 过去20日日收益率的标准差
    daily_ret = close.pct_change(fill_method=None)
    df["volatility_20d"] = daily_ret.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()

    # 4. volume_ratio_20d: 当日成交量 / 20日均量
    vol_ma = volume.rolling(VOLUME_MA_WINDOW, min_periods=VOLUME_MA_WINDOW).mean()
    df["volume_ratio_20d"] = volume / vol_ma.replace(0, np.nan)

    # 5. rsi_14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=RSI_WINDOW, min_periods=RSI_WINDOW, adjust=False).mean()
    avg_loss = loss.ewm(span=RSI_WINDOW, min_periods=RSI_WINDOW, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 6. ma_deviation_20d: (Close / MA20) - 1
    ma20 = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    df["ma_deviation_20d"] = close / ma20 - 1

    return df


# ── 截面 z-score 标准化 ──────────────────────────────
def _cross_sectional_zscore(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        raw_col = col
        z_col = f"{col}_zscore"
        grouped = panel.groupby("Date")[raw_col]
        panel[z_col] = (panel[raw_col] - grouped.transform("median")) / grouped.transform("std")
        # clip 极端值到 [-3, 3]
        panel[z_col] = panel[z_col].clip(-3, 3)
    return panel


# ── 主流程 ────────────────────────────────────────────
def main() -> None:
    print(f"Reading {INPUT_PATH} ...")
    panel = pd.read_parquet(INPUT_PATH)
    panel = panel[["Date", "ticker", "Open", "High", "Low", "Close", "Volume", "ret_1d"]].copy()
    panel = panel.rename(columns={"ret_1d": "return_1d"})

    print("Computing factors per ticker ...")
    parts = []
    for ticker, grp in panel.groupby("ticker"):
        computed = _compute_factors(grp)
        computed["ticker"] = ticker
        parts.append(computed)
    panel = pd.concat(parts, ignore_index=True)

    # 删掉窗口期内 NaN 行（前20个交易日算不出因子）
    before_drop = len(panel)
    panel = panel.dropna(subset=RAW_FACTOR_COLS).reset_index(drop=True)
    after_drop = len(panel)

    print("Cross-sectional z-score ...")
    panel = _cross_sectional_zscore(panel, RAW_FACTOR_COLS)

    ZSCORE_COLS = [f"{c}_zscore" for c in RAW_FACTOR_COLS]

    # ── 输出 ──
    out_cols = ["Date", "ticker", "Close"] + RAW_FACTOR_COLS + ZSCORE_COLS
    result = panel[out_cols].sort_values(["Date", "ticker"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}  ({result.shape[0]} rows × {result.shape[1]} cols)")

    # ── 报告 ──
    report = {
        "input_rows": int(before_drop + (before_drop - before_drop)),
        "rows_before_dropna": int(before_drop),
        "rows_after_dropna": int(after_drop),
        "dropped_warmup_rows": int(before_drop - after_drop),
        "output_rows": int(len(result)),
        "tickers": int(result["ticker"].nunique()),
        "date_range": [str(result["Date"].min()), str(result["Date"].max())],
        "raw_factors": RAW_FACTOR_COLS,
        "zscore_factors": ZSCORE_COLS,
        "factor_stats": {},
    }
    for col in RAW_FACTOR_COLS:
        s = result[col]
        report["factor_stats"][col] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "nan_rate": round(float(s.isna().mean()), 6),
        }
    # 因子相关性矩阵
    corr = result[RAW_FACTOR_COLS].corr()
    report["correlation_matrix"] = {
        c: {r: round(float(corr.loc[c, r]), 4) for r in RAW_FACTOR_COLS}
        for c in RAW_FACTOR_COLS
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")

    # 打印摘要
    print("\n── Factor Stats ──")
    print(result[RAW_FACTOR_COLS].describe().round(4).to_string())
    print("\n── Correlation Matrix ──")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
