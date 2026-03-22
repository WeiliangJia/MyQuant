"""
第二轮因子工程 — 25 个因子
==========================
在 V1（6个）基础上扩展到 25 个，覆盖 6 大类：

动量/反转 (6):  return_1d, momentum_5d, momentum_20d, momentum_60d,
                return_5d_reversal, high_52w_dist
波动率 (4):     volatility_20d, volatility_60d, vol_ratio_20_60,
                intraday_range
量价 (4):       volume_ratio_20d, volume_ratio_5d, volume_std_20d,
                price_volume_corr_20d
技术指标 (5):   rsi_14, ma_deviation_20d, macd_signal, bollinger_pos,
                atr_14_pct
微观结构 (3):   close_position, gap_open, amihud_illiq_20d
高阶统计 (3):   return_skew_20d, return_kurt_20d, downside_vol_20d
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
OUTPUT_PATH = SCRIPT_DIR / "factors_v2.parquet"
REPORT_PATH = SCRIPT_DIR / "factors_v2_report.json"

# ── 全部 25 个因子名 ─────────────────────────────────
RAW_FACTOR_COLS = [
    # 动量/反转
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "return_5d_reversal",
    "high_52w_dist",
    # 波动率
    "volatility_20d",
    "volatility_60d",
    "vol_ratio_20_60",
    "intraday_range",
    # 量价
    "volume_ratio_20d",
    "volume_ratio_5d",
    "volume_std_20d",
    "price_volume_corr_20d",
    # 技术指标
    "rsi_14",
    "ma_deviation_20d",
    "macd_signal",
    "bollinger_pos",
    "atr_14_pct",
    # 微观结构
    "close_position",
    "gap_open",
    "amihud_illiq_20d",
    # 高阶统计
    "return_skew_20d",
    "return_kurt_20d",
    "downside_vol_20d",
]


# ── 因子计算 ─────────────────────────────────────────
def _compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]
    volume = df["Volume"]
    daily_ret = close.pct_change(fill_method=None)

    # ═══ 动量/反转 (6) ═══
    df["return_1d"] = daily_ret
    df["momentum_5d"] = close.pct_change(5, fill_method=None)
    df["momentum_20d"] = close.pct_change(20, fill_method=None)
    df["momentum_60d"] = close.pct_change(60, fill_method=None)
    # 5日反转：过去5日收益取反（短期均值回归信号）
    df["return_5d_reversal"] = -close.pct_change(5, fill_method=None)
    # 52周新高距离：当前价 / 过去252日最高价 - 1
    rolling_high_252 = high.rolling(252, min_periods=252).max()
    df["high_52w_dist"] = close / rolling_high_252 - 1

    # ═══ 波动率 (4) ═══
    df["volatility_20d"] = daily_ret.rolling(20, min_periods=20).std()
    df["volatility_60d"] = daily_ret.rolling(60, min_periods=60).std()
    # 波动率变化率：短期波动/长期波动
    df["vol_ratio_20_60"] = df["volatility_20d"] / df["volatility_60d"].replace(0, np.nan)
    # 日内振幅：(High - Low) / Close 的20日均值
    df["intraday_range"] = ((high - low) / close).rolling(20, min_periods=20).mean()

    # ═══ 量价 (4) ═══
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    vol_ma5 = volume.rolling(5, min_periods=5).mean()
    df["volume_ratio_20d"] = volume / vol_ma20.replace(0, np.nan)
    df["volume_ratio_5d"] = volume / vol_ma5.replace(0, np.nan)
    # 成交量波动率
    df["volume_std_20d"] = volume.rolling(20, min_periods=20).std() / vol_ma20.replace(0, np.nan)
    # 量价相关性：过去20日 收益率 vs 成交量 的相关系数
    df["price_volume_corr_20d"] = daily_ret.rolling(20, min_periods=20).corr(volume)

    # ═══ 技术指标 (5) ═══
    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 均线偏离度
    ma20 = close.rolling(20, min_periods=20).mean()
    df["ma_deviation_20d"] = close / ma20 - 1

    # MACD signal: (EMA12 - EMA26) / Close
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    df["macd_signal"] = (macd_line - macd_signal_line) / close

    # 布林带位置：(Close - MA20) / (2 * std20)，归一化到 [-1, 1] 附近
    std20 = close.rolling(20, min_periods=20).std()
    df["bollinger_pos"] = (close - ma20) / (2 * std20.replace(0, np.nan))

    # ATR 占比：ATR14 / Close（标准化后的波动幅度）
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, min_periods=14, adjust=False).mean()
    df["atr_14_pct"] = atr14 / close

    # ═══ 微观结构 (3) ═══
    # 收盘价在日内范围的位置：(Close - Low) / (High - Low)
    hl_range = (high - low).replace(0, np.nan)
    df["close_position"] = (close - low) / hl_range

    # 跳空缺口：(Open - 前日Close) / 前日Close
    df["gap_open"] = (opn - close.shift(1)) / close.shift(1)

    # Amihud 非流动性：|return| / dollar_volume 的20日均值
    dollar_vol = close * volume
    amihud_daily = daily_ret.abs() / dollar_vol.replace(0, np.nan)
    df["amihud_illiq_20d"] = amihud_daily.rolling(20, min_periods=20).mean()

    # ═══ 高阶统计 (3) ═══
    # 收益偏度
    df["return_skew_20d"] = daily_ret.rolling(20, min_periods=20).skew()
    # 收益峰度
    df["return_kurt_20d"] = daily_ret.rolling(20, min_periods=20).kurt()
    # 下行波动率：只看负收益的标准差
    neg_ret = daily_ret.copy()
    neg_ret[neg_ret > 0] = 0
    df["downside_vol_20d"] = neg_ret.rolling(20, min_periods=20).std()

    return df


# ── 截面 z-score 标准化 ──────────────────────────────
def _cross_sectional_zscore(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        z_col = f"{col}_zscore"
        grouped = panel.groupby("Date")[col]
        panel[z_col] = (panel[col] - grouped.transform("median")) / grouped.transform("std")
        panel[z_col] = panel[z_col].clip(-3, 3)
    return panel


# ── 主流程 ────────────────────────────────────────────
def main() -> None:
    print(f"Reading {INPUT_PATH} ...")
    panel = pd.read_parquet(INPUT_PATH)
    panel = panel[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]].copy()

    print("Computing 25 factors per ticker ...")
    parts = []
    for ticker, grp in panel.groupby("ticker"):
        computed = _compute_factors(grp)
        computed["ticker"] = ticker
        parts.append(computed)
    panel = pd.concat(parts, ignore_index=True)

    # 删掉窗口期 NaN（最大窗口252日）
    before_drop = len(panel)
    panel = panel.dropna(subset=RAW_FACTOR_COLS).reset_index(drop=True)
    after_drop = len(panel)
    print(f"  Dropped warmup rows: {before_drop - after_drop}")

    print("Cross-sectional z-score ...")
    panel = _cross_sectional_zscore(panel, RAW_FACTOR_COLS)

    ZSCORE_COLS = [f"{c}_zscore" for c in RAW_FACTOR_COLS]

    # 输出
    out_cols = ["Date", "ticker", "Close"] + RAW_FACTOR_COLS + ZSCORE_COLS
    result = panel[out_cols].sort_values(["Date", "ticker"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}  ({result.shape[0]} rows x {result.shape[1]} cols)")

    # 报告
    report = {
        "rows_before_dropna": int(before_drop),
        "rows_after_dropna": int(after_drop),
        "dropped_warmup_rows": int(before_drop - after_drop),
        "output_rows": int(len(result)),
        "output_cols": int(result.shape[1]),
        "tickers": int(result["ticker"].nunique()),
        "date_range": [str(result["Date"].min()), str(result["Date"].max())],
        "raw_factors": RAW_FACTOR_COLS,
        "zscore_factors": ZSCORE_COLS,
        "factor_categories": {
            "momentum_reversal": ["return_1d", "momentum_5d", "momentum_20d", "momentum_60d",
                                  "return_5d_reversal", "high_52w_dist"],
            "volatility": ["volatility_20d", "volatility_60d", "vol_ratio_20_60", "intraday_range"],
            "volume_price": ["volume_ratio_20d", "volume_ratio_5d", "volume_std_20d",
                             "price_volume_corr_20d"],
            "technical": ["rsi_14", "ma_deviation_20d", "macd_signal", "bollinger_pos", "atr_14_pct"],
            "microstructure": ["close_position", "gap_open", "amihud_illiq_20d"],
            "higher_order_stats": ["return_skew_20d", "return_kurt_20d", "downside_vol_20d"],
        },
        "factor_stats": {},
        "correlation_matrix": {},
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

    corr = result[RAW_FACTOR_COLS].corr()
    # 只保存高相关对（|corr| > 0.7）
    high_corr_pairs = []
    for i, c1 in enumerate(RAW_FACTOR_COLS):
        for c2 in RAW_FACTOR_COLS[i + 1:]:
            r = float(corr.loc[c1, c2])
            if abs(r) > 0.7:
                high_corr_pairs.append({"factor_1": c1, "factor_2": c2, "corr": round(r, 4)})
    report["high_corr_pairs_above_0.7"] = high_corr_pairs

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")

    # 摘要
    print(f"\n── 25 Factors Stats ──")
    print(result[RAW_FACTOR_COLS].describe().round(4).to_string())
    print(f"\n── High Correlation Pairs (|r| > 0.7) ──")
    for p in high_corr_pairs:
        print(f"  {p['factor_1']:25s} <-> {p['factor_2']:25s}  r={p['corr']:.4f}")
    if not high_corr_pairs:
        print("  None")


if __name__ == "__main__":
    main()
