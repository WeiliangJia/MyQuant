"""
第三轮因子工程 — 25 个股票因子 + 8 个跨资产因子 = 33 个因子
===========================================================
在 V2（25个股票因子）基础上新增 8 个跨资产因子：

跨资产-VIX (3):   vix_level, vix_change_5d, vix_term_structure
跨资产-利率 (3):  tnx_level, tnx_change_20d, tnx_momentum_60d
跨资产-美元 (2):  dxy_change_20d, dxy_momentum_60d

跨资产因子特点：每天所有股票共享同一值（宏观因子），
但与个股因子交互后可以产生差异化信号。
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
RAW_DIR = PROJECT_DIR / "data" / "data" / "raw"
INPUT_PATH = PROCESSED_DIR / "panel_tradable.parquet"
VIX_PATH = RAW_DIR / "VIX.parquet"
TNX_PATH = RAW_DIR / "TNX.parquet"
DXY_PATH = RAW_DIR / "DXY.parquet"
OUTPUT_PATH = SCRIPT_DIR / "factors_v3.parquet"
REPORT_PATH = SCRIPT_DIR / "factors_v3_report.json"

# ── 原有 25 个股票因子 ─────────────────────────────────
STOCK_FACTOR_COLS = [
    # 动量/反转
    "return_1d", "momentum_5d", "momentum_20d", "momentum_60d",
    "return_5d_reversal", "high_52w_dist",
    # 波动率
    "volatility_20d", "volatility_60d", "vol_ratio_20_60", "intraday_range",
    # 量价
    "volume_ratio_20d", "volume_ratio_5d", "volume_std_20d", "price_volume_corr_20d",
    # 技术指标
    "rsi_14", "ma_deviation_20d", "macd_signal", "bollinger_pos", "atr_14_pct",
    # 微观结构
    "close_position", "gap_open", "amihud_illiq_20d",
    # 高阶统计
    "return_skew_20d", "return_kurt_20d", "downside_vol_20d",
]

# ── 新增 8 个跨资产因子 ────────────────────────────────
CROSS_ASSET_COLS = [
    "vix_level", "vix_change_5d", "vix_term_structure",
    "tnx_level", "tnx_change_20d", "tnx_momentum_60d",
    "dxy_change_20d", "dxy_momentum_60d",
]

ALL_RAW_FACTOR_COLS = STOCK_FACTOR_COLS + CROSS_ASSET_COLS


# ── 股票因子计算（和 V2 一样） ─────────────────────────
def _compute_stock_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    opn = df["Open"]
    volume = df["Volume"]
    daily_ret = close.pct_change(fill_method=None)

    # 动量/反转
    df["return_1d"] = daily_ret
    df["momentum_5d"] = close.pct_change(5, fill_method=None)
    df["momentum_20d"] = close.pct_change(20, fill_method=None)
    df["momentum_60d"] = close.pct_change(60, fill_method=None)
    df["return_5d_reversal"] = -close.pct_change(5, fill_method=None)
    rolling_high_252 = high.rolling(252, min_periods=252).max()
    df["high_52w_dist"] = close / rolling_high_252 - 1

    # 波动率
    df["volatility_20d"] = daily_ret.rolling(20, min_periods=20).std()
    df["volatility_60d"] = daily_ret.rolling(60, min_periods=60).std()
    df["vol_ratio_20_60"] = df["volatility_20d"] / df["volatility_60d"].replace(0, np.nan)
    df["intraday_range"] = ((high - low) / close).rolling(20, min_periods=20).mean()

    # 量价
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    vol_ma5 = volume.rolling(5, min_periods=5).mean()
    df["volume_ratio_20d"] = volume / vol_ma20.replace(0, np.nan)
    df["volume_ratio_5d"] = volume / vol_ma5.replace(0, np.nan)
    df["volume_std_20d"] = volume.rolling(20, min_periods=20).std() / vol_ma20.replace(0, np.nan)
    df["price_volume_corr_20d"] = daily_ret.rolling(20, min_periods=20).corr(volume)

    # 技术指标
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ma20 = close.rolling(20, min_periods=20).mean()
    df["ma_deviation_20d"] = close / ma20 - 1

    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    df["macd_signal"] = (macd_line - macd_signal_line) / close

    std20 = close.rolling(20, min_periods=20).std()
    df["bollinger_pos"] = (close - ma20) / (2 * std20.replace(0, np.nan))

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, min_periods=14, adjust=False).mean()
    df["atr_14_pct"] = atr14 / close

    # 微观结构
    hl_range = (high - low).replace(0, np.nan)
    df["close_position"] = (close - low) / hl_range
    df["gap_open"] = (opn - close.shift(1)) / close.shift(1)
    dollar_vol = close * volume
    amihud_daily = daily_ret.abs() / dollar_vol.replace(0, np.nan)
    df["amihud_illiq_20d"] = amihud_daily.rolling(20, min_periods=20).mean()

    # 高阶统计
    df["return_skew_20d"] = daily_ret.rolling(20, min_periods=20).skew()
    df["return_kurt_20d"] = daily_ret.rolling(20, min_periods=20).kurt()
    neg_ret = daily_ret.copy()
    neg_ret[neg_ret > 0] = 0
    df["downside_vol_20d"] = neg_ret.rolling(20, min_periods=20).std()

    return df


# ── 跨资产因子计算 ────────────────────────────────────
def _build_cross_asset_factors(vix: pd.DataFrame, tnx: pd.DataFrame, dxy: pd.DataFrame) -> pd.DataFrame:
    """构建跨资产因子，返回 Date + 8 个因子列。"""

    def _clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Date", "Close"]].copy().sort_values("Date")
        df = df.rename(columns={"Close": f"{name}_close"})
        return df

    vix = _clean(vix, "vix")
    tnx = _clean(tnx, "tnx")
    dxy = _clean(dxy, "dxy")

    cross = vix.merge(tnx, on="Date", how="outer").merge(dxy, on="Date", how="outer")
    cross = cross.sort_values("Date").reset_index(drop=True)

    # 前向填充（跨资产数据可能有不同交易日）
    cross = cross.ffill()

    # ── VIX 因子 ──
    # 1. VIX 水平（对数化，因为 VIX 右偏）
    cross["vix_level"] = np.log(cross["vix_close"].clip(lower=1))
    # 2. VIX 5日变化率
    cross["vix_change_5d"] = cross["vix_close"].pct_change(5, fill_method=None)
    # 3. VIX 期限结构代理：短期波动 vs 长期均值
    vix_ma60 = cross["vix_close"].rolling(60, min_periods=60).mean()
    cross["vix_term_structure"] = cross["vix_close"] / vix_ma60.replace(0, np.nan) - 1

    # ── 利率因子 ──
    # 4. 10Y yield 水平
    cross["tnx_level"] = cross["tnx_close"]
    # 5. 利率 20日变化（绝对值变化，因为利率本身是百分比）
    cross["tnx_change_20d"] = cross["tnx_close"] - cross["tnx_close"].shift(20)
    # 6. 利率 60日动量
    cross["tnx_momentum_60d"] = cross["tnx_close"] - cross["tnx_close"].shift(60)

    # ── 美元因子 ──
    # 7. 美元指数 20日变化率
    cross["dxy_change_20d"] = cross["dxy_close"].pct_change(20, fill_method=None)
    # 8. 美元指数 60日动量
    cross["dxy_momentum_60d"] = cross["dxy_close"].pct_change(60, fill_method=None)

    out_cols = ["Date"] + CROSS_ASSET_COLS
    return cross[out_cols].dropna()


# ── 截面 z-score 标准化 ──────────────────────────────
def _cross_sectional_zscore(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        z_col = f"{col}_zscore"
        grouped = panel.groupby("Date")[col]
        panel[z_col] = (panel[col] - grouped.transform("median")) / grouped.transform("std")
        panel[z_col] = panel[z_col].clip(-3, 3)
    return panel


def _time_series_zscore(panel: pd.DataFrame, cols: list[str], window: int = 60) -> pd.DataFrame:
    """跨资产因子用时序标准化（因为截面上所有股票同一值）。"""
    for col in cols:
        z_col = f"{col}_zscore"
        rolling_mean = panel.groupby("ticker")[col].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        rolling_std = panel.groupby("ticker")[col].transform(
            lambda s: s.rolling(window, min_periods=window).std()
        )
        panel[z_col] = (panel[col] - rolling_mean) / rolling_std.replace(0, np.nan)
        panel[z_col] = panel[z_col].clip(-3, 3)
    return panel


# ── 主流程 ────────────────────────────────────────────
def main() -> None:
    print("="*60)
    print("  FACTOR V3: 25 Stock + 8 Cross-Asset = 33 Factors")
    print("="*60)

    # 加载数据
    print(f"\nReading stock data: {INPUT_PATH}")
    panel = pd.read_parquet(INPUT_PATH)
    panel = panel[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]].copy()

    print(f"Reading cross-asset data ...")
    vix = pd.read_parquet(VIX_PATH)
    tnx = pd.read_parquet(TNX_PATH)
    dxy = pd.read_parquet(DXY_PATH)

    # 构建跨资产因子
    print("Building cross-asset factors ...")
    cross_factors = _build_cross_asset_factors(vix, tnx, dxy)
    print(f"  Cross-asset factor table: {cross_factors.shape[0]} days, "
          f"{cross_factors['Date'].min().date()} ~ {cross_factors['Date'].max().date()}")

    # 构建股票因子
    print("Computing 25 stock factors per ticker ...")
    parts = []
    for ticker, grp in panel.groupby("ticker"):
        computed = _compute_stock_factors(grp)
        computed["ticker"] = ticker
        parts.append(computed)
    panel = pd.concat(parts, ignore_index=True)

    # 合并跨资产因子到面板
    print("Merging cross-asset factors ...")
    panel = panel.merge(cross_factors, on="Date", how="left")

    # 删掉窗口期 NaN
    before_drop = len(panel)
    panel = panel.dropna(subset=ALL_RAW_FACTOR_COLS).reset_index(drop=True)
    after_drop = len(panel)
    print(f"  Dropped warmup rows: {before_drop - after_drop}")

    # 标准化：股票因子用截面 z-score，跨资产因子用时序 z-score
    print("Cross-sectional z-score for stock factors ...")
    panel = _cross_sectional_zscore(panel, STOCK_FACTOR_COLS)

    print("Time-series z-score for cross-asset factors ...")
    panel = _time_series_zscore(panel, CROSS_ASSET_COLS, window=60)

    ALL_ZSCORE_COLS = [f"{c}_zscore" for c in ALL_RAW_FACTOR_COLS]

    # 输出
    out_cols = ["Date", "ticker", "Close"] + ALL_RAW_FACTOR_COLS + ALL_ZSCORE_COLS
    result = panel[out_cols].sort_values(["Date", "ticker"]).reset_index(drop=True)

    # 删掉时序 z-score 产生的 NaN
    result = result.dropna(subset=ALL_ZSCORE_COLS).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}  ({result.shape[0]} rows x {result.shape[1]} cols)")

    # 报告
    report = {
        "rows": int(len(result)),
        "cols": int(result.shape[1]),
        "tickers": int(result["ticker"].nunique()),
        "date_range": [str(result["Date"].min()), str(result["Date"].max())],
        "stock_factors": STOCK_FACTOR_COLS,
        "cross_asset_factors": CROSS_ASSET_COLS,
        "all_factors": ALL_RAW_FACTOR_COLS,
        "total_factor_count": len(ALL_RAW_FACTOR_COLS),
        "factor_stats": {},
    }

    for col in ALL_RAW_FACTOR_COLS:
        s = result[col]
        report["factor_stats"][col] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
        }

    # 跨资产因子与股票因子的相关性
    zscore_cols = [f"{c}_zscore" for c in ALL_RAW_FACTOR_COLS]
    corr = result[zscore_cols].corr()
    cross_stock_corr = {}
    for ca in CROSS_ASSET_COLS:
        ca_z = f"{ca}_zscore"
        for sf in STOCK_FACTOR_COLS[:5]:  # 只看几个关键股票因子
            sf_z = f"{sf}_zscore"
            if ca_z in corr.columns and sf_z in corr.columns:
                cross_stock_corr[f"{ca} vs {sf}"] = round(float(corr.loc[ca_z, sf_z]), 4)
    report["cross_asset_vs_stock_correlation"] = cross_stock_corr

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")

    # 摘要
    print(f"\n── Cross-Asset Factor Stats ──")
    print(result[CROSS_ASSET_COLS].describe().round(4).to_string())

    print(f"\n── Cross-Asset vs Stock Factor Correlation ──")
    for pair, r in cross_stock_corr.items():
        print(f"  {pair:40s}  r={r:.4f}")


if __name__ == "__main__":
    main()
