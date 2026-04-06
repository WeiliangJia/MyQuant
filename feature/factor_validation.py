"""
因子检验 — 方案 A
==================
全历史每日计算 Rank IC，用 Newey-West 标准误差修正重叠窗口带来的序列相关。

分析模块：
  1. IC 分析        每日截面 Rank IC + NW 修正 t 统计量
  2. IC 衰减分析    同一因子对 1/5/10/20/40/60 日收益的预测力
  3. 分层回测       五分位组合收益 + 多空收益
  4. Fama-MacBeth   每日横截面 OLS 回归，NW 修正，验证因子溢价显著性

输入：  feature/factors_v2.parquet
        data/data/raw/SPY.parquet
输出：  feature/factor_validation_report.json
        feature/ic_series.parquet          （每个因子的日 IC 时序）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ─────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR  = SCRIPT_DIR.parent
FACTOR_PATH  = SCRIPT_DIR / "factors_v2.parquet"
SPY_PATH     = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"
REPORT_PATH  = SCRIPT_DIR / "factor_validation_report.json"
IC_TS_PATH   = SCRIPT_DIR / "ic_series.parquet"

# ── 因子列表 ──────────────────────────────────────────────────────
FACTOR_COLS = [
    "return_1d", "momentum_5d", "momentum_20d", "momentum_60d",
    "return_5d_reversal", "high_52w_dist",
    "volatility_20d", "volatility_60d", "vol_ratio_20_60", "intraday_range",
    "volume_ratio_20d", "volume_ratio_5d", "volume_std_20d", "price_volume_corr_20d",
    "rsi_14", "ma_deviation_20d", "macd_signal", "bollinger_pos", "atr_14_pct",
    "close_position", "gap_open", "amihud_illiq_20d",
    "return_skew_20d", "return_kurt_20d", "downside_vol_20d",
]
ZSCORE_COLS = [f"{c}_zscore" for c in FACTOR_COLS]

# ── 参数 ─────────────────────────────────────────────────────────
PRIMARY_HORIZON  = 5              # 主要预测窗口（天）
FORWARD_WINDOWS  = [1, 5, 10, 20, 40, 60]
N_QUANTILES      = 5
# Newey-West 滞后阶数 = 预测窗口 - 1
# 原理：5日收益窗口相邻4天重叠，序列相关最多4阶
NW_LAGS_MAP = {w: max(w - 1, 0) for w in FORWARD_WINDOWS}

# IC 有效性门槛（行业常用标准）
IC_THRESHOLD    = 0.02
ICIR_THRESHOLD  = 0.30
T_THRESHOLD     = 2.0


# ════════════════════════════════════════════════════════════════
# 1. 数据加载与标签构建
# ════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载因子面板和 SPY 数据。"""
    factors = pd.read_parquet(FACTOR_PATH)
    factors["Date"] = pd.to_datetime(factors["Date"])

    spy = pd.read_parquet(SPY_PATH)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy["Date"] = pd.to_datetime(spy["Date"])
    return factors, spy


def build_labels(factors: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """
    构建多预测窗口的超额收益标签。

    标签定义：fwd_{w}d = 个股未来 w 日收益 − SPY 未来 w 日收益
    注意：使用超额收益，截面 Rank IC 结果与使用原始收益相同
    （因为 SPY 在同一天对所有股票是常数，不影响排名）。
    """
    spy_close = spy.set_index("Date")["Close"].sort_index()
    data = factors.sort_values(["ticker", "Date"]).copy()

    for w in FORWARD_WINDOWS:
        # SPY 未来 w 日收益（按日期索引）
        spy_fwd = spy_close.pct_change(w, fill_method=None).shift(-w)

        # 个股未来 w 日收益
        stock_fwd = (
            data.groupby("ticker")["Close"]
            .transform(lambda s: s.pct_change(w, fill_method=None).shift(-w))
        )

        # 超额收益
        data[f"fwd_{w}d"] = stock_fwd - data["Date"].map(spy_fwd)

    return data


# ════════════════════════════════════════════════════════════════
# 2. Newey-West 标准误差（手动实现，不依赖额外库）
# ════════════════════════════════════════════════════════════════

def newey_west_se(x: np.ndarray, lags: int) -> float:
    """
    计算样本均值的 Newey-West HAC 标准误差。

    修正了序列相关（如重叠收益窗口）导致的 t 统计量虚高问题。

    原理：
      普通 SE = std(x) / sqrt(T)  ← 假设 x 序列无相关
      NW SE   = sqrt(V_NW / T)   ← 用 Bartlett 核加权自协方差修正

    V_NW = γ₀ + 2 * Σ_{j=1}^L (1 - j/(L+1)) * γⱼ
    γⱼ  = (1/T) * Σ_{t=j+1}^T (xₜ - x̄)(xₜ₋ⱼ - x̄)
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 4:
        return float(np.std(x) / np.sqrt(max(T, 1)))

    x_dm = x - x.mean()                          # 去均值
    V = np.dot(x_dm, x_dm) / T                   # γ₀（方差项）

    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1)                 # Bartlett 核权重
        gamma_j = np.dot(x_dm[j:], x_dm[:-j]) / T
        V += 2.0 * w * gamma_j

    return float(np.sqrt(max(V, 1e-12) / T))


# ════════════════════════════════════════════════════════════════
# 3. IC 分析
# ════════════════════════════════════════════════════════════════

def compute_ic_series(
    data: pd.DataFrame, factor_col: str, label_col: str
) -> pd.Series:
    """
    每日计算截面 Rank IC（Spearman 相关系数）。

    每天用当天约 100 只股票的因子值与未来收益做排名相关，
    得到一条长度 ≈ 交易日数 的时序。
    """
    def _day_ic(g: pd.DataFrame) -> float:
        sub = g[[factor_col, label_col]].dropna()
        if len(sub) < 10:          # 样本太少不可靠
            return np.nan
        return float(sub[factor_col].corr(sub[label_col], method="spearman"))

    return data.groupby("Date").apply(_day_ic).dropna().rename(factor_col)


def summarize_ic(ic_series: pd.Series, lags: int) -> dict:
    """
    从 IC 时序提取关键统计量。

    关键指标：
      mean_ic       IC 均值：因子的平均预测强度
      icir          IC / std(IC)：信号稳定性，类比夏普比率
      t_stat_naive  普通 t 统计量（忽略序列相关，会高估显著性）
      t_stat_nw     Newey-West 修正 t 统计量（更可靠）
      positive_rate IC > 0 的天数占比：因子方向稳定性
    """
    arr = ic_series.dropna().values
    T = len(arr)
    if T == 0:
        return {}

    mean_ic = float(arr.mean())
    std_ic  = float(arr.std())
    icir    = mean_ic / std_ic if std_ic > 0 else 0.0

    # 普通 t（假设序列独立）
    t_naive = mean_ic / (std_ic / np.sqrt(T)) if std_ic > 0 else 0.0

    # NW 修正 t（考虑序列相关）
    se_nw = newey_west_se(arr, lags)
    t_nw  = mean_ic / se_nw if se_nw > 0 else 0.0

    positive_rate = float((arr > 0).mean())

    return {
        "mean_ic":       round(mean_ic, 5),
        "std_ic":        round(std_ic, 5),
        "icir":          round(icir, 4),
        "t_stat_naive":  round(t_naive, 3),
        "t_stat_nw":     round(t_nw, 3),
        "se_nw":         round(float(se_nw), 6),
        "positive_rate": round(positive_rate, 4),
        "is_significant_naive": bool(abs(t_naive) > T_THRESHOLD),
        "is_significant_nw":    bool(abs(t_nw) > T_THRESHOLD),
        "n_days":        int(T),
        "nw_lags_used":  int(lags),
    }


# ════════════════════════════════════════════════════════════════
# 4. IC 衰减分析
# ════════════════════════════════════════════════════════════════

def compute_ic_decay(data: pd.DataFrame, factor_col: str) -> dict:
    """
    计算因子在不同预测窗口（1~60 日）的 IC。

    衰减曲线告诉你：
    - 因子对短期（1-5天）更有效，还是对中期（20-60天）更有效
    - 应该设置什么样的调仓频率
    """
    decay = {}
    for w in FORWARD_WINDOWS:
        label_col = f"fwd_{w}d"
        sub = data[[factor_col, label_col, "Date"]].dropna()
        if len(sub) < 100:
            continue
        ic_s = compute_ic_series(sub, factor_col, label_col)
        lags  = NW_LAGS_MAP[w]
        summary = summarize_ic(ic_s, lags)
        decay[f"ic_{w}d"]   = summary.get("mean_ic", 0)
        decay[f"icir_{w}d"] = summary.get("icir", 0)
        decay[f"t_nw_{w}d"] = summary.get("t_stat_nw", 0)

    # 找绝对 IC 最大的预测窗口
    ic_by_window = {w: abs(decay.get(f"ic_{w}d", 0)) for w in FORWARD_WINDOWS}
    best_w = max(ic_by_window, key=ic_by_window.get)
    decay["best_horizon_days"] = best_w
    decay["best_ic"]           = round(decay.get(f"ic_{best_w}d", 0), 5)
    return decay


# ════════════════════════════════════════════════════════════════
# 5. 分层回测（五分位）
# ════════════════════════════════════════════════════════════════

def compute_quantile_returns(
    data: pd.DataFrame, factor_col: str, label_col: str = f"fwd_{PRIMARY_HORIZON}d"
) -> dict:
    """
    每天把股票按因子值分成 N_QUANTILES 组，统计各组平均超额收益。

    Q1 = 因子值最低的 20%
    Q5 = 因子值最高的 20%

    有效因子：Q1 到 Q5 应单调递增（或递减），多空收益（Q5-Q1）显著为正。
    """
    sub = data[["Date", factor_col, label_col]].dropna().copy()

    # 每天内部排名后分组（rank 避免 qcut 因重复值失败）
    sub["q"] = sub.groupby("Date")[factor_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), N_QUANTILES,
                          labels=[f"Q{i+1}" for i in range(N_QUANTILES)])
    )

    # 每天每组的平均超额收益
    label_days = int(label_col.replace("fwd_", "").replace("d", ""))
    group_daily = (
        sub.groupby(["Date", "q"])[label_col]
        .mean()
        .unstack("q")
        / label_days                  # 5日标签 → 近似日均收益
    )

    annual = group_daily.mean() * 252  # 年化

    result = {}
    for i in range(1, N_QUANTILES + 1):
        key = f"Q{i}"
        result[f"Q{i}_annual"] = round(float(annual.get(key, np.nan)), 4)

    # 多空年化收益
    q5 = annual.get(f"Q{N_QUANTILES}", np.nan)
    q1 = annual.get("Q1", np.nan)
    if not (np.isnan(q5) or np.isnan(q1)):
        result["long_short_annual"] = round(float(q5 - q1), 4)
    else:
        result["long_short_annual"] = None

    # 单调性得分：Q1<Q2<...<Q5 中满足递增的相邻对数（满分 4）
    vals = [annual.get(f"Q{i+1}", np.nan) for i in range(N_QUANTILES)]
    if not any(np.isnan(v) for v in vals):
        result["monotonicity_score"] = int(
            sum(vals[i] < vals[i+1] for i in range(len(vals) - 1))
        )

    return result


# ════════════════════════════════════════════════════════════════
# 6. Fama-MacBeth 回归
# ════════════════════════════════════════════════════════════════

def fama_macbeth_single(
    data: pd.DataFrame, factor_col: str,
    label_col: str = f"fwd_{PRIMARY_HORIZON}d",
    lags: int = NW_LAGS_MAP[PRIMARY_HORIZON],
) -> dict:
    """
    单因子 Fama-MacBeth 回归。

    步骤：
      每天做横截面 OLS：fwd_ret_i = α_t + β_t * factor_i + ε_it
      收集 {β_t} 时序（约 2000 个值）
      对 {β_t} 做均值 t 检验，用 NW 修正标准误差

    与 IC 的区别：
      IC 用 Spearman（秩相关），对极值不敏感
      FM 用 OLS，β 有经济含义：因子上升 1 标准差 → 超额收益变化 β%
    """
    betas: list[float] = []

    for _, g in data.groupby("Date"):
        sub = g[[factor_col, label_col]].dropna()
        if len(sub) < 10:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[factor_col].values])
        y = sub[label_col].values
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            betas.append(float(coef[1]))
        except np.linalg.LinAlgError:
            continue

    if len(betas) < 30:
        return {"error": "insufficient data"}

    arr      = np.array(betas)
    mean_b   = float(arr.mean())
    se_nw    = newey_west_se(arr, lags)
    t_nw     = mean_b / se_nw if se_nw > 0 else 0.0

    return {
        "mean_beta":      round(mean_b, 6),
        "se_nw":          round(se_nw, 6),
        "t_stat_nw":      round(t_nw, 3),
        "is_significant": bool(abs(t_nw) > T_THRESHOLD),
        "n_days":         len(arr),
        "interpretation": f"因子 +1σ → 超额收益 {mean_b*100:.3f}%/期",
    }


# ════════════════════════════════════════════════════════════════
# 7. 打印 & 汇总
# ════════════════════════════════════════════════════════════════

def _bar(v: float, width: int = 12) -> str:
    """在终端用字符画一个小横条，直观显示 IC 大小。"""
    filled = int(abs(v) / 0.08 * width)
    filled = min(filled, width)
    char   = "+" if v >= 0 else "-"
    return char * filled + "·" * (width - filled)


def print_ic_table(ic_results: dict[str, dict]) -> None:
    """打印 IC 分析汇总表，按 NW t 统计量绝对值排序。"""
    rows = [(f, r) for f, r in ic_results.items() if r]
    rows.sort(key=lambda x: abs(x[1].get("t_stat_nw", 0)), reverse=True)

    header = (f"  {'Factor':<35} {'MeanIC':>8} {'ICIR':>6} "
              f"{'t(naive)':>9} {'t(NW)':>7} {'Win%':>6} {'Sig':>4}  Bar")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for factor, r in rows:
        fname  = factor.replace("_zscore", "")
        sig    = "Y" if r["is_significant_nw"] else " "
        bar    = _bar(r["mean_ic"])
        print(f"  {fname:<35} {r['mean_ic']:>8.4f} {r['icir']:>6.3f} "
              f"{r['t_stat_naive']:>9.2f} {r['t_stat_nw']:>7.2f} "
              f"{r['positive_rate']:>6.1%} {sig:>4}  {bar}")


def print_decay_table(decay_results: dict[str, dict], top_n: int = 10) -> None:
    """打印 IC 衰减表，只显示主要因子。"""
    # 取 fwd_5d IC 绝对值最大的 top_n 个因子
    ranked = sorted(decay_results.items(),
                    key=lambda x: abs(x[1].get("ic_5d", 0)), reverse=True)[:top_n]

    header = (f"  {'Factor':<30} "
              + "".join(f"{'ic_'+str(w)+'d':>8}" for w in FORWARD_WINDOWS)
              + f"  {'Best':>6}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for factor, d in ranked:
        fname = factor.replace("_zscore", "")
        vals  = "".join(f"{d.get(f'ic_{w}d', 0):>8.4f}" for w in FORWARD_WINDOWS)
        print(f"  {fname:<30} {vals}  {d.get('best_horizon_days', '?'):>6}d")


def print_quantile_table(qr_results: dict[str, dict], top_n: int = 10) -> None:
    """打印分层回测表。"""
    ranked = sorted(
        qr_results.items(),
        key=lambda x: abs(x[1].get("long_short_annual") or 0), reverse=True
    )[:top_n]

    header = (f"  {'Factor':<30} "
              + "".join(f"{'Q'+str(i):>8}" for i in range(1, N_QUANTILES+1))
              + f"  {'Q5-Q1':>8}  {'Mono':>4}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for factor, r in ranked:
        fname = factor.replace("_zscore", "")
        vals  = "".join(f"{r.get(f'Q{i}_annual', 0):>8.2%}" for i in range(1, N_QUANTILES+1))
        ls    = r.get("long_short_annual")
        ls_s  = f"{ls:>8.2%}" if ls is not None else "     N/A"
        mono  = r.get("monotonicity_score", "?")
        print(f"  {fname:<30} {vals}  {ls_s}  {mono:>4}")


def print_fm_table(fm_results: dict[str, dict]) -> None:
    """打印 Fama-MacBeth 结果表，按 |t| 排序。"""
    rows = [(f, r) for f, r in fm_results.items()
            if isinstance(r, dict) and "t_stat_nw" in r]
    rows.sort(key=lambda x: abs(x[1]["t_stat_nw"]), reverse=True)

    header = f"  {'Factor':<35} {'Beta':>10} {'SE(NW)':>9} {'t(NW)':>7} {'Sig':>4}  Interpretation"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for factor, r in rows:
        fname = factor.replace("_zscore", "")
        sig   = "Y" if r["is_significant"] else " "
        interp = r.get("interpretation", "")
        print(f"  {fname:<35} {r['mean_beta']:>10.5f} {r['se_nw']:>9.5f} "
              f"{r['t_stat_nw']:>7.2f} {sig:>4}  {interp}")


# ════════════════════════════════════════════════════════════════
# 8. 主流程
# ════════════════════════════════════════════════════════════════

def main() -> None:
    w = PRIMARY_HORIZON
    label_col = f"fwd_{w}d"
    nw_lags   = NW_LAGS_MAP[w]

    print("=" * 70)
    print("  FACTOR VALIDATION  (Method A: Full History + Newey-West)")
    print(f"  Primary horizon: {w}d   NW lags: {nw_lags}")
    print("=" * 70)

    # ── 数据准备 ──────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    factors, spy = load_data()

    print("[2/6] Building multi-horizon labels ...")
    data = build_labels(factors, spy)
    data_clean = data.dropna(subset=ZSCORE_COLS).reset_index(drop=True)
    print(f"      Data: {data_clean.shape[0]:,} rows × {data_clean.shape[1]} cols"
          f"  |  {data_clean['ticker'].nunique()} tickers"
          f"  |  {data_clean['Date'].min().date()} ~ {data_clean['Date'].max().date()}")

    # 主分析子集（primary horizon 标签有效的行）
    data_primary = data_clean.dropna(subset=[label_col]).copy()
    n_dates      = data_primary["Date"].nunique()
    print(f"      Primary subset ({label_col}): {len(data_primary):,} rows, {n_dates} dates")

    # ── IC 分析 ───────────────────────────────────────────────
    print(f"\n[3/6] IC Analysis (fwd_{w}d, Rank IC, NW lags={nw_lags}) ...")
    ic_results: dict[str, dict] = {}
    all_ic_series: dict[str, pd.Series] = {}

    for fcol in ZSCORE_COLS:
        sub     = data_primary[["Date", fcol, label_col]].dropna()
        ic_s    = compute_ic_series(sub, fcol, label_col)
        summary = summarize_ic(ic_s, nw_lags)
        ic_results[fcol]    = summary
        all_ic_series[fcol] = ic_s

    print(f"\n  IC Summary Table (sorted by |t_NW|, fwd_{w}d):\n")
    print_ic_table(ic_results)

    # ── IC 衰减 ───────────────────────────────────────────────
    print("\n[4/6] IC Decay Analysis ...")
    decay_results: dict[str, dict] = {}
    for fcol in ZSCORE_COLS:
        decay_results[fcol] = compute_ic_decay(data_clean, fcol)

    print(f"\n  IC Decay Table (top 10 by |ic_5d|):\n")
    print_decay_table(decay_results)

    # ── 分层回测 ──────────────────────────────────────────────
    print("\n[5/6] Quantile Returns ...")
    qr_results: dict[str, dict] = {}
    for fcol in ZSCORE_COLS:
        qr_results[fcol] = compute_quantile_returns(data_primary, fcol, label_col)

    print(f"\n  Quantile Returns Table (top 10 by |Q5-Q1|, annualized):\n")
    print_quantile_table(qr_results)

    # ── Fama-MacBeth ──────────────────────────────────────────
    print("\n[6/6] Fama-MacBeth Regression ...")
    fm_results: dict[str, dict] = {}
    for fcol in ZSCORE_COLS:
        fm_results[fcol] = fama_macbeth_single(data_primary, fcol, label_col, nw_lags)

    print(f"\n  Fama-MacBeth Table (sorted by |t_NW|):\n")
    print_fm_table(fm_results)

    # ── 因子筛选汇总 ──────────────────────────────────────────
    effective, weak = [], []
    for fcol in ZSCORE_COLS:
        r = ic_results.get(fcol, {})
        if (abs(r.get("mean_ic", 0)) > IC_THRESHOLD
                and abs(r.get("icir", 0)) > ICIR_THRESHOLD
                and r.get("is_significant_nw", False)):
            effective.append(fcol.replace("_zscore", ""))
        else:
            weak.append(fcol.replace("_zscore", ""))

    print(f"\n{'='*70}")
    print(f"  FACTOR SELECTION SUMMARY")
    print(f"  Threshold: |MeanIC| > {IC_THRESHOLD}, |ICIR| > {ICIR_THRESHOLD}, |t_NW| > {T_THRESHOLD}")
    print(f"{'='*70}")
    print(f"\n  [Y] Effective ({len(effective)}):")
    for f in effective:
        r = ic_results[f + "_zscore"]
        print(f"    {f:<35} IC={r['mean_ic']:+.4f}  ICIR={r['icir']:.3f}  t_NW={r['t_stat_nw']:.2f}")
    print(f"\n  [N] Weak ({len(weak)}):")
    print(f"    {', '.join(weak)}")

    # ── 保存输出 ──────────────────────────────────────────────
    ic_ts_df = pd.DataFrame(all_ic_series)
    ic_ts_df.index.name = "Date"
    ic_ts_df.to_parquet(IC_TS_PATH)
    print(f"\nSaved IC time series: {IC_TS_PATH}")

    report = {
        "meta": {
            "primary_horizon_days": w,
            "nw_lags":              nw_lags,
            "forward_windows":      FORWARD_WINDOWS,
            "n_quantiles":          N_QUANTILES,
            "ic_threshold":         IC_THRESHOLD,
            "icir_threshold":       ICIR_THRESHOLD,
            "t_threshold":          T_THRESHOLD,
            "n_tickers":            int(data_primary["ticker"].nunique()),
            "n_dates":              int(n_dates),
            "date_range": [
                str(data_primary["Date"].min().date()),
                str(data_primary["Date"].max().date()),
            ],
        },
        "ic_analysis":    ic_results,
        "ic_decay":       decay_results,
        "quantile_returns": qr_results,
        "fama_macbeth":   fm_results,
        "factor_selection": {
            "effective": effective,
            "weak":      weak,
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Saved report: {REPORT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
