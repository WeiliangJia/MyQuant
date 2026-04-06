"""
多策略框架 V5
=============
与 multi_strategy.py 完全相同，仅将策略 A 的 ML 预测换成 V5 版本。

策略 A: ML 多因子 — 使用 predictions_v5.parquet（25 特征精简版）
策略 B: 短期反转  — 不变
策略 C: 动量趋势  — 不变
策略 D: 低波动    — 不变
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR         = Path(__file__).resolve().parent
PROJECT_DIR        = SCRIPT_DIR.parent
FEATURE_PATH       = PROJECT_DIR / "feature" / "factors_v3.parquet"
ML_PREDICTIONS_PATH = SCRIPT_DIR / "predictions_v5.parquet"   # V5
SPY_PATH           = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"

HOLDINGS_PATH = SCRIPT_DIR / "holdings_multi_v5.parquet"      # V5
REPORT_PATH   = SCRIPT_DIR / "multi_strategy_v5_report.json"  # V5

TOP_K        = 20
FORWARD_DAYS = 5


def strategy_a_ml(predictions: pd.DataFrame) -> pd.DataFrame:
    rebalance_dates = sorted(predictions["Date"].unique())[::10]
    current_holdings: set[str] = set()
    holdings = []
    SELL_THRESHOLD = 30
    HOLDING_BONUS  = 0.002

    for rb_date in rebalance_dates:
        day = predictions[predictions["Date"] == rb_date].copy()
        if len(day) == 0:
            continue

        day["adj_score"] = day["pred_lgb"]
        day.loc[day["ticker"].isin(current_holdings), "adj_score"] += HOLDING_BONUS
        day = day.sort_values("adj_score", ascending=False).reset_index(drop=True)
        day["rank"] = range(1, len(day) + 1)

        keep  = set(day[(day["ticker"].isin(current_holdings)) & (day["rank"] <= SELL_THRESHOLD)]["ticker"])
        slots = TOP_K - len(keep)
        new   = set(day[(~day["ticker"].isin(keep)) & (day["rank"] <= TOP_K)].head(max(0, slots))["ticker"])
        new_holdings = keep | new
        if len(new_holdings) < TOP_K:
            extra = day[~day["ticker"].isin(new_holdings)].head(TOP_K - len(new_holdings))
            new_holdings |= set(extra["ticker"])
        final = day[day["ticker"].isin(new_holdings)].nlargest(TOP_K, "adj_score")
        current_holdings = set(final["ticker"])

        window_dates = sorted(predictions[predictions["Date"] >= rb_date]["Date"].unique())[:10]
        w = 1.0 / len(current_holdings)
        for d in window_dates:
            for tkr in current_holdings:
                row   = predictions[(predictions["Date"] == d) & (predictions["ticker"] == tkr)]
                label = float(row["label"].iloc[0]) if len(row) > 0 else 0.0
                holdings.append({"Date": d, "ticker": tkr, "weight": w, "actual_label": label})

    df = pd.DataFrame(holdings)
    print(f"  A(V5): {df['Date'].nunique()} days, {len(df)} rows")
    return df


def strategy_b_reversal(factors: pd.DataFrame) -> pd.DataFrame:
    required = factors[["Date", "ticker", "Close", "momentum_5d"]].dropna().copy()
    dates    = sorted(required["Date"].unique())
    rebalance_dates = dates[::5]
    holdings = []
    for rb_date in rebalance_dates:
        day = required[required["Date"] == rb_date].copy()
        if len(day) < TOP_K:
            continue
        bottom   = day.nsmallest(TOP_K, "momentum_5d")
        selected = set(bottom["ticker"])
        window_dates = [d for d in dates if d >= rb_date][:5]
        w = 1.0 / len(selected)
        for d in window_dates:
            for tkr in selected:
                holdings.append({"Date": d, "ticker": tkr, "weight": w, "actual_label": 0.0})
    df = pd.DataFrame(holdings).drop_duplicates(subset=["Date", "ticker"], keep="last")
    print(f"  B: {df['Date'].nunique()} days, {len(df)} rows")
    return df


def strategy_c_momentum(factors: pd.DataFrame) -> pd.DataFrame:
    required = factors[["Date", "ticker", "Close", "momentum_60d"]].dropna().copy()
    dates    = sorted(required["Date"].unique())
    rebalance_dates = dates[::20]
    holdings = []
    for rb_date in rebalance_dates:
        day = required[required["Date"] == rb_date].copy()
        if len(day) < TOP_K:
            continue
        top      = day.nlargest(TOP_K, "momentum_60d")
        selected = set(top["ticker"])
        window_dates = [d for d in dates if d >= rb_date][:20]
        w = 1.0 / len(selected)
        for d in window_dates:
            for tkr in selected:
                holdings.append({"Date": d, "ticker": tkr, "weight": w, "actual_label": 0.0})
    df = pd.DataFrame(holdings).drop_duplicates(subset=["Date", "ticker"], keep="last")
    print(f"  C: {df['Date'].nunique()} days, {len(df)} rows")
    return df


def strategy_d_low_vol(factors: pd.DataFrame) -> pd.DataFrame:
    required = factors[["Date", "ticker", "Close", "volatility_20d"]].dropna().copy()
    dates    = sorted(required["Date"].unique())
    rebalance_dates = dates[::20]
    holdings = []
    for rb_date in rebalance_dates:
        day = required[required["Date"] == rb_date].copy()
        if len(day) < TOP_K:
            continue
        bottom   = day.nsmallest(TOP_K, "volatility_20d")
        selected = set(bottom["ticker"])
        window_dates = [d for d in dates if d >= rb_date][:20]
        w = 1.0 / len(selected)
        for d in window_dates:
            for tkr in selected:
                holdings.append({"Date": d, "ticker": tkr, "weight": w, "actual_label": 0.0})
    df = pd.DataFrame(holdings).drop_duplicates(subset=["Date", "ticker"], keep="last")
    print(f"  D: {df['Date'].nunique()} days, {len(df)} rows")
    return df


def fill_actual_labels(holdings: pd.DataFrame, factors: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    spy = spy[["Date", "Close"]].copy().sort_values("Date")
    spy["spy_fwd_ret"] = spy["Close"].pct_change(FORWARD_DAYS, fill_method=None).shift(-FORWARD_DAYS)

    fwd_ret = factors[["Date", "ticker", "Close"]].copy().sort_values(["ticker", "Date"])
    fwd_ret["stock_fwd_ret"] = (
        fwd_ret.groupby("ticker")["Close"]
        .transform(lambda s: s.pct_change(FORWARD_DAYS, fill_method=None).shift(-FORWARD_DAYS))
    )
    fwd_ret = fwd_ret.merge(spy[["Date", "spy_fwd_ret"]], on="Date", how="left")
    fwd_ret["label"] = fwd_ret["stock_fwd_ret"] - fwd_ret["spy_fwd_ret"]
    label_map = fwd_ret.set_index(["Date", "ticker"])["label"]

    holdings["actual_label"] = holdings.apply(
        lambda r: label_map.get((r["Date"], r["ticker"]), np.nan), axis=1
    )
    return holdings


def combine_strategies(strat_holdings: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    combined = []
    for name, h in strat_holdings.items():
        h = h.copy()
        h["weight"]   = h["weight"] * weights[name]
        h["strategy"] = name
        combined.append(h)
    merged = pd.concat(combined, ignore_index=True)
    final  = (
        merged.groupby(["Date", "ticker"])
        .agg({"weight": "sum", "actual_label": "first"})
        .reset_index()
    )
    daily_sum      = final.groupby("Date")["weight"].transform("sum")
    final["weight"] = final["weight"] / daily_sum
    print(f"  Combined: {final['Date'].nunique()} days, avg {final.groupby('Date').size().mean():.1f} stocks/day")
    return final


def evaluate_strategy(holdings: pd.DataFrame, name: str) -> dict:
    clean = holdings.dropna(subset=["actual_label"])
    if len(clean) == 0:
        return {"name": name, "error": "no valid labels"}
    daily_port = (
        clean.groupby("Date")
        .apply(lambda g: (g["weight"] * g["actual_label"]).sum())
        .sort_index()
    )
    daily_ret  = daily_port / FORWARD_DAYS
    total_ret  = float((1 + daily_ret).prod() - 1)
    n_days     = max(len(daily_ret), 1)
    annual_ret = float((1 + total_ret) ** (252 / n_days) - 1)
    annual_vol = float(daily_ret.std() * np.sqrt(252))
    sharpe     = annual_ret / annual_vol if annual_vol > 0 else 0
    max_dd     = float((daily_ret.cumsum() - daily_ret.cumsum().cummax()).min())
    win_rate   = float((daily_ret > 0).mean())
    return {
        "name": name,
        "annual_return": round(annual_ret, 4),
        "annual_vol":    round(annual_vol, 4),
        "sharpe":        round(sharpe, 4),
        "max_drawdown":  round(max_dd, 4),
        "win_rate":      round(win_rate, 4),
        "trading_days":  int(n_days),
        "_daily_ret":    daily_ret,
    }


def main():
    print("=" * 60)
    print("  MULTI-STRATEGY V5 (Strategy A uses V5 predictions)")
    print("=" * 60)

    factors  = pd.read_parquet(FEATURE_PATH)
    ml_preds = pd.read_parquet(ML_PREDICTIONS_PATH)
    spy_raw  = pd.read_parquet(SPY_PATH)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]

    print("\nRunning 4 strategies ...")
    h_a = strategy_a_ml(ml_preds)
    h_b = strategy_b_reversal(factors)
    h_c = strategy_c_momentum(factors)
    h_d = strategy_d_low_vol(factors)

    print("\nFilling actual labels ...")
    h_b = fill_actual_labels(h_b, factors, spy_raw)
    h_c = fill_actual_labels(h_c, factors, spy_raw)
    h_d = fill_actual_labels(h_d, factors, spy_raw)

    common_start = max(h_a["Date"].min(), h_b["Date"].min(), h_c["Date"].min(), h_d["Date"].min())
    common_end   = min(h_a["Date"].max(), h_b["Date"].max(), h_c["Date"].max(), h_d["Date"].max())
    h_a = h_a[(h_a["Date"] >= common_start) & (h_a["Date"] <= common_end)]
    h_b = h_b[(h_b["Date"] >= common_start) & (h_b["Date"] <= common_end)]
    h_c = h_c[(h_c["Date"] >= common_start) & (h_c["Date"] <= common_end)]
    h_d = h_d[(h_d["Date"] >= common_start) & (h_d["Date"] <= common_end)]
    print(f"\nCommon period: {common_start.date()} ~ {common_end.date()}")

    strats  = {"A_ML": h_a, "B_Reversal": h_b, "C_Momentum": h_c, "D_LowVol": h_d}
    results = {}
    daily_rets = {}

    print(f"\n{'='*60}\n  INDIVIDUAL STRATEGY PERFORMANCE\n{'='*60}")
    for name, h in strats.items():
        r = evaluate_strategy(h, name)
        results[name] = r
        daily_rets[name] = r.pop("_daily_ret")
        print(f"  {name:<15} Return={r['annual_return']:7.2%}  Vol={r['annual_vol']:6.2%}  "
              f"Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_drawdown']:7.2%}")

    ret_df = pd.DataFrame(daily_rets).dropna()
    corr   = ret_df.corr()
    print(f"\n-- Strategy Correlation --")
    print(corr.round(3).to_string())

    equal_weights = {k: 0.25 for k in strats}
    h_equal       = combine_strategies(strats, equal_weights)
    r_equal       = evaluate_strategy(h_equal, "Combined_Equal")
    _ = r_equal.pop("_daily_ret")

    sharpes      = {k: max(results[k]["sharpe"], 0.01) for k in strats}
    total_sharpe = sum(sharpes.values())
    sharpe_weights = {k: v / total_sharpe for k, v in sharpes.items()}
    h_sharpe     = combine_strategies(strats, sharpe_weights)
    r_sharpe     = evaluate_strategy(h_sharpe, "Combined_Sharpe")
    _ = r_sharpe.pop("_daily_ret")

    print(f"\n{'='*60}\n  COMBINED STRATEGY PERFORMANCE\n{'='*60}")
    for name, r in [("Equal Weight", r_equal), ("Sharpe Weight", r_sharpe)]:
        print(f"  {name:<15} Return={r['annual_return']:7.2%}  Vol={r['annual_vol']:6.2%}  "
              f"Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_drawdown']:7.2%}")

    h_sharpe["model"] = "LightGBM"
    h_sharpe.to_parquet(HOLDINGS_PATH, index=False)
    print(f"\nSaved holdings: {HOLDINGS_PATH}")

    report = {
        "individual_strategies": results,
        "correlation_matrix":    corr.round(4).to_dict(),
        "combination_weights": {
            "equal":          equal_weights,
            "sharpe_weighted": {k: round(v, 4) for k, v in sharpe_weights.items()},
        },
        "combined_results": {
            "equal_weight":  r_equal,
            "sharpe_weight": r_sharpe,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
