"""
策略 V5：因子验证精简版
========================
基于 V4，根据因子检验结果移除 4 个无效因子（IC |t_NW| < 0.5）：

  volume_std_20d_zscore    IC t_NW = -0.13  (IC ≈ 0)
  amihud_illiq_20d_zscore  IC t_NW = +0.11  (IC ≈ 0)
  return_kurt_20d_zscore   IC t_NW = +0.30  (IC ≈ 0)
  return_skew_20d_zscore   IC t_NW = -0.48  (IC ≈ 0)

V4: 21 股票因子 + 8 跨资产 = 29 个特征
V5: 17 股票因子 + 8 跨资产 = 25 个特征（-4 无效因子）

其余参数（调仓周期、Top-K、换手控制）与 V4 完全一致，保证对比公平。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR  = SCRIPT_DIR.parent
FEATURE_PATH = PROJECT_DIR / "feature" / "factors_v3.parquet"
SPY_PATH     = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"
PREDICTIONS_PATH = SCRIPT_DIR / "predictions_v5.parquet"
HOLDINGS_PATH    = SCRIPT_DIR / "holdings_v5.parquet"
REPORT_PATH      = SCRIPT_DIR / "strategy_v5_report.json"

# ── 参数（与 V4 完全一致） ────────────────────────────
FORWARD_DAYS    = 5
TRAIN_YEARS     = 2
REBALANCE_DAYS  = 10
TOP_K           = 20
SELL_THRESHOLD  = 30
HOLDING_BONUS   = 0.002

# ── V5 特征集：删除 4 个无效股票因子 ─────────────────
STOCK_FEATURES = [
    # 动量/反转（IC t_NW 显著）
    "return_1d_zscore",
    "momentum_5d_zscore",
    "momentum_20d_zscore",
    "momentum_60d_zscore",
    "high_52w_dist_zscore",
    # 波动率（FM t_NW 显著）
    "volatility_20d_zscore",
    "vol_ratio_20_60_zscore",
    "intraday_range_zscore",
    # 量价（保留有信号的）
    "volume_ratio_20d_zscore",
    # volume_std_20d_zscore   ← 删除：IC t_NW = -0.13
    "price_volume_corr_20d_zscore",
    # 技术指标（IC t_NW 显著）
    "rsi_14_zscore",
    "ma_deviation_20d_zscore",
    "macd_signal_zscore",
    "atr_14_pct_zscore",
    # 微观结构
    "close_position_zscore",
    "gap_open_zscore",
    # amihud_illiq_20d_zscore ← 删除：IC t_NW = +0.11
    # 高阶统计（保留有信号的）
    # return_skew_20d_zscore  ← 删除：IC t_NW = -0.48
    # return_kurt_20d_zscore  ← 删除：IC t_NW = +0.30
    "downside_vol_20d_zscore",
]

CROSS_ASSET_FEATURES = [
    "vix_level_zscore",
    "vix_change_5d_zscore",
    "vix_term_structure_zscore",
    "tnx_level_zscore",
    "tnx_change_20d_zscore",
    "tnx_momentum_60d_zscore",
    "dxy_change_20d_zscore",
    "dxy_momentum_60d_zscore",
]

FEATURE_COLS = STOCK_FEATURES + CROSS_ASSET_FEATURES

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 300,
    "early_stopping_rounds": 30,
}


def build_labels(factors: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    spy = spy[["Date", "Close"]].copy().sort_values("Date")
    spy["spy_fwd_ret"] = spy["Close"].pct_change(FORWARD_DAYS, fill_method=None).shift(-FORWARD_DAYS)
    spy = spy[["Date", "spy_fwd_ret"]]

    factors = factors.sort_values(["ticker", "Date"]).copy()
    factors["stock_fwd_ret"] = (
        factors.groupby("ticker")["Close"]
        .transform(lambda s: s.pct_change(FORWARD_DAYS, fill_method=None).shift(-FORWARD_DAYS))
    )
    factors = factors.merge(spy, on="Date", how="left")
    factors["label"] = factors["stock_fwd_ret"] - factors["spy_fwd_ret"]

    n_before = len(factors)
    factors = factors.dropna(subset=["label"] + FEATURE_COLS).reset_index(drop=True)
    print(f"  Label built: {n_before} -> {len(factors)} rows")
    return factors


def rolling_train_predict(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("Rolling train & predict ...")
    dates      = sorted(data["Date"].unique())
    train_size = TRAIN_YEARS * 252

    all_preds = []
    n_windows = 0
    rebalance_indices = list(range(train_size, len(dates), REBALANCE_DAYS))
    print(f"  Total dates: {len(dates)}, rebalance points: {len(rebalance_indices)}")

    feature_importance_sum = np.zeros(len(FEATURE_COLS))

    for rb_idx in rebalance_indices:
        train_start_idx = max(0, rb_idx - train_size)
        train_dates     = dates[train_start_idx:rb_idx]
        test_end_idx    = min(rb_idx + REBALANCE_DAYS, len(dates))
        test_dates      = dates[rb_idx:test_end_idx]

        if len(test_dates) == 0:
            continue

        train = data[data["Date"].isin(train_dates)]
        test  = data[data["Date"].isin(test_dates)]

        if len(train) < 100 or len(test) == 0:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train["label"].values
        X_test  = test[FEATURE_COLS].values

        # Ridge
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_sc, y_train)
        pred_ridge = ridge.predict(X_test_sc)

        # LightGBM
        split_idx       = int(len(X_train) * 0.8)
        X_tr, X_val     = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val     = y_train[:split_idx], y_train[split_idx:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {k: v for k, v in LGB_PARAMS.items()
                  if k not in ("n_estimators", "early_stopping_rounds")}
        model = lgb.train(
            params, dtrain,
            num_boost_round=LGB_PARAMS["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_PARAMS["early_stopping_rounds"], verbose=False)],
        )
        pred_lgb = model.predict(X_test)
        feature_importance_sum += model.feature_importance(importance_type="gain")

        pred_df = test[["Date", "ticker", "label"]].copy()
        pred_df["pred_ridge"] = pred_ridge
        pred_df["pred_lgb"]   = pred_lgb
        all_preds.append(pred_df)
        n_windows += 1

    print(f"  Completed {n_windows} rolling windows")

    fi = pd.Series(feature_importance_sum / max(n_windows, 1), index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)
    print(f"\n-- Feature Importance (V5, {len(FEATURE_COLS)} features) --")
    for fname, fval in fi.items():
        tag = " [CROSS-ASSET]" if fname in CROSS_ASSET_FEATURES else ""
        print(f"  {fname:<45s}  {fval:8.1f}{tag}")

    predictions = pd.concat(all_preds, ignore_index=True)
    return predictions, fi


def build_portfolio_with_turnover_control(
    predictions: pd.DataFrame, model_col: str,
) -> tuple[pd.DataFrame, dict]:
    rebalance_dates = sorted(predictions["Date"].unique())[::REBALANCE_DAYS]

    current_holdings: set[str] = set()
    holdings_list    = []
    turnover_records = []

    for rb_date in rebalance_dates:
        day_preds = predictions[predictions["Date"] == rb_date].copy()
        if len(day_preds) == 0:
            continue

        day_preds["adjusted_score"] = day_preds[model_col]
        day_preds.loc[day_preds["ticker"].isin(current_holdings), "adjusted_score"] += HOLDING_BONUS

        day_preds = day_preds.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
        day_preds["rank"] = range(1, len(day_preds) + 1)

        keep = set(day_preds[
            (day_preds["ticker"].isin(current_holdings)) &
            (day_preds["rank"] <= SELL_THRESHOLD)
        ]["ticker"])

        slots_remaining = TOP_K - len(keep)
        new_candidates  = day_preds[
            (~day_preds["ticker"].isin(keep)) &
            (day_preds["rank"] <= TOP_K)
        ].head(max(0, slots_remaining))
        new_tickers  = set(new_candidates["ticker"])
        new_holdings = keep | new_tickers

        if len(new_holdings) < TOP_K:
            extra = day_preds[~day_preds["ticker"].isin(new_holdings)].head(TOP_K - len(new_holdings))
            new_holdings |= set(extra["ticker"])

        final_df     = day_preds[day_preds["ticker"].isin(new_holdings)].nlargest(TOP_K, "adjusted_score")
        new_holdings = set(final_df["ticker"])

        sold   = current_holdings - new_holdings
        bought = new_holdings - current_holdings
        turnover_records.append({
            "Date": rb_date,
            "sold": len(sold), "bought": len(bought),
            "turnover_pct": (len(sold) + len(bought)) / (2 * TOP_K) if TOP_K > 0 else 0,
        })

        current_holdings = new_holdings
        weight = 1.0 / len(new_holdings)
        window_dates = sorted(predictions[predictions["Date"] >= rb_date]["Date"].unique())[:REBALANCE_DAYS]

        for d in window_dates:
            day_data = predictions[predictions["Date"] == d]
            for tkr in new_holdings:
                row    = day_data[day_data["ticker"] == tkr]
                actual = float(row["label"].iloc[0])   if len(row) > 0 else 0.0
                pred   = float(row[model_col].iloc[0]) if len(row) > 0 else 0.0
                holdings_list.append({
                    "Date": d, "ticker": tkr, "weight": weight,
                    "pred_score": pred, "actual_label": actual,
                })

    holdings     = pd.DataFrame(holdings_list)
    turnover_df  = pd.DataFrame(turnover_records)
    avg_turnover = turnover_df["turnover_pct"].mean() if len(turnover_df) > 0 else 0

    return holdings, {
        "n_rebalances":      len(turnover_df),
        "avg_turnover_pct":  round(float(avg_turnover), 4),
    }


def evaluate(holdings: pd.DataFrame, model_name: str) -> dict:
    daily_port = (
        holdings.groupby("Date")
        .apply(lambda g: (g["weight"] * g["actual_label"]).sum())
        .sort_index()
    )
    daily_ret = daily_port / FORWARD_DAYS

    total_ret  = float((1 + daily_ret).prod() - 1)
    n_days     = max(len(daily_ret), 1)
    annual_ret = float((1 + total_ret) ** (252 / n_days) - 1)
    annual_vol = float(daily_ret.std() * np.sqrt(252))
    sharpe     = annual_ret / annual_vol if annual_vol > 0 else 0.0
    max_dd     = float((daily_ret.cumsum() - daily_ret.cumsum().cummax()).min())
    win_rate   = float((daily_ret > 0).mean())

    ic_series = (
        holdings.groupby("Date")
        .apply(lambda g: g["pred_score"].corr(g["actual_label"]) if len(g) > 3 else np.nan)
        .dropna()
    )
    mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
    ic_ir   = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0

    return {
        "model":            model_name,
        "total_return":     round(total_ret, 4),
        "annual_return":    round(annual_ret, 4),
        "annual_volatility":round(annual_vol, 4),
        "sharpe_ratio":     round(sharpe, 4),
        "max_drawdown":     round(max_dd, 4),
        "win_rate":         round(win_rate, 4),
        "mean_ic":          round(mean_ic, 4),
        "ic_ir":            round(ic_ir, 4),
        "trading_days":     int(n_days),
    }


def main() -> None:
    print("=" * 60)
    print("  STRATEGY V5: 25 Features (V4 minus 4 weak factors)")
    print(f"  Stock: {len(STOCK_FEATURES)}  Cross-Asset: {len(CROSS_ASSET_FEATURES)}  Total: {len(FEATURE_COLS)}")
    print("=" * 60)

    factors = pd.read_parquet(FEATURE_PATH)
    spy_raw = pd.read_parquet(SPY_PATH)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] for c in spy_raw.columns]

    print("\nBuilding labels ...")
    data = build_labels(factors, spy_raw)

    predictions, feature_importance = rolling_train_predict(data)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"\nSaved predictions: {PREDICTIONS_PATH}")

    results      = {}
    all_holdings = []

    for model_col, model_name in [("pred_ridge", "Ridge"), ("pred_lgb", "LightGBM")]:
        print(f"\n-- {model_name} Portfolio --")
        h, turnover_stats = build_portfolio_with_turnover_control(predictions, model_col)
        h["model"] = model_name
        all_holdings.append(h)
        metrics = evaluate(h, model_name)
        results[model_name] = metrics
        results[model_name]["turnover"] = turnover_stats

        print(f"  Annual Return:  {metrics['annual_return']:.2%}")
        print(f"  Annual Vol:     {metrics['annual_volatility']:.2%}")
        print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate:       {metrics['win_rate']:.2%}")
        print(f"  Mean IC:        {metrics['mean_ic']:.4f}")
        print(f"  Avg Turnover:   {turnover_stats['avg_turnover_pct']:.2%}")

    holdings_all = pd.concat(all_holdings, ignore_index=True)
    holdings_all.to_parquet(HOLDINGS_PATH, index=False)
    print(f"\nSaved holdings: {HOLDINGS_PATH}")

    report = {
        "version": "V5",
        "removed_factors": [
            "volume_std_20d_zscore (IC t_NW=-0.13)",
            "amihud_illiq_20d_zscore (IC t_NW=+0.11)",
            "return_kurt_20d_zscore (IC t_NW=+0.30)",
            "return_skew_20d_zscore (IC t_NW=-0.48)",
        ],
        "parameters": {
            "forward_days":    FORWARD_DAYS,
            "train_years":     TRAIN_YEARS,
            "rebalance_days":  REBALANCE_DAYS,
            "top_k":           TOP_K,
            "sell_threshold":  SELL_THRESHOLD,
            "holding_bonus":   HOLDING_BONUS,
            "n_features_total":       len(FEATURE_COLS),
            "n_stock_features":       len(STOCK_FEATURES),
            "n_cross_asset_features": len(CROSS_ASSET_FEATURES),
            "stock_features":         STOCK_FEATURES,
            "cross_asset_features":   CROSS_ASSET_FEATURES,
        },
        "results": results,
        "feature_importance_lgb": {k: round(float(v), 2) for k, v in feature_importance.items()},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
