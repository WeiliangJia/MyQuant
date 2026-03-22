"""
策略 V3：降换手优化
===================
在 V2 基础上，三招降换手：
  1. 持仓惩罚（holding bonus）：已持有股票预测分 +bonus，减少不必要换仓
  2. 缓冲带（buffer band）：只有排名掉出 Top 30 才卖出（买入 Top20，卖出阈值 Top30）
  3. 调仓频率：5天 → 10天

标签/模型/因子不变，只改组合构建层。
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
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FEATURE_PATH = PROJECT_DIR / "feature" / "factors_v2.parquet"
SPY_PATH = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"
OUTPUT_DIR = SCRIPT_DIR
PREDICTIONS_PATH = OUTPUT_DIR / "predictions_v3.parquet"
HOLDINGS_PATH = OUTPUT_DIR / "holdings_v3.parquet"
REPORT_PATH = OUTPUT_DIR / "strategy_v3_report.json"

# ── 参数 ──────────────────────────────────────────────
FORWARD_DAYS = 5
TRAIN_YEARS = 2
REBALANCE_DAYS = 10         # 改：5 → 10 天调仓
TOP_K = 20                  # 买入阈值
SELL_THRESHOLD = 30         # 缓冲带：掉出 Top30 才卖
HOLDING_BONUS = 0.002       # 持仓加分：已持有股票预测分 +0.2%

FEATURE_COLS = [
    "return_1d_zscore",
    "momentum_5d_zscore",
    "momentum_20d_zscore",
    "momentum_60d_zscore",
    "high_52w_dist_zscore",
    "volatility_20d_zscore",
    "vol_ratio_20_60_zscore",
    "intraday_range_zscore",
    "volume_ratio_20d_zscore",
    "volume_std_20d_zscore",
    "price_volume_corr_20d_zscore",
    "rsi_14_zscore",
    "ma_deviation_20d_zscore",
    "macd_signal_zscore",
    "atr_14_pct_zscore",
    "close_position_zscore",
    "gap_open_zscore",
    "amihud_illiq_20d_zscore",
    "return_skew_20d_zscore",
    "return_kurt_20d_zscore",
    "downside_vol_20d_zscore",
]

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


# ── 1. 构造标签 ──────────────────────────────────────
def build_labels(factors: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    print("Building labels ...")
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
    factors = factors.dropna(subset=["label"]).reset_index(drop=True)
    print(f"  Label built: {n_before} -> {len(factors)} rows")
    return factors


# ── 2. 滚动窗口训练 & 预测 ────────────────────────────
def rolling_train_predict(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("Rolling train & predict ...")
    dates = sorted(data["Date"].unique())
    train_size = TRAIN_YEARS * 252

    all_preds = []
    n_windows = 0
    rebalance_indices = list(range(train_size, len(dates), REBALANCE_DAYS))
    print(f"  Total dates: {len(dates)}, rebalance points: {len(rebalance_indices)}")

    feature_importance_sum = np.zeros(len(FEATURE_COLS))

    for rb_idx in rebalance_indices:
        train_start_idx = max(0, rb_idx - train_size)
        train_dates = dates[train_start_idx:rb_idx]
        test_end_idx = min(rb_idx + REBALANCE_DAYS, len(dates))
        test_dates = dates[rb_idx:test_end_idx]

        if len(test_dates) == 0:
            continue

        train = data[data["Date"].isin(train_dates)]
        test = data[data["Date"].isin(test_dates)]

        if len(train) < 100 or len(test) == 0:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train["label"].values
        X_test = test[FEATURE_COLS].values

        # ── Ridge ──
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_sc, y_train)
        pred_ridge = ridge.predict(X_test_sc)

        # ── LightGBM ──
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

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
        pred_df["pred_lgb"] = pred_lgb
        all_preds.append(pred_df)
        n_windows += 1

    print(f"  Completed {n_windows} rolling windows")

    fi = pd.Series(feature_importance_sum / max(n_windows, 1), index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)

    predictions = pd.concat(all_preds, ignore_index=True)
    return predictions, fi


# ── 3. 组合构建（带持仓惩罚 + 缓冲带） ────────────────
def build_portfolio_with_turnover_control(
    predictions: pd.DataFrame,
    model_col: str,
) -> tuple[pd.DataFrame, dict]:
    """
    带换手控制的组合构建：
    - 已持有的股票给 holding_bonus 加分
    - 缓冲带：持有股票只要还在 Top SELL_THRESHOLD 内就保留
    - 新买入只从 Top K 中选
    """
    # 以调仓日为单位（每个调仓日有一组预测，持续 REBALANCE_DAYS 天）
    rebalance_dates = sorted(predictions["Date"].unique())[::REBALANCE_DAYS]

    current_holdings: set[str] = set()
    holdings_list = []
    turnover_records = []

    for rb_date in rebalance_dates:
        # 取调仓日的预测
        day_preds = predictions[predictions["Date"] == rb_date].copy()
        if len(day_preds) == 0:
            continue

        # 持仓加分
        day_preds["adjusted_score"] = day_preds[model_col]
        day_preds.loc[day_preds["ticker"].isin(current_holdings), "adjusted_score"] += HOLDING_BONUS

        # 排名
        day_preds = day_preds.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
        day_preds["rank"] = range(1, len(day_preds) + 1)

        # 缓冲带逻辑
        # 保留：已持有且排名在 SELL_THRESHOLD 内
        keep = set(day_preds[
            (day_preds["ticker"].isin(current_holdings)) &
            (day_preds["rank"] <= SELL_THRESHOLD)
        ]["ticker"])

        # 还需要多少新股票
        slots_remaining = TOP_K - len(keep)

        # 从未持有的 Top 排名中选新股票
        new_candidates = day_preds[
            (~day_preds["ticker"].isin(keep)) &
            (day_preds["rank"] <= TOP_K)
        ].head(max(0, slots_remaining))
        new_tickers = set(new_candidates["ticker"])

        # 最终持仓
        new_holdings = keep | new_tickers

        # 如果还没满 TOP_K（极端情况），从排名最高的补
        if len(new_holdings) < TOP_K:
            extra = day_preds[~day_preds["ticker"].isin(new_holdings)].head(TOP_K - len(new_holdings))
            new_holdings |= set(extra["ticker"])

        # 截取到 TOP_K
        # 按 adjusted_score 排序，取前 TOP_K
        final_df = day_preds[day_preds["ticker"].isin(new_holdings)].nlargest(TOP_K, "adjusted_score")
        new_holdings = set(final_df["ticker"])

        # 计算换手
        sold = current_holdings - new_holdings
        bought = new_holdings - current_holdings
        turnover_records.append({
            "Date": rb_date,
            "held": len(current_holdings),
            "kept": len(current_holdings & new_holdings),
            "sold": len(sold),
            "bought": len(bought),
            "turnover_pct": (len(sold) + len(bought)) / (2 * TOP_K) if TOP_K > 0 else 0,
        })

        current_holdings = new_holdings
        weight = 1.0 / len(new_holdings)

        # 填充这个调仓窗口内所有交易日的持仓
        window_dates = sorted(predictions[
            (predictions["Date"] >= rb_date)
        ]["Date"].unique())[:REBALANCE_DAYS]

        for d in window_dates:
            day_data = predictions[predictions["Date"] == d]
            for tkr in new_holdings:
                row = day_data[day_data["ticker"] == tkr]
                actual = float(row["label"].iloc[0]) if len(row) > 0 else 0.0
                pred = float(row[model_col].iloc[0]) if len(row) > 0 else 0.0
                holdings_list.append({
                    "Date": d,
                    "ticker": tkr,
                    "weight": weight,
                    "pred_score": pred,
                    "actual_label": actual,
                })

    holdings = pd.DataFrame(holdings_list)
    turnover_df = pd.DataFrame(turnover_records)

    avg_turnover = turnover_df["turnover_pct"].mean() if len(turnover_df) > 0 else 0
    avg_sold = turnover_df["sold"].mean() if len(turnover_df) > 0 else 0

    turnover_stats = {
        "n_rebalances": len(turnover_df),
        "avg_sold_per_rebalance": round(float(avg_sold), 2),
        "avg_turnover_pct": round(float(avg_turnover), 4),
    }

    return holdings, turnover_stats


# ── 4. 评估 ──────────────────────────────────────────
def evaluate(holdings: pd.DataFrame, model_name: str) -> dict:
    daily_port = (
        holdings.groupby("Date")
        .apply(lambda g: (g["weight"] * g["actual_label"]).sum())
        .sort_index()
    )
    daily_ret = daily_port / FORWARD_DAYS

    total_ret = float((1 + daily_ret).prod() - 1)
    n_days = max(len(daily_ret), 1)
    annual_ret = float((1 + total_ret) ** (252 / n_days) - 1)
    annual_vol = float(daily_ret.std() * np.sqrt(252))
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
    max_dd = float((daily_ret.cumsum() - daily_ret.cumsum().cummax()).min())
    win_rate = float((daily_ret > 0).mean())

    ic_series = (
        holdings.groupby("Date")
        .apply(lambda g: g["pred_score"].corr(g["actual_label"]) if len(g) > 3 else np.nan)
        .dropna()
    )
    mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
    ic_ir = float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0.0

    return {
        "model": model_name,
        "total_return": round(total_ret, 4),
        "annual_return": round(annual_ret, 4),
        "annual_volatility": round(annual_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "mean_ic": round(mean_ic, 4),
        "ic_ir": round(ic_ir, 4),
        "trading_days": int(n_days),
    }


# ── 主流程 ────────────────────────────────────────────
def main() -> None:
    factors = pd.read_parquet(FEATURE_PATH)
    spy_raw = pd.read_parquet(SPY_PATH)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] if isinstance(c, tuple) else c for c in spy_raw.columns]

    data = build_labels(factors, spy_raw)
    predictions, feature_importance = rolling_train_predict(data)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"\nSaved predictions: {PREDICTIONS_PATH}")

    results = {}
    all_holdings = []
    all_turnover_stats = {}

    for model_col, model_name in [("pred_ridge", "Ridge"), ("pred_lgb", "LightGBM")]:
        print(f"\n── {model_name} Portfolio (Turnover-Controlled) ──")
        h, turnover_stats = build_portfolio_with_turnover_control(predictions, model_col)
        h["model"] = model_name
        all_holdings.append(h)
        all_turnover_stats[model_name] = turnover_stats
        metrics = evaluate(h, model_name)
        results[model_name] = metrics

        print(f"  Annual Return:  {metrics['annual_return']:.2%}")
        print(f"  Annual Vol:     {metrics['annual_volatility']:.2%}")
        print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate:       {metrics['win_rate']:.2%}")
        print(f"  Mean IC:        {metrics['mean_ic']:.4f}")
        print(f"  IC IR:          {metrics['ic_ir']:.4f}")
        print(f"  Avg sold/rebalance: {turnover_stats['avg_sold_per_rebalance']}")
        print(f"  Avg turnover:   {turnover_stats['avg_turnover_pct']:.2%}")

    holdings_all = pd.concat(all_holdings, ignore_index=True)
    holdings_all.to_parquet(HOLDINGS_PATH, index=False)
    print(f"\nSaved holdings: {HOLDINGS_PATH}")

    report = {
        "parameters": {
            "forward_days": FORWARD_DAYS,
            "train_years": TRAIN_YEARS,
            "rebalance_days": REBALANCE_DAYS,
            "top_k": TOP_K,
            "sell_threshold": SELL_THRESHOLD,
            "holding_bonus": HOLDING_BONUS,
            "n_features": len(FEATURE_COLS),
            "features_used": FEATURE_COLS,
            "lgb_params": LGB_PARAMS,
            "turnover_control": {
                "holding_bonus": HOLDING_BONUS,
                "sell_threshold": SELL_THRESHOLD,
                "rebalance_days": REBALANCE_DAYS,
            },
        },
        "results": results,
        "turnover_stats": all_turnover_stats,
        "feature_importance_lgb": {
            k: round(float(v), 2) for k, v in feature_importance.items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
