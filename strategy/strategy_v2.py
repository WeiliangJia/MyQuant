"""
策略 V2：25 因子 + LightGBM + Ridge
====================================
标签：  未来5日超额收益（个股 - SPY）
模型：  LightGBM + Ridge（baseline 对照）
组合：  Top 20 等权做多，每5个交易日调仓
验证：  滚动窗口时序CV（2年训练 → 滚动）
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
PREDICTIONS_PATH = OUTPUT_DIR / "predictions_v2.parquet"
HOLDINGS_PATH = OUTPUT_DIR / "holdings_v2.parquet"
REPORT_PATH = OUTPUT_DIR / "strategy_v2_report.json"

# ── 参数 ──────────────────────────────────────────────
FORWARD_DAYS = 5
TRAIN_YEARS = 2
REBALANCE_DAYS = 5
TOP_K = 20

FEATURE_COLS = [
    "return_1d_zscore",
    "momentum_5d_zscore",
    "momentum_20d_zscore",
    "momentum_60d_zscore",
    # return_5d_reversal 和 momentum_5d 完全共线（r=-1），去掉
    "high_52w_dist_zscore",
    "volatility_20d_zscore",
    # volatility_60d 和 volatility_20d 高度相关（r=0.83），去掉
    "vol_ratio_20_60_zscore",
    "intraday_range_zscore",
    "volume_ratio_20d_zscore",
    # volume_ratio_5d 和 volume_ratio_20d 高度相关（r=0.80），去掉
    "volume_std_20d_zscore",
    "price_volume_corr_20d_zscore",
    "rsi_14_zscore",
    "ma_deviation_20d_zscore",
    "macd_signal_zscore",
    # bollinger_pos 和 rsi_14 高度相关（r=0.94），去掉
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
def rolling_train_predict(data: pd.DataFrame) -> pd.DataFrame:
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

    # 特征重要性
    fi = pd.Series(feature_importance_sum / max(n_windows, 1), index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)
    print(f"\n── LightGBM Feature Importance (avg gain) ──")
    for fname, fval in fi.items():
        print(f"  {fname:35s}  {fval:.1f}")

    predictions = pd.concat(all_preds, ignore_index=True)
    return predictions, fi


# ── 3. 组合构建 ──────────────────────────────────────
def build_portfolio(predictions: pd.DataFrame, model_col: str) -> pd.DataFrame:
    holdings_list = []
    for date, grp in predictions.groupby("Date"):
        top_k = grp.nlargest(TOP_K, model_col)
        weight = 1.0 / TOP_K
        for _, row in top_k.iterrows():
            holdings_list.append({
                "Date": date,
                "ticker": row["ticker"],
                "weight": weight,
                "pred_score": row[model_col],
                "actual_label": row["label"],
            })
    return pd.DataFrame(holdings_list)


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

    for model_col, model_name in [("pred_ridge", "Ridge"), ("pred_lgb", "LightGBM")]:
        print(f"\n── {model_name} Portfolio ──")
        h = build_portfolio(predictions, model_col)
        h["model"] = model_name
        all_holdings.append(h)
        metrics = evaluate(h, model_name)
        results[model_name] = metrics

        print(f"  Annual Return:  {metrics['annual_return']:.2%}")
        print(f"  Annual Vol:     {metrics['annual_volatility']:.2%}")
        print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate:       {metrics['win_rate']:.2%}")
        print(f"  Mean IC:        {metrics['mean_ic']:.4f}")
        print(f"  IC IR:          {metrics['ic_ir']:.4f}")

    holdings_all = pd.concat(all_holdings, ignore_index=True)
    holdings_all.to_parquet(HOLDINGS_PATH, index=False)
    print(f"\nSaved holdings: {HOLDINGS_PATH}")

    report = {
        "parameters": {
            "forward_days": FORWARD_DAYS,
            "train_years": TRAIN_YEARS,
            "rebalance_days": REBALANCE_DAYS,
            "top_k": TOP_K,
            "n_features": len(FEATURE_COLS),
            "features_used": FEATURE_COLS,
            "features_dropped": [
                "return_5d_reversal_zscore (r=-1.0 with momentum_5d)",
                "volatility_60d_zscore (r=0.83 with volatility_20d)",
                "volume_ratio_5d_zscore (r=0.80 with volume_ratio_20d)",
                "bollinger_pos_zscore (r=0.94 with rsi_14)",
            ],
            "lgb_params": LGB_PARAMS,
        },
        "results": results,
        "feature_importance_lgb": {
            k: round(float(v), 2) for k, v in feature_importance.items()
        },
        "v1_vs_v2_comparison_note": "Compare with strategy_v1_report.json",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
