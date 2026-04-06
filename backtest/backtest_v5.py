"""
回测 V5：三层风控 + V4 vs V5 对比
===================================
与 backtest_v2.py 完全相同，仅输入换为 holdings_multi_v5.parquet。
最后打印 V4（29特征）vs V5（25特征）的完整对比表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from risk.risk_manager import RiskManager

HOLDINGS_PATH  = PROJECT_DIR / "strategy" / "holdings_multi_v5.parquet"
PRICE_PATH     = PROJECT_DIR / "data" / "data" / "processed" / "panel_aligned.parquet"
SPY_PATH       = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"

NAV_PATH       = SCRIPT_DIR / "nav_v5.parquet"
REPORT_PATH    = SCRIPT_DIR / "backtest_v5_report.json"
YEARLY_PATH    = SCRIPT_DIR / "backtest_v5_yearly.csv"
DRAWDOWN_PATH  = SCRIPT_DIR / "backtest_v5_drawdowns.csv"
RISK_LOG_PATH  = SCRIPT_DIR / "backtest_v5_risk_log.parquet"

# V4 report path (for comparison)
V4_REPORT_PATH = SCRIPT_DIR / "backtest_v2_report.json"

INITIAL_CAPITAL = 1_000_000.0
COMMISSION_BPS  = 5
SLIPPAGE_BPS    = 5
COST_PER_SIDE   = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000
REBALANCE_DAYS  = 5


def load_data():
    holdings = pd.read_parquet(HOLDINGS_PATH)
    holdings = holdings[holdings["model"] == "LightGBM"].copy()
    holdings = holdings[["Date", "ticker", "weight"]].copy()

    prices = pd.read_parquet(PRICE_PATH)
    prices = prices[["Date", "ticker", "Open", "Close"]].copy()

    spy = pd.read_parquet(SPY_PATH)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] for c in spy.columns]
    spy = spy[["Date", "Close"]].rename(columns={"Close": "spy_close"}).copy()

    return holdings, prices, spy


def run_backtest(holdings, prices, spy):
    print("Running backtest with risk controls ...")

    open_dict  = {}
    close_dict = {}
    for _, row in prices.iterrows():
        key = (row["Date"], row["ticker"])
        open_dict[key]  = row["Open"]
        close_dict[key] = row["Close"]

    all_dates   = sorted(prices["Date"].unique())
    bt_start    = holdings["Date"].min()
    bt_end      = holdings["Date"].max()
    trade_dates = [d for d in all_dates if bt_start <= d <= bt_end]

    rebalance_dates = sorted(holdings["Date"].unique())[::REBALANCE_DAYS]
    rebalance_set   = set(rebalance_dates)
    holdings_by_date = {
        date: grp.set_index("ticker")["weight"].to_dict()
        for date, grp in holdings.groupby("Date")
    }

    rm         = RiskManager()
    cash       = INITIAL_CAPITAL
    positions: dict[str, float] = {}
    nav_records    = []
    total_turnover = 0.0
    total_cost     = 0.0
    n_rebalances   = 0
    prev_nav       = INITIAL_CAPITAL

    for i, date in enumerate(trade_dates):
        if i % 500 == 0:
            print(f"  Day {i}/{len(trade_dates)} ...")

        is_rebalance = date in rebalance_set

        port_value_open = cash
        for tkr, shares in positions.items():
            opn = open_dict.get((date, tkr))
            if opn is not None:
                port_value_open += shares * opn

        if is_rebalance and date in holdings_by_date:
            scale, _ = rm.compute_position_scale(port_value_open, date=date)
            raw_weights     = holdings_by_date[date]
            adjusted_weights = RiskManager.apply_sector_limit(raw_weights)
            scaled_weights  = {tkr: w * scale for tkr, w in adjusted_weights.items()}

            target_positions: dict[str, float] = {}
            for tkr, w in scaled_weights.items():
                opn = open_dict.get((date, tkr))
                if opn is not None and opn > 0:
                    target_positions[tkr] = (port_value_open * w) / opn

            all_tickers = set(list(positions.keys()) + list(target_positions.keys()))
            day_turnover = 0.0

            for tkr in all_tickers:
                old_shares = positions.get(tkr, 0.0)
                new_shares = target_positions.get(tkr, 0.0)
                delta      = new_shares - old_shares
                if abs(delta) < 1e-6:
                    continue
                opn = open_dict.get((date, tkr))
                if opn is None:
                    continue
                trade_value   = abs(delta) * opn
                cost          = trade_value * COST_PER_SIDE
                day_turnover += trade_value
                cash         -= delta * opn + cost
                total_cost   += cost
                positions[tkr] = new_shares

            total_turnover += day_turnover
            n_rebalances   += 1
            positions = {k: v for k, v in positions.items() if abs(v) > 1e-6}

        port_value_eod = cash
        for tkr, shares in positions.items():
            cls = close_dict.get((date, tkr))
            if cls is not None:
                port_value_eod += shares * cls

        daily_ret = (port_value_eod - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        rm.add_return(daily_ret)
        rm.update_nav(port_value_eod, date)

        stock_value  = port_value_eod - cash
        invested_pct = stock_value / port_value_eod if port_value_eod > 0 else 0

        nav_records.append({
            "Date": date, "nav": port_value_eod,
            "cash": cash,  "n_positions": len(positions),
            "is_rebalance": is_rebalance,
            "invested_pct": invested_pct,
            "daily_return": daily_ret,
        })
        prev_nav = port_value_eod

    nav_df = pd.DataFrame(nav_records)
    nav_df = nav_df.merge(spy, on="Date", how="left")
    nav_df["spy_close"]  = nav_df["spy_close"].ffill()
    nav_df["spy_return"] = nav_df["spy_close"].pct_change().fillna(0)
    nav_df["spy_nav"]    = INITIAL_CAPITAL * (1 + nav_df["spy_return"]).cumprod()
    nav_df["excess_return"] = nav_df["daily_return"] - nav_df["spy_return"]

    risk_log = pd.DataFrame(rm.risk_log)

    print(f"  Period:      {trade_dates[0].date()} ~ {trade_dates[-1].date()}")
    print(f"  Trade days:  {len(trade_dates)}")
    print(f"  Rebalances:  {n_rebalances}")
    print(f"  Total cost:  ${total_cost:,.0f}")

    return nav_df, risk_log, total_turnover, total_cost, n_rebalances


def compute_metrics(nav_df, total_turnover, total_cost, n_rebalances):
    ret    = nav_df["daily_return"]
    excess = nav_df["excess_return"]
    n_days = len(ret)
    n_years = n_days / 252

    total_ret  = nav_df["nav"].iloc[-1] / INITIAL_CAPITAL - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe     = annual_ret / annual_vol if annual_vol > 0 else 0
    sharpe_rf  = (annual_ret - 0.04) / annual_vol if annual_vol > 0 else 0

    spy_total_ret  = nav_df["spy_nav"].iloc[-1] / INITIAL_CAPITAL - 1
    spy_annual_ret = (1 + spy_total_ret) ** (1 / n_years) - 1
    spy_annual_vol = nav_df["spy_return"].std() * np.sqrt(252)

    excess_annual_ret = annual_ret - spy_annual_ret
    excess_annual_vol = excess.std() * np.sqrt(252)
    info_ratio = excess_annual_ret / excess_annual_vol if excess_annual_vol > 0 else 0

    cummax   = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax
    max_dd   = drawdown.min()
    calmar   = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    win_rate    = (ret > 0).mean()
    profit_days = ret[ret > 0]
    loss_days   = ret[ret < 0]
    avg_win     = profit_days.mean() if len(profit_days) > 0 else 0
    avg_loss    = abs(loss_days.mean()) if len(loss_days) > 0 else 0
    pl_ratio    = avg_win / avg_loss if avg_loss > 0 else 0

    downside     = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino      = annual_ret / downside_vol if downside_vol > 0 else 0

    avg_invested = nav_df["invested_pct"].mean()
    cost_drag    = total_cost / INITIAL_CAPITAL / n_years

    return {
        "backtest_period":  f"{nav_df['Date'].iloc[0].date()} ~ {nav_df['Date'].iloc[-1].date()}",
        "trading_days":     int(n_days),
        "years":            round(n_years, 2),
        "final_nav":        round(float(nav_df["nav"].iloc[-1]), 2),
        "total_return":     round(float(total_ret), 4),
        "annual_return":    round(float(annual_ret), 4),
        "annual_volatility":round(float(annual_vol), 4),
        "sharpe_ratio":     round(float(sharpe), 4),
        "sharpe_rf_4pct":   round(float(sharpe_rf), 4),
        "sortino_ratio":    round(float(sortino), 4),
        "calmar_ratio":     round(float(calmar), 4),
        "max_drawdown":     round(float(max_dd), 4),
        "win_rate":         round(float(win_rate), 4),
        "profit_loss_ratio":round(float(pl_ratio), 4),
        "avg_invested_pct": round(float(avg_invested), 4),
        "benchmark_spy": {
            "total_return":    round(float(spy_total_ret), 4),
            "annual_return":   round(float(spy_annual_ret), 4),
            "annual_volatility":round(float(spy_annual_vol), 4),
        },
        "excess_vs_spy": {
            "excess_annual_return": round(float(excess_annual_ret), 4),
            "excess_annual_vol":    round(float(excess_annual_vol), 4),
            "information_ratio":    round(float(info_ratio), 4),
        },
        "costs": {
            "total_cost_dollar":  round(float(total_cost), 2),
            "annual_cost_drag":   round(float(cost_drag), 4),
            "annual_turnover":    round(float(total_turnover / n_years), 2),
            "n_rebalances":       n_rebalances,
        },
    }


def compute_yearly(nav_df):
    nav_df = nav_df.copy()
    nav_df["year"] = nav_df["Date"].dt.year
    records = []
    for year, grp in nav_df.groupby("year"):
        ret     = grp["daily_return"]
        spy_ret = grp["spy_return"]
        yr_ret  = (1 + ret).prod() - 1
        yr_vol  = ret.std() * np.sqrt(252)
        cum = (1 + ret).cumprod()
        yr_dd = float((cum / cum.cummax() - 1).min())
        records.append({
            "year":          int(year),
            "return":        round(float(yr_ret), 4),
            "volatility":    round(float(yr_vol), 4),
            "sharpe":        round(float(yr_ret / yr_vol if yr_vol > 0 else 0), 4),
            "max_drawdown":  round(yr_dd, 4),
            "spy_return":    round(float((1 + spy_ret).prod() - 1), 4),
            "excess_return": round(float(yr_ret - (1 + spy_ret).prod() + 1), 4),
            "win_rate":      round(float((ret > 0).mean()), 4),
            "trading_days":  int(len(grp)),
        })
    return pd.DataFrame(records)


def compute_drawdowns(nav_df, top_n=10):
    nav     = nav_df.set_index("Date")["nav"]
    cummax  = nav.cummax()
    drawdown = (nav - cummax) / cummax
    in_dd   = drawdown < 0
    dd_groups = (~in_dd).cumsum()[in_dd]
    records = []
    for _, grp_idx in dd_groups.groupby(dd_groups):
        dd_slice      = drawdown.loc[grp_idx.index]
        trough_date   = dd_slice.idxmin()
        peak_date     = cummax.loc[:grp_idx.index[0]].idxmax()
        recovered     = nav.loc[trough_date:][nav.loc[trough_date:] >= cummax.loc[trough_date]]
        recovery_date = recovered.index[0] if len(recovered) > 0 else None
        records.append({
            "peak_date":               peak_date,
            "trough_date":             trough_date,
            "recovery_date":           recovery_date,
            "max_drawdown":            round(float(dd_slice.min()), 4),
            "duration_to_trough_days": (trough_date - peak_date).days,
            "duration_to_recovery_days": (recovery_date - peak_date).days if recovery_date else None,
        })
    return pd.DataFrame(records).sort_values("max_drawdown").head(top_n).reset_index(drop=True)


# ── V4 vs V5 对比 ─────────────────────────────────────
def print_comparison(v5_metrics: dict) -> None:
    """并排打印 V4（29特征）和 V5（25特征）的回测结果。"""
    if not V4_REPORT_PATH.exists():
        print("V4 report not found, skip comparison.")
        return

    v4_m = json.loads(V4_REPORT_PATH.read_text(encoding="utf-8"))["metrics"]
    v5_m = v5_metrics

    def fmt(val, spec):
        if val is None:
            return "    N/A"
        if spec == "pct":
            return f"{val:.2%}"
        if spec == "f2":
            return f"{val:.2f}"
        if spec == "f3":
            return f"{val:.3f}"
        return str(val)

    def diff_str(v4, v5, higher_is_better=True):
        if v4 is None or v5 is None:
            return ""
        d = v5 - v4
        sign = "+" if d >= 0 else ""
        arrow = " ^" if (d > 0) == higher_is_better else " v"
        if abs(d) < 1e-6:
            arrow = " ="
        return f"{sign}{d:.4f}{arrow}"

    rows = [
        ("Annual Return",     "annual_return",   "pct",  True),
        ("Annual Volatility", "annual_volatility","pct",  False),
        ("Sharpe Ratio",      "sharpe_ratio",    "f3",   True),
        ("Sharpe (rf=4%)",    "sharpe_rf_4pct",  "f3",   True),
        ("Sortino Ratio",     "sortino_ratio",   "f2",   True),
        ("Calmar Ratio",      "calmar_ratio",    "f2",   True),
        ("Max Drawdown",      "max_drawdown",    "pct",  False),
        ("Win Rate",          "win_rate",        "pct",  True),
        ("Profit/Loss Ratio", "profit_loss_ratio","f2",  True),
        ("SPY Annual Return", None,              "pct",  None),
        ("Excess vs SPY",     None,              "pct",  True),
        ("Info Ratio",        None,              "f3",   True),
        ("Annual Cost Drag",  None,              "pct",  False),
    ]

    print(f"\n{'='*65}")
    print(f"  COMPARISON: V4 (29 features) vs V5 (25 features, -4 weak)")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'V4':>12} {'V5':>12} {'Diff (V5-V4)':>14}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*14}")

    for label, key, spec, higher in rows:
        if key is not None:
            v4_val = v4_m.get(key)
            v5_val = v5_m.get(key)
        elif label == "SPY Annual Return":
            v4_val = v4_m["benchmark_spy"]["annual_return"]
            v5_val = v5_m["benchmark_spy"]["annual_return"]
        elif label == "Excess vs SPY":
            v4_val = v4_m["excess_vs_spy"]["excess_annual_return"]
            v5_val = v5_m["excess_vs_spy"]["excess_annual_return"]
        elif label == "Info Ratio":
            v4_val = v4_m["excess_vs_spy"]["information_ratio"]
            v5_val = v5_m["excess_vs_spy"]["information_ratio"]
        elif label == "Annual Cost Drag":
            v4_val = v4_m["costs"]["annual_cost_drag"]
            v5_val = v5_m["costs"]["annual_cost_drag"]
        else:
            v4_val = v5_val = None

        d_str = diff_str(v4_val, v5_val, higher) if higher is not None else ""
        print(f"  {label:<25} {fmt(v4_val, spec):>12} {fmt(v5_val, spec):>12} {d_str:>14}")

    print(f"\n  Feature count:            {'29':>12} {'25':>12} {'  -4 (removed)':>14}")
    print(f"{'='*65}")
    print(f"  ^ = V5 improved  v = V5 worse  = = same")


def main():
    print("=" * 60)
    print("  BACKTEST V5 (Risk Controls + V4 vs V5 Comparison)")
    print("=" * 60)

    holdings, prices, spy = load_data()
    nav_df, risk_log, total_turnover, total_cost, n_rebalances = run_backtest(
        holdings, prices, spy
    )

    nav_df.to_parquet(NAV_PATH, index=False)
    if not risk_log.empty:
        risk_log.to_parquet(RISK_LOG_PATH, index=False)

    metrics = compute_metrics(nav_df, total_turnover, total_cost, n_rebalances)

    print(f"\n{'='*55}")
    print(f"  BACKTEST V5 RESULTS")
    print(f"{'='*55}")
    print(f"  Period:         {metrics['backtest_period']}")
    print(f"  Final NAV:      ${metrics['final_nav']:,.0f}")
    print(f"  Annual Return:  {metrics['annual_return']:.2%}")
    print(f"  Annual Vol:     {metrics['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino:        {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:       {metrics['win_rate']:.2%}")
    print(f"  SPY Annual:     {metrics['benchmark_spy']['annual_return']:.2%}")
    print(f"  Excess Return:  {metrics['excess_vs_spy']['excess_annual_return']:.2%}")
    print(f"  Info Ratio:     {metrics['excess_vs_spy']['information_ratio']:.3f}")
    print(f"  Cost Drag:      {metrics['costs']['annual_cost_drag']:.2%}")

    yearly = compute_yearly(nav_df)
    yearly.to_csv(YEARLY_PATH, index=False)
    print(f"\n-- Yearly Performance --")
    print(yearly.to_string(index=False))

    drawdowns = compute_drawdowns(nav_df)
    drawdowns.to_csv(DRAWDOWN_PATH, index=False)

    report = {
        "version": "V5",
        "metrics": metrics,
        "yearly":  yearly.to_dict(orient="records"),
        "top_drawdowns": drawdowns.to_dict(orient="records"),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSaved report: {REPORT_PATH}")

    # ── V4 vs V5 对比表 ──
    print_comparison(metrics)


if __name__ == "__main__":
    main()
