"""
回测 V2：集成三层风控
=====================
在 V1 基础上加入 risk/risk_manager.py 的三层风控：
  1. 目标波动率缩放（12% 目标）
  2. 最大回撤熔断（15% / 25%）
  3. 行业集中度限制（30%）

未投资部分放入现金（无收益，保守假设）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from risk.risk_manager import RiskManager

# ── 路径 ──────────────────────────────────────────────
HOLDINGS_PATH = PROJECT_DIR / "strategy" / "holdings_multi.parquet"
PRICE_PATH = PROJECT_DIR / "data" / "data" / "processed" / "panel_aligned.parquet"
SPY_PATH = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"

NAV_PATH = SCRIPT_DIR / "nav_v2.parquet"
REPORT_PATH = SCRIPT_DIR / "backtest_v2_report.json"
YEARLY_PATH = SCRIPT_DIR / "backtest_v2_yearly.csv"
DRAWDOWN_PATH = SCRIPT_DIR / "backtest_v2_drawdowns.csv"
RISK_LOG_PATH = SCRIPT_DIR / "backtest_v2_risk_log.parquet"

# ── 参数 ──────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_BPS = 5
SLIPPAGE_BPS = 5
COST_PER_SIDE = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000
REBALANCE_DAYS = 5  # 多策略混合调仓频率


def load_data():
    holdings = pd.read_parquet(HOLDINGS_PATH)
    holdings = holdings[holdings["model"] == "LightGBM"].copy()
    holdings = holdings[["Date", "ticker", "weight"]].copy()

    prices = pd.read_parquet(PRICE_PATH)
    prices = prices[["Date", "ticker", "Open", "Close"]].copy()

    spy = pd.read_parquet(SPY_PATH)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy = spy[["Date", "Close"]].rename(columns={"Close": "spy_close"}).copy()

    return holdings, prices, spy


def run_backtest_with_risk(holdings, prices, spy):
    print("Running backtest with risk controls ...")

    # 构建快速查找字典（比 .loc 快很多）
    open_dict = {}
    close_dict = {}
    for _, row in prices.iterrows():
        key = (row["Date"], row["ticker"])
        open_dict[key] = row["Open"]
        close_dict[key] = row["Close"]

    all_dates = sorted(prices["Date"].unique())
    bt_start = holdings["Date"].min()
    bt_end = holdings["Date"].max()
    trade_dates = [d for d in all_dates if bt_start <= d <= bt_end]

    rebalance_dates = sorted(holdings["Date"].unique())[::REBALANCE_DAYS]
    rebalance_set = set(rebalance_dates)
    holdings_by_date = {
        date: grp.set_index("ticker")["weight"].to_dict()
        for date, grp in holdings.groupby("Date")
    }

    # 初始化
    rm = RiskManager()
    cash = INITIAL_CAPITAL
    positions: dict[str, float] = {}
    nav_records = []
    total_turnover = 0.0
    total_cost = 0.0
    n_rebalances = 0
    prev_nav = INITIAL_CAPITAL

    for i, date in enumerate(trade_dates):
        if i % 500 == 0:
            print(f"  Day {i}/{len(trade_dates)} ...")

        is_rebalance = date in rebalance_set

        # 日初估值
        port_value_open = cash
        for tkr, shares in positions.items():
            opn = open_dict.get((date, tkr))
            if opn is not None:
                port_value_open += shares * opn

        if is_rebalance and date in holdings_by_date:
            # ── 风控计算 ──
            scale, risk_detail = rm.compute_position_scale(port_value_open, date=date)

            # 原始目标权重
            raw_weights = holdings_by_date[date]

            # 行业集中度限制
            adjusted_weights = RiskManager.apply_sector_limit(raw_weights)

            # 波动率 + 回撤缩放
            scaled_weights = {tkr: w * scale for tkr, w in adjusted_weights.items()}

            # 目标持仓股数
            target_positions: dict[str, float] = {}
            for tkr, w in scaled_weights.items():
                opn = open_dict.get((date, tkr))
                if opn is not None and opn > 0:
                    target_positions[tkr] = (port_value_open * w) / opn

            # 交易执行
            all_tickers = set(list(positions.keys()) + list(target_positions.keys()))
            day_turnover = 0.0

            for tkr in all_tickers:
                old_shares = positions.get(tkr, 0.0)
                new_shares = target_positions.get(tkr, 0.0)
                delta_shares = new_shares - old_shares

                if abs(delta_shares) < 1e-6:
                    continue

                opn = open_dict.get((date, tkr))
                if opn is None:
                    continue

                trade_value = abs(delta_shares) * opn
                cost = trade_value * COST_PER_SIDE
                day_turnover += trade_value
                cash -= delta_shares * opn + cost
                total_cost += cost
                positions[tkr] = new_shares

            total_turnover += day_turnover
            n_rebalances += 1
            positions = {k: v for k, v in positions.items() if abs(v) > 1e-6}

        # ── 日终估值 ──
        port_value_eod = cash
        for tkr, shares in positions.items():
            cls = close_dict.get((date, tkr))
            if cls is not None:
                port_value_eod += shares * cls

        # 更新风控状态
        daily_ret = (port_value_eod - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        rm.add_return(daily_ret)
        rm.update_nav(port_value_eod, date)

        # 计算当前持仓占比
        stock_value = port_value_eod - cash
        invested_pct = stock_value / port_value_eod if port_value_eod > 0 else 0

        nav_records.append({
            "Date": date,
            "nav": port_value_eod,
            "cash": cash,
            "n_positions": len(positions),
            "is_rebalance": is_rebalance,
            "invested_pct": invested_pct,
            "daily_return": daily_ret,
        })

        prev_nav = port_value_eod

    nav_df = pd.DataFrame(nav_records)

    # 合并 SPY
    nav_df = nav_df.merge(spy, on="Date", how="left")
    nav_df["spy_close"] = nav_df["spy_close"].ffill()
    nav_df["spy_return"] = nav_df["spy_close"].pct_change().fillna(0)
    nav_df["spy_nav"] = INITIAL_CAPITAL * (1 + nav_df["spy_return"]).cumprod()
    nav_df["excess_return"] = nav_df["daily_return"] - nav_df["spy_return"]

    # 风控日志
    risk_log = pd.DataFrame(rm.risk_log)

    print(f"  Backtest period: {trade_dates[0].date()} ~ {trade_dates[-1].date()}")
    print(f"  Trading days: {len(trade_dates)}")
    print(f"  Rebalances: {n_rebalances}")
    print(f"  Total turnover: ${total_turnover:,.0f}")
    print(f"  Total cost: ${total_cost:,.0f}")
    print(f"  Avg invested: {nav_df['invested_pct'].mean():.1%}")

    return nav_df, risk_log, total_turnover, total_cost, n_rebalances


def compute_metrics(nav_df, total_turnover, total_cost, n_rebalances):
    ret = nav_df["daily_return"]
    excess = nav_df["excess_return"]
    n_days = len(ret)
    n_years = n_days / 252

    total_ret = nav_df["nav"].iloc[-1] / INITIAL_CAPITAL - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    rf_rate = 0.04
    sharpe_rf = (annual_ret - rf_rate) / annual_vol if annual_vol > 0 else 0

    spy_total_ret = nav_df["spy_nav"].iloc[-1] / INITIAL_CAPITAL - 1
    spy_annual_ret = (1 + spy_total_ret) ** (1 / n_years) - 1
    spy_annual_vol = nav_df["spy_return"].std() * np.sqrt(252)

    excess_annual_ret = annual_ret - spy_annual_ret
    excess_annual_vol = excess.std() * np.sqrt(252)
    info_ratio = excess_annual_ret / excess_annual_vol if excess_annual_vol > 0 else 0

    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax
    max_dd = drawdown.min()
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    win_rate = (ret > 0).mean()
    profit_days = ret[ret > 0]
    loss_days = ret[ret < 0]
    avg_win = profit_days.mean() if len(profit_days) > 0 else 0
    avg_loss = abs(loss_days.mean()) if len(loss_days) > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    avg_annual_turnover = total_turnover / n_years
    cost_drag = total_cost / INITIAL_CAPITAL / n_years

    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0

    avg_invested = nav_df["invested_pct"].mean()

    return {
        "backtest_period": f"{nav_df['Date'].iloc[0].date()} ~ {nav_df['Date'].iloc[-1].date()}",
        "trading_days": int(n_days),
        "years": round(n_years, 2),
        "initial_capital": INITIAL_CAPITAL,
        "final_nav": round(float(nav_df["nav"].iloc[-1]), 2),
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(annual_ret), 4),
        "annual_volatility": round(float(annual_vol), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "sharpe_rf_4pct": round(float(sharpe_rf), 4),
        "sortino_ratio": round(float(sortino), 4),
        "calmar_ratio": round(float(calmar), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "profit_loss_ratio": round(float(profit_loss_ratio), 4),
        "avg_invested_pct": round(float(avg_invested), 4),
        "benchmark_spy": {
            "total_return": round(float(spy_total_ret), 4),
            "annual_return": round(float(spy_annual_ret), 4),
            "annual_volatility": round(float(spy_annual_vol), 4),
        },
        "excess_vs_spy": {
            "excess_annual_return": round(float(excess_annual_ret), 4),
            "excess_annual_vol": round(float(excess_annual_vol), 4),
            "information_ratio": round(float(info_ratio), 4),
        },
        "costs": {
            "commission_bps": COMMISSION_BPS,
            "slippage_bps": SLIPPAGE_BPS,
            "total_cost_dollar": round(float(total_cost), 2),
            "annual_cost_drag": round(float(cost_drag), 4),
            "annual_turnover": round(float(avg_annual_turnover), 2),
            "n_rebalances": n_rebalances,
        },
        "risk_controls": {
            "target_vol": 0.12,
            "dd_level_1": "15% -> 50% position",
            "dd_level_2": "25% -> 20% position",
            "sector_max": "30%",
        },
    }


def compute_yearly(nav_df):
    nav_df = nav_df.copy()
    nav_df["year"] = nav_df["Date"].dt.year
    records = []
    for year, grp in nav_df.groupby("year"):
        ret = grp["daily_return"]
        spy_ret = grp["spy_return"]
        yr_ret = (1 + ret).prod() - 1
        yr_vol = ret.std() * np.sqrt(252)
        yr_sharpe = yr_ret / yr_vol if yr_vol > 0 else 0
        spy_yr_ret = (1 + spy_ret).prod() - 1
        excess = yr_ret - spy_yr_ret
        cummax = (1 + ret).cumprod().cummax()
        dd = ((1 + ret).cumprod() - cummax) / cummax
        max_dd = dd.min()
        avg_inv = grp["invested_pct"].mean()
        records.append({
            "year": int(year),
            "return": round(float(yr_ret), 4),
            "volatility": round(float(yr_vol), 4),
            "sharpe": round(float(yr_sharpe), 4),
            "max_drawdown": round(float(max_dd), 4),
            "spy_return": round(float(spy_yr_ret), 4),
            "excess_return": round(float(excess), 4),
            "win_rate": round(float((ret > 0).mean()), 4),
            "avg_invested": round(float(avg_inv), 4),
            "trading_days": int(len(grp)),
        })
    return pd.DataFrame(records)


def compute_drawdowns(nav_df, top_n=10):
    nav = nav_df.set_index("Date")["nav"]
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    in_dd = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    dd_groups = dd_groups[in_dd]
    records = []
    for group_id, grp_idx in dd_groups.groupby(dd_groups):
        dd_slice = drawdown.loc[grp_idx.index]
        trough_date = dd_slice.idxmin()
        peak_date = cummax.loc[:grp_idx.index[0]].idxmax()
        recovery_candidates = nav.loc[trough_date:]
        recovered = recovery_candidates[recovery_candidates >= cummax.loc[trough_date]]
        recovery_date = recovered.index[0] if len(recovered) > 0 else None
        records.append({
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "max_drawdown": round(float(dd_slice.min()), 4),
            "duration_to_trough_days": (trough_date - peak_date).days,
            "duration_to_recovery_days": (recovery_date - peak_date).days if recovery_date else None,
        })
    return pd.DataFrame(records).sort_values("max_drawdown").head(top_n).reset_index(drop=True)


def main():
    holdings, prices, spy = load_data()
    nav_df, risk_log, total_turnover, total_cost, n_rebalances = run_backtest_with_risk(
        holdings, prices, spy
    )

    nav_df.to_parquet(NAV_PATH, index=False)
    print(f"Saved NAV: {NAV_PATH}")

    if not risk_log.empty:
        risk_log.to_parquet(RISK_LOG_PATH, index=False)
        print(f"Saved risk log: {RISK_LOG_PATH}")

    metrics = compute_metrics(nav_df, total_turnover, total_cost, n_rebalances)

    print(f"\n{'='*55}")
    print(f"  BACKTEST V2 (with Risk Controls)")
    print(f"{'='*55}")
    print(f"  Period:           {metrics['backtest_period']}")
    print(f"  Initial Capital:  ${metrics['initial_capital']:,.0f}")
    print(f"  Final NAV:        ${metrics['final_nav']:,.0f}")
    print(f"  Total Return:     {metrics['total_return']:.2%}")
    print(f"  Annual Return:    {metrics['annual_return']:.2%}")
    print(f"  Annual Vol:       {metrics['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"  Sharpe (rf=4%):   {metrics['sharpe_rf_4pct']:.2f}")
    print(f"  Sortino Ratio:    {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:     {metrics['calmar_ratio']:.2f}")
    print(f"  Max Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:         {metrics['win_rate']:.2%}")
    print(f"  Profit/Loss:      {metrics['profit_loss_ratio']:.2f}")
    print(f"  Avg Invested:     {metrics['avg_invested_pct']:.1%}")
    print(f"  ── Benchmark (SPY) ──")
    print(f"  SPY Annual Ret:   {metrics['benchmark_spy']['annual_return']:.2%}")
    print(f"  Excess Return:    {metrics['excess_vs_spy']['excess_annual_return']:.2%}")
    print(f"  Info Ratio:       {metrics['excess_vs_spy']['information_ratio']:.2f}")
    print(f"  ── Costs ──")
    print(f"  Total Cost:       ${metrics['costs']['total_cost_dollar']:,.0f}")
    print(f"  Annual Cost Drag: {metrics['costs']['annual_cost_drag']:.2%}")

    yearly = compute_yearly(nav_df)
    yearly.to_csv(YEARLY_PATH, index=False)
    print(f"\n── Yearly Performance ──")
    print(yearly.to_string(index=False))

    drawdowns = compute_drawdowns(nav_df)
    drawdowns.to_csv(DRAWDOWN_PATH, index=False)
    print(f"\n── Top Drawdowns ──")
    print(drawdowns.to_string(index=False))

    # 风控生效统计
    if not risk_log.empty:
        print(f"\n── Risk Control Summary ──")
        print(f"  Avg vol_scale:    {risk_log['vol_scale'].mean():.3f}")
        print(f"  Avg dd_scale:     {risk_log['dd_scale'].mean():.3f}")
        print(f"  Avg final_scale:  {risk_log['final_scale'].mean():.3f}")
        print(f"  Min final_scale:  {risk_log['final_scale'].min():.3f}")
        print(f"  Times dd < 1.0:   {(risk_log['dd_scale'] < 1.0).sum()}")

    report = {
        "metrics": metrics,
        "yearly": yearly.to_dict(orient="records"),
        "top_drawdowns": drawdowns.to_dict(orient="records"),
        "risk_summary": {
            "avg_vol_scale": round(float(risk_log["vol_scale"].mean()), 4) if not risk_log.empty else None,
            "avg_dd_scale": round(float(risk_log["dd_scale"].mean()), 4) if not risk_log.empty else None,
            "avg_final_scale": round(float(risk_log["final_scale"].mean()), 4) if not risk_log.empty else None,
            "min_final_scale": round(float(risk_log["final_scale"].min()), 4) if not risk_log.empty else None,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
