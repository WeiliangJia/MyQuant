"""
回测模块 V1：基于持仓表 + 真实价格的完整回测
=============================================
输入：  strategy/holdings_v2.parquet（模型产出的每日持仓）
        data/data/processed/panel_aligned.parquet（真实价格）
        data/data/raw/SPY.parquet（基准）

模拟真实交易环节：
  - 用次日 Open 价成交（不能用当日 Close，那是信号产生时刻）
  - 交易成本：单边 5bp（万五）
  - 滑点：单边 5bp
  - 每 5 个交易日调仓
  - 初始资金 $1,000,000

输出：
  - 净值曲线（含/不含成本）
  - 完整绩效指标
  - 分年度归因
  - 回撤分析
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
HOLDINGS_PATH = PROJECT_DIR / "strategy" / "holdings_v3.parquet"
PRICE_PATH = PROJECT_DIR / "data" / "data" / "processed" / "panel_aligned.parquet"
SPY_PATH = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"

NAV_PATH = SCRIPT_DIR / "nav_v1.parquet"
REPORT_PATH = SCRIPT_DIR / "backtest_v1_report.json"
YEARLY_PATH = SCRIPT_DIR / "backtest_v1_yearly.csv"
DRAWDOWN_PATH = SCRIPT_DIR / "backtest_v1_drawdowns.csv"

# ── 参数 ──────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_BPS = 5          # 单边手续费 5bp
SLIPPAGE_BPS = 5            # 单边滑点 5bp
COST_PER_SIDE = (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000  # 0.001
REBALANCE_DAYS = 5


def load_data():
    """加载持仓、价格、SPY基准。"""
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


def build_rebalance_schedule(holdings: pd.DataFrame) -> list[pd.Timestamp]:
    """提取调仓日列表（每5个交易日调仓一次）。"""
    all_dates = sorted(holdings["Date"].unique())
    # holdings 里每个 date 都有持仓，但只有调仓日才换仓
    # 策略是每5天调一次，找出调仓日
    rebalance_dates = all_dates[::REBALANCE_DAYS]
    return rebalance_dates


def run_backtest(holdings: pd.DataFrame, prices: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """
    逐日模拟：
    - 调仓日：按 holdings 目标权重调整持仓，差额部分计算交易成本
    - 非调仓日：持仓不变，按价格变动更新市值
    """
    print("Running backtest ...")

    # 合并价格：每只股票每天的 Open 和 Close
    price_map = prices.set_index(["Date", "ticker"])[["Open", "Close"]]

    # 所有交易日
    all_dates = sorted(prices["Date"].unique())
    # 只取回测期内的交易日
    bt_start = holdings["Date"].min()
    bt_end = holdings["Date"].max()
    trade_dates = [d for d in all_dates if bt_start <= d <= bt_end]

    rebalance_dates = set(build_rebalance_schedule(holdings))
    holdings_by_date = holdings.groupby("Date")

    # 状态变量
    cash = INITIAL_CAPITAL
    positions: dict[str, float] = {}  # ticker -> shares
    nav_records = []
    total_turnover = 0.0
    total_cost = 0.0
    n_rebalances = 0

    for i, date in enumerate(trade_dates):
        is_rebalance = date in rebalance_dates

        if is_rebalance and date in holdings_by_date.groups:
            # ── 调仓逻辑 ──
            target = holdings_by_date.get_group(date).set_index("ticker")["weight"]

            # 当前组合总市值（用今日 Open 估值）
            port_value = cash
            for tkr, shares in positions.items():
                try:
                    opn = price_map.loc[(date, tkr), "Open"]
                    port_value += shares * opn
                except KeyError:
                    pass

            # 目标持仓股数（用 Open 价计算）
            target_positions: dict[str, float] = {}
            for tkr, w in target.items():
                try:
                    opn = price_map.loc[(date, tkr), "Open"]
                    if opn > 0:
                        target_positions[tkr] = (port_value * w) / opn
                except KeyError:
                    pass

            # 计算交易量 & 成本
            all_tickers = set(list(positions.keys()) + list(target_positions.keys()))
            day_turnover = 0.0

            for tkr in all_tickers:
                old_shares = positions.get(tkr, 0.0)
                new_shares = target_positions.get(tkr, 0.0)
                delta_shares = new_shares - old_shares

                if abs(delta_shares) < 1e-6:
                    continue

                try:
                    opn = price_map.loc[(date, tkr), "Open"]
                except KeyError:
                    continue

                trade_value = abs(delta_shares) * opn
                cost = trade_value * COST_PER_SIDE
                day_turnover += trade_value
                cash -= delta_shares * opn + cost
                total_cost += cost
                positions[tkr] = new_shares

            total_turnover += day_turnover
            n_rebalances += 1

            # 清理空仓
            positions = {k: v for k, v in positions.items() if abs(v) > 1e-6}

        # ── 日终估值（用 Close） ──
        port_value_eod = cash
        for tkr, shares in positions.items():
            try:
                cls = price_map.loc[(date, tkr), "Close"]
                port_value_eod += shares * cls
            except KeyError:
                pass

        nav_records.append({
            "Date": date,
            "nav": port_value_eod,
            "cash": cash,
            "n_positions": len(positions),
            "is_rebalance": is_rebalance,
        })

    nav_df = pd.DataFrame(nav_records)
    nav_df["daily_return"] = nav_df["nav"].pct_change().fillna(0)

    # 合并 SPY
    nav_df = nav_df.merge(spy, on="Date", how="left")
    nav_df["spy_close"] = nav_df["spy_close"].ffill()
    nav_df["spy_return"] = nav_df["spy_close"].pct_change().fillna(0)
    nav_df["spy_nav"] = INITIAL_CAPITAL * (1 + nav_df["spy_return"]).cumprod()
    nav_df["excess_return"] = nav_df["daily_return"] - nav_df["spy_return"]

    # 不含成本的净值（用 actual_label 反推）
    nav_df["nav_no_cost"] = nav_df["nav"] + total_cost * (nav_df["nav"] / nav_df["nav"].iloc[-1])

    print(f"  Backtest period: {trade_dates[0].date()} ~ {trade_dates[-1].date()}")
    print(f"  Trading days: {len(trade_dates)}")
    print(f"  Rebalances: {n_rebalances}")
    print(f"  Total turnover: ${total_turnover:,.0f}")
    print(f"  Total cost: ${total_cost:,.0f}")

    return nav_df, total_turnover, total_cost, n_rebalances


def compute_metrics(nav_df: pd.DataFrame, total_turnover: float,
                    total_cost: float, n_rebalances: int) -> dict:
    """计算完整绩效指标。"""
    ret = nav_df["daily_return"]
    excess = nav_df["excess_return"]
    n_days = len(ret)
    n_years = n_days / 252

    # 基础指标
    total_ret = nav_df["nav"].iloc[-1] / INITIAL_CAPITAL - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
    rf_rate = 0.04  # 假设无风险利率4%
    sharpe_rf = (annual_ret - rf_rate) / annual_vol if annual_vol > 0 else 0

    # SPY 基准
    spy_total_ret = nav_df["spy_nav"].iloc[-1] / INITIAL_CAPITAL - 1
    spy_annual_ret = (1 + spy_total_ret) ** (1 / n_years) - 1
    spy_annual_vol = nav_df["spy_return"].std() * np.sqrt(252)

    # 超额
    excess_annual_ret = annual_ret - spy_annual_ret
    excess_annual_vol = excess.std() * np.sqrt(252)
    info_ratio = excess_annual_ret / excess_annual_vol if excess_annual_vol > 0 else 0

    # 回撤
    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax
    max_dd = drawdown.min()
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 0 else 0

    # 胜率
    win_rate = (ret > 0).mean()
    profit_days = ret[ret > 0]
    loss_days = ret[ret < 0]
    avg_win = profit_days.mean() if len(profit_days) > 0 else 0
    avg_loss = abs(loss_days.mean()) if len(loss_days) > 0 else 0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 成本
    avg_annual_turnover = total_turnover / n_years
    cost_drag = total_cost / INITIAL_CAPITAL / n_years  # 年化成本拖累

    # Sortino（下行波动率）
    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0

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
            "total_cost_per_side_bps": COMMISSION_BPS + SLIPPAGE_BPS,
            "total_cost_dollar": round(float(total_cost), 2),
            "annual_cost_drag": round(float(cost_drag), 4),
            "total_turnover": round(float(total_turnover), 2),
            "annual_turnover": round(float(avg_annual_turnover), 2),
            "n_rebalances": n_rebalances,
        },
    }


def compute_yearly(nav_df: pd.DataFrame) -> pd.DataFrame:
    """分年度绩效。"""
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

        records.append({
            "year": int(year),
            "return": round(float(yr_ret), 4),
            "volatility": round(float(yr_vol), 4),
            "sharpe": round(float(yr_sharpe), 4),
            "max_drawdown": round(float(max_dd), 4),
            "spy_return": round(float(spy_yr_ret), 4),
            "excess_return": round(float(excess), 4),
            "win_rate": round(float((ret > 0).mean()), 4),
            "trading_days": int(len(grp)),
        })

    return pd.DataFrame(records)


def compute_drawdowns(nav_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """找出最大的 N 次回撤。"""
    nav = nav_df.set_index("Date")["nav"]
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    # 找回撤区间
    in_dd = drawdown < 0
    dd_groups = (~in_dd).cumsum()
    dd_groups = dd_groups[in_dd]

    records = []
    for group_id, grp_idx in dd_groups.groupby(dd_groups):
        dd_slice = drawdown.loc[grp_idx.index]
        trough_date = dd_slice.idxmin()
        peak_date = grp_idx.index[0] - pd.Timedelta(days=1)
        # 找真正的peak date
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

    dd_df = pd.DataFrame(records).sort_values("max_drawdown").head(top_n).reset_index(drop=True)
    return dd_df


def main() -> None:
    holdings, prices, spy = load_data()
    nav_df, total_turnover, total_cost, n_rebalances = run_backtest(holdings, prices, spy)

    # 保存净值
    nav_df.to_parquet(NAV_PATH, index=False)
    print(f"Saved NAV: {NAV_PATH}")

    # 绩效指标
    metrics = compute_metrics(nav_df, total_turnover, total_cost, n_rebalances)

    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS (LightGBM Top20 Long-Only)")
    print(f"{'='*50}")
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
    print(f"  ── Benchmark (SPY) ──")
    print(f"  SPY Annual Ret:   {metrics['benchmark_spy']['annual_return']:.2%}")
    print(f"  Excess Return:    {metrics['excess_vs_spy']['excess_annual_return']:.2%}")
    print(f"  Info Ratio:       {metrics['excess_vs_spy']['information_ratio']:.2f}")
    print(f"  ── Costs ──")
    print(f"  Total Cost:       ${metrics['costs']['total_cost_dollar']:,.0f}")
    print(f"  Annual Cost Drag: {metrics['costs']['annual_cost_drag']:.2%}")
    print(f"  Annual Turnover:  ${metrics['costs']['annual_turnover']:,.0f}")

    # 分年度
    yearly = compute_yearly(nav_df)
    yearly.to_csv(YEARLY_PATH, index=False)
    print(f"\n── Yearly Performance ──")
    print(yearly.to_string(index=False))

    # 回撤分析
    drawdowns = compute_drawdowns(nav_df)
    drawdowns.to_csv(DRAWDOWN_PATH, index=False)
    print(f"\n── Top Drawdowns ──")
    print(drawdowns.to_string(index=False))

    # 保存报告
    report = {
        "metrics": metrics,
        "yearly": yearly.to_dict(orient="records"),
        "top_drawdowns": drawdowns.to_dict(orient="records"),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
