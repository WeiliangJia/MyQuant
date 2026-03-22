"""
模拟实盘：真实成本环境下的策略表现
====================================
在 backtest_v2 的基础上，加入真实世界的全部成本：

1. 券商佣金（Alpaca免佣 / IBKR $0.005/股）
2. 买卖价差（Bid-Ask Spread）
3. 滑点（市场冲击）
4. SEC 费用（卖出时 $22.90 / $1,000,000）
5. FINRA TAF 费用（$0.000166/股）
6. 月度固定成本（数据/服务器）
7. 不同本金规模对比（$1K ~ $100K）

输出：各本金规模下的净收益、夏普、成本占比
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from risk.risk_manager import RiskManager

# ── 路径 ──────────────────────────────────────────────
HOLDINGS_PATH = PROJECT_DIR / "strategy" / "holdings_multi.parquet"
PRICE_PATH = PROJECT_DIR / "data" / "data" / "processed" / "panel_aligned.parquet"
SPY_PATH = PROJECT_DIR / "data" / "data" / "raw" / "SPY.parquet"
REPORT_PATH = SCRIPT_DIR / "simulate_live_report.json"

# ── 真实成本参数 ──────────────────────────────────────
COST_SCENARIOS = {
    "Alpaca_Free": {
        "description": "Alpaca 免佣 + 免费数据 (yfinance)",
        "commission_per_share": 0.0,
        "spread_bps": 3,           # Alpaca 走 PFOF，价差略大
        "slippage_bps": 3,         # 小资金市场冲击小
        "sec_fee_per_million_sold": 22.90,  # SEC fee
        "finra_taf_per_share": 0.000166,    # FINRA TAF
        "monthly_fixed_cost": 0,   # 全免费
    },
    "IBKR_Pro": {
        "description": "Interactive Brokers Pro + Polygon数据",
        "commission_per_share": 0.005,  # $0.005/股，最低$1/单
        "min_commission_per_order": 1.0,
        "spread_bps": 1,           # IBKR 直连交易所，价差更小
        "slippage_bps": 2,
        "sec_fee_per_million_sold": 22.90,
        "finra_taf_per_share": 0.000166,
        "monthly_fixed_cost": 35,  # Polygon $29 + 服务器 $6
    },
}

CAPITAL_LEVELS = [1_000, 5_000, 10_000, 50_000, 100_000]
REBALANCE_DAYS = 5


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


def compute_trade_cost(
    shares: float,
    price: float,
    is_sell: bool,
    scenario: dict,
) -> float:
    """计算单笔交易的全部成本。"""
    trade_value = abs(shares) * price
    if trade_value < 0.01:
        return 0.0

    cost = 0.0

    # 1. 佣金
    commission = abs(shares) * scenario["commission_per_share"]
    min_comm = scenario.get("min_commission_per_order", 0)
    if commission > 0 and commission < min_comm:
        commission = min_comm
    cost += commission

    # 2. 买卖价差（单边 spread/2）
    cost += trade_value * scenario["spread_bps"] / 10_000 / 2

    # 3. 滑点
    cost += trade_value * scenario["slippage_bps"] / 10_000

    # 4. SEC 费（仅卖出）
    if is_sell:
        cost += trade_value * scenario["sec_fee_per_million_sold"] / 1_000_000

    # 5. FINRA TAF
    cost += abs(shares) * scenario["finra_taf_per_share"]

    return cost


def run_simulation(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    initial_capital: float,
    scenario: dict,
) -> dict:
    """运行单次模拟。"""

    # 构建快速查找字典
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
    cash = initial_capital
    positions: dict[str, float] = {}
    total_commission = 0.0
    total_spread = 0.0
    total_slippage = 0.0
    total_sec_finra = 0.0
    total_fixed_cost = 0.0
    total_trades = 0
    prev_nav = initial_capital
    nav_records = []
    last_month = None

    for date in trade_dates:
        is_rebalance = date in rebalance_set

        # 月度固定成本
        current_month = (date.year, date.month)
        if last_month is not None and current_month != last_month:
            monthly_cost = scenario["monthly_fixed_cost"]
            cash -= monthly_cost
            total_fixed_cost += monthly_cost
        last_month = current_month

        # 日初估值
        port_value_open = cash
        for tkr, shares in positions.items():
            opn = open_dict.get((date, tkr))
            if opn is not None:
                port_value_open += shares * opn

        if is_rebalance and date in holdings_by_date:
            scale, _ = rm.compute_position_scale(port_value_open, date=date)
            raw_weights = holdings_by_date[date]
            adjusted_weights = RiskManager.apply_sector_limit(raw_weights)
            scaled_weights = {tkr: w * scale for tkr, w in adjusted_weights.items()}

            target_positions: dict[str, float] = {}
            for tkr, w in scaled_weights.items():
                opn = open_dict.get((date, tkr))
                if opn is not None and opn > 0:
                    # 碎股支持（Alpaca 支持碎股）
                    target_positions[tkr] = (port_value_open * w) / opn

            all_tickers = set(list(positions.keys()) + list(target_positions.keys()))

            for tkr in all_tickers:
                old_shares = positions.get(tkr, 0.0)
                new_shares = target_positions.get(tkr, 0.0)
                delta = new_shares - old_shares

                if abs(delta) < 1e-6:
                    continue

                opn = open_dict.get((date, tkr))
                if opn is None:
                    continue

                is_sell = delta < 0
                cost = compute_trade_cost(abs(delta), opn, is_sell, scenario)

                # 分类记录成本
                trade_value = abs(delta) * opn
                total_commission += abs(delta) * scenario["commission_per_share"]
                total_spread += trade_value * scenario["spread_bps"] / 10_000 / 2
                total_slippage += trade_value * scenario["slippage_bps"] / 10_000
                if is_sell:
                    total_sec_finra += trade_value * scenario["sec_fee_per_million_sold"] / 1_000_000
                total_sec_finra += abs(delta) * scenario["finra_taf_per_share"]

                cash -= delta * opn + cost
                positions[tkr] = new_shares
                total_trades += 1

            positions = {k: v for k, v in positions.items() if abs(v) > 1e-6}

        # 日终估值
        port_value_eod = cash
        for tkr, shares in positions.items():
            cls = close_dict.get((date, tkr))
            if cls is not None:
                port_value_eod += shares * cls

        daily_ret = (port_value_eod - prev_nav) / prev_nav if prev_nav > 0 else 0.0
        rm.add_return(daily_ret)
        rm.update_nav(port_value_eod, date)

        nav_records.append({
            "Date": date,
            "nav": port_value_eod,
            "daily_return": daily_ret,
        })
        prev_nav = port_value_eod

    # 计算指标
    nav_df = pd.DataFrame(nav_records)
    nav_df = nav_df.merge(spy, on="Date", how="left")
    nav_df["spy_close"] = nav_df["spy_close"].ffill()
    nav_df["spy_return"] = nav_df["spy_close"].pct_change().fillna(0)
    nav_df["spy_nav"] = initial_capital * (1 + nav_df["spy_return"]).cumprod()

    ret = nav_df["daily_return"]
    n_days = len(ret)
    n_years = n_days / 252

    total_ret = nav_df["nav"].iloc[-1] / initial_capital - 1
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    spy_total_ret = nav_df["spy_nav"].iloc[-1] / initial_capital - 1
    spy_annual_ret = (1 + spy_total_ret) ** (1 / n_years) - 1

    cummax = nav_df["nav"].cummax()
    drawdown = (nav_df["nav"] - cummax) / cummax
    max_dd = drawdown.min()

    win_rate = (ret > 0).mean()

    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0

    total_cost = total_commission + total_spread + total_slippage + total_sec_finra + total_fixed_cost
    final_nav = nav_df["nav"].iloc[-1]
    net_profit = final_nav - initial_capital

    return {
        "initial_capital": initial_capital,
        "final_nav": round(float(final_nav), 2),
        "net_profit": round(float(net_profit), 2),
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(annual_ret), 4),
        "annual_vol": round(float(annual_vol), 4),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "spy_annual_return": round(float(spy_annual_ret), 4),
        "excess_return": round(float(annual_ret - spy_annual_ret), 4),
        "n_years": round(n_years, 2),
        "total_trades": total_trades,
        "costs": {
            "total": round(float(total_cost), 2),
            "commission": round(float(total_commission), 2),
            "spread": round(float(total_spread), 2),
            "slippage": round(float(total_slippage), 2),
            "sec_finra": round(float(total_sec_finra), 2),
            "fixed_monthly": round(float(total_fixed_cost), 2),
            "cost_pct_of_capital": round(float(total_cost / initial_capital * 100), 2),
            "annual_cost_drag": round(float(total_cost / initial_capital / n_years), 4),
        },
    }


def main():
    print("=" * 70)
    print("  SIMULATE LIVE TRADING — Real-World Cost Analysis")
    print("=" * 70)

    holdings, prices, spy = load_data()
    print(f"  Data loaded. Holdings: {holdings['Date'].nunique()} days")
    print(f"  Building price lookup (this takes a moment) ...")

    all_results = {}

    for scenario_name, scenario in COST_SCENARIOS.items():
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {scenario_name}")
        print(f"  {scenario['description']}")
        print(f"{'─' * 70}")

        scenario_results = {}

        for capital in CAPITAL_LEVELS:
            # 按比例缩放 holdings（权重不变，只是资金量变了）
            result = run_simulation(holdings, prices, spy, float(capital), scenario)
            scenario_results[f"${capital:,}"] = result

            marker = ""
            if result["annual_return"] < 0:
                marker = " ← 亏损"
            elif result["annual_return"] < result["spy_annual_return"]:
                marker = " ← 跑输SPY"

            print(f"\n  本金 ${capital:>7,}:")
            print(f"    最终净值:    ${result['final_nav']:>12,.2f}  (净赚 ${result['net_profit']:>10,.2f})")
            print(f"    年化收益:    {result['annual_return']:>8.2%}  (SPY: {result['spy_annual_return']:.2%}){marker}")
            print(f"    夏普比率:    {result['sharpe']:>8.2f}")
            print(f"    最大回撤:    {result['max_drawdown']:>8.2%}")
            print(f"    胜率:        {result['win_rate']:>8.2%}")
            print(f"    ── 成本明细 ──")
            c = result["costs"]
            print(f"    佣金:        ${c['commission']:>10,.2f}")
            print(f"    价差:        ${c['spread']:>10,.2f}")
            print(f"    滑点:        ${c['slippage']:>10,.2f}")
            print(f"    SEC/FINRA:   ${c['sec_finra']:>10,.2f}")
            print(f"    月费:        ${c['fixed_monthly']:>10,.2f}")
            print(f"    总成本:      ${c['total']:>10,.2f}  (占本金 {c['cost_pct_of_capital']:.1f}%)")
            print(f"    年化成本率:  {c['annual_cost_drag']:>8.2%}")

        all_results[scenario_name] = scenario_results

    # ── 对比总结 ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: 各本金 × 各方案 年化收益对比")
    print(f"{'=' * 70}")
    print(f"  {'本金':>10}  |  {'Alpaca免费':>14}  |  {'IBKR+Polygon':>14}  |  {'SPY买入持有':>14}")
    print(f"  {'─' * 10}  |  {'─' * 14}  |  {'─' * 14}  |  {'─' * 14}")

    for capital in CAPITAL_LEVELS:
        key = f"${capital:,}"
        alpaca = all_results["Alpaca_Free"][key]
        ibkr = all_results["IBKR_Pro"][key]
        print(f"  ${capital:>8,}  |  {alpaca['annual_return']:>12.2%}  |  {ibkr['annual_return']:>12.2%}  |  {alpaca['spy_annual_return']:>12.2%}")

    print(f"\n  {'本金':>10}  |  {'Alpaca夏普':>14}  |  {'IBKR夏普':>14}  |  {'Alpaca年化成本':>14}")
    print(f"  {'─' * 10}  |  {'─' * 14}  |  {'─' * 14}  |  {'─' * 14}")

    for capital in CAPITAL_LEVELS:
        key = f"${capital:,}"
        alpaca = all_results["Alpaca_Free"][key]
        ibkr = all_results["IBKR_Pro"][key]
        print(f"  ${capital:>8,}  |  {alpaca['sharpe']:>14.2f}  |  {ibkr['sharpe']:>14.2f}  |  {alpaca['costs']['annual_cost_drag']:>12.2%}")

    # 保存报告
    REPORT_PATH.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
