"""
风控模块
========
三层风控机制：

1. 目标波动率缩放（Target Volatility Scaling）
   - 用过去 60 日实现波动率估算当前风险水平
   - 动态调整仓位 = 目标波动率 / 实现波动率
   - 波动率高时自动降仓位，低时加仓位

2. 最大回撤熔断（Drawdown Circuit Breaker）
   - 回撤 > 15%：仓位降至 50%
   - 回撤 > 25%：仓位降至 20%
   - 从高水位恢复后自动解除

3. 行业集中度限制（Sector Concentration Limit）
   - 单行业权重不超过 30%
   - 超出部分等比例分配给其他行业
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 参数 ──────────────────────────────────────────────
TARGET_VOL = 0.12           # 目标年化波动率 12%
VOL_LOOKBACK = 60           # 波动率计算窗口
VOL_SCALE_MIN = 0.2         # 最低仓位比例
VOL_SCALE_MAX = 1.0         # 最高仓位比例（不加杠杆）

DD_LEVEL_1 = 0.15           # 回撤 15% → 降仓
DD_SCALE_1 = 0.50           # 降到 50%
DD_LEVEL_2 = 0.25           # 回撤 25% → 重度降仓
DD_SCALE_2 = 0.20           # 降到 20%

SECTOR_MAX_WEIGHT = 0.30    # 单行业最大权重 30%

# S&P 500 股票的行业映射（简化版，覆盖我们的 100 只）
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology", "AVGO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "AMD": "Technology", "INTC": "Technology",
    "CSCO": "Technology", "ACN": "Technology", "TXN": "Technology", "QCOM": "Technology",
    "AMAT": "Technology", "ADI": "Technology", "CDNS": "Technology", "SNPS": "Technology",
    "ADSK": "Technology", "ANET": "Technology", "APH": "Technology", "CDW": "Technology",
    "AKAM": "Technology", "APP": "Technology",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "ABT": "Healthcare", "TMO": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare", "ABBV": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare", "MDT": "Healthcare",
    "BAX": "Healthcare", "BDX": "Healthcare", "BSX": "Healthcare", "CAH": "Healthcare",
    "CNC": "Healthcare", "COR": "Healthcare", "BIIB": "Healthcare", "ALGN": "Healthcare",
    "CRL": "Healthcare",
    # Financials
    "BRK-B": "Financials", "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials", "BX": "Financials",
    "COF": "Financials", "AXP": "Financials", "AIG": "Financials", "ALL": "Financials",
    "AFL": "Financials", "AMP": "Financials", "AIZ": "Financials", "ACGL": "Financials",
    "APO": "Financials", "ARES": "Financials", "BK": "Financials", "BRO": "Financials",
    "CBOE": "Financials", "AON": "Financials", "AJG": "Financials",
    # Consumer Discretionary
    "TSLA": "Consumer Disc.", "HD": "Consumer Disc.", "NKE": "Consumer Disc.",
    "MCD": "Consumer Disc.", "SBUX": "Consumer Disc.", "LOW": "Consumer Disc.",
    "BBY": "Consumer Disc.", "CCL": "Consumer Disc.", "BKNG": "Consumer Disc.",
    "ABNB": "Consumer Disc.", "APTV": "Consumer Disc.", "CVNA": "Consumer Disc.",
    "BLDR": "Consumer Disc.", "XYZ": "Consumer Disc.",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "MO": "Consumer Staples",
    "CPB": "Consumer Staples", "ADM": "Consumer Staples", "BG": "Consumer Staples",
    "BF-B": "Consumer Staples",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "GE": "Industrials", "MMM": "Industrials", "AOS": "Industrials", "AME": "Industrials",
    "AXON": "Industrials", "CARR": "Industrials", "CHRW": "Industrials", "BR": "Industrials",
    "ALLE": "Industrials", "BALL": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AEP": "Utilities", "AEE": "Utilities", "AES": "Utilities",
    "ATO": "Utilities", "AWK": "Utilities", "CNP": "Utilities", "LNT": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "ARE": "Real Estate", "AVB": "Real Estate", "BXP": "Real Estate",
    "CPT": "Real Estate", "CBRE": "Real Estate",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "APA": "Energy", "BKR": "Energy", "CF": "Energy",
    # Communication
    "DIS": "Communication", "NFLX": "Communication", "CMCSA": "Communication",
    "T": "Communication", "TMUS": "Communication",
    # Materials
    "APD": "Materials", "ALB": "Materials", "AVY": "Materials", "AMCR": "Materials",
    "TECH": "Materials",
}


class RiskManager:
    """三层风控管理器，在回测循环中逐日调用。"""

    def __init__(
        self,
        target_vol: float = TARGET_VOL,
        vol_lookback: int = VOL_LOOKBACK,
        dd_level_1: float = DD_LEVEL_1,
        dd_scale_1: float = DD_SCALE_1,
        dd_level_2: float = DD_LEVEL_2,
        dd_scale_2: float = DD_SCALE_2,
        sector_max_weight: float = SECTOR_MAX_WEIGHT,
    ):
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.dd_level_1 = dd_level_1
        self.dd_scale_1 = dd_scale_1
        self.dd_level_2 = dd_level_2
        self.dd_scale_2 = dd_scale_2
        self.sector_max_weight = sector_max_weight

        # 状态
        self.return_history: list[float] = []
        self.high_water_mark: float = 0.0
        self.risk_log: list[dict] = []

    def update_nav(self, nav: float, date) -> None:
        """每日更新净值，追踪高水位。"""
        self.high_water_mark = max(self.high_water_mark, nav)

    def add_return(self, daily_ret: float) -> None:
        """记录日收益率，用于波动率估算。"""
        self.return_history.append(daily_ret)

    # ── 1. 目标波动率缩放 ──
    def vol_scale_factor(self) -> float:
        """根据近期波动率计算仓位缩放系数。"""
        if len(self.return_history) < self.vol_lookback:
            return 1.0  # 数据不足时满仓

        recent = self.return_history[-self.vol_lookback:]
        realized_vol = float(np.std(recent) * np.sqrt(252))

        if realized_vol <= 0:
            return 1.0

        scale = self.target_vol / realized_vol
        scale = max(VOL_SCALE_MIN, min(VOL_SCALE_MAX, scale))
        return scale

    # ── 2. 回撤熔断 ──
    def drawdown_scale_factor(self, current_nav: float) -> float:
        """根据当前回撤深度计算降仓系数。"""
        if self.high_water_mark <= 0:
            return 1.0

        drawdown = (self.high_water_mark - current_nav) / self.high_water_mark

        if drawdown >= self.dd_level_2:
            return self.dd_scale_2
        elif drawdown >= self.dd_level_1:
            # 线性插值：15% 回撤 → 50%，25% 回撤 → 20%
            t = (drawdown - self.dd_level_1) / (self.dd_level_2 - self.dd_level_1)
            return self.dd_scale_1 + t * (self.dd_scale_2 - self.dd_scale_1)
        else:
            return 1.0

    # ── 3. 行业集中度 ──
    @staticmethod
    def apply_sector_limit(
        target_weights: dict[str, float],
        max_sector_weight: float = SECTOR_MAX_WEIGHT,
    ) -> dict[str, float]:
        """限制单行业权重，超出部分等比例分配给未超限行业。"""
        if not target_weights:
            return target_weights

        # 归类行业
        sector_weights: dict[str, float] = {}
        ticker_sector: dict[str, str] = {}
        for tkr, w in target_weights.items():
            sec = SECTOR_MAP.get(tkr, "Other")
            ticker_sector[tkr] = sec
            sector_weights[sec] = sector_weights.get(sec, 0) + w

        # 检查是否有行业超限
        over_sectors = {s: w for s, w in sector_weights.items() if w > max_sector_weight}
        if not over_sectors:
            return target_weights

        # 按比例缩减超限行业的股票权重
        adjusted = dict(target_weights)
        total_excess = 0.0

        for sec, sec_w in over_sectors.items():
            ratio = max_sector_weight / sec_w
            for tkr, w in target_weights.items():
                if ticker_sector[tkr] == sec:
                    new_w = w * ratio
                    total_excess += w - new_w
                    adjusted[tkr] = new_w

        # 把多余的权重按比例分配给未超限行业的股票
        under_tickers = {
            tkr: w for tkr, w in adjusted.items()
            if ticker_sector[tkr] not in over_sectors
        }
        under_total = sum(under_tickers.values())

        if under_total > 0 and total_excess > 0:
            for tkr in under_tickers:
                adjusted[tkr] += total_excess * (adjusted[tkr] / under_total)

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    # ── 综合缩放 ──
    def compute_position_scale(self, current_nav: float, date=None) -> tuple[float, dict]:
        """综合三层风控，返回最终仓位比例。"""
        vol_scale = self.vol_scale_factor()
        dd_scale = self.drawdown_scale_factor(current_nav)

        # 取两者最小值（最保守的那个生效）
        final_scale = min(vol_scale, dd_scale)
        final_scale = max(VOL_SCALE_MIN, min(VOL_SCALE_MAX, final_scale))

        drawdown = 0.0
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - current_nav) / self.high_water_mark

        realized_vol = 0.0
        if len(self.return_history) >= self.vol_lookback:
            realized_vol = float(np.std(self.return_history[-self.vol_lookback:]) * np.sqrt(252))

        detail = {
            "vol_scale": round(vol_scale, 4),
            "dd_scale": round(dd_scale, 4),
            "final_scale": round(final_scale, 4),
            "realized_vol": round(realized_vol, 4),
            "drawdown": round(drawdown, 4),
        }

        if date is not None:
            detail["Date"] = date
            self.risk_log.append(detail)

        return final_scale, detail
