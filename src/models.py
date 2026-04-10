"""Shared data models used across the pipeline."""

from datetime import date, datetime
from typing import Literal, Optional
from pydantic import BaseModel


class NewsArticle(BaseModel):
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime


class TickerSnapshot(BaseModel):
    ticker: str
    price: float
    pct_change_1d: float
    pct_change_5d: float
    volume: int
    market_cap: Optional[float] = None


_BULLISH_TX_TYPES: frozenset = frozenset({
    "unusual_call", "13d_activist_stake", "13g_passive_stake",
    "13f_new_position", "13f_increase",
})
_BEARISH_TX_TYPES: frozenset = frozenset({
    "unusual_put", "planned_sale_144", "13f_exit", "13f_decrease",
})
_ACTION_LABELS: dict = {
    "unusual_call":       "UNUSUAL CALL SWEEP",
    "unusual_put":        "UNUSUAL PUT SWEEP",
    "13d_activist_stake": "ACTIVIST STAKE (13D)",
    "13g_passive_stake":  "PASSIVE STAKE (13G)",
    "planned_sale_144":   "PLANNED SALE (Form 144)",
    "13f_new_position":   "NEW POSITION (13F)",
    "13f_increase":       "INCREASED (13F)",
    "13f_exit":           "EXITED (13F)",
    "13f_decrease":       "DECREASED (13F)",
    "form_4_filing":      "FORM 4 FILING",
    "exchange":           "EXCHANGED",
    "purchase":           "BOUGHT",
    "sale":               "SOLD",
    "sale_partial":       "SOLD (partial)",
}


class InsiderTrade(BaseModel):
    ticker: str
    trader_name: str
    trader_type: str           # "politician" | "corporate_insider" | "options_flow" | "institutional"
    role: str                  # "Representative", "Senator", "CEO", "CFO", etc.
    transaction_type: str      # "purchase" | "sale" | "unusual_call" | "13d_activist_stake" | etc.
    amount_range: str          # e.g. "$15,001 - $50,000"
    transaction_date: date
    disclosure_date: date
    notes: str = ""

    @property
    def is_bullish(self) -> bool:
        tx = self.transaction_type
        return "purchase" in tx or tx in _BULLISH_TX_TYPES

    @property
    def action_label(self) -> str:
        return _ACTION_LABELS.get(
            self.transaction_type,
            "BOUGHT" if "purchase" in self.transaction_type else "SOLD",
        )


Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class MacroContext(BaseModel):
    """Macro regime snapshot derived from FRED indicators."""
    # Yield curve
    yield_spread_10y2y: Optional[float] = None   # 10Y - 2Y spread in %
    yield_curve_signal: str = "UNKNOWN"           # INVERTED | FLAT | NORMAL | STEEP

    # Monetary policy
    fed_funds_rate: Optional[float] = None        # Effective Fed Funds Rate %

    # Inflation
    cpi_yoy: Optional[float] = None              # CPI year-over-year %
    inflation_signal: str = "UNKNOWN"            # HIGH | ELEVATED | MODERATE | LOW

    # Labor market
    unemployment_rate: Optional[float] = None
    unemployment_trend: str = "STABLE"           # RISING | STABLE | FALLING

    # Credit markets
    hy_spread: Optional[float] = None            # HY OAS spread %
    ig_spread: Optional[float] = None            # IG OAS spread %
    credit_signal: str = "UNKNOWN"               # STRESSED | ELEVATED | NORMAL | TIGHT

    # Liquidity
    m2_growth_yoy: Optional[float] = None        # M2 year-over-year %

    # Overall
    regime: str = "UNKNOWN"                      # EXPANSION | SLOWDOWN | LATE_CYCLE | RECESSION
    summary: str = ""                            # Human-readable 2-3 sentence summary


class TickerSignal(BaseModel):
    ticker: str
    direction: Direction
    confidence: float          # 0.0 – 1.0
    sentiment_score: float     # -1.0 to +1.0
    technical_score: float     # -1.0 to +1.0
    insider_score: float = 0.0 # -1.0 to +1.0
    rationale: str
    insider_summary: str = ""  # human-readable insider/politician trade context
    sources_agreeing: int = 0  # how many enabled signal layers agree with the direction


class Recommendation(BaseModel):
    ticker: str
    type: str = "STOCK"        # "STOCK" or "ETF"
    direction: Direction
    confidence: float
    action: str                # "BUY", "SELL", "HOLD", "WATCH"
    time_horizon: str = "N/A"  # "SWING", "SHORT-TERM", "POSITION", "N/A"
    rationale: str
    generated_at: datetime
