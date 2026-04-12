"""Shared data models used across the pipeline."""

from datetime import date, datetime
from typing import Dict, List, Literal, Optional
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


class COTSignal(BaseModel):
    """Positioning signal for a single futures contract from the CFTC COT report."""
    contract: str                   # friendly name, e.g. "Gold"
    tickers: List[str]              # related ETF/futures tickers, e.g. ["GLD", "IAU", "GDX"]
    report_date: date
    net_speculator_pct: float       # (longs - shorts) / OI × 100
    net_change_wow: float           # week-over-week change in net_speculator_pct
    percentile_52w: float           # 0-100: where current positioning sits in 52-week range
    signal: str                     # EXTREME_LONG | BULLISH_TREND | NEUTRAL | BEARISH_TREND | EXTREME_SHORT
    direction: str                  # BULLISH | BEARISH | NEUTRAL (contrarian applied at extremes)
    summary: str


class COTContext(BaseModel):
    """All COT signals for the current week."""
    signals: List[COTSignal]
    report_date: date
    summary: str                    # Human-readable overview of notable extremes/trends


class IPOFiling(BaseModel):
    """A single S-1 or S-11 registration statement from SEC EDGAR."""
    company: str
    filing_date: date
    form_type: str          # "S-1" | "S-1/A" | "S-11" | "S-11/A"
    sector: str             # inferred sector
    is_amendment: bool


class IPOContext(BaseModel):
    """IPO pipeline intelligence derived from recent S-1/S-11 filings."""
    filings: List[IPOFiling]         # initial registrations (S-1, S-11)
    amendments: List[IPOFiling]      # amendments (S-1/A, S-11/A) — pipeline advancing
    sector_counts: Dict[str, int]    # sector → count of initial filings
    hot_sectors: List[str]           # top sectors by filing count
    total_new: int
    total_amendments: int
    lookback_days: int
    report_date: date
    summary: str


class VIXContext(BaseModel):
    """VIX volatility regime and term structure."""
    # Spot levels
    vix:   Optional[float] = None   # 30-day S&P 500 implied vol (^VIX)
    vxn:   Optional[float] = None   # 30-day Nasdaq implied vol (^VXN)
    vvix:  Optional[float] = None   # vol-of-vol / VIX of VIX (^VVIX)
    vix9d: Optional[float] = None   # 9-day VIX (^VIX9D)
    vix3m: Optional[float] = None   # 3-month VIX (^VIX3M)
    vix6m: Optional[float] = None   # 6-month mid-term VIX (^VXMT)

    # Term structure
    term_structure: str = "UNKNOWN"    # CONTANGO | FLAT | BACKWARDATION
    slope_1m_3m: Optional[float] = None  # VIX3M − VIX (positive = contango)

    # Regime
    vix_signal: str = "UNKNOWN"        # PANIC | EXTREME_FEAR | HIGH | ELEVATED | NORMAL | LOW | COMPLACENCY
    vix_direction: str = "NEUTRAL"     # contrarian: EXTREME_FEAR → BULLISH bias

    report_date: date
    summary: str


class PutCallSignal(BaseModel):
    """Put/call ratio signal for a single ticker."""
    ticker: str
    put_volume: int
    call_volume: int
    put_call_ratio: float
    signal: str    # EXTREME_PUTS | PUTS_HEAVY | BALANCED | CALLS_HEAVY | EXTREME_CALLS
    direction: str # BEARISH | BEARISH | NEUTRAL | BULLISH | BULLISH
    summary: str


class PutCallContext(BaseModel):
    """Market-wide and per-ticker put/call ratio sentiment."""
    # Market-wide (CBOE equity P/C ratio) — contrarian indicator
    market_pc_ratio: Optional[float] = None
    market_signal: str = "UNKNOWN"    # EXTREME_GREED | GREED | NEUTRAL | FEAR | EXTREME_FEAR
    market_direction: str = "NEUTRAL" # contrarian: FEAR → BULLISH bias

    # Per-ticker extremes (only tickers with non-BALANCED readings)
    ticker_signals: List["PutCallSignal"] = []

    report_date: date
    summary: str


class EarningsEvent(BaseModel):
    """An upcoming earnings report for a single ticker."""
    ticker: str
    earnings_date: date
    days_until: int
    estimated_eps: Optional[float] = None
    is_confirmed: bool = False


class EarningsContext(BaseModel):
    """Upcoming earnings calendar for the watchlist."""
    upcoming: List[EarningsEvent]   # sorted by date ascending
    report_date: date
    summary: str


class TickerSignal(BaseModel):
    ticker: str
    direction: Direction
    confidence: float           # 0.0 – 1.0
    sentiment_score: float      # -1.0 to +1.0  (news + all article-based sources)
    technical_score: float      # -1.0 to +1.0
    insider_score: float = 0.0  # -1.0 to +1.0  (smart money: insider trades, options flow, SEC)
    put_call_score: float = 0.0 # -1.0 to +1.0  (per-ticker options put/call sentiment)
    rationale: str
    insider_summary: str = ""   # human-readable insider/politician trade context
    sources_agreeing: int = 0   # how many enabled signal layers agree with the direction


class Recommendation(BaseModel):
    ticker: str
    type: str = "STOCK"        # "STOCK" or "ETF"
    direction: Direction
    confidence: float
    action: str                # "BUY", "SELL", "HOLD", "WATCH"
    time_horizon: str = "N/A"  # "SWING", "SHORT-TERM", "POSITION", "N/A"
    rationale: str
    generated_at: datetime
