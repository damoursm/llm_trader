"""Shared data models used across the pipeline."""

from datetime import datetime
from typing import List, Literal, Optional
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


Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class TickerSignal(BaseModel):
    ticker: str
    direction: Direction
    confidence: float          # 0.0 – 1.0
    sentiment_score: float     # -1.0 to +1.0
    technical_score: float     # -1.0 to +1.0
    rationale: str


class Recommendation(BaseModel):
    ticker: str
    direction: Direction
    confidence: float
    action: str                # e.g. "BUY", "SELL", "HOLD", "WATCH"
    rationale: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Performance tracking (Priority 0 feedback loop)
# ---------------------------------------------------------------------------


class GradedRec(BaseModel):
    """A past recommendation graded against its realized forward return."""
    ticker: str
    action: str
    direction: Direction
    confidence: float
    generated_at: datetime
    horizon_days: int
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    raw_return: float          # decimal, e.g. 0.012 = +1.2% price move
    aligned_return: float      # raw_return signed by the call (BUY +, SELL -); >0 means the call was right
    hit: bool                  # aligned_return > 0


class HorizonStats(BaseModel):
    """Aggregate hit-rate / return stats for directional calls at one horizon."""
    horizon_days: int
    graded: int                # number of directional (BUY/SELL) calls graded
    hits: int
    hit_rate: float            # 0.0 – 1.0
    avg_aligned_return: float  # mean signed return; >0 means calls add value on average
    avg_raw_return: float
    best: float
    worst: float


class Scorecard(BaseModel):
    """Realized-performance summary built from graded recommendations."""
    horizons: List[HorizonStats]
    recent: List[GradedRec]
    total_logged: int          # total recommendations ever logged
    total_graded: int          # total (rec, horizon) pairs graded
