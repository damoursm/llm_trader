"""Shared data models used across the pipeline."""

from datetime import datetime
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
