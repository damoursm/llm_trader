"""
Massive/Polygon technical-indicator score — a server-side alternative to our
locally-computed technical methods, run ALONGSIDE them for comparison.

Composite of Massive's RSI (level/momentum) + MACD (histogram momentum), each
mapped to [-1, +1] and averaged. Tracked as the `massive` method so the dashboard's
Model Performance can compare it head-to-head with our own `tech` — and we can
decide whether the server-side path (no local indicator math) is worth keeping.

Two API calls per ticker (RSI + MACD); the aggregator caps how many tickers attempt
it via ``massive_tech_max_tickers``. Fail-soft: 0.0 (= no view) on any miss.
"""

from math import tanh

from loguru import logger

from config import settings
from src.data import polygon_client
from src.data.market_data import is_valid_ticker


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_massive_tech_score(ticker: str) -> float:
    """Composite [-1, +1] technical score from Massive's RSI + MACD.

    0.0 means "no view" (feature off, Polygon unavailable, invalid ticker, or both
    indicators missing). RSI is centred at 50 (80→+1, 20→−1); MACD uses the
    histogram normalised by the line+signal magnitude (price-independent)."""
    if (not settings.enable_massive_tech or not polygon_client.is_available()
            or not is_valid_ticker(ticker)):
        return 0.0
    try:
        rsi = polygon_client.get_rsi(ticker)
        macd = polygon_client.get_macd(ticker)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"[massive_tech] {ticker} fetch failed: {e}")
        return 0.0

    parts = []
    if rsi is not None:
        parts.append(_clamp((rsi - 50.0) / 30.0))           # 50 neutral, 80→+1, 20→−1
    if macd is not None:
        denom = abs(macd["value"]) + abs(macd["signal"]) + 1e-9
        parts.append(_clamp(tanh(2.0 * macd["histogram"] / denom)))
    if not parts:
        return 0.0
    return round(sum(parts) / len(parts), 3)
