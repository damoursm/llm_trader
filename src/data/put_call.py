"""
Put/Call ratio — market sentiment and per-ticker directional bias.

Two data points:
1. Market-wide P/C  (CBOE equity put/call ratio, free CSV)
   Contrarian indicator: extreme call buying = complacency = bearish warning;
   extreme put buying = fear/capitulation = bullish signal.

2. Per-ticker P/C   (computed from yfinance options volume, already a dependency)
   Directional indicator: heavy puts = bearish positioning; heavy calls = bullish.
   Only surfaces tickers with extreme readings (P/C > 1.5 or P/C < 0.5).

Market-wide thresholds (contrarian):
  P/C < 0.60 → EXTREME_GREED → contrarian BEARISH warning
  P/C 0.60–0.80 → GREED → mild caution
  P/C 0.80–1.00 → NEUTRAL
  P/C 1.00–1.20 → FEAR → mild bullish
  P/C > 1.20 → EXTREME_FEAR → contrarian BULLISH signal

Per-ticker thresholds (directional):
  P/C > 2.0 → EXTREME_PUTS → BEARISH
  P/C 1.5–2.0 → PUTS_HEAVY → BEARISH
  P/C 0.5–1.5 → BALANCED → NEUTRAL (not surfaced)
  P/C 0.3–0.5 → CALLS_HEAVY → BULLISH
  P/C < 0.3 → EXTREME_CALLS → BULLISH

Cached daily.
"""

import csv
import io
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from loguru import logger

from config import settings
from src.models import PutCallContext, PutCallSignal

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.3

# CBOE publishes a daily equity P/C CSV at this long-standing URL
_CBOE_CSV_URL = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/putcallratio.csv"

# Per-ticker: only surface extremes to avoid noise
_TICKER_EXTREME_MIN = 0.5   # below this → CALLS_HEAVY or EXTREME_CALLS
_TICKER_EXTREME_MAX = 1.5   # above this → PUTS_HEAVY or EXTREME_PUTS
_MIN_TOTAL_VOLUME   = 500   # ignore tickers with very thin options activity


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"put_call_{date.today().isoformat()}.json"


def _load_cache() -> Optional[PutCallContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = PutCallContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(
            f"[put_call] Loaded from cache — market P/C: "
            f"{ctx.market_pc_ratio:.2f if ctx.market_pc_ratio else 'N/A'}, "
            f"{len(ctx.ticker_signals)} ticker extreme(s)"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[put_call] Cache load failed: {e}")
        return None


def _save_cache(ctx: PutCallContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[put_call] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Market-wide P/C (CBOE)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_cboe_equity_pc() -> Optional[float]:
    """
    Fetch the most recent CBOE equity put/call ratio from their daily CSV.
    CSV columns: DATE, P/C TOTAL, P/C INDEX, P/C EQUITY
    Returns the equity P/C (column index 3) which excludes index options.
    """
    try:
        resp = requests.get(
            _CBOE_CSV_URL,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; llm_trader/1.0)"},
        )
        resp.raise_for_status()
        reader = list(csv.reader(io.StringIO(resp.text)))

        # Header rows vary — find data rows by checking first column parses as a date
        data_rows = []
        for row in reader:
            if len(row) < 4:
                continue
            try:
                datetime.strptime(row[0].strip(), "%m/%d/%Y")
                equity_pc = float(row[3].strip())
                data_rows.append(equity_pc)
            except (ValueError, IndexError):
                continue

        if not data_rows:
            logger.warning("[put_call] CBOE CSV: no parseable rows found")
            return None

        latest = data_rows[-1]
        logger.info(f"[put_call] CBOE equity P/C ratio: {latest:.2f}")
        return latest

    except Exception as e:
        logger.warning(f"[put_call] CBOE CSV fetch failed: {e}")
        return None


def _classify_market_pc(ratio: float) -> Tuple[str, str, str]:
    """Return (signal, direction, summary_snippet) for market-wide P/C (contrarian)."""
    if ratio < 0.60:
        return (
            "EXTREME_GREED",
            "BEARISH",
            f"Equity P/C at {ratio:.2f} — extreme call buying signals investor complacency. "
            "Contrarian: crowded longs historically precede market pullbacks.",
        )
    if ratio < 0.80:
        return (
            "GREED",
            "BEARISH",
            f"Equity P/C at {ratio:.2f} — calls dominating, mild complacency. "
            "Slight contrarian caution warranted on broad market longs.",
        )
    if ratio < 1.00:
        return (
            "NEUTRAL",
            "NEUTRAL",
            f"Equity P/C at {ratio:.2f} — balanced options activity, no extreme sentiment.",
        )
    if ratio < 1.20:
        return (
            "FEAR",
            "BULLISH",
            f"Equity P/C at {ratio:.2f} — elevated put buying indicates market fear. "
            "Contrarian: fear-driven hedging can mark near-term lows.",
        )
    return (
        "EXTREME_FEAR",
        "BULLISH",
        f"Equity P/C at {ratio:.2f} — extreme put buying signals capitulation/panic. "
        "Historically one of the strongest contrarian BUY signals for the broad market.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker P/C (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_expiry(s: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _compute_ticker_pc(sym: str, max_expirations: int = 3) -> Optional[Tuple[int, int]]:
    """
    Return (total_put_volume, total_call_volume) for near-term options (≤60 days).
    Returns None if insufficient data.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(sym)
        expirations = t.options
        if not expirations:
            return None

        today = date.today()
        near = [
            e for e in expirations
            if (d := _parse_expiry(e)) and 0 <= (d - today).days <= 60
        ][:max_expirations]

        if not near:
            return None

        total_calls = 0
        total_puts  = 0
        for exp in near:
            try:
                chain = t.option_chain(exp)
                total_calls += int(chain.calls["volume"].fillna(0).sum())
                total_puts  += int(chain.puts["volume"].fillna(0).sum())
                time.sleep(_REQUEST_DELAY)
            except Exception:
                time.sleep(_REQUEST_DELAY)
                continue

        return (total_puts, total_calls) if (total_puts + total_calls) >= _MIN_TOTAL_VOLUME else None

    except Exception as e:
        logger.debug(f"[put_call] {sym} options fetch failed: {e}")
        return None


def _classify_ticker_pc(ratio: float) -> Tuple[str, str]:
    """Return (signal, direction) for a per-ticker P/C ratio."""
    if ratio > 2.0:
        return "EXTREME_PUTS", "BEARISH"
    if ratio > 1.5:
        return "PUTS_HEAVY", "BEARISH"
    if ratio > 0.5:
        return "BALANCED", "NEUTRAL"
    if ratio > 0.3:
        return "CALLS_HEAVY", "BULLISH"
    return "EXTREME_CALLS", "BULLISH"


def _build_ticker_signal(sym: str, put_vol: int, call_vol: int) -> Optional[PutCallSignal]:
    """Build a PutCallSignal for a ticker; returns None if the reading is balanced."""
    if call_vol == 0:
        ratio = 99.0 if put_vol > 0 else 1.0
    else:
        ratio = put_vol / call_vol

    signal, direction = _classify_ticker_pc(ratio)
    if signal == "BALANCED":
        return None   # not extreme enough to surface

    desc = "heavy put" if direction == "BEARISH" else "heavy call"
    summary = (
        f"{sym}: P/C={ratio:.2f} ({put_vol:,} puts / {call_vol:,} calls over near-term expirations). "
        f"{desc.capitalize()} buying — {'bearish' if direction == 'BEARISH' else 'bullish'} positioning signal."
    )
    return PutCallSignal(
        ticker=sym,
        put_volume=put_vol,
        call_volume=call_vol,
        put_call_ratio=round(ratio, 2),
        signal=signal,
        direction=direction,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_put_call_context(tickers: List[str]) -> Optional[PutCallContext]:
    """
    Fetch market-wide CBOE equity P/C ratio and compute per-ticker P/C for the watchlist.
    Only surfaces per-ticker extremes (PUTS_HEAVY / CALLS_HEAVY or stronger).

    Cached daily.

    Args:
        tickers: watchlist tickers to compute per-ticker P/C for

    Returns:
        PutCallContext or None if no data could be fetched.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    today = date.today()

    # ── 1. Market-wide P/C (CBOE) ───────────────────────────────────────────
    market_pc = _fetch_cboe_equity_pc()
    if market_pc is not None:
        market_signal, market_direction, market_summary = _classify_market_pc(market_pc)
    else:
        market_signal    = "UNKNOWN"
        market_direction = "NEUTRAL"
        market_summary   = "CBOE equity P/C data unavailable."

    # ── 2. Per-ticker P/C ────────────────────────────────────────────────────
    ticker_signals: List[PutCallSignal] = []
    if not settings.enable_fetch_data:
        logger.debug("[put_call] ENABLE_FETCH_DATA=false — skipping per-ticker yfinance fetch")
    else:
        for sym in tickers:
            result = _compute_ticker_pc(sym)
            if result is None:
                continue
            put_vol, call_vol = result
            sig = _build_ticker_signal(sym, put_vol, call_vol)
            if sig is not None:
                ticker_signals.append(sig)
                logger.info(
                    f"[put_call] {sym}: P/C={sig.put_call_ratio:.2f} "
                    f"({sig.signal}) → {sig.direction}"
                )

    # Sort: most extreme first (largest deviation from 1.0)
    ticker_signals.sort(key=lambda s: abs(s.put_call_ratio - 1.0), reverse=True)

    if market_pc is None and not ticker_signals:
        logger.info("[put_call] No P/C data available — skipping")
        return None

    # ── 3. Build overall summary ─────────────────────────────────────────────
    summary_parts = [market_summary]
    if ticker_signals:
        bearish_tickers = [s.ticker for s in ticker_signals if s.direction == "BEARISH"]
        bullish_tickers = [s.ticker for s in ticker_signals if s.direction == "BULLISH"]
        parts = []
        if bearish_tickers:
            parts.append(f"Heavy put buying: {', '.join(bearish_tickers[:5])}")
        if bullish_tickers:
            parts.append(f"Heavy call buying: {', '.join(bullish_tickers[:5])}")
        summary_parts.append(" | ".join(parts) + ".")

    ctx = PutCallContext(
        market_pc_ratio=market_pc,
        market_signal=market_signal,
        market_direction=market_direction,
        ticker_signals=ticker_signals,
        report_date=today,
        summary=" ".join(summary_parts),
    )
    _save_cache(ctx)
    logger.info(
        f"[put_call] Context built — market P/C: "
        f"{market_pc:.2f if market_pc else 'N/A'}, "
        f"{len(ticker_signals)} ticker extreme(s)"
    )
    return ctx
