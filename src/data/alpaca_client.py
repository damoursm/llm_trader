"""
Alpaca Market Data — real-time SIP intraday bars (Algo Trader Plus).

Primary source for 30-minute candles when ALPACA_API_KEY / ALPACA_API_SECRET are
configured: full US-market (SIP) coverage, real-time on the Plus plan, and none of
yfinance's ~60-day cap or per-IP 429s. The caller falls back to yfinance when Alpaca
is absent, lacks the ticker, or errors.

Uses direct REST via httpx (no SDK dependency, no version drift) — the same approach
as polygon_client. The client also supports the daily/weekly timeframes, but the
pipeline keeps those on the existing Polygon + resample path (already full-coverage,
and the deterministic daily CLOSE is the anchor for NAV/IC); Alpaca is wired only for
the 30-min gap.

Covers US equities / ETFs. Indices (^VIX …), futures (GC=F …) and options stay on
their existing providers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
import pandas as pd
from loguru import logger

from config import settings

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo as _ZoneInfo  # type: ignore

_DATA_BASE = "https://data.alpaca.markets"
_TIMEOUT = 30.0
_NY_TZ = _ZoneInfo("America/New_York")

# Alpaca bar timeframe strings keyed by this project's interval tokens.
_TIMEFRAME = {"30m": "30Min", "1d": "1Day", "1w": "1Week"}

# Alpaca's /v2/stocks/bars multi-symbol cap (one request covers up to 200 symbols).
_MAX_SYMBOLS_PER_REQ = 200

# Warn-once on an entitlement 403 (e.g. the SIP feed isn't on the plan) so a
# misconfigured key degrades to yfinance without spamming the log every tick.
_WARNED_FORBIDDEN: set = set()


def is_available() -> bool:
    """True when the Alpaca data keys are configured and the feed is enabled."""
    return bool(
        settings.enable_alpaca_intraday
        and settings.alpaca_api_key
        and settings.alpaca_api_secret
    )


def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret,
    }


def _to_alpaca_symbol(ticker: str) -> str:
    """Map this project's yfinance-style class shares (BRK-B) to Alpaca's dot form (BRK.B)."""
    return ticker.replace("-", ".")


def _from_alpaca_symbol(symbol: str) -> str:
    """Inverse of _to_alpaca_symbol (Alpaca's BRK.B → this project's BRK-B)."""
    return symbol.replace(".", "-")


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session 30-min bars (09:30–16:00 ET).

    Alpaca SIP returns extended-hours bars too, but the 30-min technical panel has
    always been RTH-only (the yfinance contract), and the intraday cache's
    'static off-hours' reuse assumes no new bar forms after the close. Filtering to
    RTH preserves both invariants. Bars are labelled by start time, so the last kept
    bar starts 15:30 (covers 15:30–16:00)."""
    if df.empty:
        return df
    et = df.index.tz_convert(_NY_TZ)
    mask = [(t.hour, t.minute) >= (9, 30) and (t.hour, t.minute) < (16, 0) for t in et]
    return df[mask]


def _get(path: str, params: dict) -> Optional[dict]:
    """Authenticated GET to the Alpaca data API. Returns parsed JSON or None."""
    try:
        r = httpx.get(f"{_DATA_BASE}{path}", params=params, headers=_headers(), timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 403:
            # Entitlement limit (e.g. SIP feed not on the plan). Not transient and
            # the yfinance fallback covers it — warn once per path, then stay silent.
            if path not in _WARNED_FORBIDDEN:
                _WARNED_FORBIDDEN.add(path)
                logger.warning(
                    f"[alpaca] HTTP 403 on {path} — feed '{settings.alpaca_data_feed}' not "
                    "available on this plan; using yfinance fallback (further 403s silenced)."
                )
        elif status == 429:
            logger.warning(f"[alpaca] Rate limited on {path} — yfinance fallback will apply")
        elif status != 404:
            logger.warning(f"[alpaca] HTTP {status} on {path}")
        return None
    except Exception as exc:
        logger.warning(f"[alpaca] GET {path}: {exc}")
        return None


def _request_bars(symbols: List[str], timeframe: str, lookback_days: int) -> Dict[str, list]:
    """Fetch raw bars for *symbols* (already in Alpaca dot form) across all pages.

    Returns ``{alpaca_symbol: [raw bar dict, ...]}``. One multi-symbol request plus
    pagination — the page token is request-wide, so bars are accumulated per symbol
    across pages. Empty dict on any failure."""
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat(),
        "feed": settings.alpaca_data_feed,
        "adjustment": "all",
        "sort": "asc",
        "limit": 10000,
    }
    acc: Dict[str, list] = {}
    page_token: Optional[str] = None
    while True:
        if page_token:
            params["page_token"] = page_token
        j = _get("/v2/stocks/bars", params)
        if j is None:
            break
        for sym, bars in (j.get("bars") or {}).items():
            acc.setdefault(sym, []).extend(bars or [])
        page_token = j.get("next_page_token")
        if not page_token:
            break
    return acc


def _rows_to_df(bars: list, interval: str) -> pd.DataFrame:
    """Convert raw Alpaca bar dicts to the standard OHLCV frame (tz-aware UTC index;
    30-min bars RTH-filtered). Empty frame when there are no bars."""
    rows = []
    for b in bars:
        rows.append({
            "ts":     pd.Timestamp(b["t"]),   # RFC3339 → tz-aware UTC
            "Open":   float(b["o"]),
            "High":   float(b["h"]),
            "Low":    float(b["l"]),
            "Close":  float(b["c"]),
            "Volume": int(b["v"]),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    df.index.name = None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if interval == "30m":
        df = _rth_only(df)
    return df


def get_bars(ticker: str, interval: str = "30m", lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a single *ticker* at *interval* ("30m" | "1d" | "1w").

    Returns a DataFrame (Open, High, Low, Close, Volume) with a tz-aware UTC
    DatetimeIndex — the format market_data's intraday merge expects. 30-min bars are
    RTH-filtered. Empty DataFrame on any failure / when Alpaca is unavailable, so the
    caller falls back to yfinance.
    """
    if not is_available():
        return pd.DataFrame()
    timeframe = _TIMEFRAME.get(interval)
    if timeframe is None:
        return pd.DataFrame()
    sym = _to_alpaca_symbol(ticker)
    acc = _request_bars([sym], timeframe, lookback_days)
    return _rows_to_df(acc.get(sym) or [], interval)


def get_bars_batch(tickers: List[str], interval: str = "30m",
                   lookback_days: int = 120) -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for MANY *tickers* at once — Alpaca's multi-symbol endpoint collapses
    the per-ticker round trips into one request per ``_MAX_SYMBOLS_PER_REQ`` (200)
    symbols (plus pagination).

    Returns ``{ticker: DataFrame}`` keyed by the project's ticker form (BRK-B, not
    BRK.B), including only tickers Alpaca returned bars for. Empty dict when Alpaca
    is unavailable / the interval is unknown; the caller falls back to the per-ticker
    path for anything missing.
    """
    if not is_available():
        return {}
    timeframe = _TIMEFRAME.get(interval)
    if timeframe is None:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    uniq = list(dict.fromkeys(tickers))
    for i in range(0, len(uniq), _MAX_SYMBOLS_PER_REQ):
        chunk = uniq[i:i + _MAX_SYMBOLS_PER_REQ]
        sym_map = {_to_alpaca_symbol(t): t for t in chunk}
        acc = _request_bars(list(sym_map.keys()), timeframe, lookback_days)
        for asym, bars in acc.items():
            proj = sym_map.get(asym) or _from_alpaca_symbol(asym)
            df = _rows_to_df(bars, interval)
            if not df.empty:
                out[proj] = df
    return out
