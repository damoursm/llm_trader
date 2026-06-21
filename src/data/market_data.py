"""Fetch real-time and historical market data.

Primary source: Polygon.io (batched API calls, no per-IP rate limit, works globally).
Fallback:       yfinance (per-ticker, subject to 429s).

Polygon covers all US equity / ETF tickers.  yfinance is kept as fallback for:
  - Tickers absent from Polygon's feed (very small caps, OTC)
  - Options chains, indices (^VIX, ^MOVE …), and futures (GC=F …) — these
    modules call yfinance directly and are not routed through this file.

Rate-limit handling (yfinance fallback only)
--------------------------------------------
  attempt 1 → wait BACKOFF_BASE  seconds  (60 s)
  attempt 2 → wait BACKOFF_BASE * 2       (120 s)
  attempt 3 → wait BACKOFF_BASE * 4       (240 s)
  …up to BACKOFF_MAX                      (600 s)
"""

import re
import time
import yfinance as yf
import pandas as pd
from datetime import datetime as _datetime, time as _dtime, timedelta as _timedelta, timezone as _timezone
from loguru import logger
from typing import Iterable, List, Optional, Tuple
from config import settings
from src.models import TickerSnapshot
from src.data import polygon_client
from src.performance.market_calendar import current_session

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo as _ZoneInfo  # type: ignore
_NY_TZ = _ZoneInfo("America/New_York")
_MARKET_CLOSE = _dtime(16, 0)


# ── Ticker validation ─────────────────────────────────────────────────────────
# A junk ticker (most often the literal "N/A" leaking from a discovery source
# whose ticker field was missing) reaches yfinance, which then raises an opaque
# "'Response' object is not subscriptable" deep inside its parser — observed 268×
# in one day's log, plus 30× "Not enough history for N/A" downstream. The legit
# universe uses only: plain symbols (AAPL), class shares (BRK-B), futures (CL=F,
# GC=F), the DXY (DX-Y.NYB), and ^-prefixed indices (^VIX, ^NYAD). None contain a
# "/" or whitespace, so a single positive-charset check rejects the junk without
# touching any real ticker.
_VALID_TICKER_RE = re.compile(r"^\^?[A-Z][A-Z0-9.\-=]{0,11}$")
_TICKER_JUNK = frozenset({"N/A", "NA", "NAN", "NONE", "NULL", "--", "-", ""})


def is_valid_ticker(ticker: Optional[str]) -> bool:
    """True iff ``ticker`` is a well-formed symbol (rejects ``N/A``/junk/blank)."""
    if not isinstance(ticker, str):
        return False
    s = ticker.strip().upper()
    if s in _TICKER_JUNK:
        return False
    return bool(_VALID_TICKER_RE.match(s))


def sanitize_tickers(tickers: Iterable[str]) -> List[str]:
    """Upper-case, de-dupe (order-preserving), and drop invalid/junk tickers."""
    seen: set = set()
    out: List[str] = []
    for t in tickers or []:
        if not is_valid_ticker(t):
            continue
        s = t.strip().upper()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── Exotic security-type filter ───────────────────────────────────────────────
# Preferred series, warrants, units, rights, and OTC foreign-ordinary symbols are
# redundant with a primary listing and/or not on the US consolidated tape, so they
# can't be priced deterministically (grouped-daily misses them) and aren't what the
# strategy trades. Drop them from DISCOVERY (the gate); the pinned/watchlist universe
# bypasses this, so an explicitly-chosen preferred is still honored. ADRs ('…Y',
# exchange-listed + tradeable) are deliberately KEPT.
_PREFERRED_RE   = re.compile(r"-P[A-Z]$")                # ALL-PJ, COF-PN  (NOT BRK-B, BF-B)
_WUR_FORM_RE    = re.compile(r"[-.](WT|WS|WI|UN|U|RT|RI|R)$")  # explicit warrant/unit/right/when-issued
_NASDAQ_5CH_RE  = re.compile(r"^[A-Z]{4}[FWU]$")         # 5-char OTC foreign(F)/warrant(W)/unit(U)


def is_exotic_security(ticker: Optional[str]) -> bool:
    """True for preferred / warrant / unit / right / OTC-foreign-ordinary symbols.

    Conservative + high-precision so it never drops a real common: class shares
    (BRK-B), plain commons, 4-char tickers ending F/W/U (INTU, LABU), ADRs (TCEHY),
    futures (GC=F) and indices (^VIX) all pass; only the 5-char Nasdaq special-type
    suffixes and the unambiguous dash/dot forms are flagged."""
    if not isinstance(ticker, str):
        return False
    s = ticker.strip().upper()
    if not s:
        return False
    return bool(_PREFERRED_RE.search(s) or _WUR_FORM_RE.search(s) or _NASDAQ_5CH_RE.match(s))


def _bar_session_date(ts):
    """Map an OHLCV index timestamp to its NYSE session date (ET)."""
    tz = getattr(ts, "tzinfo", None) or getattr(ts, "tz", None)
    if tz is not None:
        try:
            return ts.tz_convert("America/New_York").date()
        except (TypeError, AttributeError):
            try:
                return ts.astimezone(_NY_TZ).date()
            except Exception:
                return None
    try:
        return ts.date()
    except Exception:
        return None


def _ts_to_utc(ts) -> Optional["pd.Timestamp"]:
    """Coerce an OHLCV index timestamp to a tz-aware UTC ``Timestamp``.

    The cache stores tz-naive timestamps in UTC (``_normalize_index_tz``), while
    a fresh yfinance frame is tz-aware ET — so a tz-naive value is interpreted as
    UTC and a tz-aware one is converted. Used by the intraday/weekly forming-bar
    drops, which need instant-level (not just date-level) comparison."""
    try:
        t = pd.Timestamp(ts)
    except (TypeError, ValueError):
        return None
    try:
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _drop_forming_bar(df: Optional[pd.DataFrame], interval: str = "1d") -> Optional[pd.DataFrame]:
    """Remove the current period's still-forming bar (look-ahead guard).

    The pipeline runs intraday, so the newest bar of any timeframe may still be
    forming — feeding it to an indicator (or the NAV walk) reads a price that is
    still moving (look-ahead within the period). This drops it until the period
    closes; once final it is kept. Live prices for fills / marks come from the
    live quote (``_fetch_price``), never from this intentionally-lagged history.

      * ``1d``  — drop today's daily bar until the 16:00 ET close (legacy logic).
      * ``30m`` — drop trailing 30-minute bars whose window (bar_start → +30 min)
                  has not yet elapsed.
      * ``1w``  — drop the current ISO-week bar until that week's Friday close.
    """
    if df is None or getattr(df, "empty", True):
        return df

    if interval == "1d":
        now_et = _datetime.now(_NY_TZ)
        if now_et.time() >= _MARKET_CLOSE:
            return df  # regular session closed — today's daily bar is complete
        today = now_et.date()
        mask = [_bar_session_date(ts) != today for ts in df.index]
        return df if all(mask) else df[mask]

    if interval == "30m":
        now_utc = _datetime.now(_timezone.utc)
        mask = []
        for ts in df.index:
            u = _ts_to_utc(ts)
            # Keep a bar only once its 30-min window has fully elapsed.
            mask.append(u is not None and (u + _timedelta(minutes=30)) <= now_utc)
        return df if all(mask) else df[mask]

    if interval == "1w":
        now_et = _datetime.now(_NY_TZ)

        def _week_complete(ts) -> bool:
            # Resample labels each weekly bar with that week's Friday (W-FRI).
            try:
                friday = pd.Timestamp(ts).date()
            except (TypeError, ValueError):
                return False
            if now_et.date() > friday:
                return True
            if now_et.date() == friday:
                return now_et.time() >= _MARKET_CLOSE
            return False

        mask = [_week_complete(ts) for ts in df.index]
        return df if all(mask) else df[mask]

    return df


# ---------------------------------------------------------------------------
# Rate-limit config (yfinance fallback)
# ---------------------------------------------------------------------------

BACKOFF_BASE = 60
BACKOFF_MAX  = 600
MAX_RL_HITS  = 3
INTER_TICKER = 1.0


# ---------------------------------------------------------------------------
# yfinance rate-limit helpers
# ---------------------------------------------------------------------------

_RATE_LIMIT_PHRASES = ("429", "too many requests", "rate limit", "ratelimit", "yfratelimiterror")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def _backoff_wait(attempt: int) -> None:
    wait = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_MAX)
    logger.warning(
        f"[market_data] Rate limit hit — backing off for {wait:.0f}s "
        f"(attempt {attempt + 1}/{MAX_RL_HITS})"
    )
    elapsed = 0
    while elapsed < wait:
        chunk = min(15, wait - elapsed)
        time.sleep(chunk)
        elapsed += chunk
        remaining = wait - elapsed
        if remaining > 0:
            logger.info(f"[market_data] Resuming in {remaining:.0f}s…")


class _RateLimitAbort(Exception):
    pass


def _fetch_ticker_yf(ticker: str) -> Tuple[Optional[yf.Ticker], Optional[pd.DataFrame]]:
    """Fetch one ticker via yfinance with exponential-backoff retry."""
    rl_hits = 0
    while True:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                return t, hist
            logger.debug(f"[market_data] {ticker}: empty history (skipping)")
            return None, None
        except Exception as e:
            if _is_rate_limit(e):
                if rl_hits >= MAX_RL_HITS:
                    raise _RateLimitAbort(f"Rate limit persists after {MAX_RL_HITS} retries")
                _backoff_wait(rl_hits)
                rl_hits += 1
            else:
                logger.debug(f"[market_data] {ticker}: {e}")
                return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extended_last_price(t) -> Optional[float]:
    """Last extended-hours print via 1-minute prepost bars (None when unavailable).

    ``fast_info.last_price`` reflects the REGULAR session only, so during
    pre/after-market it silently returns the stale RTH close — the wrong
    anchor for extended-session snapshots (``signals.price`` is the entry
    anchor of the recommendation-stream pseudo-trades). The Polygon batch
    path needs no equivalent: its snapshot already prefers ``lastTrade.p``,
    which includes extended prints.
    """
    try:
        bars = t.history(period="1d", interval="1m", prepost=True)
        if bars is None or bars.empty or "Close" not in bars.columns:
            return None
        closes = bars["Close"].dropna()
        if closes.empty:
            return None
        px = float(closes.iloc[-1])
        return px if px > 0 else None
    except Exception:
        return None


def get_snapshots(tickers: List[str]) -> List[TickerSnapshot]:
    """
    Return latest price, 1-day and 5-day % change for each ticker.

    Tries Alpaca first (single batch call for all tickers).  Any ticker not
    returned by Alpaca is retried via yfinance.  Stops early on yfinance rate-
    limit exhaustion but always returns whatever was collected.
    """
    if not settings.enable_fetch_data:
        logger.debug("[market_data] ENABLE_FETCH_DATA=false — skipping snapshot fetch")
        return []

    snapshots: List[TickerSnapshot] = []

    # ── 1. Polygon batch (two REST calls for all tickers, no per-ticker throttle) ──
    polygon_data = polygon_client.get_snapshots_batch(tickers)
    covered: set = set()

    for ticker, data in polygon_data.items():
        if data.get("price") is not None:
            snapshots.append(TickerSnapshot(
                ticker=ticker,
                price=data["price"],
                pct_change_1d=data["pct_change_1d"],
                pct_change_5d=data["pct_change_5d"],
                volume=data["volume"],
                market_cap=None,  # Polygon free tier does not expose market cap
            ))
            covered.add(ticker)
            logger.debug(
                f"[market_data] {ticker}: ${data['price']:.2f} "
                f"({data['pct_change_1d']:+.2f}% 1d) [polygon]"
            )

    # ── 2. yfinance fallback for tickers Polygon didn't cover ────────────
    remaining = [t for t in tickers if t not in covered]
    if remaining:
        if polygon_client.is_available():
            logger.info(
                f"[market_data] yfinance fallback for {len(remaining)} ticker(s) "
                f"not in Polygon: {remaining}"
            )
        for i, ticker in enumerate(remaining):
            try:
                t, hist = _fetch_ticker_yf(ticker)
                if t is None or hist is None:
                    logger.warning(f"[market_data] No data for {ticker} — skipping")
                    continue

                info          = t.fast_info
                current_price = None
                if current_session() != "rth":
                    current_price = _extended_last_price(t)
                if current_price is None:
                    current_price = float(info.last_price)
                prev_close    = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current_price
                pct_change    = (current_price - prev_close) / prev_close * 100
                week_open     = float(hist["Close"].iloc[0])
                week_return   = (current_price - week_open) / week_open * 100

                snapshots.append(TickerSnapshot(
                    ticker=ticker,
                    price=current_price,
                    pct_change_1d=round(pct_change, 2),
                    pct_change_5d=round(week_return, 2),
                    volume=int(info.three_month_average_volume or 0),
                    market_cap=getattr(info, "market_cap", None),
                ))
                logger.debug(
                    f"[market_data] {ticker}: ${current_price:.2f} "
                    f"({pct_change:+.2f}% 1d) [yfinance]"
                )

                if i < len(remaining) - 1:
                    time.sleep(INTER_TICKER)

            except _RateLimitAbort as e:
                logger.error(
                    f"[market_data] {e}. Stopping yfinance fallback early — "
                    f"{len(snapshots)}/{len(tickers)} tickers fetched. "
                    "Pipeline will continue with news-only signals."
                )
                break
            except Exception as e:
                logger.warning(f"[market_data] Unexpected error for {ticker}: {e}")

    # ── 3. Deterministic grouped-daily close fallback ─────────────────────
    # Polygon's per-ticker snapshot endpoint 403s on the free tier and the
    # yfinance fallback rate-limits/aborts on a wide universe, leaving many
    # tickers with NO price (~63% on a 568-name discovered universe). The
    # grouped-daily aggregates endpoint DOES work on the free tier and returns
    # every US ticker's completed close in ONE call — deterministic and
    # near-100% coverage. Fill anything still uncovered with the last completed
    # close, marked price_source="prev_close" so it's NOT a live quote (the
    # price-provenance check skips these; live execution uses tracker._fetch_price,
    # never this bulk snapshot). A recent close beats a null for the learning
    # panel + scoring context.
    covered_now = {s.ticker for s in snapshots}
    still = [t for t in tickers if t not in covered_now]
    if still:
        try:
            grouped = polygon_client.get_grouped_daily_closes()
        except Exception as e:
            logger.warning(f"[market_data] grouped-daily close fallback failed: {e}")
            grouped = {}
        filled = 0
        for ticker in still:
            # Class shares: our universe uses 'BRK-B', Polygon grouped uses 'BRK.B'.
            close = grouped.get(ticker) or grouped.get(ticker.replace("-", "."))
            if close and close > 0:
                snapshots.append(TickerSnapshot(
                    ticker=ticker, price=float(close),
                    pct_change_1d=0.0, pct_change_5d=0.0, volume=0,
                    market_cap=None, price_source="prev_close",
                ))
                filled += 1
        if filled:
            logger.info(
                f"[market_data] grouped-daily close fallback filled {filled}/{len(still)} "
                "uncovered ticker(s) with the last completed close (deterministic)"
            )

    source = (
        "yfinance" if not polygon_client.is_available()
        else ("polygon" if not remaining else "polygon+yfinance")
    )
    logger.info(
        f"[market_data] Fetched snapshots for {len(snapshots)}/{len(tickers)} tickers [{source}]"
    )
    return snapshots


def _normalize_index_tz(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return a copy of df with the index normalised to tz-naive UTC.

    Why: yfinance returns tz-aware timestamps (often America/New_York or, after
    a JSON round-trip through the cache, UTC at 04:00/05:00); Polygon returns
    tz-naive timestamps at midnight UTC of the session date. Mixing these in
    a single concat triggers ``TypeError: Cannot compare tz-naive and tz-aware
    timestamps`` on the subsequent sort. Both representations ultimately point
    at the same NYSE session, so collapsing to tz-naive UTC is safe and gives
    sort_index something it can actually compare.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df = df.copy()
        df.index = idx.tz_convert("UTC").tz_localize(None)
    return df


def _merge_ohlcv(cached: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine cached bars with a fresh fetch, preferring fresh on overlap.

    Rationale: a force-refresh fetch typically covers only the last 3 months,
    but the cache may already hold years of older history.  Naive overwrite
    would silently shrink the cache.  Merging keeps the longest possible
    history while letting the fresh bars rule within their window so any
    split / dividend rescaling is propagated forward — the cache stays
    internally consistent on the new adjustment scale for everything in the
    fresh window, and retains pre-fresh-window bars as-is.

    Old rows whose dates fall inside the fresh window are dropped (the fresh
    bars are the source of truth for that range — they reflect the current
    adjustment scale).  Rows older than the fresh window's first bar are
    kept untouched.

    Both inputs go through ``_normalize_index_tz`` first so a tz-aware cache
    + tz-naive fresh fetch (or vice versa) doesn't crash the sort.
    """
    if fresh is None or fresh.empty:
        return cached if cached is not None else pd.DataFrame()
    if cached is None or cached.empty:
        return _normalize_index_tz(fresh)

    cached = _normalize_index_tz(cached)
    fresh = _normalize_index_tz(fresh)

    fresh_dates = {ts.date() for ts in fresh.index}
    keep_mask   = [ts.date() not in fresh_dates for ts in cached.index]
    cached_kept = cached.iloc[keep_mask]
    if cached_kept.empty:
        return fresh
    combined = pd.concat([cached_kept, fresh]).sort_index()
    # Drop any accidental duplicate timestamps (defensive — shouldn't happen
    # after the keep_mask filter, but tz-aware/naive mixes can sneak through).
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def get_history(ticker: str, period: str = "3mo", force_refresh: bool = False,
                interval: str = "1d") -> pd.DataFrame:
    """
    Return OHLCV history for chart generation / technical analysis.

    Checks the OHLCV disk cache first (TTL 3 days).  On a cache miss:
      1. Tries Polygon.io (no per-IP rate-limit concern).
      2. Falls back to yfinance with exponential-backoff retry.
    Successful fetches are **merged** with whatever was already cached
    (`_merge_ohlcv`): fresh bars take precedence inside their window so any
    split / dividend rescaling propagates, but older history outside the
    fresh window is retained.  This prevents the force-refresh path from
    silently truncating long-tail OHLCV history.

    ``force_refresh``: bypass the TTL check and re-fetch even when the
    cached last bar is within the 3-day window. Used by the performance
    tracker to keep open-trade OHLCV fully up to date.

    ``interval``: "1d" (default, the daily path below) or "30m" (yfinance-only
    intraday path — see ``_get_intraday_history``). Weekly bars come from
    ``get_weekly_history`` (resampled from the daily cache), not this function.
    """
    if interval != "1d":
        return _get_intraday_history(ticker, interval=interval, force_refresh=force_refresh)

    from src.data.cache import load_ohlcv, save_ohlcv
    from datetime import date as _date

    # Fail fast on a junk ticker ("N/A" etc.) so it never reaches yfinance — which
    # raises an opaque "'Response' object is not subscriptable" on a bad symbol and
    # spams the log (268× in one day). Defense-in-depth: the pipeline also sanitizes
    # the universe, but the tracker / liquidity warm-up / other callers route here too.
    if not is_valid_ticker(ticker):
        logger.debug(f"[market_data] get_history: skipping invalid ticker {ticker!r}")
        return pd.DataFrame()

    cached = load_ohlcv(ticker)
    if cached is not None and not cached.empty and not force_refresh:
        last_bar = cached.index[-1].date()
        if (_date.today() - last_bar).days <= 3:
            logger.debug(f"[market_data] get_history: cache hit for {ticker} (last bar {last_bar})")
            return _drop_forming_bar(cached)

    if not settings.enable_fetch_data:
        logger.debug(f"[market_data] ENABLE_FETCH_DATA=false — skipping history fetch for {ticker}")
        return cached if (cached is not None and not cached.empty) else pd.DataFrame()

    # ── 1. Polygon ────────────────────────────────────────────────────────
    df = polygon_client.get_bars(ticker, period)
    if not df.empty:
        merged = _drop_forming_bar(_merge_ohlcv(cached, df))
        save_ohlcv(ticker, merged)
        logger.debug(
            f"[market_data] get_history: fetched {len(df)} bars for {ticker} [polygon] "
            f"(cache now {len(merged)} bars)"
        )
        return merged

    # ── 2. yfinance fallback ──────────────────────────────────────────────
    rl_hits = 0
    while True:
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d")
            if not df.empty:
                merged = _drop_forming_bar(_merge_ohlcv(cached, df))
                save_ohlcv(ticker, merged)
                time.sleep(INTER_TICKER)
                logger.debug(
                    f"[market_data] get_history: fetched {len(df)} bars for {ticker} [yfinance] "
                    f"(cache now {len(merged)} bars)"
                )
                return merged
            logger.warning(f"[market_data] get_history: empty data for {ticker}")
            return cached if (cached is not None and not cached.empty) else pd.DataFrame()

        except Exception as e:
            if _is_rate_limit(e):
                if rl_hits >= MAX_RL_HITS:
                    logger.error(f"[market_data] get_history rate-limited out for {ticker}")
                    return cached if (cached is not None and not cached.empty) else pd.DataFrame()
                _backoff_wait(rl_hits)
                rl_hits += 1
            else:
                logger.warning(f"[market_data] get_history failed for {ticker}: {e}")
                return cached if (cached is not None and not cached.empty) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Multi-timeframe history — 30-minute (intraday) and weekly
# ---------------------------------------------------------------------------

def _fetch_intraday_yf(ticker: str, interval: str = "30m") -> pd.DataFrame:
    """Fetch intraday OHLCV via yfinance (RTH only). Fail-soft per ticker.

    yfinance caps 30-minute history at ~60 days. On a rate limit we give up on
    THIS ticker immediately rather than running the daily path's 60–240 s
    backoff — blocking a 30-min tick on hundreds of sleeps would be worse than
    a patchy 30m panel (the ticker simply scores on daily/weekly this tick).
    """
    try:
        hist = yf.Ticker(ticker).history(period="60d", interval=interval)
    except Exception as e:
        if _is_rate_limit(e):
            logger.debug(f"[market_data] {interval} rate-limited for {ticker} — skipping")
        else:
            logger.debug(f"[market_data] {interval} fetch failed for {ticker}: {e}")
        return pd.DataFrame()
    if hist is None or hist.empty:
        return pd.DataFrame()
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in hist.columns]
    return hist[cols] if cols else pd.DataFrame()


def _merge_intraday(cached: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine cached intraday bars with a fresh fetch, deduping by exact
    timestamp (preferring fresh on overlap).

    Unlike ``_merge_ohlcv`` (which dedupes by *date*, right for one-bar-per-day
    daily history) this keeps intraday granularity, so the 30-min cache grows
    beyond yfinance's 60-day window over successive runs."""
    if fresh is None or fresh.empty:
        return cached if cached is not None else pd.DataFrame()
    fresh = _normalize_index_tz(fresh)
    if cached is None or cached.empty:
        return fresh
    cached = _normalize_index_tz(cached)
    combined = pd.concat([cached, fresh]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def _get_intraday_history(ticker: str, interval: str = "30m",
                          force_refresh: bool = False) -> pd.DataFrame:
    """Intraday OHLCV with a short-TTL cache (``interval`` namespace).

    The forming bar changes every tick, so the cache uses a minute-level TTL
    (``intraday_30m_ttl_minutes``) instead of the daily 3-day TTL. RTH 30-min
    bars don't change off-hours (we fetch RTH-only), so when the market is
    closed a non-empty cache is reused without a refetch."""
    from src.data.cache import load_ohlcv, save_ohlcv

    if interval != "30m" or not is_valid_ticker(ticker):
        return pd.DataFrame()

    cached = load_ohlcv(ticker, interval)
    have_cache = cached is not None and not cached.empty
    if have_cache and not force_refresh:
        if current_session() != "rth":
            return _drop_forming_bar(cached, interval)   # RTH 30m bars are static off-hours
        last = _ts_to_utc(cached.index[-1])
        if last is not None:
            age_min = (_datetime.now(_timezone.utc) - last).total_seconds() / 60.0
            if age_min <= settings.intraday_30m_ttl_minutes:
                return _drop_forming_bar(cached, interval)

    if not settings.enable_fetch_data:
        return _drop_forming_bar(cached, interval) if have_cache else pd.DataFrame()

    fresh = _fetch_intraday_yf(ticker, interval)
    if fresh is not None and not fresh.empty:
        merged = _drop_forming_bar(_merge_intraday(cached, fresh), interval)
        save_ohlcv(ticker, merged, interval)
        logger.debug(
            f"[market_data] {interval}: fetched {len(fresh)} bars for {ticker} "
            f"(cache now {len(merged) if merged is not None else 0} bars)"
        )
        return merged
    return _drop_forming_bar(cached, interval) if have_cache else pd.DataFrame()


def _resample_weekly(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Resample a daily OHLCV frame to weekly bars (week ending Friday)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = _normalize_index_tz(df)
    agg = {}
    for col, how in (("Open", "first"), ("High", "max"), ("Low", "min"),
                     ("Close", "last"), ("Volume", "sum")):
        if col in d.columns:
            agg[col] = how
    if "Close" not in agg:
        return pd.DataFrame()
    weekly = d.resample("W-FRI").agg(agg).dropna(subset=["Close"])
    return weekly


def get_weekly_history(ticker: str) -> pd.DataFrame:
    """Weekly OHLCV resampled from the daily cache (no extra fetch).

    Uses whatever daily history is already cached (warmed by the daily scorers
    each tick); only fetches if the daily cache is empty. The current,
    still-forming week is dropped so weekly indicators read completed bars only.
    A thin daily cache yields few weekly bars — the weekly scorers fail-soft to
    'no view' below their minimum-row thresholds."""
    from src.data.cache import load_ohlcv

    if not is_valid_ticker(ticker):
        return pd.DataFrame()
    daily = load_ohlcv(ticker)
    if daily is None or daily.empty:
        daily = get_history(ticker, period="18mo")
    if daily is None or daily.empty:
        return pd.DataFrame()
    return _drop_forming_bar(_resample_weekly(daily), interval="1w")
