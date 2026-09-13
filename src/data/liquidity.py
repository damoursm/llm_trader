"""Discovery liquidity gate (Section F).

Sections A–E widen the discovery funnel — trending/open-vocabulary, the opportunity screener,
the macro→discovery loop, market-wide earnings/analyst catalysts, and cointegration peers. Without
a uniform quality floor that breadth would inject untradeable microcaps: exactly the names the
tracker's bid-ask model charges up to 250 bp a side (``_dynamic_half_spread``), which quietly
destroys realised P&L. This module applies one consistent filter — a minimum price and a minimum
20-day average dollar volume — to every DISCOVERED candidate before it enters the analysis universe.
``dollar_volume`` is THE definition of that average for every liquidity gate in the system
(discovery, Gate 4, the rank pool, follow-through, the Tier-2/NBBO pools, the screener and the
short-horizon MR floors); ``is_liquid`` is the shared fail-closed verdict.

What is NOT gated: the pinned/intentional universe (static watchlist, sector ETFs, commodities,
factor/thematic ETFs) and open-trade tickers — the caller decides those and passes only discovered
names here.

Data is loaded cache-first with a bounded warm-up fetch (shared across a run via the ``budget``
dict), mirroring the screener. A candidate whose liquidity cannot be verified (no/short data and
no fetch budget left) is dropped — **fail-closed**: the gate's whole purpose is to keep
unverifiable/illiquid names out, and a genuinely liquid name re-appears next run once its OHLCV
cache is warm. The base watchlist is never subject to this, so dropping a discovered name is safe.
"""

from typing import Dict, List, Optional, Iterable

import pandas as pd
from loguru import logger

from config import settings
from src.data.cache import load_ohlcv
from src.data.market_data import get_history, is_valid_ticker, is_exotic_security

_CACHE_MIN_BARS = 20    # accept the cache outright at/above this many bars
_EVAL_MIN_BARS  = 10    # below this, too little data to judge liquidity → fail-closed
DOLLAR_VOLUME_WINDOW = 20   # the "20-day average dollar volume" every gate names


def dollar_volume(df: Optional[pd.DataFrame], window: int = DOLLAR_VOLUME_WINDOW,
                  min_bars: int = 1) -> Optional[float]:
    """Trailing MEAN of Close × Volume over the last ``window`` valid bars — the
    ONE definition of "20-day average dollar volume" (2026-09-02 audit).

    Seven sites used to compute this three different ways, and the live Gate 4
    was the outlier: ``mean(Volume[-20:]) × Close[-1]`` multiplies a 20-day
    volume average by a SINGLE price, so it admits recent rallies and rejects
    recent declines (names admitted only by that formula had a median 20d
    return of +36.5%, those admitted only by this one −22.7%) — the chase
    direction Gate 5 exists to fight — and it read HYMCW at 13.6M against a true
    14.2k (961×). The per-bar product is what the setting names and docstrings
    already claimed, and what the funnel's point-in-time simulation
    (``gate_funnel._pit_ohlcv_features``) was already judging on.

    None when the frame is unusable or fewer than ``min_bars`` valid rows exist
    (callers treat None as fail-closed). NOT for ``spread.adv_of`` — the cost
    liquidity classes were FITTED on its median-60 basis.
    """
    if df is None or getattr(df, "empty", True):
        return None
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    try:
        sub = df[["Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        dollar = (sub["Close"] * sub["Volume"]).tail(max(1, int(window)))
        if len(dollar) < max(1, int(min_bars)):
            return None
        val = float(dollar.mean())
        return val if val == val else None
    except Exception as e:
        logger.debug(f"[liquidity] dollar_volume failed: {e}")
        return None


def _load(ticker: str, budget: Dict[str, int]) -> Optional[pd.DataFrame]:
    """Cache-first OHLCV; one bounded warm-up fetch when the cache is cold and allowed.

    ``budget["attempted"]`` (optional set) marks tickers the concurrent
    pre-warm below already tried — they are never re-fetched here, so a failed
    pre-warm can't double-spend the budget or re-trigger a rate-limit backoff.
    """
    df = load_ohlcv(ticker)
    if df is not None and not df.empty and len(df) >= _CACHE_MIN_BARS:
        return df
    attempted = budget.get("attempted")
    already_attempted = attempted is not None and ticker in attempted
    if settings.enable_fetch_data and budget.get("n", 0) > 0 and not already_attempted:
        budget["n"] -= 1
        try:
            fetched = get_history(ticker, period="3mo")
            if fetched is not None and not fetched.empty:
                return fetched
        except Exception as e:
            logger.debug(f"[liquidity] fetch failed for {ticker}: {e}")
    return df


def _prewarm_cold(cands: List[str], budget: Dict[str, int]) -> None:
    """Concurrently warm the OHLCV cache for cold candidates, within the budget.

    The sequential per-ticker cold fetch inside ``is_liquid`` was the 7-minute
    stall on the first tick after midnight (2026-07-08 profile: ~200 uncached
    smart-money names × sequential get_history with backoff). ``get_history``
    is Polygon-first (no per-IP rate concern; yfinance only as fallback) and
    SAVES to the OHLCV cache, so after this pass the ``is_liquid`` loop below
    runs entirely on warm cache. Fail-soft: any fetch error just leaves that
    ticker cold and it fails the gate closed, exactly as before.
    """
    workers = max(1, int(getattr(settings, "liquidity_gate_fetch_workers", 1) or 1))
    if workers <= 1 or not settings.enable_fetch_data or budget.get("n", 0) <= 0:
        return
    cold = []
    for t in cands:
        if budget.get("n", 0) - len(cold) <= 0:
            break
        df = load_ohlcv(t)
        if df is None or df.empty or len(df) < _CACHE_MIN_BARS:
            cold.append(t)
    if len(cold) <= 1:
        return
    attempted = budget.setdefault("attempted", set())

    def _fetch(t: str) -> None:
        try:
            get_history(t, period="3mo")
        except Exception as e:
            logger.debug(f"[liquidity] pre-warm fetch failed for {t}: {e}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(cold)),
                            thread_name_prefix="liq-warm") as ex:
        list(ex.map(_fetch, cold))
    budget["n"] = max(0, budget.get("n", 0) - len(cold))
    attempted.update(cold)
    logger.info(f"[liquidity] pre-warmed {len(cold)} cold ticker(s) concurrently "
                f"({budget['n']} fetch budget left)")


def tradeable_pool(tickers, budget=None) -> set:
    """The Gate-4-eligible subset of ``tickers``, CACHE-ONLY (budget n=0 — never
    spends an API call; an uncached name fails CLOSED and stays out).

    One definition for every consumer that needs "the names this book could
    actually trade" as a SET rather than a per-candidate verdict: the rank pool
    the combine ranks over, and the mechanical entry selector
    (`src/signals/rank_entry.py`). `build_signals` keeps its own inline copy
    because it also carries the universe-wide collapse guard — that is the one
    place every scored name's verdict exists per run.
    """
    from config import settings as _s

    b = {"n": 0} if budget is None else budget
    out = set()
    for tk in tickers:
        try:
            if is_liquid(tk, b, _s.trade_min_price, _s.trade_min_dollar_volume):
                out.add(tk)
        except Exception:                      # fail CLOSED, per is_liquid itself
            continue
    return out



def is_liquid(
    ticker: str,
    budget: Dict[str, int],
    min_price: float,
    min_dollar_volume: float,
    price: Optional[float] = None,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """True iff ``ticker`` clears the price + 20-day average dollar-volume floors.

    The price floor is tested on the last VALID cached close AND, when the caller
    has one, the tick's ``price`` (the live snapshot) — both must clear. The
    cached close alone is stale by up to a session: measured 2026-09-02, 0.27%
    of panel rows straddled the $5 floor and 10 of 16 in the dangerous direction
    (live below, cached above → wrongly tradeable). A wrong SNAPSHOT (it
    happens) can only defer a name to the next tick, never admit one.
    ``df`` lets a caller that already holds the OHLCV frame skip the cache read.

    Fail-closed: returns False when liquidity cannot be verified (missing columns,
    too few bars, or no data after the bounded fetch attempt).
    """
    try:
        if price is not None and float(price) < float(min_price):
            return False
    except (TypeError, ValueError):
        pass                                   # an unparseable live price is "no live price"
    if df is None:
        df = _load(ticker, budget)
    if df is None or df.empty:
        return False
    if "Close" not in df.columns or "Volume" not in df.columns:
        return False
    try:
        # Drop NaN rows before evaluating: the last bar is often the still-forming
        # intraday bar (or a trailing yfinance NaN), so reading iloc[-1] raw made
        # `NaN * NaN >= floor` → False and wrongly gated out liquid large-caps
        # during RTH. Judge on the last VALID completed bars instead.
        sub = df[["Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < _EVAL_MIN_BARS:
            return False
        last_close = float(sub["Close"].iloc[-1])
        if last_close < min_price:
            return False
        dv = dollar_volume(sub, DOLLAR_VOLUME_WINDOW, min_bars=_EVAL_MIN_BARS)
        return dv is not None and dv >= float(min_dollar_volume)
    except Exception as e:
        # The module's one formerly-UNLOGGED failure path (2026-09-02 audit): a
        # systematic breakage here (pandas API, corrupt cache) would gate out the
        # whole universe and read exactly like "no liquid names today". Logged,
        # and the actionable filter carries a universe-wide guard on top.
        logger.debug(f"[liquidity] is_liquid({ticker}) failed — fail-closed: {e}")
        return False


def apply_liquidity_gate(
    candidates: Iterable[str],
    source: str = "discovery",
    min_price: Optional[float] = None,
    min_dollar_volume: Optional[float] = None,
    budget: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Return the subset of ``candidates`` that clears the discovery liquidity gate.

    De-duplicates and upper-cases input. A no-op (returns the cleaned list unchanged) when
    ``enable_discovery_liquidity_gate`` is false, so callers can route every discovery source
    through it unconditionally. ``budget`` is a shared mutable ``{"n": int}`` cold-fetch allowance
    so repeated calls within one run together stay under ``discovery_gate_max_fetch``.
    """
    seen: set = set()
    cands: List[str] = []
    exotic: List[str] = []
    for c in candidates:
        s = (c or "").strip().upper()
        if not s or s in seen or not is_valid_ticker(s):   # drop junk ("N/A" etc.) up front
            continue
        if settings.enable_security_type_filter and is_exotic_security(s):
            exotic.append(s)                               # preferred/warrant/unit/OTC-foreign
            continue
        seen.add(s)
        cands.append(s)
    if exotic:
        logger.info(
            f"[liquidity] {source}: dropped {len(exotic)} exotic security-type(s) "
            f"(preferred/warrant/unit/OTC-foreign): {exotic[:12]}{' …' if len(exotic) > 12 else ''}"
        )
    if not cands or not settings.enable_discovery_liquidity_gate:
        return cands

    mp  = settings.discovery_min_price if min_price is None else min_price
    mdv = settings.discovery_min_dollar_volume if min_dollar_volume is None else min_dollar_volume
    if budget is None:
        budget = {"n": max(0, settings.discovery_gate_max_fetch)}

    # Warm the cold candidates' OHLCV concurrently first (bounded by the shared
    # budget), so the sequential verdict loop below reads warm cache throughout.
    _prewarm_cold(cands, budget)

    kept: List[str] = []
    dropped: List[str] = []
    for t in cands:
        (kept if is_liquid(t, budget, mp, mdv) else dropped).append(t)

    if dropped:
        logger.info(
            f"[liquidity] {source}: gated out {len(dropped)} illiquid/unverified "
            f"(<${mp:.0f} or <${mdv / 1e6:.0f}M 20d ADV): "
            f"{dropped[:12]}{' …' if len(dropped) > 12 else ''}"
        )
    return kept
