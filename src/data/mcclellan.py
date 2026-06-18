"""
McClellan Oscillator & Summation Index — NYSE advance/decline breadth momentum.

The McClellan Oscillator measures the momentum of market breadth by applying two
exponential moving averages to the daily net advances (advancing NYSE issues minus
declining issues):

  McClellan Oscillator = EMA(19) of Net Advances − EMA(39) of Net Advances

  When positive: the 19-day EMA is above the 39-day EMA → breadth accelerating
  When negative: the 19-day EMA is below the 39-day EMA → breadth decelerating

The McClellan Summation Index is the running cumulative total of oscillator readings
and gives a longer-term perspective on market breadth:

  Summation Index(t) = Summation Index(t-1) + Oscillator(t)

Oscillator signal thresholds:
  > +100   OVERBOUGHT        → contrarian BEARISH (momentum stretched; fade rallies)
  > +50    BULLISH_MOMENTUM  → BULLISH (strong net advances; support buying)
  -50–50   NEUTRAL           → no directional edge
  < -50    BEARISH_MOMENTUM  → BEARISH (net declines dominating; avoid longs)
  < -100   OVERSOLD          → contrarian BULLISH (capitulation; coiling for reversal)

Summation Index thresholds (relative to our data window):
  > +500   EXTENDED_BULL → trend overstretched; reduce new long exposure
  > 0      BULL_TREND    → positive breadth trend; buy dips
  < 0      BEAR_TREND    → negative breadth trend; sell rallies
  < -500   EXTENDED_BEAR → deeply oversold trend; approaching major reversal zone

Zero-line crossings of the oscillator are the best swing-trade timing signals:
  Crossing above 0 → bullish momentum shift (EMA19 now above EMA39)
  Crossing below 0 → bearish momentum shift (EMA19 now below EMA39)

Data source: Polygon grouped-daily (every US stock's OHLC in one call) →
Ratio-Adjusted Net Advances (RANA). Yahoo delisted ^NYAD and every A/D variant
(0 bars), so the legacy yfinance path was a silent zombie. RANA normalises the
advance/decline count to a fixed scale so the classic oscillator thresholds still
apply regardless of how many issues trade:

    RANA = 1000 × (advancing − declining) / (advancing + declining)

where advancing/declining is each stock's close vs its PRIOR close. The daily
RANA series is accumulated in cache/mcclellan_ad_history.json (one Polygon call
per new trading day; daily-cached so only the first tick of each day fetches).
One-time backfill:  python -m src.data.mcclellan
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
from loguru import logger

from config import settings
from src.models import McClellanContext

CACHE_DIR  = Path("cache")
_EMA_SHORT = 19       # "10% trend" — fast EMA
_EMA_LONG  = 39       # "5% trend"  — slow EMA

# Polygon grouped-daily breadth source (replaces the delisted ^NYAD)
_POLY_GROUPED    = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"
_AD_HISTORY_PATH = CACHE_DIR / "mcclellan_ad_history.json"
_HEADERS         = {"User-Agent": "llm-trader research"}
_BACKFILL_DAYS   = 55       # trading days to seed (≥ _EMA_LONG+10 = 49 required)
_MAX_FETCH_RUN   = 6        # cap Polygon calls per normal run (daily-cached → ~1 fetch/day)
_POLY_DELAY      = 13.0     # seconds between grouped calls (free tier ≈ 5/min)
_MIN_PRICE       = 1.0      # exclude sub-$1 names from the A/D count (penny noise)

# Oscillator thresholds
_OB_HIGH = 100.0   # overbought
_OB_MID  =  50.0   # bullish momentum
_OS_MID  = -50.0   # bearish momentum
_OS_HIGH = -100.0  # oversold

# Summation thresholds. The RANA-based oscillator cumsums to a larger range than
# the legacy raw-net-advance scale, so these match the classic McClellan SI
# "extended" convention (~±2000) rather than the old ±500 (which pinned the
# reading at EXTENDED). The Summation Index grows toward a stable continuous
# series as the A/D history accumulates day by day.
_SI_BULL_EXT =  2000.0
_SI_BEAR_EXT = -2000.0


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"mcclellan_{date.today().isoformat()}.json"


def _load_cache() -> Optional[McClellanContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = McClellanContext.model_validate(json.loads(path.read_text(encoding="utf-8")))
        logger.info(
            f"[mcclellan] Loaded from cache — osc={ctx.oscillator:+.1f} ({ctx.osc_signal}), "
            f"SI={ctx.summation:+.0f} ({ctx.sum_signal})"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[mcclellan] Cache load failed: {e}")
        return None


def _save_cache(ctx: McClellanContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[mcclellan] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────────────────────

def _classify_oscillator(osc: float) -> str:
    if osc >  _OB_HIGH: return "OVERBOUGHT"
    if osc >  _OB_MID:  return "BULLISH_MOMENTUM"
    if osc >= _OS_MID:  return "NEUTRAL"
    if osc >= _OS_HIGH: return "BEARISH_MOMENTUM"
    return "OVERSOLD"


def _classify_summation(si: float) -> str:
    if si >  _SI_BULL_EXT: return "EXTENDED_BULL"
    if si >  0:            return "BULL_TREND"
    if si >= _SI_BEAR_EXT: return "BEAR_TREND"
    return "EXTENDED_BEAR"


def _compute_direction(
    osc: float,
    osc_prev: Optional[float],
    si: float,
    is_bull_cross: bool,
    is_bear_cross: bool,
) -> str:
    """Determine directional bias. Crossings take priority."""
    if is_bull_cross:
        return "BULLISH"
    if is_bear_cross:
        return "BEARISH"
    # Contrarian at extremes: oversold + turning up / overbought + turning down
    if osc <= _OS_HIGH and osc_prev is not None and osc > osc_prev:
        return "BULLISH"   # deeply oversold and starting to recover
    if osc >= _OB_HIGH and osc_prev is not None and osc < osc_prev:
        return "BEARISH"   # overbought and rolling over
    # Momentum + trend alignment
    if osc >= _OB_MID and si > 0:
        return "BULLISH"
    if osc <= _OS_MID and si < 0:
        return "BEARISH"
    # Summation-only direction
    if si >  200:
        return "BULLISH"
    if si < -200:
        return "BEARISH"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Breadth A/D source — Polygon grouped-daily → Ratio-Adjusted Net Advances
# ─────────────────────────────────────────────────────────────────────────────

def _load_ad_history() -> Dict[str, float]:
    if not _AD_HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(_AD_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[mcclellan] A/D history load failed: {e}")
        return {}


def _save_ad_history(history: Dict[str, float]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _AD_HISTORY_PATH.write_text(json.dumps(history, indent=0, sort_keys=True), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[mcclellan] A/D history save failed: {e}")


def _recent_weekdays(n: int) -> List[date]:
    """The last ``n`` weekdays up to yesterday (oldest first). Holidays are
    fetched too but Polygon returns no results for them, so they're skipped."""
    out: List[date] = []
    d = date.today() - timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def _prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _grouped_closes(iso_day: str) -> Dict[str, float]:
    """{ticker: close} for every US stock on ``iso_day`` (empty for a non-trading
    day). Retries once on a 429 (free-tier rate limit)."""
    for attempt in range(2):
        r = httpx.get(_POLY_GROUPED.format(date=iso_day),
                      params={"adjusted": "true", "apiKey": settings.polygon_api_key},
                      headers=_HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(_POLY_DELAY)
            continue
        r.raise_for_status()
        res = r.json().get("results") or []
        return {x["T"]: x["c"] for x in res
                if x.get("T") and isinstance(x.get("c"), (int, float)) and x["c"] > 0}
    return {}


def _rana(today: Dict[str, float], prev: Dict[str, float]) -> float:
    """Ratio-Adjusted Net Advances: 1000 × (adv − dec) / (adv + dec), each
    stock's close vs its prior close, sub-$1 names excluded."""
    adv = dec = 0
    for t, c in today.items():
        pc = prev.get(t)
        if pc is None or c < _MIN_PRICE or pc < _MIN_PRICE:
            continue
        if c > pc:
            adv += 1
        elif c < pc:
            dec += 1
    tot = adv + dec
    return round(1000.0 * (adv - dec) / tot, 3) if tot else 0.0


def _build_rana_series(max_fetch: int = _MAX_FETCH_RUN, backfill: bool = False) -> Optional[pd.Series]:
    """Daily RANA series, accumulated in cache. Normal runs fetch only the few
    newest missing trading days (daily-cached → ~1/day); ``backfill=True`` seeds
    the full window. Returns None until ``_EMA_LONG+10`` days exist."""
    if not settings.polygon_api_key:
        logger.warning("[mcclellan] POLYGON_API_KEY not set — cannot build breadth A/D")
        return None

    history = _load_ad_history()
    if not history and not backfill:
        logger.warning("[mcclellan] no breadth A/D history yet — run the one-time "
                       "backfill: python -m src.data.mcclellan")
        return None

    target = _BACKFILL_DAYS + 12 if backfill else 7
    weekdays = _recent_weekdays(target)
    missing = [d for d in weekdays if d.isoformat() not in history]
    if missing:
        # Fetch closes from the weekday before the earliest gap (baseline) onward;
        # cap per normal run, but take the NEWEST days so incremental stays recent.
        cap = (10 ** 6) if backfill else max_fetch
        fetch_days = [d for d in weekdays if d >= _prev_weekday(min(missing))][-cap:]
        closes: Dict[str, Dict[str, float]] = {}
        for i, d in enumerate(fetch_days):
            try:
                c = _grouped_closes(d.isoformat())
            except Exception as e:
                logger.debug(f"[mcclellan] grouped {d} failed: {e}")
                c = {}
            if c:
                closes[d.isoformat()] = c
            if i < len(fetch_days) - 1:
                time.sleep(_POLY_DELAY)
        ordered = sorted(closes)
        added = 0
        for i in range(1, len(ordered)):
            ds, prev = ordered[i], ordered[i - 1]
            if ds not in history:
                history[ds] = _rana(closes[ds], closes[prev])
                added += 1
        if added:
            _save_ad_history(history)
            logger.info(f"[mcclellan] breadth A/D: +{added} day(s) via Polygon, {len(history)} cached")

    if len(history) < _EMA_LONG + 10:
        logger.warning(f"[mcclellan] only {len(history)} A/D day(s) — need {_EMA_LONG + 10}; "
                       "run `python -m src.data.mcclellan` to backfill")
        return None
    return pd.Series({pd.Timestamp(k): v for k, v in history.items()}).sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mcclellan_context() -> Optional[McClellanContext]:
    """
    Compute the McClellan Oscillator and Summation Index from market-wide
    advance/decline breadth (Polygon RANA). Returns McClellanContext or None if
    data is unavailable. Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[mcclellan] ENABLE_FETCH_DATA=false — skipping breadth fetch")
        return None

    net_adv = _build_rana_series()
    if net_adv is None or len(net_adv) < _EMA_LONG + 10:
        return None

    # McClellan Oscillator = EMA(19) − EMA(39) of net advances
    ema19_s = net_adv.ewm(span=_EMA_SHORT, adjust=False).mean()
    ema39_s = net_adv.ewm(span=_EMA_LONG,  adjust=False).mean()
    osc_s   = ema19_s - ema39_s

    # McClellan Summation Index = cumulative sum of oscillator (relative to our window)
    si_s = osc_s.cumsum()

    # Extract current and historical snapshots
    osc_now  = float(osc_s.iloc[-1])
    osc_5d   = float(osc_s.iloc[-6]) if len(osc_s) >= 6 else None
    si_now   = float(si_s.iloc[-1])
    si_5d    = float(si_s.iloc[-6])  if len(si_s) >= 6  else None
    ema19_now = float(ema19_s.iloc[-1])
    ema39_now = float(ema39_s.iloc[-1])

    # Detect zero crossings in the last 3 sessions
    if len(osc_s) >= 4:
        prev3 = osc_s.iloc[-4:-1].tolist()
        is_bull_cross = (osc_now >= 0) and any(v < 0 for v in prev3)
        is_bear_cross = (osc_now < 0)  and any(v >= 0 for v in prev3)
    else:
        is_bull_cross = is_bear_cross = False

    osc_signal = _classify_oscillator(osc_now)
    sum_signal = _classify_summation(si_now)
    direction  = _compute_direction(osc_now, osc_5d, si_now, is_bull_cross, is_bear_cross)

    # ── Summary ───────────────────────────────────────────────────────────────
    osc_desc = {
        "OVERBOUGHT":       "Net-advance momentum is stretched — breadth exhaustion; fade further rallies.",
        "BULLISH_MOMENTUM": "Strong positive breadth momentum — advancing issues dominating on a sustained basis.",
        "NEUTRAL":          "Breadth momentum neutral — no clear A/D directional edge.",
        "BEARISH_MOMENTUM": "Negative breadth momentum — declining issues dominating; avoid new longs.",
        "OVERSOLD":         "Breadth deeply oversold — potential mean-reversion and swing-low setup forming.",
    }.get(osc_signal, "")

    cross_str = ""
    if is_bull_cross:
        cross_str = " Oscillator just crossed ABOVE zero → bullish momentum shift confirmed."
    elif is_bear_cross:
        cross_str = " Oscillator just crossed BELOW zero → bearish momentum shift confirmed."

    si_trend_str = ""
    if si_5d is not None:
        si_delta = si_now - si_5d
        si_trend_str = f" (Δ{si_delta:+.0f} over 5 sessions)"

    osc_delta_str = ""
    if osc_5d is not None:
        osc_delta_str = f" (5d ago: {osc_5d:+.1f})"

    summary = (
        f"McClellan Oscillator: {osc_now:+.1f}{osc_delta_str} ({osc_signal}). "
        f"Summation Index: {si_now:+.0f}{si_trend_str} ({sum_signal}). "
        f"{osc_desc}{cross_str}"
    )

    ctx = McClellanContext(
        oscillator=round(osc_now, 2),
        oscillator_5d_ago=round(osc_5d, 2) if osc_5d is not None else None,
        summation=round(si_now, 1),
        summation_5d_ago=round(si_5d, 1) if si_5d is not None else None,
        ema19=round(ema19_now, 1),
        ema39=round(ema39_now, 1),
        osc_signal=osc_signal,
        sum_signal=sum_signal,
        direction=direction,
        is_bullish_cross=is_bull_cross,
        is_bearish_cross=is_bear_cross,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)
    cross_tag = " | BULL CROSS" if is_bull_cross else (" | BEAR CROSS" if is_bear_cross else "")
    logger.info(
        f"[mcclellan] Osc={osc_now:+.1f} ({osc_signal}) | SI={si_now:+.0f} ({sum_signal}) "
        f"| {direction}{cross_tag}"
    )
    return ctx


def backfill() -> None:
    """One-time seed of the breadth A/D history from Polygon grouped-daily.
    Throttled to the free-tier rate limit (~5/min), so ~55 days takes ~12 min.
    Safe to re-run — it only fetches days not already cached."""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    logger.info(f"[mcclellan] backfilling ~{_BACKFILL_DAYS} trading days of breadth A/D "
                "from Polygon (throttled for free tier — this takes a while)...")
    s = _build_rana_series(backfill=True)
    if s is None:
        logger.warning("[mcclellan] backfill produced insufficient history")
    else:
        logger.info(f"[mcclellan] backfill complete — {len(s)} A/D days cached")


if __name__ == "__main__":
    backfill()
