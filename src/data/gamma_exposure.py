"""
Gamma Exposure (GEX): options market structure as a short-term timing signal.

How it works
────────────
Market makers sell options to customers and delta-hedge with the underlying.
As price moves they must re-hedge, and the direction of that hedging determines
whether they stabilise or amplify price moves.

  Long gamma (positive GEX):  dealer SELLS rallies, BUYS dips → vol SUPPRESSED, price PINNED
  Short gamma (negative GEX): dealer BUYS rallies, SELLS dips → vol AMPLIFIED, moves ACCELERATE

Formula (SpotGamma / SqueezeMetrics convention)
───────────────────────────────────────────────
  GEX per strike = (Call_OI × Call_Γ − Put_OI × Put_Γ) × 100 × S²

  Assumes dealers are net SHORT calls (sold to retail/spec buyers) and net LONG puts
  (bought from institutional put-sellers).  Positive total GEX = calls dominate =
  stabilising.  Negative total GEX = puts dominate = destabilising.

Additional outputs
──────────────────
  Gamma flip      — price where cumulative GEX (scanning down from highest strike)
                    first crosses zero.  Breaking this level often accelerates moves.
  Max pain        — strike where total in-the-money options dollar value is minimised.
                    Options market makers profit maximally when price expires here;
                    there is a weak gravitational pull toward this level into expiry.
  Expected move   — ATM straddle price / spot → market-implied ±1σ move to nearest expiry.

Tickers covered
───────────────
  Always: SPY, QQQ, IWM (most liquid, most reliable GEX data).
  Plus: any watchlist ticker with total OI ≥ MIN_OI contracts.
  Skipped: futures-style tickers (contain "="), index symbols (start with "^"),
           and any ticker whose options chain is too thin.

Cached daily.
"""

import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_RATE_LIMIT_BACKOFF = [30, 60, 120]   # seconds to wait on successive 429s

import yfinance as yf
from loguru import logger

from config import settings
from src.models import GEXContext, GEXSignal

CACHE_DIR = Path("cache")
_REQUEST_DELAY = 0.40
_MIN_OI        = 1_000      # skip tickers with fewer total open-interest contracts
_MAX_EXPIRY_DAYS = 30       # only look at options expiring within this window
_MAX_EXPIRIES    = 3        # cap expiries processed per ticker to limit rate-limit exposure
_RISK_FREE_RATE  = 0.043    # approximate US risk-free rate (update periodically)

# Index ETFs always processed regardless of watchlist
_INDEX_TICKERS = ["SPY", "QQQ", "IWM"]

# GEX classification thresholds (applied to gex_normalized)
_PIN_THRESHOLD = 0.20       # > +0.20  → PINNED
_AMP_THRESHOLD = -0.20      # < -0.20  → AMPLIFIED


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"gex_{date.today().isoformat()}.json"


def _load_cache() -> Optional[GEXContext]:
    # Try today's cache first, then fall back to any recent file (up to 5 days old).
    # GEX data is valid for several days — max pain / gamma flip levels shift slowly.
    candidates = [_cache_path()] + sorted(
        CACHE_DIR.glob("gex_*.json"), reverse=True  # most recent first
    )
    for p in candidates:
        if not p.exists():
            continue
        try:
            ctx = GEXContext.model_validate(json.loads(p.read_text(encoding="utf-8")))
            age = (date.today() - ctx.report_date).days
            if age > 5:
                logger.debug(f"[gex] Cache {p.name} is {age}d old — skipping")
                continue
            label = "today" if age == 0 else f"{age}d old"
            logger.info(f"[gex] Loaded {len(ctx.signals)} signals from cache ({label}) — {ctx.summary}")
            return ctx
        except Exception as exc:
            logger.warning(f"[gex] Cache load failed ({p.name}): {exc}")
    return None


def _save_cache(ctx: GEXContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning(f"[gex] Cache save failed: {exc}")


# ── Black-Scholes gamma (no scipy dependency) ─────────────────────────────────

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes gamma (identical for calls and puts)."""
    if T < 1e-6 or sigma < 1e-6 or S <= 0 or K <= 0:
        return 0.0
    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (_RISK_FREE_RATE + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
        return _norm_pdf(d1) / (S * sigma * sqrt_T)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ── Per-expiry helpers ────────────────────────────────────────────────────────

def _gex_for_expiry(
    spot: float,
    expiry: str,
    chain,
) -> Tuple[Dict[float, float], int, int]:
    """
    Compute GEX per strike for one expiry.

    Returns:
        gex_by_strike  — {strike: net_gex_dollars}
        total_call_oi  — aggregate call open interest
        total_put_oi   — aggregate put open interest
    """
    T = max(0.0, (date.fromisoformat(expiry) - date.today()).days) / 365.0
    if T <= 0:
        return {}, 0, 0

    calls = chain.calls[["strike", "openInterest", "impliedVolatility"]].copy()
    puts  = chain.puts[["strike",  "openInterest", "impliedVolatility"]].copy()

    calls["openInterest"]    = calls["openInterest"].fillna(0).astype(int)
    puts["openInterest"]     = puts["openInterest"].fillna(0).astype(int)
    calls["impliedVolatility"] = calls["impliedVolatility"].fillna(0.0)
    puts["impliedVolatility"]  = puts["impliedVolatility"].fillna(0.0)

    total_call_oi = int(calls["openInterest"].sum())
    total_put_oi  = int(puts["openInterest"].sum())

    call_data = {
        float(row.strike): (int(row.openInterest), float(row.impliedVolatility))
        for row in calls.itertuples(index=False)
    }
    put_data = {
        float(row.strike): (int(row.openInterest), float(row.impliedVolatility))
        for row in puts.itertuples(index=False)
    }

    gex_by_strike: Dict[float, float] = {}
    for K in sorted(set(call_data) | set(put_data)):
        c_oi, c_iv = call_data.get(K, (0, 0.0))
        p_oi, p_iv = put_data.get(K, (0, 0.0))
        c_gamma = _bs_gamma(spot, K, T, c_iv)
        p_gamma = _bs_gamma(spot, K, T, p_iv)
        # GEX in dollars; contract multiplier = 100
        gex_by_strike[K] = (c_gamma * c_oi - p_gamma * p_oi) * 100.0 * spot * spot

    return gex_by_strike, total_call_oi, total_put_oi


def _compute_max_pain(calls, puts) -> Optional[float]:
    """Strike that minimises total in-the-money dollar loss across all options."""
    all_strikes = sorted(set(calls["strike"].tolist()) | set(puts["strike"].tolist()))
    if not all_strikes:
        return None

    c_map = dict(zip(calls["strike"], calls["openInterest"].fillna(0)))
    p_map = dict(zip(puts["strike"],  puts["openInterest"].fillna(0)))

    min_pain  = float("inf")
    pain_strike = all_strikes[len(all_strikes) // 2]

    for test in all_strikes:
        pain = sum(max(0.0, test - K) * c_map.get(K, 0) for K in all_strikes)
        pain += sum(max(0.0, K - test) * p_map.get(K, 0) for K in all_strikes)
        if pain < min_pain:
            min_pain    = pain
            pain_strike = test

    return float(pain_strike)


def _compute_expected_move(spot: float, calls, puts) -> float:
    """
    OI-weighted expected move as a percentage of spot (market-implied ±1σ range).

    Weights each strike's straddle mid-price by its total open interest so that
    strikes where the market is most active contribute proportionally.  This is
    more accurate than the plain ATM straddle when OI is spread across many strikes
    (e.g. after a large options roll or ahead of earnings).
    """
    if spot <= 0:
        return 0.0

    has_bid_ask = "bid" in calls.columns and "ask" in calls.columns

    def _mid(df, K):
        row = df[df["strike"] == K]
        if row.empty:
            return 0.0
        if has_bid_ask:
            bid = float(row["bid"].values[0] or 0)
            ask = float(row["ask"].values[0] or 0)
            if ask > 0:
                return (bid + ask) / 2.0
        if "lastPrice" in row.columns:
            return float(row["lastPrice"].values[0] or 0)
        return 0.0

    c_oi = dict(zip(calls["strike"], calls["openInterest"].fillna(0)))
    p_oi = dict(zip(puts["strike"],  puts["openInterest"].fillna(0)))
    all_strikes = sorted(set(c_oi) | set(p_oi))

    weighted_straddle = 0.0
    total_oi = 0.0
    for K in all_strikes:
        oi = float(c_oi.get(K, 0)) + float(p_oi.get(K, 0))
        if oi <= 0:
            continue
        straddle = _mid(calls, K) + _mid(puts, K)
        if straddle <= 0:
            continue
        weighted_straddle += straddle * oi
        total_oi += oi

    if total_oi == 0:
        # Fall back to plain ATM straddle if OI data is missing
        all_strikes_list = sorted(all_strikes)
        if not all_strikes_list:
            return 0.0
        atm = min(all_strikes_list, key=lambda k: abs(k - spot))
        straddle = _mid(calls, atm) + _mid(puts, atm)
        return round(straddle / spot * 100.0, 2)

    return round(weighted_straddle / total_oi / spot * 100.0, 2)


def _compute_oi_skew(spot: float, calls, puts) -> float:
    """
    OI-weighted directional skew ∈ [-1, +1].

    Measures whether the market is positioning for upside or downside by comparing
    the dollar-distance-weighted open interest on calls (above spot) vs puts (below spot).

    Formula
    ───────
      bullish = Σ  call_OI[K] × (K - spot)/spot   for K > spot
      bearish = Σ  put_OI[K]  × (spot - K)/spot   for K < spot
      skew    = (bullish − bearish) / (bullish + bearish)

    Interpretation
    ──────────────
      +1  all call OI concentrated far above spot → strong bullish positioning
      −1  all put OI concentrated far below spot  → strong bearish / hedging
       0  balanced or no OI

    Unlike the put/call *ratio* (a contrarian signal), this is a *directional*
    signal: heavy call OI above spot means large players are positioned for upside.
    """
    if spot <= 0:
        return 0.0

    c_oi = dict(zip(calls["strike"], calls["openInterest"].fillna(0)))
    p_oi = dict(zip(puts["strike"],  puts["openInterest"].fillna(0)))

    bullish = sum(float(oi) * (K - spot) / spot
                  for K, oi in c_oi.items() if K > spot and float(oi) > 0)
    bearish = sum(float(oi) * (spot - K) / spot
                  for K, oi in p_oi.items() if K < spot and float(oi) > 0)

    total = bullish + bearish
    if total < 1e-10:
        return 0.0
    return round((bullish - bearish) / total, 3)


def _find_gamma_flip(gex_by_strike: Dict[float, float], spot: float) -> Optional[float]:
    """
    Scan strikes from highest to lowest, accumulating GEX.
    The gamma flip is the strike where the running total first crosses from positive
    to negative — below this level dealers are net short gamma and moves accelerate.
    Returns None if no flip exists below current price.
    """
    strikes = sorted(gex_by_strike.keys(), reverse=True)  # high → low
    if not strikes:
        return None

    # Start from total GEX and remove each strike's contribution as we scan down
    cumulative = sum(gex_by_strike.values())
    prev = cumulative

    for strike in strikes:
        prev = cumulative
        cumulative -= gex_by_strike[strike]
        # Flip: previous positive, now negative (or vice versa)
        if prev > 0 >= cumulative or prev < 0 <= cumulative:
            return round(strike, 2)

    return None


# ── Per-ticker GEX signal ─────────────────────────────────────────────────────

def _fetch_gex_signal(ticker: str) -> Optional[GEXSignal]:
    """Compute GEX, max pain, and expected move for one ticker. Returns None on failure."""
    for attempt, backoff in enumerate([0] + _RATE_LIMIT_BACKOFF):
        if backoff:
            logger.info(f"[gex] {ticker}: rate-limited, retrying in {backoff}s (attempt {attempt+1})")
            time.sleep(backoff)
        try:
            return _fetch_gex_signal_once(ticker)
        except Exception as exc:
            if "Too Many Requests" in str(exc) or "Rate limit" in str(exc) or "429" in str(exc):
                if attempt < len(_RATE_LIMIT_BACKOFF):
                    continue   # will retry with backoff
            logger.warning(f"[gex] {ticker} failed: {exc}")
            return None
    logger.warning(f"[gex] {ticker}: gave up after {len(_RATE_LIMIT_BACKOFF)+1} attempts")
    return None


def _fetch_gex_signal_once(ticker: str) -> Optional[GEXSignal]:
    """Single attempt — raises on rate-limit so the caller can back off and retry."""
    try:
        tk = yf.Ticker(ticker)
        spot = tk.fast_info.last_price
        if not spot or spot <= 0:
            return None
        spot = float(spot)

        expiries = tk.options
        if not expiries:
            logger.debug(f"[gex] {ticker}: no options available")
            return None

        today = date.today()
        near_expiries = [
            exp for exp in expiries
            if 0 < (date.fromisoformat(exp) - today).days <= _MAX_EXPIRY_DAYS
        ][:_MAX_EXPIRIES]

        if not near_expiries:
            logger.debug(f"[gex] {ticker}: no expiries within {_MAX_EXPIRY_DAYS} days")
            return None

        combined_gex: Dict[float, float] = {}
        total_call_oi = 0
        total_put_oi  = 0
        max_pain_val: Optional[float] = None
        expected_move_pct = 0.0
        oi_skew_val = 0.0
        dominant_expiry = near_expiries[0]
        has_bid_ask = False

        for i, expiry in enumerate(near_expiries):
            chain = tk.option_chain(expiry)

            gex_map, c_oi, p_oi = _gex_for_expiry(spot, expiry, chain)
            total_call_oi += c_oi
            total_put_oi  += p_oi

            for k, v in gex_map.items():
                combined_gex[k] = combined_gex.get(k, 0.0) + v

            # Only use the nearest expiry for max pain + expected move + skew
            if i == 0:
                max_pain_val = _compute_max_pain(chain.calls, chain.puts)
                expected_move_pct = _compute_expected_move(spot, chain.calls, chain.puts)
                if expected_move_pct > 0:
                    has_bid_ask = True
                oi_skew_val = _compute_oi_skew(spot, chain.calls, chain.puts)

            time.sleep(_REQUEST_DELAY)

        total_oi = total_call_oi + total_put_oi
        if total_oi < _MIN_OI:
            logger.debug(f"[gex] {ticker}: OI={total_oi} below threshold — skipping")
            return None

        # Net GEX summary metrics
        total_net_gex = sum(combined_gex.values())
        total_abs_gex = sum(abs(v) for v in combined_gex.values())
        net_gex_bn    = round(total_net_gex / 1e9, 3)
        gex_norm      = round(total_net_gex / max(total_abs_gex, 1e-10), 3)

        # Classify
        if gex_norm > _PIN_THRESHOLD:
            gex_signal = "PINNED"
        elif gex_norm < _AMP_THRESHOLD:
            gex_signal = "AMPLIFIED"
        else:
            gex_signal = "NEUTRAL"

        # Gamma flip
        gamma_flip = _find_gamma_flip(combined_gex, spot)

        # Max pain directional bias
        max_pain_bias = "NEUTRAL"
        if max_pain_val is not None:
            pct_diff = (spot - max_pain_val) / spot
            if pct_diff > 0.01:
                max_pain_bias = "BEARISH"   # spot > max pain → gravity pulls price down
            elif pct_diff < -0.01:
                max_pain_bias = "BULLISH"   # spot < max pain → gravity pulls price up

        # Build summary
        flip_str = f" Gamma flip ${gamma_flip:.2f}." if gamma_flip else ""
        pain_str = (
            f" Max pain ${max_pain_val:.2f} ({max_pain_bias} pull)."
            if max_pain_val else ""
        )
        em_str   = f" Expected move ±{expected_move_pct:.1f}% to {near_expiries[0]}." if has_bid_ask else ""
        skew_str = f" OI skew={oi_skew_val:+.2f}." if abs(oi_skew_val) >= 0.05 else ""
        summary = (
            f"{ticker}: {gex_signal} (norm={gex_norm:+.2f}, ${net_gex_bn:+.2f}B)."
            f"{flip_str}{pain_str}{em_str}{skew_str}"
        )

        logger.info(
            f"[gex] {ticker}: {gex_signal} | norm={gex_norm:+.2f} | "
            f"flip={gamma_flip} | pain={max_pain_val} | "
            f"em=±{expected_move_pct:.1f}% | skew={oi_skew_val:+.2f}"
        )
        return GEXSignal(
            ticker=ticker,
            spot_price=round(spot, 2),
            net_gex_bn=net_gex_bn,
            gex_normalized=gex_norm,
            gex_signal=gex_signal,
            gamma_flip=round(gamma_flip, 2) if gamma_flip else None,
            max_pain=round(max_pain_val, 2) if max_pain_val else None,
            expected_move_pct=expected_move_pct,
            max_pain_bias=max_pain_bias,
            oi_skew=round(oi_skew_val, 3),
            dominant_expiry=dominant_expiry,
            report_date=date.today(),
            summary=summary,
        )

    except Exception as exc:
        # Re-raise rate-limit errors so the outer retry loop can back off
        if "Too Many Requests" in str(exc) or "Rate limit" in str(exc) or "429" in str(exc):
            raise
        logger.warning(f"[gex] {ticker} failed: {exc}")
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_gex_context(tickers: List[str]) -> Optional[GEXContext]:
    """
    Compute GEX for the major index ETFs plus liquid individual tickers.
    Cached daily — yfinance options chains are expensive to fetch.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[gex] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    # Always cover the major index ETFs; add watchlist tickers that have liquid options.
    # Skip futures/crypto tickers (contain "=") and CBOE indices (start with "^").
    index_set  = set(_INDEX_TICKERS)
    extra      = [t for t in tickers if t not in index_set and "=" not in t and t[0] != "^"]
    to_process = list(index_set) + extra

    signals: List[GEXSignal] = []
    for ticker in to_process:
        sig = _fetch_gex_signal(ticker)
        if sig is not None:
            signals.append(sig)

    if not signals:
        logger.warning("[gex] No GEX signals computed")
        return None

    # Build an overall summary from index ETF signals
    idx_map = {s.ticker: s for s in signals if s.ticker in index_set}
    summary_parts = []
    for idx in ["SPY", "QQQ", "IWM"]:
        if idx in idx_map:
            s = idx_map[idx]
            flip = f"flip=${s.gamma_flip:.2f}" if s.gamma_flip else "no flip"
            summary_parts.append(f"{idx}: {s.gex_signal} ({flip})")

    ctx = GEXContext(
        signals=signals,
        report_date=date.today(),
        summary="; ".join(summary_parts) if summary_parts else "GEX computed.",
    )
    _save_cache(ctx)
    logger.info(f"[gex] Context built: {len(signals)} signals — {ctx.summary}")
    return ctx
