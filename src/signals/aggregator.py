"""
Build a TickerSignal for each ticker by combining all enabled analysis methods.

Score sign convention (THE invariant the combine, direction, IC + inversion rely on)
────────────────────────────────────────────────────────────────────────────────────
EVERY method returns a score in [-1, +1] whose SIGN is the predicted direction of
the STOCK's forward return — never the raw indicator state, and never relative to a
position you happen to hold:

    score > 0  →  the stock is more likely to go UP    (bullish)
    score < 0  →  the stock is more likely to go DOWN  (bearish)
    score ≈ 0  →  no view

Mean-reversion / contrarian methods bake the reversal INTO the sign; they do NOT
report state. So "stretched / extreme" never decides the sign — the EXPECTED NEXT
MOVE does:
    • vwap     → + when price is BELOW its VWAP (expects a bounce UP), − when above.
    • put_call → + on EXTREME_PUTS (fear ⇒ contrarian bullish), − on EXTREME_CALLS.
    • iv_rank  → + on a high-vol capitulation selloff (CAPITULATION_BUY), − on a
                 high-vol euphoric rally (FADE_EXTREME).
    • coint    → + when the ticker is the cheap/long leg of a stretched pair.

Because the sign is uniform, everything downstream is unambiguous and position-
independent:
    combined_score = Σ wᵢ·scoreᵢ ;  ≥ +0.15 → BULLISH → BUY/long,  ≤ −0.15 → BEARISH
    → SELL/short. The score describes the STOCK, so a BEARISH (negative) read is ONE
    action set: it OPENS a short when flat AND CLOSES a long when held (a positive
    score is NOT "good for a short" — it is "the stock should rise").

Inversion is therefore just "use −score": a method whose score is anti-correlated
with forward returns (negative IC / ICIR) is flipped. It is applied PER HORIZON
(``edge_curve``), which is exactly what keeps it correct for mean-reversion methods —
their sign can be momentum-like at very short horizons and reversion-like at longer
ones, so a method may legitimately flip at some horizons and not others. Only a sign
that is wrong across ALL horizons is a true sign bug worth a permanent inversion (or a
scorer fix). See ``simulated_trades --directional`` for the per-side ICIR readout.

Methods
───────
  1  News sentiment        (enable_news_sentiment)
  2  Technical analysis    (enable_technical_analysis)
  3  Smart money / insider (enable_insider_trades / options_flow / sec_filings)
  4  Put/call ratio        (enable_put_call, per-ticker contrarian)

Confidence formula
──────────────────
  confidence = raw_conf × coherence_factor × movement_factor × volume_factor

  raw_conf         — abs(combined_score) normalised to [0, 1].
  coherence_factor — continuous [0.45, 1.35]: how strongly do all methods agree?
                     Replaces the old binary 1.25×/0.60× toggle. Magnitude-weighted,
                     so a weak outlier costs less than a strong one pointing opposite.
  movement_factor  — [0.80, 1.20]: is the stock actually likely to move?
                     Derived from ATR% + BB-width%. Low-volatility stocks are
                     unlikely to deliver on a signal; high-volatility ones are "in play".
  volume_factor    — [0.90, 1.15]: does recent volume confirm the aggregate direction?
                     Separate from the vol multiplier already baked into technical_score;
                     this operates at the cross-method level.

Interaction adjustments (additive, applied before confidence, capped ±0.15)
────────────────────────────────────────────────────────────────────────────
  • Insider accumulation at technical support — insiders buying while price is
    technically weak signals contrarian value accumulation.
  • Options extreme + technical aligned — extreme put/call skew at a price
    inflection is a classic squeeze/unwind setup.
  • News catalyst confirmed by volume — a strong sentiment signal backed by
    above-average volume is more likely to be a real market-moving event.

Sector alignment (second pass, unchanged from before)
──────────────────────────────────────────────────────
  Individual stocks are cross-referenced against their sector ETF direction.
  Alignment → mild boost (1.10×). Contradiction → meaningful penalty (0.75×).
"""

from datetime import date
from loguru import logger
from typing import List, Optional

from config import settings
from src.models import NewsArticle, Direction, TickerSignal, InsiderTrade
from src.analysis.sentiment import analyse_sentiment, filter_relevant_articles
from src.signals.sentiment_velocity import compute_sentiment_velocity
from src.data.insider_trades import build_insider_summary
from src.analysis.technical import TechnicalResult, EMPTY_RESULT, compute_technical_score
from src.signals.vwap import compute_vwap_score
from src.signals.pattern_recognition import compute_pattern_score
from src.signals.price_momentum import compute_price_momentum_score
from src.signals.sector_relative_momentum import (
    compute_sector_relative_momentum_score,
    compute_market_relative_momentum_score,
)
from src.signals.money_flow import compute_money_flow_score
from src.signals.trend_strength import compute_trend_strength_score
from src.signals.trend_predictability import (calibrate_trend_orientation,
                                              compute_trend_predictability_scores)
from src.signals.iv_rank import compute_iv_rank_score
from src.signals.iv_expr import compute_iv_expr_score
from src.signals.extended_session import compute_extended_gap_score
from src.signals.broker_advisor import compute_broker_advisor_score


# ── Smart-money signal direction + strength ──────────────────────────────────

_TX_SCORE: dict = {
    "unusual_call":       (+1.0, 1.5),
    "13d_activist_stake": (+1.0, 2.0),
    "13g_passive_stake":  (+1.0, 0.8),
    "13f_new_position":   (+1.0, 1.0),
    "13f_increase":       (+1.0, 0.8),
    "unusual_put":        (-1.0, 1.5),
    "planned_sale_144":   (-1.0, 0.9),
    "13f_exit":           (-1.0, 1.2),
    "13f_decrease":       (-1.0, 0.7),
}

# Base weights — normalised across active methods at runtime
_BASE_WEIGHTS = {
    "news":      0.40,
    "sent_velocity": 0.12,  # Δsentiment (rate of change of news tone) — short-horizon timing overlay
    "tech":      0.30,
    "massive":   0.15,   # Massive/Polygon server-side RSI+MACD composite — overlaps `tech`, so kept modest
    "insider":   0.30,
    "put_call":  0.15,
    "max_pain":  0.12,   # options-expiry gravity; weight fades automatically via expiry_factor
    "oi_skew":   0.15,   # OI-weighted call/put directional positioning
    "vwap":      0.12,   # mean-reversion: price vs rolling 20-day VWAP
    "pattern":    0.18,   # chart pattern recognition: historical win-rate based score
    "momentum":   0.18,   # perceived value: normalised 1m/3m price trend vs own history
    "sector_momentum": 0.20,  # sector-relative momentum: ticker − sector ETF (cleaner alpha factor)
    "market_momentum": 0.08,  # market-relative (ticker − SPY) — LIGHT: overlaps sector_momentum (shared idiosyncratic move), so kept small to bound the beta double-count
    "money_flow": 0.15,   # accumulation/distribution: MFI + CMF + OBV slope composite
    "trend_strength": 0.15,  # ADX/DMI trend quality + Donchian breakout (trend-following)
    "pead":       0.15,   # Post-Earnings Announcement Drift: SUE × time-decay
    "iv_rank":    0.13,   # IV Rank + directional: regime-aware contrarian / trend bias from RV percentile
    "iv_expr":    0.12,   # IV Expression: stock-vs-options bias from real chain IV + oi_skew
    "coint":      0.12,   # Cointegration pairs: per-ticker lean from stat-arb spread z-scores
    "ext_gap":    0.15,   # Extended-session gap momentum: live off-hours print vs last close / ATR
    "broker_advisor": 0.10,  # IBKR short-borrow squeeze tilt (hard/expensive to short → bullish, fades a SELL)
}

# Extended-session weight overlay (runs outside RTH). Options chains close at
# the end of the regular session, so every options-derived method reads
# YESTERDAY'S positioning during a pre/after-market tick — informative context
# but not live confirmation → scaled down. News flow (and its velocity) plus
# the extended gap are the genuinely live information off-hours → scaled up.
_EXTENDED_STALE_METHODS = ("put_call", "max_pain", "oi_skew", "iv_expr")
_EXTENDED_FRESH_METHODS = ("news", "sent_velocity", "ext_gap")


def _extended_session_weight_overlay(profile: dict) -> dict:
    """Scale a weight profile for an extended/overnight run (see above)."""
    stale_mult = float(settings.extended_stale_options_weight_mult)
    fresh_mult = float(settings.extended_news_weight_mult)
    out = dict(profile)
    for m in _EXTENDED_STALE_METHODS:
        if m in out:
            out[m] = out[m] * stale_mult
    for m in _EXTENDED_FRESH_METHODS:
        if m in out:
            out[m] = out[m] * fresh_mult
    return out

# Put/call contrarian score mapping
_PC_SIGNAL_SCORE = {
    "EXTREME_PUTS":  +0.70,
    "PUTS_HEAVY":    +0.35,
    "BALANCED":       0.00,
    "CALLS_HEAVY":   -0.35,
    "EXTREME_CALLS": -0.70,
}

# Sector → ETF mapping for sector-alignment second pass
_SECTOR_MAP: dict = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "INTC": "XLK", "ORCL": "XLK", "ADBE": "XLK", "CRM": "XLK",
    "AMZN": "XLY", "TSLA": "XLY", "UBER": "XLY", "LYFT": "XLY",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC",
    "JPM":  "XLF", "BAC":  "XLF", "GS":   "XLF", "MS":   "XLF",
    "PYPL": "XLF",
}
_SECTOR_ALIGN_BOOST   = 1.10
_SECTOR_ALIGN_PENALTY = 0.75

_AGREE_THRESHOLD = 0.0    # any non-zero method score counts as a view (was 0.10)


# ── Adaptive weight multipliers (data-informed reweighting) ──────────────────

def _adaptive_weight_multipliers() -> dict:
    """Return ``{method: multiplier}`` derived from each method's solo win rate.

    Pulls per-method standalone performance from
    ``tracker.compute_solo_method_performance`` (which is "what direction
    would this method alone have signalled? was that direction right?") and
    maps each win rate to a multiplier centred at 1.0× for a 50% (coin-flip)
    win rate.

    Bayesian shrinkage with ``adaptive_weight_prior_n`` virtual trials at 50%
    keeps a method that's only fired a few times from getting an outsized
    boost or cut — a 3-for-3 method gets a much smaller boost than a 30-for-
    30 one, since the second one is much harder to do by chance.

      shrunk_wr  = (wins + 0.5 × prior_n) / (n + prior_n)
      raw_mult   = shrunk_wr / 0.5            → 1.0 at 50% WR
      multiplier = clip(raw_mult, min_mult, max_mult)

    Returns ``{method: 1.0}`` for every method when:
      * the feature flag is off,
      * the tracker has no attribution data yet, or
      * the perf lookup raises (unavailable cache, import failure, …).

    The result is meant to be multiplied into the active weight profile
    *before* normalisation, so it composes with regime overrides and the
    OpEx amplifier instead of overriding them.
    """
    default = {m: 1.0 for m in _BASE_WEIGHTS}
    if not settings.enable_adaptive_weights:
        return default

    try:
        from src.performance.tracker import compute_solo_method_performance
        # Train on the OOS train slice only so the holdout slice remains an
        # honest evaluation set. When OOS validation is disabled (or the
        # holdout pct is 0) the helper treats every trade as "train" so the
        # call still returns the full sample — no behaviour change.
        split = "train" if settings.enable_oos_validation else None
        perf = compute_solo_method_performance(split=split)
    except Exception as e:
        logger.debug(f"[aggregator] adaptive weights — solo perf unavailable: {e}")
        return default
    if not perf:
        return default

    prior_n = max(0, settings.adaptive_weight_prior_n)
    min_m = settings.adaptive_weight_min_multiplier
    max_m = settings.adaptive_weight_max_multiplier

    multipliers = {}
    for method in _BASE_WEIGHTS:
        overall = perf.get(method, {}).get("overall", {})
        n = int(overall.get("trades", 0) or 0)
        wr_pct = float(overall.get("win_rate", 50.0) or 50.0)
        wins = n * (wr_pct / 100.0)
        # Bayesian shrinkage toward the 50% prior
        shrunk_wr = (wins + 0.5 * prior_n) / (n + prior_n) if (n + prior_n) > 0 else 0.5
        raw_mult = shrunk_wr / 0.5  # 1.0 at 50% WR; 2.0 at 100% WR; 0.0 at 0% WR
        multipliers[method] = round(max(min_m, min(max_m, raw_mult)), 3)

    return multipliers


def _apply_adaptive_multipliers(weight_profile: dict) -> tuple[dict, dict]:
    """Multiply every weight in ``weight_profile`` by its adaptive multiplier.

    Returns ``(new_profile, multipliers)`` so the caller can log which
    methods got boosted or cut. When adaptivity is off or yields all-1.0
    multipliers, the profile is returned unchanged.
    """
    mults = _adaptive_weight_multipliers()
    if all(abs(m - 1.0) < 1e-6 for m in mults.values()):
        return dict(weight_profile), mults
    new_profile = {
        m: weight_profile.get(m, 0.0) * mults.get(m, 1.0)
        for m in weight_profile
    }
    return new_profile, mults


# ── IC-informed adaptive weights (panel-driven; default off) ─────────────────

_IC_WEIGHT_CACHE: dict = {}


def _inverted_methods() -> frozenset:
    """Methods (lower-cased) flagged for sign INVERSION in the combine — their raw
    score is treated as reliably-backwards and contributes with a flipped sign
    (config ``inverted_methods``, comma-separated; empty = none). The panel keeps the
    RAW score, so the inversion stays re-validatable."""
    raw = str(getattr(settings, "inverted_methods", "") or "")
    return frozenset(m.strip().lower() for m in raw.split(",") if m.strip())


def _ic_mults_from_ic_table(ic, key: str) -> dict:
    """Map an IC table (``icir_<key>`` + ``icdays_<key>`` columns) → ``{method:
    multiplier}``, reweighting ONLY methods whose IC is statistically confident.

    ``key`` selects the horizon column: ``"5d"`` for the edge basis
    (``signal_panel.compute_ic``), ``"1w"`` for the shadow basis (the ``both``-side
    rows of ``simulated_trades.compute_directional_perf``).

    A method is reweighted only when its mean-daily-IC t-statistic clears the bar::

        t = |ICIR| · sqrt(n_days)  ≥  ic_weight_min_t        (t ≥ 2 ≈ 95% significant)

    Methods below the bar are OMITTED → they keep their base weight (we are not yet
    confident the IC is real, so we do not touch it). Among the CONFIDENT methods::

        ICIR > 0 → boost: mult = clip(ICIR / median(confident positive ICIR), min, max)
                   (typical confident method → 1.0×, better → up, weaker → down)
        ICIR ≤ 0 → floor: mult = ic_weight_min_multiplier
                   (confidently anti-predictive — minimise until inversion flips it)

    Returns ``{}`` when no method clears the bar (a thin panel ⇒ a pure no-op)."""
    import pandas as pd
    from math import sqrt
    if ic is None or getattr(ic, "empty", True):
        return {}
    icir_col, days_col = f"icir_{key}", f"icdays_{key}"
    if icir_col not in ic.columns or days_col not in ic.columns:
        return {}
    lo = float(settings.ic_weight_min_multiplier)
    hi = float(settings.ic_weight_max_multiplier)
    min_t = max(0.0, float(settings.ic_weight_min_t))

    by_method = {str(r["method"]): r for _, r in ic.iterrows()}

    # Pass 1 — keep only pool methods whose IC clears the confidence (t-stat) gate.
    confident: dict = {}    # method -> ICIR
    for m in _BASE_WEIGHTS:
        r = by_method.get(m)
        if r is None:
            continue
        icir, n_days = r.get(icir_col), r.get(days_col)
        if (icir is None or pd.isna(icir) or n_days is None or pd.isna(n_days)
                or int(n_days) <= 0):
            continue                  # no usable ICIR yet → keep base weight
        if abs(float(icir)) * sqrt(int(n_days)) >= min_t:
            confident[m] = float(icir)
    if not confident:
        return {}                     # nothing is confident yet → no-op (base weights)

    pos = [v for v in confident.values() if v > 0]
    if not pos:                       # every confident method is anti-predictive → floor all
        return {m: lo for m in confident}
    from statistics import median
    ref = median(pos)                 # typical confident positive ICIR = the 1.0× anchor
    if ref <= 1e-9:
        return {m: lo for m in confident}

    mults: dict = {}
    for m, icir in confident.items():
        mults[m] = lo if icir <= 0 else round(max(lo, min(hi, icir / ref)), 3)
    return mults


def _ic_weight_multipliers() -> dict:
    """``{method: multiplier}`` from each method's confidence-gated IC over the
    unbiased signals panel (see ``_ic_mults_from_ic_table``).

    ``ic_weight_basis`` selects the IC:
      * ``"shadow"`` (default) — MARKET-NEUTRAL IC (ticker − SPY, pooled both
        directions): the ``both``-side rows of ``compute_directional_perf`` at the
        ``ic_weight_shadow_horizon`` label. Weights by regime-robust alpha, not beta.
      * ``"edge"`` — absolute-return IC: ``compute_ic`` at ``ic_weight_horizon_days``.

    Cached for ``ic_weight_cache_seconds`` because ``build_signals`` runs many times
    per tick (the hold-review pool). Returns ``{}`` (→ no tilt) when disabled or when
    no method clears the confidence gate yet."""
    if not settings.enable_ic_weights:
        return {}
    import time
    now = time.time()
    hit = _IC_WEIGHT_CACHE.get("v")
    if hit and (now - hit["ts"]) < settings.ic_weight_cache_seconds:
        return hit["mults"]
    mults: dict = {}
    try:
        if str(settings.ic_weight_basis).lower() == "edge":
            from src.analysis.signal_panel import build_panel, compute_ic
            h = max(1, int(settings.ic_weight_horizon_days))
            panel = build_panel(horizons=(h,), days=settings.horizon_ic_days)
            if panel is not None and not panel.empty:
                ic = compute_ic(panel, horizons=(h,),
                                min_n=settings.horizon_ic_min_n,
                                min_per_day=settings.ic_weight_min_per_day,
                                min_days=settings.ic_weight_min_days)
                mults = _ic_mults_from_ic_table(ic, f"{h}d")
        else:  # shadow (default) — market-neutral, pooled-both-directions IC
            from src.analysis.simulated_trades import compute_directional_perf
            lbl = str(settings.ic_weight_shadow_horizon)
            df = compute_directional_perf(days=settings.horizon_ic_days,
                                          benchmark=settings.horizon_market_benchmark,
                                          min_n=settings.horizon_ic_min_n,
                                          min_per_day=settings.ic_weight_min_per_day,
                                          min_days=settings.ic_weight_min_days)
            if df is not None and not df.empty and "side" in df.columns:
                both = df[df["side"] == "both"]
                if not both.empty:
                    mults = _ic_mults_from_ic_table(both, lbl)
    except Exception as e:  # pragma: no cover - defensive (DB / cache hiccup)
        logger.debug(f"[aggregator] IC weights unavailable ({e}) — base weights used")
        mults = {}
    _IC_WEIGHT_CACHE["v"] = {"ts": now, "mults": mults}
    return mults


def reset_ic_weight_cache() -> None:
    """Drop the cached IC-weight multipliers (tests / forced refresh)."""
    _IC_WEIGHT_CACHE.clear()


# ── Per-method score helpers ──────────────────────────────────────────────────

def _detect_insider_cluster(ticker: str, trades: List[InsiderTrade]) -> tuple[bool, int]:
    """
    Detect if ≥3 DIFFERENT insiders bought the same ticker within any 5-day window.
    Only counts corporate_insider and politician purchases (not options flow or institutional).
    Returns (cluster_detected, max_distinct_insiders_in_any_5d_window).
    """
    from datetime import timedelta

    buys = [
        t for t in trades
        if t.ticker.upper() == ticker.upper()
        and "purchase" in t.transaction_type
        and t.trader_type in ("corporate_insider", "politician")
    ]

    if len(buys) < 3:
        return False, len({t.trader_name for t in buys})

    buys.sort(key=lambda t: t.transaction_date)

    max_cluster = 0
    for anchor in buys:
        window_end = anchor.transaction_date + timedelta(days=5)
        names_in_window = {
            t.trader_name
            for t in buys
            if anchor.transaction_date <= t.transaction_date <= window_end
        }
        if len(names_in_window) > max_cluster:
            max_cluster = len(names_in_window)

    return max_cluster >= 3, max_cluster


def _detect_insider_persistence(ticker: str, trades: List[InsiderTrade]) -> tuple[bool, int, str]:
    """
    Detect whether the SAME insider bought ``ticker`` on multiple DISTINCT days
    within the lookback window — repeated accumulation by one name signals far
    stronger conviction than a one-off purchase.

    Distinct from ``_detect_insider_cluster`` (which measures *breadth* — many
    DIFFERENT insiders buying at once). This measures *depth* — one insider's
    sustained, repeated buying over time, escalating personal capital at risk.

    Only counts corporate_insider and politician purchases (not options flow,
    13F institutional, or bare Form 4 filings whose direction is unknown — same
    filter as the cluster detector, for consistency).

    Returns ``(persistence_detected, max_distinct_buy_days, top_buyer_name)``.
    """
    from collections import defaultdict

    min_buys = max(2, settings.insider_persistence_min_buys)

    buys = [
        t for t in trades
        if t.ticker.upper() == ticker.upper()
        and "purchase" in t.transaction_type
        and t.trader_type in ("corporate_insider", "politician")
    ]
    if len(buys) < min_buys:
        return False, 0, ""

    # Count DISTINCT transaction dates per insider (case-insensitive name key)
    # so a single multi-line filing on one day counts once, not as repetition.
    dates_by_name: dict = defaultdict(set)
    display_name: dict = {}
    for t in buys:
        key = t.trader_name.strip().lower()
        if not key:
            continue
        dates_by_name[key].add(t.transaction_date)
        display_name[key] = t.trader_name.strip()

    if not dates_by_name:
        return False, 0, ""

    top_key      = max(dates_by_name, key=lambda k: len(dates_by_name[k]))
    max_distinct = len(dates_by_name[top_key])
    if max_distinct < min_buys:
        return False, max_distinct, ""

    return True, max_distinct, display_name.get(top_key, "")


def _persistence_factor(count: int) -> float:
    """Map distinct repeat-buy days to a score amplifier.

    2 buys → 1.25×, 3 → 1.50×, 4+ → 1.75× (capped at the same ceiling as the
    cluster amplifier so a single conviction tell can't dominate the score).
    """
    return min(1.75, 1.0 + 0.25 * max(0, count - 1))


def _insider_score(
    ticker: str, trades: List[InsiderTrade]
) -> tuple[float, bool, int, bool, int, str]:
    """Derive a [-1, +1] score from all smart money signals for a ticker.

    Returns ``(score, cluster_detected, cluster_size,
               persistence_detected, persistence_count, persistence_buyer)``.

    Two independent conviction amplifiers can stack on a positive base score:
      • Cluster (breadth) — ≥3 DIFFERENT insiders buying within 5 days → 1.75×.
      • Persistence (depth) — the SAME insider buying on multiple separate days
        → up to 1.75× (scales with repeat count).
    Both are applied only when the base score is bullish (insider buying is far
    more informative than insider selling), and the result is clamped to +1.0.
    """
    relevant = [t for t in trades if t.ticker.upper() == ticker.upper()]
    if not relevant:
        return 0.0, False, 0, False, 0, ""
    from src.data.insider_trades import _amount_weight
    total = 0.0
    for t in relevant:
        w  = _amount_weight(t.amount_range)
        tx = t.transaction_type
        if tx in _TX_SCORE:
            direction, multiplier = _TX_SCORE[tx]
            total += direction * w * multiplier
        elif "purchase" in tx:
            total += w
        elif "sale" in tx:
            total -= w
    base_score = round(max(-1.0, min(1.0, total / 3.0)), 3)

    # Cluster amplifier — 3+ different insiders buying within 5 days
    cluster_detected, cluster_size = _detect_insider_cluster(ticker, trades)
    if cluster_detected and base_score > 0:
        amplified = round(min(1.0, base_score * 1.75), 3)
        logger.debug(
            f"[cluster] {ticker}: {cluster_size} insiders within 5d — "
            f"score {base_score:+.3f} → {amplified:+.3f} (1.75×)"
        )
        base_score = amplified

    # Persistence amplifier — the SAME insider buying repeatedly across distinct days
    persistence_detected, persistence_count, persistence_buyer = False, 0, ""
    if settings.enable_insider_persistence:
        persistence_detected, persistence_count, persistence_buyer = (
            _detect_insider_persistence(ticker, trades)
        )
        if persistence_detected and base_score > 0:
            pf = _persistence_factor(persistence_count)
            amplified = round(min(1.0, base_score * pf), 3)
            logger.debug(
                f"[persistence] {ticker}: {persistence_buyer} bought {persistence_count}× "
                f"on separate days — score {base_score:+.3f} → {amplified:+.3f} ({pf:.2f}×)"
            )
            base_score = amplified

    return (base_score, cluster_detected, cluster_size,
            persistence_detected, persistence_count, persistence_buyer)


def _put_call_score_for(ticker: str, put_call_context) -> float:
    if put_call_context is None:
        return 0.0
    for sig in put_call_context.ticker_signals:
        if sig.ticker.upper() == ticker.upper():
            return _PC_SIGNAL_SCORE.get(sig.signal, 0.0)
    return 0.0


def _normalised_weights(active_flags: dict, weight_profile: Optional[dict] = None) -> dict:
    profile = weight_profile or _BASE_WEIGHTS
    raw   = {m: (profile.get(m, _BASE_WEIGHTS.get(m, 0.0)) if on else 0.0) for m, on in active_flags.items()}
    total = sum(raw.values())
    if total == 0:
        return {m: 0.0 for m in active_flags}
    return {m: w / total for m, w in raw.items()}


# ── Coherence ─────────────────────────────────────────────────────────────────

def _coherence_factor(combined: float, method_scores: list) -> tuple[float, float]:
    """
    Return (coherence_ratio, coherence_factor).

    coherence_ratio ∈ [0, 1]:
        Magnitude-weighted fraction of active methods that agree with the
        combined direction. A weak outlier pointing opposite hurts less than
        a strong one; methods with |score| < AGREE_THRESHOLD are ignored.

    coherence_factor ∈ [0.45, 1.35]:
        Continuous multiplier — replaces the old binary 1.25×/0.60× toggle.
        Maps coherence_ratio linearly: 0 → 0.45, 1 → 1.35.
    """
    agree_w = 0.0
    total_w = 0.0
    for enabled, score in method_scores:
        if not enabled or abs(score) < _AGREE_THRESHOLD:
            continue
        total_w += abs(score)
        if (combined > 0 and score > 0) or (combined < 0 and score < 0):
            agree_w += abs(score)

    if total_w == 0:
        return 0.0, 0.60   # no method has a view — moderate penalty

    ratio  = agree_w / total_w                  # ∈ [0, 1]
    factor = round(0.45 + ratio * 0.90, 3)      # ∈ [0.45, 1.35]
    return ratio, factor


# ── Movement potential ────────────────────────────────────────────────────────

def _interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation with clamping at the endpoints.

    Used by the movement / volume factors to convert what used to be coarse
    band lookups into smooth piecewise-linear functions. Endpoints below x0
    return y0; above x1 return y1; in between scale linearly. The previous
    band-based implementations had discrete jumps at the boundaries (e.g.
    ATR 1.99% → 1.00, ATR 2.00% → 1.10) that this function smooths out
    while preserving the same anchor values.
    """
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _movement_factor(atr_pct: float, bb_width_pct: float) -> float:
    """
    Answers: "if a signal fires, how likely is this stock to actually move?"

    Uses ATR% (typical daily range / price) and BB-width% (Bollinger Band
    squeeze vs expansion) as complementary volatility regime measures.

    Thresholds are calibrated for typical large-cap stocks:
      ATR% ~0.8–2% = normal, > 2.5% = high vol, < 0.5% = very quiet.
      BB-width ~4–8% = normal, > 10% = wide/trending, < 2% = squeeze.

    A stock in a vol squeeze (narrow BB) may be about to move — but the
    direction is unknown, so we don't boost; we leave it neutral. The boost
    only applies when ATR confirms the stock already moves meaningfully.

    Both sub-scores are piecewise-linear functions of their input (previously
    quantized into 4-5 discrete bands). Anchor points preserve the original
    band values so the overall behaviour is unchanged at the band centres.
    """
    # ATR-based score: continuous in [0.80, 1.20] anchored to the previous bands.
    # The anchor map is: 0.5% → 0.80, 1.0% → 0.90, 1.5% → 1.00, 2.5% → 1.10, 3.0%+ → 1.20.
    # Interpolating between adjacent anchors gives a smooth ATR→score curve.
    if atr_pct <= 0.005:
        atr_score = 0.80
    elif atr_pct <= 0.010:
        atr_score = _interp(atr_pct, 0.005, 0.010, 0.80, 0.90)
    elif atr_pct <= 0.015:
        atr_score = _interp(atr_pct, 0.010, 0.015, 0.90, 1.00)
    elif atr_pct <= 0.025:
        atr_score = _interp(atr_pct, 0.015, 0.025, 1.00, 1.10)
    elif atr_pct <= 0.030:
        atr_score = _interp(atr_pct, 0.025, 0.030, 1.10, 1.20)
    else:
        atr_score = 1.20

    # BB-width score: continuous in [0.88, 1.10].
    # Anchors: 2% → 0.88 (squeeze), 4.5% → 0.95, 9% → 1.00, 12%+ → 1.10.
    # The squeeze region remains dampened (direction unclear) — only the
    # quantization is removed.
    if bb_width_pct <= 0.02:
        bb_score = 0.88
    elif bb_width_pct <= 0.045:
        bb_score = _interp(bb_width_pct, 0.02, 0.045, 0.88, 0.95)
    elif bb_width_pct <= 0.09:
        bb_score = _interp(bb_width_pct, 0.045, 0.09, 0.95, 1.00)
    elif bb_width_pct <= 0.12:
        bb_score = _interp(bb_width_pct, 0.09, 0.12, 1.00, 1.10)
    else:
        bb_score = 1.10

    # Average the two, clamp to [0.80, 1.20]
    raw = (atr_score + bb_score) / 2
    return round(max(0.80, min(1.20, raw)), 3)


# ── Volume cross-method factor ────────────────────────────────────────────────

def _volume_factor(vol_ratio: float, abs_combined: float, coherence_ratio: float) -> float:
    """
    Cross-method volume confirmation: does current volume support the aggregate signal?

    This is distinct from the volume multiplier already baked into technical_score.
    That one shapes the *technical sub-score*. This one asks: given that multiple
    methods are pointing in the same direction, is the market actually trading on it?

    Smooth implementation (previously a 4-step tier function):
      - The volume-only base scales linearly from 0.92 (suspiciously low, ≤0.5×)
        to 1.00 (normal) to 1.15 (boost potential at ≥2×).
      - The *boost* portion (any contribution above 1.00) is then gated by a
        quality factor = signal_strength × coherence_quality. The original
        intent — boosts only apply when volume, signal, AND coherence all
        line up — is preserved continuously: weak signal or scattered methods
        smoothly fade the boost back toward 1.00 instead of cliff-edging it.
      - The dampening side (vol_ratio < 1.0) applies unconditionally, same
        as before.

    Anchor points preserve the prior tier values: 0.5× vol → 0.92,
    2.0× vol + strong signal + coherent methods → 1.15.
    """
    # Volume-only base — symmetric ramp around 1.0× volume.
    if vol_ratio <= 0.5:
        base = 0.92
    elif vol_ratio < 1.0:
        base = _interp(vol_ratio, 0.5, 1.0, 0.92, 1.00)
    elif vol_ratio < 2.0:
        base = _interp(vol_ratio, 1.0, 2.0, 1.00, 1.15)
    else:
        base = 1.15

    # Below 1.0× volume — dampening applies regardless of signal quality
    # (the market simply isn't trading on it).
    if base <= 1.00:
        return round(base, 3)

    # Above 1.0× — gate the boost magnitude by signal/coherence quality.
    # signal_q ramps 0 → 1 over abs_combined ∈ [0.20, 0.35]
    signal_q = _interp(abs_combined,    0.20, 0.35, 0.0, 1.0)
    # coh_q   ramps 0 → 1 over coherence_ratio ∈ [0.40, 0.60]
    coh_q    = _interp(coherence_ratio, 0.40, 0.60, 0.0, 1.0)
    quality  = signal_q * coh_q                            # multiplicative AND
    boost    = (base - 1.00) * quality
    return round(1.00 + boost, 3)


# ── Interaction adjustments ───────────────────────────────────────────────────

def _interaction_adjustment(
    combined: float,
    sentiment_score: float,
    technical_score: float,
    insider_sc: float,
    pc_score: float,
    vol_ratio: float,
) -> float:
    """
    Detect three specific market setups and return a small additive correction
    to the combined score (capped to ±0.15 in total).

    These are cases where two methods used together are more informative than
    their individual contributions to the weighted sum suggest.

    1. Smart money accumulation at technical support
       Insiders buying while the stock looks technically weak is a contrarian
       value-accumulation setup — institutions building a position before a
       technical breakout. The combined score may be artificially low because
       the technical drag cancels the insider signal.

    2. Options extreme + technical alignment
       Extreme put/call skew (heavy puts) at a technical support is a classic
       short-squeeze or capitulation setup. The contrarian put/call signal and
       the technical support are reinforcing.

    3. Strong news + volume confirmation
       A high-magnitude sentiment signal backed by 1.5× normal volume is more
       likely to be a genuine market-moving event than the same news on thin
       volume. Boost when both are present and aligned.
    """
    direction = 1 if combined >= 0 else -1
    adj = 0.0

    # 1. Insider accumulation at technical support
    #    Criteria: insiders bullish (>0.30), technical score weak/bearish (<-0.20),
    #              but combined still leans bullish — insiders are more right.
    if insider_sc > 0.30 and technical_score < -0.20 and combined > 0:
        adj += 0.10
        logger.debug("Interaction: insider accumulation at technical support (+0.10)")
    elif insider_sc < -0.30 and technical_score > 0.20 and combined < 0:
        adj -= 0.10
        logger.debug("Interaction: insider distribution at technical peak (-0.10)")

    # 2. Options extreme aligned with combined direction
    #    pc_score is contrarian: large positive = extreme puts → bullish bias.
    #    If options market is at an extreme AND it agrees with combined direction,
    #    the setup is more compelling.
    if abs(pc_score) >= 0.50 and (pc_score * direction) > 0:
        adj += 0.07 * direction
        logger.debug(f"Interaction: options extreme aligned with signal ({adj:+.2f})")

    # 3. News catalyst confirmed by volume
    #    High-magnitude sentiment AND elevated volume: news is being acted on.
    if abs(sentiment_score) >= 0.40 and vol_ratio >= 1.5 and (sentiment_score * direction) > 0:
        adj += 0.06 * direction
        logger.debug(f"Interaction: news catalyst confirmed by volume spike ({adj:+.2f})")

    return max(-0.15, min(0.15, adj))


# ── Sector alignment (second pass) ───────────────────────────────────────────

def _sector_alignment_factor(ticker: str, combined: float, signals_by_ticker: dict) -> float:
    sector_etf = _SECTOR_MAP.get(ticker.upper())
    if sector_etf is None:
        return 1.0
    sector_sig = signals_by_ticker.get(sector_etf)
    if sector_sig is None:
        return 1.0
    sector_combined = (
        sector_sig.sentiment_score * 0.40 +
        sector_sig.technical_score * 0.30 +
        sector_sig.insider_score   * 0.30
    )
    if abs(sector_combined) < 0.10:
        return 1.0
    if (combined > 0) == (sector_combined > 0):
        logger.debug(f"{ticker} ↔ {sector_etf} ALIGNED (sector={sector_combined:+.2f}) → +boost")
        return _SECTOR_ALIGN_BOOST
    logger.debug(f"{ticker} ↔ {sector_etf} CONTRADICTS (sector={sector_combined:+.2f}) → penalty")
    return _SECTOR_ALIGN_PENALTY


# ── Main entry point ──────────────────────────────────────────────────────────

def _gex_for(ticker: str, gex_context) -> tuple[str, Optional[float], str, float, float]:
    """
    Look up GEX data for a ticker.
    Returns (gex_signal, gamma_flip, max_pain_bias, expected_move_pct, oi_skew).
    """
    if gex_context is None:
        return "", None, "", 0.0, 0.0
    for sig in gex_context.signals:
        if sig.ticker.upper() == ticker.upper():
            return sig.gex_signal, sig.gamma_flip, sig.max_pain_bias, sig.expected_move_pct, sig.oi_skew
    return "", None, "", 0.0, 0.0


def _max_pain_score_for(ticker: str, gex_context) -> float:
    """
    Convert max pain distance into a [-1, +1] signal score.

    Logic
    ─────
    Max pain is the strike price where the aggregate dollar loss across all
    open options is minimised.  Market makers with net-short options books
    have an incentive to pin price near this level into expiration, creating
    a weak but statistically observable gravitational pull.

    Score formula
    ─────────────
      distance  = (max_pain - spot) / spot        signed, positive → bullish pull
      expiry_factor ∈ [0.05, 1.0]                 falls quickly beyond 7 days
      raw_score = distance × expiry_factor × 5    scale to make typical values ~±0.2–0.5

    The 5× scale compensates for typical distances (0.5–5%).  A 2% deviation
    3 days before expiry → 0.02 × 0.90 × 5 = 0.09 — a modest but real input.
    Beyond 14 days the weight is < 0.20, making this a near-zero contributor
    until the final two weeks of the cycle.
    """
    if gex_context is None:
        return 0.0
    for sig in gex_context.signals:
        if sig.ticker.upper() != ticker.upper():
            continue
        if sig.max_pain is None or sig.spot_price <= 0:
            return 0.0

        # Days until nearest expiry (use dominant_expiry stored on the signal)
        try:
            days = (date.fromisoformat(sig.dominant_expiry) - date.today()).days
        except (ValueError, AttributeError):
            return 0.0

        if days <= 0:
            return 0.0

        # Expiry-decay factor — strongest in the final week
        if days <= 2:
            expiry_factor = 1.00
        elif days <= 4:
            expiry_factor = 0.85
        elif days <= 7:
            expiry_factor = 0.65
        elif days <= 10:
            expiry_factor = 0.40
        elif days <= 14:
            expiry_factor = 0.22
        elif days <= 21:
            expiry_factor = 0.10
        else:
            expiry_factor = 0.05

        distance = (sig.max_pain - sig.spot_price) / sig.spot_price
        raw = distance * expiry_factor * 5.0
        score = round(max(-1.0, min(1.0, raw)), 3)

        if abs(score) >= 0.05:
            logger.debug(
                f"[max_pain] {ticker}: pain=${sig.max_pain:.2f} spot=${sig.spot_price:.2f} "
                f"dist={distance:+.2%} expiry={sig.dominant_expiry}({days}d) "
                f"factor={expiry_factor} → score={score:+.3f}"
            )
        return score
    return 0.0


def _gex_movement_modifier(gex_signal: str) -> float:
    """
    Adjust the movement_factor based on dealer gamma positioning.

    PINNED:    dealers are long gamma → they suppress moves → signal less likely to fire
               quickly → dampen movement factor.
    AMPLIFIED: dealers are short gamma → moves will be amplified → signal more likely
               to materialise and do so fast → boost movement factor.
    """
    if gex_signal == "PINNED":
        return 0.85
    if gex_signal == "AMPLIFIED":
        return 1.15
    return 1.0


def _coint_score_for(ticker: str, coint_context) -> float:
    """Look up the per-ticker cointegration directional lean ∈ [-1, +1].

    Positive = ticker is the *cheap/long* leg of one or more stretched
    cointegrated pairs (bullish mean-reversion bet). Negative = the *rich/short*
    leg. Zero when the ticker is in no tradeable pair.
    """
    if coint_context is None:
        return 0.0
    return float(coint_context.ticker_scores.get(ticker.upper(), 0.0))


def _pead_score_for(ticker: str, pead_context) -> tuple[float, float, int]:
    """Look up the PEAD signal for ``ticker``.

    Returns ``(pead_score, surprise_pct, days_since_report)``. Defaults to
    ``(0.0, 0.0, 0)`` when no entry exists — the ticker either has no recent
    earnings report or its drift window has already decayed to zero.
    """
    if pead_context is None:
        return 0.0, 0.0, 0
    for sig in pead_context.signals:
        if sig.ticker.upper() == ticker.upper():
            return float(sig.pead_score), float(sig.surprise_pct), int(sig.days_since_report)
    return 0.0, 0.0, 0


def build_signals(
    tickers: List[str],
    articles: List[NewsArticle],
    insider_trades: Optional[List[InsiderTrade]] = None,
    put_call_context=None,
    gex_context=None,         # Optional[GEXContext]
    market_mode_context=None, # Optional[MarketModeContext]
    opex_context=None,        # Optional[OpExContext]
    pead_context=None,        # Optional[PEADContext]
    coint_context=None,       # Optional[CointPairsContext]
    snapshots=None,           # Optional[List[TickerSnapshot]] — live prices for ext_gap
    session: Optional[str] = None,  # "rth" | "extended" | "overnight" | None (=rth)
    force_sentiment_engine: Optional[str] = None,  # pin news scoring to one engine (hold-review)
    corp_factors: Optional[dict] = None,  # {ticker: {f_split, f_dividend}} — additive corporate-action overlay
    fundamental_factors: Optional[dict] = None,  # {ticker: {f_value, f_quality, f_growth, f_short_squeeze}} — additive overlay
    borrow_context: Optional[dict] = None,  # {ticker: BorrowInfo} — IBKR short-borrow for the broker_advisor method
) -> List[TickerSignal]:
    """Build a TickerSignal for each ticker using all enabled methods.

    ``session`` activates the extended-hours profile: the stale-options /
    fresh-news weight overlay plus the ext_gap scorer (which needs the live
    snapshot prices in ``snapshots``). ``None`` behaves exactly like "rth" so
    every pre-extended-hours caller is bit-identical.
    """

    use_news      = settings.enable_news_sentiment
    # Sentiment velocity reuses article timestamps; no live fetch / LLM call required.
    use_sent_velocity = settings.enable_sentiment_velocity
    use_tech      = settings.enable_technical_analysis and settings.enable_fetch_data
    # Massive/Polygon server-side technicals (RSI+MACD). PROMOTED into the weighted
    # combined_score (2026-06-24) — in _BASE_WEIGHTS + active_flags + the combine +
    # coherence, and scored for ALL tickers (massive_tech_max_tickers defaults to 0,
    # so a weighted method never leaves capped-out tickers dampened). Still scored
    # alongside `tech` for the dashboard head-to-head; the two overlap, so the weight
    # is kept modest.
    use_massive   = settings.enable_massive_tech
    use_insider   = (
        (settings.enable_insider_trades or
         settings.enable_options_flow or
         settings.enable_sec_filings)
        and insider_trades is not None
    )
    use_put_call  = settings.enable_put_call and put_call_context is not None
    use_max_pain  = settings.enable_gex and gex_context is not None
    use_oi_skew   = settings.enable_gex and gex_context is not None
    # VWAP uses the OHLCV chart cache first, so it works even with ENABLE_FETCH_DATA=false.
    use_vwap      = settings.enable_vwap
    # Pattern recognition uses OHLCV chart cache first; fetches 2y history only on cold cache.
    use_pattern   = settings.enable_pattern_recognition
    # Price momentum uses OHLCV chart cache first; works with ENABLE_FETCH_DATA=false.
    use_momentum   = settings.enable_price_momentum
    # Sector-relative momentum: ticker − benchmark ETF. Same cache strategy as
    # price momentum; benchmark resolution is cached forever.
    use_sector_momentum = settings.enable_sector_relative_momentum
    # Market-relative momentum: ticker − SPY (diagnostic only, not weighted).
    use_market_momentum = settings.enable_market_relative_momentum
    # Money flow uses OHLCV chart cache first; works with ENABLE_FETCH_DATA=false.
    use_money_flow = settings.enable_money_flow
    # Trend strength (ADX/DMI + Donchian) is cache-first too; works with ENABLE_FETCH_DATA=false.
    use_trend_strength = settings.enable_trend_strength
    # PEAD requires a precomputed context (per-ticker SUE × time-decay derived from earnings).
    use_pead       = settings.enable_pead and pead_context is not None
    # IV Rank uses OHLCV chart cache first; works with ENABLE_FETCH_DATA=false.
    use_iv_rank    = settings.enable_iv_rank
    # IV Expression reads the GEX context's real options-chain IV — requires GEX context.
    use_iv_expr    = settings.enable_iv_expr and gex_context is not None
    # Cointegration reads per-ticker directional leans from a precomputed pairs context.
    use_coint      = settings.enable_cointegration and coint_context is not None
    # Extended-session gap. With the session profile OFF (default), ext_gap is in
    # the active method set in EVERY session so the method set + weight
    # normalisation are identical across the trading day (comparable scores) — it
    # still reads 0.0 in RTH by design (the daily technical stack already captures
    # the gap; the scorer fails closed). With the profile ON, it's off-hours-only.
    is_extended    = session is not None and session != "rth"
    use_ext_gap    = (settings.enable_extended_gap and snapshots is not None
                      and (is_extended or not settings.enable_extended_signal_profile))
    snap_price_by_ticker = (
        {s.ticker: float(s.price) for s in (snapshots or []) if getattr(s, "price", None)}
        if use_ext_gap else {}
    )
    # Broker advisor — IBKR short-borrow tilt. Needs the borrow context (fetched by
    # the pipeline from the live broker); inactive otherwise → method scores 0.
    use_broker_advisor = settings.enable_broker_advisor and borrow_context is not None

    if not use_news and not use_tech and not use_insider and not use_put_call:
        logger.warning("All analysis methods disabled — no signals will be generated.")
        return []

    # Dynamic weight profile from market mode (TRENDING/NEUTRAL/CHOPPY)
    weight_profile: Optional[dict] = None
    if (settings.enable_market_mode_switching
            and market_mode_context is not None
            and market_mode_context.mode != "NEUTRAL"):
        weight_profile = market_mode_context.weight_profile

    # Adaptive reweighting — multiply each method's weight by a multiplier
    # derived from its rolling solo win rate. Composes with whichever profile
    # is active (base or market-mode override); the OpEx amplifier below
    # still overrides max_pain explicitly.
    if settings.enable_adaptive_weights:
        base_for_adaptive = dict(weight_profile or _BASE_WEIGHTS)
        weight_profile, adaptive_mults = _apply_adaptive_multipliers(base_for_adaptive)
        non_trivial = {m: v for m, v in adaptive_mults.items() if abs(v - 1.0) >= 0.05}
        if non_trivial:
            ranked = sorted(non_trivial.items(), key=lambda kv: -kv[1])
            logger.info(
                "Adaptive weights — solo-WR multipliers: "
                + "  ".join(f"{m}={v:.2f}x" for m, v in ranked)
            )

    # IC-informed weights — tilt by reliability-adjusted IC (ICIR) over the UNBIASED
    # signals panel, shrunk toward the base weight by evidence. Default off; composes
    # as another multiplier on whatever profile is active (stacks with the win-rate
    # layer above when both are on). A thin panel ⇒ {} ⇒ base weights unchanged.
    if settings.enable_ic_weights:
        ic_mults = _ic_weight_multipliers()
        # An INVERTED method has been CORRECTED (its sign is flipped in the combine
        # below), so the IC anti-predictive floor no longer applies — lift it to base
        # weight rather than the 0.25× floor its (raw, backwards) IC would earn.
        for _m in _inverted_methods():
            if _m in ic_mults:
                ic_mults[_m] = 1.0
        if ic_mults:
            base_for_ic = dict(weight_profile or _BASE_WEIGHTS)
            weight_profile = {
                m: base_for_ic.get(m, _BASE_WEIGHTS.get(m, 0.0)) * ic_mults.get(m, 1.0)
                for m in base_for_ic
            }
            ranked = sorted(ic_mults.items(), key=lambda kv: -kv[1])
            logger.info(
                f"IC weights — {len(ic_mults)} method(s) cleared the confidence gate "
                f"(t≥{settings.ic_weight_min_t:g}): "
                + "  ".join(f"{m}={v:.2f}x" for m, v in ranked))
        else:
            logger.debug("IC weights — no method cleared the confidence gate yet "
                         "(panel too thin); base weights unchanged")

    # OpEx max-pain amplifier — boost max_pain weight during OpEx / Triple Witching week
    if (settings.enable_catalyst_timing
            and opex_context is not None
            and opex_context.in_opex_week):
        boosted_mp = 0.28 if opex_context.is_triple_witching else 0.20
        weight_profile = dict(weight_profile or _BASE_WEIGHTS)
        weight_profile["max_pain"] = boosted_mp
        logger.info(
            f"[catalyst] OpEx max-pain boost: weight → {boosted_mp:.2f}"
            + (" [TRIPLE WITCHING]" if opex_context.is_triple_witching else "")
        )

    # Extended-session overlay LAST so the staleness scaling applies to
    # whatever each options weight ended up as (mode profile, adaptivity, and
    # the OpEx boost all describe RTH dynamics — pinning happens in the
    # regular session, not in a 4 AM pre-market book). Gated by
    # enable_extended_signal_profile (default OFF → one fixed profile all day, so
    # scores are comparable across the trading day).
    if is_extended and settings.enable_extended_signal_profile:
        weight_profile = _extended_session_weight_overlay(dict(weight_profile or _BASE_WEIGHTS))
        logger.info(
            f"[extended] {session} session weight overlay: options-derived ×"
            f"{settings.extended_stale_options_weight_mult:g} (stale since RTH close), "
            f"news/velocity/gap ×{settings.extended_news_weight_mult:g}"
        )

    active_flags = {
        "news":       use_news,
        "sent_velocity": use_sent_velocity,
        "tech":       use_tech,
        "massive":    use_massive,
        "insider":    use_insider,
        "put_call":   use_put_call,
        "max_pain":   use_max_pain,
        "oi_skew":    use_oi_skew,
        "vwap":       use_vwap,
        "pattern":    use_pattern,
        "momentum":   use_momentum,
        "sector_momentum": use_sector_momentum,
        "market_momentum": use_market_momentum,
        "money_flow": use_money_flow,
        "trend_strength": use_trend_strength,
        "pead":       use_pead,
        "iv_rank":    use_iv_rank,
        "iv_expr":    use_iv_expr,
        "coint":      use_coint,
        "ext_gap":    use_ext_gap,
        "broker_advisor": use_broker_advisor,
    }
    weights      = _normalised_weights(active_flags, weight_profile=weight_profile)
    # Method INVERSION: a method whose raw signal is reliably anti-predictive across
    # horizons (net of beta) contributes with a FLIPPED sign — its backwards read
    # becomes a correct one. Config-driven (inverted_methods) + reversible; the panel
    # keeps the RAW score so the inversion stays re-validatable. (Coherence / source
    # agreement read the raw scores — a minor, accepted inconsistency.)
    _inv = _inverted_methods()
    for _m in _inv:
        if _m in weights and weights[_m]:
            weights[_m] = -weights[_m]
    if _inv:
        logger.debug(f"[aggregator] method inversion active: {sorted(_inv)} "
                     "(sign-flipped in the combine; raw kept in the panel)")
    active_count = sum(active_flags.values())

    mode_label = f" [{market_mode_context.mode}]" if market_mode_context else ""
    logger.info(
        f"Signal weights{mode_label} — "
        f"news={weights['news']:.0%}  sv={weights['sent_velocity']:.0%}  tech={weights['tech']:.0%}  "
        f"massive={weights['massive']:.0%}  "
        f"insider={weights['insider']:.0%}  put_call={weights['put_call']:.0%}  "
        f"max_pain={weights['max_pain']:.0%}  oi_skew={weights['oi_skew']:.0%}  "
        f"vwap={weights['vwap']:.0%}  pattern={weights['pattern']:.0%}  "
        f"momentum={weights['momentum']:.0%}  "
        f"sector_mom={weights['sector_momentum']:.0%}  market_mom={weights['market_momentum']:.0%}  money_flow={weights['money_flow']:.0%}  "
        f"trend={weights['trend_strength']:.0%}  "
        f"pead={weights['pead']:.0%}  iv_rank={weights['iv_rank']:.0%}  iv_expr={weights['iv_expr']:.0%}  "
        f"coint={weights['coint']:.0%}  ext_gap={weights['ext_gap']:.0%}  "
        f"broker_advisor={weights['broker_advisor']:.0%}"
    )

    signals        = []
    combined_scores: dict = {}

    # Per-ticker scoring is parallelised across a bounded thread pool — the loop body
    # is independent per ticker (DeepSeek sentiment + Massive/OHLCV reads are the
    # cost, all I/O-bound), so concurrency collapses the wall time from sum→max with
    # IDENTICAL scores. The two per-run caps that were sequential counters
    # (intraday_30m_max_tickers / massive_tech_max_tickers = "first N in order") are
    # PRE-SELECTED into deterministic sets, so which tickers get the capped treatment
    # is unchanged and order-independent. build_signals is already invoked
    # concurrently from the hold-review pool, so this whole stack is proven
    # thread-safe (sentiment counters are lock-guarded; the OHLCV cache is per-ticker).
    _n30_cap = settings.intraday_30m_max_tickers
    if not settings.enable_intraday_30m or _n30_cap <= 0:
        allow_30m_set = frozenset(tickers)
    else:
        allow_30m_set = frozenset(tickers[:_n30_cap])
    _mass_cap = settings.massive_tech_max_tickers
    if not use_massive or _mass_cap <= 0:
        allow_massive_set = frozenset(tickers)
    else:
        allow_massive_set = frozenset(tickers[:_mass_cap])

    # Pre-import the lazily-loaded scorers ONCE (not per worker thread) so the pool
    # never contends on the import lock / first-import.
    _compute_tf = _blend_tf = None
    if settings.enable_multi_timeframe_signals:
        from src.signals.multi_timeframe import (
            compute_timeframe_scores as _compute_tf,
            blend_timeframes as _blend_tf,
        )
    _compute_massive = None
    if use_massive:
        from src.signals.massive_tech import compute_massive_tech_score as _compute_massive

    # Learned continuation/reversal orientation for the trend-predictability
    # methods — computed ONCE per batch (cached + fail-soft), captured by the
    # per-ticker closure so every ticker is scored on one consistent calibration.
    _trend_orientation = (calibrate_trend_orientation()
                          if settings.enable_trend_predictability_methods else None)

    def _score_ticker(ticker):

        # ── Method 1: News sentiment (the LEVEL) ──────────────────────────
        sentiment_score = 0.0
        news_rationale  = "News sentiment disabled."
        relevant_articles: list = []
        if use_news or use_sent_velocity:
            relevant_articles = filter_relevant_articles(ticker, articles)
        if use_news:
            sentiment_score, news_rationale = analyse_sentiment(
                ticker, relevant_articles, force_engine=force_sentiment_engine)

        # ── Method 1b: Sentiment velocity (Δsentiment, not level) ─────────
        # Rate of change of news tone (recent window − prior window). The change
        # leads short-horizon moves better than the static level. Deterministic,
        # no extra LLM/API cost — reuses the article timestamps already stored.
        sent_velocity_score = 0.0
        sent_recent         = 0.0
        sent_prior          = 0.0
        if use_sent_velocity:
            sent_velocity_score, sent_recent, sent_prior, _sv_n = compute_sentiment_velocity(
                ticker, relevant_articles,
                recent_hours=settings.sentiment_velocity_recent_hours,
                prior_hours=settings.sentiment_velocity_prior_hours,
            )

        # ── Method 2: Technical analysis ─────────────────────────────────
        tech_result: TechnicalResult = EMPTY_RESULT
        if use_tech:
            tech_result = compute_technical_score(ticker)
        technical_score = tech_result.score
        vol_ratio       = tech_result.vol_ratio
        atr_pct         = tech_result.atr_pct
        bb_width_pct    = tech_result.bb_width_pct

        # ── Method 3: Smart money ─────────────────────────────────────────
        insider_sc          = 0.0
        insider_summary     = ""
        cluster_detected    = False
        cluster_size        = 0
        persist_detected    = False
        persist_count       = 0
        persist_buyer       = ""
        if use_insider:
            (insider_sc, cluster_detected, cluster_size,
             persist_detected, persist_count, persist_buyer) = _insider_score(ticker, insider_trades)
            insider_summary = build_insider_summary(ticker, insider_trades)

        # ── Method 4: Put/call ratio (contrarian) ─────────────────────────
        pc_score = 0.0
        if use_put_call:
            pc_score = _put_call_score_for(ticker, put_call_context)

        # ── Method 5: Max pain gravity (expiry-weighted) ──────────────────
        mp_score = 0.0
        if use_max_pain:
            mp_score = _max_pain_score_for(ticker, gex_context)

        # ── GEX lookup (used by Method 6, movement factor, and signal fields) ──
        gex_sig, gamma_flip, max_pain_bias, expected_move_pct, oi_skew = _gex_for(ticker, gex_context)

        # ── Method 6: OI-weighted directional skew ────────────────────────
        # Positive = call OI concentrated above spot (bullish positioning)
        # Negative = put OI concentrated below spot (bearish positioning)
        oi_skew_score = round(oi_skew, 3) if use_oi_skew else 0.0

        # ── Method 7: VWAP distance (mean-reversion) ─────────────────────
        # Score > 0: price below VWAP → institutional buyers expected (bullish)
        # Score < 0: price above VWAP → institutional sellers expected (bearish)
        vwap_score    = 0.0
        vwap_dist_pct = 0.0
        if use_vwap:
            vwap_score, vwap_dist_pct = compute_vwap_score(ticker)

        # ── Method 8: Chart pattern recognition ──────────────────────────
        # Score derived from the pattern's historical win rate for this ticker.
        # Positive = bullish pattern with good historical accuracy.
        # Negative = bearish pattern with good historical accuracy.
        pattern_score = 0.0
        pattern_name  = ""
        if use_pattern:
            pattern_score, pattern_name = compute_pattern_score(ticker)

        # ── Method 9: Price Momentum (Perceived Value) ────────────────────
        # Captures the self-reinforcing dynamic where perceived value creates
        # trends: rising prices attract more capital, confirming the trend.
        # Score normalised against the ticker's own return distribution so it
        # adapts to each security's volatility regime.
        momentum_score   = 0.0
        momentum_1m_pct  = 0.0
        momentum_3m_pct  = 0.0
        if use_momentum:
            momentum_score, momentum_1m_pct, momentum_3m_pct = compute_price_momentum_score(ticker)

        # ── Method 9b: Sector-Relative Momentum (beta-stripped alpha) ────
        # Same multi-period structure as absolute momentum, but on residual
        # returns: ticker − sector ETF. Strips out sector beta so a strong
        # sector tailwind doesn't masquerade as ticker-level alpha. The
        # benchmark resolution is asset-type aware (sector ETF for stocks,
        # SPY for ETFs, none for commodities).
        sector_momentum_score   = 0.0
        sector_momentum_1m_pct  = 0.0
        sector_momentum_3m_pct  = 0.0
        sector_benchmark_used   = ""
        if use_sector_momentum:
            (sector_momentum_score, sector_momentum_1m_pct,
             sector_momentum_3m_pct, sector_benchmark_used) = compute_sector_relative_momentum_score(ticker)

        # ── Method 9c: Market-Relative Momentum (ticker − SPY) ───────────
        # Residual vs the broad market. PROMOTED into the weighted combine
        # (2026-06-24) but kept LIGHT (market_momentum=0.08): market_rel =
        # sector_rel + (sector − market), so a heavy weight would double-count
        # the beta already carried by sector_momentum — the small weight bounds
        # that overlap. Also stored on the signal for the email / Claude prompt
        # so divergences from sector_momentum stay visible.
        market_momentum_score   = 0.0
        market_momentum_1m_pct  = 0.0
        market_momentum_3m_pct  = 0.0
        if use_market_momentum:
            (market_momentum_score, market_momentum_1m_pct,
             market_momentum_3m_pct, _) = compute_market_relative_momentum_score(ticker)

        # ── Method 10: Money Flow Indicators ─────────────────────────────
        # Composite of MFI (14-period volume-weighted RSI), CMF (20-period
        # Chaikin Money Flow), and OBV slope z-score.
        # Positive = institutional accumulation; negative = distribution.
        money_flow_score = 0.0
        mfi_value        = 50.0
        cmf_value        = 0.0
        if use_money_flow:
            money_flow_score, mfi_value, cmf_value = compute_money_flow_score(ticker)

        # ── Method 10b: Trend Strength (ADX/DMI + Donchian breakout) ─────
        # Wilder directional movement scaled by ADX trend strength + Turtle
        # 20-day channel breakout. Positive = confirmed uptrend; negative =
        # confirmed downtrend; ~0 = chop (signal intentionally dampened).
        trend_strength_score = 0.0
        adx_value            = 0.0
        trend_label          = "NO_DATA"
        if use_trend_strength:
            trend_strength_score, adx_value, trend_label = compute_trend_strength_score(ticker)

        # ── Method 10c: Trend-predictability (Kaufman/ADX, split long/short) ─
        # Signed Kaufman efficiency + ADX·DMI as four one-sided methods. Sparse
        # (a name is up- XOR down-trending), so they fold into combined_score as
        # an additive overlay below, NOT the normalised pool. Cache-first fetch
        # reuses the bar trend_strength just warmed above — no extra network hit.
        kaufman_long_v = kaufman_short_v = adx_long_v = adx_short_v = 0.0
        if settings.enable_trend_predictability_methods:
            _tp = compute_trend_predictability_scores(ticker, orientation=_trend_orientation)
            kaufman_long_v  = _tp["kaufman_long"]
            kaufman_short_v = _tp["kaufman_short"]
            adx_long_v      = _tp["adx_long"]
            adx_short_v     = _tp["adx_short"]

        # ── Method 11: Post-Earnings Announcement Drift (PEAD) ───────────
        # SUE × time-decay. Positive = bullish drift from recent earnings beat;
        # negative = bearish drift from recent miss. Decays linearly to 0 over
        # ``pead_decay_window_days`` (default 60) so signals fade naturally.
        pead_score_v    = 0.0
        pead_surprise   = 0.0
        pead_days       = 0
        if use_pead:
            pead_score_v, pead_surprise, pead_days = _pead_score_for(ticker, pead_context)

        # ── Method 12: IV Rank + Directional ─────────────────────────────
        # Uses 21-day realized vol percentile as a proxy for IV Rank.
        # High IR + extreme move → contrarian; low IR + trend → confirmation.
        # Self-normalised inputs make this robust to regime shifts.
        iv_rank_score_v   = 0.0
        iv_rank_v         = 50.0
        iv_rank_ret5d     = 0.0
        iv_rank_label_v   = "NEUTRAL"
        if use_iv_rank:
            iv_rank_score_v, iv_rank_v, iv_rank_ret5d, iv_rank_label_v = compute_iv_rank_score(ticker)

        # ── Method 13: IV Expression (real options chain) ────────────────
        # Reads true market-implied vol from the options chain (via gex_context)
        # and combines with options-market oi_skew to decide stock-vs-options
        # expression. Cheap options + strong directional skew → confirm; expensive
        # options + strong skew → fade. Uses historical gex caches for rank.
        iv_expr_score_v = 0.0
        iv_expr_rank_v  = 50.0
        iv_expr_skew_v  = 0.0
        iv_expr_label_v = "NEUTRAL"
        if use_iv_expr:
            iv_expr_score_v, iv_expr_rank_v, iv_expr_skew_v, iv_expr_label_v = compute_iv_expr_score(ticker, gex_context)

        # ── Method 14: Cointegration pairs (statistical arbitrage) ───────
        # Per-ticker net directional lean from the ticker's cheap/rich role across
        # stretched cointegrated pairs. Positive = long (cheap) leg; negative = short.
        coint_score_v = 0.0
        if use_coint:
            coint_score_v = _coint_score_for(ticker, coint_context)

        # ── Method 15: Extended-session gap momentum (off-hours only) ────
        # Live extended print vs last completed close in ATR units: the
        # pre-market gap forming in real time / the after-hours reaction to a
        # just-released catalyst. Always 0.0 during RTH runs.
        ext_gap_score_v = 0.0
        ext_gap_pct_v   = 0.0
        if use_ext_gap:
            ext_gap_score_v, ext_gap_pct_v = compute_extended_gap_score(
                ticker, snap_price_by_ticker.get(ticker), session=session,
            )

        # ── Method 16: Broker advisor (IBKR short-borrow squeeze tilt) ────
        # + = hard/expensive to short (squeeze tell → bullish, fades a SELL); 0 =
        # easy borrow / no data. The first method in the broker-aware group.
        broker_advisor_score_v = 0.0
        if use_broker_advisor:
            broker_advisor_score_v = compute_broker_advisor_score(borrow_context.get(ticker))

        # ── Multi-timeframe blend (30m / daily / weekly) ──────────────────
        # Recompute the 8 OHLCV methods on the faster + slower candles and
        # renormalise-blend them with the daily score. The blended value is what
        # the combine + coherence consume, so the live strategy reflects all
        # three timeframes. The daily component stays on the signal unchanged
        # (the panel's "Daily" IC category + email). Flag off ⇒ daily passthrough.
        tf_scores: dict = {}
        tech_eff, vwap_eff, momentum_eff, money_flow_eff = (
            technical_score, vwap_score, momentum_score, money_flow_score)
        trend_strength_eff, iv_rank_eff, pattern_eff, sector_momentum_eff = (
            trend_strength_score, iv_rank_score_v, pattern_score, sector_momentum_score)
        if settings.enable_multi_timeframe_signals:
            allow_30m = ticker in allow_30m_set
            tf_scores = _compute_tf(ticker, allow_30m=allow_30m)

            def _eff(method: str, daily_v: float) -> float:
                return _blend_tf({
                    "30m": tf_scores.get(f"{method}_30m"),
                    "1d":  daily_v,
                    "1w":  tf_scores.get(f"{method}_1w"),
                })

            tech_eff            = _eff("tech", technical_score)
            vwap_eff            = _eff("vwap", vwap_score)
            momentum_eff        = _eff("momentum", momentum_score)
            money_flow_eff      = _eff("money_flow", money_flow_score)
            trend_strength_eff  = _eff("trend_strength", trend_strength_score)
            iv_rank_eff         = _eff("iv_rank", iv_rank_score_v)
            pattern_eff         = _eff("pattern", pattern_score)
            sector_momentum_eff = _eff("sector_momentum", sector_momentum_score)

        # ── Massive server-side technicals (RSI+MACD) ─────────────────────
        # A WEIGHTED member of the combine below. allow_massive_set is every ticker
        # when massive_tech_max_tickers=0 (the default), so the pool is never dampened.
        massive_score = 0.0
        if use_massive and ticker in allow_massive_set:
            massive_score = _compute_massive(ticker)

        # ── Weighted combination ──────────────────────────────────────────
        # Technical methods use their multi-timeframe BLEND (*_eff); non-OHLCV
        # methods use their single live score.
        combined = (
            weights["news"]       * sentiment_score +
            weights["sent_velocity"] * sent_velocity_score +
            weights["tech"]       * tech_eff +
            weights["massive"]    * massive_score +
            weights["insider"]    * insider_sc +
            weights["put_call"]   * pc_score +
            weights["max_pain"]   * mp_score +
            weights["oi_skew"]    * oi_skew_score +
            weights["vwap"]       * vwap_eff +
            weights["pattern"]    * pattern_eff +
            weights["momentum"]   * momentum_eff +
            weights["sector_momentum"] * sector_momentum_eff +
            weights["market_momentum"] * market_momentum_score +
            weights["money_flow"] * money_flow_eff +
            weights["trend_strength"] * trend_strength_eff +
            weights["pead"]       * pead_score_v +
            weights["iv_rank"]    * iv_rank_eff +
            weights["iv_expr"]    * iv_expr_score_v +
            weights["coint"]      * coint_score_v +
            weights["ext_gap"]    * ext_gap_score_v +
            weights["broker_advisor"] * broker_advisor_score_v
        )

        # ── Interaction adjustments ───────────────────────────────────────
        # Small additive corrections for setups where two methods together
        # are more informative than their linear contributions suggest.
        if active_count >= 2:
            combined += _interaction_adjustment(
                combined, sentiment_score, technical_score,
                insider_sc, pc_score, vol_ratio,
            )

        # ── Corporate-action directional overlay (additive, event-driven) ──
        # f_split (forward-drift / reverse-distress) + f_dividend (increase / cut),
        # added OUTSIDE the normalised weight pool so a corporate action nudges the
        # handful of event tickers without dampening everyone else. Weight is a
        # placeholder — review once the f_split/f_dividend IC accrues.
        if corp_factors:
            _cf = corp_factors.get(ticker) or corp_factors.get(ticker.upper())
            if _cf:
                combined += settings.corp_action_factor_weight * (
                    float(_cf.get("f_split", 0.0)) + float(_cf.get("f_dividend", 0.0)))

        # ── Massive fundamental factors directional overlay (additive) ─────
        # value/quality/growth/short-squeeze, added OUTSIDE the normalised pool
        # (same idiom as corp_factors) so the capped/sparse fundamentals nudge the
        # event tickers without dampening the rest. Already IC-monitored in the panel.
        if fundamental_factors:
            _ff = fundamental_factors.get(ticker) or fundamental_factors.get(ticker.upper())
            if _ff:
                combined += settings.fundamental_factor_weight * (
                    float(_ff.get("f_value", 0.0)) + float(_ff.get("f_quality", 0.0))
                    + float(_ff.get("f_growth", 0.0)) + float(_ff.get("f_short_squeeze", 0.0)))

        # ── Trend-predictability directional overlay (additive) ────────────
        # The four scores are already oriented (continuation OR reversal, learned
        # per method + scaled by trend strength/confidence), so a single symmetric
        # weight suffices — the orientation, not the weight, encodes each context's
        # direction quality. Added outside the normalised pool (sparse per context).
        if settings.enable_trend_predictability_methods:
            combined += settings.trend_method_weight * (
                kaufman_long_v + kaufman_short_v + adx_long_v + adx_short_v)

        # ── Direction ─────────────────────────────────────────────────────
        direction: Direction
        if combined >= 0.15:
            direction = "BULLISH"
        elif combined <= -0.15:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # ── Coherence factor (continuous, replaces binary 1.25×/0.60×) ───
        # Technical methods use the blended (*_eff) values so coherence reflects
        # exactly what the combine consumed.
        method_scores = [
            (use_news,       sentiment_score),
            (use_sent_velocity, sent_velocity_score),
            (use_tech,       tech_eff),
            (use_massive,    massive_score),
            (use_insider,    insider_sc),
            (use_put_call,   pc_score),
            (use_max_pain,   mp_score),
            (use_oi_skew,    oi_skew_score),
            (use_vwap,       vwap_eff),
            (use_pattern,    pattern_eff),
            (use_momentum,   momentum_eff),
            (use_sector_momentum, sector_momentum_eff),
            (use_market_momentum, market_momentum_score),
            (use_money_flow, money_flow_eff),
            (use_trend_strength, trend_strength_eff),
            (use_pead,       pead_score_v),
            (use_iv_rank,    iv_rank_eff),
            (use_iv_expr,    iv_expr_score_v),
            (use_coint,      coint_score_v),
            (use_ext_gap,    ext_gap_score_v),
            (use_broker_advisor, broker_advisor_score_v),
        ]
        coherence_ratio, coherence_factor = _coherence_factor(combined, method_scores)

        # sources_agreeing kept for backward-compat with Claude's prompt context
        sources_agreeing = sum(
            1 for enabled, score in method_scores
            if enabled and abs(score) >= _AGREE_THRESHOLD
            and ((combined > 0 and score > 0) or (combined < 0 and score < 0))
        )

        # ── Movement potential (ATR + BB width + GEX) ────────────────────
        movement_factor = _movement_factor(atr_pct, bb_width_pct) * _gex_movement_modifier(gex_sig)
        movement_factor = round(max(0.70, min(1.30, movement_factor)), 3)

        # ── Cross-method volume confirmation ──────────────────────────────
        volume_factor = _volume_factor(vol_ratio, abs(combined), coherence_ratio)

        # ── Final confidence ──────────────────────────────────────────────
        raw_confidence = min(1.0, abs(combined) / 0.5)
        confidence = round(
            min(1.0, raw_confidence * coherence_factor * movement_factor * volume_factor),
            2,
        )

        # ── Rationale ─────────────────────────────────────────────────────
        rationale_parts = []
        if use_news:
            rationale_parts.append(news_rationale)
        if use_tech and technical_score != 0:
            rationale_parts.append(f"Technical score: {technical_score:+.2f}")
        if use_put_call and pc_score != 0:
            rationale_parts.append(f"Put/call signal: {pc_score:+.2f}")
        rationale = " | ".join(rationale_parts) if rationale_parts else "No rationale available."

        _sig = TickerSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            combined_score=round(combined, 4),
            sentiment_score=round(sentiment_score, 3),
            sentiment_velocity_score=round(sent_velocity_score, 3),
            sentiment_recent=round(sent_recent, 3),
            sentiment_prior=round(sent_prior, 3),
            technical_score=round(technical_score, 3),
            massive_score=round(massive_score, 3),
            insider_score=round(insider_sc, 3),
            put_call_score=round(pc_score, 3),
            max_pain_score=round(mp_score, 3),
            oi_skew_score=round(oi_skew_score, 3),
            vwap_score=round(vwap_score, 3),
            vwap_distance_pct=round(vwap_dist_pct, 2),
            rationale=rationale,
            insider_summary=insider_summary,
            sources_agreeing=sources_agreeing,
            gex_signal=gex_sig,
            gamma_flip=gamma_flip,
            max_pain_bias=max_pain_bias,
            expected_move_pct=expected_move_pct,
            insider_cluster_detected=cluster_detected,
            insider_cluster_size=cluster_size,
            insider_persistence_detected=persist_detected,
            insider_persistence_count=persist_count,
            insider_persistence_buyer=persist_buyer,
            pattern_score=round(pattern_score, 3),
            pattern_name=pattern_name,
            momentum_score=round(momentum_score, 3),
            momentum_1m_pct=round(momentum_1m_pct, 2),
            momentum_3m_pct=round(momentum_3m_pct, 2),
            sector_momentum_score=round(sector_momentum_score, 3),
            sector_benchmark=sector_benchmark_used,
            sector_momentum_1m_pct=round(sector_momentum_1m_pct, 2),
            sector_momentum_3m_pct=round(sector_momentum_3m_pct, 2),
            market_momentum_score=round(market_momentum_score, 3),
            market_momentum_1m_pct=round(market_momentum_1m_pct, 2),
            market_momentum_3m_pct=round(market_momentum_3m_pct, 2),
            money_flow_score=round(money_flow_score, 3),
            mfi_value=round(mfi_value, 2),
            cmf_value=round(cmf_value, 4),
            trend_strength_score=round(trend_strength_score, 3),
            adx_value=round(adx_value, 2),
            trend_strength_label=trend_label,
            kaufman_long_score=round(kaufman_long_v, 3),
            kaufman_short_score=round(kaufman_short_v, 3),
            adx_long_score=round(adx_long_v, 3),
            adx_short_score=round(adx_short_v, 3),
            pead_score=round(pead_score_v, 3),
            pead_surprise_pct=round(pead_surprise, 2),
            pead_days_since_report=int(pead_days),
            iv_rank_score=round(iv_rank_score_v, 3),
            iv_rank=round(iv_rank_v, 1),
            iv_rank_ret_5d_pct=round(iv_rank_ret5d, 2),
            iv_rank_label=iv_rank_label_v,
            iv_expr_score=round(iv_expr_score_v, 3),
            iv_expr_rank=round(iv_expr_rank_v, 1),
            iv_expr_oi_skew=round(iv_expr_skew_v, 3),
            iv_expr_label=iv_expr_label_v,
            coint_score=round(coint_score_v, 3),
            ext_gap_score=round(ext_gap_score_v, 3),
            ext_gap_pct=round(ext_gap_pct_v, 2),
            broker_advisor_score=round(broker_advisor_score_v, 3),
            timeframe_scores=tf_scores,
        )

        gex_str     = f"  gex={gex_sig}({gamma_flip})" if gex_sig else ""
        mp_str      = f"  mp={mp_score:+.2f}" if mp_score != 0.0 else ""
        skew_str    = f"  skew={oi_skew_score:+.2f}" if oi_skew_score != 0.0 else ""
        vwap_str    = f"  vwap={vwap_score:+.2f}({vwap_dist_pct:+.1f}%)" if vwap_score != 0.0 else ""
        pat_str  = f"  pat={pattern_score:+.2f}[{pattern_name}]" if pattern_name else ""
        mom_str2 = f"  mom={momentum_score:+.2f}({momentum_1m_pct:+.1f}%/1m)" if momentum_score != 0.0 else ""
        sm_str   = (
            f"  smom={sector_momentum_score:+.2f}(vs {sector_benchmark_used} {sector_momentum_1m_pct:+.1f}pp/1m)"
            if sector_momentum_score != 0.0 else ""
        )
        mm_str   = (
            f"  mmom={market_momentum_score:+.2f}(vs SPY {market_momentum_1m_pct:+.1f}pp/1m)"
            if market_momentum_score != 0.0 else ""
        )
        mf_str   = f"  mf={money_flow_score:+.2f}(mfi={mfi_value:.0f},cmf={cmf_value:+.2f})" if money_flow_score != 0.0 else ""
        ts_str   = f"  trend={trend_strength_score:+.2f}(adx={adx_value:.0f},{trend_label})" if trend_strength_score != 0.0 else ""
        pead_str = f"  pead={pead_score_v:+.2f}({pead_surprise:+.1f}%/{pead_days}d)" if pead_score_v != 0.0 else ""
        ivr_str  = f"  ivr={iv_rank_score_v:+.2f}(ir={iv_rank_v:.0f},{iv_rank_label_v})" if iv_rank_score_v != 0.0 else ""
        ivx_str  = f"  ivx={iv_expr_score_v:+.2f}(ir={iv_expr_rank_v:.0f},{iv_expr_label_v})" if iv_expr_score_v != 0.0 else ""
        coint_str = f"  coint={coint_score_v:+.2f}" if coint_score_v != 0.0 else ""
        gap_str  = f"  gap={ext_gap_score_v:+.2f}({ext_gap_pct_v:+.1f}%)" if ext_gap_score_v != 0.0 else ""
        cluster_str = f"  CLUSTER({cluster_size})" if cluster_detected else ""
        persist_str = f"  PERSIST({persist_count}×)" if persist_detected else ""
        sv_str      = f"  sv={sent_velocity_score:+.2f}" if sent_velocity_score != 0.0 else ""
        logger.info(
            f"{ticker}: {direction} (conf={confidence:.0%}, {sources_agreeing}/{active_count} agree) | "
            f"news={sentiment_score:+.2f}{sv_str}  tech={technical_score:+.2f}  "
            f"insider={insider_sc:+.2f}{cluster_str}{persist_str}  pc={pc_score:+.2f}{mp_str}{skew_str}{vwap_str}{pat_str}{mom_str2}{sm_str}{mm_str}{mf_str}{ts_str}{pead_str}{ivr_str}{ivx_str}{coint_str}{gap_str}  combined={combined:+.2f} | "
            f"coherence={coherence_ratio:.2f}({coherence_factor:.2f}x)  "
            f"movement={movement_factor:.2f}x  volume={volume_factor:.2f}x  "
            f"atr={atr_pct:.3f}  vol_ratio={vol_ratio:.2f}x{gex_str}"
        )
        return ticker, combined, _sig

    # Run the per-ticker scoring concurrently (bounded) — identical scores, ~max
    # wall time instead of the serial sum (DeepSeek sentiment was the long pole).
    # ex.map preserves input order, so `signals` is assembled in `tickers` order,
    # exactly as the sequential loop produced it. workers<=1 keeps the legacy path.
    _workers = max(1, int(getattr(settings, "signal_scoring_max_workers", 8) or 1))
    if _workers > 1 and len(tickers) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(_workers, len(tickers)),
                                thread_name_prefix="score") as _ex:
            _results = list(_ex.map(_score_ticker, tickers))
    else:
        _results = [_score_ticker(t) for t in tickers]
    for _tk, _combined, _sig in _results:
        combined_scores[_tk] = _combined
        signals.append(_sig)

    # ── Second pass: cross-sectional ranking overlay ─────────────────────────
    # Computes how each ticker's per-method scores deviate from the universe
    # mean (z-score per method), averages them, and adds the result × weight
    # to combined_score. Adds a relative-value view on top of the absolute
    # aggregation so the system stays calibrated when one regime makes every
    # absolute score skew the same way.
    cs_scores: dict = {}
    if settings.enable_cross_sectional and len(signals) >= 3:
        from src.signals.cross_sectional import compute_cross_sectional_scores
        cs_scores = compute_cross_sectional_scores(
            signals, zcap=settings.cross_sectional_zcap,
        )
        if cs_scores:
            cs_w = float(settings.cross_sectional_weight)
            updated_signals = []
            for sig in signals:
                cs = float(cs_scores.get(sig.ticker, 0.0))
                if abs(cs) < 1e-4:
                    updated_signals.append(sig.model_copy(update={"cross_sectional_score": 0.0}))
                    combined_scores[sig.ticker] = sig.combined_score
                    continue
                new_combined = sig.combined_score + cs_w * cs
                new_combined = max(-2.0, min(2.0, new_combined))
                # Re-derive direction + confidence on the adjusted combined
                if new_combined >= 0.15:
                    new_direction: Direction = "BULLISH"
                elif new_combined <= -0.15:
                    new_direction = "BEARISH"
                else:
                    new_direction = "NEUTRAL"
                # confidence is derived from |combined| / 0.5 in the original
                # build; reapply the same mapping but keep the existing
                # coherence/movement/volume factors baked into sig.confidence
                # by scaling proportionally to the change in |combined|.
                old_abs = max(abs(sig.combined_score), 0.05)
                scale = abs(new_combined) / old_abs
                new_conf = round(min(1.0, sig.confidence * scale), 2)
                updated_signals.append(sig.model_copy(update={
                    "combined_score":         round(new_combined, 4),
                    "cross_sectional_score":  round(cs, 4),
                    "direction":              new_direction,
                    "confidence":             new_conf,
                }))
                combined_scores[sig.ticker] = new_combined
                if abs(cs) >= 0.20:
                    logger.debug(
                        f"[cross_sectional] {sig.ticker}: cs={cs:+.3f}  "
                        f"combined {sig.combined_score:+.3f} -> {new_combined:+.3f}  "
                        f"conf {sig.confidence:.2f} -> {new_conf:.2f}"
                    )
            signals = updated_signals

    # ── Third pass: sector alignment ─────────────────────────────────────────
    signals_by_ticker = {s.ticker: s for s in signals}
    final_signals = []
    for sig in signals:
        raw_combined  = combined_scores.get(sig.ticker, 0.0)
        sector_factor = _sector_alignment_factor(sig.ticker, raw_combined, signals_by_ticker)
        if sector_factor != 1.0:
            adjusted = round(min(1.0, sig.confidence * sector_factor), 2)
            sig = sig.model_copy(update={"confidence": adjusted})
        final_signals.append(sig)

    return final_signals
