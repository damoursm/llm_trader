"""
IV Expression — Stock-vs-options directional bias from the real options chain.

Distinct from the RV-proxy ``iv_rank`` method: this one pulls **true market-
implied volatility** straight from the options chain that's already fetched
for GEX (``gex_context.signals[*].expected_move_pct``), and combines it with
the options market's own directional positioning (``oi_skew``). The output
is a [-1, +1] score that tells you both *which direction* the options
market is pricing and *how cheaply* it's pricing it — which together
decide whether the directional thesis should be expressed in stock, in
long options, or faded entirely.

Inputs (all from the day's GEX context — no new fetches)
────────────────────────────────────────────────────────
  expected_move_pct  Market-implied ±1σ move from ATM straddle ÷ spot.
                     This is the *current* IV reading from the live chain.
  oi_skew            OI-weighted directional positioning ∈ [-1, +1].
                     +1 = call OI piled far above spot (bullish).
                     -1 = put OI piled far below spot (bearish).
  gex_signal         Dealer gamma regime: PINNED | AMPLIFIED | NEUTRAL.

Real IV Rank (no synthetic proxy)
─────────────────────────────────
The ``expected_move_pct`` value is volatile in absolute terms — a 2% expected
move on SPY is sleepy; 2% on NVDA is sleepy too; 19% on XLK is a vol-event.
To compare meaningfully we *rank* each ticker's current IV against its **own
recent history**, reconstructed by reading the prior ``cache/gex_*.json``
files on disk (saved daily by ``gamma_exposure._save_cache``). A ticker
with 6+ historical readings is ranked by percentile; with fewer than 6, the
fallback compares the current IV to the universe-median IV in today's GEX
context so the method still produces a useful signal on cold-cache days.

Score logic (stock-vs-options expression decision)
───────────────────────────────────────────────────

  HIGH IV (rank ≥ 75) — options priced rich, vol mean-reverts.
    Strong oi_skew (|x| ≥ 0.50) → "FADE_PREMIUM": options market pricing
      a directional move *and* premium is rich. Vol-sellers fade; we score
      AGAINST the options direction with -0.55 × sign(oi_skew). Stock
      expression preferred over long options.
    Weak oi_skew → "EXPENSIVE_NEUTRAL": event-pricing without conviction.
      Small fade: -0.20 × sign(oi_skew).

  LOW IV (rank ≤ 25) — options priced cheap, no event being priced.
    Strong oi_skew → "CHEAP_DIRECTIONAL_LONG"/"CHEAP_DIRECTIONAL_SHORT":
      cheap options + decisive positioning is a high-conviction expression
      setup. Score WITH the options direction at +0.55 × sign(oi_skew).
      Either stock or long options work as expression.
    Weak oi_skew → "CHEAP_COMPLACENT": low conviction, mild trend bias of
      +0.20 × sign(oi_skew).

  MID IV (25 < rank < 75) — typical regime.
    Score = 0.30 × oi_skew. Options market's directional view contributes,
    but neither cheap nor expensive enough to amplify.

  AMPLIFIED dealer gamma adjustment (+0.10 × sign):
    When the same direction signal is amplified by short-gamma dealer
    positioning, dampen the contrarian fade (HIGH IV) or boost the
    confirmatory expression (LOW IV) by 0.10 in the favourable direction.

The score is clipped to [-1, +1] after additive adjustment.

Returns
───────
  (score, iv_rank_real, oi_skew, expression_label)

Tickers without a GEX entry return (0.0, 50.0, 0.0, "NO_OPTIONS_DATA")
— the score does not contribute to combined for those tickers (zero weight
effect).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


CACHE_DIR = Path("cache")
_HISTORY_DAYS = 60      # only consider gex caches from the trailing 60 days
_MIN_HIST_N   = 6       # minimum historical IV readings to use percentile mode

_HIGH_IV = 75.0
_LOW_IV  = 25.0
_STRONG_OI_SKEW = 0.50


def _load_iv_history(ticker: str, today: date) -> List[float]:
    """Read ``expected_move_pct`` for *ticker* from prior gex_*.json caches.

    Caches older than ``_HISTORY_DAYS`` (file mtime / parsed report_date) are
    skipped. Today's cache file is excluded so the function returns *prior*
    readings — the caller compares them against the live value.
    """
    if not CACHE_DIR.exists():
        return []

    cutoff = today - timedelta(days=_HISTORY_DAYS)
    history: List[Tuple[date, float]] = []

    for path in CACHE_DIR.glob("gex_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        try:
            rep_date = date.fromisoformat(data.get("report_date", ""))
        except (ValueError, TypeError):
            continue

        if rep_date >= today or rep_date < cutoff:
            continue

        for sig in data.get("signals", []) or []:
            if str(sig.get("ticker", "")).upper() != ticker.upper():
                continue
            em = sig.get("expected_move_pct")
            if em is None:
                continue
            try:
                history.append((rep_date, float(em)))
            except (TypeError, ValueError):
                continue
            break  # one entry per ticker per cache file

    history.sort(key=lambda kv: kv[0])
    return [v for _, v in history]


def _universe_median_iv(gex_context) -> Optional[float]:
    if gex_context is None or not gex_context.signals:
        return None
    vals = [
        float(s.expected_move_pct)
        for s in gex_context.signals
        if s.expected_move_pct is not None and s.expected_move_pct > 0
    ]
    if len(vals) < 3:
        return None
    return float(np.median(vals))


def _get_gex_entry(ticker: str, gex_context):
    if gex_context is None:
        return None
    for sig in gex_context.signals:
        if sig.ticker.upper() == ticker.upper():
            return sig
    return None


def compute_iv_expr_score(
    ticker: str,
    gex_context,
) -> Tuple[float, float, float, str]:
    """Return ``(score, iv_rank_real, oi_skew, label)``.

    ``score`` ∈ [-1, +1] — stock-vs-options expression directional bias.
    ``iv_rank_real`` ∈ [0, 100] — percentile of current IV within trailing
      history, or universe-relative score when history is too thin.
    ``oi_skew`` ∈ [-1, +1] — options-market directional positioning (raw).
    ``label`` — expression-decision token (see module docstring).
    """
    sig = _get_gex_entry(ticker, gex_context)
    if sig is None or sig.expected_move_pct is None or sig.expected_move_pct <= 0:
        return 0.0, 50.0, 0.0, "NO_OPTIONS_DATA"

    current_iv  = float(sig.expected_move_pct)
    oi_skew     = float(sig.oi_skew or 0.0)
    gex_signal  = str(sig.gex_signal or "NEUTRAL")
    today       = date.today()

    # ── Real IV rank vs trailing history (preferred) ─────────────────────
    history = _load_iv_history(ticker, today)
    if len(history) >= _MIN_HIST_N:
        hist_arr = np.asarray(history, dtype=float)
        iv_rank_real = float((hist_arr < current_iv).sum()) / float(len(hist_arr)) * 100.0
        mode = "history"
    else:
        # ── Fallback: universe-relative rank ─────────────────────────────
        # Compare current_iv to today's universe median. Mapped to a [0, 100]
        # range with median = 50 by tanh-shrinkage. Not a true percentile but
        # carries the regime information needed when no history exists.
        median = _universe_median_iv(gex_context) or current_iv
        if median > 0:
            ratio = current_iv / median
            # ratio = 1 → 50; ratio = 2 → ~85; ratio = 0.5 → ~15
            iv_rank_real = float(50.0 + 50.0 * np.tanh((ratio - 1.0) * 1.5))
        else:
            iv_rank_real = 50.0
        mode = "universe"
    iv_rank_real = round(max(0.0, min(100.0, iv_rank_real)), 1)

    # ── Score logic — expression decision ────────────────────────────────
    skew_sign = float(np.sign(oi_skew)) if abs(oi_skew) > 0.05 else 0.0
    strong_skew = abs(oi_skew) >= _STRONG_OI_SKEW

    if iv_rank_real >= _HIGH_IV:
        # Expensive options — contrarian / fade-premium regime
        if strong_skew:
            score = -0.55 * skew_sign
            label = "FADE_PREMIUM"
        elif skew_sign != 0.0:
            score = -0.20 * skew_sign
            label = "EXPENSIVE_NEUTRAL"
        else:
            score = 0.0
            label = "EXPENSIVE_NEUTRAL"
    elif iv_rank_real <= _LOW_IV:
        # Cheap options — confirmation / cheap-directional regime
        if strong_skew:
            score = 0.55 * skew_sign
            label = "CHEAP_DIRECTIONAL_LONG" if skew_sign > 0 else "CHEAP_DIRECTIONAL_SHORT"
        elif skew_sign != 0.0:
            score = 0.20 * skew_sign
            label = "CHEAP_COMPLACENT"
        else:
            score = 0.0
            label = "CHEAP_COMPLACENT"
    else:
        # Mid IV — modest directional bias from options positioning
        score = 0.30 * oi_skew  # use the raw skew here for smoother gradient
        if abs(score) >= 0.05:
            label = "MID_IV_DIRECTIONAL"
        else:
            label = "NEUTRAL"

    # ── Amplified dealer gamma adjustment ────────────────────────────────
    # When dealers are short gamma (AMPLIFIED), directional moves accelerate.
    # Add +0.10 × sign(score) to push slightly further in the favourable
    # direction. PINNED has no boost — dealers are stabilising vol.
    if gex_signal == "AMPLIFIED" and abs(score) >= 0.05:
        score += 0.10 * float(np.sign(score))

    score = round(max(-1.0, min(1.0, score)), 3)

    logger.debug(
        f"[iv_expr] {ticker}: iv={current_iv:.2f}%  rank={iv_rank_real:.0f} ({mode})  "
        f"oi_skew={oi_skew:+.2f}  gex={gex_signal}  label={label}  score={score:+.3f}"
    )
    return score, iv_rank_real, round(oi_skew, 3), label
