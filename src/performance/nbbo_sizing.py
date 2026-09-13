"""NBBO-based position sizing — size by the book you are actually crossing.

Measured 2026-09-02 over 30 days of the live ledger (385 trades since
2026-08-03; 343 with both a measured entry NBBO and an H/L pivot label anchored
at the trade's own entry price; 247 tickers, 24 entry days, 83% settled), the
two sides behave differently enough that one shared curve would be the wrong
shape:

**LONG — the DIRECTION is unaffected by the spread; only the COST is.**
Day-clustered keep-minus-drop *pivot* excess is ~0 at every threshold
(t −0.40..+0.68, halves flipping), while the *net* excess RISES as the cut
loosens: +0.88 pp/day at <4 bp, +0.91 at <6 bp, ~+1.0 at <8..15 bp, and
**+1.38 (t +2.08) at <20 bp** — the only cell of eighteen to clear t 2. Net
TOTAL over the window: −102.4% ungated → +8.6% at <4 bp. So a long-side spread
rule is an execution-cost filter: cut the widest names hard, leave the rest
alone. Tightening a long book below ~10 bp buys nothing (its pivot win rate
peaks at <2 bp on halves that flip 0.69/0.46 — noise).

**SHORT — the direction itself improves, monotonically, as the book tightens.**
Pivot win rate 58.8% ungated → 65.6% (<12 bp) → 72.7% (<5 bp) → 76.7% (<4 bp,
p .003) → 86.4% (<3 bp). Net TOTAL −46.9% → +25.7% (<4 bp), positive through
~15 bp and negative from 20 bp. Not a session artifact: inside RTH alone, <5 bp
shorts ran 81% win / +0.95 net against 50% / −1.61 for ≥5 bp (+31 pp). It is
the LLM funnel's shorts that improve most (55% → 78% at <4 bp); follow-through
shorts start at 69% and gain less.

Hence a near-flat long curve with a hard step past ``nbbo_size_long_wide_bps``,
and a short curve boosted below ``nbbo_size_short_knee_bps`` then monotonically
decreasing above it.

Both curves are SMOOTHED across their boundary, for the same reason
``enable_continuous_regime_threshold`` exists: a quoted spread is a noisy
instantaneous read, and a bare step would let 19.9 vs 20.1 bp change a position
by 3×. Each ramp sits on the LENIENT side of its boundary, so "over 20 bp" still
means "sized a lot less than under 20 bp" and "under 5 bp" still means "sized
up".

Input is the QUOTED point-in-time half-spread on the RTH basis
(``liquidity_forecast.quoted_halfspread_bps``) — not the mapped expected
deviation, and not the session-widened value, because the session is already
charged by ``extended_size_multiplier`` / ``overnight_size_multiplier``.

**NO MEASURED BOOK = NO TILT.** When neither a live NBBO nor an IBKR-measured
book is available the multiplier is exactly 1.0. In the same study the
structural proxies (Corwin-Schultz / class table) separated no outcomes at any
threshold — the contrast they produced was +0.09 pp with p 0.91 — so consuming
one here would be sizing on noise while looking identical to sizing on evidence.

Caveats this shipped with (see memory/liquidity-forecast-2026-08.md): 24 entry
days; most day-clustered t's ≤ 1.6; SHORT net flips sign across the two time
halves (H1 +0.56, H2 −0.36 at <6 bp) though its win rate stays >50% in both;
the window spans the 08-19 confidence fix and the 08-22/24 retrains. The
multipliers are audited per trade (``nbbo_size_multiplier``,
``nbbo_half_bps_at_entry``, ``nbbo_estimator_at_entry``) so the tilt can be
judged on its own record rather than re-argued.
"""
from __future__ import annotations

from typing import Optional, Tuple

from loguru import logger

from config.settings import settings

# Smoothing-band widths, in bp. Module constants rather than settings: they are
# the shape of the transition, not the decision, and a knob here would only
# invite turning a smooth curve back into the cliff it was written to avoid.
_LONG_RAMP_BPS = 4.0
_SHORT_TIGHT_BAND_BPS = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def long_multiplier(bps: float) -> float:
    """Full size up to the cut, then a ramp down to the wide-book multiplier."""
    wide_bps = float(settings.nbbo_size_long_wide_bps)
    wide = float(settings.nbbo_size_long_wide_mult)
    floor = float(settings.nbbo_size_floor_mult)
    if bps <= wide_bps:
        return 1.0
    if bps >= wide_bps + _LONG_RAMP_BPS:
        return max(wide, floor)
    frac = (bps - wide_bps) / _LONG_RAMP_BPS
    return max(1.0 + frac * (wide - 1.0), floor)


def short_multiplier(bps: float) -> float:
    """Boosted below the knee, then monotonically decreasing above it.

    Non-increasing over the whole domain and continuous at the knee, where it
    passes through exactly 1.0.
    """
    knee = float(settings.nbbo_size_short_knee_bps)
    tight = float(settings.nbbo_size_short_tight_mult)
    decay = max(1e-6, float(settings.nbbo_size_short_decay_bps))
    floor = float(settings.nbbo_size_floor_mult)
    band = min(_SHORT_TIGHT_BAND_BPS, max(knee, 0.0))
    if bps <= knee - band:
        return max(tight, 1.0)
    if bps < knee:
        frac = (bps - (knee - band)) / band if band > 0 else 1.0
        return max(tight, 1.0) + frac * (1.0 - max(tight, 1.0))
    return max(1.0 / (1.0 + (bps - knee) / decay), floor)


def nbbo_size_multiplier(ticker: str, action: Optional[str],
                         session: Optional[str] = None) -> Tuple[float, dict]:
    """``(multiplier, diag)`` for one entry. 1.0 whenever there is no view.

    ``action`` is the ledger's BUY/SELL; anything not SELL takes the long curve
    (a WATCH/HOLD never reaches this path).
    """
    diag = {"mult": 1.0, "bps": None, "raw_bps": None,
            "estimator": None, "side": None, "reason": "disabled"}
    if not bool(getattr(settings, "enable_nbbo_sizing", False)):
        return 1.0, diag
    is_short = str(action or "").upper() == "SELL"
    diag["side"] = "short" if is_short else "long"
    try:
        from src.performance.liquidity_forecast import quoted_halfspread_bps
        q = quoted_halfspread_bps(ticker, session=session)
    except Exception as exc:                                   # fail-soft: never block an entry
        logger.debug(f"[nbbo_sizing] {ticker}: quote lookup failed ({exc}) — no tilt")
        q = None
    if not q or q.get("bps") is None:
        diag["reason"] = "no measured book"
        return 1.0, diag
    bps = float(q["bps"])
    mult = short_multiplier(bps) if is_short else long_multiplier(bps)
    cap = max(1.0, float(settings.nbbo_size_short_tight_mult))
    mult = round(_clamp(mult, float(settings.nbbo_size_floor_mult), cap), 3)
    diag.update(mult=mult, bps=bps, raw_bps=q.get("raw_bps"),
                estimator=q.get("estimator"), reason="ok")
    return mult, diag
