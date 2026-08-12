"""Market-relative method skill — win rates net of the benchmark's own move.

An ABSOLUTE win rate cannot separate "this signal works" from "the market went
up". Measured 2026-07-27 on 296k ticker-days, a low-volatility screen won 53.5%
of the time at 5 days — and only 46.1% net of SPY, against a 48.6% baseline. The
entire apparent edge was beta.

Two corrections, and the second is the one that is easy to miss:

1. **Net of the benchmark.** A method's call is a win only if the stock beat SPY
   in the direction called, so a rising tide lifts nothing.
2. **The baseline is NOT 50%.** The cap-weighted index beats its typical
   constituent, so the median stock is market-relative-negative: measured at
   **48.8% / 48.6% / 48.2%** up-share at 1/5/10 days. Judging a method against
   50% on this basis holds it to a bar ~1.4pp too high and makes a perfectly
   ordinary method look broken. The baseline is therefore MEASURED from the same
   panel rather than assumed — a half-migrated basis (relative numerator, 50%
   bar) is worse than either pure basis.

Deliberately scoped to WEIGHTING, not to the hard filter or to P&L reporting.
The split follows what each measurement decides:

    drop / invert a method   -> signal quality -> market-relative
    method weighting         -> signal quality -> market-relative  (this module)
    sizing, edge blend, P&L  -> money          -> absolute

You do not trade market-relative: the book is outright long/short and its P&L is
absolute, so alpha you cannot capture must not drive sizing. But whether a
signal CONTAINS information is a different question from what it earns, and beta
is a confound there.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from config.settings import settings

from loguru import logger  # project configures loguru sinks only

# Fallback baseline (market-relative up-share of the median stock) when it
# cannot be measured. Measured value at the 1-week horizon on 2026-07-27.
_FALLBACK_BASELINE = 48.6

_CACHE: dict = {}
_SKILL_LOCK = threading.Lock()


def _horizon_col() -> str:
    """Which `compute_directional_perf` horizon the weighting reads."""
    return str(getattr(settings, "market_relative_horizon", "1w"))


def market_relative_skill(side: Optional[str] = None) -> Dict[str, dict]:
    """``{method: {"trades": n, "win_rate": pct, "baseline": pct}}`` — each
    method's market-relative hit rate on ``side`` (``None`` = both sides pooled).

    ``win_rate`` is the share of that method's calls where the stock beat the
    benchmark IN THE DIRECTION CALLED; ``baseline`` is the same statistic over
    every call in the panel, i.e. the number a method must exceed to be adding
    anything. Both come from `compute_directional_perf`, which is already the
    project's market-relative surface.

    Cached for ``ic_weight_cache_seconds`` — `build_signals` runs many times per
    tick and the panel advances once. ``{}`` on any failure, so every caller
    falls back to the absolute basis rather than to a wrong number.
    """
    if not getattr(settings, "enable_market_relative_weighting", False):
        return {}
    key = f"skill:{side or 'both'}"
    hit = _CACHE.get(key)
    if hit and (time.time() - hit["ts"]) < float(settings.ic_weight_cache_seconds):
        return hit["data"]                      # fast path — no lock when fresh
    # MISS → serialise, so the concurrent build_signals callers (main pass +
    # _HoldReviewBranch + shadow arms) share ONE pass over the directional panel
    # instead of each running their own (measured 2x per tick).
    with _SKILL_LOCK:
        return _market_relative_skill_compute(side, key)


def _market_relative_skill_compute(side: Optional[str], key: str) -> Dict[str, dict]:
    """The heavy path, always under ``_SKILL_LOCK``; re-checks the cache so a
    caller that queued behind another thread reuses its result."""
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit["ts"]) < float(settings.ic_weight_cache_seconds):
        return hit["data"]

    out: Dict[str, dict] = {}
    try:
        from src.analysis.simulated_trades import compute_directional_perf
        df = compute_directional_perf(min_n=int(settings.market_relative_min_obs))
        if df is None or df.empty or "side" not in df.columns:
            raise ValueError("no directional panel")
        want = {"buy": "bull", "sell": "bear"}.get(str(side or "").lower(), "both")
        sub = df[df["side"] == want]
        h = _horizon_col()
        hit_col, n_col = f"hit_{h}", f"n_{h}"
        if hit_col not in sub.columns:
            raise ValueError(f"horizon {h} not in the directional panel")

        baseline = _measure_baseline(df, h)
        for _, r in sub.iterrows():
            m, w, n = r.get("method"), r.get(hit_col), r.get(n_col)
            if m is None or w is None or n is None:
                continue
            try:
                w, n = float(w), int(n)
            except (TypeError, ValueError):
                continue
            if n <= 0 or w != w:            # NaN guard
                continue
            out[m] = {"trades": n, "win_rate": round(w, 2), "baseline": baseline}
    except Exception as e:
        logger.debug(f"[market_relative] unavailable: {e}")
        out = {}

    _CACHE[key] = {"ts": now, "data": out}
    return out


def _measure_baseline(df, horizon: str) -> float:
    """The market-relative up-share of the typical call — the bar a method has
    to beat. Measured, not assumed: it is ~48.6%, not 50%, because the
    cap-weighted index beats its median constituent."""
    try:
        both = df[df["side"] == "both"]
        col, ncol = f"hit_{horizon}", f"n_{horizon}"
        if col not in both.columns or both.empty:
            return _FALLBACK_BASELINE
        w = both[col].astype(float)
        n = both[ncol].astype(float)
        ok = w.notna() & n.notna() & (n > 0)
        if not ok.any():
            return _FALLBACK_BASELINE
        # Observation-weighted mean across methods = the panel's own up-share.
        return round(float((w[ok] * n[ok]).sum() / n[ok].sum()), 2)
    except Exception:
        return _FALLBACK_BASELINE


def market_relative_baseline() -> float:
    """The measured bar a method must beat (%), for display and for centring."""
    skill = market_relative_skill(None)
    for rec in skill.values():
        b = rec.get("baseline")
        if b:
            return float(b)
    return _FALLBACK_BASELINE


def reset_cache() -> None:
    _CACHE.clear()
