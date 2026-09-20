"""A name's standing against ITS OWN recent scores — the freshness filter's input.

The entry rules select on the CROSS-SECTION: is this name high against the other
names scored in this run. That question has a blind spot the panel measures
directly: a name the model rates highly *every* run carries no news on the day it
rates it highly, and those picks underperform. Sorting judged picks by the level
of their own score history, long net falls +2.12 → +0.83 per entry from the
second quintile to the top one.

So a second, orthogonal question is asked of every candidate: is this score also
unusual FOR THIS NAME. "Unusual" is the strict form — strictly above every score
the same model gave it over the previous `rank_entry_fresh_window_days` trading
days, or strictly below for a short. Measured over 1,773 runs / 65 days it lifts
every one of eleven selection rules (+0.39 to +1.85 per entry) and its held-out
half-window median was the best of the five conditions tested.

**A name with no standing is KEPT, not dropped.** Fewer than
`rank_entry_fresh_min_history` prior scores in the window means the model has
never rated it, not that it rated it badly — there is nothing for today's score
to be normal against, which makes it the limiting case of freshness rather than
the absence of it. That cohort measured BEST in nine of the eleven rules
(+1.75..+3.28 per entry) and is emphatically not a data-poverty cohort: its
median stock carries 16.6 years of daily bars, it is simply new to the scored
universe (earnings and trending discovery are 11.9% and 11.6% of its rows against
0.6% and 2.5% for the rest). Verified 2026-09-18: a RANDOM name from that cohort
LOSES on both sides (-0.41% long, -0.68% short, i.e. the round trip on a cohort
with no drift) while the cut's picks from it earn +1.35%/+1.55%, so the return is
the model's selection and not a favourable population.

Only the prior MAX, MIN and COUNT are needed, so this is one aggregate query per
trading day rather than a per-ticker history load. Fails OPEN: on any error every
name reads as "no standing" and the filter becomes a no-op, which reverts the
caller to its unfiltered cut (measured positive) rather than emptying the book.

History: `memory/ml-ohlcv-hl5-retrain-2026-09.md`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from loguru import logger

from config import settings

# {(asof_date, column, window_days): {ticker: (n_prior, hi, lo)}}
_CACHE: Dict[tuple, Dict[str, Tuple[int, float, float]]] = {}
_WARNED = False


def _window_days() -> int:
    return max(1, int(getattr(settings, "rank_entry_fresh_window_days", 30) or 30))


def min_history() -> int:
    return max(1, int(getattr(settings, "rank_entry_fresh_min_history", 10) or 10))


def _today() -> str:
    """The signal date whose history we are building — the cutoff is EXCLUSIVE,
    so a run never sees its own day's earlier runs. That is deliberate: the
    condition was measured that way (strictly-before), and it also makes the
    answer stable across the day, so a name cannot flip from fresh to stale
    between two ticks of the same session."""
    try:                                              # honour a walk-forward cutoff
        from src.analysis.asof import current_asof
        a = current_asof()
        if a:
            return str(a)[:10]
    except Exception:
        pass
    from datetime import datetime, timezone
    try:                                              # the SIGNAL date is an ET date
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _epoch_floor(column: str) -> Optional[str]:
    """The first signal date whose stored scores came from the CURRENT scorer.

    `combined_score` has no method epoch of its own; it is fingerprinted as a
    scorer by `code_version`, and its producer changes with the combine, so the
    ml_ohlcv epoch is the right floor there too while ml_ohlcv IS the combine.
    Fails open (None) — a missing epoch must not empty the window.
    """
    try:
        from src.signals.method_epochs import epoch_for
        d = epoch_for("ml_ohlcv")
        return str(d)[:10] if d else None
    except Exception:
        return None


def load_standings(column: str = "ml_ohlcv") -> Dict[str, Tuple[int, float, float]]:
    """``{ticker: (n_prior_scores, max_prior, min_prior)}`` over the window.

    Cached per (day, column, window): the window is strictly-before-today, so it
    changes once a day, not once a run.
    """
    global _WARNED
    day, w = _today(), _window_days()
    key = (day, column, w)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out: Dict[str, Tuple[int, float, float]] = {}
    try:
        from src.db import repo
        # The window is the last `w` DISTINCT signal dates strictly before today,
        # so a holiday or a dark day costs the window a calendar day, not a
        # trading one.
        days = repo.fetch_df(
            "SELECT DISTINCT signal_date FROM signals WHERE signal_date < ? "
            "ORDER BY signal_date DESC LIMIT ?", [day, w])
        if days is None or days.empty:
            _CACHE[key] = out
            return out
        start = str(days.signal_date.min())[:10]
        # EPOCH GUARD. "A new high against its own history" is only a statement
        # about the NAME if every score in the window came from the same scorer.
        # It does not: `ml_ohlcv`'s scale moves with each retrain (the daily 1%
        # artifact averages |0.0375| against the 30-minute 5% model's |0.097|,
        # 2.6x), so a window straddling a retrain would read almost every name
        # as a new high on day one and the filter would silently degenerate into
        # the unfiltered cut. The standing instead starts at the scorer epoch:
        # fewer names carry one for a day, and "no standing" is the state this
        # module already handles correctly. Same rule as every calibration here
        # — restrict to comparable data, never convert across the boundary.
        start = max(start, _epoch_floor(column) or start)
        df = repo.fetch_df(
            f"SELECT ticker, count(*) AS n, max({column}) AS hi, min({column}) AS lo "
            f"FROM signals WHERE signal_date >= ? AND signal_date < ? "
            f"AND {column} IS NOT NULL GROUP BY ticker", [start, day])
        if df is not None and not df.empty:
            for tk, n, hi, lo in zip(df.ticker, df.n, df.hi, df.lo):
                try:
                    out[str(tk)] = (int(n), float(hi), float(lo))
                except (TypeError, ValueError):
                    continue
        logger.info(f"[freshness] {column}: standings for {len(out):,} tickers over "
                    f"{len(days)} trading days ({start} → {day}, exclusive)")
    except Exception as e:                                     # fail OPEN
        if not _WARNED:
            logger.warning(f"[freshness] standings unavailable ({e}) — the filter is a no-op this run")
            _WARNED = True
        out = {}
    _CACHE[key] = out
    if len(_CACHE) > 8:
        for k in list(_CACHE)[:-4]:
            _CACHE.pop(k, None)
    return out


def is_fresh(ticker: str, value: Optional[float], side: str,
             standings: Optional[Dict[str, Tuple[int, float, float]]] = None,
             column: str = "ml_ohlcv") -> bool:
    """True when this score is a new extreme for the name, or the name has no
    standing. ``side`` is "L"/"BUY" (a new high) or "S"/"SELL" (a new low)."""
    if value is None or value != value:
        return False
    st = standings if standings is not None else load_standings(column)
    rec = st.get(ticker)
    if rec is None or rec[0] < min_history():
        return True                       # no standing — kept, see the docstring
    _n, hi, lo = rec
    v = float(value)
    return v > hi if str(side).upper() in ("L", "BUY", "BULLISH") else v < lo


def describe(ticker: str, value: Optional[float], side: str,
             standings: Optional[Dict[str, Tuple[int, float, float]]] = None,
             column: str = "ml_ohlcv") -> str:
    """One clause for the recommendation rationale, so the reason a name was
    taken is visible in the ledger and the email rather than inferred."""
    st = standings if standings is not None else load_standings(column)
    rec = st.get(ticker)
    if rec is None or rec[0] < min_history():
        return "no score history in the window"
    n, hi, lo = rec
    ref = hi if str(side).upper() in ("L", "BUY", "BULLISH") else lo
    return f"new {_extreme(side)} vs {n} prior scores ({ref:+.3f})"


def _extreme(side: str) -> str:
    return "high" if str(side).upper() in ("L", "BUY", "BULLISH") else "low"


def reset_cache() -> None:
    """Tests only."""
    _CACHE.clear()
