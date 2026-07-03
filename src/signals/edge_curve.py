"""Term-structure of edge — per-ticker horizon synthesis from MEASURED IC.

For each ticker we weight every method's LIVE score by that method's measured
information coefficient at each horizon (from the ``simulated_trades`` panel), so
the weights are purely empirical and SIGN-AWARE:

* positive IC  → the method's score counts in its own direction,
* negative IC  → the score is FLIPPED (an anti-predictive method is a contrarian
                 tell, not noise),
* |IC| ≈ 0     → the method drops out (no measured skill at that horizon).

The result is an **edge curve** ``edge(h)`` over the horizons
30m/3h/6h/1d/3d/1w/2w/1m (directional conviction), plus a cost-aware
``exp_gross(h)`` / ``net(h)`` expected gross return per horizon. The holding
horizon is the argmax net-of-cost return among horizons that also clear a
conviction floor; the LLM may CONFIRM or SHORTEN it (``cap_horizon``); the matched
exit (in ``tracker``) closes a position once its horizon window passes unless it
is still strongly confirmed.

NO static weights — per design the weighting is the IC alone. With a thin panel
the curve is noisy by construction; that is an accepted trade-off (the matched
exit and the LLM cap are the safety rails).
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

from loguru import logger

from config import settings
from src.analysis.simulated_trades import HORIZON_LABELS

# Wall-clock duration of each horizon (hours) — used by the matched-exit time-stop.
HORIZON_HOURS: Dict[str, float] = {
    "30m": 0.5, "3h": 3.0, "6h": 6.0, "1d": 24.0,
    "3d": 72.0, "1w": 168.0, "2w": 336.0, "1m": 720.0,
}
# Map each horizon to the LLM's coarse ``time_horizon`` bucket.
HORIZON_BUCKET: Dict[str, str] = {
    "30m": "SHORT-TERM", "3h": "SHORT-TERM", "6h": "SHORT-TERM",
    "1d": "SWING", "3d": "SWING",
    "1w": "POSITION", "2w": "POSITION", "1m": "POSITION",
}
# Ascending duration order (for the "never lengthen" cap).
HORIZON_ORDER: Dict[str, int] = {h: i for i, h in enumerate(HORIZON_LABELS)}
# Representative horizon for each LLM bucket — the longest member, so capping to a
# bucket only shortens when the LLM picked a strictly shorter bucket.
BUCKET_HORIZON: Dict[str, str] = {
    "SHORT-TERM": "6h", "SWING": "3d", "POSITION": "1m", "N/A": "1m", "": "1m",
}

# Cache the heavy IC matrix across ticks (the OHLCV join is expensive).
_ic_cache: dict = {}


def _num(x) -> Optional[float]:
    """Coerce to a finite float, else None (NaN / below-min_n IC cells)."""
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def get_ic_matrix(days: Optional[int] = None, min_n: Optional[int] = None,
                  ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """``{method: {horizon: (ic, ret)}}`` from the simulated_trades panel.

    Cached for ``horizon_ic_cache_seconds`` — the simulated-trade forward-return
    join is heavy, and the matrix only shifts as new runs accrue. Returns ``{}``
    (horizon synthesis idles) when the panel has no usable IC yet."""
    days = settings.horizon_ic_days if days is None else days
    min_n = settings.horizon_ic_min_n if min_n is None else min_n
    key = (days, min_n)
    now = time.time()
    hit = _ic_cache.get(key)
    if hit and (now - hit["ts"]) < settings.horizon_ic_cache_seconds:
        return hit["matrix"]

    matrix: Dict[str, Dict[str, Tuple[float, float]]] = {}
    try:
        from src.analysis.simulated_trades import compute_method_perf
        # dedupe="last" PINNED: the horizon system's live IC matrix keeps the
        # documented one-row-per-(day, ticker, method) convention. The dashboard's
        # trades view moved to entry-EVENT semantics (dedupe="events", 2026-07-02)
        # — a live decision input must not change as a side effect of that.
        df = compute_method_perf(days=days, min_n=min_n, dedupe="last")
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                cells: Dict[str, Tuple[float, float]] = {}
                for h in HORIZON_LABELS:
                    ic = _num(r.get(f"ic_{h}"))
                    ret = _num(r.get(f"ret_{h}"))
                    if ic is not None and ret is not None:
                        cells[h] = (ic, ret)
                if cells:
                    matrix[str(r["method"])] = cells
    except Exception as e:  # pragma: no cover - defensive (DB / cache hiccup)
        logger.warning(f"[edge_curve] IC matrix unavailable ({e}) — horizon synthesis idle this run")
    _ic_cache[key] = {"ts": now, "matrix": matrix}
    return matrix


def reset_cache() -> None:
    """Drop the cached IC matrices (tests / forced refresh)."""
    _ic_cache.clear()
    _dir_cache.clear()


# Cache the heavy direction-conditional matrix (separate from the pooled one).
_dir_cache: dict = {}


def get_directional_ic_matrix(days: Optional[int] = None, min_n: Optional[int] = None,
                              ) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, int]]]]:
    """``{method: {horizon: {"bull"|"bear"|"both": (skill, mkt_rel_yield, n)}}}``.

    From ``simulated_trades.compute_directional_perf`` (market-relative, split by
    side). ``skill = 2·(market-relative hit% / 100 − 0.5)`` ∈ [-1, 1] — 0 at a
    market coin-flip (drift removed), +1 if the side's calls always beat the
    market, negative if they lose (→ the side gets flipped). Cached like the pooled
    matrix. Returns ``{}`` when the panel has no usable directional skill yet."""
    days = settings.horizon_ic_days if days is None else days
    min_n = settings.horizon_ic_min_n if min_n is None else min_n
    key = (days, min_n)
    now = time.time()
    hit = _dir_cache.get(key)
    if hit and (now - hit["ts"]) < settings.horizon_ic_cache_seconds:
        return hit["matrix"]

    matrix: Dict[str, Dict[str, Dict[str, Tuple[float, float, int]]]] = {}
    try:
        from src.analysis.simulated_trades import compute_directional_perf
        df = compute_directional_perf(days=days, min_n=min_n,
                                      benchmark=settings.horizon_market_benchmark)
        if df is not None and not df.empty:
            for _, r in df.iterrows():
                method, side = str(r["method"]), str(r["side"])
                for h in HORIZON_LABELS:
                    hitpct = _num(r.get(f"hit_{h}"))
                    yld = _num(r.get(f"yield_{h}"))
                    n = _num(r.get(f"n_{h}"))
                    if hitpct is None or yld is None:
                        continue
                    skill = 2.0 * (hitpct / 100.0 - 0.5)
                    (matrix.setdefault(method, {}).setdefault(h, {})
                        [side]) = (skill, yld, int(n or 0))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[edge_curve] directional matrix unavailable ({e}) — shadow curve idle")
    _dir_cache[key] = {"ts": now, "matrix": matrix}
    return matrix


def compute_directional_edge_curve(scores: Dict[str, float],
                                   dmatrix: Dict[str, Dict[str, Dict[str, Tuple[float, float, int]]]],
                                   cost_hurdle_pct: Optional[float] = None,
                                   prior_n: Optional[int] = None) -> Dict[str, dict]:
    """Direction-aware, MARKET-NEUTRAL edge curve.

    For each method, the side matching the live score's sign (bull/bear) is used,
    its per-side skill SHRUNK toward the method's both-sides skill by sample size
    (the thin per-side estimate borrows strength from the pooled one — still a
    measured prior, not a static weight). ``edge``/``exp_gross``/``net`` mirror the
    pooled curve but on market-relative yields, so ``net`` is alpha over the market
    net of cost."""
    from src.performance.spread import effective_cost_hurdle_pct
    hurdle = float(effective_cost_hurdle_pct() if cost_hurdle_pct is None else cost_hurdle_pct)
    prior_n = int(settings.horizon_dir_shrink_prior_n if prior_n is None else prior_n)
    out: Dict[str, dict] = {}
    for h in HORIZON_LABELS:
        w_num = w_den = a_num = a_den = 0.0
        for method, s in scores.items():
            if not s:
                continue
            cells = dmatrix.get(method, {}).get(h)
            if not cells:
                continue
            side = "bull" if s > 0 else "bear"
            side_cell, both_cell = cells.get(side), cells.get("both")
            if side_cell is None and both_cell is None:
                continue
            if side_cell is not None:
                skill, yld, n = side_cell
                if both_cell is not None:           # shrink toward the both-sides skill
                    b_skill, b_yld, _bn = both_cell
                    denom = n + prior_n
                    if denom > 0:
                        skill = (n * skill + prior_n * b_skill) / denom
                        yld = (n * yld + prior_n * b_yld) / denom
            else:
                skill, yld, n = both_cell
            w_num += skill * s
            w_den += abs(skill)
            a = abs(skill * s)
            a_num += a * (yld if skill >= 0 else -yld)   # sign-corrected market-rel yield
            a_den += a
        edge = (w_num / w_den) if w_den > 1e-12 else 0.0
        exp_gross = (a_num / a_den) if a_den > 1e-12 else 0.0
        out[h] = {"edge": round(edge, 4), "exp_gross": round(exp_gross, 4),
                  "net": round(exp_gross - hurdle, 4)}
    return out


def compute_edge_curve(scores: Dict[str, float],
                       ic_matrix: Dict[str, Dict[str, Tuple[float, float]]],
                       cost_hurdle_pct: Optional[float] = None) -> Dict[str, dict]:
    """Per-horizon ``{edge, exp_gross, net}`` from live scores × the IC matrix.

    edge(h)      = Σ ic(m,h)·s_m / Σ|ic(m,h)|                 directional conviction ∈ [-1,1]
                   (sign-aware: a negative-IC method flips the sense of its score)
    exp_gross(h) = Σ |ic·s| · sign(ic)·ret(m,h) / Σ |ic·s|    expected GROSS return %
                   (sign-corrected per-method yield, weighted by reliability×strength)
    net(h)       = exp_gross(h) − cost_hurdle_pct
    """
    from src.performance.spread import effective_cost_hurdle_pct
    hurdle = float(effective_cost_hurdle_pct() if cost_hurdle_pct is None else cost_hurdle_pct)
    out: Dict[str, dict] = {}
    for h in HORIZON_LABELS:
        w_num = w_den = a_num = a_den = 0.0
        for method, s in scores.items():
            if not s:
                continue                       # exactly-zero score = no view
            cell = ic_matrix.get(method, {}).get(h)
            if cell is None:
                continue                       # no measured skill at this horizon
            ic, ret = cell
            w_num += ic * s
            w_den += abs(ic)
            a = abs(ic) * abs(s)
            a_num += a * (ret if ic >= 0 else -ret)   # sign-corrected gross yield
            a_den += a
        edge = (w_num / w_den) if w_den > 1e-12 else 0.0
        exp_gross = (a_num / a_den) if a_den > 1e-12 else 0.0
        out[h] = {
            "edge": round(edge, 4),
            "exp_gross": round(exp_gross, 4),
            "net": round(exp_gross - hurdle, 4),
        }
    return out


def select_horizon(curve: Dict[str, dict]) -> dict:
    """Cost-aware holding-horizon pick: the argmax net edge among horizons that
    both clear the cost hurdle (net > 0) AND carry directional conviction
    (|edge| ≥ ``horizon_min_conviction``). Falls back to the best-net horizon with
    ``tradeable=False`` when none clear — still informative for the prompt."""
    min_conv = float(settings.horizon_min_conviction)
    candidates = [(h, c) for h, c in curve.items()
                  if c["net"] > 0 and abs(c["edge"]) >= min_conv]
    tradeable = bool(candidates)
    pool = candidates or list(curve.items())
    if not pool:
        return {"target_horizon": "", "horizon_label": "N/A", "direction": "NEUTRAL",
                "conviction": 0.0, "net_edge_pct": 0.0, "tradeable": False}
    h, c = max(pool, key=lambda kv: kv[1]["net"])
    direction = "BULLISH" if c["edge"] > 0 else "BEARISH" if c["edge"] < 0 else "NEUTRAL"
    return {
        "target_horizon": h,
        "horizon_label": HORIZON_BUCKET.get(h, "N/A"),
        "direction": direction,
        "conviction": round(abs(c["edge"]), 4),
        "net_edge_pct": c["net"],
        "expected_move_pct": c["exp_gross"],   # gross expected FAVOURABLE move (magnitude, pre-cost)
        "tradeable": tradeable,
    }


def cap_horizon(mechanical_h: str, llm_bucket: Optional[str]) -> str:
    """The LLM may CONFIRM or SHORTEN the mechanical horizon, never lengthen it.

    Returns the shorter (by ascending duration) of the mechanical horizon and the
    LLM bucket's representative horizon. An empty mechanical horizon (no view) is
    returned unchanged. Used at trade time so the synthesis cannot stretch a
    position past the horizon its measured edge supports."""
    if not mechanical_h:
        return mechanical_h
    llm_h = BUCKET_HORIZON.get((llm_bucket or "").upper().strip(), mechanical_h)
    if HORIZON_ORDER.get(mechanical_h, 0) <= HORIZON_ORDER.get(llm_h, len(HORIZON_LABELS)):
        return mechanical_h
    return llm_h


def horizon_hours(h: str) -> Optional[float]:
    """Wall-clock duration of a horizon label, or None if unknown/empty."""
    return HORIZON_HOURS.get(h)


# ── Expected-move / market-aligned upside ranking ──────────────────────────

def market_direction_from_regime(regime: Optional[str]) -> str:
    """Map a macro-regime label to a coarse market direction: UP | DOWN | NEUTRAL.
    The regime layer owns the market call (alpha/beta split) — selection only
    amplifies in the direction the regime already chose."""
    r = (regime or "").upper()
    if r == "RISK_ON":
        return "UP"
    if r in ("RISK_OFF", "PANIC"):
        return "DOWN"
    return "NEUTRAL"


def market_alignment(position_direction, market_direction: str) -> str:
    """``aligned`` | ``counter`` | ``neutral`` — does the position lean with the
    regime? (A long in a risk-on regime is aligned — beta is a tailwind.)"""
    pos = str(getattr(position_direction, "value", position_direction) or "").upper()
    if market_direction == "NEUTRAL" or pos not in ("BULLISH", "BEARISH"):
        return "neutral"
    is_long = pos == "BULLISH"
    if (is_long and market_direction == "UP") or (not is_long and market_direction == "DOWN"):
        return "aligned"
    return "counter"


def upside_score(conviction: float, expected_move_pct: float, alignment: str) -> float:
    """The selection rank key — probability (≈conviction) × magnitude (expected
    FAVOURABLE move %) × alignment factor. Picks the biggest expected move in the
    regime's direction; a counter-regime position is haircut (soft, not banned)."""
    factor = float(settings.horizon_counter_market_mult) if alignment == "counter" else 1.0
    return round(max(0.0, conviction) * max(0.0, expected_move_pct) * factor, 5)
