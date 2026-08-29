"""Payoff-shaped rank mapping (2026-08-14 user directive).

The plain centered rank assumes a method's payoff is LINEAR in its within-run
rank. The measured decile curves say it is not: ``ext_gap`` is convex (the
extremes carry nearly everything) while the momentum family is n-shaped — its
top decile is BEARISH at the pivot horizon (the 2026-08-13 rank experiment:
decile-extreme selection helped ext_gap t +3.5 and hurt momentum t −2..−3.3).
This module replaces the linear grid with each method's OWN measured
rank→payoff curve, so an anti-predictive extreme becomes reversal signal
instead of damage.

Construction (`calibrate_rank_shapes`), per method over the signals panel:

1.  within-day percentile of the method's non-zero raw scores, bucketed into
    deciles against the settled H/L pivot target (winsorized 1/99);
2.  each decile's mean payoff is SHRUNK toward the method's own ordinary-
    least-squares LINE through the curve (per-decile prior ``rank_shape_prior_n``
    observations) — shrinkage removes the wiggles a 43-day window invents while
    keeping the slope the data insists on;
3.  the curve is DEMEANED (observation-weighted): the pivot target's +0.3-0.5%
    drift and the method's own mean payoff must not become a permanent
    directional tilt — the shape is the rank-CONDITIONAL payoff only;
4.  normalized by its max |value| onto [−1, +1] at the decile midpoints.

Serving (`shape_score`) linearly interpolates the run percentile through the
curve. Fail-soft everywhere, in order: shaping disabled → identity; a method
below ``rank_shape_min_rows`` settled rows (or the whole panel unavailable) →
identity; a curve whose demeaned amplitude is below ``_AMP_FLOOR`` (rank
carries ~no payoff — noise must not zero a method; dropping methods is the
win-rate machinery's job) → identity. Identity = the plain centered rank.

Point-in-time: the calibration reads ``build_panel``, which honours
``analysis_asof``; the TTL cache is registered in
``asof.reset_all_calibration_caches`` so a walk-forward step cannot reuse a
stale curve. Consumption-time only (the inversion architecture) — persisted
scores stay RAW, covered by the same CONFIDENCE_EPOCH 2026-08-14 as the rank
basis itself when both ship in one restart.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from config.settings import settings

# Decile midpoints the curves are anchored at (percentile space).
_MIDS = np.arange(0.05, 1.0, 0.10)

# Minimum demeaned-curve amplitude (payoff %) below which the rank→payoff
# relationship is treated as noise and the plain centered rank is kept.
_AMP_FLOOR = 0.05

_CACHE: dict = {"ts": 0.0, "shapes": None}


def calibrate_rank_shapes(days: Optional[int] = None) -> Dict[str, List[float]]:
    """``{method: [10 curve values in [-1, 1]]}`` for every method that clears
    the evidence gate AND the amplitude floor. Missing method = identity."""
    import pandas as pd

    from src.analysis.signal_panel import build_panel
    from src.signals.agreement import FAMILY_OF

    panel = build_panel(horizons=(5,), days=days)
    if panel is None or panel.empty or "fwd_ret_pivot" not in panel.columns:
        return {}
    fp = panel.copy()
    # Gate-4 subset (2026-08-14, matching the serve-time TRADEABLE rank pool):
    # curves must be fit on the same population the live ranks are computed
    # over. Fail-soft: if the liquidity features can't be attached, calibrate
    # on the full panel rather than not at all.
    if bool(getattr(settings, "rank_tradeable_only", True)):
        try:
            from src.analysis.predictability import build_feature_panel
            fq = build_feature_panel(fp)
            pr = pd.to_numeric(fq.get("price"), errors="coerce")
            dv = pd.to_numeric(fq.get("dollar_vol"), errors="coerce")   # $M
            keep = (pr >= float(settings.trade_min_price))                    & (dv >= float(settings.trade_min_dollar_volume) / 1e6)
            if int(keep.sum()) >= 1000:
                fp = fq[keep].copy()
        except Exception as e:
            logger.warning(f"[rank_shaping] Gate-4 subset unavailable ({e}) — "
                           f"calibrating on the full panel")
    fp["fwd_ret_pivot"] = pd.to_numeric(fp["fwd_ret_pivot"], errors="coerce")
    fp = fp[fp["fwd_ret_pivot"].notna()]
    if fp.empty:
        return {}
    fp["day"] = fp["signal_date"].astype(str).str[:10]
    lo, hi = np.percentile(fp["fwd_ret_pivot"], [1, 99])
    fp["retw"] = fp["fwd_ret_pivot"].clip(lo, hi)

    prior_n = max(0, int(getattr(settings, "rank_shape_prior_n", 200)))
    min_rows = max(1, int(getattr(settings, "rank_shape_min_rows", 500)))
    shapes: Dict[str, List[float]] = {}
    for m in FAMILY_OF:
        if m not in fp.columns:
            continue
        sc = pd.to_numeric(fp[m], errors="coerce")
        sub = fp[sc.notna() & (sc != 0)]
        if len(sub) < min_rows or sub["day"].nunique() < 10:
            continue
        pct = sub.groupby("day")[m].rank(pct=True)
        b = np.clip((pct * 10).astype(int), 0, 9)
        g = sub.groupby(b)["retw"].agg(["mean", "size"]).reindex(range(10))
        n_d = g["size"].fillna(0.0).to_numpy(float)
        m_d = g["mean"].to_numpy(float)
        ok = np.isfinite(m_d) & (n_d > 0)
        if ok.sum() < 6:
            continue
        # OLS line through the observed deciles (the method's own linear trend)
        x = _MIDS[ok]
        try:
            slope, intercept = np.polyfit(x, m_d[ok], 1)
        except Exception:
            continue
        line = slope * _MIDS + intercept
        # shrink observed deciles toward the line; unobserved deciles = line
        m_filled = np.where(ok, np.nan_to_num(m_d), line)
        shrunk = (n_d * m_filled + prior_n * line) / np.maximum(n_d + prior_n, 1.0)
        # demean (observation-weighted) — the shape is rank-CONDITIONAL payoff
        w = np.maximum(n_d, 1.0)
        shrunk = shrunk - float(np.average(shrunk, weights=w))
        amp = float(np.max(np.abs(shrunk)))
        if amp < _AMP_FLOOR:
            continue                      # flat curve -> keep the linear rank
        shapes[m] = [round(float(v), 4) for v in (shrunk / amp)]
    logger.info(f"[rank_shaping] calibrated {len(shapes)} shaped curve(s) "
                f"over {fp['day'].nunique()} settled days")
    return shapes


def get_rank_shapes() -> Dict[str, List[float]]:
    """TTL-cached calibration; {} when disabled or unavailable (fail-soft)."""
    if not bool(getattr(settings, "enable_rank_shaping", True)):
        return {}
    now = time.time()
    ttl = float(getattr(settings, "rank_shape_ttl_seconds", 21600))
    if _CACHE["shapes"] is not None and (now - _CACHE["ts"]) < ttl:
        return _CACHE["shapes"]
    try:
        shapes = calibrate_rank_shapes()
    except Exception as e:
        logger.warning(f"[rank_shaping] calibration failed (identity kept): {e}")
        shapes = {}
    _CACHE.update(ts=now, shapes=shapes)
    return shapes


def shape_score(method: str, pct: float,
                shapes: Optional[Dict[str, List[float]]] = None) -> float:
    """Map a within-run percentile ∈ [0,1] to the method's shaped score.
    No curve -> the plain centered rank ``2·pct − 1`` (identity)."""
    shapes = get_rank_shapes() if shapes is None else shapes
    curve = shapes.get(method)
    if not curve:
        return 2.0 * pct - 1.0
    return float(np.interp(pct, _MIDS, curve))


def reset_cache() -> None:
    """asof / test hook."""
    _CACHE.update(ts=0.0, shapes=None)


# ── Walk-forward SHAPE history (2026-08-21) ──────────────────────────────────
# `weight_history` walk-forwards the WEIGHTS, but the shaped curves — a fitted
# CONSUMPTION layer — rode at today's fit, so a tier-2 backtest read curves
# fitted on the very window it was scoring (measured 2026-08-20: that alone
# flips the current arch's pivot IC from −0.041 to +0.049). These helpers give
# curves their own as-of series, the exact sibling of walkforward.py: each row
# is `calibrate_rank_shapes()` run under `analysis_asof(as_of)`, so a curve can
# never see rows at or after its own date.

def materialize_shape_history(start: Optional[str] = None,
                              end: Optional[str] = None,
                              step_days: int = 5) -> int:
    """Append missing as-of shape rows every ``step_days`` calendar days from
    ``start`` (default: the panel's first day) through ``end`` (default:
    today). Idempotent — existing as_of dates are skipped, so the EOD call
    appends at most one step. Returns rows written."""
    import json as _json
    from datetime import date as _date, timedelta as _td

    from src.analysis.asof import analysis_asof
    from src.db import repo
    from src.db.connection import connect

    have = set()
    try:
        df = repo.fetch_df("SELECT DISTINCT as_of FROM shape_history")
        if df is not None and not df.empty:
            have = set(df["as_of"].astype(str))
    except Exception:
        pass
    if start is None:
        try:
            d0 = repo.fetch_df("SELECT min(substr(signal_date,1,10)) AS d FROM signals")
            start = str(d0["d"].iloc[0])
        except Exception:
            return 0
    end_d = _date.fromisoformat(str(end)[:10]) if end else _date.today()
    cur = _date.fromisoformat(str(start)[:10])
    stamp = None
    written = 0
    while cur <= end_d:
        as_of = cur.isoformat()
        if as_of not in have:
            try:
                with analysis_asof(as_of):
                    shapes = calibrate_rank_shapes()
            except Exception as e:
                logger.warning(f"[rank_shaping] shape history {as_of} failed: {e}")
                shapes = None
            if shapes is not None:
                if stamp is None:
                    from datetime import datetime, timezone
                    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                rows = ([(as_of, m, _json.dumps(c), stamp) for m, c in sorted(shapes.items())]
                        or [(as_of, "__none__", "[]", stamp)])   # "computed, no curves"
                with connect() as con:
                    con.executemany(
                        "INSERT INTO shape_history (as_of, method, curve, computed_at) "
                        "VALUES (?, ?, ?, ?)", rows)
                written += len(rows)
        cur += _td(days=max(1, int(step_days)))
    # Always ensure TODAY has a row so the EOD append keeps the series current
    # even when today falls off the stride grid.
    today = _date.today().isoformat()
    if today not in have and today != as_of and end is None:
        return written + materialize_shape_history(start=today, end=today, step_days=1)
    if written:
        logger.info(f"[rank_shaping] shape history: wrote {written} row(s)")
    return written


def load_shape_history():
    """The full as-of series as ``{as_of: {method: curve}}`` (sorted keys)."""
    import json as _json

    from src.db import repo
    try:
        df = repo.fetch_df("SELECT as_of, method, curve FROM shape_history")
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    out: dict = {}
    for r in df.itertuples(index=False):
        d = out.setdefault(str(r.as_of), {})
        if r.method != "__none__":
            try:
                d[str(r.method)] = list(_json.loads(r.curve))
            except Exception:
                continue
    return dict(sorted(out.items()))


def shapes_for_date(day: str, hist: Optional[dict] = None):
    """The curves in force ON ``day``: the latest as_of STRICTLY BEFORE it
    (a same-day calibration saw that day's rows under asof semantics only if
    strictly earlier — mirror `weights_for_date`'s convention). ``None`` when
    no earlier calibration exists — the caller must treat that as identity,
    never as today's curves."""
    hist = load_shape_history() if hist is None else hist
    if not hist:
        return None
    day = str(day)[:10]
    best = None
    for as_of in hist:                      # sorted
        if as_of < day:
            best = as_of
        else:
            break
    return dict(hist[best]) if best is not None else None


if __name__ == "__main__":       # pragma: no cover - operator CLI
    import argparse
    ap = argparse.ArgumentParser(description="Walk-forward shape history")
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--step-days", type=int, default=5)
    a = ap.parse_args()
    if a.materialize:
        print(f"wrote {materialize_shape_history(step_days=a.step_days)} row(s)")
    else:
        h = load_shape_history()
        print(f"shape_history: {len(h)} as-of date(s)")
        for k in list(h)[-5:]:
            print(f"  {k}: {len(h[k])} curve(s)")
