"""Calibrated exit-confidence floor — learned from the hold-review trajectory.

The exit gate closes a position when its opener's same-direction re-affirmation
confidence falls below ``max(absolute floor, relative × entry confidence)``.
The ABSOLUTE floor was a frozen constant (0.45) — but the system records, every
tick, exactly the data that decides where that floor should sit: each
``trade_reviews`` row pairs a re-affirmation confidence with what the position
did next. If positions re-affirmed at 0.55 keep making money, 0.45 is too
tight; if 0.65-confidence holds go on to bleed, it's too loose.

Calibration: same-direction (re-affirming) reviews are joined to the position's
DIRECTION-ORIENTED next-day return (the same forward engine as the exit panel).
Scanning candidate floors, we pick the lowest confidence level where holding is
still profitable — i.e. the mean oriented forward return of reviews ABOVE the
floor is positive and the ones BELOW it are not adding money. The observed
floor is then Bayesian-shrunk toward the static prior while evidence is thin
(``shrink``, prior_n = ``exit_floor_prior_n``) and clamped to
``[exit_floor_min, exit_floor_max]`` — the floor drifts with evidence, never
jumps, and never leaves the sane band. Fail-soft: any data problem returns the
static setting. Deterministic given the DB; cached (the forward join is heavy).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from loguru import logger

from config.settings import settings
from src.performance.calibration import report_calibration, shrink

_CACHE_TTL_S = 1800.0
_cache: dict = {"ts": 0.0, "value": None}

# Candidate floors scanned (confidence units). The observed floor is the lowest
# candidate whose below-floor population LOSES money while the above-floor one
# makes it — the empirical boundary between "conviction still worth holding"
# and "conviction that precedes bleed".
_GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]      # 0.30 … 0.70


def _review_outcomes(days: Optional[int]) -> List[Tuple[float, float]]:
    """(reaffirm confidence, oriented next-day forward return %) pairs from
    trade_reviews — same-direction reviews only (the floor only ever tests
    re-affirmations; flips close unconditionally).

    DEDUPED to the LAST review per (position, day): a held position is
    re-judged every ~30-min tick, so the raw rows are heavily autocorrelated
    (near-identical confidence, IDENTICAL next-day return) and counting them
    all would inflate the evidence weight ~30× and over-confidently move the
    floor. One decision point per position-day matches the house convention
    (exit panel / periodic IC stats)."""
    from src.analysis.exit_panel import _load_trade_reviews, _daily_series, _fwd_daily
    from src.utils import ET
    import pandas as pd

    df = _load_trade_reviews(days)
    if df is None or getattr(df, "empty", True):
        return []
    df = df.copy()
    # Last review per (position_id, day). position_id falls back to ticker on
    # legacy rows; both are stable within a held position.
    pid = df["position_id"] if "position_id" in df.columns else df["ticker"]
    df["grp_pid"] = pid.fillna(df["ticker"])
    ts = pd.to_datetime(df["reviewed_at"], errors="coerce", utc=True)
    df["grp_day"] = ts.dt.tz_convert("America/New_York").dt.date
    df = (df.dropna(subset=["grp_day"]).sort_values("reviewed_at")
            .groupby(["grp_pid", "grp_day"], as_index=False).tail(1))

    out: List[Tuple[float, float]] = []
    daily_cache: dict = {}
    for r in df.itertuples(index=False):
        ea = str(getattr(r, "entry_action", "") or "").upper()
        act = str(getattr(r, "action", "") or "").upper()
        conf = float(getattr(r, "confidence", 0.0) or 0.0)
        if ea not in ("BUY", "SELL") or act != ea or conf <= 0:
            continue                                   # re-affirmations only
        tk = getattr(r, "ticker")
        sigd = getattr(r, "grp_day")
        if tk not in daily_cache:
            daily_cache[tk] = _daily_series(tk)
        dates, closes = daily_cache[tk]
        fwd = _fwd_daily(dates, closes, sigd, 1)
        if fwd is None:
            continue
        out.append((conf, fwd if ea == "BUY" else -fwd))
    return out


def _observed_floor(pairs: List[Tuple[float, float]], min_side: int) -> Optional[float]:
    """The lowest grid floor where the below-floor reviews lose money and the
    above-floor ones make it — None when no candidate separates the outcomes
    with at least ``min_side`` observations on each side."""
    for cand in _GRID:
        below = [f for c, f in pairs if c < cand]
        above = [f for c, f in pairs if c >= cand]
        if len(below) < min_side or len(above) < min_side:
            continue
        if sum(below) / len(below) <= 0.0 < sum(above) / len(above):
            return cand
    return None


def calibrated_exit_floor() -> float:
    """The absolute exit-confidence floor in force: the static setting, pulled
    toward the empirically observed boundary as review evidence accrues."""
    static = float(settings.signal_decay_confidence_floor)
    if not settings.exit_floor_calibration_enabled:
        return static
    now = time.time()
    if _cache["value"] is not None and (now - _cache["ts"]) < _CACHE_TTL_S:
        return _cache["value"]
    value = static
    try:
        pairs = _review_outcomes(int(settings.exit_floor_calibration_days))
        n = len(pairs)
        obs = _observed_floor(pairs, int(settings.exit_floor_min_side)) if n else None
        post = shrink(static, int(settings.exit_floor_prior_n), obs,
                      n if obs is not None else 0)
        value = min(float(settings.exit_floor_max),
                    max(float(settings.exit_floor_min), post))
        report_calibration(
            "exit_confidence_floor", value=value, prior=static,
            n_evidence=n if obs is not None else 0, unit="confidence",
            note=("observed hold/bleed boundary "
                  + (f"{obs:.2f}" if obs is not None else "none separable")
                  + f" over {n} re-affirming review(s)"))
    except Exception as e:
        logger.debug(f"[exit_floor] calibration skipped ({e}) — static floor")
    _cache.update(ts=now, value=value)
    return value


def reset_cache() -> None:
    """Tests."""
    _cache.update(ts=0.0, value=None)
