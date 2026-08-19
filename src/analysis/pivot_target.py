"""Signed pivot target + leg-state features — the ``ml_ohlcv`` v2 basis.

Ported verbatim from the 2026-08 scratchpad harness that measured it (see
``memory/pivot-horizon-target-2026-08.md`` for the full experimental record).
The quantities:

**The SIGNED pivot target** (``sp_buy``): at day *i*, the % return from
close(*i*) to the NEXT pivot's EXTREME — the peak bar's HIGH or the trough
bar's LOW, whichever pivot comes first (H/L basis + the
``pivot_min_move_pct`` threshold zigzag since 2026-08-12, user directives; the
construction previously ran zero-threshold on closes — those changes are why
``pivot_basis()`` exists and why the ml_ohlcv artifact is basis-stamped). One pivot, not two: the earlier
two-pivot variant (buy→next peak, sell→next trough) made both sides ~75%
positive by construction and turned the model into a volatility ranker; the
signed redefinition drops the base rate to ~50.4% and kills that artifact
(``atr_pct_14`` alone: ICIR +0.959 on the two-pivot target, −0.123 on this one).
``sell = -buy`` exactly, so ONE model serves both sides. Median horizon 1
trading day, mean ~2, p90 4. Rows whose next pivot is further than
``MAX_PIVOT_DAYS`` out, or not yet printed, are excluded from training.

**The 9 LEG FEATURES** (``LEG_FEATURES``): where the stock sits within its own
swing — the state the 76 generic features never encode, and measured to be
pivot-SPECIFIC (they add +0.0011 IC here and NOTHING on a static 5d target).
A pivot is CONFIRMED only once its ``pivot_min_move_pct`` reversal has
PRINTED (variable lag — `_resolved_pivots` returns each pivot's confirmation
bar), so every leg feature at bar *i* uses pivots with ``conf ≤ i`` — that
confirmation seam is the entire causality story, because any leak across it
hands the model a still-revisable extreme as if it were settled. ``tests/test_pivot_target.py``
probes exactly that seam.

**The training label** (``within_day_rank``): the within-day centred percentile
rank of ``sp_buy``, trained with day-equal row weights under L2 — the objective
switch worth +26% IC over pooled-L1 on the raw target (the eval metric is
per-day Spearman, so train on the statistic being scored). The 2026-08-08
weight-scheme experiment confirmed uniform day-equal weights beat every
magnitude/recency/signed alternative, and that weighting by the SIGNED target
is structurally unsound here (the weight becomes a monotone function of the
label).

Data path: ``predictability._hlc_by_session`` → the daily OHLCV cache —
completed bars only, identical to ``ml_dataset.ticker_feature_frame``, so
dataset rows and live serving read the same series by construction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MAX_PIVOT_DAYS = 60

LEG_FEATURES: List[str] = [
    "leg_dir", "leg_age", "leg_ret", "leg_age_ratio", "leg_ret_ratio",
    "swing_len_med10", "swing_amp_med10", "run_signed", "close_loc_1",
]


def _min_move_pct() -> float:
    """The minimum swing threshold (%). Lazy settings read, fail-soft to 1.0 —
    the value the 2026-08-12 directive named ("not worth buying and selling
    weak price runs" against the spread)."""
    try:
        from config.settings import settings
        return max(0.0, float(settings.pivot_min_move_pct))
    except Exception:
        return 1.0


def pivot_basis() -> str:
    """The pivot-definition fingerprint the ml_ohlcv artifact is stamped with:
    ``hl`` (highs/lows, extremes) + the confirmation threshold. Any change to
    either is a CATEGORICAL change — a stale-basis artifact must abstain, so
    the threshold is part of the string rather than a silent setting."""
    return f"hl{_min_move_pct():g}"


def _pivot_scan(c: np.ndarray, h: np.ndarray, lo: np.ndarray):
    """The zigzag pass underlying ``_resolved_pivots``, additionally returning
    the machine's terminal state: ``(resolved, pending)`` where ``resolved`` is
    the usual ``(indices, prices, is_peak_flags, confirm_indices)`` and
    ``pending`` is the still-UNCONFIRMED running candidate
    ``(ext_idx, ext_price, is_peak)`` — the extreme the NEXT pivot will be, if
    and when its ``pivot_min_move_pct`` reversal prints, still extending with
    every new extreme until then. ``None`` before the first pivot confirms
    (direction unknown). The loop is the single implementation; the pending
    state is exactly what the confirmation seam withholds from the resolved
    sequence, so it must only ever feed PROVISIONAL surfaces (live trade
    monitoring), never a settled label."""
    n = len(c)
    empty = (np.asarray([], dtype=int), np.asarray([], dtype=float),
             np.asarray([], dtype=bool), np.asarray([], dtype=int))
    if n == 0:
        return empty, None
    thr = _min_move_pct() / 100.0
    idxs: List[int] = []
    prices: List[float] = []
    flags: List[bool] = []
    confs: List[int] = []

    up: Optional[bool] = None
    hi_i, hi_px = 0, float(h[0])
    lo_i, lo_px = 0, float(lo[0])
    ext_i, ext_px = 0, float(h[0])

    for i in range(1, n):
        if up is None:
            # Direction unknown: triggers test against the extremes as of the
            # PRIOR bar; only afterwards does bar i update the running extremes.
            down_trig = hi_px > 0 and lo[i] <= hi_px * (1.0 - thr)
            up_trig = lo_px > 0 and h[i] >= lo_px * (1.0 + thr)
            if down_trig and up_trig:
                # one bar resolves both ways — take the larger relative move
                down_mag = (hi_px - lo[i]) / hi_px
                up_mag = (h[i] - lo_px) / lo_px
                if down_mag >= up_mag:
                    up_trig = False
                else:
                    down_trig = False
            if down_trig:
                idxs.append(hi_i); prices.append(hi_px); flags.append(True); confs.append(i)
                up = False
                ext_i, ext_px = i, float(lo[i])
            elif up_trig:
                idxs.append(lo_i); prices.append(lo_px); flags.append(False); confs.append(i)
                up = True
                ext_i, ext_px = i, float(h[i])
            else:
                if h[i] > hi_px:
                    hi_i, hi_px = i, float(h[i])
                if lo[i] < lo_px:
                    lo_i, lo_px = i, float(lo[i])
        elif up:
            if h[i] > ext_px:
                ext_i, ext_px = i, float(h[i])          # extend — no reversal test this bar
            elif ext_px > 0 and lo[i] <= ext_px * (1.0 - thr):
                idxs.append(ext_i); prices.append(ext_px); flags.append(True); confs.append(i)
                up = False
                ext_i, ext_px = i, float(lo[i])
        else:
            if lo[i] < ext_px:
                ext_i, ext_px = i, float(lo[i])         # extend — no reversal test this bar
            elif ext_px > 0 and h[i] >= ext_px * (1.0 + thr):
                idxs.append(ext_i); prices.append(ext_px); flags.append(False); confs.append(i)
                up = True
                ext_i, ext_px = i, float(h[i])

    pending = None if up is None else (int(ext_i), float(ext_px), bool(up))
    return (np.asarray(idxs, dtype=int), np.asarray(prices, dtype=float),
            np.asarray(flags, dtype=bool), np.asarray(confs, dtype=int)), pending


def _resolved_pivots(c: np.ndarray, h: np.ndarray, lo: np.ndarray):
    """The single causal pivot SEQUENCE as a THRESHOLD zigzag on highs/lows:
    ``(indices, prices, is_peak_flags, confirm_indices)``.

    A swing high is the running maximum of the HIGH series; it becomes a PEAK
    pivot only once some later bar's LOW prints ``pivot_min_move_pct`` percent
    below it (the mirror on lows for troughs) — so every leg moved at least
    the threshold and sub-threshold wiggles never become pivots (2026-08-12
    user directive: weak runs aren't worth the spread). ``confirm_indices[k]``
    is the bar at which pivot ``k`` became knowable — the VARIABLE-lag
    confirmation seam that replaced the fixed one-bar lag: leg features at bar
    *i* may only use pivots with ``conf <= i``, and a truncated series simply
    never emits an unconfirmed pivot, which is the whole point-in-time story.

    Per bar the machine does exactly ONE of extend or reverse, and the
    reversal always tests against the extreme as of the PRIOR bar: a bar whose
    own range spans the threshold must not confirm a reversal against the
    extreme it itself just set (with typical daily ranges above 1% that
    self-trigger degenerates into a pivot on almost every bar — caught by the
    monotone-series probe). Extension wins an outside bar (continuation-
    favoring, deterministic; intra-bar order is unknowable from daily data),
    so its reversal, if real, confirms on a later bar against the updated
    extreme. Ties keep the FIRST extreme bar. Implemented by ``_pivot_scan``;
    this wrapper discards the pending (unconfirmed) state."""
    return _pivot_scan(c, h, lo)[0]


def live_next_pivot(c: np.ndarray, h: np.ndarray, lo: np.ndarray, i0: int) -> Optional[dict]:
    """The LIVE view of "the first pivot after bar ``i0``" — settled when its
    confirming reversal has printed, PROVISIONAL until then.

    Returns ``{"price", "idx", "is_peak", "resolved", "confirm_idx"}`` or
    ``None`` (short window / no anchor / direction never established).

    Semantics, in resolution order:
    - If a CONFIRMED pivot with index > ``i0`` exists, that is the answer
      (``resolved=True``) — identical pivot sequence as ``next_pivot_targets``,
      but WITHOUT the ``MAX_PIVOT_DAYS`` training cap: a monitoring surface
      wants the swing's actual end, the cap only excludes stale rows from
      training.
    - Otherwise the answer is the machine's PENDING candidate — the running
      leg extreme, which by construction keeps extending through sub-threshold
      wiggles and freezes only when the ``pivot_min_move_pct`` reversal prints
      (``resolved=False``, ``confirm_idx=None``).
    - If the pending candidate's own bar is ≤ ``i0`` (the anchor sits past the
      leg's extreme-so-far), the first pivot after ``i0`` belongs to a LATER
      leg. We take the as-if-the-current-leg-stands view: alternate forward —
      the opposite-side running extreme strictly after the candidate — until
      the index clears ``i0``. Genuinely ambiguous (a new leg extreme would
      re-route it), which is exactly what provisional means; ties keep the
      first bar, matching the machine.
    - If NO completed bar after the anchor has printed yet (an entry on/after
      the newest session), the freshest running candidate is returned as a
      LEG-CONTINUATION SEED (``seed=True``, its bar at/before the anchor) —
      the weakest provisional state, superseded the moment a completed bar
      lands past the anchor. All other returns carry ``seed=False``.

    PROVISIONAL values must never feed a settled-label surface (panel labels,
    calibrations, training sets) — they are biased small early in a leg and
    revisable by construction. Trade monitoring / open-position evaluation
    only."""
    n = len(c)
    if n < 50 or not (0 <= int(i0) < n):
        return None
    i0 = int(i0)
    (P, PP, FL, CF), pending = _pivot_scan(c, h, lo)
    k = int(np.searchsorted(P, i0, side="right"))
    if k < len(P):
        return {"price": float(PP[k]), "idx": int(P[k]), "is_peak": bool(FL[k]),
                "resolved": True, "confirm_idx": int(CF[k]), "seed": False}
    if pending is None:
        return None
    ext_i, ext_px, is_peak = pending
    while ext_i <= i0:
        seg0 = ext_i + 1
        if seg0 >= n:
            return {"price": float(ext_px), "idx": int(ext_i), "is_peak": bool(is_peak),
                    "resolved": False, "confirm_idx": None, "seed": True}
        if is_peak:
            j = seg0 + int(np.argmin(lo[seg0:]))
            ext_i, ext_px, is_peak = j, float(lo[j]), False
        else:
            j = seg0 + int(np.argmax(h[seg0:]))
            ext_i, ext_px, is_peak = j, float(h[j]), True
    return {"price": float(ext_px), "idx": int(ext_i), "is_peak": bool(is_peak),
            "resolved": False, "confirm_idx": None, "seed": False}


def next_pivot_targets(c: np.ndarray, h: np.ndarray, lo: np.ndarray):
    """Evaluation-side labels: ``(sp, end_idx)`` per bar — ``sp[i]`` = the
    signed % return from close(i) to the next resolved pivot's EXTREME (the
    peak bar's high / the trough bar's low; NaN unsettled), ``end_idx[i]`` =
    that pivot's bar index (−1 unsettled). EXACTLY the training target: same
    resolved sequence (`_resolved_pivots`), same settle rule (next pivot
    printed AND within ``MAX_PIVOT_DAYS``), same 50-bar minimum — a parity
    test pins ``sp`` against ``pivot_frame``'s ``sp_buy``. The caller owns
    point-in-time discipline by truncating all three series at its visible-
    history cutoff: a pivot then settles only if its CONFIRMING bar is inside
    the window."""
    n = len(c)
    sp = np.full(n, np.nan)
    end = np.full(n, -1, dtype=int)
    if n < 50:
        return sp, end
    P, PP, _fl, _cf = _resolved_pivots(c, h, lo)
    if len(P) == 0:
        return sp, end
    pos = np.searchsorted(P, np.arange(n), side="right")   # first pivot > i
    for i in range(n):
        k = pos[i]
        if k >= len(P):
            continue
        j = int(P[k])
        if j > 0 and (j - i) <= MAX_PIVOT_DAYS and c[i] > 0:
            sp[i] = (PP[k] / c[i] - 1.0) * 100.0
            end[i] = j
    return sp, end


def _leg_target_rows(c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                     dates: Sequence, only_last: bool = False,
                     require_target: bool = True) -> List[dict]:
    """The shared causal pass: per-bar leg state + (optionally) the signed target.

    ``only_last`` emits just bar n-1 (serving); ``require_target=False`` keeps
    rows whose next pivot has not printed (serving again — today's bar almost
    never has a settled target). Training uses the defaults.
    """
    n = len(c)
    if n < 50:
        return []
    d = np.diff(c)
    P, PP, P_is_peak, P_conf = _resolved_pivots(c, h, lo)

    if len(P) >= 2:
        s_len = (P[1:] - P[:-1]).astype(float)
        # Swing amplitude between the RESOLVED EXTREMES (H/L basis) — the real
        # size of each completed swing, not its close-to-close shadow.
        s_amp = np.abs(PP[1:] / PP[:-1] - 1.0) * 100.0
        med_len = pd.Series(s_len).rolling(10, min_periods=3).median().to_numpy()
        med_amp = pd.Series(s_amp).rolling(10, min_periods=3).median().to_numpy()
    else:
        med_len = med_amp = np.array([])

    run = np.zeros(n)
    for i in range(1, n):
        if d[i - 1] > 0:
            run[i] = run[i - 1] + 1 if run[i - 1] > 0 else 1
        elif d[i - 1] < 0:
            run[i] = run[i - 1] - 1 if run[i - 1] < 0 else -1
        else:
            run[i] = 0

    # A pivot is USABLE at bar i only once CONFIRMED (its threshold reversal
    # printed): the variable-lag seam. `n_prior[i]` counts pivots with
    # conf <= i — NOT extreme-index < i, which would leak pre-confirmation
    # knowledge of a still-revisable extreme.
    n_prior = np.searchsorted(P_conf, np.arange(n), side="right")
    n_next = np.searchsorted(P, np.arange(n), side="right")   # first pivot > i
    rows: List[dict] = []
    idxs = [n - 1] if only_last else range(n)
    for i in idxs:
        if c[i] <= 0:
            continue
        k_next = n_next[i]
        jn = int(P[k_next]) if k_next < len(P) else -1
        settled = jn > 0 and (jn - i) <= MAX_PIVOT_DAYS
        if require_target and not settled:
            continue
        rec: dict = {"signal_date": dates[i].isoformat()}
        if settled:
            rec["sp_buy"] = (PP[k_next] / c[i] - 1.0) * 100.0
            rec["sp_end"] = dates[jn].isoformat()
        pos = n_prior[i] - 1                    # last pivot < i (confirmed by bar i)
        if pos >= 0:
            jp = int(P[pos])
            rec["leg_dir"] = -1.0 if P_is_peak[pos] else 1.0
            rec["leg_age"] = float(i - jp)
            # Position in the current leg vs the pivot's EXTREME (H/L basis).
            rec["leg_ret"] = (c[i] / PP[pos] - 1.0) * 100.0
            k = pos                              # swings 1..pos are complete
            if k >= 1 and len(med_len) >= k and np.isfinite(med_len[k - 1]):
                ml, ma = med_len[k - 1], med_amp[k - 1]
                rec["swing_len_med10"] = ml
                rec["swing_amp_med10"] = ma
                rec["leg_age_ratio"] = (i - jp) / ml if ml > 0 else np.nan
                rec["leg_ret_ratio"] = abs(rec["leg_ret"]) / ma if ma > 0 else np.nan
        rec["run_signed"] = run[i]
        rng = h[i] - lo[i]
        rec["close_loc_1"] = (c[i] - lo[i]) / rng if rng > 0 else np.nan
        rows.append(rec)
    return rows


def _series(tk: str):
    from src.analysis.predictability import _hlc_by_session
    hlc = _hlc_by_session(tk)
    if hlc is None:
        return None
    idx, high, low, close, _v = hlc
    c = np.asarray([float(x) for x in close.values], dtype=float)
    h = np.asarray([float(x) for x in high.values], dtype=float)
    lo = np.asarray([float(x) for x in low.values], dtype=float)
    return idx, c, h, lo


def pivot_frame(tickers: Sequence[str]) -> pd.DataFrame:
    """Training frame: one row per (ticker, date) with a SETTLED target —
    ``sp_buy``/``sp_end`` + the 9 leg features. Merge onto the deep feature set
    on (ticker, signal_date)."""
    rows: List[dict] = []
    for tk in tickers:
        s = _series(tk)
        if s is None:
            continue
        idx, c, h, lo = s
        for rec in _leg_target_rows(c, h, lo, list(idx)):
            rec["ticker"] = tk
            rows.append(rec)
    return pd.DataFrame(rows)


def latest_leg_features(tk: str) -> Optional[Dict[str, float]]:
    """The 9 leg features for the LAST completed bar — the serving-side call.
    Same pass as ``pivot_frame`` (a parity test pins this), target not required."""
    s = _series(tk)
    if s is None:
        return None
    idx, c, h, lo = s
    rows = _leg_target_rows(c, h, lo, list(idx), only_last=True, require_target=False)
    if not rows:
        return None
    return {f: float(rows[0][f]) for f in LEG_FEATURES if f in rows[0]
            and rows[0][f] == rows[0][f]}


def within_day_rank(y: np.ndarray, day: np.ndarray) -> np.ndarray:
    """Percentile rank of y within each day, centred to [-0.5, 0.5] — the
    training label. Computed over whatever rows are passed in, so the caller's
    point-in-time gate (pivot end < cutoff) is inherited rather than re-derived."""
    out = np.empty_like(y, dtype=np.float32)
    order = np.argsort(day, kind="stable")
    ds = day[order]
    bounds = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        r = np.argsort(np.argsort(y[idx])).astype(np.float32)
        out[idx] = (r / max(1, len(idx) - 1)) - 0.5
    return out
