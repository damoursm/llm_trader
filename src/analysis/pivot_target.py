"""Signed pivot target + leg-state features — the ``ml_ohlcv`` v2 basis.

Ported verbatim from the 2026-08 scratchpad harness that measured it (see
``memory/pivot-horizon-target-2026-08.md`` for the full experimental record).
The quantities:

**The SIGNED pivot target** (``sp_buy``): at day *i*, the % return to the NEXT
pivot — peak or trough, whichever comes first. One pivot, not two: the earlier
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
A pivot at bar *j* is CONFIRMED only once bar *j+1* prints, so every leg
feature at bar *i* uses pivots at *j ≤ i−1* — the one-bar confirmation lag is
the entire causality story, because sign(target) = sign(tomorrow's move) makes
any leak at this seam a perfect answer key. ``tests/test_pivot_target.py``
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
    is_peak = np.zeros(n, dtype=bool)
    is_trough = np.zeros(n, dtype=bool)
    is_peak[1:-1] = (d[:-1] >= 0) & (d[1:] < 0)
    is_trough[1:-1] = (d[:-1] <= 0) & (d[1:] > 0)

    # next pivot at/after i+1 (the target end), -1 when none has printed
    nxt = np.full(n, -1, dtype=int)
    j = -1
    for i in range(n - 1, -1, -1):
        nxt[i] = j
        if is_peak[i] or is_trough[i]:
            j = i

    P = np.where(is_peak | is_trough)[0]
    P_is_peak = is_peak[P]
    if len(P) >= 2:
        s_len = (P[1:] - P[:-1]).astype(float)
        s_amp = np.abs(c[P[1:]] / c[P[:-1]] - 1.0) * 100.0
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

    n_prior = np.searchsorted(P, np.arange(n), side="left")   # pivots strictly < i
    rows: List[dict] = []
    idxs = [n - 1] if only_last else range(n)
    for i in idxs:
        if c[i] <= 0:
            continue
        jn = nxt[i]
        settled = jn > 0 and (jn - i) <= MAX_PIVOT_DAYS
        if require_target and not settled:
            continue
        rec: dict = {"signal_date": dates[i].isoformat()}
        if settled:
            rec["sp_buy"] = (c[jn] / c[i] - 1.0) * 100.0
            rec["sp_end"] = dates[jn].isoformat()
        pos = n_prior[i] - 1                    # last pivot < i (confirmed by bar i)
        if pos >= 0:
            jp = P[pos]
            rec["leg_dir"] = -1.0 if P_is_peak[pos] else 1.0
            rec["leg_age"] = float(i - jp)
            rec["leg_ret"] = (c[i] / c[jp] - 1.0) * 100.0
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
