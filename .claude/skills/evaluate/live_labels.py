"""Panel-wide LIVE H/L pivot labels — settled where the pivot has confirmed,
PROVISIONAL (running leg extreme, extended by the last price) otherwise.

`pivot_target.live_next_pivot` answers one anchor at a time; this vectorises it
over every bar from a SINGLE `_pivot_scan`, and supports as-of truncation so the
proxy can be backtested ("what would we have known at date T?").

Convention matches the PANEL label (`next_pivot_targets`): anchor = the signal
date's own bar, pivots strictly AFTER it, target = % from close(anchor) to the
pivot's extreme, MARKET-SIGNED. For a POSITION multiply by the direction sign
(+1 long / −1 short) to get the oriented "is this still running my way" value.

Validated 2026-08-18 as an evaluation proxy: within-day rank correlation with
the settled label 0.963–1.000, sign agreement ~100%.

**Provisional values are for EVALUATION AND MONITORING ONLY** — never let them
feed a calibration, a training label, or any surface that decides live trades.
They keep extending until the confirming bar prints, so fitting on them is
fitting a moving target.

Usage:
    import sys; sys.path.insert(0, ".claude/skills/evaluate")
    from live_labels import label_frame
    lab = label_frame(sorted(df["ticker"].unique()))          # current prices
    df["y_mkt"] = [lab.get((t, d), (float("nan"), False))[0]
                   for t, d in zip(df["ticker"], df["signal_date"])]
    df["settled"] = [lab.get((t, d), (float("nan"), False))[1]
                     for t, d in zip(df["ticker"], df["signal_date"])]
"""
from __future__ import annotations

import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np

# Resolve the repo root whether this is imported from the repo or the skill dir.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.pivot_target import _pivot_scan, MAX_PIVOT_DAYS  # noqa: E402


def live_targets(c, h, lo, cap_days: int = MAX_PIVOT_DAYS):
    """Per-bar ``(target_pct, resolved)`` using every bar in the arrays passed.

    ``resolved[i]`` = the next pivot after i has CONFIRMED (the settled label,
    identical to `next_pivot_targets` modulo the MAX_PIVOT_DAYS cap, which is
    applied only to settled rows to match the panel). Otherwise the value is
    PROVISIONAL: the running leg extreme so far (extending, sign already fixed
    by the leg's direction) — i.e. the LAST PRICE stands in as the resolution.
    NaN when nothing is knowable.
    """
    n = len(c)
    tgt = np.full(n, np.nan)
    res = np.zeros(n, dtype=bool)
    if n < 50:
        return tgt, res
    (P, PP, FL, CF), pending = _pivot_scan(c, h, lo)
    Pl = list(P)
    for i in range(n):
        if not c[i] > 0:
            continue
        k = bisect_right(Pl, i)
        if k < len(Pl):
            j = int(P[k])
            if (j - i) <= cap_days:
                tgt[i] = (PP[k] / c[i] - 1.0) * 100.0
                res[i] = True
            continue
        if pending is None:
            continue
        ext_i, ext_px, is_peak = pending
        while ext_i <= i:
            seg0 = ext_i + 1
            if seg0 >= n:
                ext_i = -1
                break
            if is_peak:
                j2 = seg0 + int(np.argmin(lo[seg0:]))
                ext_i, ext_px, is_peak = j2, float(lo[j2]), False
            else:
                j2 = seg0 + int(np.argmax(h[seg0:]))
                ext_i, ext_px, is_peak = j2, float(h[j2]), True
        if ext_i < 0:
            continue
        tgt[i] = (ext_px / c[i] - 1.0) * 100.0
        res[i] = False
    return tgt, res


def label_frame(tickers, asof: str | None = None):
    """``{(ticker, 'YYYY-MM-DD'): (target_pct, resolved)}`` for every bar, with
    the series TRUNCATED at ``asof`` (exclusive of later bars) when given — the
    point-in-time view a run at ``asof`` would have had."""
    from src.analysis.pivot_target import _series
    out = {}
    for tk in tickers:
        s = _series(tk)
        if s is None:
            continue
        idx, c, h, lo = s
        if asof:
            keep = [k for k, d in enumerate(idx) if d.isoformat() <= asof]
            if len(keep) < 50:
                continue
            m = keep[-1] + 1
            idx, c, h, lo = idx[:m], c[:m], h[:m], lo[:m]
        tgt, res = live_targets(c, h, lo)
        for k, d in enumerate(idx):
            if tgt[k] == tgt[k]:
                out[(tk, d.isoformat())] = (float(tgt[k]), bool(res[k]))
    return out
