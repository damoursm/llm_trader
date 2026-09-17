"""The PIVOT label — the next H/L pivot on 30-MINUTE bars — and the leg-state features.

**The label (the only one, 2026-09-16).** For a row anchored at a tick
``(time, price)``, the signed % move from ``price`` to the EXTREME of the next
RESOLVED pivot on 30-minute regular-hours bars: swing highs on each bar's HIGH,
swing lows on its LOW (``pivot_label_basis="hl"``; ``"close"`` marks both on
the bar's close), a swing confirmed once price retraces ``pivot_min_move_pct``
from the running extreme (`_resolved_pivots`, the threshold zigzag). Eligible
bars START at or after the tick — the bar containing the tick is excluded, it
cannot be split — so the pivot may print later the SAME session, strictly after
the tick, or days out: the horizon is whatever the pivot takes. A row that
carries a DATE but no tick (a deep training row, a position-day, an exit-day
close) anchors at that session's 16:00 ET close with the session close as the
price, so the search starts at the next session's first bar. Whether a model's
FEATURES are daily or intraday does not matter to the label — it is the next
pivot in the subsequent bars either way (user, 2026-09-16), which is why
``ml_ohlcv`` keeps its daily features and trains on this label.

Point-in-time: a pivot exists only once its CONFIRMING bar is visible — a
truncated series never emits it, and `pivot_rows.pivot_fwd_row` applies the
same rule per row through the confirming index on one memoised scan. Training
uses SETTLED rows only; evaluation marks the unresolved tail at the LAST
VISIBLE CLOSE (never the running leg's extreme, which nothing guarantees was
capturable). ``pivot_basis()`` (``hl1@30m``) fingerprints marks + threshold +
resolution; an artifact stamped with another basis abstains at serving.

The daily H/L label (`next_pivot_targets` on daily bars, anchored at the
close, first candidate bar tomorrow) was DECOMMISSIONED on 2026-09-16: on daily
bars a same-session swing was unrepresentable, and the 2026-09-15 five-label
comparison ranked it apart from every 30-minute label
(`memory/pivot-label-verdict-2026-09.md`). Its history: the 2026-08-12 H/L +
threshold directive, `memory/pivot-horizon-target-2026-08.md`.

**The 9 LEG FEATURES** (``LEG_FEATURES``): where the stock sits within its own
swing on DAILY bars — the state the 76 generic features never encode, measured
pivot-SPECIFIC (they add IC on a pivot target and nothing on a static one).
They ride the same zigzag on the daily series with the same threshold; a pivot
is usable at bar *i* only once CONFIRMED (``conf <= i``) — the confirmation seam
`tests/test_pivot_target.py` probes. Features, not the label: they stay daily
while the label is intraday.

**The training label** (``within_day_rank``): the within-day centred percentile
rank of the signed target, trained with day-equal row weights (the objective
switch worth +26% IC over pooled-L1 on the raw target, 2026-08).

Data paths: daily bars from ``predictability._hlc_by_session`` (features);
30-minute bars from the tick cache (`_series_30m`, the newest 260 sessions —
panel/evaluation rows) or the deep store (`data/intraday_store`, 2021→ —
training rows).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MAX_PIVOT_DAYS = 60
BARS_PER_SESSION_30M = 13                     # 09:30–16:00 ET in 30-min bars
MAX_PIVOT_BARS_30M = MAX_PIVOT_DAYS * BARS_PER_SESSION_30M
MIN_BARS_30M = 50
_NY_TZ_NAME = "America/New_York"

LEG_FEATURES: List[str] = [
    "leg_dir", "leg_age", "leg_ret", "leg_age_ratio", "leg_ret_ratio",
    "swing_len_med10", "swing_amp_med10", "run_signed", "close_loc_1",
]


def _min_move_pct() -> float:
    """The confirmation threshold (%) of the zigzag — label and leg features
    alike. Lazy settings read, fail-soft to 1.0 (the 2026-08-12 directive's
    value: "not worth buying and selling weak price runs")."""
    try:
        from config.settings import settings
        return max(0.0, float(settings.pivot_min_move_pct))
    except Exception:
        return 1.0


def pivot_label_basis() -> str:
    """Where the 30-minute pivot marks live: ``"hl"`` (swing highs on the bar's
    HIGH, lows on its LOW — the user's choice, 2026-09-16) or ``"close"`` (both
    on the bar's CLOSE, the measured alternative of 2026-09-15). Setting
    ``pivot_label_basis``."""
    try:
        from config.settings import settings
        b = str(getattr(settings, "pivot_label_basis", "hl")).lower()
        return b if b in ("hl", "close") else "hl"
    except Exception:
        return "hl"


def pivot_basis() -> str:
    """The label fingerprint every trained artifact is stamped with —
    marks basis + confirmation threshold + bar resolution, e.g. ``hl1@30m``.
    Any change to any of the three is CATEGORICAL: an artifact stamped with
    another basis abstains at serving until retrained."""
    return f"{pivot_label_basis()}{_min_move_pct():g}@30m"


def _pivot_scan(c: np.ndarray, h: np.ndarray, lo: np.ndarray, thr_pct: Optional[float] = None):
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
    thr = (_min_move_pct() if thr_pct is None else max(0.0, float(thr_pct))) / 100.0
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


def _resolved_pivots(c: np.ndarray, h: np.ndarray, lo: np.ndarray, thr_pct: Optional[float] = None):
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
    return _pivot_scan(c, h, lo, thr_pct)[0]


# ── leg-state FEATURES on daily bars ─────────────────────────────────────────

def leg_feature_rows(c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                     dates: Sequence, only_last: bool = False) -> List[dict]:
    """Per-bar leg state on a DAILY series: one dict per bar with
    ``signal_date`` + the ``LEG_FEATURES`` that are defined at that bar.
    ``only_last`` emits just bar n-1 (serving). Features only — the label is
    `intraday_pivot_targets` on 30-minute bars, never this pass."""
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
    rows: List[dict] = []
    idxs = [n - 1] if only_last else range(n)
    for i in idxs:
        if c[i] <= 0:
            continue
        rec: dict = {"signal_date": dates[i].isoformat()}
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
    """``(session_dates, c, h, lo)`` from the DAILY cache — the leg features'
    series, and the session closes the deep training rows anchor at."""
    from src.analysis.predictability import _hlc_by_session
    hlc = _hlc_by_session(tk)
    if hlc is None:
        return None
    idx, high, low, close, _v = hlc
    c = np.asarray([float(x) for x in close.values], dtype=float)
    h = np.asarray([float(x) for x in high.values], dtype=float)
    lo = np.asarray([float(x) for x in low.values], dtype=float)
    return idx, c, h, lo


def latest_leg_features(tk: str) -> Optional[Dict[str, float]]:
    """The 9 leg features for the LAST completed daily bar — the serving-side
    call. Same pass as the training rows (a parity test pins this)."""
    s = _series(tk)
    if s is None:
        return None
    idx, c, h, lo = s
    rows = leg_feature_rows(c, h, lo, list(idx), only_last=True)
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


# ── the LABEL on 30-minute bars ───────────────────────────────────────────────

def to_naive_utc(ts) -> Optional[pd.Timestamp]:
    """Any timestamp / ISO string → naive UTC ``Timestamp`` (the 30-min cache's
    index convention). Naive input is taken as UTC. None on junk."""
    try:
        t = pd.Timestamp(ts)
        if t is pd.NaT:
            return None
        # pandas 2 parses junk like "t1" as year 0001 at second resolution; forcing
        # nanoseconds turns that into the OutOfBounds error we want to swallow
        # instead of an overflow deep inside a searchsorted.
        t = t.as_unit("ns")
    except Exception:
        return None
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def session_close_utc(day) -> pd.Timestamp:
    """16:00 ET of ``day`` as naive UTC — the anchor for a row that carries a
    date but no tick (deep training rows, position-days, exit-day closes).
    DST-correct."""
    d = pd.Timestamp(day).normalize()
    return (d + pd.Timedelta(hours=16)).tz_localize(_NY_TZ_NAME).tz_convert("UTC").tz_localize(None)


def _series_30m(tk: str):
    """``(index, c, h, lo)`` from the 30-minute TICK cache, index naive UTC.
    NaN / non-positive bars are dropped; None below ``MIN_BARS_30M``."""
    from src.data.cache import load_ohlcv
    df = load_ohlcv(tk, interval="30m")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    c = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float) if "High" in df.columns else c
    lo = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float) if "Low" in df.columns else c
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(lo) & (c > 0)
    if int(ok.sum()) < MIN_BARS_30M:
        return None
    return idx[ok], c[ok], h[ok], lo[ok]


def intraday_pivot_targets(idx, c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                           anchors, asof=None, basis: Optional[str] = None,
                           thr_pct: Optional[float] = None) -> List[Optional[dict]]:
    """Per-anchor next-pivot labels on a 30-minute series — THE label.

    ``basis`` — ``"hl"`` or ``"close"`` (default: ``pivot_label_basis()``); on
    the close basis the marks and the target are the bar's CLOSE.
    ``thr_pct`` — the confirmation threshold (default: ``pivot_min_move_pct``).
    ``anchors`` is an iterable of ``(tick_time, tick_price)``. For each: eligible
    bars start at or after the tick; the label is the % move from ``tick_price``
    to the first RESOLVED pivot's extreme on an eligible bar (within
    ``MAX_PIVOT_BARS_30M``), ``resolved=True``; otherwise PROVISIONAL — the move
    to the last visible close, ``resolved=False``; ``None`` when the tick is at
    or after the last visible bar or the price is unusable.

    Point-in-time: pass ``asof`` (any timestamp) and the series is TRUNCATED to
    bars starting before it before the scan, so a pivot counts only once its
    confirming bar is visible and "last close" is the last one that run saw.

    Returned dict: ``target_pct, resolved, end_ts (pivot bar start, or last bar),
    end_idx, bars_ahead, same_session, is_peak, confirm_ts``.
    """
    idx = pd.DatetimeIndex(idx)
    if asof is not None:
        cut = to_naive_utc(asof)
        if cut is not None:
            m = int(idx.searchsorted(cut, side="left"))
            idx, c, h, lo = idx[:m], c[:m], h[:m], lo[:m]
    if (basis or pivot_label_basis()) == "close":
        h, lo = c, c
    n = len(c)
    out: List[Optional[dict]] = []
    if n < MIN_BARS_30M:
        return [None for _ in anchors]
    P, PP, FL, CF = _resolved_pivots(c, h, lo, _min_move_pct() if thr_pct is None else thr_pct)
    idx_et_dates = idx.tz_localize("UTC").tz_convert(_NY_TZ_NAME).date
    last_close = float(c[-1])
    for tick_time, tick_price in anchors:
        t = to_naive_utc(tick_time)
        try:
            px = float(tick_price)
        except (TypeError, ValueError):
            px = float("nan")
        if t is None or not (px == px and px > 0):
            out.append(None)
            continue
        if t < idx[0]:                                  # before the series' first bar: the warm-up
            out.append(None)                            # would masquerade as the next pivot
            continue
        i0 = int(idx.searchsorted(t, side="left"))      # first bar starting AT/AFTER the tick
        if i0 >= n:
            out.append(None)
            continue
        k = int(np.searchsorted(P, i0, side="left")) if len(P) else 0
        t_day = t.tz_localize("UTC").tz_convert(_NY_TZ_NAME).date()
        if k < len(P) and (int(P[k]) - i0) <= MAX_PIVOT_BARS_30M:
            j = int(P[k])
            out.append(dict(target_pct=(float(PP[k]) / px - 1.0) * 100.0, resolved=True,
                            end_ts=idx[j], end_idx=j, bars_ahead=j - i0 + 1,
                            same_session=bool(idx_et_dates[j] == t_day), is_peak=bool(FL[k]),
                            confirm_ts=idx[int(CF[k])]))       # the bar the pivot became knowable
        else:
            out.append(dict(target_pct=(last_close / px - 1.0) * 100.0, resolved=False,
                            end_ts=idx[-1], end_idx=n - 1, bars_ahead=n - i0,
                            same_session=False, is_peak=None, confirm_ts=None))
    return out


def next_pivot_targets(tk: str, anchors, asof=None) -> List[Optional[dict]]:
    """Ticker-level convenience over the TICK cache + `intraday_pivot_targets`.
    ``[None, ...]`` when the ticker has no usable 30-minute history — callers
    must treat that as NO LABEL; there is no other basis to fall back to."""
    s = _series_30m(tk)
    if s is None:
        return [None for _ in anchors]
    idx, c, h, lo = s
    return intraday_pivot_targets(idx, c, h, lo, anchors, asof=asof)


def live_next_pivot(tk: str, anchor_time, anchor_price: float,
                    live_mark: Optional[float] = None) -> Optional[dict]:
    """The open-position view: the first resolved 30-minute pivot after the
    entry tick, else PROVISIONAL at the freshest close — the live mark when it
    is newer than the last cached bar. Monitoring only."""
    r = next_pivot_targets(tk, [(anchor_time, anchor_price)])[0]
    if r is None:
        return None
    if not r["resolved"] and live_mark is not None and live_mark == live_mark and live_mark > 0:
        r = dict(r, target_pct=(float(live_mark) / float(anchor_price) - 1.0) * 100.0, price=float(live_mark))
    else:
        r = dict(r, price=float(anchor_price) * (1.0 + r["target_pct"] / 100.0))
    return r


def session_close_labels(idx, c: np.ndarray, h: np.ndarray, lo: np.ndarray,
                         session_dates: Sequence, session_closes: np.ndarray):
    """The TRAINING label for date-only rows: per session date, the settled %
    move from that session's close (anchored at its 16:00 ET close, so the
    search starts at the next session's first bar) to the next pivot on the
    30-minute series. Returns ``(sp, end_dates)`` — ``sp`` NaN and ``end`` None
    where the pivot has not printed. Exactly what `pivot_rows.pivot_fwd_row`
    answers for a date-only row, on the same series (a parity test pins it)."""
    n = len(session_dates)
    sp = np.full(n, np.nan)
    end: List[Optional[str]] = [None] * n
    idx = pd.DatetimeIndex(idx)
    if len(idx) < MIN_BARS_30M:
        return sp, end
    et_dates = idx.tz_localize("UTC").tz_convert(_NY_TZ_NAME).date
    first_day = et_dates[0]
    # Only sessions the 30-minute series covers can carry a label (a session
    # before its first bar would be answered by the warm-up's first swing).
    keep = [i for i, d in enumerate(session_dates) if d >= first_day]
    anchors = [(session_close_utc(session_dates[i]), float(session_closes[i])) for i in keep]
    res = intraday_pivot_targets(idx, c, h, lo, anchors)
    for i, r in zip(keep, res):
        if r is not None and r["resolved"]:
            sp[i] = float(r["target_pct"])
            end[i] = et_dates[int(r["end_idx"])].isoformat()
    return sp, end


def pivot_label_frame(tickers: Sequence[str], deep: bool = True) -> pd.DataFrame:
    """One row per (ticker, session_date) with a SETTLED label — ``sp_buy`` /
    ``sp_end`` — plus the 9 daily leg features: the offline training frame
    (`ml_validate`). ``deep=True`` reads the deep 30-minute store (+ the tick
    cache's newer tail); ``False`` the tick cache only. Built per ticker from
    arrays and concatenated, so it stays lean on thousands of names."""
    from src.data.intraday_store import deep_series_30m
    frames: List[pd.DataFrame] = []
    for tk in tickers:
        s = _series(tk)
        if s is None:
            continue
        idx_d, c_d, h_d, lo_d = s
        s30 = deep_series_30m(tk) if deep else _series_30m(tk)
        if s30 is None:
            continue
        sp, end = session_close_labels(*s30, list(idx_d), c_d)
        legs = leg_feature_rows(c_d, h_d, lo_d, list(idx_d))
        if not legs:
            continue
        lf = pd.DataFrame(legs)
        lab = pd.DataFrame({"signal_date": [d.isoformat() for d in idx_d], "sp_buy": sp, "sp_end": end})
        f = lf.merge(lab, on="signal_date", how="inner")
        f = f[np.isfinite(f["sp_buy"].to_numpy(dtype=float))]
        if f.empty:
            continue
        f.insert(0, "ticker", tk)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["ticker", "signal_date", "sp_buy", "sp_end"] + LEG_FEATURES)
    return pd.concat(frames, ignore_index=True)
