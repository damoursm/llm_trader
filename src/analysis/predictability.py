"""Predictability-feature IC panel — which per-stock features make OUR direction
call more forecastable (Tier 0 of the "find easier-to-predict swing names" work).

The discovery gate filters for *tradeability* (valid security, liquid, priced).
Nothing filters for *predictability* — whether a name's direction is forecastable
at a swing horizon by the signals we actually compute. Predictability is not a
property of the stock alone; it is a property of (stock × our signal set ×
horizon), so the honest way to find it is to MEASURE it, not assert it.

This module does exactly that. For every row of the signals panel it computes a
few cheap per-stock features from the OHLCV cache **as of the signal date** (no
look-ahead — the feature window ends at the same bar ``build_panel`` anchors the
forward return on), then buckets the panel by each feature and reports, per
horizon, how well the aggregate ``combined_score`` predicted the forward return
INSIDE each bucket (Spearman IC, directional hit rate, and the signed
solo-return). A feature is a useful predictability filter iff the metrics
separate across its buckets — e.g. hit rate rises Low→High for trend efficiency,
or is hump-shaped for volatility (mid predictable, extremes noisy).

Crucially this uses the WHOLE panel — features are OHLCV-derived, independent of
the ``universe_source`` stamp — so it produces signal from the existing history
immediately (unlike the forward-collected source table).

Features (all cheap, causal, cache-only):
* ``eff_ratio``    — Kaufman efficiency ratio (|net move| ÷ path length, 20d):
                     clean directional move vs chop. Higher ⇒ expect more skill.
* ``adx``          — Wilder ADX (14d): trend strength. Higher ⇒ established trend.
* ``realized_vol`` — stdev of daily % returns (20d): expect a HUMP (moderate vol
                     predictable, too-low = no swing, too-high = noise).
* ``breadth``      — signal methods agreeing (from the panel, not OHLCV): the
                     known strongest entry discriminator (Spearman +0.48) — the
                     built-in sanity check that the machinery measures correctly.

Usage:  python -m src.analysis.predictability [--horizons 1,5,10] [--days 90]
                                              [--buckets 3] [--min-n 30]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.signal_panel import _spearman, build_panel

_EPS = 1e-12
# ADX value that normalises a signed-ADX signal to ~±1 (rank-preserving for IC —
# only the magnitude scale, never the sign, is affected).
_ADX_NORM = 40.0

# The signed directional trend signals (Kaufman efficiency ratio + ADX·DMI),
# evaluated for STOCK DISCOVERY in their own dedicated dashboard table with the
# IC / win / return split by side (buys vs sells) — NOT mixed into the method IC
# table. Computed on the fly from OHLCV (never persisted).
FEATURE_SIGNAL_COLUMNS = ("er_signed", "adx_signed")
FEATURE_SIGNAL_LABELS = {
    "er_signed":  "Kaufman efficiency (signed trend)",
    "adx_signed": "ADX · DMI (signed trend)",
}

# (column, display label, hypothesis note). ``breadth`` is sourced from the
# panel's n_methods_agreeing, ``price`` from the panel's snapshot price; the rest
# are computed from OHLCV.
FEATURES: List[Tuple[str, str, str]] = [
    ("eff_ratio",    "Trend efficiency (Kaufman ER, 20d)", "higher = cleaner trend → expect higher hit/IC"),
    ("adx",          "Trend strength (ADX, 14d)",          "higher = stronger established trend"),
    ("realized_vol", "Realized volatility (20d, %/day)",   "expect a HUMP — mid predictable, extremes noisy"),
    ("breadth",      "Signal breadth (methods agreeing)",  "higher = more convergence (known +0.48 edge)"),
    ("price",        "Stock price ($)",                    "penny (low) vs large-cap (high) — are cheap names easier or harder to predict?"),
    ("dollar_vol",   "Dollar volume (20d avg, $M)",        "does liquidity matter? thin books may be noisier / harder to predict"),
]
# Attached to the panel from OHLCV by build_feature_panel. ``price`` is NOT here —
# it's already a column on the signals panel (the snapshot price).
_OHLCV_FEATURES = ("eff_ratio", "adx", "realized_vol", "dollar_vol")
FEATURE_LABELS = {c: lbl for c, lbl, _ in FEATURES}
BASELINE_KEY = "(all rows)"


# ── per-stock feature series (causal, so a full-series read at the anchor bar
#    equals the as-of-signal-date value — Wilder smoothing is a causal recursion) ──

def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA smoothing (EMA with alpha = 1/period) — matches trend_strength."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _dmi_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    """Full Wilder ``(adx, plus_di, minus_di)`` series (the series form of
    trend_strength._compute_dmi). +DI vs −DI gives the trend DIRECTION that
    orients the otherwise-unsigned ADX into a directional signal."""
    prev_close = close.shift(1)
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, period).replace(0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr
    minus_di = 100.0 * _wilder(minus_dm, period) / atr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return _wilder(dx.fillna(0.0), period), plus_di, minus_di


def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Full Wilder ADX series (unsigned trend strength)."""
    return _dmi_series(high, low, close, period)[0]


def _hlc_by_session(ticker: str):
    """``(session_dates_sorted, High, Low, Close, Volume)`` for a ticker from the
    OHLCV cache, keyed by ET session date so it aligns with ``build_panel``'s
    anchor. None when there's no usable history. Close-only cached series fall
    back to close for High/Low (ADX degrades gracefully) and 0 for Volume."""
    from src.data.cache import load_ohlcv
    from src.performance.daily_nav import _session_date
    df = load_ohlcv(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    has_hl = "High" in df.columns and "Low" in df.columns
    has_vol = "Volume" in df.columns
    sdates = [_session_date(ts) for ts in df.index]
    closes = pd.to_numeric(df["Close"], errors="coerce").tolist()
    highs = pd.to_numeric(df["High"], errors="coerce").tolist() if has_hl else closes
    lows = pd.to_numeric(df["Low"], errors="coerce").tolist() if has_hl else closes
    vols = pd.to_numeric(df["Volume"], errors="coerce").tolist() if has_vol else [0.0] * len(closes)
    rows: dict = {}
    for d, h, l, c, v in zip(sdates, highs, lows, closes, vols):
        if d is None or c is None or not (c > 0):
            continue
        rows[d] = (h if (h is not None and h == h) else c,   # h==h drops NaN
                   l if (l is not None and l == l) else c, c,
                   float(v) if (v is not None and v == v) else 0.0)
    if not rows:
        return None
    idx = sorted(rows)
    return (idx,
            pd.Series([rows[d][0] for d in idx], index=idx),
            pd.Series([rows[d][1] for d in idx], index=idx),
            pd.Series([rows[d][2] for d in idx], index=idx),
            pd.Series([rows[d][3] for d in idx], index=idx))


def _feature_frame(ticker: str, er_window: int, vol_window: int, adx_period: int):
    """``(session_dates, DataFrame)`` of causal per-stock features indexed by
    session date. Columns: ``eff_ratio`` / ``adx`` / ``realized_vol`` /
    ``dollar_vol`` (unsigned, for the bucket analysis) and the SIGNED directional
    trend signals ``er_signed`` / ``adx_signed`` (for the standalone IC). None
    when no history."""
    hlc = _hlc_by_session(ticker)
    if hlc is None:
        return None
    idx, high, low, close, volume = hlc
    net = close - close.shift(er_window)                          # signed net move
    den = close.diff().abs().rolling(er_window).sum().replace(0, np.nan)   # path length
    adx, plus_di, minus_di = _dmi_series(high, low, close, adx_period)
    fs = pd.DataFrame(index=idx)
    fs["eff_ratio"] = net.abs() / den                            # Kaufman ER ∈ [0, 1]
    fs["er_signed"] = net / den                                  # signed ER ∈ [−1, 1]
    fs["realized_vol"] = close.pct_change().rolling(vol_window).std(ddof=1) * 100.0
    fs["adx"] = adx.values
    fs["adx_signed"] = np.sign((plus_di - minus_di).values) * (adx.values / _ADX_NORM)
    # 20-day average dollar volume in $M (liquidity feature; 0 when no volume data).
    fs["dollar_vol"] = (close * volume).rolling(vol_window).mean() / 1e6
    return idx, fs


def _attach_asof(panel: pd.DataFrame, columns, er_window: int, vol_window: int,
                 adx_period: int) -> pd.DataFrame:
    """Attach the named feature columns to a ``build_panel`` frame, each read at
    the anchor bar (first session ≥ signal_date — the same bar ``build_panel``
    anchors the forward return on), so every feature is known at entry (no
    look-ahead). One feature-frame per ticker, shared across its rows."""
    fp = panel.copy()
    fp["_sig"] = fp["signal_date"].map(date.fromisoformat)
    frames: dict = {}
    for tk in fp["ticker"].unique():
        try:
            frames[tk] = _feature_frame(tk, er_window, vol_window, adx_period)
        except Exception:
            frames[tk] = None

    def lookup(row) -> pd.Series:
        pair = frames.get(row["ticker"])
        blank = pd.Series({c: None for c in columns})
        if pair is None:
            return blank
        idx, fs = pair
        i = bisect_left(idx, row["_sig"])
        if i >= len(idx):
            return blank
        r = fs.iloc[i]
        return pd.Series({c: (float(r[c]) if (c in fs.columns and pd.notna(r[c])) else None)
                          for c in columns})

    fp[list(columns)] = fp.apply(lookup, axis=1)
    return fp.drop(columns=["_sig"])


def build_feature_panel(panel: pd.DataFrame, er_window: int = 20, vol_window: int = 20,
                        adx_period: int = 14) -> pd.DataFrame:
    """Attach the as-of-signal-date bucket features (``eff_ratio`` / ``adx`` /
    ``realized_vol``) plus ``breadth`` (from ``n_methods_agreeing``) to a
    ``build_panel`` frame — the input to ``compute_predictability_ic``."""
    if panel is None or panel.empty:
        return pd.DataFrame()
    fp = _attach_asof(panel, _OHLCV_FEATURES, er_window, vol_window, adx_period)
    if "n_methods_agreeing" in fp.columns:
        fp["breadth"] = pd.to_numeric(fp["n_methods_agreeing"], errors="coerce")
    return fp


def attach_feature_signals(panel: pd.DataFrame, er_window: int = 20, vol_window: int = 20,
                           adx_period: int = 14) -> pd.DataFrame:
    """Attach ONLY the signed directional trend signals (``er_signed`` /
    ``adx_signed``) to a ``build_panel`` frame, so ``signal_panel.compute_ic``
    evaluates their standalone IC alongside the methods. Fail-soft: returns the
    panel unchanged on any error (compute_ic then simply skips the columns)."""
    if panel is None or panel.empty:
        return panel
    try:
        return _attach_asof(panel, FEATURE_SIGNAL_COLUMNS, er_window, vol_window, adx_period)
    except Exception:
        return panel


# ── directional IC of the signed trend signals (for stock discovery) ─────────

# The three sides each signal is evaluated on: everything, the long calls only
# (signal points up, score > 0), and the short calls only (score < 0). Splitting
# the IC this way shows whether a trend signal predicts on ONE side but not the
# other (e.g. clean-uptrend longs work while clean-downtrend shorts are noise).
_SIDES = ("all", "buy", "sell")
_SIDE_LABELS = {"all": "All", "buy": "Buys (long calls)", "sell": "Sells (short calls)"}


def _side_mask(scores: pd.Series, side: str) -> pd.Series:
    if side == "buy":
        return scores > _EPS
    if side == "sell":
        return scores < -_EPS
    return scores.abs() > _EPS


def compute_directional_feature_ic(panel: pd.DataFrame, horizons: Sequence[int] = (1, 5, 10),
                                   min_n: int = 20,
                                   features: Sequence[str] = FEATURE_SIGNAL_COLUMNS) -> pd.DataFrame:
    """Per signed-trend-feature × side (all / buy / sell) × horizon: the
    observation count ``n``, Spearman ``ic`` (of the signal vs the forward
    return), directional ``win`` % (share whose sign matched — a long call right
    when the stock rose, a short call right when it fell), and ``sim`` % (mean
    ``sign(score) × forward_return`` — the side's gross P&L if traded in the
    signal's direction). ``panel`` must already carry the ``er_signed`` /
    ``adx_signed`` columns (via ``attach_feature_signals``) and ``fwd_ret_<h>d``.

    A feature "works" on a side when its win > 50 and sim > 0 PERSISTING across
    horizons; the per-side split reveals a signal that predicts one direction but
    not the other. IC keeps the standard convention — positive means the signal's
    magnitude ranks forward returns in the direction it points, on that side."""
    if panel is None or panel.empty:
        return pd.DataFrame()
    rows = []
    for col in features:
        if col not in panel.columns:
            continue
        s_all = pd.to_numeric(panel[col], errors="coerce")
        for side in _SIDES:
            base = _side_mask(s_all, side) & s_all.notna()
            row = {"feature": col, "label": FEATURE_SIGNAL_LABELS.get(col, col),
                   "side": side, "side_label": _SIDE_LABELS[side]}
            for h in horizons:
                f_all = pd.to_numeric(panel.get(f"fwd_ret_{h}d"), errors="coerce")
                valid = base & f_all.notna()
                n = int(valid.sum())
                row[f"n_{h}d"] = n
                if n < min_n:
                    row[f"ic_{h}d"] = row[f"win_{h}d"] = row[f"sim_{h}d"] = None
                    continue
                s, f = s_all[valid], f_all[valid]
                ic = _spearman(s, f)
                row[f"ic_{h}d"] = round(ic, 4) if ic is not None else None
                moved = f != 0
                row[f"win_{h}d"] = (round(float(((s > 0) == (f > 0))[moved].mean() * 100.0), 2)
                                    if bool(moved.any()) else None)
                row[f"sim_{h}d"] = round(float(f.where(s > 0, -f).mean()), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def load_directional_feature_ic(horizons: Sequence[int] = (1, 5, 10),
                                days: Optional[int] = None, min_n: int = 20) -> pd.DataFrame:
    """Build the signals panel, enrich it with the signed trend signals, and
    compute their by-direction IC. Empty when there's no forward-return history."""
    panel = build_panel(horizons=horizons, days=days)
    if panel is None or panel.empty:
        return pd.DataFrame()
    return compute_directional_feature_ic(attach_feature_signals(panel),
                                          horizons=horizons, min_n=min_n)


# ── bucketed conditional IC of combined_score ────────────────────────────────

def _score_stats(sub: pd.DataFrame, horizons: Sequence[int], min_n: int) -> dict:
    """For a subset of rows: per horizon, how well ``combined_score`` predicted
    the forward return — ``n``, Spearman ``ic``, directional ``hit`` %, and signed
    ``simret`` % (mean sign(score)×fwd_ret). Zero scores ("no view") excluded."""
    score = pd.to_numeric(sub.get("combined_score"), errors="coerce")
    has_view = score.notna() & (score.abs() > _EPS)
    out: dict = {}
    for h in horizons:
        fwd = pd.to_numeric(sub.get(f"fwd_ret_{h}d"), errors="coerce")
        valid = has_view & fwd.notna()
        n = int(valid.sum())
        out[f"n_{h}d"] = n
        if n < min_n:
            out[f"ic_{h}d"] = out[f"hit_{h}d"] = out[f"simret_{h}d"] = None
            continue
        s, f = score[valid], fwd[valid]
        ic = _spearman(s, f)
        out[f"ic_{h}d"] = round(ic, 4) if ic is not None else None
        moved = f != 0
        out[f"hit_{h}d"] = (round(float(((s > 0) == (f > 0))[moved].mean() * 100.0), 2)
                            if bool(moved.any()) else None)
        out[f"simret_{h}d"] = round(float(f.where(s > 0, -f).mean()), 4)
    return out


def _bucket_names(k: int) -> List[str]:
    if k == 3:
        return ["Low", "Mid", "High"]
    if k == 2:
        return ["Low", "High"]
    return [f"Q{i + 1}" for i in range(k)]


def compute_predictability_ic(feature_panel: pd.DataFrame, horizons: Sequence[int] = (1, 5, 10),
                              min_n: int = 30, n_buckets: int = 3,
                              features: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Bucket the feature panel by each feature (into ``n_buckets`` quantiles) and
    report ``combined_score``'s conditional prediction quality per bucket. Row 0 is
    the unconditional ``(all rows)`` baseline; then, per feature, its ordered
    Low→High buckets. Columns: ``feature``, ``label``, ``bucket``, ``lo``/``hi``
    (the bucket's feature range), ``n_rows``, and per horizon ``n_/ic_/hit_/simret_``."""
    if feature_panel is None or feature_panel.empty:
        return pd.DataFrame()
    feats = list(features) if features is not None else [f[0] for f in FEATURES]
    rows = [{"feature": BASELINE_KEY, "label": "All scored rows (baseline)", "bucket": "all",
             "lo": None, "hi": None, "n_rows": int(len(feature_panel)),
             **_score_stats(feature_panel, horizons, min_n)}]

    for col in feats:
        if col not in feature_panel.columns:
            continue
        s = pd.to_numeric(feature_panel[col], errors="coerce")
        if int(s.notna().sum()) < min_n or int(s.dropna().nunique()) < 2:
            continue
        try:
            cats = pd.qcut(s, n_buckets, duplicates="drop")
        except Exception:
            continue
        categories = list(cats.cat.categories)
        names = _bucket_names(len(categories))
        for bi, interval in enumerate(categories):
            sub = feature_panel[cats == interval]
            rows.append({
                "feature": col, "label": FEATURE_LABELS.get(col, col), "bucket": names[bi],
                "lo": round(float(interval.left), 4), "hi": round(float(interval.right), 4),
                "n_rows": int(len(sub)), **_score_stats(sub, horizons, min_n),
            })
    return pd.DataFrame(rows)


def summarize_feature_edges(bucket_df: pd.DataFrame,
                            horizons: Sequence[int] = (1, 5, 10)) -> pd.DataFrame:
    """The Tier-0 punchline: per feature, how much its buckets SEPARATE prediction
    quality — the best-minus-worst-bucket spread in hit % and sim-return %, plus
    which bucket is best, per horizon. A large positive spread with a sensible
    best bucket (High for trend features, Mid for volatility) = a predictability
    filter worth acting on; a spread ≈ 0 = the feature doesn't sort predictable
    from unpredictable."""
    if bucket_df is None or bucket_df.empty:
        return pd.DataFrame()
    rows = []
    body = bucket_df[bucket_df["feature"] != BASELINE_KEY]
    for feat, g in body.groupby("feature", sort=False):
        row = {"feature": feat, "label": g.iloc[0]["label"], "buckets": int(len(g))}
        for h in horizons:
            for m in ("hit", "simret"):
                vals = [(r["bucket"], r.get(f"{m}_{h}d")) for _, r in g.iterrows()
                        if pd.notna(r.get(f"{m}_{h}d"))]
                if len(vals) >= 2:
                    best = max(vals, key=lambda x: x[1])
                    worst = min(vals, key=lambda x: x[1])
                    row[f"{m}_spread_{h}d"] = round(best[1] - worst[1], 3)
                    row[f"{m}_best_{h}d"] = best[0]
                else:
                    row[f"{m}_spread_{h}d"] = None
                    row[f"{m}_best_{h}d"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ── live-DB convenience layer ────────────────────────────────────────────────

def load_predictability(horizons: Sequence[int] = (1, 5, 10), days: Optional[int] = None,
                        min_n: int = 30, n_buckets: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(bucket_table, edge_summary)`` over the signals panel. Empty frames when
    there's no forward-return history yet."""
    panel = build_panel(horizons=horizons, days=days)
    if panel is None or panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    fp = build_feature_panel(panel)
    ic = compute_predictability_ic(fp, horizons=horizons, min_n=min_n, n_buckets=n_buckets)
    return ic, summarize_feature_edges(ic, horizons=horizons)


def _print_report(horizons: Sequence[int], days: Optional[int], min_n: int, n_buckets: int) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from src.db import repo
    repo.set_read_only(True)

    ic, edges = load_predictability(horizons=horizons, days=days, min_n=min_n, n_buckets=n_buckets)
    if ic is None or ic.empty:
        print("No signals-panel rows with forward returns yet — warm forward closes with "
              "`python -m src.analysis.signal_panel --refresh`, then re-run.")
        return

    print(f"\nPredictability-feature IC — combined_score vs forward return, bucketed "
          f"({n_buckets} quantiles/feature)")
    print("hit = directional hit rate %; IC = Spearman(score, fwd_ret); sim = mean signed "
          "return %. A feature helps iff these SEPARATE across its buckets.\n")
    head = f"{'feature / bucket':<40}{'n_rows':>8}"
    for h in horizons:
        head += f"{f'n@{h}':>6}{f'IC@{h}':>8}{f'hit@{h}':>8}{f'sim@{h}':>8}"
    for _, r in ic.iterrows():
        if r["feature"] != BASELINE_KEY and r["bucket"] in ("Low", "Q1", "all"):
            print("-" * len(head))
            tag = r["label"] if r["feature"] != BASELINE_KEY else ""
            if tag:
                print(tag)
            print(head)
        rng = "" if r["lo"] is None else f" [{r['lo']:g}, {r['hi']:g}]"
        name = ("BASELINE" if r["feature"] == BASELINE_KEY else f"  {r['bucket']}{rng}")
        line = f"{name:<40}{int(r['n_rows']):>8}"
        for h in horizons:
            n, ic_v, hit, sim = (r.get(f"n_{h}d"), r.get(f"ic_{h}d"),
                                 r.get(f"hit_{h}d"), r.get(f"simret_{h}d"))
            line += f"{int(n) if pd.notna(n) else 0:>6}"
            line += f"{ic_v:>+8.3f}" if pd.notna(ic_v) else f"{'—':>8}"
            line += f"{hit:>7.1f}%" if pd.notna(hit) else f"{'—':>8}"
            line += f"{sim:>+8.2f}" if pd.notna(sim) else f"{'—':>8}"
        print(line)

    if edges is not None and not edges.empty:
        print("\n\nFeature edge — best-minus-worst bucket separation (the punchline)")
        h_head = f"{'feature':<40}"
        for h in horizons:
            h_head += f"{f'hitΔ@{h}':>9}{f'best@{h}':>8}{f'simΔ@{h}':>9}"
        print(h_head)
        print("-" * len(h_head))
        for _, r in edges.iterrows():
            line = f"{r['label']:<40}"
            for h in horizons:
                hd, hb, sd = (r.get(f"hit_spread_{h}d"), r.get(f"hit_best_{h}d"),
                              r.get(f"simret_spread_{h}d"))
                line += f"{hd:>9.2f}" if pd.notna(hd) else f"{'—':>9}"
                line += f"{str(hb):>8}" if hb is not None and pd.notna(hb) else f"{'—':>8}"
                line += f"{sd:>9.3f}" if pd.notna(sd) else f"{'—':>9}"
            print(line)

    # Signed trend signals evaluated for stock discovery, split by side.
    dir_ic = load_directional_feature_ic(horizons=horizons, days=days)
    if dir_ic is not None and not dir_ic.empty:
        print("\n\nStock discovery — signed trend signal IC by direction "
              "(does the signal predict on longs / shorts?)")
        head = f"{'signal / side':<40}"
        for h in horizons:
            head += f"{f'n@{h}':>7}{f'IC@{h}':>8}{f'win@{h}':>8}{f'sim@{h}':>8}"
        for _, r in dir_ic.iterrows():
            if r["side"] == "all":
                print("-" * len(head))
                print(r["label"])
                print(head)
            line = f"  {_SIDE_LABELS[r['side']]:<38}"
            for h in horizons:
                n, ic_v, win, sim = (r.get(f"n_{h}d"), r.get(f"ic_{h}d"),
                                     r.get(f"win_{h}d"), r.get(f"sim_{h}d"))
                line += f"{int(n) if pd.notna(n) else 0:>7}"
                line += f"{ic_v:>+8.3f}" if pd.notna(ic_v) else f"{'—':>8}"
                line += f"{win:>7.1f}%" if pd.notna(win) else f"{'—':>8}"
                line += f"{sim:>+8.2f}" if pd.notna(sim) else f"{'—':>8}"
            print(line)

    print("\nSeparation across buckets (not the level) is the signal: trend features should "
          "improve Low→High, volatility should peak in the MID bucket. For the directional "
          "table, a signal EARNS a side when win > 50 and sim > 0 persist there. "
          "Forward-collected — judge the longer horizons only as the panel thickens.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Predictability-feature IC panel: does combined_score predict better "
                    "inside high-trend / moderate-vol / high-breadth buckets?")
    p.add_argument("--horizons", default="1,5,10", help="forward horizons in sessions (default 1,5,10)")
    p.add_argument("--days", type=int, default=None, help="only signals from the last N days (default all)")
    p.add_argument("--min-n", type=int, default=30, help="min rows per bucket before its stats report (default 30)")
    p.add_argument("--buckets", type=int, default=3, help="quantile buckets per feature (default 3)")
    args = p.parse_args(list(argv) if argv is not None else None)
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())
    _print_report(horizons, args.days, args.min_n, args.buckets)


if __name__ == "__main__":
    main()
