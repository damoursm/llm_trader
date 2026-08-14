"""ML dataset builder — features and labels for the OHLCV-only base-signal model
(Phase 0 of the "trained models as new methods" work; see
``memory/ml-methods-plan-2026-07.md``).

The whole point of an OHLCV-only model is that every input is reconstructable
from the backfilled daily cache, so training escapes the ~5-week/one-regime
``signals`` panel and sees years x thousands of tickers. This module turns the
cache into a supervised table:

    one row per (ticker, session_date)
      features    : cheap causal price/volume features known AT THE CLOSE of the
                    session_date bar (no look-ahead — see the causality note)
      labels      : forward return over the next h sessions, on TWO bases —
                    raw (``fwd_ret_raw_<h>d``) and market-relative net of the
                    benchmark (``fwd_ret_rel_<h>d``) — plus ``end_date_<h>d``,
                    the calendar date the label's end bar prints, which the
                    walk-forward split needs to know a label is realised.

**Causality is the one property that matters here, and it is by construction.**
Every feature is a rolling / ewm / shift over ``close.iloc[:i+1]`` — a value at
bar ``i`` depends only on bars ``<= i`` — and every label uses bars ``i+1 ..
i+h`` strictly in the future, so features and labels never overlap. The
observation is defined as "the close of session D is known; predict D -> D+h",
which mirrors ``replay.visible_history`` with a post-close run (bars ``<= D``).
``tests/test_ml_dataset.py`` pins this adversarially: appending future bars must
not move any already-emitted row.

**Survivorship caveat (deliberate, documented).** The cache is TODAY's universe,
so deep reconstruction silently drops names that delisted or went to zero — the
training set is optimistic. Per the plan this is acceptable because the
PROMOTION decision is made later on the forward-collected ``signals`` panel
(survivorship-free forward); the deep cache is only for LEARNING robust features
across regimes. Split-adjustment contamination is handled by reusing the replay
split guards.

Reuses the live scorers' own causal primitives (``predictability._hlc_by_session``,
the Wilder/DMI helpers, the SPY benchmark) so the features cannot drift from the
indicators the rest of the system computes.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.analysis.predictability import _hlc_by_session, _dmi_series, _wilder

# Minimum completed bars before a row is emitted — long enough for the 252-day
# windows to be defined. Rows shorter than this yield NaN features and are of no
# use to the model, so they are dropped at the source.
_MIN_BARS = 260

# The feature columns, in a stable order (the model keys on this list). Grouped
# by family for readability; every one is causal (rolling/ewm/shift only).
FEATURE_COLUMNS: List[str] = [
    # returns / momentum
    "ret_1", "ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "ret_252", "ret_12_1",
    # trend quality
    "eff_ratio", "er_signed", "adx", "adx_signed",
    "ma_dist_20", "ma_dist_50", "ma_dist_200", "donchian_pos_20", "donchian_pos_55",
    # mean-reversion
    "rsi_14", "z_20", "bb_width_20",
    # volatility
    "realized_vol_20", "realized_vol_60", "atr_pct_14",
    # volume / flow
    "rvol_20", "updown_vol_10", "dollar_vol_log",
    # range position
    "pct_from_52w_high", "pct_from_52w_low",
    # size / liquidity conditioning
    "log_price",
    # engineered orthogonal features (2026-07-31)
    "mom_accel", "vol_regime", "vol_accel", "close_loc_5", "bb_width_delta", "ret_skew_20",
]

# ── within-ticker normalisation (2026-08-05) ─────────────────────────────────
# A pooled tree splits on ABSOLUTE feature values across a heterogeneous
# universe, so `realized_vol_20 > 2.5` means "calm" for a utility and "wild" for
# a biotech and every split is a compromise. Each feature's EXPANDING
# within-ticker z-score gives the model that stock's OWN context, while the raw
# column keeps the cross-sectional LEVEL the market-relative label ranks on.
#
# Measured (paired walk-forward, 279 tickers x 131,947 OOS rows, 23 retrains):
# raw+z beats raw-only on the per-ticker IC at BOTH horizons — 1d 164/279 (59%,
# p=0.0020), 10d 156/279 (56%, p=0.0276) — and at 10d it flips the mean
# per-ticker IC from -0.0047 to +0.0037 while cutting simret loss -0.124 ->
# -0.045. Dropping the raw columns and keeping only z was WORSE (daily IC
# +0.0239 vs +0.0290), so BOTH halves are load-bearing. This was the byproduct of
# testing per-ticker training, which itself measured as no gain at 10d — see
# memory/per-ticker-vs-pooled-2026-08.md.
#
# CAUSAL by construction: an expanding window over rows <= i, so the look-ahead
# probe in tests/test_ml_dataset.py covers these columns too.
_TZ_SUFFIX = "_tz"
_TZ_MIN_OBS = 60                      # z is NaN until the ticker has this many
TZ_FEATURE_COLUMNS: List[str] = [f + _TZ_SUFFIX for f in FEATURE_COLUMNS]

# Everything ``ticker_feature_frame`` emits (raw + within-ticker z). Consumers
# that read a per-ticker frame must iterate THIS, not FEATURE_COLUMNS.
FRAME_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + TZ_FEATURE_COLUMNS

# Cross-sectional features are added AFTER the per-ticker frames are stacked
# (they rank within a signal_date, so they need the whole day's universe). Still
# causal — a same-day rank uses only same-day values. Aligned with the
# market-relative label: relative positioning across the universe is genuinely
# different information from a single stock's absolute values. One source tuple
# so build_dataset and build_panel_dataset cannot drift.
_XRANK_SOURCES = (("ret_5", "xrank_ret5"), ("ret_21", "xrank_ret21"),
                  ("ret_63", "xrank_ret63"), ("realized_vol_20", "xrank_vol20"),
                  ("rsi_14", "xrank_rsi14"), ("dollar_vol_log", "xrank_dvol"))
_XSECTION_COLUMNS: List[str] = [dst for _, dst in _XRANK_SOURCES]

ALL_FEATURE_COLUMNS: List[str] = FRAME_FEATURE_COLUMNS + _XSECTION_COLUMNS

_EPS = 1e-12


# ── per-ticker causal feature frame ──────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI — causal (ewm recursion). NaN until ``period`` bars accrue."""
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0), period)
    loss = _wilder((-delta).clip(lower=0.0), period)
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def ticker_feature_frame(ticker: str) -> Optional[pd.DataFrame]:
    """Causal feature frame for one ticker, indexed by session date.

    Every column is a rolling / ewm / shift over the cached daily bars, so the
    value at row ``i`` uses only bars ``<= i``. Returns None when the ticker has
    no usable history. Also carries ``Close`` so the caller can build forward
    labels off the identical session grid.
    """
    hlc = _hlc_by_session(ticker)
    if hlc is None:
        return None
    idx, high, low, close, volume = hlc
    if len(idx) < 20:
        return None
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    fs = pd.DataFrame(index=idx)
    fs["Close"] = close.values

    # returns / momentum (%)
    for w in (1, 5, 10, 21, 63, 126, 252):
        fs[f"ret_{w}"] = (close / close.shift(w) - 1.0) * 100.0
    # 12-1 skip-month momentum: 252d ago -> 21d ago, excluding the last month.
    fs["ret_12_1"] = (close.shift(21) / close.shift(252) - 1.0) * 100.0

    # trend quality
    net = close - close.shift(20)
    path = close.diff().abs().rolling(20).sum().replace(0.0, np.nan)
    fs["eff_ratio"] = (net.abs() / path).values
    fs["er_signed"] = (net / path).values
    adx, plus_di, minus_di = _dmi_series(high, low, close, 14)
    fs["adx"] = adx.values
    fs["adx_signed"] = (np.sign((plus_di - minus_di).values) * (adx.values / 40.0))
    for w in (20, 50, 200):
        sma = close.rolling(w).mean()
        fs[f"ma_dist_{w}"] = ((close / sma - 1.0) * 100.0).values
    for w in (20, 55):
        hi = high.rolling(w).max()
        lo = low.rolling(w).min()
        rng = (hi - lo).replace(0.0, np.nan)
        fs[f"donchian_pos_{w}"] = ((close - lo) / rng).values

    # mean-reversion
    fs["rsi_14"] = _rsi(close, 14).values
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=1).replace(0.0, np.nan)
    fs["z_20"] = ((close - sma20) / std20).values
    fs["bb_width_20"] = ((4.0 * std20) / sma20.replace(0.0, np.nan)).values  # (upper-lower)/mid, 2σ bands

    # volatility
    pct = close.pct_change()
    fs["realized_vol_20"] = (pct.rolling(20).std(ddof=1) * 100.0).values
    fs["realized_vol_60"] = (pct.rolling(60).std(ddof=1) * 100.0).values
    tr = _true_range(high, low, close)
    fs["atr_pct_14"] = ((_wilder(tr, 14) / close.replace(0.0, np.nan)) * 100.0).values

    # volume / flow
    vol_sma20 = volume.rolling(20).mean().replace(0.0, np.nan)
    fs["rvol_20"] = (volume / vol_sma20).values
    up_day = (close.diff() > 0).astype(float)
    up_vol = (volume * up_day).rolling(10).sum()
    tot_vol = volume.rolling(10).sum().replace(0.0, np.nan)
    fs["updown_vol_10"] = (2.0 * (up_vol / tot_vol) - 1.0).values          # signed share ∈ [-1, 1]
    dvol = (close * volume).rolling(20).mean()
    fs["dollar_vol_log"] = np.log10(dvol.replace(0.0, np.nan).clip(lower=1.0)).values

    # range position
    hi_252 = close.rolling(252, min_periods=60).max()
    lo_252 = close.rolling(252, min_periods=60).min()
    fs["pct_from_52w_high"] = ((close / hi_252 - 1.0) * 100.0).values
    fs["pct_from_52w_low"] = ((close / lo_252 - 1.0) * 100.0).values

    # size / liquidity conditioning
    fs["log_price"] = np.log10(close.replace(0.0, np.nan).clip(lower=0.01)).values

    # ── engineered features (2026-07-31) — chosen for ORTHOGONALITY to the return
    #    block above (a different aspect of the tape), not more collinear windows.
    #    All causal (rolling/ewm/shift/pct_change), so the look-ahead probe covers them.
    # momentum acceleration: the recent 10d vs the prior 10d (momentum of momentum).
    fs["mom_accel"] = (2.0 * fs["ret_10"] - fs["ret_21"]).values
    # volatility regime: short vs long realized vol (>0 expanding, <0 contracting) —
    # distinct from the vol LEVEL already captured by realized_vol_20/60.
    fs["vol_regime"] = (fs["realized_vol_20"] / pd.Series(fs["realized_vol_60"]).replace(0.0, np.nan) - 1.0).values
    # volume surge: recent 5d average volume vs the 20d baseline.
    fs["vol_accel"] = (volume.rolling(5).mean() / vol_sma20 - 1.0).values
    # intraday close location in the H-L range, averaged 5d, centered to [-1, 1]
    # (closing near the high vs the low — orthogonal to close-to-close returns).
    _rng = (high - low).replace(0.0, np.nan)
    _loc = ((close - low) / _rng).clip(0.0, 1.0)
    fs["close_loc_5"] = (2.0 * _loc.rolling(5).mean() - 1.0).values
    # Bollinger width vs 20 bars ago: coiling (<0) vs expanding (>0) — the squeeze read.
    _bbw = (4.0 * std20) / sma20.replace(0.0, np.nan)
    fs["bb_width_delta"] = (_bbw / _bbw.shift(20) - 1.0).values
    # return skew (20d): tail asymmetry of the daily-return distribution.
    fs["ret_skew_20"] = pct.rolling(20).skew().values

    return _add_within_ticker_z(fs)


def _add_within_ticker_z(fs: pd.DataFrame) -> pd.DataFrame:
    """Append each feature's EXPANDING within-ticker z-score (``<feat>_tz``).

    Causal: ``expanding`` at row ``i`` sees only rows ``<= i`` of THIS ticker, and
    ``min_periods`` counts non-NaN observations, so a feature still warming up
    yields NaN rather than a z-score off two points. A full-sample z would leak
    the ticker's future distribution into every past row — the exact error the
    walk-forward machinery exists to prevent.

    ``ddof=0`` (population) matches the cumulative-sum implementation the effect
    was measured with; over >= 60 observations the ddof choice moves the z by
    <1%, but keeping it identical means the shipped feature is the validated one.
    """
    base = fs[FEATURE_COLUMNS]
    win = base.expanding(min_periods=_TZ_MIN_OBS)
    sd = win.std(ddof=0).replace(0.0, np.nan)          # constant feature -> NaN, not inf
    z = (base - win.mean()) / sd
    z.columns = TZ_FEATURE_COLUMNS
    return pd.concat([fs, z], axis=1)


# ── benchmark (market-relative label leg) ────────────────────────────────────

def _benchmark_series(benchmark: str) -> Tuple[List[date], List[float]]:
    """``(sorted_session_dates, closes)`` for the benchmark, for same-window
    market-relative returns. Empty when the benchmark isn't cached."""
    hlc = _hlc_by_session(benchmark)
    if hlc is None:
        return [], []
    idx, _h, _l, close, _v = hlc
    return list(idx), [float(c) for c in close.values]


def _benchmark_return(b_dates: List[date], b_closes: List[float],
                      d0: date, d1: date) -> Optional[float]:
    """Benchmark % return over the wall-clock window [d0, d1], using the last
    close at or before each endpoint. None when the benchmark can't cover it."""
    if not b_dates or d1 <= d0:
        return None
    i0 = bisect_right(b_dates, d0) - 1
    i1 = bisect_right(b_dates, d1) - 1
    if i0 < 0 or i1 < 0:
        return None
    c0, c1 = b_closes[i0], b_closes[i1]
    if c0 <= 0:
        return None
    return (c1 / c0 - 1.0) * 100.0


# ── dataset assembly ─────────────────────────────────────────────────────────

def cached_universe(limit: Optional[int] = None) -> List[str]:
    """Ticker symbols with a cached daily OHLCV file. Deterministic (sorted).

    This is TODAY's universe — the survivorship caveat in the module docstring
    applies. ``limit`` takes the first N alphabetically for quick runs.
    """
    from src.data.cache import _ohlcv_dir
    d = _ohlcv_dir("1d")
    if not d.exists():
        return []
    names = sorted(p.stem.upper() for p in d.glob("*.json"))
    return names[:limit] if limit else names


def build_dataset(tickers: Optional[Sequence[str]] = None,
                  horizons: Sequence[int] = (1, 3),
                  benchmark: Optional[str] = None,
                  limit_tickers: Optional[int] = None,
                  date_stride: int = 1,
                  min_date: Optional[str] = None) -> pd.DataFrame:
    """Build the (ticker, date) feature/label table over the cached universe.

    One row per emitted (ticker, session_date). Feature columns are causal;
    label columns are ``fwd_ret_raw_<h>d`` / ``fwd_ret_rel_<h>d`` / ``end_date_<h>d``
    per horizon. ``date_stride`` subsamples every Nth eligible session per ticker
    (1 = all), ``min_date`` drops earlier history.

    The benchmark leg is loaded once. A row's raw and relative labels are NaN
    independently: a ticker with a forward bar but no benchmark coverage still
    contributes its raw label.
    """
    benchmark = benchmark or settings.horizon_market_benchmark
    tickers = list(tickers) if tickers is not None else cached_universe(limit_tickers)
    horizons = list(horizons)
    hmax = max(horizons)
    b_dates, b_closes = _benchmark_series(benchmark)
    min_d = date.fromisoformat(min_date) if min_date else None

    frames: List[pd.DataFrame] = []
    n_skipped = 0
    for tk in tickers:
        if tk == benchmark:
            continue
        try:
            fs = ticker_feature_frame(tk)
        except Exception as e:                          # a broken cache file must not abort the build
            logger.debug(f"[ml_dataset] {tk} feature build failed: {e}")
            fs = None
        if fs is None or len(fs) < _MIN_BARS:
            n_skipped += 1
            continue
        idx = list(fs.index)
        closes = [float(c) for c in fs["Close"].values]
        n = len(idx)
        rows: List[dict] = []
        # Only sessions with (a) the full lookback behind them and (b) at least
        # hmax forward bars ahead can carry both features and a label.
        for i in range(_MIN_BARS - 1, n - hmax):
            if (i - (_MIN_BARS - 1)) % date_stride != 0:
                continue
            d0 = idx[i]
            if min_d is not None and d0 < min_d:
                continue
            base = closes[i]
            if not base or base <= 0:
                continue
            rec = {"ticker": tk, "signal_date": d0.isoformat()}
            frow = fs.iloc[i]
            for c in FRAME_FEATURE_COLUMNS:
                v = frow.get(c)
                rec[c] = float(v) if v is not None and pd.notna(v) else np.nan
            for h in horizons:
                j = i + h
                d1 = idx[j]
                raw = (closes[j] / base - 1.0) * 100.0
                rec[f"fwd_ret_raw_{h}d"] = raw
                rec[f"end_date_{h}d"] = d1.isoformat()
                bench = _benchmark_return(b_dates, b_closes, d0, d1)
                rec[f"fwd_ret_rel_{h}d"] = (raw - bench) if bench is not None else np.nan
            rows.append(rec)
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        logger.info(f"[ml_dataset] no rows built ({n_skipped} tickers had too little history)")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Cross-sectional ranks — computed AFTER stacking because they rank within a
    # signal_date across the day's universe. Still causal: a same-day rank uses
    # only same-day feature values. Scaled to [-1, 1] so an incomplete day (few
    # names) doesn't blow up the scale.
    for src, dst in _XRANK_SOURCES:
        if src in df.columns:
            r = df.groupby("signal_date")[src].rank(pct=True)
            df[dst] = (2.0 * r - 1.0)

    df = df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    logger.info(f"[ml_dataset] built {len(df):,} rows over {df['ticker'].nunique()} tickers "
                f"({df['signal_date'].min()} .. {df['signal_date'].max()})")
    return df


def build_panel_dataset(horizons: Sequence[int] = (5, 10), days: Optional[int] = None,
                        benchmark: Optional[str] = None) -> pd.DataFrame:
    """The SURVIVORSHIP-FREE evaluation set: the live ``signals`` panel (every
    ticker actually scored each tick), with the SAME causal features as
    ``build_dataset`` computed as-of each row's signal_date, plus the panel's own
    forward returns (raw) and the market-relative version.

    This is the honest promotion gate. The deep-cache dataset trains the model;
    THIS set — which includes the penny/thin/soon-delisted names the deep cache
    silently drops — judges it. Same column schema as ``build_dataset`` so the
    walk-forward validator can train on one and evaluate on the other.

    Caveat: the 6 cross-sectional features (``_XRANK_SOURCES``) rank within each
    day's universe, which differs between the deep-cache (training) and panel
    (eval) sets — a distribution shift on 6 of the columns, noted rather than
    hidden. (Said "2" until 2026-08-12; the set grew to 6 and the count was
    never updated — the number here must track ``_XRANK_SOURCES``.)
    """
    from src.analysis.signal_panel import build_panel
    benchmark = benchmark or settings.horizon_market_benchmark
    horizons = list(horizons)
    panel = build_panel(horizons=horizons, days=days)
    if panel is None or panel.empty:
        return pd.DataFrame()
    b_dates, b_closes = _benchmark_series(benchmark)

    frames: Dict[str, Optional[pd.DataFrame]] = {}

    def _frame(tk: str) -> Optional[pd.DataFrame]:
        if tk not in frames:
            try:
                frames[tk] = ticker_feature_frame(tk)
            except Exception:
                frames[tk] = None
        return frames[tk]

    rows: List[dict] = []
    for r in panel.itertuples(index=False):
        tk = getattr(r, "ticker", None)
        sd = getattr(r, "signal_date", None)
        if tk is None or sd is None or tk == benchmark:
            continue
        fs = _frame(tk)
        if fs is None:
            continue
        idx = list(fs.index)
        d0 = date.fromisoformat(str(sd)[:10])
        i = bisect_left(idx, d0)                       # anchor bar = first session >= signal_date
        if i >= len(idx):
            continue
        frow = fs.iloc[i]
        rec = {"ticker": tk, "signal_date": d0.isoformat()}
        for c in FRAME_FEATURE_COLUMNS:
            v = frow.get(c)
            rec[c] = float(v) if v is not None and pd.notna(v) else np.nan
        for h in horizons:
            raw = getattr(r, f"fwd_ret_{h}d", np.nan)   # panel's authoritative raw fwd return
            rec[f"fwd_ret_raw_{h}d"] = float(raw) if raw is not None and pd.notna(raw) else np.nan
            end = idx[i + h] if i + h < len(idx) else None
            rec[f"end_date_{h}d"] = end.isoformat() if end is not None else None
            bench = _benchmark_return(b_dates, b_closes, idx[i], end) if end is not None else None
            rec[f"fwd_ret_rel_{h}d"] = (rec[f"fwd_ret_raw_{h}d"] - bench) if (bench is not None
                                        and pd.notna(rec[f"fwd_ret_raw_{h}d"])) else np.nan
        rows.append(rec)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for src, dst in _XRANK_SOURCES:
        if src in df.columns:
            rk = df.groupby("signal_date")[src].rank(pct=True)
            df[dst] = (2.0 * rk - 1.0)
    df = df.sort_values(["signal_date", "ticker"]).reset_index(drop=True)
    logger.info(f"[ml_dataset] panel eval set: {len(df):,} rows over {df['ticker'].nunique()} "
                f"tickers ({df['signal_date'].min()} .. {df['signal_date'].max()})")
    return df


def materialize(path: str = "cache/ml/dataset.parquet", **kwargs) -> int:
    """Build the dataset and write it to Parquet via DuckDB (no pyarrow needed).

    Returns the row count. A batch job, not a hot path — the walk-forward harness
    reads this back instead of rebuilding every run.
    """
    import duckdb
    from pathlib import Path

    df = build_dataset(**kwargs)
    if df.empty:
        return 0
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("_ds", df)
        con.execute(f"COPY _ds TO '{out.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.close()
    logger.info(f"[ml_dataset] materialized {len(df):,} rows -> {out}")
    return len(df)


def load_materialized(path: str = "cache/ml/dataset.parquet") -> pd.DataFrame:
    """Read a materialized dataset back, or empty when it doesn't exist."""
    import duckdb
    from pathlib import Path
    if not Path(path).exists():
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        return con.execute(f"SELECT * FROM read_parquet('{Path(path).as_posix()}')").df()
    finally:
        con.close()


def missing_feature_columns(df: Optional[pd.DataFrame]) -> List[str]:
    """Feature columns the CURRENT code emits but this frame lacks.

    A materialized parquet outlives the feature set that built it, and every
    consumer here intersects its feature list with the frame's columns — so a
    stale file does not error, it silently trains or validates on the OLD
    features and reports a confident number for a model nobody asked for. (This
    is the same trap that made a shallow-cache parquet 'successfully' retrain
    ml_ohlcv on 13,380 rows in 2026-08-04.) Callers use this to REBUILD rather
    than proceed. Empty frame -> empty list; the caller's own emptiness check
    owns that case.
    """
    if df is None or df.empty:
        return []
    return [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the OHLCV-only ML feature/label dataset")
    p.add_argument("--horizons", default="1,3", help="forward horizons in sessions (default 1,3)")
    p.add_argument("--limit-tickers", type=int, default=None, help="first N cached tickers (quick runs)")
    p.add_argument("--date-stride", type=int, default=1, help="emit every Nth eligible session (default 1)")
    p.add_argument("--min-date", default=None, help="drop sessions before YYYY-MM-DD")
    p.add_argument("--write", default=None, help="materialize to this Parquet path")
    a = p.parse_args(argv)
    horizons = tuple(int(h) for h in str(a.horizons).split(",") if h.strip())
    if a.write:
        n = materialize(a.write, horizons=horizons, limit_tickers=a.limit_tickers,
                        date_stride=a.date_stride, min_date=a.min_date)
        print(f"materialized {n:,} rows -> {a.write}")
        return
    df = build_dataset(horizons=horizons, limit_tickers=a.limit_tickers,
                       date_stride=a.date_stride, min_date=a.min_date)
    if df.empty:
        print("No rows built.")
        return
    print(f"\n{len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df['signal_date'].min()} .. {df['signal_date'].max()}")
    print(f"features: {len(ALL_FEATURE_COLUMNS)} | horizons: {horizons}")
    for h in horizons:
        for basis in ("raw", "rel"):
            col = f"fwd_ret_{basis}_{h}d"
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col:<20} n={int(s.notna().sum()):>8}  mean={s.mean():+.3f}%  std={s.std():.3f}")


if __name__ == "__main__":
    main()
