"""Signals-panel analysis — forward returns + per-method information coefficients.

The pipeline persists the FULL per-ticker signal cross-section of every run to
the DuckDB ``signals`` table (see ``schema.py``). This module turns that panel
into evidence: it joins each (signal_date, ticker) row against forward returns
computed from the OHLCV cache the system already maintains, then reports the
Spearman rank information coefficient (IC) and directional hit rate per method
per horizon. This is the counterfactual view the trade ledger can't give —
every scored ticker counts, not just the gate-filtered top-10 that became
trades — so it's the dataset for threshold tuning and weight calibration
without selection bias.

Conventions
-----------
* One row per (signal_date, ticker): when the intraday scheduler produced
  several runs in a day, only the LAST run's row is kept (``dedupe="last"``) —
  intraday repeats are highly autocorrelated and would pseudo-replicate.
* Forward return at horizon h = close(base + h sessions) / close(base) − 1,
  where base is the first session ≥ signal_date (= same-day close for a
  trading-day signal). Close-to-close, no spread/commission — IC measures
  ranking skill, not net P&L.
* Zero scores mean "no view / method disabled" and are EXCLUDED from that
  method's IC and hit rate.

Usage:  python -m src.analysis.signal_panel [--horizons 1,5,10] [--days 90]
                                            [--min-n 20] [--dedupe last|all]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date, timedelta
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.db.schema import (SIGNAL_METHOD_COLUMNS, SIGNAL_TIMEFRAME_COLUMNS,
                           SIGNAL_FUNDAMENTAL_COLUMNS)
from src.signals.method_epochs import epoch_for as _epoch_for

# Scored columns the IC report covers: every per-method column plus the
# aggregator's weighted combined score (the "all methods together" row) and the
# buy/sell split sides (2026-07-22). The sides are persisted as MAGNITUDES
# (combined_buy_score / combined_sell_score, both >= 0); build_panel derives the
# signed evaluation columns below so the standard sign-convention machinery
# (hit = directional, simret = sign(score)×fwd) applies unchanged: cmb_buy is
# the bullish camp's conviction as-is, cmb_sell is the bearish camp's conviction
# NEGATED (a bearish view in the standard signed convention).
PANEL_SIDE_EVAL_COLUMNS = ["cmb_buy", "cmb_sell"]
PANEL_SCORE_COLUMNS = (list(SIGNAL_METHOD_COLUMNS) + ["combined_score"]
                       + PANEL_SIDE_EVAL_COLUMNS)


# ── IC categories (the dashboard's section grouping) ───────────────────────
# The OHLCV methods are split by candle size; everything else (news, sentiment,
# smart money, options, PEAD, catalysts…) keeps using the most-recent data and
# lands in "Other". The 8 daily-technical method names are exactly those that
# have a 30-min variant — derived from the schema so there's one source of truth.
IC_CATEGORY_30M = "Technical · 30-min"
IC_CATEGORY_1D = "Technical · Daily"
IC_CATEGORY_1W = "Technical · Weekly"
IC_CATEGORY_FUND = "Fundamentals & corporate actions (Massive: value · quality · growth · short · split · dividend)"
IC_CATEGORY_OTHER = "Other (news · sentiment · smart money · options · catalysts)"
IC_CATEGORY_ORDER = (IC_CATEGORY_30M, IC_CATEGORY_1D, IC_CATEGORY_1W,
                     IC_CATEGORY_FUND, IC_CATEGORY_OTHER)

_DAILY_TECHNICAL = frozenset(c[:-4] for c in SIGNAL_TIMEFRAME_COLUMNS if c.endswith("_30m"))
_FUNDAMENTAL_SET = frozenset(SIGNAL_FUNDAMENTAL_COLUMNS)


def fwd_col(h) -> tuple:
    """Horizon token -> ``(forward-return column, key suffix)``.

    The canonical resolution every panel-reading evaluation shares (2026-08-13
    standardization directive): ``"pv"`` — the H/L signed-pivot pseudo-horizon,
    the decision basis — maps to ``("fwd_ret_pivot", "pv")``; an int ``h`` maps
    to ``(f"fwd_ret_{h}d", f"{h}d")``, the fixed monitoring grid."""
    return ("fwd_ret_pivot", "pv") if h == "pv" else (f"fwd_ret_{h}d", f"{h}d")


def int_horizons(horizons: Sequence) -> list:
    """The fixed-day subset of a horizon list — what ``build_panel(horizons=)``
    accepts (the pivot label rides every panel unconditionally, so ``"pv"``
    never needs to reach it)."""
    ints = [h for h in horizons if h != "pv"]
    return ints or [5]


def category_for(method: str) -> str:
    """Map a method/score column to its IC category."""
    if method.endswith("_30m"):
        return IC_CATEGORY_30M
    if method.endswith("_1w"):
        return IC_CATEGORY_1W
    if method in _DAILY_TECHNICAL:
        return IC_CATEGORY_1D
    if method in _FUNDAMENTAL_SET:
        return IC_CATEGORY_FUND
    return IC_CATEGORY_OTHER


def _close_series(ticker: str) -> dict:
    """date → close for one ticker, from the OHLCV cache. Module-level seam so
    tests can monkeypatch it (mirrors tests' fake_closes pattern)."""
    from src.performance.daily_nav import _load_close_series
    return _load_close_series(ticker)


def refresh_panel_ohlcv(tickers: Sequence[str], max_tickers: Optional[int] = None) -> int:
    """Force-refresh the OHLCV cache for each panel ticker so the forward closes
    the IC join needs actually exist.

    The cache is otherwise only warmed incidentally by a running pipeline, so a
    signal_date's forward bars are frequently missing (the last tick of the day
    runs pre-close and `_drop_forming_bar` drops the forming bar — observed: cache
    frozen at the signal day, every forward return NaN, every IC `n=0`). This
    decouples measurement from cache warmth. Bounded + fail-soft; offline use only
    (writes the file cache, never the DB). Returns the count successfully warmed."""
    from src.data.market_data import get_history
    uniq = list(dict.fromkeys(t for t in tickers if t))
    if max_tickers and max_tickers > 0:
        uniq = uniq[:max_tickers]
    warmed = 0
    for i, tk in enumerate(uniq, 1):
        try:
            df = get_history(tk, force_refresh=True)
            if df is not None and not df.empty:
                warmed += 1
        except Exception:
            pass
        if i % 50 == 0:
            print(f"  …OHLCV refresh {i}/{len(uniq)}")
    print(f"OHLCV refresh: warmed {warmed}/{len(uniq)} panel tickers")
    return warmed


def _load_signals(days: Optional[int], dedupe: Optional[str] = None) -> pd.DataFrame:
    """Rows from the ``signals`` table, optionally windowed by ``days``.

    When ``dedupe == "last"`` the last-row-per-(signal_date, ticker) reduction is
    pushed into DuckDB as a ``row_number()`` window instead of being done in
    pandas afterwards. The panel keeps only ~12k of 275k rows (96% discarded), so
    doing it in SQL means the other 263k never cross into Python. Falls back to
    loading everything and letting ``build_panel`` dedupe if the window query
    fails — an optimisation, never a behaviour change.
    """
    from src.db import repo
    from src.analysis.asof import current_asof
    where, params = "", []
    if days:
        where = " WHERE signal_date >= ?"
        params = [(date.today() - timedelta(days=days)).isoformat()]
    # Point-in-time cutoff (walk-forward). Applied HERE, at the choke point, so
    # every panel consumer is restricted without knowing the mechanism exists.
    _asof = current_asof()
    if _asof:
        where += (" AND " if where else " WHERE ") + f"signal_date < '{_asof}'"
    if dedupe == "last":
        try:
            return repo.fetch_df(
                "SELECT * EXCLUDE (_rn) FROM ("
                "  SELECT *, row_number() OVER ("
                "    PARTITION BY signal_date, ticker ORDER BY generated_at DESC"
                "  ) AS _rn FROM signals" + where +
                ") WHERE _rn = 1 ORDER BY generated_at", params)
        except Exception as e:
            logger.debug(f"[signal_panel] SQL dedupe unavailable ({e}) — loading all rows")
    try:
        return repo.fetch_df("SELECT * FROM signals" + where + " ORDER BY generated_at", params)
    except Exception as e:
        # DB missing or table not created yet (it appears on the first pipeline
        # run after the schema gained the signals table) — report, don't crash.
        print(f"Could not read the signals table ({e}).\n"
              "It is created automatically on the next pipeline run; rows accumulate per run.")
        return pd.DataFrame()


# ── panel memo ──────────────────────────────────────────────────────────────
_PANEL_CACHE: dict = {}
_PANEL_VER: dict = {"ts": 0.0, "val": None}
_PANEL_VER_TTL = 15.0


def _panel_version() -> Optional[str]:
    """Latest run_id — the panel's data version. The panel derives from the
    ``signals`` table, which only changes when a run persists."""
    import time as _t
    now = _t.time()
    if (now - _PANEL_VER["ts"]) < _PANEL_VER_TTL:
        return _PANEL_VER["val"]
    try:
        from src.db import repo
        d = repo.fetch_df("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")
        val = None if d is None or d.empty else str(d.iloc[0]["run_id"])
    except Exception:
        val = _PANEL_VER["val"]
    _PANEL_VER.update(ts=now, val=val)
    return val


def reset_panel_cache() -> None:
    """Drop the memoised panels (tests / forced refresh)."""
    _PANEL_CACHE.clear()
    _PANEL_VER.update(ts=0.0, val=None)


def build_panel(horizons: Sequence[int] = (1, 5, 10), days: Optional[int] = None,
                dedupe: str = "last", signals_df: Optional[pd.DataFrame] = None,
                ) -> pd.DataFrame:
    """The signals table joined with forward returns: one row per
    (signal_date, ticker), plus a ``fwd_ret_<h>d`` column per horizon (in %,
    NaN where the OHLCV cache doesn't yet reach signal_date + h sessions).

    Memoised per (horizons, days, dedupe) against the latest run_id: a dashboard
    warm sweep asked for the IDENTICAL panel 6 times out of 9 calls (~19s of
    rebuilds), and the analysis modules that share it — predictability,
    price_volume_perf, horizon_edge, policy_eval, source_performance,
    confidence_components — each built their own. Callers get a COPY (25.8 MB
    panel, ~4.5ms to copy versus ~6.1s to rebuild — 1,355x), so nobody can
    corrupt a shared frame. An explicit ``signals_df`` is never cached.
    """
    _key = None
    if signals_df is None:
        _key = (tuple(horizons), days, dedupe, _panel_version())
        _hit = _PANEL_CACHE.get(_key)
        if _hit is not None:
            return _hit.copy()

    _sql_deduped = False
    if signals_df is not None:
        df = signals_df
    else:
        df = _load_signals(days, dedupe=dedupe)
        _sql_deduped = dedupe == "last"
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    if dedupe == "last" and not _sql_deduped and "generated_at" in df.columns:
        df = (df.sort_values("generated_at")
                .groupby(["signal_date", "ticker"], as_index=False).tail(1))

    # Signed evaluation columns for the buy/sell split sides (see
    # PANEL_SIDE_EVAL_COLUMNS): buy side as-is, sell side negated so its sign
    # says "bearish view" in the standard convention. Rows persisted before the
    # split (NaN columns) simply contribute no views.
    if "combined_buy_score" in df.columns:
        df["cmb_buy"] = pd.to_numeric(df["combined_buy_score"], errors="coerce")
    if "combined_sell_score" in df.columns:
        df["cmb_sell"] = -pd.to_numeric(df["combined_sell_score"], errors="coerce")

    # ── Scorer-epoch masking (2026-07-24) ────────────────────────────────────
    # Blank out stored values that a SUPERSEDED implementation produced, so no
    # analysis can silently attribute them to the scorer that exists today.
    # Applied HERE, at the single entry point every panel consumer goes
    # through (predictability, horizon_edge, policy_eval, scorecard,
    # source_performance, confidence_components, price_volume_perf, the
    # IC table, …), rather than in each of them — a new analysis is then
    # protected by default instead of having to remember the registry.
    # Masked → NaN, which every consumer already treats as "no view".
    # The rows themselves are KEPT: their forward returns, prices and every
    # unchanged method column remain valid evidence.
    # Restore-before-mask: where a method is faithfully replayable from the
    # cached OHLCV, a superseded score is REGENERATED by the current scorer
    # rather than blanked. The mask below then only blanks what could not be
    # restored, so it keeps its correctness guarantee while discarding far less
    # evidence (money_flow alone: 98% of its rows were being dropped).
    # Fail-soft — an absent/stale `signals_replay` just leaves the mask to run.
    _restored = {}
    if getattr(settings, "enable_panel_replay_restore", True):
        try:
            from src.analysis.replay import restore_replayed
            df, _restored = restore_replayed(df)
        except Exception as _e:
            logger.debug(f"[panel] replay restore unavailable: {_e}")
    if _restored:
        logger.info("[panel] replayed with current scorers: "
                    + ", ".join(f"{k}({int(v.sum())})" for k, v in sorted(_restored.items())))

    _masked = []
    for _col in list(df.columns):
        _ep = _epoch_for(_col.split("_30m")[0].split("_1w")[0])
        if _ep is None or _col not in df.columns:
            continue
        _pre = df["signal_date"].astype(str) < _ep.isoformat()
        # A replayed cell already holds the CURRENT scorer's value, so it is not
        # superseded any more and must survive the mask — otherwise the restore
        # above is undone and the panel is exactly where it started.
        _rp = _restored.get(_col)
        if _rp is not None:
            _pre = _pre & ~_rp.reindex(_pre.index, fill_value=False).values
        if _pre.any():
            df.loc[_pre, _col] = float("nan")
            _masked.append(f"{_col}({int(_pre.sum())})")
    # Confidence epoch — same contract, different registry. The confidence
    # column and its six components mix formulas across the panel's life, so
    # pre-epoch values are blanked rather than compared with post-epoch ones.
    try:
        from src.signals.method_epochs import confidence_epoch, CONFIDENCE_EPOCH_COLUMNS
        _cep = confidence_epoch()
        if _cep is not None:
            _pre = df["signal_date"].astype(str) < _cep.isoformat()
            if _pre.any():
                for _c in CONFIDENCE_EPOCH_COLUMNS:
                    if _c not in df.columns:
                        continue
                    # movement_factor is the one component that does NOT depend
                    # on the weights, so the replay can genuinely recover it.
                    # A recovered cell carries the CURRENT formula and must
                    # survive this mask, exactly as for the method columns —
                    # otherwise the restore is silently undone.
                    _cm = _pre
                    _crp = _restored.get(_c)
                    if _crp is not None:
                        _cm = _pre & ~_crp.reindex(_pre.index, fill_value=False).values
                    if _cm.any():
                        df.loc[_cm, _c] = float("nan")
                _masked.append(f"confidence+components({int(_pre.sum())})")
    except Exception as _e:
        logger.debug(f"[signal_panel] confidence epoch unavailable: {_e}")

    if _masked:
        logger.debug(f"[signal_panel] scorer-epoch masked: {', '.join(_masked)}")

    df["_sig_date"] = df["signal_date"].map(date.fromisoformat)

    # One cache read per ticker, shared across all its rows/horizons.
    closes_by_ticker: dict = {}
    for tk in df["ticker"].unique():
        try:
            closes_by_ticker[tk] = _close_series(tk) or {}
        except Exception:
            closes_by_ticker[tk] = {}
    dates_by_ticker = {tk: sorted(c.keys()) for tk, c in closes_by_ticker.items()}

    # Point-in-time horizon guard (walk-forward). A forward return is only
    # KNOWN once its end bar has printed: at cutoff D a row dated D-1 has no
    # 5-day return, only rows dated <= D-5 do. Without this the cutoff filters
    # signal_date but the forward return still reaches past it — the calibration
    # would be fitted on outcomes that had not happened, which is exactly the
    # look-ahead walk-forward exists to remove, and it is invisible because the
    # ROW looks correctly dated.
    from src.analysis.asof import current_asof as _cur_asof
    _asof_s = _cur_asof()
    _asof_d = date.fromisoformat(_asof_s) if _asof_s else None

    def fwd(row, h: int) -> Optional[float]:
        dates = dates_by_ticker.get(row["ticker"]) or []
        closes = closes_by_ticker[row["ticker"]]
        i = bisect_left(dates, row["_sig_date"])
        if i >= len(dates) or i + h >= len(dates):
            return None
        if _asof_d is not None and dates[i + h] >= _asof_d:
            return None                 # end bar has not printed by the cutoff
        base = closes[dates[i]]
        if not base or base <= 0:
            return None
        return (closes[dates[i + h]] / base - 1.0) * 100.0

    for h in horizons:
        df[f"fwd_ret_{h}d"] = df.apply(lambda r: fwd(r, h), axis=1)

    # PIVOT label (2026-08-12, user directive): the signed % return to the next
    # pivot (`pivot_target.next_pivot_targets` — the ml_ohlcv-v2 training
    # target) as `fwd_ret_pivot`, with `end_date_pivot` for walk-forward
    # consumers (a variable-horizon label settles at its OWN end date, so
    # "label printed" cutoffs must read the row's end, not a fixed offset).
    # Point-in-time by TRUNCATION at the as-of cutoff: a pivot settles only if
    # its CONFIRMING bar (pivot+1) is inside the visible window — one bar
    # stricter than the fixed-horizon guard above, matching the label's real
    # information timing. Unsettled rows stay NaN (never backfilled).
    try:
        import numpy as _np

        from src.analysis.pivot_target import next_pivot_targets
        from src.data.cache import load_ohlcv as _load_hl
        pv_ret = pd.Series(float("nan"), index=df.index)
        pv_end = pd.Series(None, index=df.index, dtype=object)
        for tk, ridx in df.groupby("ticker").groups.items():
            dts = dates_by_ticker.get(tk) or []
            if _asof_d is not None:
                dts = dts[:bisect_left(dts, _asof_d)]
            if len(dts) < 50:
                continue
            closes = closes_by_ticker[tk]
            c = _np.asarray([closes[d] for d in dts], dtype=float)
            # H/L basis (2026-08-12): marks live on each bar's high/low; a
            # missing frame degrades to closes-as-extremes rather than dropping
            # the ticker's label.
            harr = larr = c
            try:
                _f = _load_hl(tk)
                if _f is not None and not _f.empty and "High" in _f.columns:
                    _idx = pd.DatetimeIndex(_f.index)
                    _hv = pd.to_numeric(_f["High"], errors="coerce").to_numpy(dtype=float)
                    _lv = pd.to_numeric(_f["Low"], errors="coerce").to_numpy(dtype=float)
                    _hm = {t.date(): v for t, v in zip(_idx, _hv) if v == v}
                    _lm = {t.date(): v for t, v in zip(_idx, _lv) if v == v}
                    harr = _np.asarray([_hm.get(d, closes[d]) for d in dts], dtype=float)
                    larr = _np.asarray([_lm.get(d, closes[d]) for d in dts], dtype=float)
            except Exception:
                harr = larr = c
            sp, end = next_pivot_targets(c, harr, larr)
            pos = _np.searchsorted(_np.array(dts), df.loc[ridx, "_sig_date"].to_numpy())
            ok = pos < len(dts)
            for r, p, k in zip(ridx, pos, ok):
                if k and end[p] >= 0:
                    pv_ret.at[r] = float(sp[p])
                    pv_end.at[r] = dts[int(end[p])].isoformat()
        df["fwd_ret_pivot"] = pv_ret
        df["end_date_pivot"] = pv_end
    except Exception as _e:
        logger.debug(f"[signal_panel] pivot label unavailable: {_e}")

    out = df.drop(columns=["_sig_date"])
    if _key is not None:
        # One panel per (args, run) — keep only the newest few so a long-lived
        # dashboard can't accumulate a frame per pipeline run (~26 MB each).
        _PANEL_CACHE[_key] = out
        while len(_PANEL_CACHE) > 4:
            _PANEL_CACHE.pop(next(iter(_PANEL_CACHE)))
        return out.copy()
    return out


def _spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Spearman rank correlation = Pearson on average-tie ranks (no scipy).

    ``np.errstate`` silences the divide/invalid FloatingPointError machinery for
    the degenerate case (a constant vector → corrcoef divides by a zero std).
    That is not cosmetic: every such call otherwise emits a RuntimeWarning to
    STDERR, and the 2026-08-08/09 scheduler freezes were exactly this — the
    nightly rescore's walk-forward re-calibration ran thousands of per-day
    Spearmans, the warning flood filled an undrained stderr pipe (~64KB), and
    from then on EVERY thread that touched stderr (numpy warnings, loguru's
    console sink) blocked forever: whole-process wedge, zero CPU, silent log.
    The NaN result for a degenerate day is already the documented contract
    ("no verdict"), so nothing is lost by not announcing it per call.
    """
    if len(a) < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = a.rank().corr(b.rank())
    return None if pd.isna(ic) else float(ic)


# ── Session filtering (shared by the simulated/exit panels) ──────────────────

def session_of_ts(series: pd.Series) -> pd.Series:
    """Fine US-market session (``rth|premarket|afterhours|overnight``) of each
    ISO timestamp, vectorized. The panel writers store tz-aware UTC ISO strings;
    a rare naive value is assumed UTC. Unparseable → "" (matches no filter).
    Boundaries mirror ``tracker._session_of_iso_fine``."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert("America/New_York")
    except Exception:
        return pd.Series([""] * len(series), index=series.index)
    mins = dt.dt.hour * 60 + dt.dt.minute
    out = pd.Series("overnight", index=series.index, dtype=object)
    out[(mins >= 9 * 60 + 30) & (mins < 16 * 60)] = "rth"
    out[(mins >= 4 * 60) & (mins < 9 * 60 + 30)] = "premarket"
    out[(mins >= 16 * 60) & (mins < 20 * 60)] = "afterhours"
    out[dt.isna()] = ""
    return out


def session_filter_mask(ts_series: pd.Series, session: Optional[str]) -> pd.Series:
    """Boolean mask selecting rows whose timestamp falls in *session* — the fine
    dashboard values (``rth|premarket|afterhours|overnight``) plus the coarse
    ``extended`` (= premarket ∪ afterhours). None/empty → all rows."""
    if not session:
        return pd.Series(True, index=ts_series.index)
    sess = session_of_ts(ts_series)
    if session == "extended":
        return sess.isin(("premarket", "afterhours"))
    return sess == session


def periodic_ic_stats(days: Sequence, scores: Sequence, fwd: Sequence,
                      min_per_day: int = 5, min_days: int = 3,
                      ) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """``(mean, std, icir, n_days)`` over the PER-DAY Spearman IC — the IC's confidence.

    The three inputs are aligned sequences for ONE method × horizon, already
    filtered to non-zero scores that have a forward return. A Spearman IC is
    computed within every signal-day carrying ≥ ``min_per_day`` joint
    observations; the sample stdev (``ddof=1``) and ``icir = mean / std`` are then
    taken ACROSS those daily ICs. Each DAY counts once, so the dispersion measures
    day-to-day stability — it is NOT inflated the way a standard error off the
    pooled stock-day ``n`` would be (same-day names share market/sector moves, so
    they are not independent draws). ``icir`` is the standard information-ratio
    reliability score (|ICIR| ≳ 0.5 is a stable signal, ≈ 0 is noise).

    Returns ``(None, None, None, n_days)`` when fewer than ``max(2, min_days)``
    usable daily ICs exist (a stdev needs ≥ 2 points; the floor stops a 1–2-day
    estimate from masquerading as real), or when the daily ICs are degenerate
    (std ≈ 0 ⇒ ``icir`` is None but std is still reported)."""
    floor = max(2, int(min_days))
    tmp = pd.DataFrame({"d": list(days), "s": list(scores), "f": list(fwd)})
    ics: list = []
    for _, g in tmp.groupby("d", sort=True):
        if len(g) < min_per_day:
            continue
        ic = _spearman(g["s"], g["f"])
        if ic is not None:
            ics.append(ic)
    n_days = len(ics)
    if n_days < floor:
        return None, None, None, n_days
    ser = pd.Series(ics)
    std = float(ser.std(ddof=1))
    mean = float(ser.mean())
    icir = (mean / std) if std > 1e-12 else None
    return mean, std, icir, n_days


def compute_ic(panel: pd.DataFrame, horizons: Sequence[int] = (1, 5, 10),
               min_n: int = 20, min_per_day: int = 5, min_days: int = 3,
               side: Optional[str] = None) -> pd.DataFrame:
    """Per-method IC table: for each score column × horizon, the observation
    count ``n_<h>d``, Spearman ``ic_<h>d``, directional ``hit_<h>d`` (the
    **simulated win rate** — % of non-zero scores whose sign matched the forward
    return's), and ``simret_<h>d`` (the **simulated return** — mean of
    ``sign(score) × forward_return`` over the same rows, i.e. the gross P&L if
    that method alone had decided the trade direction). It also reports the IC's
    confidence — ``icstd_<h>d`` (stdev of the per-day IC) and ``icir_<h>d`` (its
    information ratio, ``mean / std`` of the per-day IC; see ``periodic_ic_stats``)
    — which populate only once ``min_days`` signal-days each carrying
    ``min_per_day`` names have accrued. Methods with fewer than ``min_n`` joint
    observations report NaN. Sorted by |IC| at the longest horizon. Each row is
    also tagged with its ``category``.

    ``side`` (2026-07-22) restricts every metric to ONE side of each method's
    calls: ``"buy"`` = positive scores only (the method's bullish calls — hit
    becomes P(fwd>0), simret the long-only P&L, IC the ranking skill WITHIN its
    bullish calls), ``"sell"`` = negative scores only (bearish calls — hit =
    P(fwd<0), simret the short-only P&L; a PREDICTIVE sell side shows a
    POSITIVE IC there too, since a more-negative score should rank a
    more-negative return). None = both (the historical behavior). The BUY-vs-
    SELL forensics found method skill is heavily side-dependent (e.g. news:
    all edge on the negative side), so the dashboard renders all three views."""
    rows = []
    for method in PANEL_SCORE_COLUMNS:
        if method not in panel.columns:
            continue
        s_all = pd.to_numeric(panel[method], errors="coerce")
        # Values from a superseded scorer are already NaN — build_panel masks
        # them once, centrally (see the scorer-epoch block there), so they drop
        # out of has_view here without a second check.
        has_view = s_all.notna() & (s_all.abs() > 1e-12)
        if side == "buy":
            has_view &= s_all > 0
        elif side == "sell":
            has_view &= s_all < 0
        row: dict = {"method": method, "category": category_for(method),
                     "views": int(has_view.sum())}
        # The PIVOT pseudo-horizon (2026-08-12) rides the same block under the
        # "pv" suffix whenever the panel carries its label — the decision-basis
        # IC beside the fixed monitoring grid.
        h_list = list(horizons) + (["pv"] if "fwd_ret_pivot" in panel.columns else [])
        for h in h_list:
            col, sfx = fwd_col(h)
            f_all = pd.to_numeric(panel.get(col), errors="coerce")
            valid = has_view & f_all.notna()
            n = int(valid.sum())
            row[f"n_{sfx}"] = n
            if n < min_n:
                row[f"ic_{sfx}"] = None
                row[f"icstd_{sfx}"] = None
                row[f"icir_{sfx}"] = None
                row[f"icdays_{sfx}"] = 0
                row[f"hit_{sfx}"] = None
                row[f"simret_{sfx}"] = None
                continue
            s, f = s_all[valid], f_all[valid]
            row[f"ic_{sfx}"] = _spearman(s, f)
            # Confidence: stdev + information-ratio of the PER-DAY IC (each day one
            # observation → not inflated by same-day cross-sectional correlation).
            # ``icdays`` = how many signal-days backed it (the evidence the IC-weight
            # shrinkage uses; a column the dashboard ignores).
            _, ic_std, icir, ic_days = periodic_ic_stats(
                panel.loc[valid, "signal_date"], s, f, min_per_day, min_days)
            row[f"icstd_{sfx}"] = round(ic_std, 4) if ic_std is not None else None
            row[f"icir_{sfx}"] = round(icir, 3) if icir is not None else None
            row[f"icdays_{sfx}"] = int(ic_days)
            moved = f != 0
            row[f"hit_{sfx}"] = (float(((s > 0) == (f > 0))[moved].mean() * 100)
                                 if moved.any() else None)
            # Simulated solo return: trade the SIGN of the score, hold to horizon.
            signed = f.where(s > 0, -f)        # +f when score>0 (long), −f when score<0 (short)
            row[f"simret_{sfx}"] = round(float(signed.mean()), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_col = f"ic_{max(horizons)}d"
    return (out.assign(_abs=pd.to_numeric(out[sort_col], errors="coerce").abs())
               .sort_values("_abs", ascending=False, na_position="last")
               .drop(columns="_abs").reset_index(drop=True))


def print_report(panel: pd.DataFrame, ic: pd.DataFrame,
                 horizons: Sequence[int]) -> None:
    try:
        from src.performance.tracker import METHOD_LABELS
    except Exception:
        METHOD_LABELS = {}
    if panel.empty:
        print("Signals panel is empty — run the pipeline first; rows accumulate per run.")
        return
    days = sorted(panel["signal_date"].unique())
    print(f"\nSignals panel — {len(panel)} rows · {panel['ticker'].nunique()} tickers · "
          f"{days[0]} → {days[-1]} ({len(days)} signal day(s))")
    print("IC = Spearman(score, forward close-to-close return); zero scores excluded.  "
          "ICsd/ICIR = stdev & info-ratio of the per-day IC (the IC's reliability).  "
          "win = simulated solo win rate (sign-agreement %); sim = simulated solo "
          "return % (mean sign(score)×fwd_ret).\n")
    head = f"{'method':<34}{'views':>7}"
    for h in horizons:
        head += (f"{f'n@{h}':>7}{f'IC@{h}':>8}{f'ICsd@{h}':>9}{f'ICIR@{h}':>8}"
                 f"{f'win@{h}':>8}{f'sim@{h}':>9}")
    width = len(head)

    def _emit_rows(subset: pd.DataFrame) -> None:
        for _, r in subset.iterrows():
            label = METHOD_LABELS.get(r["method"], r["method"])
            line = f"{label:<34}{int(r['views']):>7}"
            for h in horizons:
                n, icv, hit, sim = (r[f"n_{h}d"], r[f"ic_{h}d"],
                                    r[f"hit_{h}d"], r[f"simret_{h}d"])
                icsd, icir = r.get(f"icstd_{h}d"), r.get(f"icir_{h}d")
                line += f"{int(n):>7}"
                line += f"{icv:>+8.3f}" if pd.notna(icv) else f"{'—':>8}"
                line += f"{icsd:>9.3f}" if pd.notna(icsd) else f"{'—':>9}"
                line += f"{icir:>+8.2f}" if pd.notna(icir) else f"{'—':>8}"
                line += f"{hit:>7.1f}%" if pd.notna(hit) else f"{'—':>8}"
                line += f"{sim:>+9.2f}" if pd.notna(sim) else f"{'—':>9}"
            print(line)

    has_cat = "category" in ic.columns
    for category in IC_CATEGORY_ORDER:
        subset = ic[ic["category"] == category] if has_cat else ic
        if subset.empty:
            continue
        print(f"\n{category}")
        print(head)
        print("-" * width)
        _emit_rows(subset)
        if not has_cat:
            break

    print("\nA well-behaved method shows IC > 0 that persists across horizons; "
          "IC ≈ 0 on a large n means the method adds noise. The 3 technical "
          "categories are the SAME indicators on 30-min / daily / weekly candles. "
          "n grows every run — judge nothing on a thin panel.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: render → glyphs
    except Exception:
        pass
    from src.db import repo
    repo.set_read_only(True)   # never contend with a running scheduler's write lock
    p = argparse.ArgumentParser(description="Per-method IC report over the persisted signals panel.")
    p.add_argument("--horizons", default="1,5,10",
                   help="comma-separated forward horizons in sessions (default 1,5,10)")
    p.add_argument("--days", type=int, default=None,
                   help="only signals from the last N calendar days (default: all)")
    p.add_argument("--min-n", type=int, default=20,
                   help="minimum joint observations before an IC is reported (default 20)")
    p.add_argument("--min-per-day", type=int, default=5,
                   help="min cross-section per signal-day before that day's IC counts toward ICstd/ICIR (default 5)")
    p.add_argument("--min-days", type=int, default=3,
                   help="min usable signal-days before IC stdev/ICIR is reported (default 3)")
    p.add_argument("--dedupe", choices=("last", "all"), default="last",
                   help="'last' keeps one row per (day, ticker) — the day's final run (default)")
    p.add_argument("--refresh", action="store_true",
                   help="force-refresh OHLCV for panel tickers first so forward returns exist "
                        "(slow — one fetch per ticker; offline, writes the file cache only)")
    p.add_argument("--refresh-max", type=int, default=0,
                   help="cap the number of tickers refreshed by --refresh (0 = all)")
    args = p.parse_args(list(argv) if argv is not None else None)
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())

    signals_df = _load_signals(args.days)
    if args.refresh and signals_df is not None and not signals_df.empty:
        refresh_panel_ohlcv(signals_df["ticker"].unique().tolist(),
                            max_tickers=args.refresh_max or None)
    panel = build_panel(horizons=horizons, days=args.days, dedupe=args.dedupe,
                        signals_df=signals_df)
    ic = (compute_ic(panel, horizons=horizons, min_n=args.min_n,
                     min_per_day=args.min_per_day, min_days=args.min_days)
          if not panel.empty else pd.DataFrame())
    print_report(panel, ic, horizons)


if __name__ == "__main__":
    main()
