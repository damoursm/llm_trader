"""Per-method simulated-trade performance over ALL scored tickers.

The ``simulated_trades`` table (a long-format reshape of ``signals``, written
every run by the pipeline) records, for each ``(run, ticker, method)`` with a
non-zero score, the side that method ALONE implied — BUY when score>0, SELL when
score<0. This module joins those rows against forward returns from the OHLCV
cache and reports each method's **directional win rate** + **mean (gross)
directional return** at several horizons.

It is the unbiased answer to "is this method predictive of direction?": every
scored ticker counts, even when the synthesized recommendation went the other
way (or produced no trade at all) — so a method's accuracy is measured on
thousands of observations instead of the few dozen the trading gates let
through.

Conventions
-----------
* Forward returns are **GROSS** close-to-close (no spread/commission): the
  question is predictiveness, not net P&L.
* Forward returns are read from the **OHLCV cache close** at ``entry + horizon``
  (deterministic, same basis as the IC panel) — NOT the stored ``entry_price``.
* Horizons: ``30m / 3h / 6h / 1d / 3d / 1w / 2w / 1m``. The daily horizons step N
  trading sessions on the daily cache; the intraday horizons (``30m``/``3h``/``6h``)
  step 1/6/12 bars on the 30-min cache — trading-time, skipping overnight gaps,
  so a late-session entry's 6h mark lands in the next session (best effort: that
  cache is yfinance ≤60d and patchy, so the longer intraday steps thin out).
* ``dedupe="last"`` keeps one row per (signal_date, ticker, method) — the day's
  final run — so highly-autocorrelated intraday repeats don't pseudo-replicate.
* A method needs ``min_n`` joint observations at a horizon before its win/return
  is reported (NaN otherwise).

Usage:  python -m src.analysis.simulated_trades [--days 90] [--backfill] [--refresh]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.analysis.signal_panel import category_for, _spearman, periodic_ic_stats

# (label, cache interval, forward steps). Daily steps are trading sessions;
# the 30m step is one 30-minute bar. 1w=5 sessions, 2w=10, 1m=21.
HORIZONS: Tuple[Tuple[str, str, int], ...] = (
    ("30m", "30m", 1),
    ("3h", "30m", 6),     # 6 thirty-min bars ≈ 3 trading-hours
    ("6h", "30m", 12),    # 12 thirty-min bars ≈ 6 trading-hours (~one RTH session)
    ("1d", "1d", 1),
    ("3d", "1d", 3),
    ("1w", "1d", 5),
    ("2w", "1d", 10),
    ("1m", "1d", 21),
)
HORIZON_LABELS = tuple(h[0] for h in HORIZONS)


# ── data access ────────────────────────────────────────────────────────────

def load_sim_trades(days: Optional[int] = None) -> pd.DataFrame:
    """Load the simulated_trades rows (optionally only the last ``days``)."""
    from src.db import repo
    try:
        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            return repo.fetch_df(
                "SELECT * FROM simulated_trades WHERE signal_date >= ? ORDER BY generated_at",
                [cutoff])
        return repo.fetch_df("SELECT * FROM simulated_trades ORDER BY generated_at")
    except Exception as e:
        print(f"Could not read simulated_trades ({e}).\n"
              "It is populated every pipeline run; backfill the existing signals "
              "history with:  python -m src.analysis.simulated_trades --backfill")
        return pd.DataFrame()


def _daily_series(ticker: str) -> Tuple[List[date], Dict[date, float]]:
    """(sorted session dates, {date: close}) from the daily OHLCV cache."""
    from src.performance.daily_nav import _load_close_series
    closes = _load_close_series(ticker) or {}
    return sorted(closes.keys()), closes


def _intraday_series(ticker: str) -> List[Tuple[int, float]]:
    """Sorted [(epoch_ns, close)] from the 30-min OHLCV cache (best effort)."""
    from src.data.cache import load_ohlcv
    df = load_ohlcv(ticker, interval="30m")
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return []
    out: List[Tuple[int, float]] = []
    for ts, close in zip(df.index, df["Close"].tolist()):
        try:
            c = float(close)
            if c > 0:
                out.append((int(pd.Timestamp(ts).value), c))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def _fwd_daily(dates: List[date], closes: Dict[date, float],
               sig_date: date, steps: int) -> Optional[float]:
    i = bisect_left(dates, sig_date)
    if i >= len(dates) or i + steps >= len(dates):
        return None
    base = closes[dates[i]]
    if not base or base <= 0:
        return None
    return (closes[dates[i + steps]] / base - 1.0) * 100.0


def _fwd_intraday(series: List[Tuple[int, float]], generated_at: str,
                  steps: int) -> Optional[float]:
    if not series:
        return None
    try:
        entry_ns = int(pd.Timestamp(generated_at).value)
    except Exception:
        return None
    times = [t for t, _ in series]
    i = bisect_left(times, entry_ns)
    if i >= len(series) or i + steps >= len(series):
        return None
    base = series[i][1]
    if not base or base <= 0:
        return None
    return (series[i + steps][1] / base - 1.0) * 100.0


# ── core computation ───────────────────────────────────────────────────────

def extract_entry_events(df: pd.DataFrame, max_gap_days: float = 3.0) -> pd.DataFrame:
    """Reduce the per-run simulated rows to ENTRY EVENTS — the tick where a
    method NEWLY decided to enter (its first call, a sign flip, or a re-emerged
    call after ``max_gap_days`` without one). A method already in a position
    doesn't re-enter, so the run-after-run re-affirmations of a standing call
    are NOT separate trades — they were pseudo-replicating the sample and, being
    re-dedupled inside each session filter, made the per-session trade counts
    sum to more than "All sessions". Events are unique moments: sessions
    partition them exactly (All = Σ sessions)."""
    if df is None or df.empty or "generated_at" not in df.columns:
        return df
    df = df.sort_values("generated_at").copy()
    ts = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
    sign = (df["score"] > 0).astype(int) - (df["score"] < 0).astype(int)
    g = df.groupby(["ticker", "method"], sort=False)
    prev_sign = g["score"].shift().pipe(lambda s: (s > 0).astype(int) - (s < 0).astype(int))
    has_prev = g["score"].shift().notna()
    prev_ts = ts.groupby([df["ticker"], df["method"]], sort=False).shift()
    gap_days = (ts - prev_ts).dt.total_seconds() / 86400.0
    is_event = (~has_prev) | (sign != prev_sign) | (gap_days > max_gap_days)
    return df[is_event]


def compute_method_perf(days: Optional[int] = None, dedupe: str = "events",
                        min_n: int = 10, sim_df: Optional[pd.DataFrame] = None,
                        min_per_day: int = 5, min_days: int = 3,
                        session: Optional[str] = None,
                        direction: Optional[str] = None,
                        ) -> pd.DataFrame:
    """Per-method directional win rate + mean gross directional return per horizon.

    Returns one row per method with: ``method``, ``category``, ``views`` (total
    simulated ENTRY events), and for each horizon label H: ``n_H`` (joint obs),
    ``win_H`` (% of trades whose signed forward return was positive), ``ret_H``
    (mean signed forward return %, gross), ``ic_H`` (Spearman rank IC between the
    method's raw score and the forward return — ranking skill, same basis as the
    signals-panel IC table), and the IC's confidence ``icstd_H`` / ``icir_H``
    (stdev and information ratio of the PER-DAY IC; see
    ``signal_panel.periodic_ic_stats`` — populate once ``min_days`` signal-days of
    ``min_per_day`` names accrue). Win/return/IC are NaN below ``min_n``. Sorted by
    views desc.

    ``dedupe="events"`` (default): a simulated trade is the tick a method NEWLY
    called the direction (``extract_entry_events``) — one trade per call, not one
    per run/day, so the session buckets are a true partition. ``"last"`` = the
    legacy one-row-per-(day, ticker, method) convention; ``"all"`` = raw rows.

    ``session`` restricts to entries DECIDED in that US-market session
    (``rth|premarket|afterhours|overnight|extended``); ``direction``
    (``long|short``) to the side of the method's call (a positive score is its
    long call). Both filters apply AFTER event extraction — filtering first
    would manufacture phantom transitions across excluded ticks. A window
    (``days``) edge can make a pre-existing call look new at the boundary."""
    df = sim_df if sim_df is not None else load_sim_trades(days)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    if dedupe == "events":
        df = extract_entry_events(df)
    elif dedupe == "last" and "generated_at" in df.columns:
        df = (df.sort_values("generated_at")
                .groupby(["signal_date", "ticker", "method"], as_index=False).tail(1))

    if direction and "direction" in df.columns:
        want = "BUY" if str(direction).lower() in ("long", "buy") else "SELL"
        df = df[df["direction"] == want]
    if session and "generated_at" in df.columns:
        from src.analysis.signal_panel import session_filter_mask
        df = df[session_filter_mask(df["generated_at"], session)]
    if df.empty:
        return pd.DataFrame()

    df["sigd"] = df["signal_date"].map(date.fromisoformat)

    tickers = df["ticker"].unique()
    daily = {tk: _daily_series(tk) for tk in tickers}
    intra: Dict[str, List[Tuple[int, float]]] = {}

    # Forward returns depend only on (ticker, date/timestamp, steps) — not the
    # method — so memoise across the many method rows that share a ticker/run.
    daily_fwd: Dict[Tuple[str, date, int], Optional[float]] = {}
    intra_fwd: Dict[Tuple[str, str, int], Optional[float]] = {}

    # Per (method, horizon): collect the (score, forward-return) pairs so n, win
    # rate, mean signed return AND the Spearman IC all come from one source.
    acc: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: {lbl: {"s": [], "f": [], "d": []} for lbl in HORIZON_LABELS})

    for row in df.itertuples(index=False):
        tk, sc, method, sigd, gen = (row.ticker, row.score, row.method,
                                     row.sigd, row.generated_at)
        dates, closes = daily.get(tk, ([], {}))
        for lbl, interval, steps in HORIZONS:
            if interval == "30m":
                key = (tk, gen, steps)
                if key not in intra_fwd:
                    if tk not in intra:
                        intra[tk] = _intraday_series(tk)
                    intra_fwd[key] = _fwd_intraday(intra[tk], gen, steps)
                fwd = intra_fwd[key]
            else:
                key = (tk, sigd, steps)
                if key not in daily_fwd:
                    daily_fwd[key] = _fwd_daily(dates, closes, sigd, steps)
                fwd = daily_fwd[key]
            if fwd is None:
                continue
            cell = acc[method][lbl]
            cell["s"].append(sc)
            cell["f"].append(fwd)
            cell["d"].append(sigd)

    views = df.groupby("method").size().to_dict()
    rows = []
    # Rows come from the views keys, not only the accumulator: a method whose
    # only entries are too recent to have ANY forward return yet still renders
    # (views > 0, every n = 0), so a session/direction filter isolating such an
    # event can't silently drop the row and break All = Σ sessions.
    for method in dict.fromkeys(list(views) + list(acc)):
        by_h = acc.get(method) or {lbl: {"s": [], "f": [], "d": []} for lbl in HORIZON_LABELS}
        rec: dict = {"method": method, "category": category_for(method),
                     "views": int(views.get(method, 0))}
        for lbl in HORIZON_LABELS:
            s_list, f_list, d_list = by_h[lbl]["s"], by_h[lbl]["f"], by_h[lbl]["d"]
            n = len(f_list)
            rec[f"n_{lbl}"] = n
            if n >= min_n:
                # signed = forward return in the method's traded direction
                signed = [f if s > 0 else -f for s, f in zip(s_list, f_list)]
                wins = sum(1 for x in signed if x > 0)
                ic = _spearman(pd.Series(s_list), pd.Series(f_list))
                rec[f"win_{lbl}"] = round(wins / n * 100, 1)
                rec[f"ret_{lbl}"] = round(sum(signed) / n, 3)
                rec[f"ic_{lbl}"] = round(ic, 3) if ic is not None else None
                # Reliability of that IC: stdev + info-ratio of the per-day IC.
                _, ic_std, icir, _ = periodic_ic_stats(d_list, s_list, f_list,
                                                       min_per_day, min_days)
                rec[f"icstd_{lbl}"] = round(ic_std, 4) if ic_std is not None else None
                rec[f"icir_{lbl}"] = round(icir, 3) if icir is not None else None
            else:
                rec[f"win_{lbl}"] = None
                rec[f"ret_{lbl}"] = None
                rec[f"ic_{lbl}"] = None
                rec[f"icstd_{lbl}"] = None
                rec[f"icir_{lbl}"] = None
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("views", ascending=False).reset_index(drop=True)


# ── direction-conditional, market-neutral skill (for the shadow edge curve) ─

def compute_directional_perf(days: Optional[int] = None, min_n: int = 10,
                             benchmark: str = "SPY", dedupe: str = "last",
                             sim_df: Optional[pd.DataFrame] = None,
                             min_per_day: int = 5, min_days: int = 3) -> pd.DataFrame:
    """Per-(method, side) MARKET-RELATIVE skill at each horizon.

    Splits every simulated solo trade by side (``bull`` = score>0, ``bear`` =
    score<0, plus a pooled ``both``) and scores it on returns NET OF the benchmark's
    same-horizon move (``ticker_fwd − benchmark_fwd``), so market drift can't make
    one side look skilful. Per (method, side, horizon): ``n_H`` obs, ``hit_H`` =
    % of that side's calls that BEAT the market in the traded direction, ``yield_H``
    = mean market-relative return in the traded direction (%), ``ic_H`` =
    Spearman(score, market-relative return), and the IC's confidence ``icstd_H`` /
    ``icir_H`` (stdev and information-ratio of the PER-DAY market-relative IC; see
    ``signal_panel.periodic_ic_stats``). The basis for the direction-aware shadow
    edge curve — and the readout for whether a side should be INVERTED: a side whose
    ``icir_H`` is confidently negative (and stays negative across horizons) is
    reliably anti-predictive net of the benchmark. Below ``min_n`` a cell reports
    NaN; ``icstd``/``icir`` need ``min_days`` signal-days of ``min_per_day`` names."""
    df = sim_df if sim_df is not None else load_sim_trades(days)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if dedupe == "events":
        df = extract_entry_events(df)
    elif dedupe == "last" and "generated_at" in df.columns:
        df = (df.sort_values("generated_at")
                .groupby(["signal_date", "ticker", "method"], as_index=False).tail(1))
    if df.empty:
        return pd.DataFrame()
    df["sigd"] = df["signal_date"].map(date.fromisoformat)

    tickers = df["ticker"].unique()
    daily = {tk: _daily_series(tk) for tk in tickers}
    intra: Dict[str, list] = {}
    b_dates, b_closes = _daily_series(benchmark)
    b_intra = _intraday_series(benchmark)

    d_fwd: dict = {}; i_fwd: dict = {}; bd_fwd: dict = {}; bi_fwd: dict = {}
    acc: Dict[tuple, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: {lbl: {"s": [], "m": [], "d": []} for lbl in HORIZON_LABELS})

    for row in df.itertuples(index=False):
        tk, sc, method, sigd, gen = (row.ticker, row.score, row.method, row.sigd, row.generated_at)
        side = "bull" if sc > 0 else "bear"
        dts, cls = daily.get(tk, ([], {}))
        for lbl, interval, steps in HORIZONS:
            if interval == "30m":
                k = (tk, gen, steps)
                if k not in i_fwd:
                    if tk not in intra:
                        intra[tk] = _intraday_series(tk)
                    i_fwd[k] = _fwd_intraday(intra[tk], gen, steps)
                fwd = i_fwd[k]
                bk = (gen, steps)
                if bk not in bi_fwd:
                    bi_fwd[bk] = _fwd_intraday(b_intra, gen, steps)
                bfwd = bi_fwd[bk]
            else:
                k = (tk, sigd, steps)
                if k not in d_fwd:
                    d_fwd[k] = _fwd_daily(dts, cls, sigd, steps)
                fwd = d_fwd[k]
                bk = (sigd, steps)
                if bk not in bd_fwd:
                    bd_fwd[bk] = _fwd_daily(b_dates, b_closes, sigd, steps)
                bfwd = bd_fwd[bk]
            if fwd is None or bfwd is None:
                continue                      # need both legs to neutralise the market
            mktrel = fwd - bfwd
            for s in (side, "both"):
                cell = acc[(method, s)][lbl]
                cell["s"].append(sc)
                cell["m"].append(mktrel)
                cell["d"].append(sigd)

    rows = []
    for (method, side), by_h in acc.items():
        rec: dict = {"method": method, "side": side}
        for lbl in HORIZON_LABELS:
            s_list, m_list, d_list = by_h[lbl]["s"], by_h[lbl]["m"], by_h[lbl]["d"]
            n = len(m_list)
            rec[f"n_{lbl}"] = n
            if n >= min_n:
                # signed = market-relative return in the method's TRADED direction
                signed = [m if s > 0 else -m for s, m in zip(s_list, m_list)]
                wins = sum(1 for x in signed if x > 0)
                ic = _spearman(pd.Series(s_list), pd.Series(m_list))
                rec[f"hit_{lbl}"] = round(wins / n * 100, 1)
                rec[f"yield_{lbl}"] = round(sum(signed) / n, 4)
                rec[f"ic_{lbl}"] = round(ic, 3) if ic is not None else None
                # Reliability of that market-relative IC: stdev + info-ratio of the
                # per-day IC — the inversion readout (confidently negative ⇒ flip side)
                # AND the source the shadow-basis IC weighting gates on (icdays = the
                # signal-day count the confidence t-stat uses).
                _, ic_std, icir, ic_days = periodic_ic_stats(d_list, s_list, m_list,
                                                             min_per_day, min_days)
                rec[f"icstd_{lbl}"] = round(ic_std, 4) if ic_std is not None else None
                rec[f"icir_{lbl}"] = round(icir, 3) if icir is not None else None
                rec[f"icdays_{lbl}"] = int(ic_days)
            else:
                rec[f"hit_{lbl}"] = None
                rec[f"yield_{lbl}"] = None
                rec[f"ic_{lbl}"] = None
                rec[f"icstd_{lbl}"] = None
                rec[f"icir_{lbl}"] = None
                rec[f"icdays_{lbl}"] = 0
        rows.append(rec)
    return pd.DataFrame(rows)


# ── backfill (reshape existing signals → simulated_trades) ─────────────────

def backfill_from_signals() -> int:
    """Populate simulated_trades from the signals already in the DB.

    The pipeline writes both tables going forward, but this materialises the
    accumulated history in one pass so the view is useful immediately. Idempotent
    per run_id (``insert_simulated_trades`` replaces a run's rows). Returns the
    number of simulated-trade rows written."""
    from src.db import repo
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    try:
        sig = repo.fetch_df("SELECT * FROM signals ORDER BY generated_at")
    except Exception as e:
        print(f"Could not read signals ({e}). Run the pipeline first.")
        return 0
    if sig is None or sig.empty:
        print("signals table is empty — nothing to backfill.")
        return 0
    method_cols = [c for c in (list(SIGNAL_METHOD_COLUMNS) + ["combined_score"])
                   if c in sig.columns]
    total = 0
    for run_id, g in sig.groupby("run_id"):
        gen = str(g["generated_at"].iloc[0])
        sdate = str(g["signal_date"].iloc[0])
        rows = []
        for r in g.itertuples(index=False):
            px = getattr(r, "price", None)
            for m in method_cols:
                sc = getattr(r, m, None)
                if sc is None or pd.isna(sc) or abs(float(sc)) < 1e-9:
                    continue
                rows.append({"ticker": r.ticker, "method": m, "score": float(sc),
                             "direction": "BUY" if float(sc) > 0 else "SELL",
                             "entry_price": px})
        if rows:
            repo.insert_simulated_trades(run_id, generated_at=gen, signal_date=sdate, rows=rows)
            total += len(rows)
    print(f"Backfilled {total} simulated-trade row(s) across "
          f"{sig['run_id'].nunique()} run(s).")
    return total


# ── CLI report ─────────────────────────────────────────────────────────────

def print_report(perf: pd.DataFrame) -> None:
    try:
        from src.performance.tracker import METHOD_LABELS
    except Exception:
        METHOD_LABELS = {}
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined)"
    if perf is None or perf.empty:
        print("No simulated single-method trades yet — they accrue every run "
              "(or run --backfill to materialise existing signals history).")
        return
    head = f"{'method':<34}{'views':>7}"
    for lbl in HORIZON_LABELS:
        head += (f"{f'n@{lbl}':>7}{f'IC@{lbl}':>9}{f'ICsd@{lbl}':>9}{f'ICIR@{lbl}':>8}"
                 f"{f'win@{lbl}':>9}{f'ret@{lbl}':>9}")
    print("\nSimulated single-method performance — directional, GROSS, over ALL scored "
          "tickers.\nIC = Spearman(score, forward return); ICsd/ICIR = stdev & info-ratio "
          "of the per-day IC (the IC's reliability); win = % of this method's solo "
          "BUY/SELL calls that were right; ret = mean signed forward return %.\n")
    print(head)
    print("-" * len(head))
    for _, r in perf.iterrows():
        line = f"{labels.get(r['method'], r['method']):<34}{int(r['views']):>7}"
        for lbl in HORIZON_LABELS:
            n, ic, win, ret = r[f"n_{lbl}"], r[f"ic_{lbl}"], r[f"win_{lbl}"], r[f"ret_{lbl}"]
            icsd, icir = r.get(f"icstd_{lbl}"), r.get(f"icir_{lbl}")
            line += f"{int(n):>7}"
            line += f"{ic:>+9.3f}" if pd.notna(ic) else f"{'—':>9}"
            line += f"{icsd:>9.3f}" if pd.notna(icsd) else f"{'—':>9}"
            line += f"{icir:>+8.2f}" if pd.notna(icir) else f"{'—':>8}"
            line += f"{win:>8.1f}%" if pd.notna(win) else f"{'—':>9}"
            line += f"{ret:>+9.2f}" if pd.notna(ret) else f"{'—':>9}"
        print(line)
    print("\nn grows every run; 5d/10d/21d horizons need a post-close cache warm "
          "(--refresh) to populate. Judge nothing on a thin panel.")


def print_directional_report(perf: pd.DataFrame) -> None:
    """Per-(method, side) MARKET-RELATIVE skill + IC reliability — the readout for
    deciding whether a method's score should be INVERTED on a given side.

    Read it as: a side whose market-relative ICIR is confidently negative (≲ −0.5)
    AND stays negative across horizons is reliably anti-predictive net of the
    benchmark → invert that side. An ICIR ≈ 0 is noise (do NOT invert — a negative
    IC at ICIR ≈ 0 is a few bad days, not skill); a side flagged here whose live
    (absolute-return) IC is positive is the clearest beta-vs-alpha disagreement."""
    try:
        from src.performance.tracker import METHOD_LABELS
    except Exception:
        METHOD_LABELS = {}
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined)"
    if perf is None or perf.empty:
        print("\nNo directional simulated trades with forward returns yet — they accrue "
              "every run (or run --backfill to materialise existing signals history).")
        return
    perf = perf.copy()
    perf["_so"] = perf["side"].map({"bull": 0, "bear": 1, "both": 2}).fillna(3)
    perf = perf.sort_values(["method", "_so"])
    head = f"{'method':<26}{'side':<6}"
    for lbl in HORIZON_LABELS:
        head += f"{f'n@{lbl}':>7}{f'IC@{lbl}':>9}{f'ICIR@{lbl}':>8}{f'hit@{lbl}':>9}"
    print("\nDirectional (market-relative) single-method skill — net of the benchmark, "
          "split by side.\nIC = Spearman(score, market-rel return); ICIR = info-ratio of "
          "the per-day IC (reliability); hit = % of that side's calls that beat the market.\n"
          "Score convention: + = predicted UP, so a NEGATIVE IC means the score points the "
          "wrong way. INVERT a side when its ICIR is confidently NEGATIVE (<= -0.5) AND stays "
          "negative across ALL horizons (a real sign bug). A mean-reversion method that is "
          "negative at SHORT horizons but positive at longer ones is a HORIZON effect, NOT an "
          "inversion (the edge curve already flips it per-horizon); ICIR ~ 0 is noise.\n")
    print(head)
    print("-" * len(head))
    for _, r in perf.iterrows():
        line = f"{labels.get(r['method'], r['method'])[:25]:<26}{str(r['side']):<6}"
        for lbl in HORIZON_LABELS:
            n, ic, icir, hit = (r[f"n_{lbl}"], r[f"ic_{lbl}"],
                                r.get(f"icir_{lbl}"), r[f"hit_{lbl}"])
            line += f"{int(n):>7}"
            line += f"{ic:>+9.3f}" if pd.notna(ic) else f"{'—':>9}"
            line += f"{icir:>+8.2f}" if pd.notna(icir) else f"{'—':>8}"
            line += f"{hit:>8.1f}%" if pd.notna(hit) else f"{'—':>9}"
        print(line)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Per-method simulated-trade directional performance.")
    p.add_argument("--days", type=int, default=None, help="only signals from the last N days")
    p.add_argument("--min-n", type=int, default=10, help="min joint obs before win/ret is reported")
    p.add_argument("--min-per-day", type=int, default=5,
                   help="min cross-section per signal-day before that day's IC counts toward ICstd/ICIR (default 5)")
    p.add_argument("--min-days", type=int, default=3,
                   help="min usable signal-days before IC stdev/ICIR is reported (default 3)")
    p.add_argument("--dedupe", choices=("events", "last", "all"), default="events",
                   help="'events' = one trade per NEW directional call (entry-event "
                        "semantics, the dashboard's view); 'last' = one row per "
                        "(day, ticker, method) — the legacy panel convention the live "
                        "horizon/edge-curve system still pins; 'all' = raw rows")
    p.add_argument("--backfill", action="store_true",
                   help="materialise simulated_trades from existing signals, then report")
    p.add_argument("--refresh", action="store_true",
                   help="force-warm the daily OHLCV cache for panel tickers first")
    p.add_argument("--directional", action="store_true",
                   help="also print the per-(method, side) MARKET-RELATIVE skill table with "
                        "ICIR — the readout for deciding score inversion per side")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.backfill:
        backfill_from_signals()

    from src.db import repo
    repo.set_read_only(not args.backfill)  # backfill needs write; reporting is read-only

    from config import settings
    if args.refresh:
        df = load_sim_trades(args.days)
        if not df.empty:
            from src.analysis.signal_panel import refresh_panel_ohlcv
            tickers = df["ticker"].unique().tolist()
            if args.directional:
                tickers.append(settings.horizon_market_benchmark)   # the shadow needs the benchmark leg
            refresh_panel_ohlcv(tickers)

    perf = compute_method_perf(days=args.days, dedupe=args.dedupe, min_n=args.min_n,
                               min_per_day=args.min_per_day, min_days=args.min_days)
    print_report(perf)

    if args.directional:
        dperf = compute_directional_perf(days=args.days, dedupe=args.dedupe, min_n=args.min_n,
                                         benchmark=settings.horizon_market_benchmark,
                                         min_per_day=args.min_per_day, min_days=args.min_days)
        print_directional_report(dperf)


if __name__ == "__main__":
    main()
