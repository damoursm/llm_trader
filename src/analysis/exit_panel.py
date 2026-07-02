"""Exit-method performance — the exit-side mirror of ``simulated_trades``.

Every tick, the pipeline decomposes each held position's exit decision into signed
**hold-conviction** scores (``analysis.exit_methods``) and persists them long-format
to the ``exit_signals`` panel. This module joins that panel against **forward
returns** from the OHLCV cache — oriented by the *position's* direction — and
reports, per exit method per horizon, the directional win rate, mean signed
return, Spearman **IC**, and the IC's reliability (**IC std** / **ICIR**), exactly
like ``compute_method_perf`` does for the entry signals.

Orientation is the one twist vs the entry engine: an exit score is already
position-relative (``+`` = keep holding, ``−`` = exit), so the forward return is
multiplied by the position's direction (``+1`` long / ``−1`` short) before it is
paired with the score. A persistently POSITIVE IC then means the method correctly
holds winners / exits losers.

The synthesized ``llm_review`` row (the method that actually decides) is computed
from the ``trade_reviews`` table instead of ``exit_signals`` — same derivation, but
``trade_reviews`` already carries months of history, so that row is populated
immediately while the decomposed panel accrues.

Usage:  python -m src.analysis.exit_panel [--days 90]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.analysis.signal_panel import _spearman, periodic_ic_stats
from src.analysis.simulated_trades import (HORIZONS, HORIZON_LABELS, _daily_series,
                                           _fwd_daily, _intraday_series, _fwd_intraday)
from src.analysis.exit_methods import exit_category_for

MIN_N = 10


# ── data access ────────────────────────────────────────────────────────────

def _load_exit_signals(days: Optional[int] = None) -> pd.DataFrame:
    """Load the exit_signals rows (optionally only the last ``days``)."""
    from src.db import repo
    try:
        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            return repo.fetch_df(
                "SELECT * FROM exit_signals WHERE signal_date >= ? ORDER BY reviewed_at",
                [cutoff])
        return repo.fetch_df("SELECT * FROM exit_signals ORDER BY reviewed_at")
    except Exception as e:
        print(f"Could not read exit_signals ({e}).\n"
              "It is populated every pipeline run once positions are held.")
        return pd.DataFrame()


def _load_trade_reviews(days: Optional[int] = None) -> pd.DataFrame:
    """Load the trade_reviews rows (optionally only the last ``days``)."""
    from src.db import repo
    try:
        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            return repo.fetch_df(
                "SELECT * FROM trade_reviews WHERE reviewed_at >= ? ORDER BY reviewed_at",
                [cutoff])
        return repo.fetch_df("SELECT * FROM trade_reviews ORDER BY reviewed_at")
    except Exception:
        return pd.DataFrame()


def _dir_sign_of(direction) -> int:
    """+1 for a long position, −1 for a short — from a BULLISH/BEARISH or BUY/SELL string."""
    d = str(direction or "").upper()
    return 1 if ("BULL" in d or d in ("BUY", "LONG")) else -1


def extract_activation_events(df: pd.DataFrame, group_cols, ts_col: str,
                              epoch_col: Optional[str] = None) -> pd.DataFrame:
    """Reduce per-tick hold-conviction rows to ACTIVATION EVENTS — the tick where
    an exit method first turned against the position (score crossed into
    negative territory: no previous tick, previous ≥ 0, or a new position epoch
    — e.g. the shadow book's aggregate direction flipped). A method that keeps
    saying "exit" tick after tick fired ONCE; counting every tick both
    pseudo-replicated the sample and made per-session counts sum past "All
    sessions". Events are unique moments, so sessions partition them exactly.
    ``group_cols`` identifies one position × method series (held book:
    position_id+method; shadow book: ticker+method)."""
    if df is None or df.empty:
        return df
    df = df.sort_values(ts_col).copy()
    g = df.groupby(list(group_cols), sort=False)
    prev_score = g["score"].shift()
    new_epoch = pd.Series(False, index=df.index)
    if epoch_col is not None and epoch_col in df.columns:
        new_epoch = g[epoch_col].shift().ne(df[epoch_col]) & g[epoch_col].shift().notna()
    is_activation = (df["score"] < 0) & (prev_score.isna() | (prev_score >= 0) | new_epoch)
    return df[is_activation]


# ── core computation ───────────────────────────────────────────────────────

def _accumulate(df: pd.DataFrame) -> Dict[str, Dict[str, dict]]:
    """Per (method, horizon) collect the (score, direction-oriented forward return,
    day) triples. ``df`` needs columns ticker/method/score/sigd(date)/ts(str)/dir_sign."""
    acc: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: {lbl: {"s": [], "f": [], "d": []} for lbl in HORIZON_LABELS})
    tickers = df["ticker"].unique()
    daily = {tk: _daily_series(tk) for tk in tickers}
    intra: Dict[str, list] = {}
    daily_fwd: dict = {}
    intra_fwd: dict = {}

    for row in df.itertuples(index=False):
        tk, sc, method, sigd, ts, dsign = (row.ticker, row.score, row.method,
                                           row.sigd, row.ts, row.dir_sign)
        dates, closes = daily.get(tk, ([], {}))
        for lbl, interval, steps in HORIZONS:
            if interval == "30m":
                key = (tk, ts, steps)
                if key not in intra_fwd:
                    if tk not in intra:
                        intra[tk] = _intraday_series(tk)
                    intra_fwd[key] = _fwd_intraday(intra[tk], ts, steps)
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
            cell["f"].append(fwd * dsign)      # orient by the POSITION's direction
            cell["d"].append(sigd)
    return acc


def _perf_rows(acc: Dict[str, Dict[str, dict]], views: Dict[str, int],
               min_n: int, min_per_day: int, min_days: int) -> List[dict]:
    """Turn the accumulator into per-method rows (same schema as
    ``simulated_trades.compute_method_perf``: n/win/ret/ic/icstd/icir per horizon).

    Rows come from the ``views`` keys, NOT only the accumulator: a method whose
    only events are too recent to have ANY forward return yet must still render
    (views > 0, every n = 0) — otherwise a session filter that isolates such an
    event silently drops the whole row and the per-session trade counts no
    longer sum to the All-sessions view."""
    rows: List[dict] = []
    for method in dict.fromkeys(list(views) + list(acc)):
        by_h = acc.get(method) or {lbl: {"s": [], "f": [], "d": []} for lbl in HORIZON_LABELS}
        rec: dict = {"method": method, "category": exit_category_for(method),
                     "views": int(views.get(method, 0))}
        for lbl in HORIZON_LABELS:
            s_list, f_list, d_list = by_h[lbl]["s"], by_h[lbl]["f"], by_h[lbl]["d"]
            n = len(f_list)
            rec[f"n_{lbl}"] = n
            if n >= min_n:
                # f_list is ALREADY direction-oriented, so signed = the P&L in the
                # direction the method advised (keep if score>0, exit if score<0).
                signed = [f if s > 0 else -f for s, f in zip(s_list, f_list)]
                wins = sum(1 for x in signed if x > 0)
                ic = _spearman(pd.Series(s_list), pd.Series(f_list))
                rec[f"win_{lbl}"] = round(wins / n * 100, 1)
                rec[f"ret_{lbl}"] = round(sum(signed) / n, 3)
                rec[f"ic_{lbl}"] = round(ic, 3) if ic is not None else None
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
    return rows


def compute_llm_review_perf_from_reviews(days: Optional[int] = None, min_n: int = MIN_N,
                                         min_per_day: int = 5, min_days: int = 3,
                                         review_df: Optional[pd.DataFrame] = None,
                                         session: Optional[str] = None,
                                         direction: Optional[str] = None) -> Optional[dict]:
    """The ``llm_review`` row (the synthesized decider) from the ``trade_reviews``
    table: hold-conviction = ``+confidence`` when the review reaffirms the entry
    action, ``−confidence`` when it flips (HOLD/WATCH skipped), reduced to
    ACTIVATION EVENTS — the review tick where the engine first turned against the
    position — joined to the position's direction-oriented forward return.
    Returns one row dict (same schema as ``compute_exit_method_perf``) or ``None``
    when there is nothing usable. ``session`` filters by the session the
    activation FIRED in; ``direction`` (long|short) by the position's side —
    both applied after event extraction, so sessions partition the events."""
    df = review_df if review_df is not None else _load_trade_reviews(days)
    if df is None or getattr(df, "empty", True):
        return None
    from src.utils import ET
    recs: List[dict] = []
    for r in df.itertuples(index=False):
        ea = str(getattr(r, "entry_action", "") or "").upper()
        act = str(getattr(r, "action", "") or "").upper()
        conf = float(getattr(r, "confidence", 0.0) or 0.0)
        if ea not in ("BUY", "SELL") or act not in ("BUY", "SELL") or conf == 0:
            continue
        score = conf if act == ea else -conf
        try:
            ts = pd.Timestamp(getattr(r, "reviewed_at"))
            sigd = (ts.tz_convert(ET) if ts.tzinfo is not None else ts).date()
        except Exception:
            continue
        recs.append({"ticker": getattr(r, "ticker"), "method": "llm_review", "score": score,
                     "sigd": sigd, "ts": str(getattr(r, "reviewed_at")),
                     "position_id": getattr(r, "position_id", None) or getattr(r, "ticker"),
                     "dir_sign": 1 if ea == "BUY" else -1})
    if not recs:
        return None
    rdf = extract_activation_events(pd.DataFrame(recs), ("position_id",), "ts")
    if session and not rdf.empty:
        from src.analysis.signal_panel import session_filter_mask
        rdf = rdf[session_filter_mask(rdf["ts"], session)]
    if direction and not rdf.empty:
        want = 1 if str(direction).lower() in ("long", "buy") else -1
        rdf = rdf[rdf["dir_sign"] == want]
    if rdf.empty:
        return None
    acc = _accumulate(rdf)
    rows = _perf_rows(acc, {"llm_review": int(len(rdf))}, min_n, min_per_day, min_days)
    return rows[0] if rows else None


def compute_exit_method_perf(days: Optional[int] = None, min_n: int = MIN_N,
                             min_per_day: int = 5, min_days: int = 3,
                             exit_df: Optional[pd.DataFrame] = None,
                             review_df: Optional[pd.DataFrame] = None,
                             session: Optional[str] = None,
                             direction: Optional[str] = None) -> pd.DataFrame:
    """Per exit-method win rate / signed return / IC / IC-std / ICIR per horizon.

    All methods come from the ``exit_signals`` panel except ``llm_review`` (the
    synthesized decider), which is taken from ``trade_reviews`` — same derivation,
    but with the history that panel lacks on day one. Dedupes to one row per
    (signal_date, ticker, position_id, method). Empty until forward returns exist.

    A row here is an ACTIVATION EVENT — the tick a method first turned against
    the held position (``extract_activation_events``), per the exit rule
    "evaluate every subsequent tick; when the method activates, it enters the
    data in that tick's session". ``session`` restricts to activations that
    fired in that US-market session (``rth|premarket|afterhours|overnight|
    extended``); ``direction`` (``long|short``) to the held position's side.
    Filters apply AFTER event extraction, so sessions partition the events
    (All = Σ sessions)."""
    ex = exit_df if exit_df is not None else _load_exit_signals(days)
    rows: List[dict] = []
    if ex is not None and not ex.empty:
        ex = extract_activation_events(ex, ("position_id", "method"), "reviewed_at")
        if direction and not ex.empty and "entry_direction" in ex.columns:
            want = 1 if str(direction).lower() in ("long", "buy") else -1
            ex = ex[ex["entry_direction"].map(_dir_sign_of) == want]
        if session and not ex.empty and "reviewed_at" in ex.columns:
            from src.analysis.signal_panel import session_filter_mask
            ex = ex[session_filter_mask(ex["reviewed_at"], session)]
    if ex is not None and not ex.empty:
        ex = ex.copy()
        ex["sigd"] = ex["signal_date"].map(date.fromisoformat)
        ex["dir_sign"] = ex["entry_direction"].map(_dir_sign_of)
        ex["ts"] = ex["reviewed_at"]
        acc = _accumulate(ex[["ticker", "method", "score", "sigd", "ts", "dir_sign"]])
        views = ex.groupby("method").size().to_dict()
        rows = _perf_rows(acc, views, min_n, min_per_day, min_days)

    out = pd.DataFrame(rows)
    # Replace the panel's llm_review row with the history-backed trade_reviews one.
    review_row = compute_llm_review_perf_from_reviews(days, min_n, min_per_day, min_days,
                                                      review_df=review_df,
                                                      session=session, direction=direction)
    if review_row is not None:
        if not out.empty:
            out = out[out["method"] != "llm_review"]
        out = pd.concat([pd.DataFrame([review_row]), out], ignore_index=True)
    if out.empty:
        return out
    return out.sort_values("views", ascending=False).reset_index(drop=True)


# ── simulated shadow book: exit methods over ALL scored tickers ─────────────
# The position-independent exit methods can be simulated over the WHOLE universe
# (not just the held book) straight from the signals panel: treat every scored
# ticker as a hypothetical position held in its OWN aggregate direction, and score
# each method as a signed hold-conviction. This escapes the held book's selection
# bias + small sample and BACKFILLS instantly from months of stored signals.
# ``horizon`` / ``llm_review`` are NOT here — they need a real (or hypothetical)
# entry time / opener, so they exist only in the held-book view.

def shadow_exit_methods() -> tuple:
    """The exit methods simulable universe-wide: aggregator (= combined_score) +
    every entry signal method re-scored as an exit signal."""
    from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
    return ("aggregator",) + tuple(SIGNAL_BASE_METHOD_COLUMNS)


def compute_shadow_exit_method_perf(days: Optional[int] = None, min_n: int = MIN_N,
                                    min_per_day: int = 5, min_days: int = 3,
                                    signals_df: Optional[pd.DataFrame] = None,
                                    session: Optional[str] = None,
                                    direction: Optional[str] = None) -> pd.DataFrame:
    """Simulate the position-independent exit methods over ALL scored tickers.

    Reads the ``signals`` panel, treats each ticker as a hypothetical position held
    in its own aggregate ``direction``, and scores every method as a signed
    hold-conviction (method score × the ticker's dir_sign; ``aggregator`` =
    combined_score), joined to the direction-oriented forward return through the
    SAME engine as the held book. Deduped to the last run per (signal_date, ticker).
    The large-sample, selection-bias-free counterpart to ``compute_exit_method_perf``
    — but only for methods that don't need a real entry (``horizon`` / ``llm_review``
    stay held-only).

    A row here is an ACTIVATION EVENT: the run where a method's hold-conviction
    for the ticker's hypothetical position first turned negative (per
    ``extract_activation_events``; an aggregate-direction flip starts a new
    position epoch). ``session`` restricts to activations that FIRED in that
    US-market session; ``direction`` (``long|short``) to hypothetical positions
    of that side (mirroring ``_dir_sign_of``). Filters apply after event
    extraction, so sessions partition the events (All = Σ sessions)."""
    from src.analysis.signal_panel import _load_signals, session_filter_mask
    df = signals_df if signals_df is not None else _load_signals(days)
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    df = df.copy()
    # (method, source column) pairs whose column is actually present in the panel.
    pairs = [(m, "combined_score" if m == "aggregator" else m) for m in shadow_exit_methods()]
    pairs = [(m, c) for (m, c) in pairs if c in df.columns]
    meta = [c for c in ("signal_date", "ticker", "direction", "generated_at") if c in df.columns]
    value_cols = list(dict.fromkeys(c for _, c in pairs))
    long = df[meta + value_cols].melt(id_vars=meta, var_name="col", value_name="raw")
    long = long.dropna(subset=["raw"])
    long = long[long["raw"] != 0.0]                      # no view → not scored
    if long.empty:
        return pd.DataFrame()
    col2method = {c: m for m, c in pairs}
    long["method"] = long["col"].map(col2method)
    long["dir_sign"] = long["direction"].map(_dir_sign_of)
    long["score"] = long["raw"].astype(float) * long["dir_sign"]
    long["ts"] = long["generated_at"]
    long = extract_activation_events(long, ("ticker", "method"), "ts", epoch_col="dir_sign")
    if session and not long.empty:
        long = long[session_filter_mask(long["ts"], session)]
    if direction and not long.empty:
        want = 1 if str(direction).lower() in ("long", "buy") else -1
        long = long[long["dir_sign"] == want]
    if long.empty:
        return pd.DataFrame()
    date_map = {d: date.fromisoformat(d) for d in long["signal_date"].unique()}
    long["sigd"] = long["signal_date"].map(date_map)
    ldf = long[["ticker", "method", "score", "sigd", "ts", "dir_sign"]]
    acc = _accumulate(ldf)
    out = pd.DataFrame(_perf_rows(acc, ldf.groupby("method").size().to_dict(),
                                 min_n, min_per_day, min_days))
    if out.empty:
        return out
    return out.sort_values("views", ascending=False).reset_index(drop=True)


# ── CLI ────────────────────────────────────────────────────────────────────

def _main() -> None:
    ap = argparse.ArgumentParser(description="Exit-method performance over the exit_signals panel.")
    ap.add_argument("--days", type=int, default=None, help="only the last N days of signal_date")
    ap.add_argument("--min-n", type=int, default=MIN_N, help="min joint obs before a horizon reports")
    ap.add_argument("--source", choices=("held", "shadow"), default="held",
                    help="held = the real exit_signals/trade_reviews book; "
                         "shadow = simulate over ALL scored tickers (signals panel)")
    args = ap.parse_args()
    df = (compute_shadow_exit_method_perf(days=args.days, min_n=args.min_n)
          if args.source == "shadow"
          else compute_exit_method_perf(days=args.days, min_n=args.min_n))
    if df is None or df.empty:
        print("No exit-method performance yet — the exit_signals panel / trade_reviews "
              "need forward-return history (accrues every run once positions are held).")
        return
    cols = ["method", "category", "views"] + [
        f"{m}_{h}" for h in HORIZON_LABELS for m in ("n", "ic", "icir", "win", "ret")]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    _main()
