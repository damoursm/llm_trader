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

from src.analysis.signal_panel import category_for, _spearman

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

def compute_method_perf(days: Optional[int] = None, dedupe: str = "last",
                        min_n: int = 10, sim_df: Optional[pd.DataFrame] = None,
                        ) -> pd.DataFrame:
    """Per-method directional win rate + mean gross directional return per horizon.

    Returns one row per method with: ``method``, ``category``, ``views`` (total
    simulated trades), and for each horizon label H: ``n_H`` (joint obs),
    ``win_H`` (% of trades whose signed forward return was positive), ``ret_H``
    (mean signed forward return %, gross), and ``ic_H`` (Spearman rank IC between
    the method's raw score and the forward return — ranking skill, same basis as
    the signals-panel IC table). Win/return/IC are NaN below ``min_n``. Sorted by
    views desc."""
    df = sim_df if sim_df is not None else load_sim_trades(days)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    if dedupe == "last" and "generated_at" in df.columns:
        df = (df.sort_values("generated_at")
                .groupby(["signal_date", "ticker", "method"], as_index=False).tail(1))

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
        lambda: {lbl: {"s": [], "f": []} for lbl in HORIZON_LABELS})

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

    views = df.groupby("method").size().to_dict()
    rows = []
    for method, by_h in acc.items():
        rec: dict = {"method": method, "category": category_for(method),
                     "views": int(views.get(method, 0))}
        for lbl in HORIZON_LABELS:
            s_list, f_list = by_h[lbl]["s"], by_h[lbl]["f"]
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
            else:
                rec[f"win_{lbl}"] = None
                rec[f"ret_{lbl}"] = None
                rec[f"ic_{lbl}"] = None
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("views", ascending=False).reset_index(drop=True)


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
        head += f"{f'n@{lbl}':>7}{f'IC@{lbl}':>9}{f'win@{lbl}':>9}{f'ret@{lbl}':>9}"
    print("\nSimulated single-method performance — directional, GROSS, over ALL scored "
          "tickers.\nIC = Spearman(score, forward return); win = % of this method's solo "
          "BUY/SELL calls that were right; ret = mean signed forward return %.\n")
    print(head)
    print("-" * len(head))
    for _, r in perf.iterrows():
        line = f"{labels.get(r['method'], r['method']):<34}{int(r['views']):>7}"
        for lbl in HORIZON_LABELS:
            n, ic, win, ret = r[f"n_{lbl}"], r[f"ic_{lbl}"], r[f"win_{lbl}"], r[f"ret_{lbl}"]
            line += f"{int(n):>7}"
            line += f"{ic:>+9.3f}" if pd.notna(ic) else f"{'—':>9}"
            line += f"{win:>8.1f}%" if pd.notna(win) else f"{'—':>9}"
            line += f"{ret:>+9.2f}" if pd.notna(ret) else f"{'—':>9}"
        print(line)
    print("\nn grows every run; 5d/10d/21d horizons need a post-close cache warm "
          "(--refresh) to populate. Judge nothing on a thin panel.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Per-method simulated-trade directional performance.")
    p.add_argument("--days", type=int, default=None, help="only signals from the last N days")
    p.add_argument("--min-n", type=int, default=10, help="min joint obs before win/ret is reported")
    p.add_argument("--dedupe", choices=("last", "all"), default="last")
    p.add_argument("--backfill", action="store_true",
                   help="materialise simulated_trades from existing signals, then report")
    p.add_argument("--refresh", action="store_true",
                   help="force-warm the daily OHLCV cache for panel tickers first")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.backfill:
        backfill_from_signals()

    from src.db import repo
    repo.set_read_only(not args.backfill)  # backfill needs write; reporting is read-only

    if args.refresh:
        df = load_sim_trades(args.days)
        if not df.empty:
            from src.analysis.signal_panel import refresh_panel_ohlcv
            refresh_panel_ohlcv(df["ticker"].unique().tolist())

    perf = compute_method_perf(days=args.days, dedupe=args.dedupe, min_n=args.min_n)
    print_report(perf)


if __name__ == "__main__":
    main()
