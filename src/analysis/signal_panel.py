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
from typing import Iterable, Optional, Sequence

import pandas as pd

from src.db.schema import SIGNAL_METHOD_COLUMNS

# Scored columns the IC report covers: every per-method column plus the
# aggregator's weighted combined score (the "all methods together" row).
PANEL_SCORE_COLUMNS = list(SIGNAL_METHOD_COLUMNS) + ["combined_score"]


def _close_series(ticker: str) -> dict:
    """date → close for one ticker, from the OHLCV cache. Module-level seam so
    tests can monkeypatch it (mirrors tests' fake_closes pattern)."""
    from src.performance.daily_nav import _load_close_series
    return _load_close_series(ticker)


def _load_signals(days: Optional[int]) -> pd.DataFrame:
    from src.db import repo
    try:
        if days:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            return repo.fetch_df(
                "SELECT * FROM signals WHERE signal_date >= ? ORDER BY generated_at",
                [cutoff])
        return repo.fetch_df("SELECT * FROM signals ORDER BY generated_at")
    except Exception as e:
        # DB missing or table not created yet (it appears on the first pipeline
        # run after the schema gained the signals table) — report, don't crash.
        print(f"Could not read the signals table ({e}).\n"
              "It is created automatically on the next pipeline run; rows accumulate per run.")
        return pd.DataFrame()


def build_panel(horizons: Sequence[int] = (1, 5, 10), days: Optional[int] = None,
                dedupe: str = "last", signals_df: Optional[pd.DataFrame] = None,
                ) -> pd.DataFrame:
    """The signals table joined with forward returns: one row per
    (signal_date, ticker), plus a ``fwd_ret_<h>d`` column per horizon (in %,
    NaN where the OHLCV cache doesn't yet reach signal_date + h sessions)."""
    df = signals_df if signals_df is not None else _load_signals(days)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    if dedupe == "last" and "generated_at" in df.columns:
        df = (df.sort_values("generated_at")
                .groupby(["signal_date", "ticker"], as_index=False).tail(1))

    df["_sig_date"] = df["signal_date"].map(date.fromisoformat)

    # One cache read per ticker, shared across all its rows/horizons.
    closes_by_ticker: dict = {}
    for tk in df["ticker"].unique():
        try:
            closes_by_ticker[tk] = _close_series(tk) or {}
        except Exception:
            closes_by_ticker[tk] = {}
    dates_by_ticker = {tk: sorted(c.keys()) for tk, c in closes_by_ticker.items()}

    def fwd(row, h: int) -> Optional[float]:
        dates = dates_by_ticker.get(row["ticker"]) or []
        closes = closes_by_ticker[row["ticker"]]
        i = bisect_left(dates, row["_sig_date"])
        if i >= len(dates) or i + h >= len(dates):
            return None
        base = closes[dates[i]]
        if not base or base <= 0:
            return None
        return (closes[dates[i + h]] / base - 1.0) * 100.0

    for h in horizons:
        df[f"fwd_ret_{h}d"] = df.apply(lambda r: fwd(r, h), axis=1)
    return df.drop(columns=["_sig_date"])


def _spearman(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Spearman rank correlation = Pearson on average-tie ranks (no scipy)."""
    if len(a) < 2:
        return None
    ic = a.rank().corr(b.rank())
    return None if pd.isna(ic) else float(ic)


def compute_ic(panel: pd.DataFrame, horizons: Sequence[int] = (1, 5, 10),
               min_n: int = 20) -> pd.DataFrame:
    """Per-method IC table: for each score column × horizon, the observation
    count ``n_<h>d``, Spearman ``ic_<h>d``, and directional ``hit_<h>d`` (% of
    non-zero scores whose sign matched the forward return's). Methods with
    fewer than ``min_n`` joint observations report NaN — too little data to
    read. Sorted by |IC| at the longest horizon."""
    rows = []
    for method in PANEL_SCORE_COLUMNS:
        if method not in panel.columns:
            continue
        s_all = pd.to_numeric(panel[method], errors="coerce")
        has_view = s_all.notna() & (s_all.abs() > 1e-12)
        row: dict = {"method": method, "views": int(has_view.sum())}
        for h in horizons:
            col = f"fwd_ret_{h}d"
            f_all = pd.to_numeric(panel.get(col), errors="coerce")
            valid = has_view & f_all.notna()
            n = int(valid.sum())
            row[f"n_{h}d"] = n
            if n < min_n:
                row[f"ic_{h}d"] = None
                row[f"hit_{h}d"] = None
                continue
            s, f = s_all[valid], f_all[valid]
            row[f"ic_{h}d"] = _spearman(s, f)
            moved = f != 0
            row[f"hit_{h}d"] = (float(((s > 0) == (f > 0))[moved].mean() * 100)
                                if moved.any() else None)
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
    print("IC = Spearman(score, forward close-to-close return); zero scores excluded; "
          "hit = sign-agreement %.\n")
    head = f"{'method':<34}{'views':>7}"
    for h in horizons:
        head += f"{f'n@{h}d':>8}{f'IC@{h}d':>9}{f'hit@{h}d':>9}"
    print(head)
    print("-" * len(head))
    for _, r in ic.iterrows():
        label = METHOD_LABELS.get(r["method"], r["method"])
        line = f"{label:<34}{r['views']:>7}"
        for h in horizons:
            n, icv, hit = r[f"n_{h}d"], r[f"ic_{h}d"], r[f"hit_{h}d"]
            line += f"{int(n):>8}"
            line += f"{icv:>+9.3f}" if pd.notna(icv) else f"{'—':>9}"
            line += f"{hit:>8.1f}%" if pd.notna(hit) else f"{'—':>9}"
        print(line)
    print("\nA well-behaved method shows IC > 0 that persists across horizons; "
          "IC ≈ 0 on a large n means the method adds noise. n grows every run — "
          "judge nothing on a thin panel.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Per-method IC report over the persisted signals panel.")
    p.add_argument("--horizons", default="1,5,10",
                   help="comma-separated forward horizons in sessions (default 1,5,10)")
    p.add_argument("--days", type=int, default=None,
                   help="only signals from the last N calendar days (default: all)")
    p.add_argument("--min-n", type=int, default=20,
                   help="minimum joint observations before an IC is reported (default 20)")
    p.add_argument("--dedupe", choices=("last", "all"), default="last",
                   help="'last' keeps one row per (day, ticker) — the day's final run (default)")
    args = p.parse_args(list(argv) if argv is not None else None)
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())

    panel = build_panel(horizons=horizons, days=args.days, dedupe=args.dedupe)
    ic = compute_ic(panel, horizons=horizons, min_n=args.min_n) if not panel.empty else pd.DataFrame()
    print_report(panel, ic, horizons)


if __name__ == "__main__":
    main()
