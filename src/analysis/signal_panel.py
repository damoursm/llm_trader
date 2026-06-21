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

from src.db.schema import SIGNAL_METHOD_COLUMNS, SIGNAL_TIMEFRAME_COLUMNS

# Scored columns the IC report covers: every per-method column plus the
# aggregator's weighted combined score (the "all methods together" row).
PANEL_SCORE_COLUMNS = list(SIGNAL_METHOD_COLUMNS) + ["combined_score"]


# ── IC categories (the dashboard's 4-section grouping) ─────────────────────
# The OHLCV methods are split by candle size; everything else (news, sentiment,
# smart money, options, PEAD, catalysts…) keeps using the most-recent data and
# lands in "Other". The 8 daily-technical method names are exactly those that
# have a 30-min variant — derived from the schema so there's one source of truth.
IC_CATEGORY_30M = "Technical · 30-min"
IC_CATEGORY_1D = "Technical · Daily"
IC_CATEGORY_1W = "Technical · Weekly"
IC_CATEGORY_OTHER = "Other (news · sentiment · smart money · options · catalysts)"
IC_CATEGORY_ORDER = (IC_CATEGORY_30M, IC_CATEGORY_1D, IC_CATEGORY_1W, IC_CATEGORY_OTHER)

_DAILY_TECHNICAL = frozenset(c[:-4] for c in SIGNAL_TIMEFRAME_COLUMNS if c.endswith("_30m"))


def category_for(method: str) -> str:
    """Map a method/score column to its IC category."""
    if method.endswith("_30m"):
        return IC_CATEGORY_30M
    if method.endswith("_1w"):
        return IC_CATEGORY_1W
    if method in _DAILY_TECHNICAL:
        return IC_CATEGORY_1D
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
    count ``n_<h>d``, Spearman ``ic_<h>d``, directional ``hit_<h>d`` (the
    **simulated win rate** — % of non-zero scores whose sign matched the forward
    return's), and ``simret_<h>d`` (the **simulated return** — mean of
    ``sign(score) × forward_return`` over the same rows, i.e. the gross P&L if
    that method alone had decided the trade direction). Methods with fewer than
    ``min_n`` joint observations report NaN. Sorted by |IC| at the longest
    horizon. Each row is also tagged with its ``category``."""
    rows = []
    for method in PANEL_SCORE_COLUMNS:
        if method not in panel.columns:
            continue
        s_all = pd.to_numeric(panel[method], errors="coerce")
        has_view = s_all.notna() & (s_all.abs() > 1e-12)
        row: dict = {"method": method, "category": category_for(method),
                     "views": int(has_view.sum())}
        for h in horizons:
            col = f"fwd_ret_{h}d"
            f_all = pd.to_numeric(panel.get(col), errors="coerce")
            valid = has_view & f_all.notna()
            n = int(valid.sum())
            row[f"n_{h}d"] = n
            if n < min_n:
                row[f"ic_{h}d"] = None
                row[f"hit_{h}d"] = None
                row[f"simret_{h}d"] = None
                continue
            s, f = s_all[valid], f_all[valid]
            row[f"ic_{h}d"] = _spearman(s, f)
            moved = f != 0
            row[f"hit_{h}d"] = (float(((s > 0) == (f > 0))[moved].mean() * 100)
                                if moved.any() else None)
            # Simulated solo return: trade the SIGN of the score, hold to horizon.
            signed = f.where(s > 0, -f)        # +f when score>0 (long), −f when score<0 (short)
            row[f"simret_{h}d"] = round(float(signed.mean()), 4)
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
          "win = simulated solo win rate (sign-agreement %); sim = simulated solo "
          "return % (mean sign(score)×fwd_ret).\n")
    head = f"{'method':<34}{'views':>7}"
    for h in horizons:
        head += f"{f'n@{h}':>7}{f'IC@{h}':>8}{f'win@{h}':>8}{f'sim@{h}':>9}"
    width = len(head)

    def _emit_rows(subset: pd.DataFrame) -> None:
        for _, r in subset.iterrows():
            label = METHOD_LABELS.get(r["method"], r["method"])
            line = f"{label:<34}{int(r['views']):>7}"
            for h in horizons:
                n, icv, hit, sim = (r[f"n_{h}d"], r[f"ic_{h}d"],
                                    r[f"hit_{h}d"], r[f"simret_{h}d"])
                line += f"{int(n):>7}"
                line += f"{icv:>+8.3f}" if pd.notna(icv) else f"{'—':>8}"
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
    ic = compute_ic(panel, horizons=horizons, min_n=args.min_n) if not panel.empty else pd.DataFrame()
    print_report(panel, ic, horizons)


if __name__ == "__main__":
    main()
