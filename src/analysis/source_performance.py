"""Per-discovery-source performance — which parts of the funnel find winners.

The universe is built from many discovery sources (a static watchlist plus
trending / screener / macro→holdings / catalyst / smart-money / related-company
expansions). The pipeline stamps the FIRST source that surfaced each ticker onto
every scored row (``signals.universe_source``) and every trade — the measurement
behind an eventual *adaptive discovery budget*: expand the sources that
demonstrably surface names that move favorably, starve the ones that don't.

This module turns that stamp into evidence two ways:

* ``compute_source_performance`` — the UNBIASED view. It groups the signals panel
  (every scored ticker, joined with forward returns from the OHLCV cache — the
  same join ``signal_panel`` uses) by ``universe_source`` and reports, per
  horizon: the funnel share, the mean forward return of the source's names
  (discovery quality, direction-agnostic — "do these names tend to move up?"),
  the up-share win rate, and the Spearman IC of the aggregate ``combined_score``
  against the forward return (signal skill ON that source's names — a negative IC
  means "they trade but predict backwards"). Every scored ticker counts, not just
  the gate-selected few that became trades, so it is free of the ledger's
  selection bias — the dataset the adaptive budget will consume.
* ``compute_source_trade_perf`` — the REALIZED view. Actual per-source trade
  outcomes (win rate, mean/median return) from the ledger: small-n and
  selection-biased, but it is what actually traded and made or lost money.

Both are pure (DataFrame / list in, plain structures out) so they're unit-
testable; ``load_*`` + ``main`` are the live-DB convenience layer.

Usage:  python -m src.analysis.source_performance [--horizons 1,5,10] [--days 90]
                                                  [--min-n 10]
"""

from __future__ import annotations

import argparse
from statistics import median
from typing import Iterable, Optional, Sequence

import pandas as pd

from src.analysis.signal_panel import _spearman, build_panel, periodic_ic_stats

# A score below this magnitude is "no view" (the aggregator had nothing) and is
# excluded from the source's IC / simulated return — matches signal_panel.py.
_EPS = 1e-12

# Label for rows whose provenance stamp is missing (pre-stamp history, or a
# ticker the marker never reached). Shown so the gap is VISIBLE, not silently
# folded into a real source.
_UNSTAMPED = "(unstamped)"


def _source_label(val) -> str:
    """Normalise a raw ``universe_source`` cell to a display label, mapping the
    various flavours of 'missing' (None / NaN / ""/ "nan") to ``_UNSTAMPED``."""
    if val is None:
        return _UNSTAMPED
    try:
        if pd.isna(val):
            return _UNSTAMPED
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none") else _UNSTAMPED


def compute_source_performance(panel: pd.DataFrame, horizons: Sequence[int] = (1, 5, 10),
                               min_n: int = 10, min_per_day: int = 5,
                               min_days: int = 3) -> pd.DataFrame:
    """Per-``universe_source`` forward-return performance over the signals panel.

    ``panel`` is ``signal_panel.build_panel`` output — one row per
    (signal_date, ticker) with ``universe_source``, ``combined_score``,
    ``signal_date`` and a ``fwd_ret_<h>d`` column per horizon. Returns one row
    per source with:

    * ``rows`` — scored ticker-rows; ``funnel_pct`` — the source's slice of the
      STAMPED funnel (None for the ``(unstamped)`` pre-stamp bucket, which is
      excluded from that denominator).
    * per horizon h: ``n_<h>d`` (rows with a forward return), ``fwd_<h>d`` (mean
      raw forward return %, discovery quality), ``win_<h>d`` (% of moved names
      that rose), ``ic_<h>d`` (Spearman ``combined_score`` vs forward return —
      signal skill on this source), ``icir_<h>d`` (info-ratio of the per-day IC,
      populated only once enough signal-days accrue), and ``simret_<h>d`` (mean
      ``sign(combined_score) × forward_return`` — the gross P&L of trading our
      aggregate direction on this source's names).

    A horizon with fewer than ``min_n`` joint observations reports ``None`` for
    that horizon's stats. Sorted by ``rows`` descending (biggest funnel
    contributors first)."""
    if panel is None or panel.empty or "universe_source" not in panel.columns:
        return pd.DataFrame()
    df = panel.copy()
    df["_source"] = df["universe_source"].map(_source_label)
    # Funnel share is over STAMPED rows only — the ``(unstamped)`` bucket is
    # pre-stamp history (the provenance stamp is forward-collected), not a
    # discovery source, so folding it into the denominator would understate every
    # real source's share until that history ages out. It gets no funnel share.
    stamped_total = int((df["_source"] != _UNSTAMPED).sum())

    rows = []
    for source, g in df.groupby("_source"):
        n_rows = int(len(g))
        is_stamped = str(source) != _UNSTAMPED
        row: dict = {"source": str(source), "rows": n_rows,
                     "funnel_pct": (round(100.0 * n_rows / stamped_total, 1)
                                    if is_stamped and stamped_total else None)}
        score = (pd.to_numeric(g["combined_score"], errors="coerce")
                 if "combined_score" in g.columns
                 else pd.Series(float("nan"), index=g.index))
        for h in horizons:
            fwd = pd.to_numeric(g.get(f"fwd_ret_{h}d"), errors="coerce")
            valid = fwd.notna()
            n = int(valid.sum())
            row[f"n_{h}d"] = n
            if n < min_n:
                for k in ("fwd", "win", "ic", "icir", "simret"):
                    row[f"{k}_{h}d"] = None
                continue
            f = fwd[valid]
            row[f"fwd_{h}d"] = round(float(f.mean()), 4)
            moved = f != 0
            row[f"win_{h}d"] = (round(float((f[moved] > 0).mean() * 100.0), 2)
                                if bool(moved.any()) else None)

            # IC / simret condition on a real aggregate signal (non-zero score).
            s = score[valid]
            has_view = s.notna() & (s.abs() > _EPS)
            if int(has_view.sum()) >= min_n:
                sv, fv = s[has_view], f[has_view]
                ic = _spearman(sv, fv)
                row[f"ic_{h}d"] = round(ic, 3) if ic is not None else None
                _, _, icir, _ = periodic_ic_stats(
                    g.loc[valid, "signal_date"][has_view], sv, fv, min_per_day, min_days)
                row[f"icir_{h}d"] = round(icir, 2) if icir is not None else None
                signed = fv.where(sv > 0, -fv)   # +f for a long call, −f for a short
                row[f"simret_{h}d"] = round(float(signed.mean()), 4)
            else:
                row[f"ic_{h}d"] = row[f"icir_{h}d"] = row[f"simret_{h}d"] = None
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Real discovery sources first (biggest funnel contributor on top); the
    # pre-stamp ``(unstamped)`` history sinks to the bottom.
    out["_unstamped"] = (out["source"] == _UNSTAMPED).astype(int)
    return (out.sort_values(["_unstamped", "rows"], ascending=[True, False])
               .drop(columns="_unstamped").reset_index(drop=True))


def compute_source_trade_perf(trades: Iterable[dict]) -> list:
    """Realized per-source trade outcomes from the ledger. ``trades`` is the list
    of trade dicts (each carries ``universe_source`` + cost-adjusted
    ``return_pct``; open trades contribute their live M2M mark). Returns
    ``[{source, trades, win_rate, avg_return, median_return, best, worst}]``
    sorted by trade count descending. Direction is already baked into
    ``return_pct`` (a profitable short is a positive return), so a win is simply
    ``return_pct > 0``."""
    groups: dict = {}
    for t in trades or []:
        r = t.get("return_pct")
        if r is None:
            continue
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        groups.setdefault(_source_label(t.get("universe_source")), []).append(r)

    out = []
    for source, rets in groups.items():
        if not rets:
            continue
        wins = sum(1 for r in rets if r > 0)
        out.append({
            "source":        source,
            "trades":        len(rets),
            "win_rate":      round(100.0 * wins / len(rets), 1),
            "avg_return":    round(sum(rets) / len(rets), 3),
            "median_return": round(float(median(rets)), 3),
            "best":          round(max(rets), 2),
            "worst":         round(min(rets), 2),
        })
    out.sort(key=lambda r: r["trades"], reverse=True)
    return out


# ── live-DB convenience layer ────────────────────────────────────────────────

def load_source_performance(horizons: Sequence[int] = (1, 5, 10),
                            days: Optional[int] = None, min_n: int = 10) -> pd.DataFrame:
    """Build the signals panel and compute per-source performance over it."""
    panel = build_panel(horizons=horizons, days=days)
    if panel is None or panel.empty:
        return pd.DataFrame()
    return compute_source_performance(panel, horizons=horizons, min_n=min_n)


def load_source_trade_perf() -> list:
    """Per-source realized trade performance from the ledger (read-only)."""
    from src.db import repo
    try:
        return compute_source_trade_perf(repo.load_trades())
    except Exception:
        return []


def _print_report(horizons: Sequence[int], days: Optional[int], min_n: int) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from src.db import repo
    repo.set_read_only(True)

    perf = load_source_performance(horizons=horizons, days=days, min_n=min_n)
    if perf is None or perf.empty:
        print("No per-source signal rows with forward returns yet — the signals panel "
              "accrues every run (warm forward closes with "
              "`python -m src.analysis.signal_panel --refresh`).")
    else:
        total_rows = int(perf["rows"].sum())
        print(f"\nDiscovery-source performance — {len(perf)} source(s), "
              f"{total_rows} scored ticker-row(s)")
        print("fwd = mean raw forward return % (discovery quality); win = up-share %; "
              "IC = Spearman(combined_score, fwd_ret) — signal skill on the source.\n")
        head = f"{'source':<18}{'rows':>7}{'funnel%':>9}"
        for h in horizons:
            head += f"{f'n@{h}':>6}{f'fwd@{h}':>9}{f'win@{h}':>8}{f'IC@{h}':>8}"
        print(head)
        print("-" * len(head))
        for _, r in perf.iterrows():
            fp = r.get("funnel_pct")
            fp_s = f"{fp:>9.1f}" if pd.notna(fp) else f"{'—':>9}"
            line = f"{str(r['source']):<18}{int(r['rows']):>7}{fp_s}"
            for h in horizons:
                n, fwd, win, ic = (r.get(f"n_{h}d"), r.get(f"fwd_{h}d"),
                                   r.get(f"win_{h}d"), r.get(f"ic_{h}d"))
                line += f"{int(n) if pd.notna(n) else 0:>6}"
                line += f"{fwd:>+9.2f}" if pd.notna(fwd) else f"{'—':>9}"
                line += f"{win:>7.1f}%" if pd.notna(win) else f"{'—':>8}"
                line += f"{ic:>+8.3f}" if pd.notna(ic) else f"{'—':>8}"
            print(line)

    trade_perf = load_source_trade_perf()
    if trade_perf:
        print(f"\nRealized trades by source — {len(trade_perf)} source(s)")
        print(f"{'source':<18}{'trades':>8}{'win%':>7}{'avg%':>8}{'med%':>8}{'best%':>8}{'worst%':>8}")
        print("-" * 65)
        for r in trade_perf:
            print(f"{r['source']:<18}{r['trades']:>8}{r['win_rate']:>7.1f}"
                  f"{r['avg_return']:>+8.2f}{r['median_return']:>+8.2f}"
                  f"{r['best']:>+8.2f}{r['worst']:>+8.2f}")

    print("\nThe signals-panel view is the large, unbiased sample; the realized-trades "
          "view is what actually traded (small n, gate-selected). Judge a source only "
          "once its n is real — this is a forward-collected accumulator.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Per-discovery-source forward-return + realized-trade performance.")
    p.add_argument("--horizons", default="1,5,10",
                   help="comma-separated forward horizons in sessions (default 1,5,10)")
    p.add_argument("--days", type=int, default=None,
                   help="only signals from the last N calendar days (default: all)")
    p.add_argument("--min-n", type=int, default=10,
                   help="minimum joint observations before a horizon's stats report (default 10)")
    args = p.parse_args(list(argv) if argv is not None else None)
    horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())
    _print_report(horizons, args.days, args.min_n)


if __name__ == "__main__":
    main()
