"""Data-quality monitoring — source reliability + per-method coverage.

Every data problem this system has hit (news-starvation, the put_call cache
crash, dead congressional / ^TICK feeds) was *silent* until it surfaced as a
weak downstream signal. The raw material to catch them proactively is already
persisted every run:

* ``run_sources`` — one row per data source per run (ok / error / duration). The
  dashboard only ever read the LATEST run's failures; aggregated over N days it
  shows which sources are chronically flaky or slow.
* ``signals`` — the full per-ticker score cross-section. A method silently reads
  ``0.0`` ("no view") when its source failed for a ticker, so a feed going dark
  shows up as collapsing *coverage* (fraction of tickers with a real score)
  before it ever shows up as bad performance.

Both functions are pure (DataFrame in, plain structures out) so they're unit-
testable; the ``load_*`` helpers + ``main`` are the live-DB convenience layer.

Usage:  python -m src.analysis.data_quality [--days 14]
"""

from __future__ import annotations

from statistics import median
from typing import Optional, Sequence

import pandas as pd

from src.db.schema import SIGNAL_METHOD_COLUMNS

# A score below this magnitude is "no view" (method had no usable data), not a
# real signal — matches the convention in signal_panel.py.
_EPS = 1e-9


# Data sources that are DEAD with no free replacement — their upstream went away,
# so an empty return is expected and NOT actionable (don't mask it as ok, don't
# spam WARNINGs). Surfaced distinctly as status "dead" so the deadness is VISIBLE,
# and a source that starts returning data again self-heals to "ok".
#   tick      — ^TICK delisted on Yahoo (needs a non-Yahoo NYSE-TICK provider)
#   insider   — House/Senate Stock Watcher S3 buckets return 403 (anonymous access
#               revoked 2026); the ONLY feeds fetch_insider_trades aggregates, so it
#               returns nothing every run. Corporate-insider Form 4 buys still flow
#               via the `sec` source. Restoring congressional needs a paid API
#               (Quiver / Unusual Whales / FMP) or a clerk/eFD scraper.
# (mcclellan was here — RESTORED via Polygon grouped-daily RANA after ^NYAD was
#  delisted; it is an always-on source again.)
KNOWN_DEAD_SOURCES = frozenset({"tick", "insider"})

# Data sources whose result is event-driven — an empty return is frequently
# legitimate (no qualifying event in the window). Emptiness from these is logged
# at INFO and is NOT flagged as a problem. Everything NOT listed here (and not
# KNOWN_DEAD) is market-wide context expected to ALWAYS return data, so an empty
# return is a WARNING and a dashboard flag: "ran OK but returned nothing".
EXPECTED_SPARSE_SOURCES = frozenset({
    "8k", "sec", "analyst", "eps", "pead", "short",
    "trends", "reddit", "whisper", "revision", "options", "gex",
    "earnings_cal", "macro_news",
})


def source_status(ok, empty, label: str = None) -> str:
    """Outcome for one source-run: ``error`` (raised), ``dead`` (a known-dead feed
    that returned nothing — no free replacement, so not actionable), ``empty``
    (ran fine but returned nothing), or ``ok`` (returned data). A known-dead
    source that returns data self-heals to ``ok``."""
    if not bool(ok):
        return "error"
    if bool(empty):
        return "dead" if label in KNOWN_DEAD_SOURCES else "empty"
    return "ok"


def is_unexpected_empty(label: str, status: str) -> bool:
    """An empty result from a source expected to always return data — the
    actionable 'a feed went silently dark' signal. Dead sources report status
    'dead' (never 'empty'), so they are never flagged here."""
    return status == "empty" and label not in EXPECTED_SPARSE_SOURCES


# ── Context hollowness / partial-fill detection ──────────────────────────────
#
# A fetcher can return a NON-None context object that is meaningless — every
# measurement field None/0 (hollow), or, for a context built from several
# upstreams, only partially filled (e.g. global_macro got DXY but copper/gold and
# oil failed). `_result_size` treats any present object as "not empty", so these
# slip through. This registry names the ESSENTIAL field(s) per always-on context
# source; ALL must be present (so a partial multi-component fetch is caught too).
# The chosen fields are levels/prices/yields/counts/lists that are NEVER a
# legitimate 0/empty when the feed is healthy — so "present" = non-None, nonzero,
# non-empty. Event-driven (EXPECTED_SPARSE) and dead sources are intentionally
# absent; their emptiness is already handled.
CONTEXT_REQUIRED_FIELDS = {
    "fred":             ("fed_funds_rate", "yield_spread_10y2y"),
    "cot":              ("signals",),
    "vix":              ("vix",),
    "credit":           ("hyg_price", "spy_price"),
    # ticker_signals is the per-ticker payload that feeds the put_call scorer;
    # market_pc_ratio (CBOE market-wide) is a bonus and is frequently None, so it
    # is NOT the populated criterion.
    "put_call":         ("ticker_signals",),
    "breadth":          ("etf_count",),
    "highs_lows":       ("total_count",),
    "fedwatch":         ("ff_upper", "ff_lower"),
    "bond_internals":   ("yield_10y", "yield_3m"),
    "move":             ("move",),
    "dix":              ("dix",),
    "global_macro":     ("dxy", "copper_gold_ratio", "oil_price"),
    "sector_rotation":  ("sectors",),
    "rotation_drivers": ("fed_rate_current",),
    "intermarket":      ("entries",),
}


def _present(val) -> bool:
    """Whether a required field carries real data: non-None, non-empty container,
    and nonzero number (the registry only lists fields where 0 means failure)."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (list, dict, tuple, set, str)):
        return len(val) > 0
    if isinstance(val, (int, float)):
        return val != 0
    return True


def is_context_populated(label: str, result) -> Optional[bool]:
    """Is a returned context object meaningfully populated?

    ``True`` / ``False`` for a source with a registered field check (or a model
    that defines its own ``is_populated()``); ``None`` when there is no check for
    this source (unknown → caller treats it as populated). A list/None result is
    not this function's concern — ``_result_size`` already handles those."""
    if result is None or isinstance(result, (list, tuple, set, dict, str)):
        return None
    checker = getattr(result, "is_populated", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return None
    required = CONTEXT_REQUIRED_FIELDS.get(label)
    if not required:
        return None
    return all(_present(getattr(result, f, None)) for f in required)


# ── Source reliability (from run_sources) ────────────────────────────────────

def _na_false(val) -> bool:
    """bool() that treats pandas NA / NaN / None as False (so it never raises)."""
    return bool(val) if pd.notna(val) else False


def compute_source_reliability(df: pd.DataFrame) -> list:
    """Per-source reliability over the rows given. ``df`` columns: source_label,
    ok, error, duration_s, started_at, and (when the DB has been migrated)
    n_items, empty. Returns a list of dicts sorted worst-first: lowest success
    rate, then UNEXPECTED-empty sources (always-on feeds that returned nothing),
    then highest empty-rate, then slowest.

    ``last_status`` is the most recent run's three-state outcome (error / empty /
    ok); ``empty_rate`` is the share of *successful* runs that returned nothing
    (None until the migrated columns carry data); ``unexpected_empty`` marks a
    source whose latest run was empty but which is expected to always have data."""
    if df is None or df.empty or "source_label" not in df.columns:
        return []
    has_empty = "empty" in df.columns
    out = []
    for source, g in df.groupby("source_label"):
        source = str(source)
        if "started_at" in g.columns:
            g = g.sort_values("started_at")
        oks = g["ok"].astype("boolean").fillna(False)
        n = int(len(g))
        n_ok = int(oks.sum())
        durs = pd.to_numeric(g.get("duration_s"), errors="coerce").dropna() if "duration_s" in g else pd.Series(dtype=float)

        fails = g[~oks]
        last_error = None
        if not fails.empty and "error" in fails.columns:
            err = fails.iloc[-1].get("error")
            last_error = str(err)[:200] if err is not None and str(err) != "nan" else None

        # Emptiness (only meaningful on migrated rows; NA on legacy rows).
        empty_rate = n_empty = empty_known = None
        last_empty = False
        if has_empty:
            ok_empty = g.loc[oks, "empty"]
            known = ok_empty.notna()
            empty_known = int(known.sum())
            n_empty = int(ok_empty[known].astype(bool).sum()) if empty_known else 0
            empty_rate = round(100.0 * n_empty / empty_known, 1) if empty_known else None
            last_empty = _na_false(g.iloc[-1].get("empty"))

        last_ok = _na_false(g.iloc[-1].get("ok"))
        last_status = source_status(last_ok, last_empty, source)
        unexpected = is_unexpected_empty(source, last_status)

        out.append({
            "source":        source,
            "runs":          n,
            "ok":            n_ok,
            "fail":          n - n_ok,
            "success_rate":  round(100.0 * n_ok / n, 1) if n else None,
            "median_s":      round(float(durs.median()), 2) if not durs.empty else None,
            "p90_s":         round(float(durs.quantile(0.9)), 2) if not durs.empty else None,
            "last_error":    last_error,
            "empty_runs":    n_empty,
            "empty_rate":    empty_rate,
            "last_status":   last_status,
            "expected_sparse": source in EXPECTED_SPARSE_SOURCES,
            "known_dead":    source in KNOWN_DEAD_SOURCES,
            "unexpected_empty": unexpected,
        })
    out.sort(key=lambda r: (
        r["success_rate"] if r["success_rate"] is not None else 100.0,
        0 if r["unexpected_empty"] else 1,
        -(r["empty_rate"] or 0.0),
        -(r["median_s"] or 0.0),
    ))
    return out


# ── Per-method coverage (from signals) ───────────────────────────────────────

def compute_method_coverage(df: pd.DataFrame,
                            methods: Sequence[str] = SIGNAL_METHOD_COLUMNS,
                            recent_days: int = 2) -> dict:
    """Per-method data coverage = fraction of scored rows with a real (non-zero)
    score. ``df`` is the signals panel (one row per run×ticker) with a
    ``signal_date`` column + one DOUBLE column per method.

    Returns ``{n_rows, days, per_method:[{method, coverage_pct, n_scored,
    n_total, recent_pct, prior_pct, delta}], by_day:{method:[{day,coverage_pct,
    n}]}}``. ``delta`` = recent (last ``recent_days``) minus prior coverage — a
    large negative delta is a feed that went dark. Sorted biggest-drop-first,
    then lowest coverage."""
    if df is None or df.empty:
        return {"n_rows": 0, "days": [], "per_method": [], "by_day": {}}
    cols = [m for m in methods if m in df.columns]
    days = sorted(str(d) for d in df["signal_date"].unique()) if "signal_date" in df.columns else []
    recent_set = set(days[-recent_days:]) if days else set()
    n_total = int(len(df))

    per_method, by_day = [], {}
    for m in cols:
        scored_mask = pd.to_numeric(df[m], errors="coerce").fillna(0.0).abs() > _EPS
        scored = int(scored_mask.sum())

        series = []
        if "signal_date" in df.columns:
            for d, g in df.groupby("signal_date"):
                gm = pd.to_numeric(g[m], errors="coerce").fillna(0.0).abs() > _EPS
                series.append({"day": str(d), "coverage_pct": round(100.0 * int(gm.sum()) / len(g), 1),
                               "n": int(len(g))})
            series.sort(key=lambda x: x["day"])
        by_day[m] = series

        recent = [x["coverage_pct"] for x in series if x["day"] in recent_set]
        prior = [x["coverage_pct"] for x in series if x["day"] not in recent_set]
        recent_pct = round(median(recent), 1) if recent else None
        prior_pct = round(median(prior), 1) if prior else None
        delta = round(recent_pct - prior_pct, 1) if (recent_pct is not None and prior_pct is not None) else None

        per_method.append({
            "method":       m,
            "coverage_pct": round(100.0 * scored / n_total, 1) if n_total else 0.0,
            "n_scored":     scored,
            "n_total":      n_total,
            "recent_pct":   recent_pct,
            "prior_pct":    prior_pct,
            "delta":        delta,
        })
    per_method.sort(key=lambda r: (r["delta"] if r["delta"] is not None else 0.0, r["coverage_pct"]))
    return {"n_rows": n_total, "days": days, "per_method": per_method, "by_day": by_day}


# ── live-DB convenience layer ────────────────────────────────────────────────

def load_source_rows(days: Optional[int] = 14) -> pd.DataFrame:
    """run_sources joined with run timestamps, ENABLED sources only, last N days.

    Pulls the emptiness columns (n_items, empty) when present; falls back to the
    legacy column set on a DB that predates them (the next pipeline write adds
    them via ensure_schema), so a read never breaks on an un-migrated DB."""
    from datetime import date, timedelta
    from src.db import repo
    base = ("FROM run_sources rs JOIN runs r ON rs.run_id = r.run_id "
            "WHERE rs.enabled = TRUE")
    params: list = []
    if days:
        base += " AND r.started_at >= ?"
        params.append((date.today() - timedelta(days=days)).isoformat())
    cols = "rs.source_label, rs.ok, rs.error, rs.duration_s, r.started_at"
    try:
        return repo.fetch_df(f"SELECT {cols}, rs.n_items, rs.empty {base}", params)
    except Exception:
        try:
            return repo.fetch_df(f"SELECT {cols} {base}", params)
        except Exception:
            return pd.DataFrame()


def load_signal_rows(days: Optional[int] = 14) -> pd.DataFrame:
    from datetime import date, timedelta
    from src.db import repo
    cols = "signal_date, " + ", ".join(SIGNAL_METHOD_COLUMNS)
    sql = f"SELECT {cols} FROM signals"
    params: list = []
    if days:
        sql += " WHERE signal_date >= ?"
        params.append((date.today() - timedelta(days=days)).isoformat())
    try:
        return repo.fetch_df(sql, params)
    except Exception:
        return pd.DataFrame()


def _print_report(days: int) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from src.db import repo
    repo.set_read_only(True)

    rel = compute_source_reliability(load_source_rows(days))
    print(f"\nSource reliability — last {days} day(s), {len(rel)} source(s)")
    print(f"{'source':<22}{'runs':>6}{'ok%':>7}{'empty%':>8}{'status':>9}{'med_s':>8}  note")
    print("-" * 78)
    for r in rel:
        empty_pct = f"{r['empty_rate']:>8.1f}" if r.get("empty_rate") is not None else f"{'—':>8}"
        flag = " ⚠ unexpected-empty (always-on feed dark)" if r.get("unexpected_empty") else ""
        note = flag or (r.get("last_error") or "")
        print(f"{r['source']:<22}{r['runs']:>6}{(r['success_rate'] or 0):>7.1f}"
              f"{empty_pct}{r.get('last_status',''):>9}{(r['median_s'] or 0):>8.2f}  {note[:44]}")
    unexpected = [r["source"] for r in rel if r.get("unexpected_empty")]
    if unexpected:
        print("\n⚠ Returned NOTHING this run though they should always have data — "
              f"INVESTIGATE: {', '.join(unexpected)}")

    cov = compute_method_coverage(load_signal_rows(days))
    print(f"\nPer-method coverage — {cov['n_rows']} signal row(s) over {len(cov['days'])} day(s)")
    print(f"{'method':<18}{'cov%':>7}{'scored':>8}{'recent%':>9}{'prior%':>8}{'Δ':>7}")
    print("-" * 60)
    for r in cov["per_method"]:
        def f(v, w=7):
            return f"{v:>{w}.1f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
        print(f"{r['method']:<18}{r['coverage_pct']:>7.1f}{r['n_scored']:>8}"
              f"{f(r['recent_pct'],9)}{f(r['prior_pct'])}{f(r['delta'])}")
    print("\nLow coverage is normal for sparse methods (pead/ext_gap/options); watch for "
          "a negative Δ — a method whose coverage DROPPED is a feed that went dark.")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Data-quality report: source reliability + method coverage.")
    p.add_argument("--days", type=int, default=14, help="lookback window in days (default 14)")
    args = p.parse_args()
    _print_report(args.days)


if __name__ == "__main__":
    main()
