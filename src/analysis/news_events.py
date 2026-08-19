"""News-event study: what KIND of news, at what MAGNITUDE, precedes what
pivot-basis return.

The dataset is one row per (ticker, signal_date) news EVENT:

  - catalyst   — the fixed-taxonomy class (`sentiment.NEWS_CATALYST_TYPES`).
                 LIVE rows carry the sentiment LLM's own classification
                 (`signals.news_catalyst`, captured since prompt v4 2026-08-15);
                 HISTORICAL rows fall back to `news_event_backfill`
                 (`python -m src.analysis.news_backfill` — a later classifier
                 over re-fetched Polygon headlines, provenance kept separate).
  - direction / magnitude — sign and size of the news read: the RAW LLM verdict
                 (`news_raw_score`) when captured, else the adjusted `news`
                 score (historical rows; scaler-shrunk, so magnitude bands
                 understate slightly there — the `era` column marks it).
  - fwd_ret_pivot — the % return from the signal-date close to the next pivot
                 EXTREME (the system's decision target), taken from
                 `signal_panel.build_panel` so the as-of settlement rules are
                 the shared ones.

DESCRIPTIVE, not promotion machinery: the news read is taken UNMASKED straight
from `signals` (the epoch mask exists so a superseded scorer's record is never
charged to the current one in weighting/filtering — an event study describing
"what happened after earnings-type news" is a different question, and the `era`
column ('current' vs 'superseded', split by the news scorer epoch) keeps the
two prompt regimes separable). Nothing here feeds live weights.

Dedupe: the FIRST qualifying read of each (ticker, day) is the event row — the
freshest reaction to the news (later ticks mostly re-serve the cached verdict
as the articles age). Labels merge from the panel's dedupe="last" frame, which
is valid because `fwd_ret_pivot` is a function of the signal DATE, and the
day's last run has the most-settled label.

CLI: python -m src.analysis.news_events [--days N] [--min-n 8] [--era] [--bands]
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Magnitude bands on |verdict| — mirrors the prompt's band rubric.
MAGNITUDE_BINS = (0.0, 0.25, 0.60, 1.0001)
MAGNITUDE_LABELS = ("minor", "moderate", "major")


def _events_sql(days: Optional[int]) -> tuple[str, list]:
    where = "s.news IS NOT NULL AND (s.news <> 0.0 OR (s.news_catalyst IS NOT NULL AND s.news_catalyst <> 'none'))"
    params: list = []
    if days:
        from datetime import date, timedelta
        where += " AND s.signal_date >= ?"
        params.append((date.today() - timedelta(days=int(days))).isoformat())
    # Walk-forward cutoff: since 2026-08-15 this loader also feeds the
    # catalyst_tilt calibration, so under analysis_asof it must see only
    # strictly-earlier events (labels are already as-of via build_panel).
    try:
        from src.analysis.asof import current_asof
        _asof = current_asof()
        if _asof:
            where += " AND s.signal_date < ?"
            params.append(str(_asof))
    except Exception:
        pass
    sql = f"""
        SELECT ticker, signal_date, news, news_catalyst, news_raw_score
        FROM (
            SELECT s.ticker, s.signal_date, s.news, s.news_catalyst, s.news_raw_score,
                   row_number() OVER (PARTITION BY s.ticker, s.signal_date
                                      ORDER BY s.generated_at) AS _rn
            FROM signals s
            WHERE {where}
        ) WHERE _rn = 1
    """
    return sql, params


def load_news_events(days: Optional[int] = None) -> pd.DataFrame:
    """One row per (ticker, signal_date) news event, labeled with the pivot
    forward return. Columns: ticker, signal_date, news, news_raw_score,
    catalyst, catalyst_source ('live'|'backfill'|None), direction
    ('bull'|'bear'|'zero'), magnitude, era, fwd_ret_pivot, end_date_pivot,
    fwd_ret_5d."""
    from src.db import repo

    sql, params = _events_sql(days)
    ev = repo.fetch_df(sql, params)
    if ev is None or ev.empty:
        return pd.DataFrame()

    try:
        bf = repo.fetch_df(
            "SELECT ticker, signal_date, catalyst AS bf_catalyst "
            "FROM news_event_backfill WHERE catalyst IS NOT NULL")
    except Exception:
        bf = pd.DataFrame(columns=["ticker", "signal_date", "bf_catalyst"])
    if not bf.empty:
        ev = ev.merge(bf, on=["ticker", "signal_date"], how="left")
    else:
        ev["bf_catalyst"] = None

    live = ev["news_catalyst"].notna()
    ev["catalyst"] = ev["news_catalyst"].where(live, ev["bf_catalyst"])
    ev["catalyst_source"] = np.where(live, "live",
                                     np.where(ev["bf_catalyst"].notna(), "backfill", None))
    ev = ev.drop(columns=["bf_catalyst"])

    news = pd.to_numeric(ev["news"], errors="coerce").fillna(0.0)
    ev["direction"] = np.where(news > 0, "bull", np.where(news < 0, "bear", "zero"))
    raw = pd.to_numeric(ev["news_raw_score"], errors="coerce")
    ev["magnitude"] = raw.abs().where(raw.notna(), news.abs())

    # Era: which news-scorer regime produced the read (descriptive label, not a
    # mask — epoch_for returns the day AFTER a mid-day change, same convention
    # the date-granular consumers use).
    try:
        from src.signals.method_epochs import epoch_for
        _ep = epoch_for("news")
        ev["era"] = (np.where(ev["signal_date"].astype(str) >= _ep.isoformat(),
                              "current", "superseded")
                     if _ep is not None else "current")
    except Exception:
        ev["era"] = "current"

    # Labels from the shared panel machinery (as-of-correct pivot settlement).
    try:
        from src.analysis.signal_panel import build_panel
        panel = build_panel(horizons=(1, 5, 10), days=days, dedupe="last")
        keep = [c for c in ("signal_date", "ticker", "fwd_ret_pivot",
                            "end_date_pivot", "fwd_ret_5d") if c in panel.columns]
        if len(keep) > 2:
            ev = ev.merge(panel[keep], on=["signal_date", "ticker"], how="left")
    except Exception as e:
        logger.warning(f"[news_events] panel labels unavailable: {e}")
    for c in ("fwd_ret_pivot", "end_date_pivot", "fwd_ret_5d"):
        if c not in ev.columns:
            ev[c] = float("nan") if c != "end_date_pivot" else None
    return ev


def catalyst_return_table(ev: pd.DataFrame, min_n: int = 8,
                          ret_col: str = "fwd_ret_pivot") -> pd.DataFrame:
    """Per-catalyst outcome table over labeled events.

    `oriented_*` orients the return by the news direction (+ret when the move
    went the way the news pointed), computed over directional (nonzero) events
    only; `bull_*`/`bear_*` are the RAW forward returns per side — "what
    actually happened after bullish-read vs bearish-read news of this class".
    Zero-score events (typed but netted to 0 — e.g. priced-in deals) aggregate
    under `zero_mean`."""
    if ev is None or ev.empty:
        return pd.DataFrame()
    df = ev.copy()
    df["catalyst"] = df["catalyst"].fillna("(untyped)")
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    df["_ret"] = ret
    sign = df["direction"].map({"bull": 1.0, "bear": -1.0}).astype(float)
    df["_oriented"] = sign * ret

    rows = []
    for cat, g in df.groupby("catalyst"):
        lab = g[g["_ret"].notna()]
        bull = lab[lab.direction == "bull"]["_ret"]
        bear = lab[lab.direction == "bear"]["_ret"]
        zero = lab[lab.direction == "zero"]["_ret"]
        ori = lab["_oriented"].dropna()
        rows.append({
            "catalyst": cat,
            "n": len(g),
            "n_labeled": len(lab),
            "n_bull": int((g.direction == "bull").sum()),
            "n_bear": int((g.direction == "bear").sum()),
            "bull_mean": bull.mean(), "bull_med": bull.median(),
            "bear_mean": bear.mean(), "bear_med": bear.median(),
            "zero_mean": zero.mean(),
            "oriented_mean": ori.mean(),
            "hit_pct": (ori > 0).mean() * 100 if len(ori) else float("nan"),
        })
    out = pd.DataFrame(rows)
    out = out[out["n_labeled"] >= int(min_n)]
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def magnitude_table(ev: pd.DataFrame, min_n: int = 8,
                    ret_col: str = "fwd_ret_pivot") -> pd.DataFrame:
    """Oriented outcome per (catalyst, magnitude band) — does a bigger claimed
    impact actually precede a bigger move? Directional events only."""
    if ev is None or ev.empty:
        return pd.DataFrame()
    df = ev[ev.direction.isin(("bull", "bear"))].copy()
    df["catalyst"] = df["catalyst"].fillna("(untyped)")
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    sign = df["direction"].map({"bull": 1.0, "bear": -1.0}).astype(float)
    df["_oriented"] = sign * ret
    df["band"] = pd.cut(pd.to_numeric(df["magnitude"], errors="coerce"),
                        bins=list(MAGNITUDE_BINS), labels=list(MAGNITUDE_LABELS),
                        include_lowest=True)
    rows = []
    for (cat, band), g in df.groupby(["catalyst", "band"], observed=True):
        ori = g["_oriented"].dropna()
        if len(ori) < int(min_n):
            continue
        rows.append({"catalyst": cat, "band": str(band), "n": len(ori),
                     "oriented_mean": ori.mean(), "oriented_med": ori.median(),
                     "hit_pct": (ori > 0).mean() * 100})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_b"] = out["band"].map({b: i for i, b in enumerate(MAGNITUDE_LABELS)})
    return (out.sort_values(["catalyst", "_b"]).drop(columns="_b")
            .reset_index(drop=True))


def _print(df: pd.DataFrame, title: str) -> None:
    print(f"\n── {title} " + "─" * max(1, 74 - len(title)))
    if df is None or df.empty:
        print("(no rows at this min-n — accrue more data or lower --min-n)")
        return
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="News-catalyst → pivot-return event study")
    ap.add_argument("--days", type=int, default=None, help="window (default: all history)")
    ap.add_argument("--min-n", type=int, default=8, help="min labeled events per row")
    ap.add_argument("--era", action="store_true", help="split by news-scorer era")
    ap.add_argument("--bands", action="store_true", help="add magnitude-band table")
    args = ap.parse_args()

    ev = load_news_events(days=args.days)
    if ev is None or ev.empty:
        print("No news events found — the dataset accrues per run (and via "
              "`python -m src.analysis.news_backfill` for history).")
        return
    typed = ev["catalyst"].notna().sum()
    lab = ev["fwd_ret_pivot"].notna().sum()
    src_counts = ev["catalyst_source"].value_counts(dropna=True).to_dict()
    print(f"events={len(ev)}  typed={typed} ({src_counts})  pivot-labeled={lab}  "
          f"window={ev.signal_date.min()}→{ev.signal_date.max()}")
    base = pd.to_numeric(ev["fwd_ret_pivot"], errors="coerce")
    print(f"all-events base: mean pivot ret {base.mean():+.2f}%  med {base.median():+.2f}%")

    if args.era:
        for era, g in ev.groupby("era"):
            _print(catalyst_return_table(g, min_n=args.min_n),
                   f"catalyst → pivot return [{era} era]")
    else:
        _print(catalyst_return_table(ev, min_n=args.min_n), "catalyst → pivot return")
    if args.bands:
        _print(magnitude_table(ev, min_n=args.min_n), "catalyst × magnitude band (oriented)")


if __name__ == "__main__":
    main()
