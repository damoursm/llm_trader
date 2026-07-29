"""Per-TICKER simulated performance — every scored name, gates or not.

Every other performance surface aggregates: by method (Signal IC), by discovery
source (source_performance), by feature bucket (predictability,
price_volume_perf), or by realized trade (the ledger). None of them answer the
plainest question about a name you deliberately pinned: **how is the strategy
doing on THIS ticker?**

The answer has to be gate-independent. The trade ledger only contains names that
survived Gates 1-5, which is exactly the selection bias the `signals` panel was
built to escape — a pinned ticker that never clears the confidence bar would be
invisible in the ledger while still being scored every tick. So the simulated
return here is computed over EVERY scored ticker-day: if the system had taken
`combined_score`'s direction on that day, what did it earn over the next N
sessions?

Alongside it, the per-ticker DECISION FUNNEL — scored → recommended → actionable
→ traded — shows where each name actually stops. A ticker with a good simulated
return and `actionable = 0` is one the gates are consistently declining; that
gap is the interesting signal, and it is only visible with both halves side by
side.

CLI:  python -m src.analysis.ticker_performance [--days 30] [--source watchlist]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.analysis.signal_panel import build_panel
from src.db import repo

from loguru import logger  # project configures loguru sinks only

# Below this, `combined_score` is treated as "no view" rather than a weak
# directional call — mirrors the exactly-zero exclusion the simulated_trades
# reshape already applies, with a small epsilon for float noise.
_NO_VIEW = 0.02


def _funnel_counts(days: Optional[int]) -> pd.DataFrame:
    """Per-ticker recommendation + actionable counts from the recs table."""
    where, params = "", []
    if days:
        where = "WHERE generated_at >= (CURRENT_DATE - INTERVAL (?) DAY)::VARCHAR"
        params = [int(days)]
    try:
        df = repo.fetch_df(
            f"""SELECT ticker,
                       COUNT(*)                                        AS recs,
                       SUM(CASE WHEN action IN ('BUY','SELL') THEN 1 ELSE 0 END) AS dir_recs,
                       SUM(CASE WHEN actionable THEN 1 ELSE 0 END)     AS actionable
                FROM recommendations {where}
                GROUP BY ticker""",
            params,
        )
    except Exception as exc:
        logger.debug(f"[ticker_perf] recommendations unavailable: {exc}")
        return pd.DataFrame(columns=["ticker", "recs", "dir_recs", "actionable"])
    return df if df is not None else pd.DataFrame()


def _trade_counts() -> Dict[str, dict]:
    """Per-ticker realized trade counts + mean return from the ledger."""
    try:
        from src.performance.tracker import _load_trades
        trades = _load_trades()
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    for t in trades:
        tk = str(t.get("ticker") or "").upper()
        if not tk:
            continue
        d = out.setdefault(tk, {"trades": 0, "open": 0, "rets": []})
        d["trades"] += 1
        if t.get("status") == "OPEN":
            d["open"] += 1
        r = t.get("return_pct")
        if r is not None:
            d["rets"].append(float(r))
    return out


def compute_ticker_perf(days: Optional[int] = None,
                        horizons: Sequence[int] = (1, 5, 10),
                        source: Optional[str] = None,
                        min_days: int = 1,
                        panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per ticker: simulated per-signal performance + the decision funnel.

    ``source`` filters to one ``universe_source`` (e.g. ``"watchlist"``).
    ``min_days`` drops names with too few scored days to read.
    """
    p = panel if panel is not None else build_panel(horizons=horizons, days=days)
    if p is None or p.empty:
        return pd.DataFrame()
    p = p.copy()
    if source:
        p = p[p["universe_source"].astype(str) == source]
        if p.empty:
            return pd.DataFrame()

    p["combined_score"] = pd.to_numeric(p["combined_score"], errors="coerce")
    p["_view"] = p["combined_score"].abs() >= _NO_VIEW
    p["_sign"] = p["combined_score"].apply(lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
    for h in horizons:
        fwd = pd.to_numeric(p.get(f"fwd_ret_{h}d"), errors="coerce")
        # Oriented: what the signal's own direction earned. NaN where there is
        # no view or no forward bar — both are "no observation", not a zero.
        p[f"_ret_{h}"] = (p["_sign"] * fwd).where(p["_view"])

    rows: List[dict] = []
    for tk, g in p.groupby("ticker"):
        n = len(g)
        if n < min_days:
            continue
        views = g[g["_view"]]
        row = {
            "ticker": tk,
            "source": (g["universe_source"].mode().iat[0]
                       if g["universe_source"].notna().any() else None),
            "signal_days": n,
            "view_days": len(views),
            "avg_score": round(float(g["combined_score"].mean()), 3),
            "avg_conf": (round(float(pd.to_numeric(g["confidence"], errors="coerce").mean()), 3)
                         if "confidence" in g else None),
            "buy_days": int((views["_sign"] > 0).sum()),
            "sell_days": int((views["_sign"] < 0).sum()),
        }
        for h in horizons:
            col = views[f"_ret_{h}"].dropna()
            row[f"ret_{h}d"] = round(float(col.mean()), 3) if len(col) else None
            row[f"hit_{h}d"] = round(100.0 * float((col > 0).mean()), 1) if len(col) else None
            row[f"n_{h}d"] = int(len(col))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # ── decision funnel: where does this name actually stop? ────────────────
    funnel = _funnel_counts(days)
    if funnel is not None and not funnel.empty:
        out = out.merge(funnel, on="ticker", how="left")
    for c in ("recs", "dir_recs", "actionable"):
        out[c] = out[c].fillna(0).astype(int) if c in out.columns else 0

    tc = _trade_counts()
    out["trades"] = out["ticker"].map(lambda t: tc.get(t, {}).get("trades", 0))
    out["open"] = out["ticker"].map(lambda t: tc.get(t, {}).get("open", 0))
    out["real_ret"] = out["ticker"].map(
        lambda t: (round(sum(tc[t]["rets"]) / len(tc[t]["rets"]), 2)
                   if tc.get(t, {}).get("rets") else None))

    sort_h = 5 if 5 in horizons else list(horizons)[0]
    return out.sort_values(f"ret_{sort_h}d", ascending=False, na_position="last")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Per-ticker simulated performance")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--source", default=None, help="filter to one universe_source")
    ap.add_argument("--horizons", default="1,5,10")
    ap.add_argument("--min-days", type=int, default=1)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    hs = [int(x) for x in a.horizons.split(",") if x.strip()]

    df = compute_ticker_perf(days=a.days, horizons=hs, source=a.source,
                             min_days=a.min_days)
    if df.empty:
        print("No scored tickers in this window.")
        raise SystemExit(0)
    cols = (["ticker", "source", "signal_days", "view_days", "avg_score", "avg_conf"]
            + [f"{p}_{h}d" for h in hs for p in ("ret", "hit")]
            + ["recs", "dir_recs", "actionable", "trades", "open", "real_ret"])
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print(df[cols].head(a.top).to_string(index=False))
