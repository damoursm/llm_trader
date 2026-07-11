"""Post-exit forward returns — what would each closed trade have earned if held longer?

The exit-quality block (``exit_quality.py``) judges an exit inside the trade's own
MFE→MAE range; the close-rule harness (``exit_policy_eval.py``) scores hypothetical
policies over the exit-signals panel. Neither answers the direct question about the
trades we ACTUALLY closed: *had we held this position N more sessions, what would it
have returned?* This module measures exactly that, per closed trade and aggregated
per exit rule (``exit_reason``) — the ledger-grounded "did this rule exit too early?"
readout.

Per CLOSED trade, anchored at the ACTUAL exit fill:

    fwd_Nd = sign × (close(exit_date + N sessions) / exit_price − 1) × 100
             sign = +1 long (BUY/BULLISH), −1 short (SELL/BEARISH)

so ``fwd_Nd`` is the ADDITIONAL oriented return the position would have earned by
staying on N more sessions. Positive ⇒ the exit left money on the table (too early);
negative ⇒ the exit dodged a drawdown (good exit). Gross of costs by design: holding
longer defers the same one-way exit cost rather than adding one, so the exit-cost
term ≈ cancels in the hold-vs-close comparison. Forward closes come from the daily
OHLCV cache (``cache/ohlcv/``) — the EOD cache warm keeps recently-exited tickers
warm (``data/cache_warm.py``) so bars keep accruing after a ticker leaves the book.
A trade whose forward bars don't exist yet (closed today / cache gap) is counted as
``pending``, never guessed.

Deterministic given the DB + OHLCV cache. Judge nothing on a handful of closes —
per-reason rows need trades to accumulate, like every other learning surface here.

Usage:  python -m src.analysis.exit_forward
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from statistics import mean, median
from typing import Dict, List, Optional, Sequence, Tuple

# Sessions-after-exit horizons. 1/3/5/10 brackets the system's swing horizon
# (the measured combined_score edge peaks ~1–2d and is gone by ~3–5d, so a
# systematic positive fwd at these horizons is a real early-exit signal).
HORIZONS: Tuple[int, ...] = (1, 3, 5, 10)


def _close_series(ticker: str) -> Dict[date, float]:
    """date → close from the OHLCV cache. Module-level seam so tests can inject
    synthetic series (mirrors ``signal_panel._close_series``)."""
    from src.performance.daily_nav import _load_close_series
    return _load_close_series(ticker) or {}


def _oriented_sign(t: dict) -> Optional[float]:
    """+1 long / −1 short from the trade's action/direction; None = unorientable."""
    tag = str(t.get("action") or t.get("direction") or "").upper()
    if tag in ("BUY", "BULLISH"):
        return 1.0
    if tag in ("SELL", "BEARISH"):
        return -1.0
    return None


def _per_trade(t: dict, horizons: Sequence[int]) -> Optional[dict]:
    """Forward-return row for one CLOSED trade, or None when it can't be anchored
    (open, no exit fill/date, unorientable). Horizons whose forward bar isn't in
    the cache yet are None — the aggregates skip them, never guess them."""
    if t.get("status") != "CLOSED":
        return None
    sign = _oriented_sign(t)
    try:
        exit_price = float(t.get("exit_price") or 0.0)
        exit_d = date.fromisoformat(str(t.get("exit_date") or "")[:10])
    except (TypeError, ValueError):
        return None
    if sign is None or exit_price <= 0:
        return None

    closes = _close_series(str(t.get("ticker") or ""))
    dates = sorted(closes.keys())
    # First session strictly AFTER the exit date: +1d is the next session's close
    # whether the exit fired intraday, after hours, or on a weekend/overnight date.
    j0 = bisect_right(dates, exit_d)
    fwd: Dict[int, Optional[float]] = {}
    for h in horizons:
        j = j0 + h - 1
        if j < len(dates) and closes[dates[j]] > 0:
            fwd[h] = round(sign * (closes[dates[j]] / exit_price - 1.0) * 100.0, 3)
        else:
            fwd[h] = None
    return {
        "ticker":      t.get("ticker"),
        "exit_date":   str(t.get("exit_date") or "")[:10],
        "exit_reason": t.get("exit_reason") or "(unspecified)",
        "return_pct":  round(float(t["return_pct"]), 3) if t.get("return_pct") is not None else None,
        **{f"fwd_{h}d": fwd[h] for h in horizons},
    }


def _aggregate(rows: List[dict], horizons: Sequence[int]) -> dict:
    """Per-horizon n / mean / median / pct_pos over a set of per-trade rows.
    ``pct_pos`` = share of exits whose forward return is POSITIVE, i.e. the
    position kept going our way — the '% exited too early' at that horizon."""
    out: dict = {"trades": len(rows)}
    for h in horizons:
        vals = [r[f"fwd_{h}d"] for r in rows if r.get(f"fwd_{h}d") is not None]
        out[f"n_{h}d"] = len(vals)
        out[f"mean_{h}d"] = round(mean(vals), 3) if vals else None
        out[f"median_{h}d"] = round(median(vals), 3) if vals else None
        out[f"pct_pos_{h}d"] = round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1) if vals else None
    return out


def compute_exit_forward(trades: List[dict], horizons: Sequence[int] = HORIZONS) -> dict:
    """Post-exit forward-return report over CLOSED trades.

    Returns ``{n, n_pending, per_trade:[...], by_reason:[...], overall:{...},
    horizons, verdict}``. ``per_trade`` rows carry ``fwd_{h}d`` per horizon
    (None = bar not cached yet); a row with NO computable horizon counts as
    pending only. ``by_reason`` aggregates per ``exit_reason`` (trade-count
    desc); ``overall`` is the same aggregate over everything."""
    horizons = tuple(horizons)
    all_rows = [r for r in (_per_trade(t, horizons) for t in (trades or [])) if r is not None]
    rows = [r for r in all_rows if any(r[f"fwd_{h}d"] is not None for h in horizons)]
    pending = len(all_rows) - len(rows)
    if not rows:
        return {"n": 0, "n_pending": pending, "per_trade": [], "by_reason": [],
                "overall": {}, "horizons": list(horizons),
                "verdict": "No closed trades with post-exit bars yet (closes too "
                           "recent, or the OHLCV cache lacks the forward sessions)."}

    by_reason: Dict[str, List[dict]] = {}
    for r in rows:
        by_reason.setdefault(r["exit_reason"], []).append(r)
    reason_rows = [{"exit_reason": reason, **_aggregate(seg, horizons)}
                   for reason, seg in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))]
    overall = _aggregate(rows, horizons)

    # Verdict at the swing horizon: the longest of {5d, else the longest horizon
    # with data} — a positive mean here says the book keeps moving our way after
    # we leave it.
    judge = 5 if 5 in horizons and overall.get("n_5d") else max(
        (h for h in horizons if overall.get(f"n_{h}d")), default=horizons[0])
    m, pp = overall.get(f"mean_{judge}d"), overall.get(f"pct_pos_{judge}d")
    parts = [f"Mean +{judge}d forward return after exit: {m:+.2f}% "
             f"({pp:.0f}% of exits kept going our way)"]
    if m is not None and m > 0.5:
        worst = max((r for r in reason_rows if (r.get(f"n_{judge}d") or 0) >= 3),
                    key=lambda r: r.get(f"mean_{judge}d") or -1e9, default=None)
        if worst is not None and (worst.get(f"mean_{judge}d") or 0) > 0:
            parts.append(f"⚠ exits leave money on the table — worst rule: "
                         f"{worst['exit_reason']} (+{worst[f'mean_{judge}d']:.2f}% @{judge}d, "
                         f"n={worst[f'n_{judge}d']})")
    elif m is not None and m < -0.5:
        parts.append("exits are dodging drawdowns (held-longer would have LOST)")

    return {
        "n":         len(rows),
        "n_pending": pending,
        "per_trade": sorted(rows, key=lambda r: r["exit_date"], reverse=True),
        "by_reason": reason_rows,
        "overall":   overall,
        "horizons":  list(horizons),
        "verdict":   "; ".join(parts) + ".",
    }


def compute_exit_forward_report(session: Optional[str] = None,
                                direction: Optional[str] = None,
                                horizons: Sequence[int] = HORIZONS) -> dict:
    """Ledger-loading wrapper with the Exit-Performance tab's filter semantics:
    ``session`` = the session the trade EXITED in, ``direction`` = long|short
    (both exactly as ``compute_exit_reason_perf`` filters them)."""
    from src.performance.tracker import _exit_session_matches, _load_trades, _match_direction
    closed = [t for t in _load_trades()
              if t.get("status") == "CLOSED"
              and _exit_session_matches(t, session)
              and _match_direction(t, direction)]
    return compute_exit_forward(closed, horizons=horizons)


def _print_report(rep: dict) -> None:
    if not rep["n"]:
        print(rep["verdict"])
        return
    hs = rep["horizons"]
    print(f"\nPost-exit forward returns — {rep['n']} closed trade(s) with forward bars"
          + (f" ({rep['n_pending']} pending)" if rep["n_pending"] else ""))
    print(rep["verdict"])
    print()
    head = f"{'exit reason':<24}{'n':>4}" + "".join(
        f"{f'mean+{h}d':>10}{f'%+{h}d':>8}" for h in hs)
    print(head)
    print("-" * len(head))
    for r in rep["by_reason"] + [{"exit_reason": "ALL exits", **rep["overall"]}]:
        cells = ""
        for h in hs:
            m, p = r.get(f"mean_{h}d"), r.get(f"pct_pos_{h}d")
            cells += (f"{m:>10.2f}" if m is not None else f"{'—':>10}")
            cells += (f"{p:>8.0f}" if p is not None else f"{'—':>8}")
        print(f"{r['exit_reason']:<24}{r['trades']:>4}{cells}")
    print()
    head2 = f"{'ticker':<8}{'exit':<12}{'ret%':>8}" + "".join(f"{f'+{h}d':>8}" for h in hs) + "  reason"
    print(head2)
    print("-" * len(head2))
    for r in rep["per_trade"]:
        cells = "".join((f"{r[f'fwd_{h}d']:>8.2f}" if r[f"fwd_{h}d"] is not None else f"{'—':>8}") for h in hs)
        ret = f"{r['return_pct']:>8.2f}" if r["return_pct"] is not None else f"{'—':>8}"
        print(f"{(r['ticker'] or ''):<8}{r['exit_date']:<12}{ret}{cells}  {r['exit_reason']}")


def main() -> None:
    import sys
    from src.db import repo
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: render ⚠/— glyphs
    except Exception:
        pass
    repo.set_read_only(True)   # never contend with a running scheduler's write lock
    try:
        trades = repo.load_trades()
    except Exception as e:
        print(f"Could not read trades ({e}).")
        return
    _print_report(compute_exit_forward(trades))


if __name__ == "__main__":
    main()
