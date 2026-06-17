"""Sim-vs-broker tracking error — the gap between what the strategy MODELED and
what IBKR actually did.

The sim ledger books every decision at its decision price through the modeled
spread/commission stack; ``broker_view`` re-anchors the same trades at the real
average fill prices with the commissions actually charged. The systematic
difference between the two is the execution gap — and a persistent, one-sided gap
is the signature of a cost-model or pricing bug (e.g. the stale-price bug would
show as a large one-sided entry-price divergence on the affected session). This
is the check that would auto-catch that class instead of a human spotting it on a
single ticker.

Per matched trade (a sim trade whose broker entry actually filled):
    d_return_pct = sim.return_pct − broker.return_pct      (modeled minus real)
    entry_bps    = (broker.entry − sim.entry) / sim.entry × 1e4   (signed)

Aggregated overall, by session, and as a time series by entry date. Thin until
paper fills accumulate — it reports the matched-trade count and degrades to a
"needs more fills" message rather than pretending precision on n=1.

Usage:  python -m src.analysis.tracking_error
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import List, Optional


def _key(t: dict):
    """Stable identity to match a sim trade to its broker-view twin."""
    return (t.get("trade_id") or t.get("recommendation_id")
            or (t.get("ticker"), t.get("entry_datetime")))


def _f(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def compute_tracking_error(trades: List[dict]) -> dict:
    """Sim-vs-broker tracking-error report.

    Returns ``{n_matched, per_trade:[...], overall, by_session, by_date, verdict}``
    where ``overall``/``by_session`` carry mean/median/std of the return gap and
    entry-price gap, and ``by_date`` is the per-entry-date mean return gap (a
    time series). ``per_trade`` rows pair the sim and broker numbers."""
    from src.performance.broker_view import build_broker_trades

    broker_by_key = {_key(b): b for b in build_broker_trades(trades or [])}

    rows = []
    for t in trades or []:
        b = broker_by_key.get(_key(t))
        if b is None:
            continue
        sim_ret, brk_ret = _f(t.get("return_pct")), _f(b.get("return_pct"))
        sim_entry, brk_entry = _f(t.get("entry_price")), _f(b.get("entry_price"))
        if sim_ret is None or brk_ret is None:
            continue
        entry_bps = (round((brk_entry - sim_entry) / sim_entry * 1e4, 1)
                     if sim_entry and brk_entry and sim_entry > 0 else None)
        rows.append({
            "ticker":      t.get("ticker"),
            "entry_date":  t.get("entry_date"),
            "session":     t.get("entry_session") or "rth",
            "sim_return":  round(sim_ret, 3),
            "broker_return": round(brk_ret, 3),
            "d_return":    round(sim_ret - brk_ret, 3),
            "sim_entry":   sim_entry,
            "broker_entry": brk_entry,
            "entry_bps":   entry_bps,
            "status":      b.get("status"),
        })

    if not rows:
        return {"n_matched": 0, "per_trade": [], "overall": None,
                "by_session": [], "by_date": [],
                "verdict": "No trades with a matching broker fill yet — tracking "
                           "error needs filled paper/live orders to compare against."}

    def _agg(rs: List[dict]) -> dict:
        dr = [r["d_return"] for r in rs]
        eb = [r["entry_bps"] for r in rs if r["entry_bps"] is not None]
        return {
            "n":              len(rs),
            "mean_d_return":  round(mean(dr), 3),
            "median_d_return": round(sorted(dr)[len(dr) // 2], 3),
            "std_d_return":   round(pstdev(dr), 3) if len(dr) > 1 else 0.0,
            "mean_entry_bps": round(mean(eb), 1) if eb else None,
            "median_entry_bps": round(sorted(eb)[len(eb) // 2], 1) if eb else None,
        }

    by_session = []
    for sess in sorted({r["session"] for r in rows}):
        by_session.append({"session": sess, **_agg([r for r in rows if r["session"] == sess])})

    by_date = []
    for d in sorted({r["entry_date"] for r in rows if r["entry_date"]}):
        drs = [r["d_return"] for r in rows if r["entry_date"] == d]
        by_date.append({"entry_date": d, "n": len(drs), "mean_d_return": round(mean(drs), 3)})

    overall = _agg(rows)
    bias = overall["mean_d_return"]
    note = ("" if abs(bias) < 0.10 else
            f"  ⚠ persistent one-sided gap ({bias:+.2f}%) — possible cost-model/pricing bug "
            "(the sim is systematically "
            + ("optimistic" if bias > 0 else "pessimistic") + " vs real fills).")
    verdict = (f"Matched {len(rows)} trade(s): sim − broker return = {bias:+.2f}% mean, "
               f"entry price gap {overall['mean_entry_bps']} bp mean.{note}")
    return {"n_matched": len(rows), "per_trade": rows, "overall": overall,
            "by_session": by_session, "by_date": by_date, "verdict": verdict}


def _print_report(rep: dict) -> None:
    if not rep["n_matched"]:
        print(rep["verdict"])
        return
    o = rep["overall"]
    print(f"\nSim-vs-broker tracking error — {rep['n_matched']} matched trade(s)")
    print(rep["verdict"])
    print(f"\noverall: Δreturn mean {o['mean_d_return']:+.3f}% · median {o['median_d_return']:+.3f}% · "
          f"std {o['std_d_return']:.3f}%  |  entry gap mean {o['mean_entry_bps']} bp")
    print("\nby session:")
    for s in rep["by_session"]:
        print(f"  {s['session']:<10} n={s['n']:<3} Δreturn {s['mean_d_return']:+.3f}%  "
              f"entry {s['mean_entry_bps']} bp")


def main() -> None:
    import sys
    from src.db import repo
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: render −/⚠ glyphs
    except Exception:
        pass
    repo.set_read_only(True)   # never contend with a running scheduler's write lock
    try:
        trades = repo.load_trades()
    except Exception as e:
        print(f"Could not read trades ({e}).")
        return
    _print_report(compute_tracking_error(trades))


if __name__ == "__main__":
    main()
