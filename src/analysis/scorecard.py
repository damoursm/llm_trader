"""Recent-period SCORECARD — one windowed read of what's working vs what isn't,
across ENTRY signals, EXIT signals/quality, broker EXECUTION, and realized TRADES.

Pulls together the existing validation surfaces (signal_panel IC, exit_panel,
exit_quality, broker_forensics, tracking_error, the realized performance table)
into a single narrative so "what changed and did it help" is answerable in one run.

    python -m src.analysis.scorecard [days=7]

Notes on horizons: entry-signal IC needs the forward return to AGE (a 5-day IC
can't exist for a signal younger than 5 days), so ENTRY IC is computed over ALL
accrued history — a short window is structurally empty. EXIT / EXECUTION / TRADES
are windowed to the last ``days`` (where the recent changes actually landed).
Everything is ~weeks of ONE regime — directional, not definitive.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pandas as pd

from src.db import repo


def _closed(trades: List[dict]) -> List[dict]:
    return [t for t in trades if t.get("status") == "CLOSED"]


def _fmt(v, nd=3):
    return round(v, nd) if isinstance(v, float) else v


def _entry_ic(lines: List[str]) -> None:
    """Per-method 5-day Spearman IC over the whole accrued signals panel — the
    unbiased 'does this signal predict forward returns' read, incl. combined_score."""
    lines.append("\n== ENTRY — signal IC (5d, all accrued history) ==")
    try:
        from src.analysis.signal_panel import build_panel, compute_ic
        panel = build_panel(horizons=(1, 5, 10), days=None)
        ic = compute_ic(panel, horizons=(1, 5, 10))
        if ic is None or ic.empty:
            lines.append("  (no forward returns yet)")
            return
        d = ic[[c for c in ["method", "n_5d", "ic_5d", "hit_5d", "icir_5d"] if c in ic.columns]]
        d = d.dropna(subset=["ic_5d"]).sort_values("ic_5d", ascending=False)
        combined = d[d["method"] == "combined_score"]
        top = d.head(8); bot = d.tail(6)
        if not combined.empty:
            r = combined.iloc[0]
            verdict = "PREDICTIVE" if r["ic_5d"] > 0.02 else ("~noise" if r["ic_5d"] > -0.02 else "ANTI-PREDICTIVE")
            lines.append(f"  AGGREGATE combined_score: IC {r['ic_5d']:+.3f}  hit {r['hit_5d']:.0f}%  ICIR {r['icir_5d']:+.2f}  -> {verdict}")
        lines.append("  Top predictive:")
        for _, r in top.iterrows():
            lines.append(f"    {r['method']:<18} IC {r['ic_5d']:+.3f}  hit {r['hit_5d']:.0f}%  ICIR {r['icir_5d']:+.2f}  (n={int(r['n_5d'])})")
        lines.append("  Most ANTI-predictive:")
        for _, r in bot.iloc[::-1].iterrows():
            lines.append(f"    {r['method']:<18} IC {r['ic_5d']:+.3f}  hit {r['hit_5d']:.0f}%  ICIR {r['icir_5d']:+.2f}  (n={int(r['n_5d'])})")
    except Exception as e:
        lines.append(f"  ERR {e!r}")


def _exit_signals(lines: List[str], days: int) -> None:
    lines.append(f"\n== EXIT — exit-signal IC (1d, last {days}d) ==")
    try:
        from src.analysis.exit_panel import compute_exit_method_perf
        em = compute_exit_method_perf(days=days, min_n=5)
        if em is not None and not em.empty and "ic_1d" in em.columns:
            d = em[["method", "views", "n_1d", "win_1d", "ic_1d"]].dropna(subset=["ic_1d"])
            for _, r in d.sort_values("ic_1d", ascending=False).head(8).iterrows():
                lines.append(f"    {r['method']:<16} IC {r['ic_1d']:+.3f}  win {r['win_1d']:.0f}%  (n={int(r['n_1d'])})")
        else:
            lines.append("  (empty)")
    except Exception as e:
        lines.append(f"  ERR {e!r}")


def _exit_reasons_and_quality(lines: List[str], trades: List[dict]) -> None:
    lines.append("\n== EXIT — realized P&L by close reason ==")
    try:
        from src.performance.tracker import compute_exit_reason_perf
        for row in compute_exit_reason_perf():
            lines.append(f"    {str(row.get('exit_reason')):<22} n={row.get('trades', 0):<3} "
                         f"win {row.get('win_rate', 0):.0f}%  avg {row.get('avg_return', 0):+.2f}%  "
                         f"compound {row.get('compound_return', 0):+.2f}%")
    except Exception as e:
        lines.append(f"  ERR {e!r}")
    lines.append("\n== EXIT — MFE/MAE capture (did we keep the gains?) ==")
    try:
        from src.analysis.exit_quality import compute_exit_quality
        eq = compute_exit_quality(_closed(trades))
        lines.append(f"  n={eq.get('n')}  avg capture {eq.get('avg_capture')}  "
                     f"gave-back-most-MFE {eq.get('pct_gave_back_most_mfe')}%  near-MAE {eq.get('pct_exited_near_mae')}%")
        lines.append(f"  {eq.get('verdict', '')}")
    except Exception as e:
        lines.append(f"  ERR {e!r}")


def _execution(lines: List[str], days: int, cutoff: dt.date, trades: List[dict]) -> None:
    lines.append(f"\n== EXECUTION — broker fills/errors by day (last {days}d) ==")
    try:
        from src.analysis.broker_forensics import load_broker_orders
        orders = load_broker_orders()
        if not orders.empty and "submitted_at" in orders.columns:
            orders["_day"] = orders["submitted_at"].astype(str).str[:10]
            for day, g in orders[orders["_day"] >= str(cutoff)].groupby("_day"):
                st = g["status"].astype(str)
                filled = int(st.str.contains("Filled").sum())
                err = int(st.isin(["ERROR", "Rejected"]).sum() + st.str.contains("FAILED").sum())
                lines.append(f"    {day}: orders={len(g):<4} filled={filled:<4} error/reject={err}")
        else:
            lines.append("  (no broker orders)")
    except Exception as e:
        lines.append(f"  ERR {e!r}")
    lines.append("\n== EXECUTION — sim vs real-fill tracking error ==")
    try:
        from src.analysis.tracking_error import compute_tracking_error
        lines.append(f"  {compute_tracking_error(trades).get('verdict', '')}")
    except Exception as e:
        lines.append(f"  ERR {e!r}")


def _trades(lines: List[str], days: int) -> None:
    lines.append(f"\n== TRADES — realized performance (last {days}d) ==")
    try:
        from src.performance.tracker import get_performance_for_email
        perf = get_performance_for_email(window_days=days)
        pm = perf.get("portfolio_metrics", {})
        lines.append(f"  compound {days}d window: {pm.get('return_1w', pm.get('compound_inception'))}%")
        rows = perf.get("performance_table", [])
        keep = {"total", "direction", "session"}
        for row in rows:
            if row.get("group") in keep:
                lines.append(f"    {str(row.get('label','')):<26} n={row.get('trades',0):<3} "
                             f"win {row.get('win_rate',0):.0f}%  avg {row.get('avg_return',0):+.2f}%  "
                             f"compound {row.get('compound_return',0):+.2f}%")
        # best / worst methods by realized compound
        meth = [r for r in rows if r.get("group") == "method" and (r.get("trades") or 0) >= 5]
        meth.sort(key=lambda r: r.get("compound_return", 0))
        if meth:
            lines.append("  worst methods (realized compound):")
            for r in meth[:4]:
                lines.append(f"    {str(r.get('label','')):<30} compound {r.get('compound_return',0):+.2f}%  win {r.get('win_rate',0):.0f}%")
            lines.append("  best methods (realized compound):")
            for r in meth[-4:][::-1]:
                lines.append(f"    {str(r.get('label','')):<30} compound {r.get('compound_return',0):+.2f}%  win {r.get('win_rate',0):.0f}%")
    except Exception as e:
        lines.append(f"  ERR {e!r}")


def build_scorecard(days: int = 7) -> str:
    """Return the full scorecard as text."""
    repo.set_read_only(True)
    cutoff = dt.date.today() - dt.timedelta(days=days)
    trades = repo.load_trades()
    lines: List[str] = [
        "=" * 68,
        f"  SCORECARD — window last {days}d (>= {cutoff}) | trades={len(trades)}",
        "  (entry IC = all accrued history; ~weeks of one regime — directional)",
        "=" * 68,
    ]
    _entry_ic(lines)
    _exit_signals(lines, days)
    _exit_reasons_and_quality(lines, trades)
    _execution(lines, days, cutoff, trades)
    _trades(lines, days)
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    days = int((argv or sys.argv[1:] or [7])[0])
    print(build_scorecard(days))


if __name__ == "__main__":
    main()
