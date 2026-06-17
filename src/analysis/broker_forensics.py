"""Broker execution forensics over the persisted ``broker_orders`` /
``broker_reconciles`` tables.

The reconciler writes one event row per order submission / fill repair / settle
action (``repo.insert_broker_report``), so the paper phase accumulates a durable
execution record. This module turns it into the four questions that validate the
execution design before real money rides on it:

  1. Slippage distribution by session — is the LMT cap (20 bp RTH / 80 bp
     extended) actually achievable, or are fills landing outside it?
  2. Fill-rate vs kill-rate — the settle-or-kill design cancels an order that
     doesn't fill in ~a minute; what fraction fill vs get killed/expired?
  3. Drift-event frequency — how often do broker positions diverge from the
     ledger (the stale-snapshot race the settle pass was built to close)?
  4. Reject-reason breakdown — what is the broker actually rejecting, and why?

All compute functions take DataFrames so they're unit-testable without a DB; the
``load_*`` helpers + ``main`` are the live-DB convenience layer.

Usage:  python -m src.analysis.broker_forensics
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

# How a raw broker order status/event maps to a fill-outcome bucket. Terminal
# buckets (everything except "working") count toward the fill rate denominator.
_WORKING = {"Submitted", "PreSubmitted", "PendingSubmit"}
_KILLED = {"SETTLE_KILL", "STALE_CANCELLED", "STALE_CANCEL", "EXPIRED", "Cancelled"}
_FAILED = {"SUBMIT_FAILED", "Inactive"}
_SKIPPED = {"DUPLICATE_REF_NOT_SUBMITTED", "NOTHING_TO_CLOSE", "DRYRUN"}


def _session_of(iso: object) -> str:
    """ET session for a broker ``submitted_at`` UTC ISO string (rth/extended/
    overnight), or 'unknown' when it can't be parsed."""
    if not iso or not isinstance(iso, str):
        return "unknown"
    try:
        from src.performance.market_calendar import current_session
        return current_session(datetime.fromisoformat(iso))
    except Exception:
        return "unknown"


def _outcome(status: object, filled_qty: object) -> str:
    """Bucket one order row into a fill outcome."""
    s = str(status or "").strip()
    try:
        fq = int(filled_qty or 0)
    except (TypeError, ValueError):
        fq = 0
    if s == "Filled" or fq > 0:
        return "filled"
    if s in _WORKING:
        return "working"
    if s in _KILLED:
        return "killed"
    if s in _FAILED:
        return "failed"
    if s in _SKIPPED:
        return "skipped"
    return "other"


def slippage_by_session(orders: pd.DataFrame) -> pd.DataFrame:
    """Per-session slippage stats over filled legs with a recorded
    ``slippage_bps`` (positive = adverse). Columns: session, n, mean_bps,
    median_bps, p90_bps, max_bps."""
    if orders is None or orders.empty or "slippage_bps" not in orders.columns:
        return pd.DataFrame(columns=["session", "n", "mean_bps", "median_bps", "p90_bps", "max_bps"])
    df = orders.copy()
    df["slippage_bps"] = pd.to_numeric(df["slippage_bps"], errors="coerce")
    df = df[df["slippage_bps"].notna()]
    if "filled_qty" in df.columns:
        df = df[pd.to_numeric(df["filled_qty"], errors="coerce").fillna(0) > 0]
    if df.empty:
        return pd.DataFrame(columns=["session", "n", "mean_bps", "median_bps", "p90_bps", "max_bps"])
    df["session"] = df.get("submitted_at").map(_session_of) if "submitted_at" in df.columns else "unknown"
    rows = []
    for sess, g in df.groupby("session"):
        s = g["slippage_bps"]
        rows.append({
            "session": sess, "n": int(len(s)),
            "mean_bps": round(float(s.mean()), 1),
            "median_bps": round(float(s.median()), 1),
            "p90_bps": round(float(s.quantile(0.9)), 1),
            "max_bps": round(float(s.max()), 1),
        })
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)


def fill_outcomes(orders: pd.DataFrame) -> dict:
    """Counts by fill outcome + the fill rate (filled / terminal outcomes,
    excluding still-working and skipped/no-op rows)."""
    if orders is None or orders.empty:
        return {"counts": {}, "fill_rate": None, "n_terminal": 0}
    fq = orders.get("filled_qty")
    outcomes = [
        _outcome(st, q)
        for st, q in zip(orders.get("status", [None] * len(orders)),
                         fq if fq is not None else [0] * len(orders))
    ]
    counts: dict = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    terminal = {k: v for k, v in counts.items() if k not in ("working", "skipped")}
    n_terminal = sum(terminal.values())
    fill_rate = round(100.0 * counts.get("filled", 0) / n_terminal, 1) if n_terminal else None
    return {"counts": counts, "fill_rate": fill_rate, "n_terminal": n_terminal}


def reject_reasons(orders: pd.DataFrame) -> pd.DataFrame:
    """Failed/rejected rows grouped by error message. Columns: reason, n."""
    empty = pd.DataFrame(columns=["reason", "n"])
    if orders is None or orders.empty or "status" not in orders.columns:
        return empty
    bad = orders[orders["status"].isin(_FAILED) | (orders.get("ok") == False)]  # noqa: E712
    if bad.empty:
        return empty
    reasons = bad.get("error").fillna("(no message)") if "error" in bad.columns else pd.Series(["(no message)"] * len(bad))
    out = (reasons.replace("", "(no message)").value_counts()
           .rename_axis("reason").reset_index(name="n"))
    return out


def drift_frequency(reconciles: Optional[pd.DataFrame]) -> dict:
    """Drift summary over reconcile runs: how often broker positions diverged
    from the ledger (the stale-snapshot race)."""
    if reconciles is None or reconciles.empty or "n_drift" not in reconciles.columns:
        return {"n_runs": 0, "runs_with_drift": 0, "total_drift_events": 0, "pct_runs_with_drift": None}
    nd = pd.to_numeric(reconciles["n_drift"], errors="coerce").fillna(0)
    n_runs = int(len(nd))
    runs_with = int((nd > 0).sum())
    return {
        "n_runs": n_runs,
        "runs_with_drift": runs_with,
        "total_drift_events": int(nd.sum()),
        "pct_runs_with_drift": round(100.0 * runs_with / n_runs, 1) if n_runs else None,
    }


def compute_forensics(orders: pd.DataFrame, reconciles: Optional[pd.DataFrame] = None) -> dict:
    """Full forensics bundle from the two broker tables (as DataFrames)."""
    return {
        "n_orders":          0 if orders is None else int(len(orders)),
        "slippage_by_session": slippage_by_session(orders),
        "fill_outcomes":     fill_outcomes(orders),
        "reject_reasons":    reject_reasons(orders),
        "drift":             drift_frequency(reconciles),
    }


# ── live-DB convenience layer ───────────────────────────────────────────────

def load_broker_orders() -> pd.DataFrame:
    from src.db import repo
    try:
        return repo.fetch_df("SELECT * FROM broker_orders")
    except Exception:
        return pd.DataFrame()


def load_broker_reconciles() -> pd.DataFrame:
    from src.db import repo
    try:
        return repo.fetch_df("SELECT * FROM broker_reconciles")
    except Exception:
        return pd.DataFrame()


def _print_report(rep: dict) -> None:
    if not rep["n_orders"]:
        print("No broker_orders recorded yet (broker_mode off or no fills).")
        return
    print(f"\nBroker forensics — {rep['n_orders']} order event(s)\n")
    fo = rep["fill_outcomes"]
    print(f"Fill rate: {fo['fill_rate']}%  (over {fo['n_terminal']} terminal orders)   counts: {fo['counts']}")
    d = rep["drift"]
    print(f"Drift: {d['runs_with_drift']}/{d['n_runs']} runs had drift "
          f"({d['pct_runs_with_drift']}%), {d['total_drift_events']} event(s) total\n")
    sl = rep["slippage_by_session"]
    print("Slippage by session (bp, +=adverse):")
    print(sl.to_string(index=False) if not sl.empty else "  (no filled legs with slippage)")
    rr = rep["reject_reasons"]
    print("\nReject reasons:")
    print(rr.to_string(index=False) if not rr.empty else "  (none)")


def main() -> None:
    import sys
    from src.db import repo
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows console: render glyphs
    except Exception:
        pass
    repo.set_read_only(True)   # never contend with a running scheduler's write lock
    _print_report(compute_forensics(load_broker_orders(), load_broker_reconciles()))


if __name__ == "__main__":
    main()
