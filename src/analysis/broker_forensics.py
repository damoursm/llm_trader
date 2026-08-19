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

from datetime import datetime, timedelta, timezone
from typing import Optional

import re

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


def filter_orders(orders: pd.DataFrame, days: Optional[int] = None,
                  session: Optional[str] = None,
                  direction: Optional[str] = None) -> pd.DataFrame:
    """Window / session / direction slice of the broker-order log.

    ``session`` is derived from ``submitted_at`` — the moment the order went to
    the broker, which is the only session that means anything for execution
    (the fill rate ranges 56.6% in RTH to 4.8% overnight, so this is the single
    most informative cut on the Execution tab). ``current_session`` returns the
    coarse ``rth | extended | overnight``, so the dashboard's finer premarket /
    afterhours choices both map onto ``extended`` rather than silently matching
    nothing.

    ``direction`` maps long→BUY / short→SELL on the order's own side, NOT the
    parent position's: an EXIT of a long is a SELL order, and for execution
    questions ("did this order cross?") the side that was sent is what matters.
    """
    if orders is None or orders.empty:
        return orders
    df = orders
    if days and "submitted_at" in df.columns:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
        ts = pd.to_datetime(df["submitted_at"], errors="coerce", utc=True)
        df = df[ts.notna() & (ts >= cutoff)]
    if session and "submitted_at" in df.columns:
        want = {"premarket": "extended", "afterhours": "extended"}.get(session, session)
        df = df[df["submitted_at"].map(_session_of) == want]
    if direction and "side" in df.columns:
        want_side = "BUY" if str(direction).lower() == "long" else "SELL"
        df = df[df["side"].astype(str).str.upper() == want_side]
    return df


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


RETRY_SUFFIX = re.compile(r"-r\d+$")


def base_ref(client_ref: object) -> str:
    """The INTENDED TRADE behind a client_ref.

    Each resubmission gets a fresh ref by appending ``-rN`` (``abc``, ``abc-r1``,
    ``abc-r2`` …; exits are ``abc-exit`` then ``abc-exit-r1``). Stripping that
    suffix collapses every retry of one intended trade back onto a single key —
    the difference between "how often does an order fill" and "how often do we
    get the trade on", which measured 20.1% and 75.8% on the same data.
    """
    return RETRY_SUFFIX.sub("", str(client_ref or "").strip())


def fill_rate_by_attempt(orders: pd.DataFrame) -> dict:
    """The fill rate at BOTH units, because they answer different questions.

    * ``per_retry`` — one row per client_ref, i.e. per ORDER PLACED. "When we
      put an order in, how often did it fill?" This is the execution-quality
      number, and it is low by design: settle-or-kill gives an order one tick,
      re-anchors it every ~6 s, and cancels it at ``broker_settle_seconds``.
    * ``per_trade`` — one row per BASE ref, retries pooled. "Of the trades we
      decided to make, how many did we actually get on?" This is the number
      that matters for whether the strategy is being executed at all.

    They differ enormously and reporting only one misleads in whichever
    direction it happens to point: measured 20.1% per retry versus 75.8% per
    trade, over 4,460 orders behind 1,185 intended trades (3.76 submissions
    each, with a tail reaching ``-r85``). Quoting 20% alone reads as a broken
    execution leg; quoting 76% alone hides that we place five orders to get one
    fill.

    Note what is NOT the unit here: ``fill_outcomes`` counts order EVENTS
    (SUBMIT, each SETTLE_REANCHOR, the terminal row), which is wrong for both
    questions and additionally drops every row still marked ``Submitted`` as
    "working" — mostly re-anchors of intents that were later killed, i.e. the
    clearest failures. That is why it reads ~2× high.

    A fill is ``status == "Filled"`` or ``filled_qty > 0``: a partial counts,
    because the position WAS established, just smaller. Rows that never reached
    the broker (duplicate ref, nothing to close, dry run) are not attempts.

    Returns ``{per_retry: {n, filled, rate}, per_trade: {n, filled, rate,
    avg_retries}, by_intent: {ENTRY: {...}, EXIT: {...}}}`` — ``by_intent`` on
    the per-TRADE basis, since an unfilled exit (a position still open that the
    ledger believes is closed) is the failure with real consequences.
    """
    blank = {"n": 0, "filled": 0, "rate": None}
    empty = {"per_retry": dict(blank), "per_trade": dict(blank, avg_retries=None),
             "by_intent": {}}
    if orders is None or orders.empty or "client_ref" not in orders.columns:
        return empty

    df = orders.copy()
    df["_ref"] = df["client_ref"].astype(str).str.strip()
    df = df[df["_ref"].ne("") & df["_ref"].ne("None")]
    if "status" in df.columns:
        df = df[~df["status"].astype(str).str.strip().isin(_SKIPPED)]
    if df.empty:
        return empty

    fq = pd.to_numeric(df.get("filled_qty"), errors="coerce").fillna(0)
    df["_filled"] = df.get("status").astype(str).str.strip().eq("Filled") | (fq > 0)
    df["_base"] = df["_ref"].map(base_ref)

    def _pack(n, filled, extra=None):
        out = {"n": int(n), "filled": int(filled),
               "rate": round(100.0 * filled / n, 1) if n else None}
        if extra:
            out.update(extra)
        return out

    retry = df.groupby("_ref").agg(filled=("_filled", "max"), base=("_base", "first"))
    per_retry = _pack(len(retry), retry["filled"].sum())

    trade = retry.groupby("base")["filled"].agg(["max", "size"])
    per_trade = _pack(len(trade), trade["max"].sum(),
                      {"avg_retries": round(float(trade["size"].mean()), 2) if len(trade) else None})

    by_intent = {}
    if "intent" in df.columns:
        intent_of = df.groupby("_base")["intent"].first()
        j = trade.join(intent_of.rename("intent"))
        for label, sub in j.dropna(subset=["intent"]).groupby("intent"):
            by_intent[str(label)] = _pack(
                len(sub), sub["max"].sum(),
                {"avg_retries": round(float(sub["size"].mean()), 2)})

    return {"per_retry": per_retry, "per_trade": per_trade, "by_intent": by_intent}


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


def compute_forensics(orders: pd.DataFrame, reconciles: Optional[pd.DataFrame] = None,
                      days: Optional[int] = None, session: Optional[str] = None,
                      direction: Optional[str] = None) -> dict:
    """Full forensics bundle from the two broker tables (as DataFrames).

    ``days`` / ``session`` / ``direction`` slice the ORDER log (see
    ``filter_orders``). The drift block is deliberately NOT sliced: it is
    per-reconcile-RUN, not per-order, so there is nothing coherent to filter it
    by — a run either found an unexplained position or it did not. The filters
    therefore describe the order-derived blocks only, and the UI says so."""
    orders = filter_orders(orders, days=days, session=session, direction=direction)
    return {
        "n_orders":          0 if orders is None else int(len(orders)),
        "slippage_by_session": slippage_by_session(orders),
        "fill_outcomes":     fill_outcomes(orders),
        # The headline fill rate. `fill_outcomes` counts EVENTS and reads ~2x
        # high; this counts intended trades per tick. Both are kept — the event
        # mix is genuinely useful for seeing WHERE orders die.
        "fill_rate":         fill_rate_by_attempt(orders),
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
