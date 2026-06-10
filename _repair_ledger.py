"""One-time ledger repair + drift adoption (2026-06-10 incident).

Restores what the stale-test wipe destroyed and adopts the four orphaned IBKR
paper positions back under management:

  1. Legacy history — cache/trades.json (27 trades, 2026-05-18..31) is merged
     back in. Restored OPEN trades get a sentinel broker_order_id so the
     reconciler never submits fresh entries for stale May signals.
  2. Drift adoption — VGT / VLUE / CPSH ledger rows are rebuilt from the
     surviving log lines (exact entry price/time/size) joined with their
     original recommendation rows in DuckDB (rec_id, rationale, method
     scores). TRUP comes back via the legacy restore (its row predates June).
     All four carry broker_status='RESTORED_ADOPTED': entries are never
     resubmitted, the fill-refresh pass ignores them, and when a signal
     closes one, the exit sizes from the LIVE broker position — flattening
     the real paper holding.
  3. Today's real broker-linked trades are kept verbatim; a legacy OPEN trade
     whose ticker is also open today is dropped (can't hold both).

Safety: DRY-RUN by default (prints the plan); pass --apply to write. A
timestamped backup of the DB file is taken before any write. Refuses to run
twice (aborts if pre-2026-06-10 trades already exist). Run it while no
pipeline tick is in flight — ideally with the scheduler stopped — because
ticks full-replace the trades table and would clobber a mid-tick repair.

Usage:
    python _repair_ledger.py            # dry run — show the plan
    python _repair_ledger.py --apply    # write to the production DB
    python _repair_ledger.py --db PATH [--apply]   # rehearse against a copy
"""
import argparse
import json
import shutil
from datetime import datetime, timezone

from config.settings import settings

# Tickers currently held at IBKR with no ledger row (today's drift report).
DRIFTED = {"VGT", "VLUE", "CPSH", "TRUP"}

# Rebuilt from logs/llm_trader_2026-06-08.log / -09.log "[tracker] Opened" lines.
ADOPTIONS = [
    {"ticker": "VGT",  "run_id": "2026-06-08_190008", "entry_date": "2026-06-08",
     "entry_price": 117.63, "multiplier": 1.217,
     "decision": "2026-06-08T19:03:23+00:00", "entry_dt": "2026-06-08T19:03:23+00:00"},
    {"ticker": "VLUE", "run_id": "2026-06-08_190008", "entry_date": "2026-06-08",
     "entry_price": 195.65, "multiplier": 0.884,
     "decision": "2026-06-08T19:03:24+00:00", "entry_dt": "2026-06-08T19:03:24+00:00"},
    {"ticker": "CPSH", "run_id": "2026-06-09_140027", "entry_date": "2026-06-09",
     "entry_price": 8.04, "multiplier": 1.0,
     "decision": "2026-06-09T14:05:44+00:00", "entry_dt": "2026-06-09T14:05:44+00:00"},
]

WIPE_DATE = "2026-06-10"   # trades entered on/after this survived the incident


def _fetch_recommendation(run_id: str, ticker: str) -> dict:
    """The original recommendation row — rec_id, rationale, method scores."""
    from src.db import repo
    df = repo.fetch_df(
        "SELECT rec_id, type, direction, action, confidence, time_horizon, "
        "rationale, dominant_method, methods_agreeing, contributing_scores "
        "FROM recommendations WHERE run_id = ? AND ticker = ?",
        [run_id, ticker], read_only=False)
    if df.empty:
        raise SystemExit(f"ABORT: no recommendation row for {ticker} in run {run_id} "
                         "— the adoption constants don't match the DB.")
    r = df.iloc[0]
    return {
        "rec_id": r["rec_id"], "type": r["type"], "direction": r["direction"],
        "action": r["action"], "confidence": float(r["confidence"]),
        "time_horizon": r["time_horizon"], "rationale": r["rationale"],
        "dominant_method": r["dominant_method"],
        "methods_agreeing": json.loads(r["methods_agreeing"] or "[]"),
        "method_scores": json.loads(r["contributing_scores"] or "{}"),
    }


def _ref_close(ticker: str, entry_date: str):
    """Cached close on entry_date, for the NAV walk's split-adjustment anchor."""
    try:
        from src.data.cache import load_ohlcv
        df = load_ohlcv(ticker)
        if df is None or df.empty:
            return None, None
        on_date = df[df.index.strftime("%Y-%m-%d") == entry_date]
        if on_date.empty:
            return None, None
        return float(on_date["Close"].iloc[-1]), entry_date
    except Exception:
        return None, None


def _build_adopted_trade(spec: dict) -> dict:
    rec = _fetch_recommendation(spec["run_id"], spec["ticker"])
    ref_close, ref_date = _ref_close(spec["ticker"], spec["entry_date"])
    return {
        "ticker": spec["ticker"],
        "run_id": spec["run_id"],
        "recommendation_id": rec["rec_id"],
        "type": rec["type"],
        "action": rec["action"],
        "direction": rec["direction"],
        "confidence": rec["confidence"],
        "time_horizon": rec["time_horizon"],
        "position_size_multiplier": spec["multiplier"],
        "confidence_size_multiplier": None,    # log shows only the final size
        "correlation_size_multiplier": None,
        "entry_date": spec["entry_date"],
        "entry_datetime": spec["entry_dt"],
        "decision_datetime": spec["decision"],
        "entry_price": spec["entry_price"],
        "entry_ref_close": ref_close,
        "entry_ref_close_date": ref_date,
        "rationale": rec["rationale"],
        "current_price": spec["entry_price"],    # refreshed on the next tick
        "current_price_datetime": spec["decision"],
        "return_pct": 0.0,
        "weighted_return_pct": 0.0,
        "days_held": 0,
        "exit_date": None, "exit_datetime": None, "exit_decision_datetime": None,
        "exit_price": None, "exit_ref_close": None, "exit_ref_close_date": None,
        "exit_reason": None,
        "status": "OPEN",
        "method_scores": rec["method_scores"],
        "methods_agreeing": rec["methods_agreeing"],
        "dominant_method": rec["dominant_method"],
        "signal_at_entry": None,    # aggregator snapshot lost — decay check #3 skips
        "max_favorable_excursion": 0.0, "mfe_date": spec["entry_date"],
        "max_adverse_excursion": 0.0, "mae_date": spec["entry_date"],
        "pattern_at_entry": None, "pattern_score_at_entry": None,
        # Broker adoption: never resubmit the entry; never fill-refresh; on
        # close, the exit sizes from the live IBKR position and flattens it.
        "broker_order_id": "ADOPTED_DRIFT_2026-06-10",
        "broker_status": "RESTORED_ADOPTED",
        "restored_from": "logs+recommendations (2026-06-10 wipe repair)",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true",
                   help="actually write (default: dry-run, print the plan only)")
    p.add_argument("--db", default=None, help="override DB path (rehearsal on a copy)")
    args = p.parse_args()
    if args.db:
        settings.db_path = args.db

    from src.db import repo

    current = repo.load_trades()
    if any((t.get("entry_date") or "") < WIPE_DATE for t in current):
        raise SystemExit("ABORT: ledger already contains pre-2026-06-10 trades — "
                         "the repair appears to have run already.")

    legacy = json.loads(open("cache/trades.json", encoding="utf-8").read())
    fake = [t for t in current if not t.get("entry_date")]
    real = [t for t in current if t.get("entry_date")]
    open_now = {t["ticker"] for t in real if t.get("status") == "OPEN"}

    restored, dropped_conflicts = [], []
    for t in legacy:
        if t.get("status") == "OPEN":
            if t["ticker"] in open_now:
                dropped_conflicts.append(t["ticker"])
                continue
            t = dict(t)
            t["broker_order_id"] = "LEGACY_RESTORED"
            # TRUP's paper short is live at IBKR (initial convergence entered it).
            t["broker_status"] = ("RESTORED_ADOPTED" if t["ticker"] in DRIFTED
                                  else "RESTORED_NOT_SUBMITTED")
            t["restored_from"] = "cache/trades.json (2026-06-10 wipe repair)"
        restored.append(t)

    adopted = [_build_adopted_trade(spec) for spec in ADOPTIONS
               if spec["ticker"] not in open_now]

    merged = restored + adopted + real

    print(f"plan ({'APPLY' if args.apply else 'DRY RUN'}) on {settings.db_path}:")
    print(f"  legacy restored:   {len(restored)} (open conflicts dropped: {dropped_conflicts or 'none'})")
    for spec in ADOPTIONS:
        a = next((x for x in adopted if x["ticker"] == spec["ticker"]), None)
        line = (f"adopt {a['action']} {a['ticker']} @ {a['entry_price']} "
                f"({a['entry_date']}, size={a['position_size_multiplier']}x, "
                f"rec={a['recommendation_id']})") if a else f"{spec['ticker']} SKIPPED (open today)"
        print(f"  {line}")
    print(f"  fake rows dropped: {[t.get('ticker') for t in fake] or 'none'}")
    print(f"  today kept:        {len(real)}")
    print(f"  merged total:      {len(merged)}")

    if not args.apply:
        print("\nDry run only — nothing written. Re-run with --apply to write.")
        return

    backup = settings.db_path.replace(
        ".db", f"_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.db")
    shutil.copyfile(settings.db_path, backup)
    print(f"\nbackup: {backup}")

    repo.save_trades(merged)

    check = repo.load_trades()
    assert len(check) == len(merged), "post-write count mismatch"
    assert all(t.get("entry_date") for t in check), "malformed row survived"
    open_tickers = sorted(t["ticker"] for t in check if t.get("status") == "OPEN")
    adopted_ok = sorted(t["ticker"] for t in check
                        if t.get("broker_status") == "RESTORED_ADOPTED")
    print(f"VERIFIED: {len(check)} rows | open: {open_tickers}")
    print(f"adopted under management (drift will clear): {adopted_ok}")
    print("entry_date range:", min(t["entry_date"] for t in check),
          "->", max(t["entry_date"] for t in check))


if __name__ == "__main__":
    main()
