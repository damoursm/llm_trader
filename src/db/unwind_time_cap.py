"""One-time correction: undo time-cap (``holding_period``) exits.

The fixed 5-session holding-period auto-close was removed — positions now exit
only when their thesis deteriorates (signal decay / regime / reversal). This
migration brings existing history in line with that rule:

  * Any trade closed with ``exit_reason == "holding_period"`` is **reopened**
    (status → OPEN, exit fields cleared) so it is treated as a continuous hold.
  * The old code often opened a same-direction **re-entry** moments after the
    cap close (same ticker/action, entry within minutes of the cap exit). That
    re-entry is a duplicate of the now-reopened position, so it is **removed** —
    leaving each ticker with a single continuous position carrying its original
    cost basis.

Idempotent: once there are no ``holding_period`` closes left, a re-run is a
no-op. After running, the next pipeline tick's ``monitor_open_positions`` will
close any reopened position whose rationale has since deteriorated.

    python -m src.db.unwind_time_cap
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from src.db import repo

# A re-entry opened within this window after a cap exit is treated as the "roll"
# of the same position (the old code reopened within a couple of seconds).
_ROLL_WINDOW_SECONDS = 600

_EXIT_FIELDS = (
    "exit_date", "exit_datetime", "exit_decision_datetime", "exit_price",
    "exit_ref_close", "exit_ref_close_date", "exit_reason",
)


def _parse(dt) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(dt))
    except (TypeError, ValueError):
        return None


def unwind() -> None:
    trades = repo.load_trades()
    capped = [t for t in trades if t.get("exit_reason") == "holding_period"]
    if not capped:
        logger.info("[unwind_time_cap] No holding_period closes — nothing to do.")
        return

    remove = set()       # id() of roll re-entries to drop
    reopened = 0
    for c in capped:
        c_exit = _parse(c.get("exit_datetime"))

        # Locate the roll re-entry: same ticker + action, entered shortly after
        # this cap exit (and not itself a cap close we're already reopening).
        roll = None
        for r in trades:
            if r is c or id(r) in remove:
                continue
            if r.get("ticker") != c.get("ticker") or r.get("action") != c.get("action"):
                continue
            if r.get("exit_reason") == "holding_period":
                continue
            r_entry = _parse(r.get("entry_datetime"))
            if c_exit and r_entry and 0 <= (r_entry - c_exit).total_seconds() <= _ROLL_WINDOW_SECONDS:
                roll = r
                break

        # Reopen the original position (its entry is the true cost basis).
        for f in _EXIT_FIELDS:
            c[f] = None
        c["status"] = "OPEN"
        reopened += 1

        if roll is not None:
            remove.add(id(roll))
            logger.info(
                f"[unwind_time_cap] {c['ticker']}: reopened original entry "
                f"{c.get('entry_datetime')} @ {c.get('entry_price')}; dropped roll re-entry "
                f"{roll.get('entry_datetime')} @ {roll.get('entry_price')}"
            )
        else:
            logger.info(f"[unwind_time_cap] {c['ticker']}: reopened (no roll re-entry found)")

    new_trades = [t for t in trades if id(t) not in remove]

    # Invariant guard: never leave two OPEN positions on one ticker.
    seen = set()
    for t in new_trades:
        if t.get("status") == "OPEN":
            if t["ticker"] in seen:
                logger.warning(
                    f"[unwind_time_cap] WARNING: {t['ticker']} has >1 OPEN position after unwind — review manually."
                )
            seen.add(t["ticker"])

    repo.save_trades(new_trades)
    logger.info(
        f"[unwind_time_cap] Done. Reopened {reopened} position(s), removed "
        f"{len(remove)} roll re-entr(ies). Total trades {len(trades)} -> {len(new_trades)}."
    )


if __name__ == "__main__":
    unwind()
