"""AN EXTRA ACCOUNT'S TICK (from 2026-10-23; `docs/SECOND_ACCOUNT.md`):

    python -m src.accounts.run --account second

launched by the main scheduler right after the main account's live path (`src.accounts.launch`), with the account's
profile as its environment (`src.accounts.profiles`). It runs the main account's live-path steps, in the same order —
the marks, the books' exits, the account's check (IBKR's liquidation, de-leveraging), the dip fills and the crash buy,
the dip orders, the shorts' entries, the broker sync (`pipeline._live_marks_and_exits` / `_live_entries_and_sync`) —
on ITS OWN ledger and state, from the decisions the main account's tick already made: the scorer's picks, the dip
signals and the crash decision are READ, never made here (no scorer is launched, no borrow file fetched). Then it
writes its summary for the main account's email (`src.accounts.summary`). One at a time (its ``busy.lock``), bounded
by ``extra_account_timeout_seconds``.

It refuses to run unless its environment names the account: a process with the main account's settings must never
trade the main ledger a second time — nor open an IBKR session: an extra account is CALCULATED from the main account
(the user, 2026-10-23), its broker a dry run (`guard`).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

MAIN_DB = "data/llm_trader.db"


def _setup_logging(name: str) -> None:
    from src.log_redaction import redaction_filter
    logger.remove()
    logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<8} | {message}", level="INFO", filter=redaction_filter)
    logger.add(f"logs/account_{name}_{{time:YYYY-MM-DD}}.log", rotation="1 day", retention="7 days", level="DEBUG",
               filter=redaction_filter)


def _same_file(a: str, b: str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


# the broker modes an extra account may run: never a gateway session (the main account's gateway is its own)
SAFE_BROKER_MODES = ("dry_run", "off")


def guard(name: str) -> None:
    """Refuse an environment that does not name account ``name``: its name, a state root of its own, a ledger
    database that is not the main account's and a broker that never opens an IBKR session."""
    from config.settings import settings
    from src.accounts import paths
    if name == paths.MAIN:
        raise SystemExit("the main account trades in the scheduler's own process, never here")
    if paths.name() != name:
        raise SystemExit(f"the environment names the account {paths.name()!r}, not {name!r}")
    if paths.is_main():
        raise SystemExit(f"the account {name!r} has no state root of its own (ACCOUNT_STATE_ROOT)")
    if _same_file(settings.db_path, MAIN_DB):
        raise SystemExit(f"the account {name!r} would write the main account's ledger ({settings.db_path})")
    mode = str(getattr(settings, "broker_mode", "") or "off").lower()
    if mode not in SAFE_BROKER_MODES:
        raise SystemExit(f"the account {name!r} would open an IBKR session (broker_mode={mode}): an extra account "
                         f"is calculated, its orders a dry run")


def _watchdog(seconds: float) -> None:
    """A hung tick (a broker sync, a price fetch) ends the process: the next tick starts clean."""
    def _kill():
        logger.critical(f"[account] the tick ran past {seconds:.0f}s — ending the process")
        os._exit(3)
    t = threading.Timer(seconds, _kill)
    t.daemon = True
    t.start()


def tick(name: str) -> dict:
    """One tick of account ``name`` (its environment already applied). Returns its summary."""
    from config.settings import settings
    from src.accounts import summary as SUM
    from src.accounts import paths
    guard(name)
    start = datetime.now(timezone.utc)
    run_id = f"{name}_{start.strftime('%Y-%m-%d_%H%M%S')}"
    lock = paths.state("busy.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{os.getpid()} {start.isoformat()}", encoding="utf-8")
    try:
        from src import pipeline as P
        if not bool(getattr(settings, "enable_live_path_first", False)):
            logger.warning("[account] enable_live_path_first is off: the account trades nothing")
            return {}
        logger.info(f"[account] {name}: tick {run_id}")
        sold = P._live_marks_and_exits()
        report = P._live_entries_and_sync(run_id, None, None, dip_closed=sold)
        P._persist_live_report(run_id, report)
        out = SUM.write(run_id, report, started=start)
        logger.info(f"[account] {name}: done in {time.time() - start.timestamp():.0f}s — {out.get('line')}")
        return out
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="An extra account's tick (docs/SECOND_ACCOUNT.md)")
    ap.add_argument("--account", required=True)
    a = ap.parse_args()
    _setup_logging(a.account)
    from config.settings import settings
    _watchdog(float(getattr(settings, "extra_account_timeout_seconds", 900)))
    try:
        tick(a.account)
    except SystemExit as e:
        logger.critical(f"[account] {a.account}: refused — {e}")
        return 2
    except Exception as e:                                         # noqa: BLE001
        logger.exception(f"[account] {a.account}: the tick failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
