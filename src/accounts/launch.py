"""THE EXTRA ACCOUNTS' LAUNCH (from 2026-10-23; `docs/SECOND_ACCOUNT.md`): the main account's tick starts, right
after its own live path (its broker report persisted), one process per extra account — ``python -m src.accounts.run
--account <name>`` with the account's profile as its environment (`src.accounts.profiles.env_for`) — and never waits
for it: the account trades on the decisions the main tick just made, on its own ledger (calculated: its orders
a dry run, IBKR's answers the main account's). A
process still running (its ``busy.lock`` fresh) is never doubled; its output goes to ``logs/account_<name>.log``.
`enable_extra_accounts` (off by default) and `extra_accounts` (the names) decide; every failure is a warning, never
the main tick's."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from config.settings import settings

_ACTIVE: Dict[str, subprocess.Popen] = {}


def _busy(name: str) -> bool:
    root = str(getattr(settings, f"{name}_account_state_root", "") or "")
    lock = Path(root) / "busy.lock"
    try:
        return lock.exists() and time.time() - lock.stat().st_mtime < float(settings.extra_account_timeout_seconds)
    except OSError:
        return False


def launch_all(now: Optional[datetime] = None) -> List[str]:
    """Start each enabled extra account's tick (non-blocking). Returns the names launched."""
    if not bool(getattr(settings, "enable_extra_accounts", False)):
        return []
    from src.accounts import profiles
    from src.accounts import paths
    if not paths.is_main():                          # an extra account never launches others
        return []
    launched = []
    for name in list(getattr(settings, "extra_accounts", []) or []):
        try:
            prev = _ACTIVE.get(name)
            if prev is not None and prev.poll() is None:
                logger.warning(f"[account] {name}: the previous tick (pid {prev.pid}) still runs — not launching")
                continue
            if _busy(name):
                logger.warning(f"[account] {name}: another process holds its busy.lock — not launching")
                continue
            if not str(getattr(settings, f"{name}_account_fresh_start", "") or ""):
                logger.warning(f"[account] {name}: no fresh start set ({name}_account_fresh_start) — not launching")
                continue
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            env.update(profiles.env_for(name))
            Path("logs").mkdir(exist_ok=True)
            log = open(Path("logs") / f"account_{name}.log", "a", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-m", "src.accounts.run", "--account", name],
                                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
            log.close()                              # the child keeps its own handle
            _ACTIVE[name] = proc
            launched.append(name)
            logger.info(f"[account] {name}: tick launched (pid {proc.pid}) at "
                        f"{(now or datetime.now(timezone.utc)).isoformat(timespec='seconds')}")
        except Exception as e:                                     # noqa: BLE001
            logger.warning(f"[account] {name}: launch failed (fail-soft): {e}")
    return launched
