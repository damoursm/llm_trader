"""AN EXTRA ACCOUNT'S SUMMARY (from 2026-10-23; `docs/SECOND_ACCOUNT.md`): written by the account's own process at
the end of each tick (`src.accounts.run`) to ``summary.json`` under its state root — its equity against its $50,000,
its open positions, today's closed trades and the broker's mode — and read by the MAIN account's email as one line in
the scorer banner's notes (`health`): the main process never opens the account's ledger database."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings

STALE_HOURS = 6.0


def path_for(name: str) -> Path:
    """The summary file of account ``name`` (read by the main account: its root from the main settings)."""
    root = str(getattr(settings, f"{name}_account_state_root", "") or "")
    return Path(root) / "summary.json"


def write(run_id: str, report: Optional[dict], started: Optional[datetime] = None) -> dict:
    """The account's summary after its tick (in the account's own process: its settings, its ledger)."""
    from src.accounts import paths
    from src.performance import live_account as LA
    from src.performance.tracker import _load_trades
    now = datetime.now(timezone.utc)
    trades = _load_trades()
    st = LA.state(trades, now)
    initial = float(getattr(settings, "sel_short_account_initial", 0.0) or 0.0)
    eq = float(st.get("equity") or 0.0)
    today = now.astimezone().date().isoformat()
    open_pos: List[Dict] = []
    closed_today: List[Dict] = []
    from src.performance import sim_account as SA
    for t in trades:
        if not SA.counts(t):
            continue
        row = {"ticker": t.get("ticker"), "book": t.get("entry_mechanism"), "arm": t.get("sel_arm"),
               "side": t.get("direction"), "entry": t.get("entry_price"), "entry_date": t.get("entry_date")}
        if t.get("status") == "OPEN":
            open_pos.append(dict(row, current=t.get("current_price"), ret_pct=t.get("return_pct")))
        elif str(t.get("exit_date") or "")[:10] == today:
            closed_today.append(dict(row, exit=t.get("exit_price"), ret_pct=t.get("return_pct"),
                                     why=t.get("exit_reason")))
    mode = (report or {}).get("mode") if isinstance(report, dict) else None
    line = (f"{paths.name()} account ${eq:,.0f} ({100 * (eq / initial - 1) if initial else 0.0:+.2f}% since its "
            f"start): {len(open_pos)} open" + (f" ({', '.join(str(p['ticker']) for p in open_pos[:6])})"
                                              if open_pos else "")
            + f", {len(closed_today)} closed today; "
            + ("calculated, no IBKR orders" if mode in (None, "dry_run", "off") else f"broker {mode}"))
    out = {"account": paths.name(), "run_id": run_id, "at": now.isoformat(),
           "started": started.isoformat() if started else None, "equity": eq, "initial": initial,
           "state": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in st.items()},
           "open": open_pos, "closed_today": closed_today, "broker_mode": mode, "line": line,
           "broker_problems": list((report or {}).get("problems") or []) if isinstance(report, dict) else []}
    p = paths.state("summary.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=str, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return out


def health() -> Dict[str, List[str]]:
    """The main account's email: one note line per extra account (its last summary; stale after `STALE_HOURS`).
    Never a problem: an extra account's trouble never alarms the main account's banner."""
    notes: List[str] = []
    if not bool(getattr(settings, "enable_extra_accounts", False)):
        return {"problems": [], "notes": notes}
    for name in list(getattr(settings, "extra_accounts", []) or []):
        p = path_for(name)
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            notes.append(f"{name} account: no summary yet")
            continue
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(rec["at"])).total_seconds() / 3600.0
        except (KeyError, ValueError, TypeError):
            age = None
        line = str(rec.get("line") or f"{name} account")
        if age is not None and age > STALE_HOURS:
            line += f" (last tick {age:.0f} h ago)"
        notes.append(line)
    return {"problems": [], "notes": notes}
