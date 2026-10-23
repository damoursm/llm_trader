"""THE LIVE ACCOUNT'S MARGIN CHECK, AS IBKR'S (PREREG174 / 175; the user 2026-10-21: "production and backtest needs to
liquidate like IBKR does").

IBKR checks an account's margin in real time and, below the maintenance requirement, liquidates the most recently opened
positions in 100-share lots until it is back at the requirement (`live_account.plan_liquidation`). Production checks at
its ticks, so each check reads what happened since the last one: every 30-minute bar of each funded position COMPLETED
since the last bar it read (Polygon's, 04:00-20:00 ET, adjusted to today's terms — each bar read once; a bar not yet
served is read at the next check) and the live price. Every position then moves together from its price at the last
check to its adverse extreme (the bars' high for a short, low for a long, or the live price when worse) — the
simulator's M_PATH (`account.simulate`'s ``liquidate_path``, `Mimic.margin_bars` "all"). The end-of-day Reg T check
runs once a session, at the first check from 15:50 ET (before 20:00) of a market day, at the live prices.

The state (`STATE`): each position's price at the last check (the path's start), the end of the last bar read and its
split factor then; the session whose Reg T check ran. A position the state does not hold starts at its entry (entered
since the last check: its bars after the entry); with no state at all (the first check after the deploy) the path starts
at the marks and reads no bar before it. Each liquidation is journaled in `JOURNAL/<ET day>.jsonl`."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from config import settings
from src.accounts import paths as _acct

ET = ZoneInfo("America/New_York")
STATE = _acct.state(Path("cache") / "live_account" / "margin_state.json")      # this account's (`src.accounts.paths`)
JOURNAL = _acct.state(Path("cache") / "live_account" / "liquidations")
BAR = timedelta(minutes=30)
REGT_FROM, REGT_TO = (15, 50), (20, 0)


def mode() -> str:
    """``live_account_liquidation``: "ibkr" (IBKR's liquidation, the default from 2026-10-21) or "all" (every funded
    position closed at a margin call, the rule before)."""
    m = str(getattr(settings, "live_account_liquidation", "ibkr") or "ibkr").lower()
    return m if m in ("ibkr", "all") else "ibkr"


def _parse(ts) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def load_state() -> dict:
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
        return st if isinstance(st, dict) else {}
    except Exception:                                              # noqa: BLE001 — missing / unreadable: a fresh start
        return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, STATE)


def journal(now: datetime, event: dict) -> None:
    try:
        JOURNAL.mkdir(parents=True, exist_ok=True)
        with open(JOURNAL / f"{now.astimezone(ET).date().isoformat()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
    except Exception as e:                                         # noqa: BLE001 — a journal never stops the check
        logger.warning(f"[live_account] liquidation journal not written: {e}")


def latest_bar_end(now: datetime) -> datetime:
    """The latest instant a stored 30-minute bar (04:00-20:00 ET) can have ended by ``now``: the last half-hour mark,
    or the last 20:00 ET between 20:00 and 04:30 (the overnight hours hold no bar; the night's late prints are the
    04:00 bar's)."""
    et = now.astimezone(ET)
    floor = et.replace(minute=(et.minute // 30) * 30, second=0, microsecond=0)
    if et.hour >= 20:
        floor = et.replace(hour=20, minute=0, second=0, microsecond=0)
    elif (et.hour, et.minute) < (4, 30):
        floor = (et - timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
    return floor.astimezone(timezone.utc)


def bars_since(ticker: str, after: datetime, now: datetime) -> Optional[List[Tuple[datetime, float, float]]]:
    """[(bar end, high, low)] of the ticker's 30-minute bars (04:00-20:00 ET, Polygon's, adjusted) ending in
    (``after``, ``now``]; None when Polygon could not be read."""
    if latest_bar_end(now) <= after:
        return []
    from src.data import polygon_client as pc
    d0 = after.astimezone(ET).date().isoformat()
    d1 = now.astimezone(ET).date().isoformat()
    try:
        df = pc.get_intraday_bars_range(ticker, d0, d1, adjusted=True, rth_only=False)
    except Exception as e:                                         # noqa: BLE001
        logger.warning(f"[live_account] {ticker}: bars since {after.isoformat()} unreadable ({e})")
        return None
    if df is None:
        return None
    out = []
    for ts, row in df.iterrows():
        end = ts.to_pydatetime() + BAR
        if after < end <= now:
            out.append((end, float(row["High"]), float(row["Low"])))
    return out


def regt_due(now: datetime, st: dict) -> bool:
    """The end-of-day Reg T check is due: a market day, from 15:50 to 20:00 ET, not yet run this session."""
    if not bool(getattr(settings, "live_account_regt_close", True)):
        return False
    from src.performance.market_calendar import is_market_day
    et = now.astimezone(ET)
    if not is_market_day(et.date()):
        return False
    if not (REGT_FROM <= (et.hour, et.minute) < REGT_TO):
        return False
    return st.get("regt_done") != et.date().isoformat()


def splittable(trade: dict) -> bool:
    """A position may be liquidated in part unless its entry order may still fill at IBKR (sent, not final): splitting a
    leg IBKR is still working would leave the lot and the remainder disagreeing with the order's fills."""
    if not trade.get("broker_order_id"):
        return True
    from src.broker.reconcile import _RESTORED_STATUSES, _TERMINAL_STATUSES
    s = trade.get("broker_status") or ""
    return s in _TERMINAL_STATUSES or s in _RESTORED_STATUSES


def inputs(trades: Sequence[dict], now: datetime, st: dict, path: bool = True) -> Tuple[List[dict], dict, List[str]]:
    """(the plan's positions, each trade's new state, notes) for the account's funded open positions at ``now`` —
    their marks the trades' ``current_price``; with ``path`` their price at the last check and the adverse extreme of
    the bars completed since (`bars_since`)."""
    from src.performance import live_account as LA
    seen = st.get("trades") or {}
    since = _parse(st.get("checked_at"))
    pos, new, notes = [], {}, []
    for t in LA.funded_open(trades):
        key = str(t.get("recommendation_id") or "")
        mark = float(t.get("current_price") or t.get("entry_price") or 0.0)
        n = LA.shares_of(t)
        if not key or n <= 0 or mark <= 0:
            continue
        side = LA.side_of(t)
        f_now = float(t.get("split_factor") or 1.0)
        prev, bar_from = mark, None
        s = seen.get(key)
        entry = _parse(t.get("entry_datetime"))
        if s is not None and s.get("px"):
            prev = float(s["px"]) * f_now / float(s.get("f") or 1.0)
            bar_from = _parse(s.get("bar_end"))
        elif since is not None and entry is not None:       # not in the state: entered since the last check
            if entry >= since:
                prev, bar_from = float(t.get("entry_price") or mark), entry
            else:                                           # (held before it, never recorded: from the last check on)
                prev, bar_from = mark, since
        ext, last = mark, bar_from
        if path and bar_from is not None:
            bars = bars_since(str(t["ticker"]), bar_from, now)
            if bars is None:
                notes.append(f"{t['ticker']}: bars unreadable — judged at its mark")
            elif bars:
                hi = max(b[1] for b in bars)
                lo = min(b[2] for b in bars)
                ext = max(mark, hi) if side < 0 else min(mark, lo)
                last = max(b[0] for b in bars)
        else:
            prev = mark
        if bar_from is None:
            last = latest_bar_end(now)                     # the first check: no bar before it is read
        pos.append({"key": key, "side": side, "n": n, "mark": mark, "prev": prev, "ext": ext,
                    "order": (str(t.get("entry_datetime") or ""), key), "whole": not splittable(t),
                    "maint": (lambda q, p, _t=t: LA.maint_of(_t, q, p)), "regt": (lambda q, p, _t=t: LA.regt_of(_t, q, p)),
                    "ticker": t["ticker"]})
        new[key] = {"bar_end": last.isoformat() if last is not None else None, "f": f_now}
    return pos, new, notes
