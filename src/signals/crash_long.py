"""THE CRASH-REBOUND BOOK — QQQ's capitulations, traded in TQQQ (user 2026-10-11: PREREG86 "Crash book only", then
PREREG88 "deploy the QQQ decisions, trade TQQQ to replace the QQQ decisions, trade QQQ").

The rule (PREREG86's CR1, decided on `crash_long_symbol`, QQQ): at the previous session's close QQQ closed BELOW its
`crash_long_trend_sessions` (200) session average with a 2-session RSI (EWM, alpha 1/2 — the dip book's `rsi2`) under
`crash_long_rsi_max` (5) -> BUY `crash_long_trade_symbol` (TQQQ, 3x the Nasdaq-100) at the regular-hours open (within
`crash_long_entry_window_minutes` of 09:30); SELL it at the open after QQQ's first close above its
`crash_long_exit_sma_sessions` (5) session average since the entry session (its own close counts; every close since
the entry is read, so a missed day delays the exit instead of losing it), or after `crash_long_max_hold_sessions` (10)
sessions — the dip book's exit. One position at a time. Both decisions read the DECISION symbol's closes, never the
traded fund's (the trade stamps ``crash_decision_symbol``). THE AFTER-HOURS EXIT (PREREG166's XC, from the 2026-10-20
evening; `crash_long_pm_exit`): the same exit is taken at the first after-hours tick at/after 16:30 ET of the session
whose close triggers it (`pm_exit_check`, `tracker.monitor_crash_long_pm_exit`) instead of the next open; a tick
missed until 20:00 ET leaves it to the open.

The daily bars are the dip book's construction (`dip_long.daily_bars`: the deep 30-minute store, one row per ET
session); the decision reads the closes through D-1 only (`tests/test_no_lookahead.py`).

SIZE (`live_account.size_crash`): 1/`crash_long_account_slices` (4) of the LIVE ACCOUNT's equity in the traded fund,
within the long side's cap (the longs' value at most the equity) and the margin room at the fund's requirement
`crash_long_margin_rate` (0.75: FINRA's 3 x 25% for a 3x fund, initial and maintenance; stamped
``crash_margin_rate``); short of room it sells the dip book's latest longs WHOLE, as a short does
(`live_account.crash_give_way_plan` — the backtest's quality 0.2 above the dip book's 0.13); a margin call closes it
with the account's other positions; de-leveraging does not sell it.

Evidence (memory/crash-book-leveraged-2026-10.md, PREREG88; PREREG86 for the rule): QQQ's decisions traded in TQQQ
1988-2020 106 trades, +2.75% a trade (95% +0.83..+4.76; TQQQ synthetic before 2010), against +1.44% for QQQ itself; the worst trade -36% (October 1987, synthetic). In the live account (five draws) +1.6 / +2.0 log points a year
over trading QQQ on sets A / B, every draw positive, the drawdown no deeper, no margin call — the tightened add rule's
pass, mostly from the 3x exposure (at the same exposure, 1/12, it does not pass).

Files (`crash_long_dir`): signals/<day>.json (the day's decision), entries/<day>.jsonl, exits/<day>.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.accounts import paths as _acct
from src.signals import dip_long as DL

MECHANISM = "crash_long"
ET = DL.ET
FINAL_OUTCOMES = ("opened", "no_slot", "account_full", "account_too_small", "held", "holding", "no_room")


def enabled() -> bool:
    return bool(getattr(settings, "enable_crash_long", False))


def root() -> Path:
    return Path(settings.crash_long_dir)


def symbol() -> str:
    """The DECISION symbol (QQQ): its closes decide the buy and the sell."""
    return str(settings.crash_long_symbol).upper()


def trade_symbol() -> str:
    """The fund bought and sold (TQQQ); the decision symbol when the setting is empty."""
    return str(settings.crash_long_trade_symbol or settings.crash_long_symbol).upper()


def margin_rate() -> float:
    """The traded fund's margin requirement, initial and maintenance, as a share of its value (0: Reg T's 50% /
    25%) — the live account's and the simulator's (`Strategy.house_init` / ``house_maint``)."""
    return max(0.0, float(settings.crash_long_margin_rate or 0.0))


def in_entry_window(now: Optional[datetime] = None) -> bool:
    """Regular hours, within `crash_long_entry_window_minutes` of the 09:30 open (the study bought at the open)."""
    now = now or datetime.now(timezone.utc)
    if not DL.in_rth(now):
        return False
    et = now.astimezone(ET)
    mins = et.hour * 60 + et.minute + et.second / 60.0 - 570.0
    return 0.0 <= mins <= float(settings.crash_long_entry_window_minutes)


# ── the rule ─────────────────────────────────────────────────────────────────

def stats_at(daily: Optional[pd.DataFrame], through: date) -> Optional[dict]:
    """The rule's numbers at the close of the last session on or before ``through`` (nothing later is read):
    ``session``, ``close``, the trend average, ``rsi2``; None below `crash_long_trend_sessions` daily bars."""
    if daily is None or daily.empty:
        return None
    d = daily[daily.index <= pd.Timestamp(through)]
    nt = int(settings.crash_long_trend_sessions)
    if len(d) < nt:
        return None
    c = d["c"].to_numpy(float)
    return {"session": pd.Timestamp(d.index[-1]).date().isoformat(), "close": float(c[-1]),
            "sma_trend": float(np.mean(c[-nt:])), "rsi2": float(DL.rsi2(c)[-1]), "bars": int(len(c))}


def qualifies(st: Optional[dict]) -> bool:
    """The entry rule on the numbers at the close of D-1."""
    if not st:
        return False
    return (math.isfinite(st["sma_trend"]) and st["close"] < st["sma_trend"]
            and math.isfinite(st["rsi2"]) and st["rsi2"] < float(settings.crash_long_rsi_max))


def _daily() -> Optional[pd.DataFrame]:
    return DL._daily(symbol())


def signals_path(d: date) -> Path:
    return root() / "signals" / f"{d.isoformat()}.json"


def signal_for(d: date, extend: bool = True) -> dict:
    """Session ``d``'s decision from the close of D-1 — computed once and kept (``signals/<d>.json``); a store
    still missing D-1's close after one extension is returned unkept (``stale``, no signal) and retried."""
    p = signals_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    _acct.decide_here(f"the crash decision of {d}")       # an extra account reads it, never computes it
    prev = DL.prev_session(d)
    sym = symbol()

    def last_stored(daily):
        if daily is None or daily.empty:
            return None
        x = daily.index[daily.index <= pd.Timestamp(prev)]
        return pd.Timestamp(x.max()).date() if len(x) else None
    daily = _daily()
    if last_stored(daily) != prev and extend:
        DL._extend([sym], prev)
        daily = _daily()
    st = stats_at(daily, prev)
    stale = last_stored(daily) != prev
    out = {"day": d.isoformat(), "symbol": sym, "trade_symbol": trade_symbol(), "signal_session": prev.isoformat(),
           "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stale": bool(stale),
           "signal": bool(qualifies(st)) and not stale, **(st or {})}
    if not stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    return out


def entry_day(trade: dict) -> Optional[date]:
    raw = trade.get("crash_entry_day") or str(trade.get("entry_date") or "")[:10]
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def rebound_session(daily: Optional[pd.DataFrame], entry: date, through: date) -> Optional[dict]:
    """The FIRST session from ``entry`` through ``through`` whose close is above its exit average."""
    if daily is None or daily.empty:
        return None
    d = daily[daily.index <= pd.Timestamp(through)]
    if d.empty:
        return None
    c = d["c"]
    sma = c.rolling(int(settings.crash_long_exit_sma_sessions)).mean()
    hit = d.index[(d.index >= pd.Timestamp(entry)) & (c > sma).to_numpy()]
    if len(hit) == 0:
        return None
    s = hit[0]
    return {"session": pd.Timestamp(s).date(), "close": float(c.loc[s]), "sma_exit": float(sma.loc[s])}


def exit_check(trade: dict, d: date) -> Tuple[Optional[str], dict]:
    """Does session ``d``'s open close this long? ``crash_rebound`` once a close of the DECISION symbol (the
    trade's ``crash_decision_symbol``, else `crash_long_symbol` — never the traded fund's) from the entry session
    through D-1 is above its exit average, ``crash_time`` once it has held `crash_long_max_hold_sessions` sessions
    through D-1; else None (``info`` carries the numbers; ``stale`` when the store had no close for D-1)."""
    e = entry_day(trade)
    prev = DL.prev_session(d)
    info: dict = {"through": prev.isoformat()}
    if e is None or prev < e:
        return None, info
    held = DL.sessions_held(e, prev)
    info["held"] = held
    tk = str(trade.get("crash_decision_symbol") or symbol()).upper()
    info["decision_symbol"] = tk
    daily = DL._daily(tk)
    last = None if daily is None or daily.empty else pd.Timestamp(daily.index[daily.index <= pd.Timestamp(prev)].max())
    if last is None or pd.isna(last) or last.date() != prev:
        DL._extend([tk], prev)
        daily = DL._daily(tk)
        last = None if daily is None or daily.empty else pd.Timestamp(daily.index[daily.index <= pd.Timestamp(prev)].max())
    if last is None or pd.isna(last) or last.date() != prev:
        info["stale"] = True
        logger.warning(f"[crash_long] {tk}: no stored close for {prev} — the rebound rule is judged on the closes "
                       f"the store has")
    hit = rebound_session(daily, e, prev)
    if hit is not None:
        info.update(rebound_session=hit["session"].isoformat(), close=hit["close"], sma_exit=hit["sma_exit"])
        return "crash_rebound", info
    if held >= int(settings.crash_long_max_hold_sessions):
        return "crash_time", info
    return None, info


# ── THE AFTER-HOURS EXIT (PREREG166's XC; user 2026-10-20: "Deploy all three.") ─────────────────────────────────
# The exit the next session's open would take (`exit_check`) is taken at the first after-hours tick at/after 16:30 ET
# of the session whose close triggers it — the backtest's 16:00-16:30 bar close (`mimic.pm_exit`). The deep store holds
# the sessions through the one before (the next pre-open run extends it), so the session's own regular-hours close is
# Polygon's last regular-hours 30-minute bar of it — the store's construction (`dip_long.daily_bars`: the last bar's
# close). A tick that cannot read it, or every tick missed until 20:00 ET, leaves the sale to the next open, as before.
PM_EXIT_FROM = 16 * 60 + 30                 # 16:30 ET
PM_EXIT_UNTIL = 20 * 60                     # the extended session's end


def pm_exit_on() -> bool:
    return bool(getattr(settings, "crash_long_pm_exit", False))


def in_pm_exit_window(now: Optional[datetime] = None) -> bool:
    """A session day's after-hours instant from 16:30 to 20:00 ET."""
    now = now or datetime.now(timezone.utc)
    et = now.astimezone(ET)
    m = et.hour * 60 + et.minute
    return DL.is_session(et.date()) and PM_EXIT_FROM <= m < PM_EXIT_UNTIL


def session_close(ticker: str, s: date) -> Optional[float]:
    """Session ``s``'s regular-hours close of ``ticker``: the close of Polygon's 15:30-16:00 ET 30-minute bar (None
    when the session's bars cannot be read or do not reach it)."""
    bars = DL.session_bars(ticker, s)
    if bars is None or bars.empty:
        return None
    st, _ = DL._bar_times(bars)
    last = st[-1].tz_convert(ET)
    if last.date() != s or (last.hour, last.minute) != (15, 30):
        return None
    v = float(pd.to_numeric(bars["Close"].iloc[-1], errors="coerce"))
    return v if math.isfinite(v) and v > 0 else None


def pm_exit_check(trade: dict, s: date) -> Tuple[Optional[str], dict]:
    """At session ``s``'s after-hours tick: the exit the next session's open would take (`exit_check`'s rule through
    the close of ``s``) — ``crash_rebound`` once a close of the decision symbol from the entry session through ``s``
    is above its exit average, ``crash_time`` once the long has held `crash_long_max_hold_sessions` sessions through
    ``s``; else None. ``info`` carries the numbers (``pm`` True; ``no_close`` when ``s``'s close cannot be read)."""
    e = entry_day(trade)
    info: dict = {"through": s.isoformat(), "pm": True}
    if e is None or s < e:
        return None, info
    held = DL.sessions_held(e, s)
    info["held"] = held
    tk = str(trade.get("crash_decision_symbol") or symbol()).upper()
    info["decision_symbol"] = tk
    c = session_close(tk, s)
    if c is None:
        info["no_close"] = True
        logger.warning(f"[crash_long] {tk}: no regular-hours close for {s} yet — the after-hours exit is judged at "
                       f"the next tick")
        return None, info
    daily = DL._daily(tk)
    before = daily[daily.index < pd.Timestamp(s)] if daily is not None and not daily.empty else None
    row = pd.DataFrame({"o": [np.nan], "h": [np.nan], "l": [np.nan], "c": [c], "v": [np.nan]},
                       index=pd.DatetimeIndex([pd.Timestamp(s)]))
    full = row if before is None or before.empty else pd.concat([before[["o", "h", "l", "c", "v"]], row])
    info["close_today"] = c
    hit = rebound_session(full, e, s)
    if hit is not None:
        info.update(rebound_session=hit["session"].isoformat(), close=hit["close"], sma_exit=hit["sma_exit"])
        return "crash_rebound", info
    if held >= int(settings.crash_long_max_hold_sessions):
        return "crash_time", info
    return None, info


# ── journals ─────────────────────────────────────────────────────────────────

def entries_path(d: date) -> Path:
    return _acct.state(root() / "entries" / f"{d.isoformat()}.jsonl")   # this account's (`src.accounts.paths`)


def exits_path(d: date) -> Path:
    return _acct.state(root() / "exits" / f"{d.isoformat()}.jsonl")


def _append(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def journal_entry(d: date, sig: dict, outcome: str, **kw) -> None:
    _append(entries_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "ticker": sig.get("symbol"), "trade_symbol": sig.get("trade_symbol") or trade_symbol(),
                              "outcome": outcome,
                              **{k: sig.get(k) for k in ("close", "sma_trend", "rsi2", "signal_session")}, **kw})


def journal_exit(d: date, trade: dict, reason: str, info: dict) -> None:
    _append(exits_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "ticker": trade.get("ticker"), "reason": reason,
                            "trade": trade.get("recommendation_id"), **info})


def settled(d: date) -> Dict[str, str]:
    """Session ``d``'s signal already settled for good (a missing price is retried)."""
    return {str(r.get("ticker")): str(r.get("outcome")) for r in _read_jsonl(entries_path(d))
            if r.get("outcome") in FINAL_OUTCOMES}


def trade_id(d: date, ticker: str) -> str:
    return hashlib.sha1(f"crash|{d.isoformat()}|{ticker}".encode("utf-8")).hexdigest()[:16]


def health(trades: Optional[Sequence[dict]] = None, now: Optional[datetime] = None) -> dict:
    """Problems for the email's scorer banner: a crash long held past its limit (its exit missed), a day's
    decision not computed by the end of the entry window."""
    now = now or datetime.now(timezone.utc)
    problems: List[str] = []
    if not enabled():
        return {"problems": problems}
    d = DL.today_et(now)
    if DL.is_session(d) and DL.in_rth(now) and not in_entry_window(now) and not signals_path(d).exists():
        et = now.astimezone(ET)
        if et.hour * 60 + et.minute > 570 + float(settings.crash_long_entry_window_minutes):
            problems.append(f"crash_long: no decision computed for {d}")
    for t in trades or []:
        if t.get("status") != "OPEN" or t.get("entry_mechanism") != MECHANISM:
            continue
        e = entry_day(t)
        if e is not None and DL.sessions_held(e, d) > int(settings.crash_long_max_hold_sessions) + 2:
            problems.append(f"crash_long: {t.get('ticker')} held past its {settings.crash_long_max_hold_sessions}-"
                            f"session limit (since {e})")
    return {"problems": problems}
