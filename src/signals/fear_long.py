"""THE FEAR BOOK — the dip book's $1B+ signals whose implied volatility sits in the top fifth of its own past year,
bought as the capitulation book's limit orders, 1/6 of the live account each, at most 4 open, giving way first
(PREREG219-226; the user 2026-10-27: "Deploy to production the $1B, 1/6 each, 4 slots."; from the 2026-10-28 session;
runbook docs/FEAR_LONG.md).

FEAR (`ivp_at`, PREREG219's `h219.pct_at` exactly): for a signal whose signal session is S (the close the dip rule
reads, the order's first session D the next), IVP = the percentile of the name's 30-day implied volatility closing S —
IBKR's OPTION_IMPLIED_VOLATILITY daily bars — among its closes of the `fear_long_ivp_window` (252) sessions through S
(at least `fear_long_ivp_min_values` (126) values; the last bar within 5 days of S; the window no longer than 252 x 1.6
sessions in calendar time); FEAR = IVP >= `fear_long_ivp_min` (0.8). A name without a series is not FEAR (no order).

THE RULE (`compute_signals`): on session D, every signal of the dip book's day (`dip_long.signals_for`: a common stock
/ ADR that closed above its 200-session average with a 2-session RSI under 10 trading $1B+ a day — its kept, not-top
and meta-model-filtered signals alike) with dv20 >= `fear_long_min_dollar_volume` ($1B) that is FEAR and is not the
dip book's own order of the day (its daily top pick) becomes a BUY LIMIT ORDER at the close minus
`fear_long_limit_atr` (1/2) x its 14-session ATR, alive `fear_long_limit_sessions` (3) sessions: the capitulation
book's order (placed at each window session's opening tick, filled at the open when it opened at/under the limit,
else at the end of the first completed 30-minute bar whose low reaches it, at most what the room allows). Placed IVP
highest first (ties: the deepest RSI2). SELL at the open after the first close above its `fear_long_exit_sma_sessions`
(3) average since the fill session, or after `fear_long_max_hold_sessions` (5) sessions (`exit_check`).

THE IMPLIED VOLATILITY (`prepare` -> `fetch`): IBKR's daily bars are final after S's close; the pre-market of D
(`fear_long_iv_start_et` 08:30 to `fear_long_iv_stop_et` 09:27 ET) a DETACHED process on its own client
(`ibkr_client_id` + `fear_long_iv_client_offset`, read-only) computes the day's candidates — the dip rule's names at
S's close from the deep store, loosely (a superset: the exact signals are the dip book's at 09:30) — and asks IBKR for
each one's history (2 years for a name the store lacks, a month to extend one it holds), one request every
`fear_long_iv_gap_seconds`, held from 2 minutes before to 3 after each :00 / :30; each answer merged into the store
(`fear_long_iv_dir`, data/ibkr_iv/: one file a name, never delete — seeded with PREREG219's 2006-2026 history; a
fetched bar replaces the stored one of its date, and every stored bar an answer revises or no longer serves is
journaled first in ``revisions/<D>.jsonl`` there — P7 of the parity audit of 2026-10-27: the simulator reads this
store, and the live calibration counts the revisions) and its IVP written to ``ivp/<D>.json`` as it comes. The opening
tick reads the table: a signal it lacks is not ordered (``no_iv``). Fail-safe: no table, no order.

THE ACCOUNT (`src/performance/live_account.py`): each order 1/`fear_long_account_slices` (6) of the equity at its
limit within the long side's cap and the room at Reg T's 50%, at most `fear_long_max_open` (4) open with its working
orders; its capital GIVES WAY FIRST — the backtest's quality 0.04, under every other book's: a short, the crash book's
buy, a dip, capitulation or insider entry short of room sells its latest longs whole before any other long
(``fear_gave_way``); de-leveraging sells them with the capitulation and insider books' (``fear_delever``); a margin call
liquidates them as IBKR does (``fear_margin_call``). ONE BOOK WITH THE DIP BOOK for one book per name (PREREG223's M1):
the dip book may buy a name the fear book holds and the fear book a name the dip book holds (two long trades in one
symbol, netted at IBKR); against the capitulation, insider and crash books and the shorts, one book per name holds.

Evidence (PREREG221 / 223 / 224 / 225 / 226): the fearful $1B+ dips 349 trades in 77 months at 3 open, +0.542 %/day
(95% +0.330 .. +0.783), at 4 open 416 trades +0.490 %/day; the $2B+ half's fear premium holds on 2007-2017, the $1-2B
half's only on 2018-2026; in the live account (40 draws a set) 88.21 -> 93.04% a year (+4.83 +- 0.27 pp, every set,
without its best year +1.98 log points), the worst 12 months +8.0 -> +7.9%, the worst draw's drawdown 34.7 -> 36.3%,
the largest one-name long exposure 71% of equity (the dip and fear books together).

Files (`fear_long_dir`, cache/ml/fear_long/): ``ivp/<D>.json`` (the day's implied-volatility table),
``signals/<D>.json`` (the day's fear signals and every dip signal's fate), ``orders/<D>.json`` (the dip book's schema),
``entries/<D>.jsonl``, ``exits/<D>.jsonl``, ``fetch.lock``; the store's journal ``data/ibkr_iv/revisions/<D>.jsonl``
(one line a stored bar a fetch changed: ``revised`` with the old and new values, ``absent``). CLI: ``python -m
src.signals.fear_long --fetch [--day D] [--until HH:MM]`` / ``--signals [--day D] [--write]`` (a preview unless
``--write``) / ``--ivp SYM [--day D]`` / ``--status`` / ``--seed DIR``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.accounts import paths as _acct
from src.performance.books import FEAR_LONG as MECHANISM      # one definition of the book's mechanism
from src.signals import dip_long as DL

ET = DL.ET
LOCK = "fetch.lock"
NO_IV_AT_IBKR = ("no_contract", "no_data", "no_value")         # the name's own lack, not the fetch's: a note
DIP_EXCLUDED_STATUS = ("filtered", "not_top", "superseded")    # dip orders that are journal-only, never the day's pick


def enabled() -> bool:
    return bool(getattr(settings, "enable_fear_long", False))


def first_session() -> Optional[date]:
    """The book's first session (`fear_long_first_session`; empty or unreadable: none)."""
    raw = str(getattr(settings, "fear_long_first_session", "") or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        logger.warning(f"[fear_long] fear_long_first_session {raw!r} unreadable — no first session")
        return None


def live_on(d: date) -> bool:
    """Session ``d`` is the book's: on or after its first session (before it: no fetch, no order, no banner check)."""
    fs = first_session()
    return fs is None or d >= fs


def root() -> Path:
    return Path(settings.fear_long_dir)


def iv_root() -> Path:
    return Path(settings.fear_long_iv_dir)


def _iso(d: date) -> str:
    return d.isoformat()


def _write_json(p: Path, obj) -> None:
    DL._write_json(p, obj)


def _hm(s: str) -> dtime:
    hh, mm = (int(x) for x in str(s).split(":"))
    return dtime(hh, mm)


def in_entry_window(now: Optional[datetime] = None) -> bool:
    """Regular hours, within `fear_long_entry_window_minutes` of the 09:30 open (the orders are placed at the open)."""
    now = now or datetime.now(timezone.utc)
    if not DL.in_rth(now):
        return False
    et = now.astimezone(ET)
    mins = et.hour * 60 + et.minute + et.second / 60.0 - 570.0
    return 0.0 <= mins <= float(settings.fear_long_entry_window_minutes)


# ── THE IMPLIED-VOLATILITY STORE (one file a name: [date, open, high, low, close] rows) ──────────────────────────
# A fetched bar replaces the stored one of its date: the store holds IBKR's latest values, as PREREG219's seed does.
# Every stored bar a fetch CHANGES is journaled first (P7, the parity audit of 2026-10-27: the simulator reads this
# store, so a revision would move the bank's fear trades unseen) — ``revisions/<D>.jsonl`` beside the names' files.

REVISIONS = "revisions"
REVISION_TOL = 1e-9          # a value moved by more than this (relative, at least absolute) is a revision


def iv_path(sym: str, root: Optional[Path] = None) -> Path:
    """The name's file in the store (``root``: another copy of it — the simulator's frozen snapshot)."""
    return (iv_root() if root is None else Path(root)) / f"{sym}.json.gz"


def load_rows(sym: str, root: Optional[Path] = None) -> List[list]:
    p = iv_path(sym, root)
    if not p.exists():
        return []
    try:
        return list(json.loads(gzip.decompress(p.read_bytes()).decode("utf-8")))
    except (OSError, ValueError) as e:
        logger.warning(f"[fear_long] {p} unreadable ({e})")
        return []


def bar_row(r: Sequence) -> list:
    """A bar as the store holds it: [date, open, high, low, close], a missing or non-finite value None."""
    return [str(r[0])[:10]] + [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else float(v)
                               for v in list(r)[1:5]]


def differs(a: Sequence, b: Sequence) -> bool:
    """Two store rows of one date hold different values (`REVISION_TOL`; a value present in one only)."""
    for x, y in zip(list(a)[1:5], list(b)[1:5]):
        if (x is None) != (y is None):
            return True
        if x is not None and abs(float(x) - float(y)) > REVISION_TOL * max(1.0, abs(float(x))):
            return True
    return False


def changes(held: Dict[str, Sequence], answer: Sequence[Sequence]) -> Tuple[List[dict], List[dict]]:
    """What IBKR's ``answer`` (store rows, `bar_row`) changes in the stored bars ``held`` ({date: row}): the bars it
    REVISES ([{"date", "old", "new"}], the four values each) and the stored bars inside its span that it no longer
    serves (``absent``, [{"date", "old"}]: the store keeps them)."""
    revised = [{"date": r[0], "old": list(held[r[0]])[1:5], "new": list(r)[1:5]}
               for r in answer if r[0] in held and differs(held[r[0]], r)]
    dates = {r[0] for r in answer}
    lo, hi = (min(dates), max(dates)) if dates else ("", "")
    absent = [{"date": k, "old": list(held[k])[1:5]} for k in sorted(held) if lo <= k <= hi and k not in dates]
    return revised, absent


def revisions_path(d: date, root: Optional[Path] = None) -> Path:
    return (iv_root() if root is None else Path(root)) / REVISIONS / f"{_iso(d)}.jsonl"


def load_revisions(d: date, root: Optional[Path] = None) -> List[dict]:
    """The journal of session ``d``'s fetch: one record a stored bar it changed (`merge_rows`)."""
    p = revisions_path(d, root)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def _journal(sym: str, revised: Sequence[dict], absent: Sequence[dict], d: date, at: Optional[str]) -> None:
    p = revisions_path(d)
    p.parent.mkdir(parents=True, exist_ok=True)
    at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(p, "a", encoding="utf-8") as fh:
        for kind, xs in (("revised", revised), ("absent", absent)):
            for x in xs:
                fh.write(json.dumps({"at": at, "day": _iso(d), "sym": sym, "kind": kind, **x}) + "\n")


def iv_series(sym: str, rows: Optional[List[list]] = None) -> Optional[pd.Series]:
    """The name's implied-volatility CLOSES by date (finite and positive) — PREREG219's `h219.iv_series`."""
    rows = load_rows(sym) if rows is None else rows
    if not rows:
        return None
    s = pd.Series({pd.Timestamp(r[0]): float(r[4]) for r in rows if r[4] is not None})
    s = s[np.isfinite(s) & (s > 0)].sort_index()
    return s if len(s) else None


def ivp_at(s: Optional[pd.Series], S, win: Optional[int] = None,
           min_n: Optional[int] = None) -> Optional[Tuple[float, float]]:
    """(the percentile of the value at ``S`` among the last ``win`` bars through ``S``, the value) or None — the bar must
    be within 5 days of S, at least ``min_n`` values, the window no longer than win x 1.6 sessions in calendar time —
    PREREG219's `h219.pct_at`, exactly. Reads no bar after ``S``."""
    win = int(settings.fear_long_ivp_window if win is None else win)
    min_n = int(settings.fear_long_ivp_min_values if min_n is None else min_n)
    if s is None or not len(s):
        return None
    S = pd.Timestamp(S)
    i = int(s.index.searchsorted(S, side="right")) - 1
    if i < 0 or (S - s.index[i]).days > 5:
        return None
    lo = max(0, i - win + 1)
    if (s.index[i] - s.index[lo]).days > win * 1.6 / 252 * 365.25:
        return None
    w = s.values[lo: i + 1]
    w = w[np.isfinite(w)]
    if len(w) < min_n:
        return None
    x = float(s.values[i])
    return float((np.sum(w < x) + 0.5 * np.sum(w == x)) / len(w)), x


def merge_rows(sym: str, new: Sequence[Sequence], day: Optional[date] = None, at: Optional[str] = None,
               counts: Optional[dict] = None) -> Tuple[int, int]:
    """Merge IBKR's bars ``new`` into the name's file (a fetched bar replaces a stored one of its date), every stored
    bar the answer changes journaled first in ``revisions/<day>.jsonl`` (``day``: the fetch's session, else today in
    New York; ``at``: the answer's instant): one line a bar — ``revised`` with the values the store held and IBKR's new
    ones, ``absent`` a stored bar inside the answer's span that IBKR no longer serves (kept in the store). ``counts``
    receives ``answered`` / ``revised`` / ``absent``. Returns (bars added, bars held)."""
    old = {str(r[0])[:10]: list(r) for r in load_rows(sym)}
    answer = [bar_row(r) for r in new]
    revised, absent = changes(old, answer)
    if revised or absent:
        _journal(sym, revised, absent, day or DL.today_et(), at)
    added = 0
    for r in answer:
        if r[0] not in old:
            added += 1
        old[r[0]] = r
    rows = [old[k] for k in sorted(old)]
    p = iv_path(sym)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(gzip.compress(json.dumps(rows).encode("utf-8")))
    os.replace(tmp, p)
    if counts is not None:
        counts.update(answered=len(answer), revised=len(revised), absent=len(absent))
    return added, len(rows)


def seed(src_dir: str) -> dict:
    """Copy a research store's files (PREREG219's `llm_trader_research/data/ibkr_iv/`) into the production store — a
    name the store already holds is left alone. Returns the counts."""
    src = Path(src_dir)
    out = {"copied": 0, "kept": 0}
    iv_root().mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*.json.gz")):
        dst = iv_root() / p.name
        if dst.exists():
            out["kept"] += 1
            continue
        shutil.copy2(p, dst)
        out["copied"] += 1
    st = src / "status.jsonl"
    if st.exists() and not (iv_root() / "seed_status.jsonl").exists():
        shutil.copy2(st, iv_root() / "seed_status.jsonl")
    return out


# ── THE DAY'S TABLE ──────────────────────────────────────────────────────────

def table_path(d: date) -> Path:
    return root() / "ivp" / f"{_iso(d)}.json"


def load_table(d: date) -> Optional[dict]:
    p = table_path(d)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def fetch_window(d: date) -> Tuple[datetime, datetime]:
    """The pre-market window of session ``d`` in which the implied volatility is fetched (UTC instants)."""
    a = datetime.combine(d, _hm(settings.fear_long_iv_start_et), tzinfo=ET).astimezone(timezone.utc)
    b = datetime.combine(d, _hm(settings.fear_long_iv_stop_et), tzinfo=ET).astimezone(timezone.utc)
    return a, b


def near_slot(now: datetime) -> bool:
    """From 2 minutes before to 3 minutes after any :00 / :30 (a tick's live path and broker sync)."""
    m = now.astimezone(ET).minute % 30
    return m >= 28 or m <= 2


def _pid_alive(pid: int) -> bool:
    from src.scheduler import db_fence
    return db_fence.pid_alive(int(pid))


def _lock_holder() -> Optional[int]:
    p = root() / LOCK
    try:
        pid = int(p.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if _pid_alive(pid) else None


def prepare(now: Optional[datetime] = None) -> Optional[dict]:
    """At a tick inside session D's pre-market fetch window: launch the DETACHED fetch (`fetch`) unless one runs or the
    day's table is done. Never raises, never waits."""
    now = now or datetime.now(timezone.utc)
    try:
        d = DL.today_et(now)
        if not enabled() or not DL.is_session(d) or not _acct.is_main() or not live_on(d):
            return None
        a, b = fetch_window(d)
        if not (a <= now < b):
            return None
        tab = load_table(d)
        if tab is not None and tab.get("status") == "done":
            return None
        holder = _lock_holder()
        if holder is not None:
            return {"running": holder}
        logs = Path("logs")
        logs.mkdir(exist_ok=True)
        log = logs / f"fear_iv_{_iso(d)}.log"
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        with open(log, "a", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable, "-m", "src.signals.fear_long", "--fetch", "--day", _iso(d)],
                                    stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
                                    close_fds=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"), cwd=os.getcwd())
        logger.info(f"[fear_long] implied-volatility fetch for {d} launched (pid {proc.pid}) -> {log}")
        return {"launched": proc.pid}
    except Exception as e:                                          # noqa: BLE001
        logger.warning(f"[fear_long] fetch launch failed (fail-soft): {type(e).__name__}: {e}")
        return None


def candidate_names(d: date, prev: date, seen: Optional[Dict[str, Optional[dict]]] = None
                    ) -> Tuple[Dict[str, dict], List[str]]:
    """The day's likely signals at the close of ``prev`` — the dip rule's names, LOOSELY (dv20 >= 90% of the floor,
    the close above 98% of its trend, RSI2 under the rule's + 3: a superset of the dip book's exact signals of 09:30) —
    and the names whose stored bars do not reach ``prev`` yet (``stale``: the pre-open run extends them; never
    extended here). ``seen``: the names already judged on current bars ({name: its numbers, or None}), updated."""
    names, _ = DL.candidates(d, min_dv=float(settings.fear_long_min_dollar_volume))
    seen = {} if seen is None else seen
    stale: List[str] = []
    floor = 0.9 * float(settings.fear_long_min_dollar_volume)
    rmax = float(settings.dip_long_rsi_max) + 3.0
    for tk in names:
        if tk in seen:
            continue
        try:
            st = DL.name_stats(tk, prev)
        except Exception as e:                                      # noqa: BLE001
            logger.debug(f"[fear_long] {tk}: stats failed ({e})")
            seen[tk] = None
            continue
        if st is None:
            seen[tk] = None
            continue
        if st["session"] != prev:
            stale.append(tk)
            continue
        ok = (math.isfinite(st["dv20"]) and st["dv20"] >= floor and math.isfinite(st["sma_trend"])
              and st["close"] > 0.98 * st["sma_trend"] and math.isfinite(st["rsi2"]) and st["rsi2"] < rmax)
        seen[tk] = {"rsi2": float(st["rsi2"]), "dv20": float(st["dv20"])} if ok else None
    return {tk: v for tk, v in seen.items() if v is not None}, stale


def _mask(s) -> str:
    return re.sub(r"\b[DU]{1,2}\d{5,}\b", "<acct>", str(s))


_INFO_CODES = (2104, 2106, 2107, 2108, 2158)          # IBKR's farm-connection notices: no answer to a request


def _ask(ib, sym: str, duration: str, errs: list):
    """IBKR's daily OPTION_IMPLIED_VOLATILITY bars of ``sym`` over ``duration`` (None: no contract; [] no data).
    ``errs``: the list IBKR's error stream appends to (`fetch`'s hook) — cleared before the bars are asked, so it holds
    that request's errors; an exception is recorded as code -1."""
    from ib_async import Stock
    from src.broker.ibkr import to_ib_symbol
    c = Stock(to_ib_symbol(sym), "SMART", "USD")
    try:
        q = ib.qualifyContracts(c)
    except Exception as e:                                          # noqa: BLE001
        errs.append((-1, f"{type(e).__name__}: {_mask(e)[:120]}"))
        q = None
    if not q or not getattr(c, "conId", 0):
        return None
    errs.clear()
    try:
        bars = ib.reqHistoricalData(c, endDateTime="", durationStr=duration, barSizeSetting="1 day",
                                    whatToShow="OPTION_IMPLIED_VOLATILITY", useRTH=True, formatDate=1)
    except Exception as e:                                          # noqa: BLE001
        errs.append((-1, f"{type(e).__name__}: {_mask(e)[:120]}"))
        bars = None
    return [[str(b.date)[:10], b.open, b.high, b.low, b.close] for b in (bars or [])]


def fetch(d: date, until: Optional[str] = None, connect=None, clock=None) -> dict:
    """The pre-market fetch of session ``d`` (module docstring): the candidates, each one's bars from IBKR merged into
    the store, its IVP at the close of D-1 written to ``ivp/<d>.json`` as it comes; ends when every candidate is asked
    and no stale name is left, at `fear_long_iv_stop_et` (``until``), on a lost connection or after three timeouts in a
    row. ``connect``: a test's IB factory; ``clock``: a test's UTC clock. Returns the table."""
    def now() -> datetime:
        return clock() if clock is not None else datetime.now(timezone.utc)
    prev = DL.prev_session(d)
    lock = root() / LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = _lock_holder()
    if holder is not None and holder != os.getpid():
        return {"skipped": f"another fetch runs (pid {holder})"}
    lock.write_text(f"{os.getpid()} {datetime.now(ET).isoformat(timespec='seconds')}", encoding="utf-8")
    stop = (datetime.combine(d, _hm(until), tzinfo=ET).astimezone(timezone.utc) if until else fetch_window(d)[1])
    tab = load_table(d) or {}
    tab.update(day=_iso(d), signal_session=_iso(prev), status="running", pid=os.getpid(),
               started=tab.get("started") or now().isoformat(timespec="seconds"))
    tab.setdefault("names", {})
    gap = max(1.0, float(settings.fear_long_iv_gap_seconds))
    ib, last, timeouts, seen = None, 0.0, 0, {}
    errs: list = []                                 # IBKR's error stream for the request being asked (PREREG219's hook)

    def on_error(req, code, text, contract=None, *a):
        errs.append((int(code), _mask(text)[:160]))

    def save():
        tab["updated"] = now().isoformat(timespec="seconds")
        _write_json(table_path(d), tab)
    try:
        while now() < stop:
            stats, stale = candidate_names(d, prev, seen)
            tab["candidates"] = sorted(stats)
            tab["stale"] = sorted(stale)
            todo = sorted((tk for tk in stats if (tab["names"].get(tk) or {}).get("status") not in
                           ("ok", "no_contract", "no_data", "no_value")), key=lambda t: (stats[t]["rsi2"], t))
            save()
            if not todo and not stale:
                tab["status"] = "done"
                break
            if not todo:
                time.sleep(60)                                     # the store is behind: the pre-open run extends it
                continue
            for tk in todo:
                while near_slot(now()) and now() < stop:
                    time.sleep(10)
                if now() >= stop:
                    break
                if ib is None:
                    if connect is not None:
                        ib = connect()
                        if hasattr(ib, "errorEvent"):
                            ib.errorEvent += on_error
                    else:
                        from ib_async import IB
                        ib = IB()
                        ib.errorEvent += on_error
                        ib.connect(settings.ibkr_host, settings.ibkr_port,
                                   clientId=int(settings.ibkr_client_id) + int(settings.fear_long_iv_client_offset),
                                   timeout=20, readonly=True)
                    ib.RequestTimeout = float(settings.fear_long_iv_timeout_seconds)
                if not ib.isConnected():
                    tab["stopped"] = "the connection to the gateway is lost"
                    break
                wait = last + gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                last = time.time()
                held = iv_series(tk)
                ok_hist = held is not None and len(held) >= int(settings.fear_long_ivp_window) and \
                    (pd.Timestamp(prev) - held.index[-1]).days <= 25
                dur = "1 M" if ok_hist else "2 Y"
                errs.clear()
                rows = _ask(ib, tk, dur, errs)
                rec = {"duration": dur, "at": now().isoformat(timespec="seconds"),
                       "rsi2": stats[tk]["rsi2"], "dv20": stats[tk]["dv20"]}
                if rows is None:
                    rec.update(status="no_contract", errors=[e for e in errs if e[0] not in _INFO_CODES][:3])
                elif rows:
                    cnt: dict = {}
                    try:
                        added, n = merge_rows(tk, rows, day=d, at=rec["at"], counts=cnt)
                        s = iv_series(tk)
                        v = ivp_at(s, pd.Timestamp(prev))
                    except Exception as e:                          # noqa: BLE001 — one name's store, not the fetch
                        rec.update(status="error", errors=[(-1, f"store: {type(e).__name__}: {_mask(e)[:120]}")])
                    else:
                        rec.update(status="ok" if v is not None else "no_value", bars=n, added=added,
                                   answered=cnt.get("answered"), revised=cnt.get("revised", 0),
                                   absent=cnt.get("absent", 0),
                                   last_bar=str(s.index[-1].date()) if s is not None else None)
                        if v is not None:
                            rec.update(ivp=v[0], iv=v[1])
                        if cnt.get("revised") or cnt.get("absent"):        # P7: the bars it changed, journaled
                            tab.setdefault("revisions", {})[tk] = {"revised": cnt.get("revised", 0),
                                                                   "absent": cnt.get("absent", 0)}
                    timeouts = 0
                else:
                    # PREREG219's reading of an empty answer: a timeout (or no word from IBKR at all) is an ``error``,
                    # retried while the window lasts; IBKR's own error (162: no data) a ``no_data``, not asked again
                    info = [e for e in errs if e[0] not in _INFO_CODES]
                    timed_out = any(e[0] == -1 and "Timeout" in e[1] for e in info) or not info
                    rec.update(status="error" if timed_out else "no_data", errors=info[:3])
                    timeouts = timeouts + 1 if timed_out else 0
                tab["names"][tk] = rec
                save()
                logger.info(f"[fear_long] {tk}: {rec['status']}" + (f" IVP {rec['ivp']:.2f}" if "ivp" in rec else ""))
                if timeouts >= 3:
                    tab["stopped"] = "three timeouts in a row"
                    break
            if tab.get("stopped"):
                break
        if tab.get("status") != "done":
            tab["status"] = "stopped"
            tab.setdefault("stopped", "the fetch window ended" if now() >= stop else "stopped")
    except Exception as e:                                          # noqa: BLE001
        tab["status"] = "stopped"
        tab["stopped"] = f"{type(e).__name__}: {_mask(e)[:200]}"
        logger.warning(f"[fear_long] fetch failed: {tab['stopped']}")
    finally:
        try:
            if ib is not None and connect is None:
                ib.disconnect()
        except Exception:                                           # noqa: BLE001
            pass
        tab["ended"] = now().isoformat(timespec="seconds")
        save()
        try:
            if _lock_holder() in (None, os.getpid()):
                lock.unlink(missing_ok=True)
        except OSError:
            pass
    return tab


# ── THE RULE ─────────────────────────────────────────────────────────────────

def signals_path(d: date) -> Path:
    return root() / "signals" / f"{_iso(d)}.json"


def limit_price(close: float, atr: float) -> Optional[float]:
    return DL.limit_price(close, atr, float(settings.fear_long_limit_atr))


def dip_signals(day: dict) -> List[dict]:
    """Every $1B+ signal of the dip book's day — the kept (``signals``), the not-top (``not_top``) and the meta-model's
    filtered ones (``filtered``) — once each."""
    out, seen = [], set()
    for k in ("signals", "not_top", "filtered"):
        for r in day.get(k) or []:
            tk = r.get("ticker")
            if tk and tk not in seen:
                seen.add(tk)
                out.append(r)
    return out


def dip_pick(d: date) -> set:
    """The names of the dip book's own orders of session ``d`` (its daily top pick; journal-only orders excluded)."""
    return {str(o.get("ticker")) for o in DL.load_orders(d) if o.get("status") not in DIP_EXCLUDED_STATUS}


def dip_ready(d: date) -> bool:
    """The dip book's day is decided for the exclusion of its own order: its signals computed and — when it kept a
    pick — its orders made (or the dip book off)."""
    if not DL.enabled():
        return True
    p = DL.signals_path(d)
    if not p.exists():
        return False
    try:
        day = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return not day.get("signals") or DL.orders_path(d).exists()


def ready(d: date, now: Optional[datetime] = None) -> bool:
    """The day's table is final: done or stopped, or its fetch window is over."""
    now = now or datetime.now(timezone.utc)
    tab = load_table(d)
    return bool(tab and tab.get("status") in ("done", "stopped")) or now >= fetch_window(d)[1]


def compute_signals(d: date, now: Optional[datetime] = None, write: bool = True) -> dict:
    """Session ``d``'s fear signals (module docstring), written to ``signals/<d>.json`` (``write``): each of the dip
    book's signals with its fate — ``fear`` (an order), ``calm`` (IVP under the cut), ``no_iv`` (no value in the day's
    table; its table status in ``iv_status``), ``dip_pick`` (the dip book's own order), ``below_floor``."""
    _acct.decide_here(f"the fear signals of {d}")     # an extra account reads them, never computes them
    t0 = datetime.now(timezone.utc)
    prev = DL.prev_session(d)
    day = DL.signals_for(d)
    tab = load_table(d) or {}
    names = tab.get("names") or {}
    pick = dip_pick(d)
    cut = float(settings.fear_long_ivp_min)
    floor = float(settings.fear_long_min_dollar_volume)
    sig, fates = [], {}
    for r in dip_signals(day):
        tk = str(r["ticker"])
        rec = names.get(tk) or {}
        ivp = rec.get("ivp")
        if float(r.get("dv20") or 0.0) < floor:
            fates[tk] = "below_floor"
        elif tk in pick:
            fates[tk] = "dip_pick"
        elif ivp is None:
            fates[tk] = "no_iv"
        elif float(ivp) < cut:
            fates[tk] = "calm"
        else:
            fates[tk] = "fear"
            sig.append({k: r.get(k) for k in ("ticker", "close", "atr14", "rsi2", "dv20", "sma_trend")} |
                       {"ivp": float(ivp), "iv": rec.get("iv"), "limit": limit_price(r.get("close"), r.get("atr14"))})
    sig.sort(key=lambda s: (-float(s["ivp"]), float(s["rsi2"]), s["ticker"]))
    out = {"day": _iso(d), "signal_session": _iso(prev), "signals": sig, "fates": fates,
           "ivp_by_name": {tk: (names.get(tk) or {}).get("ivp") for tk in fates},
           "iv_status": {tk: (names.get(tk) or {}).get("status") or "not_asked"
                         for tk, f in fates.items() if f == "no_iv"},
           "table": {k: tab.get(k) for k in ("status", "stopped", "started", "ended")} if tab else None,
           "rule": {"ivp_min": cut, "min_dollar_volume": floor, "limit_atr": float(settings.fear_long_limit_atr),
                    "limit_sessions": int(settings.fear_long_limit_sessions)},
           "computed_at": t0.isoformat(timespec="seconds")}
    if not write:
        return out
    _write_json(signals_path(d), out)
    if sig or fates:
        logger.info(f"[fear_long] {d}: {len(sig)} fear signal(s) of {len(fates)} dip signal(s)"
                    + (": " + ", ".join(f"{s['ticker']} IVP {s['ivp']:.2f}" for s in sig) if sig else "")
                    + (f" — no implied volatility for {sorted(t for t, f in fates.items() if f == 'no_iv')}"
                       if any(f == "no_iv" for f in fates.values()) else ""))
    return out


def signals_for(d: date, now: Optional[datetime] = None) -> Optional[dict]:
    """The day's fear signals — computed once (when the day's table is final), then read from ``signals/<d>.json``;
    None while the table may still grow."""
    p = signals_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                                      # noqa: BLE001
            logger.warning(f"[fear_long] {p} unreadable ({e}) — recomputing")
    if not ready(d, now):
        return None
    return compute_signals(d, now)


# ── the orders (the dip book's schema and life) ──────────────────────────────

def window(d: date) -> List[date]:
    """The sessions an order of session ``d`` lives in: ``d`` and the next `fear_long_limit_sessions` - 1."""
    from src.signals.sel_short import session_after
    n = max(1, int(settings.fear_long_limit_sessions))
    return [d] + [session_after(d, k) for k in range(1, n)]


def orders_path(d: date) -> Path:
    return _acct.state(root() / "orders" / f"{_iso(d)}.json")         # this account's (`src.accounts.paths`)


def load_orders(d: date) -> List[dict]:
    p = orders_path(d)
    if not p.exists():
        return []
    try:
        return list(json.loads(p.read_text(encoding="utf-8")).get("orders") or [])
    except Exception as e:                                          # noqa: BLE001
        logger.warning(f"[fear_long] {p} unreadable ({e})")
        return []


def save_orders(d: date, orders: Sequence[dict]) -> None:
    _write_json(orders_path(d), {"day": _iso(d), "orders": list(orders)})


def order_days(today: date) -> List[date]:
    """The order days whose windows can include ``today``."""
    from src.signals.sel_short import sessions_before
    n = max(1, int(settings.fear_long_limit_sessions))
    back = sessions_before(today, n - 1) if n > 1 else []
    return back + ([today] if DL.is_session(today) else [])


def active_orders(today: date, now: Optional[datetime] = None) -> Dict[date, List[dict]]:
    """``{order day: its orders}`` for the days whose windows can include ``today``, each pending or working order
    rescaled through the splits executed since its day (`dip_long.apply_order_splits`)."""
    out: Dict[date, List[dict]] = {}
    for d in order_days(today):
        o = load_orders(d)
        if o:
            DL.apply_order_splits(o, now)
            out[d] = o
    return out


def make_order(d: date, sig: dict, rank: int, signal_session: Optional[str]) -> Optional[dict]:
    """A pending order for signal ``sig`` of session ``d`` (None when its limit cannot be priced)."""
    lim = sig.get("limit")
    if lim is None and sig.get("atr14") is not None and math.isfinite(float(sig["atr14"])):
        lim = limit_price(sig.get("close"), sig["atr14"])
    if lim is None:
        return None
    return {"oid": f"{sig['ticker']}|{_iso(d)}", "ticker": str(sig["ticker"]), "day": _iso(d),
            "signal_session": signal_session, "rank": int(rank), "ivp": sig.get("ivp"), "iv": sig.get("iv"),
            "rsi2": sig.get("rsi2"), "close": sig.get("close"), "atr14": sig.get("atr14"), "limit": float(lim),
            "dv20": sig.get("dv20"), "sma_trend": sig.get("sma_trend"),
            "window": [_iso(x) for x in window(d)], "status": "pending", "qty": None, "placed_at": None,
            "placed_session": None, "ibkr_qty": 0, "ibkr": None, "ibkr_n": 0, "checked_through": None,
            "unfunded_session": None, "trade": None, "events": []}


def order_key(o: dict) -> tuple:
    """The orders' place when they compete (the backtest's ``rank`` = -IVP + RSI2 x 1e-5: IVP highest first, then the
    deepest dip)."""
    v = o.get("ivp")
    r = o.get("rsi2")
    return (-float(v) if v is not None else 1e9,
            float(r) if r is not None and math.isfinite(float(r)) else 1e9, str(o.get("day")), str(o.get("ticker")))


order_event = DL.order_event


# ── exits ────────────────────────────────────────────────────────────────────

def entry_day(trade: dict) -> Optional[date]:
    raw = trade.get("fear_entry_day") or str(trade.get("entry_date") or "")[:10]
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
    sma = c.rolling(int(settings.fear_long_exit_sma_sessions)).mean()
    hit = d.index[(d.index >= pd.Timestamp(entry)) & (c > sma).to_numpy()]
    if len(hit) == 0:
        return None
    s = hit[0]
    return {"session": pd.Timestamp(s).date(), "close": float(c.loc[s]), "sma_exit": float(sma.loc[s])}


def exit_check(trade: dict, d: date) -> Tuple[Optional[str], dict]:
    """Does session ``d``'s open close this long? ``fear_rebound`` once a close from the fill session through D-1 is
    above its exit average, ``fear_time`` once the long has held `fear_long_max_hold_sessions` sessions through D-1;
    else None (``info``: the numbers; ``stale`` when the store had no close for D-1, extended once first)."""
    e = entry_day(trade)
    prev = DL.prev_session(d)
    info: dict = {"through": _iso(prev)}
    if e is None or prev < e:
        return None, info
    held = DL.sessions_held(e, prev)
    info["held"] = held
    tk = str(trade["ticker"])
    daily = DL._daily(tk, prev)
    last = None if daily is None or daily.empty else pd.Timestamp(daily.index[daily.index <= pd.Timestamp(prev)].max())
    if last is None or pd.isna(last) or last.date() != prev:
        DL._extend([tk], prev)
        daily = DL._daily(tk, prev)
        last = None if daily is None or daily.empty else pd.Timestamp(daily.index[daily.index <= pd.Timestamp(prev)].max())
    if last is None or pd.isna(last) or last.date() != prev:
        info["stale"] = True
        logger.warning(f"[fear_long] {tk}: no stored close for {prev} — the rebound rule is judged on the closes the "
                       f"store has")
    hit = rebound_session(daily, e, prev)
    if hit is not None:
        info.update(rebound_session=_iso(hit["session"]), close=hit["close"], sma_exit=hit["sma_exit"])
        return "fear_rebound", info
    if held >= int(settings.fear_long_max_hold_sessions):
        return "fear_time", info
    return None, info


# ── journals ─────────────────────────────────────────────────────────────────

def entries_path(d: date) -> Path:
    return _acct.state(root() / "entries" / f"{_iso(d)}.jsonl")


def exits_path(d: date) -> Path:
    return _acct.state(root() / "exits" / f"{_iso(d)}.jsonl")


def journal_entry(d: date, sig: dict, outcome: str, **kw) -> None:
    DL._append(entries_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "ticker": sig.get("ticker"), "outcome": outcome,
                                 **{k: sig.get(k) for k in ("ivp", "rsi2", "close", "sma_trend", "dv20")}, **kw})


def journal_exit(d: date, trade: dict, reason: str, info: dict) -> None:
    DL._append(exits_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "ticker": trade.get("ticker"), "reason": reason,
                               "trade": trade.get("recommendation_id"), **info})


def trade_id(d: date, ticker: str) -> str:
    return hashlib.sha1(f"fear|{_iso(d)}|{ticker}".encode("utf-8")).hexdigest()[:16]


# ── health (the email's scorer banner) ───────────────────────────────────────

def health(now: Optional[datetime] = None, trades: Optional[Sequence[dict]] = None) -> dict:
    """``problems``: on a session after the entry window, no fear signals decided for today (the dip book's signals or
    the day's table missing); a day's table that never ran or holds errors for a signal; an open long held past its
    limit. ``notes``: the day's fear orders, the names IBKR answered without a value."""
    now = now or datetime.now(timezone.utc)
    out: dict = {"problems": [], "notes": []}
    if not enabled():
        return out
    d = DL.today_et(now)
    et = now.astimezone(ET)
    if DL.is_session(d) and live_on(d) and DL.in_rth(now) and \
            et.hour * 60 + et.minute > 570 + float(settings.fear_long_entry_window_minutes):
        p = signals_path(d)
        day = None
        if p.exists():
            try:
                day = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                day = None
        if day is None:
            out["problems"].append(f"fear_long: no fear signals decided for {d} (the dip book's signals or the "
                                   f"implied-volatility table missing)")
        else:
            tab = day.get("table")
            if tab is None:
                out["problems"].append(f"fear_long: no implied-volatility table for {d} — the pre-market fetch did "
                                       f"not run, no fear order")
            no_iv = sorted(t for t, f in (day.get("fates") or {}).items() if f == "no_iv")
            ivs = day.get("iv_status") or {}
            none_at_ibkr = [t for t in no_iv if ivs.get(t) in NO_IV_AT_IBKR]
            missed = [t for t in no_iv if t not in none_at_ibkr]
            if tab is not None and missed:
                why = ", ".join(t + " " + str(ivs.get(t) or "not_asked") for t in missed[:6])
                out["problems"].append(f"fear_long: no implied volatility for {len(missed)} signal(s) of {d} "
                                       f"({why}) — not ordered"
                                       + (f"; the fetch stopped: {tab.get('stopped')}" if tab.get("stopped") else ""))
            if none_at_ibkr:
                why = ", ".join(t + " " + str(ivs[t]) for t in none_at_ibkr[:6])
                out["notes"].append(f"fear_long: IBKR has no implied volatility for {len(none_at_ibkr)} signal(s) of "
                                    f"{d} ({why}) — not ordered")
            if day.get("signals"):
                out["notes"].append(f"fear_long: {len(day['signals'])} fear order(s) for {d}: "
                                    + ", ".join(f"{s['ticker']} IVP {float(s['ivp']):.2f}" for s in day["signals"][:6]))
    for t in trades or []:
        if t.get("status") != "OPEN" or t.get("entry_mechanism") != MECHANISM:
            continue
        e = entry_day(t)
        if e is not None and DL.sessions_held(e, d) > int(settings.fear_long_max_hold_sessions) + 2:
            out["problems"].append(f"fear_long: {t.get('ticker')} held past its {settings.fear_long_max_hold_sessions}-"
                                   f"session limit (since {e})")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="The fear book (src/signals/fear_long.py)")
    ap.add_argument("--fetch", action="store_true", help="the pre-market implied-volatility fetch of --day")
    ap.add_argument("--until", help="HH:MM ET: stop the fetch then (default fear_long_iv_stop_et)")
    ap.add_argument("--signals", action="store_true", help="the day's fear signals, previewed (not written)")
    ap.add_argument("--write", action="store_true",
                    help="with --signals: write signals/<D>.json — the opening tick's file, decided once: never before "
                         "the day's table is final")
    ap.add_argument("--ivp", help="a name's IVP at the close before --day, from the store")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--seed", help="copy a research store's files into the production store")
    ap.add_argument("--day", help="YYYY-MM-DD (default today)")
    a = ap.parse_args(argv)
    d = date.fromisoformat(a.day) if a.day else DL.today_et()
    if a.seed:
        print(json.dumps(seed(a.seed)))
    if a.fetch:
        tab = fetch(d, until=a.until)
        print(json.dumps({k: v for k, v in tab.items() if k != "names"} | {"names": len(tab.get("names") or {})},
                         default=str))
    if a.ivp:
        print(json.dumps({"name": a.ivp, "session": _iso(DL.prev_session(d)),
                          "ivp": ivp_at(iv_series(a.ivp), pd.Timestamp(DL.prev_session(d)))}, default=str))
    if a.signals:
        print(json.dumps(compute_signals(d, write=a.write), indent=1, default=str)[:4000])
    if a.status:
        print(json.dumps(health(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
