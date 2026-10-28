"""PREREG99 — trading halts at the short arms' entry passes: production's halt guard in the simulator (V0, a
correction) and the registered variants (H1a / H1b / H2 / H3). Production's guard is `Mimic()`'s DEFAULT from
2026-10-19 (AUDIT151's H_GUARD, -0.68 pp a year); ``Mimic(halts=None)`` (`Mimic.off()`, `mimic.PRE151`) builds every
book as before (`mimic.live_books` then never imports this module); ``Mimic(halts="guard" | "wait" | "wait_half" |
"backup" | "both")`` applies one of them to the vol, fall and ETF arms' entries.

PRODUCTION (`tracker.record_sel_short_trades` -> ``attempt``, ``enable_sel_short_halt_guard``): a pick whose name is on
NYSE's current-halt list at an entry pass is not entered then (``halted_retry``, not consumed) and is re-checked at
every later pass while it is at most ``sel_short_entry_max_age_minutes`` (75) old — a turn entry (fall, ETF arms)
while its turn bar is; halted at every pass, it expires. A halt never triggers the vol arm's fallback (only no_borrow /
unlendable / ibkr_refused do); the fallback's backups and the pressure extras go through the same guard.

THE PASSES of a pick (or a turn entry) whose bar ends at T (`Schedule`): two at each of the scheduler's slots at or
after T — the live path at slot + 90 s, the end-of-tick block at slot + 11 min — on market days (the research
calendar), over the regular session's 30-minute slots and the extended windows (`runner._session_slots`; the overnight
venue's slots are not modelled), while at most 75 minutes after T.

THE HALTS (`HaltBook`: NYSE's history of every US listing, ``cache/ml/halts/nyse_halts_<year>.csv``, times in ET —
checked on GME's 2021-03-10 pauses against Polygon's 1-minute bars): intervals [halt, resume); one halt listed twice
(its notice without a resume, then again with it) is one halt, the resume kept. A halt WITHOUT a resume in the file is
over once a Polygon 1-minute bar with volume that started at/after it has ENDED, and at the latest at the next regular
session's open — PREREG99's rule ("resumes at the first Polygon 1-minute bar with volume after it, else at the next
regular session's open") read without look-ahead: a pass sees only the bars that ended by then.

NO LOOK-AHEAD (`HaltModel.halt_at`, `HaltModel.price`): a pass sees the halts that STARTED at/before it — whether each
is still in force (no resume by then) and nothing more; a resume time decides nothing but "not resumed yet" — and the
1-minute bars that ENDED at/before it (`tests/test_backtest_halts.py` rewrites everything later and checks nothing
moves).

PRICES at a later pass (`HaltModel.price`): the close of the last Polygon 1-minute bar ENDING at/before the pass
(extended hours included; split-adjusted bars) in the candidate's traded dollars — scaled by its bar's close over the
last 1-minute bar ending at/before its bar's end. Fetched only for the candidates a halt touches and cached per
(symbol, ET day) under `configure` 's ``minute_dir`` (default ``<research>/studies/prereg99/minute``), at most
``rate`` (5) requests a second.

V0 ("guard", production): a candidate enters at its FIRST pass where it is not halted — at the first pass the
simulator's own convention (its bar's close, stamped at the bar's end: nothing changes for a candidate not halted
then); at a later pass at the 1-minute price then, after production's checks at that moment: at/under the arm's
target -> ``target_reached``; lendability judged when it enters, as the turn entries (`mimic.turn_lend`: IBKR's file in
force then where an archive holds it, the pick's draw in between; the ETF arm its own ``lend_after`` on the entry's
day). Halted at every pass -> no trade (``halted_expired``). The vol arm runs production's pass loop (`VolMachine`): at
each pass its pending picks, the oldest bar first — the pick, its same-bar fallback when IBKR cannot lend it (the
backups in rank order; a halted backup is skipped and stays eligible at later passes), then the pressure extras (V6,
every HIGH backup not yet settled) while the pick is pending. Lendability is judged once per candidate, when it is
first priced (the simulator models no in-window borrow retries); the trade's deadline stays counted from the pick and
a delayed entry's exits are judged from the first bar ending after it.

The variants change only the picks halted at their first pass:

* ``"wait"`` (H1a): a pick halted at EVERY pass of its window stays pending while halted, to 16:00 ET of the next
  regular session; at the first later pass where it is not halted (and has a price) it enters if its price is above
  the arm's target (the fall arm has none) and it is lendable then; else no trade. Only the pick waits: its backups
  and extras keep V0's window.
* ``"wait_half"`` (H1b): H1a, entered only at a price of at least the pick close - 1/2 x (pick close - target).
* ``"backup"`` (H2, the vol arm; the fall and ETF arms have no backups): a pick halted at its first pass is replaced
  there by the bar's first backup in rank order that is not halted then, lendable and not traded today — the fallback
  with "halted" as one more trigger; the pick is settled ``halted_fallback``. No backup qualifies -> V0.
* ``"both"`` (H3): H2, and the pick itself also waits as in H1a (two trades when both happen); the halted fallback
  counts as the pick's fallback (none again if it later proves unlendable).
"""
from __future__ import annotations

import glob
import heapq
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings
from src.backtest.mimic import HALT_MODES as MODES

ET = "America/New_York"
EPOCH = pd.Timestamp("1970-01-01")
S = 10**9
MINUTE = 60 * S
DAY_NS = 86400 * S
WAIT_MODES = ("wait", "wait_half", "both")
BACKUP_MODES = ("backup", "both")
PASS_OFFSETS = (90, 11 * 60)            # seconds after a slot: the live path, the end-of-tick block
OPEN_MIN, CLOSE_MIN = 9 * 60 + 30, 16 * 60
HALT_DIR = Path(__file__).resolve().parents[2] / "cache" / "ml" / "halts"
# ``halt_paths``: the halt files the model reads (None = every year's file under `HALT_DIR`; PREREG101's set A0 adds its
# own years from `cache/ml/halts_a0/`, `src.backtest.a0.harness`)
_CONF: Dict[str, object] = {"minute_dir": None, "fetch": True, "rate": 5.0, "halt_paths": None}


def configure(minute_dir=None, fetch: Optional[bool] = None, rate: Optional[float] = None) -> None:
    """Where the 1-minute bars are cached (and fetched to), whether a missing day is fetched, the request rate."""
    if minute_dir is not None:
        _CONF["minute_dir"] = str(minute_dir)
    if fetch is not None:
        _CONF["fetch"] = bool(fetch)
    if rate is not None:
        _CONF["rate"] = float(rate)


def minute_dir() -> Path:
    d = _CONF.get("minute_dir")
    return Path(str(d)) if d else Path(settings.backtest_research_dir) / "studies" / "prereg99" / "minute"


def set_halt_paths(paths: Optional[Sequence] = None) -> None:
    """The halt files the default model reads (None: every year's file under `HALT_DIR`, the default)."""
    _CONF["halt_paths"] = [str(p) for p in paths] if paths is not None else None


def halt_paths() -> Optional[List[str]]:
    """The halt files set by `set_halt_paths` (None: the default, every file under `HALT_DIR`)."""
    v = _CONF.get("halt_paths")
    return list(v) if v is not None else None


# ── time helpers ─────────────────────────────────────────────────────────────

def day_of(ns) -> int:
    """The ET calendar day number (days since 1970-01-01) of a UTC nanosecond instant."""
    t = pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET)
    return int((pd.Timestamp(t.date()) - EPOCH).days)


def iso_day(day: int) -> str:
    return (EPOCH + pd.Timedelta(days=int(day))).date().isoformat()


def iso_et(ns) -> Optional[str]:
    return None if ns is None else pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET).isoformat()


def plain(tkn) -> str:
    """The traded symbol of a backtest name (``XYZ@D`` / ``@E`` / ``@X`` -> ``XYZ``; `mimic.plain`)."""
    return str(tkn).split("@")[0]


@lru_cache(maxsize=None)
def halt_symbol(sym: str) -> str:
    """The halt file's spelling (class shares with a space: ``BRK-B`` -> ``BRK B``; production's `trade_halts._symbol`)."""
    return str(plain(sym)).strip().upper().replace("-", " ").replace(".", " ")


@lru_cache(maxsize=1)
def production_slots() -> Tuple[int, ...]:
    """The scheduler's tick slots in the regular session and the extended windows, minutes after midnight ET
    (`runner._session_slots`; the overnight venue's slots are not modelled)."""
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    return tuple(sorted({t.hour * 60 + t.minute for t, kind in slots if kind in ("rth", "extended")}))


# ── the halt file ────────────────────────────────────────────────────────────

def _et_ns(d: pd.Series, t: pd.Series) -> np.ndarray:
    """UTC ns of ET date + time strings; -1 where either is blank or unparseable."""
    d, t = d.astype(str).str.strip(), t.astype(str).str.strip()
    s = (d + " " + t).where((d != "") & (t != ""))
    ts = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    ts = ts.dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC").dt.tz_localize(None)
    out = ts.to_numpy("datetime64[ns]").astype(np.int64)
    out[ts.isna().to_numpy()] = -1
    return out


def read_halts(paths: Optional[Sequence] = None) -> pd.DataFrame:
    """NYSE's halt history as one table — ``sym`` (the file's spelling), ``start`` / ``resume`` (UTC ns; resume -1 =
    none in the file), ``reason`` (lower case), ``exchange`` — one row per halt: a halt listed twice (its notice
    without a resume, then with it; exact duplicates) is merged, its resume kept."""
    files = [str(p) for p in paths] if paths is not None else sorted(glob.glob(str(HALT_DIR / "nyse_halts_*.csv")))
    if not files:
        raise FileNotFoundError(f"no halt file under {HALT_DIR}")
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    t = pd.DataFrame({"sym": df["Symbol"].astype(str).str.strip().str.upper(),
                      "start": _et_ns(df["Halt Date"], df["Halt Time"]),
                      "resume": _et_ns(df["Resume Date"], df["NYSE Resume Time"]),
                      "reason": df["Reason"].astype(str).str.strip().str.lower(),
                      "exchange": df["Exchange"].astype(str).str.strip()})
    t = t[(t["sym"] != "") & (t["start"] >= 0)]
    t = t.sort_values(["sym", "start", "resume"], kind="mergesort")
    t = t.groupby(["sym", "start"], as_index=False, sort=True).agg(
        resume=("resume", "max"), reason=("reason", "first"), exchange=("exchange", "first"))
    return t.reset_index(drop=True)


class HaltBook:
    """Each symbol's halts as arrays sorted by start (UTC ns; resume -1 = none in the file), with the running maximum
    of the file's resumes (a halt still in force at t: one started by t whose resume is after t)."""

    def __init__(self, table: pd.DataFrame):
        self.table = table
        self._by: Dict[str, tuple] = {}
        low = np.iinfo(np.int64).min
        for sym, g in table.groupby("sym", sort=False):
            g = g.sort_values("start", kind="mergesort")
            st = g["start"].to_numpy(np.int64)
            rs = g["resume"].to_numpy(np.int64)
            self._by[str(sym)] = (st, rs, np.maximum.accumulate(np.where(rs >= 0, rs, low)),
                                  g["reason"].to_numpy(object))

    def get(self, sym: str):
        return self._by.get(sym)

    def __len__(self) -> int:
        return len(self.table)


_BOOKS: Dict[tuple, HaltBook] = {}


def halt_book(paths: Optional[Sequence] = None) -> HaltBook:
    """The halt book of ``paths`` (default: every year's file), read once per process."""
    key = tuple(str(p) for p in paths) if paths is not None else ("*",)
    if key not in _BOOKS:
        _BOOKS[key] = HaltBook(read_halts(paths))
    return _BOOKS[key]


# ── Polygon's 1-minute bars ─────────────────────────────────────────────────

def _safe(sym: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(sym))


class MinuteBars:
    """Polygon's 1-minute bars of one symbol on one ET day (04:00-20:00, split-adjusted): ``{"s": bar starts (UTC
    ns), "c": closes, "v": volumes}`` (empty arrays: no trade that day), or None when unknown (not cached, fetching
    off). Cached on disk per (symbol, day) — an empty day too — written atomically; fetched at most ``rate`` times a
    second. ``source(sym, day)``: an injected provider instead of the cache and Polygon (tests)."""

    def __init__(self, cache_dir=None, fetch: bool = True, rate: float = 5.0,
                 source: Optional[Callable[[str, int], Optional[dict]]] = None):
        self.dir = Path(cache_dir) if cache_dir is not None else None
        self.fetch, self.rate, self.source = bool(fetch), float(rate), source
        self._mem: Dict[Tuple[str, int], Optional[dict]] = {}
        self._last = 0.0
        self.fetched = 0

    def path(self, sym: str, day: int) -> Path:
        return self.dir / _safe(plain(sym)) / f"{iso_day(day)}.npz"

    def day(self, sym: str, day: int) -> Optional[dict]:
        k = (plain(sym), int(day))
        if k in self._mem:
            return self._mem[k]
        out = None
        if self.source is not None:
            out = self.source(k[0], k[1])
        elif self.dir is not None:
            p = self.path(*k)
            if p.exists():
                with np.load(p) as z:
                    out = {"s": z["s"].astype(np.int64), "c": z["c"].astype(float), "v": z["v"].astype(float)}
            elif self.fetch:
                out = self._fetch(*k)
                self._save(p, out)
        self._mem[k] = out
        return out

    def _pace(self) -> None:
        wait = self._last + 1.0 / max(self.rate, 1e-9) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _fetch(self, sym: str, day: int) -> dict:
        from src.data import polygon_client as pc
        iso = iso_day(day)
        path = f"/v2/aggs/ticker/{pc.to_polygon_symbol(sym)}/range/1/minute/{iso}/{iso}"
        js = None
        for k in range(3):
            self._pace()
            js = pc._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
            if js is not None:
                break
            time.sleep(2.0 * (k + 1))
        if js is None:
            raise RuntimeError(f"Polygon's 1-minute bars for {sym} on {iso}: no answer after 3 tries")
        res = list(js.get("results") or [])
        nxt, pages = js.get("next_url"), 0
        while nxt and pages < 20:
            self._pace()
            try:
                sep = "&" if "?" in nxt else "?"
                r = pc._http_get(f"{nxt}{sep}apiKey={settings.polygon_api_key}")
                r.raise_for_status()
                j2 = r.json()
            except Exception:                                  # noqa: BLE001 — the URL carries the key: never shown
                raise RuntimeError(f"Polygon's 1-minute bars for {sym} on {iso}: a further page failed") from None
            res.extend(j2.get("results") or [])
            nxt, pages = j2.get("next_url"), pages + 1
        self.fetched += 1
        s = np.array([int(b["t"]) for b in res], np.int64) * 1_000_000
        o = np.argsort(s, kind="mergesort")
        return {"s": s[o], "c": np.array([float(b["c"]) for b in res], float)[o],
                "v": np.array([float(b.get("v") or 0.0) for b in res], float)[o]}

    @staticmethod
    def _save(p: Path, out: dict) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, s=out["s"], c=out["c"], v=out["v"])
        try:
            os.replace(tmp, p)
        except PermissionError:                            # another process wrote (and holds) the same day: keep it
            if not p.exists():
                raise
            tmp.unlink(missing_ok=True)


def _last_close(b: Optional[dict], t: int) -> Optional[float]:
    """The close of the last bar of ``b`` ENDING at/before ``t``."""
    if not b or not len(b["s"]):
        return None
    k = int(np.searchsorted(np.asarray(b["s"], np.int64) + MINUTE, int(t), side="right")) - 1
    return float(b["c"][k]) if k >= 0 else None


# ── production's entry passes ────────────────────────────────────────────────

class Schedule:
    """Production's entry passes over the market days ``sessions`` (day numbers) and the slots ``slots`` (minutes
    after midnight ET): two per slot (``offsets`` seconds after it) from a bar's end, within ``window_min``."""

    def __init__(self, sessions, slots: Sequence[int], window_min: float = 75.0,
                 offsets: Sequence[int] = PASS_OFFSETS):
        self.cal = np.unique(np.asarray(sessions, np.int64))
        self._set = {int(d) for d in self.cal}
        self.slots = tuple(sorted({int(m) for m in slots}))
        self.window = int(round(float(window_min) * 60)) * S
        self.offsets = tuple(int(o) * S for o in offsets)
        self._slot_ns: Dict[int, np.ndarray] = {}
        self._at: Dict[Tuple[int, int], int] = {}

    def is_session(self, day: int) -> bool:
        return int(day) in self._set

    def next_session(self, day: int) -> int:
        i = int(np.searchsorted(self.cal, int(day), side="right"))
        if i >= len(self.cal):
            raise ValueError(f"no session after {iso_day(day)} in the calendar")
        return int(self.cal[i])

    def sessions_between(self, d0: int, d1: int) -> np.ndarray:
        return self.cal[(self.cal >= int(d0)) & (self.cal <= int(d1))]

    def at(self, day: int, minutes: int) -> int:
        k = (int(day), int(minutes))
        v = self._at.get(k)
        if v is None:
            v = int(pd.Timestamp(EPOCH + pd.Timedelta(days=k[0], minutes=k[1])).tz_localize(ET).value)
            self._at[k] = v
        return v

    def slot_ns(self, day: int) -> np.ndarray:
        v = self._slot_ns.get(int(day))
        if v is None:
            v = np.array([self.at(day, m) for m in self.slots], np.int64)
            self._slot_ns[int(day)] = v
        return v

    def passes(self, T: int, until: Optional[int] = None) -> np.ndarray:
        """The passes of a candidate whose bar ends at ``T``: production's two passes at each slot at/after ``T`` on
        market days, while at most the window after ``T`` — or, with ``until``, every one up to that instant."""
        T = int(T)
        d = day_of(T)
        days = [d]
        if until is not None:
            dd = d
            while True:
                dd = self.next_session(dd)
                if self.at(dd, 0) > int(until):
                    break
                days.append(dd)
        out = []
        for dd in days:
            if not self.is_session(dd):
                continue
            for s in self.slot_ns(dd):
                if s < T:
                    continue
                for off in self.offsets:
                    P = int(s) + off
                    if (P - T > self.window) if until is None else (P > int(until)):
                        continue
                    out.append(P)
        return np.array(sorted(out), np.int64)

    def wait_end(self, T: int) -> int:
        """H1's horizon: the end (16:00 ET) of the regular session after the one ``T`` belongs to."""
        return self.at(self.next_session(day_of(T)), CLOSE_MIN)

    def next_open(self, t: int) -> int:
        """The next regular session's open (09:30 ET) after instant ``t``."""
        d = day_of(t)
        if self.is_session(d) and int(t) < self.at(d, OPEN_MIN):
            return self.at(d, OPEN_MIN)
        return self.at(self.next_session(d), OPEN_MIN)


# ── the halt model ───────────────────────────────────────────────────────────

class HaltModel:
    """What a pass knows: whether a name is halted at it, its 1-minute price then."""

    def __init__(self, book: HaltBook, minutes: MinuteBars, schedule: Schedule):
        self.book, self.minutes, self.schedule = book, minutes, schedule
        self.n_halt_checks = 0

    def halt_at(self, sym: str, t: int) -> Optional[dict]:
        """The halt in force on ``sym`` at instant ``t``, or None: a halt that STARTED at/before ``t`` with its resume
        in the file after ``t`` — or, without a resume in the file, no 1-minute bar with volume that started at/after
        it has ENDED by ``t`` and ``t`` is before the next regular session's open. The latest-starting one is
        returned (start, resume or None, reason, missing). Reads nothing after ``t``."""
        self.n_halt_checks += 1
        a = self.book.get(halt_symbol(sym))
        if a is None:
            return None
        st, rs, rmax, why = a
        t = int(t)
        i = int(np.searchsorted(st, t, side="right"))         # the halts started at/before t: [0, i)
        if i == 0:
            return None
        best = -1
        if rmax[i - 1] > t:                                   # one with a resume in the file still in force
            j = i - 1
            while not rs[j] > t:
                j -= 1
            best = j
        j, lo = i - 1, t - 6 * DAY_NS                         # without a resume: over by the next open at the latest
        while j > best and st[j] >= lo:
            if rs[j] < 0 and t < self.schedule.next_open(int(st[j])) and not self.traded_between(sym, int(st[j]), t):
                best = j
                break
            j -= 1
        if best < 0:
            return None
        return {"start": int(st[best]), "resume": int(rs[best]) if rs[best] >= 0 else None,
                "reason": str(why[best]), "missing": bool(rs[best] < 0)}

    def halted_at(self, sym: str, t: int) -> bool:
        """``sym`` is halted at instant ``t`` (`halt_at`)."""
        return self.halt_at(sym, t) is not None

    def traded_between(self, sym: str, s: int, t: int) -> bool:
        """A 1-minute bar with volume started at/after ``s`` and ENDED at/before ``t``."""
        for d in self.schedule.sessions_between(day_of(s), day_of(t)):
            b = self.minutes.day(sym, int(d))
            if not b or not len(b["s"]):
                continue
            bs = np.asarray(b["s"], np.int64)
            if ((bs >= int(s)) & (bs + MINUTE <= int(t)) & (np.asarray(b["v"], float) > 0)).any():
                return True
        return False

    def price(self, sym: str, t: int, ref_t: int, ref_close: float) -> Optional[Tuple[float, float]]:
        """(price, scale): the close of the last 1-minute bar ENDING at/before ``t`` (back to ``ref_t`` 's day) in the
        terms of ``ref_close`` — the candidate's own bar close at ``ref_t``, traded dollars — via the last 1-minute bar
        ending at/before ``ref_t`` (its day); None without either."""
        if not (ref_close is not None and np.isfinite(ref_close) and ref_close > 0):
            return None
        d_ref = day_of(ref_t)
        c_ref = _last_close(self.minutes.day(sym, d_ref), ref_t)
        if c_ref is None or not c_ref > 0:
            return None
        scale = float(ref_close) / float(c_ref)
        for d in self.schedule.sessions_between(d_ref, day_of(t))[::-1]:
            c = _last_close(self.minutes.day(sym, int(d)), t)
            if c is not None:
                return float(c) * scale, scale
        return None

    # ── reporting only (may read after the pass: never used to decide) ──
    def reopen(self, sym: str, h: dict) -> Optional[int]:
        """When the halt ``h`` ended: its resume in the file, else the end of the first 1-minute bar with volume after
        it, at the latest the next regular session's open."""
        if h.get("resume") is not None:
            return int(h["resume"])
        cap = self.schedule.next_open(int(h["start"]))
        for d in self.schedule.sessions_between(day_of(h["start"]), day_of(cap)):
            b = self.minutes.day(sym, int(d))
            if b and len(b["s"]):
                bs = np.asarray(b["s"], np.int64)
                m = np.flatnonzero((bs >= int(h["start"])) & (np.asarray(b["v"], float) > 0))
                if len(m):
                    return int(min(bs[m[0]] + MINUTE, cap))
        return int(cap)

    def event(self, sym: str, T: int, P1: int, h: dict) -> dict:
        """The incidence record of a candidate halted at its first pass ``P1``."""
        rp = self.reopen(sym, h)
        free = None
        for P in self.schedule.passes(T, until=self.schedule.wait_end(T)):
            if P >= P1 and self.halt_at(sym, int(P)) is None:
                free = int(P)
                break
        return {"sym": plain(sym), "T": iso_et(T), "first_pass": iso_et(P1), "reason": h["reason"],
                "missing_resume": bool(h["missing"]), "halt_start": iso_et(h["start"]), "reopen": iso_et(rp),
                "reopen_min": None if rp is None else (rp - int(P1)) / MINUTE,
                "free_pass": iso_et(free), "free_min": None if free is None else (free - int(P1)) / MINUTE}


class NoHalts(HaltModel):
    """Nothing halted (the parity check: the pass loop without halts must reproduce the simulator's selection)."""

    def __init__(self, schedule: Schedule):
        super().__init__(HaltBook(pd.DataFrame(columns=["sym", "start", "resume", "reason", "exchange"])),
                         MinuteBars(source=lambda s, d: None), schedule)

    def halt_at(self, sym, t):
        return None


def _wait_ok(mode: str, price: float, target: Optional[float], pick_close: Optional[float]) -> bool:
    """H1's condition at the reopen: above the target (H1a; no target: always), or at least half the run-up still to
    give back (H1b)."""
    if target is None:
        return True
    if mode == "wait_half":
        return price >= float(pick_close) - 0.5 * (float(pick_close) - float(target))
    return price > float(target)


# ── one candidate on its own (a fall or ETF turn entry) ──────────────────────

@dataclass
class Entry:
    kind: str                              # "first" (unchanged) / "delayed" (a later pass) / "lost"
    k: int = 0                             # the pass it enters at
    at: Optional[int] = None               # that pass (UTC ns)
    price: Optional[float] = None          # traded dollars
    scale: Optional[float] = None
    why: str = ""
    event: Optional[dict] = None           # the incidence record (a candidate halted at its first pass)


def decide(model: HaltModel, mode: str, sym: str, T: int, close: float, target: Optional[float] = None,
           pick_close: Optional[float] = None) -> Entry:
    """One candidate whose bar ends at ``T`` (close ``close``, traded dollars) under ``mode``: unchanged when not
    halted at its first pass; else its first later pass where it is not halted and has a price, the target checked
    there (``target_reached``); halted at every pass of its window -> ``halted_expired``, or under H1 the wait to the
    next session's close (H1a: above the target; H1b: half the run-up left). Lendability is the caller's (judged at
    the entry)."""
    W = model.schedule.passes(T)
    if not len(W):
        return Entry("first")
    h1 = model.halt_at(sym, int(W[0]))
    if h1 is None:
        return Entry("first")
    ev = model.event(sym, T, int(W[0]), h1)
    halted_all = True
    for k in range(1, len(W)):
        P = int(W[k])
        if model.halt_at(sym, P) is not None:
            continue
        halted_all = False
        px = model.price(sym, P, T, close)
        if px is None:
            continue                                         # no price: retried at the next pass (production)
        if target is not None and px[0] <= target:
            return Entry("lost", k, P, px[0], px[1], "target_reached", ev)
        return Entry("delayed", k, P, px[0], px[1], "", ev)
    if not halted_all:
        return Entry("lost", why="expired_no_price", event=ev)
    if mode not in WAIT_MODES:
        return Entry("lost", why="halted_expired", event=ev)
    V = model.schedule.passes(T, until=model.schedule.wait_end(T))
    V = V[V > W[-1]]
    for k2, P in enumerate(V):
        P = int(P)
        if model.halt_at(sym, P) is not None:
            continue
        px = model.price(sym, P, T, close)
        if px is None:
            continue
        if _wait_ok(mode, px[0], target, pick_close):
            return Entry("delayed", len(W) + k2, P, px[0], px[1], "wait", ev)
        return Entry("lost", len(W) + k2, P, px[0], px[1], "wait_half" if mode == "wait_half" else "wait_target", ev)
    return Entry("lost", why="wait_expired", event=ev)


# ── the vol arm: production's pass loop ──────────────────────────────────────

@dataclass
class Cand:
    src: str                     # "L": the bar's pick; "P": a backup of the bar's pool (the engine glue's arrays)
    j: int                       # its place in that selection's arrays
    pos: int                     # its place in the evaluator's arrays
    row: int                     # its store row (a piece's "R")
    sym: str                     # its traded symbol
    tk: int                      # the name (one trade per name per day per arm)
    T: int                       # its bar's end (UTC ns)
    close: float                 # its bar's close, traded dollars
    target: Optional[float]      # its give-back target, traded dollars (None: none)
    lend0: bool                  # lendable at the first pass (the simulator's own lendability at the bar)
    day: int                     # its bar's session (day number)
    hi: bool = False             # under borrow pressure (V6's extras)


@dataclass
class Run:
    pick: Cand
    backups: List[Cand] = field(default_factory=list)


@dataclass
class Trade:
    cand: Cand
    k: int                       # the pass index (0: the bar's close at its end)
    at: int                      # its entry instant
    price: float                 # traded dollars
    scale: Optional[float]
    role: str                    # pick / fallback / halted_fallback / extra / wait / alt (PREREG227's walk)


class VolMachine:
    """Production's entry passes for the vol arm (`tracker.record_sel_short_trades`): at each pass every pending pick,
    the oldest bar first — ``attempt`` (halt, price, target, lendability), the same-bar FALLBACK when IBKR cannot lend
    it (``pool``: the backups in rank order, a halted one skipped and left for a later pass; the first lendable one not
    traded today opens and settles the pick), then the pressure EXTRAS while the pick is pending (``extras``: every
    HIGH backup not yet settled). A pick expires after its window's last pass; H1 lets one halted at every pass wait
    alone; H2 / H3 make a halt at the first pass one more fallback trigger. ``lend_later(cand, at, price)``: its
    lendability when first priced at a later pass. Without halts the result is the simulator's own selection
    (PREREG16's fallback + V6's extras) — `Applier` checks it every time.

    ``walk`` (PREREG227's S3, `Mimic.vol_walk`): production walks a pick's backups until one OPENS — a backup the
    ACCOUNT refuses (another book's name, two shorts of the name, no room) passes to the next. The pass loop cannot see
    the account, so when a backup enters as the fallback it also records the backups the walk would reach after it at
    that pass (``chain``: store row -> (chain key, its place after the chosen one, HIGH)): each later backup that is
    lendable there and not traded that day becomes an ALTERNATIVE trade (role "alt"; not "traded today", nothing
    settled), a HIGH one is the extras loop's as before and joins the chain; the engine opens the first member the
    account takes (`account.simulate`'s ``alt_of``)."""

    def __init__(self, model: HaltModel, mode: str, pool: bool, extras: bool,
                 lend_later: Callable[[Cand, int, float], bool], walk: bool = False):
        if mode not in MODES:
            raise ValueError(f"halt mode must be one of {MODES}, not {mode!r}")
        self.model, self.mode, self.pool, self.extras, self.lend_later = model, mode, pool, extras, lend_later
        self.walk = bool(walk)

    def run(self, runs: Sequence[Run]) -> "VolMachine":
        sched = self.model.schedule
        n = len(runs)
        self.trades: List[Trade] = []
        self.events: List[dict] = []
        self._wait: Dict[int, np.ndarray] = {}
        W = [sched.passes(r.pick.T) for r in runs]
        # the processing order at a pass: the simulator's run order (the oldest bar first); a run whose pick's bar
        # ended before an earlier run's (the run is a bar POSITION, live a CLOCK bar) keeps its place
        base = [0] * n
        prev = None
        for r in range(n):
            P1 = int(W[r][0]) if len(W[r]) else int(runs[r].pick.T)
            base[r] = P1 if prev is None else max(P1, prev)
            prev = base[r]
        heap = [(max(int(P), base[r]), r, k, 0) for r in range(n) for k, P in enumerate(W[r])]
        heapq.heapify(heap)
        self._done = [False] * n
        self._fb = [False] * n
        self._halted_all = [True] * n
        self._lend: Dict[tuple, bool] = {}
        self._bdone: set = set()
        self._traded: set = set()
        self._ev: Dict[tuple, dict] = {}
        self.chain: Dict[int, Tuple[str, int, bool]] = {}        # PREREG227's walk: store row -> (key, place, HIGH)
        self._head: Dict[Tuple[int, int], int] = {}              # (run, pass) -> the chosen backup's index
        while heap:
            _, r, k, phase = heapq.heappop(heap)
            if self._done[r]:
                continue
            run = runs[r]
            P = int(W[r][k]) if phase == 0 else int(self._wait[r][k - len(W[r])])
            if phase == 1:
                self._wait_pass(r, run, k, P)
                continue
            first = k == 0
            out = self._pick(r, run, k, P, first)
            if out != "halted":
                self._halted_all[r] = False
            if out == "halted":
                if first and self.mode in BACKUP_MODES and self.pool:
                    b = self._fallback(r, run, k, P, first, "halted_fallback")
                    if b is not None:
                        self._fb[r] = True
                        e = self._ev.get(("p", r))
                        if e is not None:
                            e["halted_fallback"] = b.cand.sym
                        if self.mode == "backup":            # H2: settled by its backup; H3: it waits too
                            self._done[r] = True
                            self._note(("p", r), outcome="halted_fallback")
            elif out == "target_reached":
                self._done[r] = True
                self._note(("p", r), outcome="target_reached", at_k=k)
            elif out == "unlendable":
                e = self._ev.get(("p", r))
                if e is not None:
                    e.setdefault("unlendable_at_k", k)
                if self.pool and not self._fb[r]:
                    b = self._fallback(r, run, k, P, first, "fallback")
                    if b is not None:
                        self._fb[r] = True
                        self._done[r] = True
                        self._note(("p", r), outcome="unlendable_fallback", fallback=b.cand.sym, at_k=k)
            elif isinstance(out, tuple):
                self.trades.append(Trade(run.pick, k, run.pick.T if first else P, out[0], out[1], "pick"))
                self._traded.add((run.pick.tk, run.pick.day if first else day_of(P)))
                self._done[r] = True
                self._note(("p", r), outcome="entered", at_k=k, at=iso_et(P), price=out[0], scale=out[1])
            if self.extras:
                self._extras(r, run, k, P, first)
            if k == len(W[r]) - 1 and not self._done[r]:
                if self._halted_all[r] and self.mode in WAIT_MODES:
                    V = sched.passes(run.pick.T, until=sched.wait_end(run.pick.T))
                    V = V[V > W[r][-1]]
                    self._wait[r] = V
                    for k2, P2 in enumerate(V):
                        heapq.heappush(heap, (int(P2), r, len(W[r]) + k2, 1))
                    if not len(V):
                        self._done[r] = True
                        self._note(("p", r), outcome="wait_expired")
                else:
                    self._done[r] = True
                    self._note(("p", r), outcome="halted_expired" if self._halted_all[r] else "expired")
        for e in self._ev.values():
            e.setdefault("outcome", "not_retried")
            self.events.append(e)
        return self

    # ── one candidate at one pass ──
    def _attempt(self, r: int, key, c: Cand, k: int, P: int, first: bool):
        h = self.model.halt_at(c.sym, P)
        if h is not None:
            if (key, r) not in self._ev:
                ev = self.model.event(c.sym, c.T, P, h) if k == 0 else {
                    "sym": c.sym, "T": iso_et(c.T), "first_pass": iso_et(P), "reason": h["reason"],
                    "missing_resume": bool(h["missing"]), "halted_from_k": k}
                ev.update(role="pick" if key == "p" else "backup", row=c.row, pick_day=iso_day(c.day),
                          close=c.close, target=c.target, halted_at_first=k == 0, hi=c.hi)
                self._ev[(key, r)] = ev
            return "halted"
        if first:
            price, scale = c.close, None
        else:
            got = self.model.price(c.sym, P, c.T, c.close)
            if got is None:
                return "no_price"
            price, scale = got
        if c.target is not None and price <= c.target:
            return "target_reached"
        lk = (r, key)
        if lk not in self._lend:
            self._lend[lk] = bool(c.lend0) if first else bool(self.lend_later(c, P, price))
        if not self._lend[lk]:
            return "unlendable"
        return (price, scale)

    def _pick(self, r: int, run: Run, k: int, P: int, first: bool):
        if self._lend.get((r, "p")) is False and not (self.pool and not self._fb[r] and self._open_backups(r, run)):
            return "unlendable_idle"                      # judged unlendable and no backup left: nothing can change
        return self._attempt(r, "p", run.pick, k, P, first)

    def _open_backups(self, r: int, run: Run) -> bool:
        return any((r, b) not in self._bdone for b in range(len(run.backups)))

    def _fallback(self, r: int, run: Run, k: int, P: int, first: bool, role: str) -> Optional[Trade]:
        """PREREG16's same-bar fallback at this pass: the backups in rank order (a halted one is skipped, left for a
        later pass); the first lendable one not traded today opens."""
        for b, c in enumerate(run.backups):
            if (r, b) in self._bdone:
                continue
            day = c.day if first else day_of(P)
            if (c.tk, day) in self._traded:
                self._bdone.add((r, b))
                self._note((b, r), outcome="traded_today")
                continue
            out = self._attempt(r, b, c, k, P, first)
            if out in ("halted", "no_price"):
                continue
            if not isinstance(out, tuple):
                self._bdone.add((r, b))
                self._note((b, r), outcome=out)
                continue
            t = Trade(c, k, c.T if first else P, out[0], out[1], role)
            self.trades.append(t)
            self._traded.add((c.tk, day))
            self._bdone.add((r, b))
            self._note((b, r), outcome="entered", at_k=k, at=iso_et(P), price=out[0], scale=out[1], as_role=role)
            if self.walk:
                self._alternatives(r, run, k, P, first, b, c)
            return t
        return None

    def _alternatives(self, r: int, run: Run, k: int, P: int, first: bool, b0: int, c0: Cand) -> None:
        """PREREG227's walk: the backups after the chosen one ``b0`` that the walk would try at this pass were the
        account to refuse it — the HIGH ones are the extras loop's (`_extras` joins them to the chain), every other one
        lendable here and not traded that day an alternative trade (role "alt", nothing settled)."""
        key = f"w{int(run.pick.row)}_{int(k)}"
        self._head[(r, k)] = b0
        self.chain[int(c0.row)] = (key, 0, bool(c0.hi))
        for b, c in enumerate(run.backups):
            if b <= b0 or c.hi or (r, b) in self._bdone:
                continue
            day = c.day if first else day_of(P)
            if (c.tk, day) in self._traded:
                continue
            out = self._attempt(r, b, c, k, P, first)
            if not isinstance(out, tuple):
                continue                                   # halted, no price, at its target or unlendable here
            self.trades.append(Trade(c, k, c.T if first else P, out[0], out[1], "alt"))
            self.chain[int(c.row)] = (key, int(b - b0), False)

    def _extras(self, r: int, run: Run, k: int, P: int, first: bool) -> None:
        """V6: every HIGH backup not yet settled (a halted one is left for a later pass while the pick is pending)."""
        for b, c in enumerate(run.backups):
            if not c.hi or (r, b) in self._bdone:
                continue
            day = c.day if first else day_of(P)
            if (c.tk, day) in self._traded:
                self._bdone.add((r, b))
                self._note((b, r), outcome="traded_today")
                continue
            out = self._attempt(r, b, c, k, P, first)
            if out in ("halted", "no_price"):
                continue
            self._bdone.add((r, b))
            if isinstance(out, tuple):
                self.trades.append(Trade(c, k, c.T if first else P, out[0], out[1], "extra"))
                self._traded.add((c.tk, day))
                self._note((b, r), outcome="entered", at_k=k, at=iso_et(P), price=out[0], scale=out[1], as_role="extra")
                b0 = self._head.get((r, k))
                if self.walk and b0 is not None and b > b0:   # PREREG227's walk: an extra after this pass's fallback
                    self.chain[int(c.row)] = (f"w{int(run.pick.row)}_{int(k)}", int(b - b0), True)
            else:
                self._note((b, r), outcome=out)

    def _wait_pass(self, r: int, run: Run, k: int, P: int) -> None:
        """H1: a pick halted at every pass of its window, alone, at a pass after it."""
        c = run.pick
        if self.model.halt_at(c.sym, P) is not None:
            if P >= int(self._wait[r][-1]):
                self._done[r] = True
                self._note(("p", r), outcome="wait_expired")
            return
        got = self.model.price(c.sym, P, c.T, c.close)
        if got is None:
            if P >= int(self._wait[r][-1]):
                self._done[r] = True
                self._note(("p", r), outcome="wait_expired")
            return
        self._done[r] = True
        if not _wait_ok(self.mode, got[0], c.target, c.close):
            self._note(("p", r), outcome="wait_half" if self.mode == "wait_half" else "wait_target", at_k=k,
                       at=iso_et(P), price=got[0])
            return
        if not bool(self.lend_later(c, P, got[0])):
            self._note(("p", r), outcome="wait_unlendable", at_k=k, at=iso_et(P), price=got[0])
            return
        self.trades.append(Trade(c, k, P, got[0], got[1], "wait"))
        self._traded.add((c.tk, day_of(P)))
        self._note(("p", r), outcome="entered", at_k=k, at=iso_et(P), price=got[0], scale=got[1], as_role="wait")

    def _note(self, key: tuple, **kw) -> None:
        """The outcome of a candidate a halt touched (``key``: ("p", run) or (backup, run); nothing for the others)."""
        e = self._ev.get(key)
        if e is not None and "outcome" not in e:
            e.update(kw)


# ── the engine glue (`mimic.live_books` with ``Mimic(halts=...)``) ───────────

class _Fixed:
    """The research assembler reads a trade table from ``trades``: these trades, the evaluator for the rest."""

    def __init__(self, e, t):
        self._e, self._t = e, t

    def trades(self, _prm):
        return self._t

    def __getattr__(self, a):
        return getattr(self._e, a)


@dataclass
class Item:
    """One vol-engine candidate to build a piece for: its evaluator position, the selection's deadline bar / mark
    flag / close / target (stored units), and its entry: instant, price (stored units), first exit bar (path index),
    session, and the ATR% and clock bar that price its spread. ``at_pick``: entered at its own bar's close (the marks
    as `VolEngine._pieces` keeps them); else as `VolEngine.turn_pieces` keeps them (none before the entry)."""
    pos: int
    eff: int
    marked: bool
    px: float
    lvl: float
    ens: int
    Ein: float
    j0: int
    start_day: int
    atr_in: float
    bod_in: int
    extra: dict = field(default_factory=dict)
    at_pick: bool = False


def build_vol_pieces(eng, ev, cfg: dict, prm: dict, items: Sequence[Item], strategy: str) -> List[dict]:
    """The research engine's pieces for vol-engine candidates entered as ``items`` say — `VolEngine.turn_pieces` 's
    construction with any entry: the exits (target, volatility exit in profit, cover of the entry) from the path bar
    ``j0`` to the deadline bar, the leg costs, the borrow from the entry's settlement; marks after the entry only.
    An entry at a bar's end with that bar's close reproduces `trades_at_pick` / `turn_pieces` (`Applier` checks)."""
    if not items:
        return []
    st, ve = eng.st, eng.ve
    ix = np.array([it.pos for it in items], np.int64)
    R = ev.r[ix]
    eff = np.array([it.eff for it in items], np.int64)
    marked = np.array([bool(it.marked) for it in items], bool)
    lvl = np.array([it.lvl for it in items], float)
    ens = np.array([it.ens for it in items], np.int64)
    Ein = np.array([it.Ein for it in items], float)
    j0 = np.array([it.j0 for it in items], np.int64)
    start_day = np.array([it.start_day for it in items], np.int64)
    n = len(items)
    ar = np.arange(n)
    PCc, PAc = np.asarray(st.PC[R], float), np.asarray(st.PA[R], float)
    J = np.arange(PCc.shape[1])[None, :]
    apk = ev.apk[ix].astype(float)
    BIG = 1 << 30
    valid = (J >= j0[:, None]) & (J <= eff[:, None])
    with np.errstate(invalid="ignore"):
        th = valid & (PCc <= lvl[:, None])
        t_first = np.where(th.any(1), th.argmax(1), BIG)
        if prm.get("volnorm") is not None:
            vh = valid & (PAc <= float(prm["volnorm"]) * apk[:, None]) & (PCc < Ein[:, None])
            v_first = np.where(vh.any(1), vh.argmax(1), BIG)
        else:
            v_first = np.full(n, BIG)
        if prm.get("cap") is not None:
            ch = valid & (PCc >= float(prm["cap"]) * Ein[:, None])
            c_first = np.where(ch.any(1), ch.argmax(1), BIG)
        else:
            c_first = np.full(n, BIG)
    first = np.minimum(np.minimum(t_first, v_first), c_first)
    hit = first < BIG
    xi = np.where(hit, first, eff)
    why = np.where(hit, np.where(first == t_first, 0, np.where(first == v_first, 1, 4)), np.where(marked, 3, 2))
    nobar = xi < 0                                         # cut before its first path bar: marked at its entry
    xi_ = np.maximum(xi, 0)
    X = np.where(nobar, Ein, PCc[ar, xi_])
    xns = np.where(nobar, ens, np.asarray(st.PNS[R, xi_], np.int64))
    cdays = np.where(nobar, 0, np.asarray(st.PSD[R, xi_], np.int64) - start_day)
    hs_in = ev.halfspread(ix, ens, Ein, np.array([it.atr_in for it in items], float),
                          np.array([it.bod_in for it in items]))
    hs_out = ev.halfspread(ix, xns, X, np.where(nobar, apk, PAc[ar, xi_]),
                           np.where(nobar, np.array([it.bod_in for it in items]), np.asarray(st.PBD[R, xi_])))
    NOT = ve.NOTIONAL
    px_, act = ev.px[ix], ev.actual[ix]

    def comm(q):
        return np.minimum(np.maximum(1.0, 0.005 * NOT / q), 0.01 * NOT) / NOT
    a_in, a_out = Ein / px_ * act, X / px_ * act
    c_in = hs_in / 1e4 + comm(a_in) + ve.SEC_FEE + np.minimum(ve.TAF_PER_SHARE * NOT / a_in, ve.TAF_MAX) / NOT
    c_out = hs_out / 1e4 + comm(a_out)
    fee = ev.fee_fn(ev.tkn[ix], start_day, cdays, ev.fee[ix])
    net = (((Ein * (1 - c_in)) - X * (1 + c_out)) / (Ein * (1 - c_in)) - fee / 100 * cdays / 365) * 100
    T = {"idx": ix, "net": net, "days": (xns - ens) / ve.DAY_NS, "d": ev.dn[ix], "year": ev.year[ix], "why": why,
         "entry": Ein, "exit": X, "ens": ens, "xns": xns, "cdays": cdays, "hs_in": hs_in, "hs_out": hs_out,
         "passive": np.zeros(n, bool), "fill_bars": j0, "model_entry": np.array([it.px for it in items], float),
         "fee_eff": fee, "lvl": lvl, "eff": eff, "ei": j0}
    eng.asm.cache.clear()                                  # its key cannot tell these entries from another entry
    P = eng.asm.pieces(_Fixed(ev, T), cfg)
    eng.asm.cache.clear()
    out = []
    for c, q in enumerate(P):
        i = int(ix[c])
        d = dict(q)
        f = float(ev.actual[i] / ev.px[i])
        xr = int(d["xi"])
        ns_ = np.asarray(st.PNS[int(R[c])][:xr + 1], np.int64)
        if items[c].at_pick:                               # `VolEngine._pieces`: every path mark to the exit
            pa = np.asarray(st.PA[int(R[c])][:xr + 1][ns_ >= 0], float)
        else:                                              # `VolEngine.turn_pieces`: no mark before the entry
            keep = np.asarray(d["pt"], np.int64) > int(d["ens"])
            for key in ("pt", "pc", "ph"):
                d[key] = np.asarray(d[key])[keep]
            okk = (ns_ >= 0) & (ns_ <= int(d["xns"]))
            pa = np.asarray(st.PA[int(R[c])][:xr + 1][okk], float)[keep]
        lv = float(lvl[c])
        d.update(strategy=strategy, pick_day=int(ev.dn[i]),
                 lvl=lv * f if np.isfinite(lv) else None, apk=float(apk[c]), pa=pa,
                 vn=None if prm.get("volnorm") is None else float(prm["volnorm"]),
                 cover_mult=None if prm.get("cap") is None else float(prm["cap"]),
                 why=str(eng.WHY[int(why[c])]), f=f)
        d.update(items[c].extra)
        out.append(d)
    return out


def etf_entry_piece(arm, r, a: dict, p: int, last: int, beyond: bool, px: float, lvl: float, f: float, turn: tuple,
                    cover: Optional[float], strategy: str, ctx: bool, ens: int, e: float) -> Optional[dict]:
    """`EtfArm._turn_piece` 's piece for an entry at instant ``ens`` and price ``e`` (stored units): the exits from the
    first bar ending after it to the deadline bar ``last``; the spread priced at the last bar ended by then. ``ens`` =
    the turning bar's end with its close reproduces the turn piece. None past the deadline."""
    from src.backtest import account as A
    k_, W_ = float(turn[0]), int(turn[1])
    c, h, atr, ns, sday, bod = a["c"], a["h"], a["atr"], a["ns"], a["sday"], a["bod"]
    atr_pick = float(np.float32(atr[p]))
    j0 = int(np.searchsorted(ns, int(ens), side="right"))
    if j0 > last:
        return None
    jb = max(j0 - 1, p)
    s2, a2 = c[j0:last + 1], atr[j0:last + 1]
    hits = [np.flatnonzero(s2 <= lvl), np.flatnonzero((a2 <= 0.5 * atr_pick) & (s2 < e))]
    if cover:
        hits.append(np.flatnonzero(s2 >= cover * e))
    first = min((int(x[0]) for x in hits if len(x)), default=None)
    xi = j0 + first if first is not None else last
    hs = arm.cm.predict(np.array([e * f, float(c[xi]) * f]), np.array([float(r.dv20)] * 2),
                        np.array([float(atr[jb]), float(atr[xi])]), np.array([int(bod[jb]), int(bod[xi])]),
                        np.array([int(r.year)] * 2))
    d_in = int(A.day_numbers([int(ens)])[0])
    d_out = int(A.day_numbers([int(ns[xi])])[0])
    d0 = arm.settle(d_in)
    days = np.arange(d0, arm.settle(d_out) + 30, dtype=np.int64)
    lastbar = np.r_[np.flatnonzero(np.diff(sday) != 0), len(sday) - 1]
    sd, sc = sday[lastbar].astype(np.int64), c[lastbar]
    jj = np.searchsorted(sd, days) - 1
    coll = np.ceil(1.02 * sc[np.maximum(jj, 0)] * f - 1e-9)
    from src.backtest.strategies.etf import FEE_MED
    fb = float(r.fee_file) if np.isfinite(r.fee_file) else FEE_MED
    cum = np.cumsum(coll * arm.rf.daily(r.tkn, days, fb))
    q = {"strategy": strategy, "tkn": r.tkn, "ens": int(ens), "xns": int(ns[xi]), "e": e * f,
         "x": float(c[xi]) * f, "hs_in": float(hs[0]), "hs_out": float(hs[1]), "dv20": float(r.dv20),
         "pt": ns[j0:xi + 1].astype(np.int64), "pc": c[j0:xi + 1] * f, "ph": h[j0:xi + 1] * f,
         "d0": d0, "cum": cum, "bfac": 1.0 / 100.0 / 360.0, "d_out": d_out, "pick_day": int(r.dn),
         "year": int(r.year), "turn_k": k_, "turn_W": W_, "pick_px": px * f}
    if ctx:
        why = (("mark" if beyond else "time") if first is None
               else "target" if len(hits[0]) and int(hits[0][0]) == first
               else "volnorm" if len(hits[1]) and int(hits[1][0]) == first else "cap")
        q.update(lvl=lvl * f, apk=atr_pick, pa=np.asarray(atr[j0:xi + 1], float), vn=0.5, cover_mult=cover,
                 why=why, f=f)
    return q


def _same(a: dict, b: dict) -> List[str]:
    """The fields two pieces differ in (the ones the account engine and the post-processing read)."""
    bad = []
    for k in ("tkn", "ens", "xns", "e", "x", "hs_in", "hs_out", "dv20", "d0", "d_out", "bfac", "full_ps", "pick_day",
              "pt", "pc", "ph", "cum", "lvl", "apk", "pa", "vn", "cover_mult", "why", "f"):
        if k not in a and k not in b:
            continue
        x, y = a.get(k), b.get(k)
        if isinstance(x, (np.ndarray, list)) or isinstance(y, (np.ndarray, list)):
            x, y = np.asarray(x if x is not None else [], float), np.asarray(y if y is not None else [], float)
            if x.shape != y.shape or not np.allclose(x, y, rtol=1e-12, atol=0.0, equal_nan=True):
                bad.append(k)
        elif isinstance(x, float) or isinstance(y, float):
            if x is None or y is None or not (x == y or (np.isnan(x) and np.isnan(y)) or abs(x - y) <= 1e-12 * max(abs(x), abs(y))):
                bad.append(k)
        elif x != y:
            bad.append(k)
    return bad


def pressure_flags(eng) -> Optional[np.ndarray]:
    """Each store row under borrow pressure (V6's extras): ``eng.pressure_hi`` when set, else — V6 installed
    (`run71.make_real6` 's ``Real6`` as the engine's evaluator) — that module's ``NFHI``; None without V6."""
    v = getattr(eng, "pressure_hi", None)
    if v is not None:
        return np.asarray(v, bool)
    cls = getattr(eng, "Real", None)
    if getattr(cls, "__name__", "") == "Real6":
        import sys
        mod = sys.modules.get(getattr(cls, "__module__", ""))
        if mod is not None and getattr(mod, "NFHI", None) is not None:
            return np.asarray(mod.NFHI, bool)
    return None


def v6_installed(eng) -> bool:
    """V6's pressure extras are part of the vol arm's selection: the engine's evaluator is V6's ``Real6``
    (`run71.make_real6`, installed by a study harness's `R75.Applied`). The plain evaluator — the reliability cards'
    construction — has none, though set A0's engine carries its pressure flags (``pressure_hi``) all the same."""
    return getattr(getattr(eng, "Real", None), "__name__", "") == "Real6"


_MODELS: Dict[tuple, HaltModel] = {}


def default_model(cal) -> HaltModel:
    """The halt model of this process: every year's halt file, the 1-minute bars under `minute_dir`, production's
    passes over the research calendar ``cal``."""
    paths = halt_paths()
    key = (str(minute_dir()), bool(_CONF["fetch"]), float(_CONF["rate"]), int(len(cal)), int(cal[0]), int(cal[-1]))
    if paths is not None:
        key = key + (tuple(paths),)
    if key not in _MODELS:
        win = float(getattr(settings, "sel_short_entry_max_age_minutes", 75.0) or 75.0)
        _MODELS[key] = HaltModel(halt_book(paths), MinuteBars(minute_dir(), fetch=bool(_CONF["fetch"]),
                                                              rate=float(_CONF["rate"])),
                                 Schedule(cal, production_slots(), win))
    return _MODELS[key]


class Applier:
    """``Mimic(halts=...)`` inside `mimic.live_books`: the ETF arm's turn entries as the adapter builds them
    (`etf_pieces`), then the vol and fall books (`apply`); each delayed or new piece carries ``halt`` (its pass, role,
    mode). V6's extras (`R75.Applied`) need each store row's borrow pressure (`pressure_flags`: ``eng.pressure_hi``, else
    V6's own NFHI): the pass loop without halts must reproduce the selection, or the build stops."""

    VALIDATE = 40                                          # original pieces rebuilt per arm to check the builder

    def __init__(self, mim, eng, L: np.ndarray, model: Optional[HaltModel] = None):
        if mim.halts not in MODES:
            raise ValueError(f"Mimic.halts must be one of {MODES}, not {mim.halts!r}")
        if not mim.entry:
            raise ValueError("the halt model needs Mimic(entry=True): the first pass enters at the bar's close")
        if mim.short_limit is not None:
            raise NotImplementedError("the halt model does not cover the short arms' sell limits (PREREG66)")
        if mim.fall_turn is None or mim.etf_turn is None:
            raise NotImplementedError("the halt model covers the fall and ETF arms on the turn (production's)")
        from src.backtest.mimic import turn_lend
        self.mim, self.mode, self.eng = mim, str(mim.halts), eng
        self.walk = bool(getattr(mim, "vol_walk", False))     # PREREG227's S3: the fallback walked to the account
        self.L = np.asarray(L, bool)
        self.model = model or default_model(np.asarray(eng.C4.CAL, np.int64))
        self.lend_at = turn_lend(eng, mim, self.L)
        self.events: List[dict] = []
        self.scope: Dict[str, Dict[str, List[int]]] = {"vol": {}, "fall": {}, "etf": {}}
        self.counts: Dict[str, int] = {}

    def _count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def _scope(self, arm: str, day: int, halted: bool, traded: Optional[bool] = None) -> None:
        v = self.scope[arm].setdefault(iso_day(day), [0, 0, 0, 0])
        v[0] += 1
        v[1] += int(bool(halted))
        if traded is not None:
            v[2] += int(bool(traded))
            v[3] += int(bool(halted) and bool(traded))

    def _lend_later(self, c: Cand, at: int, price: float) -> bool:
        if self.lend_at is None:
            return bool(c.lend0)
        return bool(self.lend_at(int(c.row), int(at), float(price)))

    # ── the ETF arm: its turn entries as `EtfArm.pieces` builds them ──
    def etf_pieces(self, etf):
        """``etf.pieces`` with each turn entry through the halt model (its lendability is then judged by the adapter
        at the entry)."""
        def call(lo, hi, **kw):
            had = "_turn_piece" in etf.__dict__
            prev = etf.__dict__.get("_turn_piece")
            orig = prev if had else type(etf)._turn_piece.__get__(etf)
            pend: List[Tuple[dict, dict]] = []

            def wrapped(r, a, p, i0, last, beyond, px, lvl, f, turn, cover, strategy, ctx):
                q = orig(r, a, p, i0, last, beyond, px, lvl, f, turn, cover, strategy, ctx)
                if q is None:
                    return None
                tgt = float(lvl) * float(f) if np.isfinite(lvl) else None
                d = decide(self.model, self.mode, plain(r.tkn), int(q["ens"]), float(q["e"]), tgt, float(px) * float(f))
                self._scope("etf", int(r.dn), d.kind != "first")
                if d.kind == "first":
                    return q
                ev = dict(d.event or {}, arm="etf", role="turn", tkn=str(r.tkn), pick_day=iso_day(int(r.dn)),
                          close=float(q["e"]), target=tgt, pick_close=float(px) * float(f), halted_at_first=True)
                self.events.append(ev)
                if d.kind == "lost":
                    ev.update(outcome=d.why, at_k=d.k, at=iso_et(d.at), price=d.price)
                    self._count(f"etf_{d.why}")
                    return None
                q2 = etf_entry_piece(etf, r, a, p, last, beyond, px, lvl, f, turn, cover, strategy, ctx,
                                     int(d.at), float(d.price) / float(f))
                ev.update(at_k=d.k, at=iso_et(d.at), price=d.price, scale=d.scale, as_role="wait" if d.why == "wait" else "turn")
                if q2 is None:
                    ev["outcome"] = "past_deadline"
                    self._count("etf_past_deadline")
                    return None
                q2["halt"] = {"k": int(d.k), "role": "wait" if d.why == "wait" else "turn", "mode": self.mode}
                pend.append((ev, q2))
                return q2
            etf._turn_piece = wrapped
            try:
                res = etf.pieces(lo, hi, **kw)
            finally:
                if had:
                    etf._turn_piece = prev
                else:
                    etf.__dict__.pop("_turn_piece", None)
            kept = {id(q) for q in res}
            for ev, q2 in pend:                                 # the adapter judged lendability at the entry
                ev["outcome"] = "entered" if id(q2) in kept else "unlendable"
                self._count(f"etf_{ev['outcome']}")
            return res
        return call

    # ── the vol and fall books ──
    def apply(self, books: Dict[str, List[dict]], lo, hi, extra: Optional[dict], log: dict) -> None:
        books["vol"] = self._vol(books["vol"], lo, hi, extra)
        books["fall"] = self._fall(books["fall"], lo, hi)
        log["halts"] = {"mode": self.mode, "events": self.events, "scope": self.scope, "counts": dict(self.counts)}

    def _vol(self, pieces: List[dict], lo, hi, extra: Optional[dict]) -> List[dict]:
        eng = self.eng
        from src.backtest.strategies.vol import LIVE
        cfg, _p = eng.cfg(dict(LIVE))
        cfg = dict(cfg, **(extra or {}))
        ev = eng._real(lo, hi)
        ev.lend_all = self.L
        ev.fee_fn = eng.rf.eff
        EVm = eng.wf10.EV
        prm = EVm.norm(cfg)
        fb = int(prm.get("fallback", 0) or 0)
        pool = int(prm["top_k"]) == 1 and fb > 1
        saved = ev.borrow
        try:
            ev.borrow = np.ones(len(saved), bool)
            SL = EVm.Evaluator.select(ev, prm)
            SP = EVm.Evaluator.select(ev, dict(prm, top_k=fb, per_bar="all")) if pool else None
        finally:
            ev.borrow = saved
        hi_rows = pressure_flags(eng)
        extras = bool(pool and hi_rows is not None and v6_installed(eng))
        runs = self._vol_runs(ev, SL, SP, fb, hi_rows)
        sched = self.model.schedule
        null = VolMachine(NoHalts(sched), "guard", pool, extras, self._lend_later).run(runs)
        orig = {int(p["R"]): p for p in pieces}
        got0 = {t.cand.row for t in null.trades}
        if got0 != set(orig) or len(orig) != len(pieces):
            raise RuntimeError(
                f"halts: the vol arm's pass loop without halts does not reproduce the simulator's selection "
                f"({len(got0 - set(orig))} rows it adds, {len(set(orig) - got0)} it misses, {len(pieces)} pieces) — "
                f"V6's extras need eng.pressure_hi; a study's own vol parameters are not supported")
        real = VolMachine(self.model, self.mode, pool, extras, self._lend_later, walk=self.walk).run(runs)
        halted = {int(e["row"]) for e in real.events if e.get("role") == "pick" and e.get("halted_at_first")}
        for run in runs:
            self._scope("vol", run.pick.day, run.pick.row in halted)
        for e in real.events:
            e["arm"] = "vol"
            self.events.append(e)
            self._count(f"vol_{e.get('role')}_{e.get('outcome')}")
        sel = {"L": SL, "P": SP}
        self._validate_vol(ev, cfg, prm, sel, null.trades, orig)
        keep = {t.cand.row for t in real.trades if t.k == 0 and t.cand.row in orig}
        out = [p for p in pieces if int(p["R"]) in keep]    # the unchanged ones, in their order
        items = []
        for t in real.trades:
            if t.k == 0 and t.cand.row in orig:
                continue
            it = self._vol_item(ev, sel, t)
            if it is None:
                self._count("vol_past_deadline")
                for e in real.events:                         # the delayed entry fell past its deadline bar
                    if e.get("row") == t.cand.row and e.get("outcome") == "entered":
                        e["outcome"] = "past_deadline"
                continue
            items.append(it)
        add = build_vol_pieces(eng, ev, cfg, prm, items, "vol")
        if not add and len(out) == len(pieces):
            res = pieces
        else:
            res = out + sorted(add, key=lambda p: int(p["ens"]))
        return self._walk_stamps(res, real) if self.walk else res

    WALKED = ("fallback", "halted_fallback", "extra", "alt")

    def _walk_stamps(self, pieces: List[dict], real: "VolMachine") -> List[dict]:
        """PREREG227's S3: each chain member's ``alt_of`` / ``alt_k`` / ``alt_hi`` (`VolMachine.chain`) and, on every
        fallback, extra and alternative, ``walk_dedupe`` (production's `_sel_traded_today` in both loops) — on copies:
        the pieces may be the engine's cached ones."""
        role = {int(t.cand.row): t.role for t in real.trades}
        out = []
        for p in pieces:
            R = int(p["R"])
            ch, rl = real.chain.get(R), role.get(R)
            if ch is None and rl not in self.WALKED:
                out.append(p)
                continue
            q = dict(p)
            if rl in self.WALKED:
                q["walk_dedupe"] = True
            if ch is not None:
                q["alt_of"], q["alt_k"], q["alt_hi"] = ch[0], int(ch[1]), bool(ch[2])
            out.append(q)
        self._count("vol_walk_chains", len({c[0] for c in real.chain.values()}))
        self._count("vol_walk_alternatives", sum(1 for t in real.trades if t.role == "alt"))
        return out

    def _vol_runs(self, ev, SL, SP, pool_n: int, hi_rows) -> List[Run]:
        idxL = np.asarray(SL[0], np.int64)
        lend = np.asarray(ev.lend_all, bool)[ev.r]
        HI = np.asarray(hi_rows, bool)[ev.r] if hi_rows is not None else np.zeros(len(ev.r), bool)
        pos_all = np.r_[idxL, np.asarray(SP[0], np.int64) if SP is not None else np.zeros(0, np.int64)]
        end = dict(zip(pos_all.tolist(), ev.pick_end_ns(pos_all).tolist())) if len(pos_all) else {}
        by_run: Dict[int, List[int]] = {}
        if SP is not None:
            idx3 = np.asarray(SP[0], np.int64)
            for jj, i in enumerate(idx3):
                if 1 <= int(ev.rank[i]) < pool_n:
                    by_run.setdefault(int(ev.run[i]), []).append(jj)
            for v in by_run.values():
                v.sort(key=lambda jj: (-ev.s[idx3[jj]], ev.tkr[idx3[jj]]))

        def cand(src: str, j: int) -> Cand:
            S_ = SL if src == "L" else SP
            i = int(S_[0][j])
            f = float(ev.actual[i] / ev.px[i])
            lv = float(S_[5][j])
            return Cand(src=src, j=int(j), pos=i, row=int(ev.r[i]), sym=plain(ev.tkn[i]), tk=int(ev.tk[i]),
                        T=int(end[i]), close=float(ev.actual[i]), target=lv * f if np.isfinite(lv) else None,
                        lend0=bool(lend[i]), day=int(ev.dn[i]), hi=bool(HI[i]))
        runs = []
        for j in np.argsort(ev.run[idxL], kind="stable"):
            i = int(idxL[j])
            runs.append(Run(cand("L", int(j)), [cand("P", jj) for jj in by_run.get(int(ev.run[i]), [])]))
        return runs

    def _vol_item(self, ev, sel: dict, t: Trade) -> Optional[Item]:
        c = t.cand
        S_ = sel[c.src]
        j = c.j
        i = c.pos
        eff, marked, px, lvl = int(S_[1][j]), bool(S_[2][j]), float(S_[3][j]), float(S_[5][j])
        extra = {"halt": {"k": int(t.k), "role": t.role, "mode": self.mode}}
        if t.k == 0:
            return Item(i, eff, marked, px, lvl, int(c.T), px, 0, int(ev.dn[i]), float(ev.apk[i]), int(ev.bod[i]),
                        extra, at_pick=True)
        return self._later_item(ev, i, eff, marked, px, lvl, int(t.at), float(t.price), extra, pick_bar=-1)

    def _later_item(self, ev, i: int, eff: int, marked: bool, px: float, lvl: float, at: int, price: float,
                    extra: dict, pick_bar: int) -> Optional[Item]:
        """An entry at pass ``at`` and traded price ``price``: exits from the first path bar ending after it (none
        past the deadline bar); the spread priced at the last bar ended by then (the pick bar ``pick_bar`` = -1, or a
        turn bar)."""
        st = self.eng.st
        R = int(ev.r[i])
        ns_row = np.asarray(st.PNS[R], np.int64)
        after = np.flatnonzero((ns_row >= 0) & (ns_row > int(at)))
        if not len(after) or int(after[0]) > eff:
            return None
        j0 = int(after[0])
        f = float(ev.actual[i] / ev.px[i])
        jb = j0 - 1
        if jb >= 0 and jb >= pick_bar:
            atr_in, bod_in = float(st.PA[R, jb]), int(st.PBD[R, jb])
        else:
            atr_in, bod_in = float(ev.apk[i]), int(ev.bod[i])
        return Item(i, eff, marked, px, lvl, int(at), float(price) / f, j0, day_of(at), atr_in, bod_in, extra)

    def _validate_vol(self, ev, cfg, prm, sel, trades0: Sequence[Trade], orig: Dict[int, dict]) -> None:
        """The piece builder at the bar's close must reproduce the simulator's own pieces."""
        items = []
        for t in list(trades0)[: self.VALIDATE]:
            c = t.cand
            S_ = sel[c.src]
            items.append(Item(c.pos, int(S_[1][c.j]), bool(S_[2][c.j]), float(S_[3][c.j]), float(S_[5][c.j]),
                              int(c.T), float(S_[3][c.j]), 0, int(ev.dn[c.pos]), float(ev.apk[c.pos]),
                              int(ev.bod[c.pos]), at_pick=True))
        for it, q in zip(items, build_vol_pieces(self.eng, ev, cfg, prm, items, "vol")):
            bad = _same(q, orig[int(q["R"])])
            if bad:
                raise RuntimeError(f"halts: the vol piece builder does not reproduce row {q['R']} ({bad})")

    def _fall(self, pieces: List[dict], lo, hi) -> List[dict]:
        """The fall arm's turn entries (`VolEngine.turn_pieces` 's selection): each turn's passes from the turning
        bar's end; one halted at its first pass enters at its first free pass (lendability judged there) or not."""
        eng, mim = self.eng, self.mim
        from src.backtest.strategies.vol import FALL, turn_candidates
        import build as BLD                                    # the research engine's store layout (on sys.path)
        k_, W_ = float(mim.fall_turn[0]), int(mim.fall_turn[1])
        fp = dict(FALL)
        cfg, _p = eng.cfg(fp)
        cfg = dict(cfg, rule201=False, entry_at_pick=True)
        ev = eng._real(lo, hi)
        base = np.ones(len(self.L), bool) if self.lend_at is not None else self.L
        pre = eng.st.PRE[:, BLD.LS.index(5)]
        px_all = eng.st.C["px"].to_numpy(float)
        ev.lend_all = base & (pre > 0) & (px_all <= pre)
        ev.fee_fn = eng.rf.eff
        prm = eng.wf10.EV.norm(cfg)
        idx, eff, marked, px, _E, lvl = ev.select(prm)
        if not len(idx):
            return pieces
        st = eng.st
        R = ev.r[idx]
        f = (ev.actual[idx] / ev.px[idx]).astype(float)
        apk = ev.apk[idx].astype(float)
        cal = np.asarray(eng.C4.CAL, np.int64)
        jt = turn_candidates(st.PC[R], st.PSD[R], ev.dn[idx], cal, px, k_ * px * apk / 100.0, lvl, eff, W_)
        rows = np.flatnonzero(jt >= 0)
        orig = {int(p["R"]): p for p in pieces}
        at_row = {int(R[r]): int(r) for r in rows}
        for q in pieces:
            r = at_row.get(int(q["R"]))
            if r is None or int(jt[r]) != int(q["turn_bar"]):
                raise RuntimeError(f"halts: the fall arm's piece of row {q['R']} is not one of its turn entries")
        self._validate_fall(ev, cfg, prm, idx, eff, marked, px, lvl, jt, rows, orig)
        out_keep, items = [], []
        drop = set()
        for r in rows:
            i, Rr, j = int(idx[r]), int(R[r]), int(jt[r])
            T = int(st.PNS[Rr, j])
            close = float(st.PC[Rr, j]) * float(f[r])
            d = decide(self.model, self.mode, plain(ev.tkn[i]), T, close, None, float(px[r]) * float(f[r]))
            self._scope("fall", int(ev.dn[i]), d.kind != "first", Rr in orig)
            if d.kind == "first":
                continue
            drop.add(Rr)
            e = dict(d.event or {}, arm="fall", role="turn", tkn=str(ev.tkn[i]), row=Rr,
                     pick_day=iso_day(int(ev.dn[i])), close=close, target=None, halted_at_first=True,
                     traded_without_halts=Rr in orig)
            self.events.append(e)
            if d.kind == "lost":
                e.update(outcome=d.why, at_k=d.k, at=iso_et(d.at), price=d.price)
                self._count(f"fall_{d.why}")
                continue
            e.update(at_k=d.k, at=iso_et(d.at), price=d.price, scale=d.scale)
            ok = self._lend_turn(Rr, int(d.at), float(d.price))
            if not ok:
                e["outcome"] = "unlendable"
                self._count("fall_unlendable")
                continue
            it = self._later_item(ev, i, int(eff[r]), bool(marked[r]), float(px[r]), float(lvl[r]), int(d.at),
                                  float(d.price), {"turn_k": k_, "turn_W": W_, "pick_px": float(px[r] * f[r]),
                                                   "turn_bar": j,
                                                   "halt": {"k": int(d.k), "role": "wait" if d.why == "wait" else "turn",
                                                            "mode": self.mode}}, pick_bar=j)
            if it is None:
                e["outcome"] = "past_deadline"
                self._count("fall_past_deadline")
                continue
            e["outcome"] = "entered"
            self._count("fall_entered")
            items.append(it)
        if not drop:
            return pieces
        out_keep = [p for p in pieces if int(p["R"]) not in drop]
        add = build_vol_pieces(eng, ev, cfg, prm, items, "fall")
        return out_keep + sorted(add, key=lambda p: int(p["ens"]))

    def _lend_turn(self, row: int, at: int, price: float) -> bool:
        if self.lend_at is None:
            return bool(self.L[row])
        return bool(self.lend_at(int(row), int(at), float(price)))

    def _validate_fall(self, ev, cfg, prm, idx, eff, marked, px, lvl, jt, rows, orig: Dict[int, dict]) -> None:
        st = self.eng.st
        items, want = [], []
        for r in rows:
            Rr = int(ev.r[idx[r]])
            if Rr not in orig:
                continue
            i, j = int(idx[r]), int(jt[r])
            items.append(Item(i, int(eff[r]), bool(marked[r]), float(px[r]), float(lvl[r]), int(st.PNS[Rr, j]),
                              float(st.PC[Rr, j]), j + 1, int(st.PSD[Rr, j]), float(st.PA[Rr, j]), int(st.PBD[Rr, j])))
            want.append(orig[Rr])
            if len(items) >= self.VALIDATE:
                break
        for q, w in zip(build_vol_pieces(self.eng, ev, cfg, prm, items, "fall"), want):
            bad = _same(q, w)
            if bad:
                raise RuntimeError(f"halts: the fall piece builder does not reproduce row {q['R']} ({bad})")
