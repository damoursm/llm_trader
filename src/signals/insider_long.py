"""THE INSIDER BOOK — opportunistic insiders' open-market purchases, bought while the market is 20% under its high
(PREREG203's O1_20; the user, 2026-10-24: "Ok. Deploy the Opportunistic, SPY 20%+ down to the live combine."; from the
2026-10-26 session; runbook docs/INSIDER_LONG.md).

THE GATE (`gate_from_closes`): SPY's close at D-1 at least `insider_long_spy_drawdown` (20%) under its highest close of
the `insider_long_spy_high_sessions` (252) sessions through D-1 (at least `insider_long_spy_min_sessions`, 200, read) —
Polygon's whole-market daily bars (`sel_pressure.grouped_ohlc`'s cache; SPY never split, so its closes are the
official ones, equal to the OHLCV cache's that the study read). Off, the book does nothing that day: it is DORMANT
outside a deep sell-off (2018-2026: 76 sessions — December 2018, March-April 2020, June 2022 - January 2023).

THE EVENTS (`events_from_rows`, the study's construction, PREREG201's `events`): every Form 4 / 4/A row of an officer or
director (``is_officer`` / ``is_director``) in the deep store's insider tables — the SEC's quarterly bulk set
(``form345.parquet``) and its live tail read from EDGAR (``form345_live.parquet``, fetched through yesterday by the
08:30 pre-open run) — with code P (an OPEN-MARKET PURCHASE: acquired, a price above 0), a non-derivative title (no
option, warrant, right, note, preferred, unit, debenture or bond), the issuer's symbol well formed (BRK-B / BRK/B ->
BRK.B), a filing and a trade date, shares above 0; a transaction already reported (the same owner, issuer, code, trade
date, shares and price — an amendment repeating its original) counted once, at its first report in the files' order.
An EVENT is one company's purchases filed on one date; it is OPPORTUNISTIC when at least one of its rows is an
opportunistic insider's and none a routine insider's — Cohen, Malloy & Pomorski's classes, point in time: an insider
who traded (bought or sold, any company) in each of the three calendar years before the purchase's trade year, by the
filings made before its own; ROUTINE when one of those trades fell in the purchase's month in all three years,
opportunistic otherwise; an insider without trades in all three years is neither.

THE RULE (`compute_signals`): on a gate session D, every opportunistic event filed from the session before D through
D's eve — its company's first session after the filing is D — whose stock (the whole-market bars as traded, the run of
its bars since its last gap of more than 30 days) closed at D-1 at `insider_long_min_price` ($5) or more, traded
`insider_long_min_dollar_volume` ($5M) a day (the mean close x volume of its last 20 bars) and holds
`insider_long_min_history` (61) bars is BOUGHT at D's open, the largest purchase (the event's summed shares x price)
first; SOLD at the open after `insider_long_hold_sessions` (60) sessions held (``ins_time``, `exit_check`). Every input is
read through D's eve (the filings filed before D, the bars through D-1; `tests/test_no_lookahead.py`).

THE ACCOUNT (`src/performance/live_account.py`): each long 1/`insider_long_account_slices` (10) of the live account's
equity within the long side's cap and the margin room at Reg T's 50%, at most 1% of its dollar volume, at most
`insider_long_max_open` (5) open (`size_insider`); its capital GIVES WAY FIRST — the backtest's quality 0.05, under the
capitulation book's 0.1: a short, the crash book's buy or a dip or capitulation fill short of room sells its latest
longs whole before any other long (``ins_gave_way``); de-leveraging sells them with the capitulation book's, the latest
first (``ins_delever``); a margin call liquidates them like any position (``ins_margin_call``). One book per name; a
second event of a name the book holds opens another long (the study's account stacked them).

Evidence (`memory/insider-books-account-prereg203-2026-10.md`, `memory/capital-place-prereg204-2026-10.md`; PREREG201-205): the rule's
signals 2007-2026 640 (23 entry months), +24.0% a trade over 60 sessions, SPY +15.1% over the same days (+8.9 pp, 95%
+1.6 .. +14.0), a 3x S&P fund +45.3%; in the live account at this place +3.68 pp a year on 40 draws a set (the add rule's
ADD, 2020 and 2022 carrying it), held out +0.84 log points a year; the arm FAILS the bank's checks (too few months, one year carrying it).

Files (`insider_long_dir`, cache/ml/insider_long/): ``signals/<D>.json`` (the day's gate, its events and signals, the
rejected), ``entries/<D>.jsonl``, ``exits/<D>.jsonl``. CLI: ``python -m src.signals.insider_long --gate [--day D]`` /
``--signals [--day D] [--no-write]`` / ``--status`` / ``--parity``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.accounts import paths as _acct
from src.performance.books import INSIDER_LONG as MECHANISM      # one definition of the book's mechanism
from src.signals import dip_long as DL

ET = DL.ET
TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
BAD_TITLE = re.compile(r"OPTION|WARRANT|RIGHT|NOTE|PREFERRED|UNIT|DEBENTURE|BOND", re.I)
FORMS = ("4", "4/A")
CLASS_YEARS = 3                        # Cohen, Malloy & Pomorski: the three calendar years before the trade's
GAP_DAYS = 30                          # a gap longer than this (calendar days) starts a new run of a symbol's bars
DEDUPE = ["owner_cik", "issuer_cik", "trans_code", "td", "shares", "price"]
_COLS = ("ticker, issuer_cik, form, filing_date, trans_date, trans_code, acq_disp, shares, price, "
         "shares_owned_after, security_title, is_director, is_officer, owner_cik, owner_title, aff_10b5_1")


def enabled() -> bool:
    return bool(getattr(settings, "enable_insider_long", False))


def root() -> Path:
    return Path(settings.insider_long_dir)


def _iso(d: date) -> str:
    return d.isoformat()


def in_entry_window(now: Optional[datetime] = None) -> bool:
    """Regular hours, within `insider_long_entry_window_minutes` of the 09:30 open (the study bought at the open)."""
    now = now or datetime.now(timezone.utc)
    if not DL.in_rth(now):
        return False
    et = now.astimezone(ET)
    mins = et.hour * 60 + et.minute + et.second / 60.0 - 570.0
    return 0.0 <= mins <= float(settings.insider_long_entry_window_minutes)


def hold_sessions() -> int:
    return max(1, int(settings.insider_long_hold_sessions))


# ── THE GATE ─────────────────────────────────────────────────────────────────

def gate_from_closes(closes: pd.Series, through: date) -> dict:
    """The market gate at the close of ``through`` from ``closes`` (a date-indexed series; only the closes on or before
    ``through`` are read): ``drawdown`` = 1 - close / the highest close of the last `insider_long_spy_high_sessions`
    (at least `insider_long_spy_min_sessions` of them), ``on`` when at least `insider_long_spy_drawdown`; ``error``
    when ``through`` has no close or too few sessions precede it."""
    n_hi = max(1, int(settings.insider_long_spy_high_sessions))
    n_min = max(1, int(settings.insider_long_spy_min_sessions))
    th = float(settings.insider_long_spy_drawdown)
    out = {"symbol": str(settings.insider_long_gate_symbol), "session": _iso(through), "threshold": th, "on": False,
           "drawdown": None, "close": None, "high": None, "high_session": None, "sessions": 0, "error": None}
    s = pd.Series(closes, dtype=float).dropna()
    s = s[pd.DatetimeIndex(s.index).normalize() <= pd.Timestamp(through)]
    s = s[s > 0]
    if s.empty or pd.Timestamp(s.index[-1]).normalize() != pd.Timestamp(through):
        out["error"] = f"no {out['symbol']} close for {through}"
        return out
    w = s.iloc[-n_hi:]
    out["sessions"] = int(len(w))
    if len(w) < n_min:
        out["error"] = f"only {len(w)} {out['symbol']} closes through {through} (at least {n_min} needed)"
        return out
    k = int(np.argmax(w.to_numpy(float)))
    hi = float(w.iloc[k])
    c = float(w.iloc[-1])
    dd = 1.0 - c / hi
    out.update(close=c, high=hi, high_session=_iso(pd.Timestamp(w.index[k]).date()), drawdown=dd,
               on=bool(dd >= th))
    return out


def _frames(d: date, n: int, fetch: bool = True) -> Tuple[List[Tuple[date, pd.DataFrame]], List[str]]:
    """The whole-market daily bars of the ``n`` sessions strictly before ``d`` (oldest first) and the sessions missing
    — the capitulation book's reader (`cap_long._session_frame`: a session the cache marked empty is asked again)."""
    from src.signals import cap_long
    from src.signals.sel_short import sessions_before
    out, missing = [], []
    for s in sessions_before(d, n):
        g = cap_long._session_frame(s, fetch=fetch)
        if g is None:
            missing.append(_iso(s))
            continue
        out.append((s, g))
    return out, missing


def spy_closes(frames: Sequence[Tuple[date, pd.DataFrame]]) -> pd.Series:
    """The gate symbol's closes in ``frames``."""
    sym = str(settings.insider_long_gate_symbol)
    vals = {}
    for s, g in frames:
        r = g.loc[g["sym"] == sym, "c"]
        if len(r) and np.isfinite(float(r.iloc[0])):
            vals[pd.Timestamp(s)] = float(r.iloc[0])
    return pd.Series(vals, dtype=float).sort_index()


def gate_at(d: date, fetch: bool = True, frames=None) -> dict:
    """The gate deciding session ``d`` — at the close of D-1 — from the whole-market bars (``frames``: the sessions
    before ``d``, read when not given)."""
    prev = DL.prev_session(d)
    missing: List[str] = []
    if frames is None:
        frames, missing = _frames(d, max(1, int(settings.insider_long_spy_high_sessions)), fetch=fetch)
    out = gate_from_closes(spy_closes(frames), prev)
    if missing:                                     # a session missing could hide the high: never a guess
        out["missing_sessions"] = missing[:10]
        out["error"] = out.get("error") or f"no whole-market bars for {len(missing)} session(s): {', '.join(missing[:5])}"
    return out


# ── THE FILINGS ──────────────────────────────────────────────────────────────

def norm_ticker(t) -> str:
    """The issuer's symbol as the study reads it: upper case, a class suffix after '/' or '-' as '.X', at most five
    letters (else "")."""
    s = str(t or "").strip().upper()
    s = re.sub(r"[/-]([A-Z])$", r".\1", s)
    return s if TICKER_RE.match(s) else ""


def project_ticker(sym: str) -> str:
    """The project's form of a symbol (BRK.B -> BRK-B: the ledger's, the whole-market bars')."""
    return sym.replace(".", "-")


def _deep() -> Path:
    from src.data.deep import DEEP_DIR
    return Path(DEEP_DIR)


def _files() -> List[Tuple[int, Path]]:
    return [(0, _deep() / "form345.parquet"), (1, _deep() / "form345_live.parquet")]


def _query(where: str, args: Sequence = ()) -> pd.DataFrame:
    """The officer / director P and S rows of the two tables matching ``where`` (SQL over their columns), in the files'
    order (the bulk table first, each in its row order — the study's union)."""
    import duckdb
    parts, params = [], []
    for src, f in _files():
        if not f.exists():
            continue
        parts.append(f"select {src} as src, file_row_number as rn, {_COLS} from read_parquet('{f.as_posix()}', "
                     f"file_row_number=true) where trans_code in ('P', 'S') and (is_officer or is_director)"
                     + (f" and ({where})" if where else ""))
        params += list(args)
    if not parts:
        return pd.DataFrame(columns=["src", "rn"] + [c.strip() for c in _COLS.split(",")])
    con = duckdb.connect()
    try:
        return con.execute(" union all ".join(parts) + " order by src, rn", params).df()
    finally:
        con.close()


def clean_rows(R: pd.DataFrame) -> pd.DataFrame:
    """The study's filters on raw rows (in the files' order, never re-sorted): a well-formed symbol (``sym``), forms 4
    and 4/A, a non-derivative title, a filing date (``fd``) and a trade date (``td``), shares above 0; ``plan`` the
    10b5-1 flag. No deduplication (`dedupe`)."""
    R = R.copy()
    R["sym"] = R["ticker"].map(norm_ticker)
    R = R[(R["sym"] != "") & R["form"].isin(list(FORMS))]
    R = R[~R["security_title"].fillna("").str.contains(BAD_TITLE)]
    R["fd"] = pd.to_datetime(R["filing_date"], errors="coerce").dt.normalize()
    R["td"] = pd.to_datetime(R["trans_date"], errors="coerce").dt.normalize()
    R = R[R["fd"].notna() & R["td"].notna() & (pd.to_numeric(R["shares"], errors="coerce") > 0)]
    R["shares"] = pd.to_numeric(R["shares"], errors="coerce").astype(float)
    R["price"] = pd.to_numeric(R["price"], errors="coerce").astype(float)
    R["plan"] = R["aff_10b5_1"].astype(str).str.lower().isin(["1", "true"])
    return R


def dedupe(R: pd.DataFrame) -> pd.DataFrame:
    """A transaction reported twice (an amendment repeating its original) counted once — its first report in the files'
    order."""
    return R.drop_duplicates(DEDUPE)


def load_rows(owners: Optional[Sequence[str]] = None, filed_before: Optional[date] = None) -> pd.DataFrame:
    """The insiders' cleaned rows (`clean_rows`, not deduplicated), only ``owners``' when given and only those filed
    before ``filed_before`` when given (the point-in-time cut)."""
    where, args = [], []
    if owners is not None:
        owners = sorted({str(o) for o in owners if o is not None and str(o)})
        if not owners:
            return clean_rows(_query("false"))
        where.append("owner_cik in (" + ", ".join("?" for _ in owners) + ")")
        args += owners
    if filed_before is not None:
        where.append("filing_date < ?")
        args.append(_iso(filed_before))
    return clean_rows(_query(" and ".join(where), args))


def window_owners(lo: date, hi: date) -> List[str]:
    """The officers / directors with an open-market purchase filed in [``lo``, ``hi``]."""
    R = _query("trans_code = 'P' and filing_date >= ? and filing_date <= ?", [_iso(lo), _iso(hi)])
    return sorted({str(o) for o in R["owner_cik"].dropna()})


def klass(hist_fd: np.ndarray, hist_td: np.ndarray, td, fd) -> str:
    """Cohen, Malloy & Pomorski's class of an insider's purchase traded ``td``, filed ``fd`` — from the insider's trades
    ``hist_td`` filed ``hist_fd`` (sorted by filing date): only those filed BEFORE ``fd`` count. "opportunistic", "routine"
    or "na" (no trade in one of the three calendar years before the trade's)."""
    k = int(np.searchsorted(hist_fd, np.datetime64(pd.Timestamp(fd)), side="left"))
    if k <= 0:
        return "na"
    t = pd.DatetimeIndex(hist_td[:k])
    years = set(t.year.tolist())
    y, m = pd.Timestamp(td).year, pd.Timestamp(td).month
    need = tuple(y - j for j in range(1, CLASS_YEARS + 1))
    if not all(x in years for x in need):
        return "na"
    months: Dict[int, set] = {}
    for yy, mm in zip(t.year.tolist(), t.month.tolist()):
        months.setdefault(yy, set()).add(mm)
    return "routine" if all(m in months.get(x, set()) for x in need) else "opportunistic"


def events_from_rows(R: pd.DataFrame, lo: date, hi: date, cut: date) -> pd.DataFrame:
    """The purchase EVENTS filed in [``lo``, ``hi``] from cleaned rows ``R`` (`clean_rows`, in the files' order; at least
    every row of the window's insiders): the rows filed before ``cut`` kept, deduplicated, the window's purchases (code
    P, acquired, a price above 0) classified (`klass`) and grouped by (symbol, filing date): ``notional`` (shares x price
    summed), ``buyers`` (distinct insiders), ``issuer``, ``td`` (the first trade date), ``n_opp`` / ``n_rout`` (their
    opportunistic / routine rows), ``opp`` (at least one opportunistic, no routine)."""
    cols = ["sym", "fd", "issuer", "notional", "buyers", "td", "n_opp", "n_rout", "opp"]
    lo_t, hi_t, cut_t = pd.Timestamp(lo), pd.Timestamp(hi), pd.Timestamp(cut)
    win = R[(R["fd"] >= lo_t) & (R["fd"] <= hi_t) & (R["trans_code"] == "P")]
    if win.empty:
        return pd.DataFrame(columns=cols)
    owners = set(win["owner_cik"].astype(str))
    Q = R[R["owner_cik"].astype(str).isin(owners) & (R["fd"] < cut_t)]
    Q = dedupe(Q)
    P = Q[(Q["trans_code"] == "P") & (Q["acq_disp"] == "A") & (Q["price"] > 0)
          & (Q["fd"] >= lo_t) & (Q["fd"] <= hi_t)].copy()
    if P.empty:
        return pd.DataFrame(columns=cols)
    hist: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for o, g in Q.groupby(Q["owner_cik"].astype(str), sort=False):
        order = np.argsort(g["fd"].to_numpy("datetime64[ns]"), kind="stable")
        hist[o] = (g["fd"].to_numpy("datetime64[ns]")[order], g["td"].to_numpy("datetime64[ns]")[order])
    P["cls"] = [klass(*hist[str(o)], td, fd) for o, td, fd in zip(P["owner_cik"], P["td"], P["fd"])]
    P["notional"] = P["shares"] * P["price"]
    E = P.groupby(["sym", "fd"]).agg(issuer=("issuer_cik", "first"), notional=("notional", "sum"),
                                     buyers=("owner_cik", "nunique"), td=("td", "min"),
                                     n_opp=("cls", lambda s: int((s == "opportunistic").sum())),
                                     n_rout=("cls", lambda s: int((s == "routine").sum()))).reset_index()
    E["opp"] = (E["n_opp"] >= 1) & (E["n_rout"] == 0)
    return E[cols]


def filing_window(d: date) -> Tuple[date, date]:
    """The filing dates whose company's first session after the filing is ``d``: from the session before ``d`` through
    ``d``'s eve."""
    return DL.prev_session(d), d - timedelta(days=1)


def _federal_holiday(x: date) -> bool:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    return bool(len(USFederalHolidayCalendar().holidays(start=pd.Timestamp(x), end=pd.Timestamp(x))))


def filings_fresh(lo: date, hi: date) -> Tuple[bool, List[str]]:
    """Does the store hold every filing day of [``lo``, ``hi``]? Each weekday is covered by the bulk table (on or before
    its last filing date) or by the live tail's rows of that date; a weekday without rows is EDGAR's holiday (no index)
    when a later day's rows exist (the refresh asked for it and found none) or when it is a federal holiday. Returns
    (fresh, the days missing)."""
    import duckdb
    bulk, live = _files()[0][1], _files()[1][1]
    con = duckdb.connect()
    try:
        seam = (con.execute(f"select max(filing_date) from '{bulk.as_posix()}'").fetchone()[0]
                if bulk.exists() else None)
        have = ({r[0] for r in con.execute(f"select distinct filing_date from '{live.as_posix()}' where filing_date >= ?",
                                           [_iso(lo)]).fetchall()} if live.exists() else set())
    finally:
        con.close()
    last = max(have) if have else None
    missing = []
    x = lo
    while x <= hi:
        if x.weekday() < 5 and not (seam and _iso(x) <= str(seam)[:10]) and _iso(x) not in have:
            if not ((last and str(last)[:10] > _iso(x)) or _federal_holiday(x)):
                missing.append(_iso(x))
        x += timedelta(days=1)
    return not missing, missing


# ── THE BARS ─────────────────────────────────────────────────────────────────

def symbol_bars(frames: Sequence[Tuple[date, pd.DataFrame]], tickers: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Each ticker's (project form) daily bars in ``frames``: a date-indexed o h l c v frame, its sessions only."""
    want = set(tickers)
    rows: Dict[str, List[tuple]] = {t: [] for t in want}
    for s, g in frames:
        sub = g[g["sym"].isin(want)]
        for r in sub.itertuples(index=False):
            rows[r.sym].append((pd.Timestamp(s), float(r.o), float(r.h), float(r.l), float(r.c), r.v))
    out = {}
    for t, rr in rows.items():
        if rr:
            df = pd.DataFrame(rr, columns=["date", "o", "h", "l", "c", "v"]).set_index("date").sort_index()
            out[t] = df
    return out


def bar_stats(bars: Optional[pd.DataFrame], d: date) -> Optional[dict]:
    """A stock's numbers before session ``d`` from its daily bars as traded (the study's `_pre` and `_run_at`): the
    ``bars`` of its current run (consecutive bars at most 30 calendar days apart) before ``d`` — 0 when its last bar is
    more than 30 days before ``d`` (a bar on ``d`` would start a new run) —, its ``close`` at its last bar
    (``session``), ``dv20`` the mean close x volume of its last 20 bars. None without a bar before ``d``."""
    if bars is None or bars.empty:
        return None
    b = bars[(pd.DatetimeIndex(bars.index) < pd.Timestamp(d)) & np.isfinite(bars["c"].to_numpy(float))
             & (bars["c"].to_numpy(float) > 0) & (bars["o"].to_numpy(float) > 0) & bars["v"].notna().to_numpy()]
    if b.empty:
        return None
    days = pd.DatetimeIndex(b.index).to_numpy("datetime64[D]")
    gaps = np.flatnonzero(np.diff(days).astype(int) > GAP_DAYS)
    start = int(gaps[-1] + 1) if len(gaps) else 0
    run = b.iloc[start:]
    c, v = run["c"].to_numpy(float), run["v"].to_numpy(float)
    k = min(20, len(run))
    last = pd.Timestamp(run.index[-1])
    n = int(len(run)) if (pd.Timestamp(d) - last).days <= GAP_DAYS else 0
    return {"session": _iso(last.date()), "close": float(c[-1]), "dv20": float(np.mean(c[-k:] * v[-k:])), "bars": n}


def qualifies(ev: dict, st: Optional[dict]) -> Tuple[bool, str]:
    """The rule on an event and its stock's numbers (the gate apart): (passes, why not)."""
    if not ev.get("opp"):
        return False, "routine" if int(ev.get("n_rout") or 0) > 0 else "not_opportunistic"
    if st is None:
        return False, "no_bars"
    if st["bars"] < int(settings.insider_long_min_history):
        return False, "history"
    if not (math.isfinite(st["close"]) and st["close"] >= float(settings.insider_long_min_price)):
        return False, "price"
    if not (math.isfinite(st["dv20"]) and st["dv20"] >= float(settings.insider_long_min_dollar_volume)):
        return False, "liquidity"
    return True, "ok"


def signal_key(s: dict) -> tuple:
    """The signals' order: the largest purchase first (the study's ``rank`` = -notional)."""
    return (-float(s.get("notional") or 0.0), str(s.get("ticker")), str(s.get("fd")))


def decide(d: date, gate: dict, events: pd.DataFrame, bars: Dict[str, pd.DataFrame]) -> Tuple[List[dict], dict]:
    """The day's signals from its gate, its events and their stocks' bars: (the signals in rank order, the rejected
    tickers by reason)."""
    sig, rej = [], {}
    if not gate.get("on"):
        return [], {}
    for ev in events.to_dict("records"):
        tk = project_ticker(str(ev["sym"]))
        st = bar_stats(bars.get(tk), d)
        ok, why = qualifies(ev, st)
        if not ok:
            rej.setdefault(why, []).append(tk)
            continue
        sig.append({"ticker": tk, "sym": str(ev["sym"]), "fd": _iso(pd.Timestamp(ev["fd"]).date()),
                    "issuer": ev.get("issuer"), "notional": float(ev["notional"]), "buyers": int(ev["buyers"]),
                    "n_opp": int(ev["n_opp"]), "n_rout": int(ev["n_rout"]),
                    "td": _iso(pd.Timestamp(ev["td"]).date()), **st})
    sig.sort(key=signal_key)
    for k, s in enumerate(sig, 1):
        s["rank"] = k
    return sig, rej


# ── the day's decision ───────────────────────────────────────────────────────

def signals_path(d: date) -> Path:
    return root() / "signals" / f"{_iso(d)}.json"


def compute_signals(d: date, write: bool = True, fetch: bool = True) -> Optional[dict]:
    """Session ``d``'s gate and signals (module docstring), written to ``signals/<d>.json``; None (nothing written:
    retried) while the gate cannot be read or, the gate on, the store lacks a filing day of the window."""
    _acct.decide_here(f"the insider signals of {d}")       # an extra account reads them, never computes them
    t0 = datetime.now(timezone.utc)
    prev = DL.prev_session(d)
    frames, missing = _frames(d, max(1, int(settings.insider_long_spy_high_sessions)), fetch=fetch)
    gate = gate_at(d, frames=frames)
    if missing:                                     # a session missing could hide the high: never a guess
        gate["missing_sessions"] = missing[:10]
        gate["error"] = gate.get("error") or f"no whole-market bars for {len(missing)} session(s): {', '.join(missing[:5])}"
    out = {"day": _iso(d), "signal_session": _iso(prev), "gate": gate,
           "rule": {"spy_drawdown": float(settings.insider_long_spy_drawdown),
                    "high_sessions": int(settings.insider_long_spy_high_sessions),
                    "min_price": float(settings.insider_long_min_price),
                    "min_dollar_volume": float(settings.insider_long_min_dollar_volume),
                    "min_history": int(settings.insider_long_min_history), "hold_sessions": hold_sessions()},
           "signals": [], "computed_at": t0.isoformat(timespec="seconds")}
    if gate.get("error"):
        logger.warning(f"[insider_long] gate for {d} not decided: {gate['error']}")
        return None
    if gate["on"]:
        lo, hi = filing_window(d)
        fresh, gone = filings_fresh(lo, hi)
        out["filings"] = {"window": [_iso(lo), _iso(hi)], "fresh": fresh, "missing": gone}
        if not fresh:
            logger.warning(f"[insider_long] {d}: the gate is ON but the insider store lacks the filings of {gone} — "
                           f"retried next tick (the pre-open run fetches them)")
            return None
        owners = window_owners(lo, hi)
        R = load_rows(owners, filed_before=d) if owners else clean_rows(_query("false"))
        E = events_from_rows(R, lo, hi, d)
        bars = symbol_bars(frames, [project_ticker(s) for s in E["sym"]]) if len(E) else {}
        sig, rej = decide(d, gate, E, bars)
        out.update(events=int(len(E)), opportunistic=int(E["opp"].sum()) if len(E) else 0, signals=sig,
                   rejected={k: sorted(v) for k, v in rej.items()})
    out["seconds"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    if write:
        DL._write_json(signals_path(d), out)
    logger.info(f"[insider_long] {d}: SPY {float(gate['drawdown']):.1%} under its {gate['sessions']}-session high at "
                f"the {prev} close — "
                + (f"the gate is ON: {out.get('events', 0)} purchase event(s), {len(out['signals'])} signal(s)"
                   + (": " + ", ".join(f"{s['ticker']} ${s['notional'] / 1e3:,.0f}k" for s in out["signals"][:8])
                      if out["signals"] else "")
                   if gate["on"] else f"under the {float(settings.insider_long_spy_drawdown):.0%} gate, no buy"))
    return out


def signals_for(d: date) -> Optional[dict]:
    """The day's gate and signals — computed once a day, then read from ``signals/<d>.json``."""
    p = signals_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                                      # noqa: BLE001
            logger.warning(f"[insider_long] {p} unreadable ({e}) — recomputing")
    return compute_signals(d)


def prepare(now: Optional[datetime] = None) -> Optional[dict]:
    """Before the open (a pre-market tick from 09:00 ET, after the pre-open run's filings): the day's decision
    computed and kept, so the opening tick reads it. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        d = DL.today_et(now)
        if not enabled() or not DL.is_session(d) or not _acct.is_main():
            return None
        et = now.astimezone(ET)
        if not (9 * 60 <= et.hour * 60 + et.minute < 570):
            return None
        return signals_for(d)
    except Exception as e:                                          # noqa: BLE001
        logger.warning(f"[insider_long] preparation failed ({type(e).__name__}: {e})")
        return None


# ── exits ────────────────────────────────────────────────────────────────────

def entry_day(trade: dict) -> Optional[date]:
    raw = trade.get("ins_entry_day") or str(trade.get("entry_date") or "")[:10]
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def exit_check(trade: dict, d: date) -> Tuple[Optional[str], dict]:
    """Does session ``d``'s open close this long? ``ins_time`` once it has held `insider_long_hold_sessions` sessions
    through D-1 (its entry session counted); else None (``info``: the sessions held)."""
    e = entry_day(trade)
    prev = DL.prev_session(d)
    info: dict = {"through": _iso(prev)}
    if e is None or prev < e:
        return None, info
    held = DL.sessions_held(e, prev)
    info["held"] = held
    if held >= hold_sessions():
        return "ins_time", info
    return None, info


# ── journals ─────────────────────────────────────────────────────────────────

def entries_path(d: date) -> Path:
    return _acct.state(root() / "entries" / f"{_iso(d)}.jsonl")


def exits_path(d: date) -> Path:
    return _acct.state(root() / "exits" / f"{_iso(d)}.jsonl")


def journal_entry(d: date, sig: dict, outcome: str, **kw) -> None:
    DL._append(entries_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "ticker": sig.get("ticker"), "outcome": outcome,
                                 **{k: sig.get(k) for k in ("fd", "notional", "buyers", "n_opp", "close", "dv20")},
                                 **kw})


def journal_exit(d: date, trade: dict, reason: str, info: dict) -> None:
    DL._append(exits_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "ticker": trade.get("ticker"), "reason": reason,
                               "trade": trade.get("recommendation_id"), **info})


def settled(d: date) -> set:
    """The signals (``ticker|fd``) the entry step settled for good on ``d`` (every outcome but a missing price)."""
    out = set()
    for r in DL._read_jsonl(entries_path(d)):
        if r.get("outcome") != "no_price":
            out.add(f"{r.get('ticker')}|{r.get('fd')}")
    return out


def trade_id(d: date, ticker: str, fd: str) -> str:
    return hashlib.sha1(f"ins|{_iso(d)}|{ticker}|{fd}".encode("utf-8")).hexdigest()[:16]


# ── health (the email's scorer banner) ───────────────────────────────────────

def health(now: Optional[datetime] = None, trades: Optional[Sequence[dict]] = None) -> dict:
    """``problems``: on a session after the entry window, no decision for today (the gate could not be read, or the gate
    on and the filings missing); an open long held past its limit. ``notes``: the gate on, the day's signals."""
    now = now or datetime.now(timezone.utc)
    out: dict = {"problems": [], "notes": []}
    if not enabled():
        return out
    d = DL.today_et(now)
    et = now.astimezone(ET)
    if DL.is_session(d) and DL.in_rth(now):
        p = signals_path(d)
        day = None
        if p.exists():
            try:
                day = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                day = None
        if day is None:
            if et.hour * 60 + et.minute > 570 + float(settings.insider_long_entry_window_minutes):
                out["problems"].append(f"insider_long: no decision for {d} (SPY's bars could not be read, or the gate "
                                       f"is on and the insider store lacks the window's filings)")
        elif (day.get("gate") or {}).get("on"):
            g = day["gate"]
            out["notes"].append(f"insider_long: the gate is ON — SPY {float(g['drawdown']):.0%} under its high at the "
                                f"{day.get('signal_session')} close, {len(day.get('signals') or [])} signal(s)")
    lim = hold_sessions()
    for t in trades or []:
        if t.get("status") != "OPEN" or t.get("entry_mechanism") != MECHANISM:
            continue
        e = entry_day(t)
        if e is not None and DL.sessions_held(e, d) > lim + 2:
            out["problems"].append(f"insider_long: {t.get('ticker')} held past its {lim}-session limit (since {e})")
    return out


# ── parity with the study ────────────────────────────────────────────────────

def research_pieces() -> Optional[List[dict]]:
    """PREREG203's O1_20 pieces (`studies/prereg203/data/pieces203.pkl`), None when the research repository is absent."""
    import pickle
    p = Path(settings.backtest_research_dir) / "studies" / "prereg203" / "data" / "pieces203.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return list(pickle.load(fh)["pieces"].get("O1_20") or [])


def parity(lo: str = "2018-01-02", hi: str = "2026-09-24", fetch: bool = True) -> dict:
    """Production's decisions against PREREG203's O1_20 entries on the gate sessions of [``lo``, ``hi``]: each session's
    signals (ticker, filing date) computed by `compute_signals` (nothing written) beside the study's pieces entering that
    session."""
    R = research_pieces()
    if R is None:
        return {"error": "no research pieces"}
    want: Dict[str, set] = {}
    for p in R:
        day = pd.Timestamp(int(p["ens"]), unit="ns", tz="UTC").tz_convert(ET).date()
        fd = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(p["pick_day"]))).date()
        want.setdefault(_iso(day), set()).add((project_ticker(str(p["tkn"])), _iso(fd)))
    from src.backtest import reliability as REL
    closes = REL.spy_daily()["Close"].astype(float)
    rows = []
    d = pd.Timestamp(lo).date()
    end = pd.Timestamp(hi).date()
    while d <= end:
        if DL.is_session(d):
            g = gate_from_closes(closes, DL.prev_session(d))
            if g.get("on") or _iso(d) in want:
                got = compute_signals(d, write=False, fetch=fetch)
                have = {(s["ticker"], s["fd"]) for s in (got or {}).get("signals") or []}
                w = want.get(_iso(d), set())
                rows.append({"day": _iso(d), "gate": (got or {}).get("gate", {}).get("on"), "study": len(w),
                             "production": len(have), "only_study": sorted(w - have),
                             "only_production": sorted(have - w)})
        d += timedelta(days=1)
    diff = [r for r in rows if r["only_study"] or r["only_production"]]
    return {"sessions": len(rows), "study": sum(r["study"] for r in rows),
            "production": sum(r["production"] for r in rows), "equal_sessions": len(rows) - len(diff),
            "differ": diff}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="The insider book (src/signals/insider_long.py)")
    ap.add_argument("--gate", action="store_true", help="the gate deciding --day (at the close before it)")
    ap.add_argument("--signals", action="store_true", help="the day's gate and signals")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--parity", action="store_true", help="production's decisions vs PREREG203's O1_20")
    ap.add_argument("--day", help="YYYY-MM-DD (default today)")
    ap.add_argument("--lo", default="2018-01-02")
    ap.add_argument("--hi", default="2026-09-24")
    a = ap.parse_args(argv)
    d = date.fromisoformat(a.day) if a.day else DL.today_et()
    if a.gate:
        print(json.dumps(gate_at(d), indent=1, default=str))
    if a.signals:
        print(json.dumps(compute_signals(d, write=not a.no_write), indent=1, default=str)[:6000])
    if a.parity:
        print(json.dumps(parity(a.lo, a.hi), indent=1, default=str)[:20000])
    if a.status:
        print(json.dumps(health(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
