"""THE CAPITULATION BOOK — the big stocks' dips under their own trend, bought together on a market-wide capitulation
day (PREREG165's CBg6_20; user 2026-10-20: "the last lead, buy large stocks together on market-wide capitulation
days, is a good lead even if one year carries it because it's a buy and we expect most years to be good for buying.
Plus, it lifts the weak months", then "Deploy all three."; from the 2026-10-21 session; runbook docs/CAP_LONG.md).

THE BREADTH (`breadth_at`): at the close of session S, among the common stocks / ADRs (Polygon type CS / ADRC, the
type table `cache/security_types.json`) in Polygon's whole-market daily bars, each name's run of bars (cut at a gap of
more than 30 calendar days: a reused ticker, a long halt) with at least `MIN_HISTORY` (220) bars through S, the
`cap_long_breadth_top` (100) with the highest 20-session mean close x volume; B = the share of them that closed UNDER
their own 200-session average with a 2-session RSI (the dip book's `rsi2`) under 10. The bars are Polygon's
UNADJUSTED ones (`sel_pressure.grouped_ohlc`'s cache, `BREADTH_SESSIONS` sessions through S), the closes brought into
S's terms by the effective split set (`intraday_store.split_rows`); close x volume needs no adjustment. The research's
construction is `llm_trader_research/studies/prereg165/cap165.breadth` (Polygon's adjusted bars of 2005-2026):
`python -m src.signals.cap_long --parity` compares the two.

THE RULE (`compute_signals`): on session D, when B(D-1) >= `cap_long_breadth_min` (0.20) — a CAPITULATION session —
every common stock / ADR that at the close of D-1 traded >= `cap_long_min_dollar_volume` ($2B) a day (the 20-session
mean of close x volume), closed UNDER its `cap_long_trend_sessions` (200) average with a 2-session RSI under
`cap_long_rsi_max` (10), on 220 daily bars or more — the dip book's construction from the deep 30-minute store, each
name from its own security (`dip_long._daily`) — becomes a BUY LIMIT ORDER at that close minus `cap_long_limit_atr`
(1/2) x its 14-session ATR, floored to the cent, alive sessions D .. D+`cap_long_limit_sessions`-1: THE DIP BOOK'S
ORDER (placed at each window session's opening tick, filled at the session's open when it opened at/under the limit,
else at the end of the first completed 30-minute regular-hours bar whose low reaches it, at most what the room allows
then; lapsed after its last session's 16:00). Every signal is an order — no top pick, no meta-model — the deepest dip
(the lowest RSI2) first when they compete. SELL at the open after the first close above its
`cap_long_exit_sma_sessions` (3) average since the fill session (its own close counts), or after
`cap_long_max_hold_sessions` (5) sessions (`exit_check`). Every input is read through the close of D-1
(`tests/test_no_lookahead.py`).

THE ACCOUNT (`src/performance/live_account.py`): each order 1/`cap_long_account_slices` (6) of the live account's
equity at its limit within the long side's cap (the longs' value at most the equity) and the margin room at Reg T's
50%, at most `cap_long_max_open` (3) open (its working orders count); its capital GIVES WAY FIRST — the backtest's
quality 0.1, under the dip book's 0.13 (`mimic.QUALITY`): a short, the crash book's buy or a dip order's fill short of
room sells its latest longs WHOLE before any dip long (``cap_gave_way``); a dip order is sized as if they were sold;
de-leveraging sells them first (``cap_delever``); a margin call closes them (``cap_margin_call``) and cancels its
working orders. One book per name: never a name another book holds, nor another book on its names.

Evidence (`memory/rpd-review-capitulation-2026-10-20.md`, PREREG165 / 167): the rule's trades 2018-2026 (pooled) 364,
+0.683 %/day (95% 0.23 .. 1.27); UNTOUCHED 2006-2017 on a survivorship-free whole-market panel 149 trades, +1.79 %/day
(1.00 .. 2.54); in the deployed configuration (RPD_IS, 20 paired draws a set) +3.38 pp a year, no added margin call,
the weak market context (below SPY's trend in high volatility) +21 -> +28% a year; its plateau fails at a 15% breadth
(+1.02 pp); with RP33 and the crash book's after-hours exit +7.19 pp (66.5 -> 73.7%), the capitulation book's own
increment +3.57 pp. 2022 carries much of it — a crisis buy: most years hold no capitulation day.

Files (`cap_long_dir`, cache/ml/cap_long/): ``breadth/<S>.json`` (the breadth at S's close: the top names, the
oversold ones), ``signals/<D>.json`` (the day's gate and signals), ``orders/<D>.json`` (each order's life, the dip
book's schema), ``entries/<D>.jsonl``, ``exits/<D>.jsonl``. CLI: ``python -m src.signals.cap_long --breadth [--day D]``
/ ``--signals [--day D]`` / ``--status`` / ``--parity [--days N]``.
"""
from __future__ import annotations

import argparse
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
from src.performance.books import CAP_LONG as MECHANISM      # one definition of the book's mechanism
from src.signals import dip_long as DL

ET = DL.ET
TYPES = ("CS", "ADRC")                 # Polygon's common stock / ADR types
MIN_HISTORY = 220                      # daily bars a name needs, in the breadth and in the rule (the study's)
GAP_DAYS = 30                          # a gap longer than this (calendar days) starts a new run of a symbol's bars
BREADTH_SESSIONS = 300                 # whole-market sessions read for the breadth through its session
FINAL_OUTCOMES = ("filled",)


def enabled() -> bool:
    return bool(getattr(settings, "enable_cap_long", False))


def root() -> Path:
    return Path(settings.cap_long_dir)


def _iso(d: date) -> str:
    return d.isoformat()


def theta() -> float:
    return float(settings.cap_long_breadth_min)


def in_entry_window(now: Optional[datetime] = None) -> bool:
    """Regular hours, within `cap_long_entry_window_minutes` of the 09:30 open (the orders are placed at the open)."""
    now = now or datetime.now(timezone.utc)
    if not DL.in_rth(now):
        return False
    et = now.astimezone(ET)
    mins = et.hour * 60 + et.minute + et.second / 60.0 - 570.0
    return 0.0 <= mins <= float(settings.cap_long_entry_window_minutes)


def _write_json(p: Path, obj) -> None:
    DL._write_json(p, obj)


# ── THE BREADTH ──────────────────────────────────────────────────────────────

def breadth_path(s: date) -> Path:
    return root() / "breadth" / f"{_iso(s)}.json"


def _session_frame(s: date, fetch: bool = True) -> Optional[pd.DataFrame]:
    """Session ``s``'s unadjusted whole-market daily bars (sym in the project's form, o h l c v) — the borrow-pressure
    model's cache (`sel_pressure.grouped_ohlc`). A session the calendar keeps but the cache marked empty (a fetch that
    failed when it was more than three days old) is asked again once."""
    from src.signals import sel_pressure as SP
    g = SP.grouped_ohlc(s, fetch=fetch)
    if (g is None or not len(g)) and fetch:
        marker = SP.root() / "grouped" / f"{_iso(s)}.empty"
        if marker.exists() and DL.is_session(s):
            try:
                marker.unlink()
            except OSError:
                pass
            g = SP.grouped_ohlc(s, fetch=True)
    return g if g is not None and len(g) else None


def _is_stock(sym: str) -> bool:
    from src.data.company_names import security_type
    return security_type(sym) in TYPES


def breadth_at(s: date, fetch: bool = True, top: Optional[int] = None) -> dict:
    """THE BREADTH at the close of session ``s`` (module docstring): ``breadth`` (None with ``error`` when the bars
    cannot be read whole), the ``top`` names with their numbers, the ``oversold`` ones, ``eligible`` (the names with
    220 bars or more through ``s``). Reads only the sessions through ``s``."""
    from src.data.intraday_store import split_rows
    from src.data.polygon_client import to_polygon_symbol
    from src.signals import sel_short
    n_top = int(top if top is not None else settings.cap_long_breadth_top)
    out: dict = {"session": _iso(s), "breadth": None, "top_n": n_top, "theta": theta(), "error": None,
                 "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    days = sel_short.sessions_before(s, BREADTH_SESSIONS - 1) + [s]
    frames, missing = [], []
    for x in days:
        g = _session_frame(x, fetch=fetch)
        if g is None:
            missing.append(_iso(x))
            continue
        frames.append(g[["sym", "o", "h", "l", "c", "v"]].assign(dn=np.int64(sel_short.dnum(x))))
    out["sessions"] = [_iso(days[0]), _iso(s)]
    if missing:
        out["error"] = f"no whole-market bars for {len(missing)} session(s): {', '.join(missing[:5])}"
        out["missing"] = missing
        return out
    G = pd.concat(frames, ignore_index=True)
    G = G[np.isfinite(G["c"].to_numpy(float)) & (G["c"].to_numpy(float) > 0) & G["v"].notna()]
    syms = pd.unique(G["sym"])
    stocks = {x for x in syms if _is_stock(str(x))}
    G = G[G["sym"].isin(stocks)].sort_values(["sym", "dn"], kind="mergesort").reset_index(drop=True)
    sym = G["sym"].to_numpy(object)
    dn = G["dn"].to_numpy(np.int64)
    c = G["c"].to_numpy(float)
    cv = c * G["v"].to_numpy(float)
    n = len(G)
    if not n:
        out["error"] = "no common stock in the whole-market bars"
        return out
    first = np.r_[True, sym[1:] != sym[:-1]]
    start = first | np.r_[True, np.diff(dn) > GAP_DAYS]           # a new run: a new symbol or a gap of 30+ days
    run = np.cumsum(start) - 1
    run_len = np.bincount(run)
    last = np.r_[sym[1:] != sym[:-1], True]
    sd = sel_short.dnum(s)
    idx = np.flatnonzero(last & (dn == sd))                        # each name's row of session s
    idx = idx[run_len[run[idx]] >= MIN_HISTORY]                    # its run holds 220 bars or more through s
    cs = np.r_[0.0, np.cumsum(cv)]
    dv20 = (cs[idx + 1] - cs[idx + 1 - 20]) / 20.0                 # the run's last 20 bars (220 >= 20)
    ok = np.isfinite(dv20)
    idx, dv20 = idx[ok], dv20[ok]
    out["eligible"] = int(len(idx))
    if len(idx) < n_top:
        out["error"] = f"only {len(idx)} names with {MIN_HISTORY} bars through {s} (fewer than {n_top})"
        return out
    order = np.argsort(-dv20, kind="mergesort")[:n_top]
    rows = split_rows()
    if rows is None:
        out["error"] = "the split data cannot be read"
        return out
    names: List[dict] = []
    for k in order:
        i = int(idx[k])
        a = i - int(run_len[run[i]]) + 1                           # the run's first row
        tk = str(sym[i])
        cc = c[a:i + 1].copy()
        # the closes in s's terms: a bar of day x times split_from / split_to of every effective split executed in
        # (x, s] (`intraday_store.split_factor_between`'s rule, vectorized over the run)
        dd = dn[a:i + 1]
        for e, sf, st in rows.get(to_polygon_symbol(tk).upper(), []):
            if e <= s:
                cc[dd < sel_short.dnum(e)] *= sf / st
        sma = float(np.mean(cc[-200:]))
        r2 = float(DL.rsi2(cc)[-1])
        low = bool(cc[-1] < sma and r2 < 10.0)
        names.append({"ticker": tk, "dv20": float(dv20[k]), "close": float(c[i]), "sma200": sma, "rsi2": r2,
                      "oversold": low, "bars": int(i - a + 1)})
    out["top"] = names
    out["oversold"] = [x["ticker"] for x in names if x["oversold"]]
    out["breadth"] = float(np.mean([x["oversold"] for x in names]))
    return out


def breadth_for(d: date, fetch: bool = True) -> dict:
    """The breadth that decides session ``d`` — at the close of D-1 — computed once and kept (``breadth/<D-1>.json``);
    a computation with an ``error`` is returned unkept (retried)."""
    s = DL.prev_session(d)
    p = breadth_path(s)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    _acct.decide_here(f"the capitulation breadth at the {s} close")     # the main account's to compute
    out = breadth_at(s, fetch=fetch)
    if out.get("error") is None:
        _write_json(p, out)
        logger.info(f"[cap_long] breadth at the {s} close: {out['breadth']:.0%} of the {out['top_n']} most-traded "
                    f"stocks oversold under their trend ({len(out['oversold'])}) — "
                    + ("a CAPITULATION session tomorrow" if out["breadth"] >= theta() else
                       f"under the {theta():.0%} gate"))
    else:
        logger.warning(f"[cap_long] breadth at the {s} close not computed: {out['error']}")
    return out


def prepare(now: Optional[datetime] = None) -> Optional[dict]:
    """Before the open (a pre-market tick of a session day): the day's breadth computed and kept, so the opening tick
    reads it. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        d = DL.today_et(now)
        if not enabled() or not DL.is_session(d) or not _acct.is_main():
            return None
        et = now.astimezone(ET)
        if not (4 * 60 <= et.hour * 60 + et.minute < 570):
            return None
        return breadth_for(d)
    except Exception as e:                                          # noqa: BLE001
        logger.warning(f"[cap_long] breadth preparation failed ({type(e).__name__}: {e})")
        return None


# ── THE RULE ─────────────────────────────────────────────────────────────────

def name_stats(ticker: str, through: date) -> Optional[dict]:
    """The rule's numbers at the close of ``through`` (`dip_long.stats_at` at the book's own averages)."""
    return DL.name_stats(ticker, through, MIN_HISTORY, trend=int(settings.cap_long_trend_sessions),
                         exit_sma=int(settings.cap_long_exit_sma_sessions))


def qualifies(st: Optional[dict]) -> bool:
    """The entry rule on a name's numbers at the close of D-1 (the breadth gate apart)."""
    if not st:
        return False
    return (math.isfinite(st["dv20"]) and st["dv20"] >= float(settings.cap_long_min_dollar_volume)
            and math.isfinite(st["sma_trend"]) and st["close"] < st["sma_trend"]
            and math.isfinite(st["rsi2"]) and st["rsi2"] < float(settings.cap_long_rsi_max))


def limit_price(close: float, atr: float) -> Optional[float]:
    return DL.limit_price(close, atr, float(settings.cap_long_limit_atr))


def signals_path(d: date) -> Path:
    return root() / "signals" / f"{_iso(d)}.json"


def compute_signals(d: date, extend: bool = True) -> Optional[dict]:
    """Session ``d``'s gate and signals (module docstring), written to ``signals/<d>.json``; None (nothing written:
    retried) while the breadth cannot be computed."""
    _acct.decide_here(f"the capitulation signals of {d}")    # an extra account reads them, never computes them
    prev = DL.prev_session(d)
    t0 = datetime.now(timezone.utc)
    br = breadth_for(d)
    if br.get("breadth") is None:
        return None
    gate = float(br["breadth"]) >= theta()
    out = {"day": _iso(d), "signal_session": _iso(prev), "breadth": br["breadth"], "theta": theta(),
           "gate": bool(gate), "oversold": br.get("oversold"),
           "rule": {"min_dollar_volume": float(settings.cap_long_min_dollar_volume),
                    "rsi_max": float(settings.cap_long_rsi_max), "trend_sessions": int(settings.cap_long_trend_sessions),
                    "limit_atr": float(settings.cap_long_limit_atr), "limit_sessions": int(settings.cap_long_limit_sessions)},
           "signals": [], "computed_at": t0.isoformat(timespec="seconds")}
    if gate:
        names, src = DL.candidates(d, min_dv=float(settings.cap_long_min_dollar_volume))
        sec = DL._security_prepare(names, prev) if DL.own_security() else None

        def evaluate(tks):
            got, late = {}, []
            for tk in tks:
                st = name_stats(tk, prev)
                if st is None:
                    continue
                if st["session"] != prev:
                    late.append(tk)
                    continue
                got[tk] = st
            return got, late
        rows, stale = evaluate(names)
        if stale and extend:
            DL._extend(stale, prev)
            more, stale = evaluate(stale)
            rows.update(more)
        if sec is not None:
            redo = DL._security_recheck([tk for tk, st in rows.items() if qualifies(st)], prev, sec)
            if redo:
                more, _ = evaluate(redo)
                for tk in redo:
                    rows.pop(tk, None)
                rows.update(more)
            DL._security_journal(sec, names)
            out["security"] = sec
        inactive: List[str] = []
        if stale:
            traded = DL._traded_on(prev)
            if traded is not None:
                inactive = [t for t in stale if t not in traded]
                stale = [t for t in stale if t in traded]
        sig = [{"ticker": tk, **st} for tk, st in rows.items() if qualifies(st)]
        for r in sig:
            r["limit"] = limit_price(r["close"], r.get("atr14"))
            if sec is not None:
                own = (sec.get("changed") or {}).get(r["ticker"]) or (sec.get("errors") or {}).get(r["ticker"])
                if own:
                    r["own_security"] = own
        sig.sort(key=lambda r: (r["rsi2"], r["ticker"]))
        out.update(signals=sig, source=src, considered=len(names), evaluated=len(rows), stale=stale,
                   inactive=inactive)
    out["seconds"] = round((datetime.now(timezone.utc) - t0).total_seconds(), 1)
    _write_json(signals_path(d), out)
    logger.info(f"[cap_long] {d}: breadth {br['breadth']:.0%} at the {prev} close "
                + (f"— a CAPITULATION session: {len(out['signals'])} signal(s)"
                   + (": " + ", ".join(f"{r['ticker']} RSI2 {r['rsi2']:.1f}" for r in out["signals"])
                      if out["signals"] else "")
                   if gate else f"— under the {theta():.0%} gate, no order"))
    return out


def signals_for(d: date) -> Optional[dict]:
    """The day's gate and signals — computed once a day, then read from ``signals/<d>.json``."""
    p = signals_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                                      # noqa: BLE001
            logger.warning(f"[cap_long] {p} unreadable ({e}) — recomputing")
    return compute_signals(d)


# ── the orders (the dip book's schema and life) ──────────────────────────────

def window(d: date) -> List[date]:
    """The sessions an order of session ``d`` lives in: ``d`` and the next `cap_long_limit_sessions` - 1."""
    from src.signals.sel_short import session_after
    n = max(1, int(settings.cap_long_limit_sessions))
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
        logger.warning(f"[cap_long] {p} unreadable ({e})")
        return []


def save_orders(d: date, orders: Sequence[dict]) -> None:
    _write_json(orders_path(d), {"day": _iso(d), "orders": list(orders)})


def order_days(today: date) -> List[date]:
    """The order days whose windows can include ``today``."""
    from src.signals.sel_short import sessions_before
    n = max(1, int(settings.cap_long_limit_sessions))
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
            "signal_session": signal_session, "rank": int(rank), "rsi2": sig.get("rsi2"), "close": sig.get("close"),
            "atr14": sig.get("atr14"), "limit": float(lim), "dv20": sig.get("dv20"), "sma_trend": sig.get("sma_trend"),
            "window": [_iso(x) for x in window(d)], "status": "pending", "qty": None, "placed_at": None,
            "placed_session": None, "ibkr_qty": 0, "ibkr": None, "ibkr_n": 0, "checked_through": None,
            "unfunded_session": None, "trade": None, "events": []}


def order_key(o: dict) -> tuple:
    """The orders' place when they compete (the backtest's: the deepest dip first — the pieces' ``rank`` is the
    signal's RSI2)."""
    r = o.get("rsi2")
    return (float(r) if r is not None and math.isfinite(float(r)) else 1e9, str(o.get("day")), str(o.get("ticker")))


order_event = DL.order_event


# ── exits ────────────────────────────────────────────────────────────────────

def entry_day(trade: dict) -> Optional[date]:
    raw = trade.get("cap_entry_day") or str(trade.get("entry_date") or "")[:10]
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
    sma = c.rolling(int(settings.cap_long_exit_sma_sessions)).mean()
    hit = d.index[(d.index >= pd.Timestamp(entry)) & (c > sma).to_numpy()]
    if len(hit) == 0:
        return None
    s = hit[0]
    return {"session": pd.Timestamp(s).date(), "close": float(c.loc[s]), "sma_exit": float(sma.loc[s])}


def exit_check(trade: dict, d: date) -> Tuple[Optional[str], dict]:
    """Does session ``d``'s open close this long? ``cap_rebound`` once a close from the fill session through D-1 is
    above its exit average, ``cap_time`` once the long has held `cap_long_max_hold_sessions` sessions through D-1;
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
        logger.warning(f"[cap_long] {tk}: no stored close for {prev} — the rebound rule is judged on the closes the "
                       f"store has")
    hit = rebound_session(daily, e, prev)
    if hit is not None:
        info.update(rebound_session=_iso(hit["session"]), close=hit["close"], sma_exit=hit["sma_exit"])
        return "cap_rebound", info
    if held >= int(settings.cap_long_max_hold_sessions):
        return "cap_time", info
    return None, info


# ── journals ─────────────────────────────────────────────────────────────────

def entries_path(d: date) -> Path:
    return _acct.state(root() / "entries" / f"{_iso(d)}.jsonl")


def exits_path(d: date) -> Path:
    return _acct.state(root() / "exits" / f"{_iso(d)}.jsonl")


def journal_entry(d: date, sig: dict, outcome: str, **kw) -> None:
    DL._append(entries_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "ticker": sig.get("ticker"), "outcome": outcome,
                                 **{k: sig.get(k) for k in ("rsi2", "close", "sma_trend", "dv20")}, **kw})


def journal_exit(d: date, trade: dict, reason: str, info: dict) -> None:
    DL._append(exits_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "ticker": trade.get("ticker"), "reason": reason,
                               "trade": trade.get("recommendation_id"), **info})


def trade_id(d: date, ticker: str) -> str:
    return hashlib.sha1(f"cap|{_iso(d)}|{ticker}".encode("utf-8")).hexdigest()[:16]


# ── health (the email's scorer banner) ───────────────────────────────────────

def health(now: Optional[datetime] = None, trades: Optional[Sequence[dict]] = None) -> dict:
    """``problems``: on a session after the entry window, no gate decided for today (the breadth could not be
    computed: the book could not look); an open long held past its limit. ``notes``: the day's breadth and gate."""
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
            if et.hour * 60 + et.minute > 570 + float(settings.cap_long_entry_window_minutes):
                out["problems"].append(f"cap_long: no breadth / gate decided for {d} (the whole-market bars or the "
                                       f"split data could not be read)")
        elif day.get("gate"):
            out["notes"].append(f"cap_long: a CAPITULATION session — breadth {float(day['breadth']):.0%} at the "
                                f"{day.get('signal_session')} close, {len(day.get('signals') or [])} order(s)")
    for t in trades or []:
        if t.get("status") != "OPEN" or t.get("entry_mechanism") != MECHANISM:
            continue
        e = entry_day(t)
        if e is not None and DL.sessions_held(e, d) > int(settings.cap_long_max_hold_sessions) + 2:
            out["problems"].append(f"cap_long: {t.get('ticker')} held past its {settings.cap_long_max_hold_sessions}-"
                                   f"session limit (since {e})")
    return out


# ── parity with the research's breadth ───────────────────────────────────────

def research_breadth() -> Optional[pd.Series]:
    """PREREG165's breadth series (`studies/prereg165/data/breadth165.pkl`), None when the research repository is
    absent."""
    p = Path(settings.backtest_research_dir) / "studies" / "prereg165" / "data" / "breadth165.pkl"
    return pd.read_pickle(p) if p.exists() else None


def parity(days: int = 60, end: Optional[date] = None) -> dict:
    """Production's breadth against the research's on its last ``days`` sessions (through ``end``)."""
    R = research_breadth()
    if R is None:
        return {"error": "no research breadth"}
    R = R.dropna()
    if end is not None:
        R = R[R.index <= pd.Timestamp(end)]
    rows = []
    for ts, b in R.iloc[-int(days):].items():
        got = breadth_at(pd.Timestamp(ts).date())
        rows.append({"session": str(pd.Timestamp(ts).date()), "research": float(b), "production": got.get("breadth"),
                     "error": got.get("error")})
    diff = [r for r in rows if r["production"] is None or abs(float(r["production"]) - r["research"]) > 1e-9]
    gate_diff = [r for r in rows if r["production"] is not None
                 and (float(r["production"]) >= theta()) != (r["research"] >= theta())]
    return {"sessions": len(rows), "equal": len(rows) - len(diff), "differ": diff, "gate_differs": gate_diff}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="The capitulation book (src/signals/cap_long.py)")
    ap.add_argument("--breadth", action="store_true", help="the breadth deciding --day (at the close before it)")
    ap.add_argument("--signals", action="store_true", help="the day's gate and signals (written)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--parity", action="store_true", help="production's breadth vs PREREG165's")
    ap.add_argument("--day", help="YYYY-MM-DD (default today)")
    ap.add_argument("--days", type=int, default=60)
    a = ap.parse_args(argv)
    d = date.fromisoformat(a.day) if a.day else DL.today_et()
    if a.breadth:
        out = breadth_at(DL.prev_session(d))
        print(json.dumps({k: v for k, v in out.items() if k != "top"}, indent=1, default=str))
    if a.signals:
        print(json.dumps(compute_signals(d), indent=1, default=str)[:4000])
    if a.parity:
        print(json.dumps(parity(a.days, d), indent=1, default=str))
    if a.status:
        print(json.dumps(health(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
