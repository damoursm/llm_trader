"""THE MEGA-CAP DIP LONG BOOK (user directive 2026-10-08: "Deploy the mega-cap dip buying strategy to live
production"; the study PREREG43, `memory/long-dip-megacaps-2026-10.md`, report
https://claude.ai/artifact/3DfPZz6KzJiW5r8znx3nVz).

The rule, the study's M1_1B (an untouched test on 2007-2020, survivorship-free on 2021-26):

* BUY at the open of session D a common stock / ADR (Polygon type CS / ADRC) that, at the close of D-1,
  traded `dip_long_min_dollar_volume` ($1B) or more a day (the 20-session mean of close x volume), closed
  above its `dip_long_trend_sessions` (200) session average, and had a 2-session RSI under
  `dip_long_rsi_max` (10) — the deepest dips (lowest RSI) first when slots run out;
* SELL at the open after the first close above its `dip_long_exit_sma_sessions` (5) session average, the
  entry session's own close included, or at the open after its `dip_long_max_hold_sessions` (10) session at
  the latest (the study's next-open exit: +0.74% / +1.06% a trade 2007-20 / 2021-26, vs +0.54 / +0.94 at
  the close).

The daily bars are the study's own construction (its `lg_build.daily_of`): the deep 30-minute store's
regular-hours bars per ET session — open = the first bar's open, high / low = max / min, close = the last
bar's close, volume = the sum — and every statistic is computed on the series through D-1, never a bar of D
(`tests/test_no_lookahead.py`). A name needs `MIN_HISTORY` (220) daily bars, as in the study.

Execution: the regular-hours ticks only. Exits first (`exit_check`: their slots free up for the same
tick), then entries within `dip_long_entry_window_minutes` of the 09:30 open — marketable limit orders
through the ordinary broker sync, a few seconds to a minute after the opening auction the study traded.
Each signal is settled once a day (`entries/<day>.jsonl`); only a missing live price is retried.

Sizing: the book's OWN simulated cash account, as in the study's account (`lg_account.simulate`):
`dip_long_account_initial` ($10,000, no deposits) plus the dollar P&L of its trades (shares x entry x the
ledger's net return); each new long = floor(min(equity / slices, cash) / price) whole shares with the
study's spread-and-commission margin, at most `dip_long_account_slices` (10) open. Never a name another
open trade holds (and the short book never stacks on a dip long: `record_sel_short_trades`).

Journals under `dip_long_dir` (cache/ml/dip_long/): ``signals/<day>.json`` (the day's computation: every
qualifying name with its numbers), ``entries/<day>.jsonl`` (each signal's outcome at the entry step) and
``exits/<day>.jsonl`` (each exit with the close and average that fired it).
CLI: ``python -m src.signals.dip_long --signals [--day D]`` / ``--status``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.performance.books import DIP_LONG as MECHANISM      # one definition of the book's mechanism
from src.utils import ET

TYPES = ("CS", "ADRC")               # Polygon's common stock / ADR types — the study's universe
MIN_HISTORY = 220                      # daily bars a name needs (the study's `len(d) < 220` skip)
PREFILTER_SHARE = 0.6                  # names within 60% of the floor on the cheap estimates get the exact test
PREFILTER_SESSIONS = 20                # whole-market daily bars read by the prefilter
PREFILTER_MIN_SESSIONS = 15
SPREAD_MARGIN = 1.0001                 # the study's sizing margin for the half-spread
FINAL_OUTCOMES = ("opened", "no_slot", "no_cash", "account_too_small", "held")


def enabled() -> bool:
    return bool(getattr(settings, "enable_dip_long", False))


def root() -> Path:
    return Path(settings.dip_long_dir)


def _iso(d: date) -> str:
    return d.isoformat()


# ── calendar (the selection short's, one definition) ─────────────────────────

def is_session(d: date) -> bool:
    from src.signals.sel_short import is_session as _is
    return _is(d)


def prev_session(d: date) -> date:
    """The last session strictly before ``d`` — the one whose close decides on ``d``."""
    from src.signals.sel_short import sessions_before
    return sessions_before(d, 1)[0]


def sessions_held(entry: date, through: date) -> int:
    """Sessions from ``entry`` through ``through``, both included (0 when ``through`` is before ``entry``)."""
    if through < entry:
        return 0
    from src.performance.market_calendar import market_days_between
    return market_days_between(entry - timedelta(days=1), through)


def today_et(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(ET).date()


def in_rth(now: Optional[datetime] = None) -> bool:
    """A regular-hours instant on a market day — the only time the book trades."""
    now = now or datetime.now(timezone.utc)
    from src.performance.market_calendar import current_session
    return is_session(today_et(now)) and current_session(now) == "rth"


def in_entry_window(now: Optional[datetime] = None) -> bool:
    """Regular hours, within `dip_long_entry_window_minutes` of the 09:30 open: the study bought at the open."""
    now = now or datetime.now(timezone.utc)
    if not in_rth(now):
        return False
    et = now.astimezone(ET)
    mins = et.hour * 60 + et.minute + et.second / 60.0 - 570.0
    return 0.0 <= mins <= float(settings.dip_long_entry_window_minutes)


# ── the study's construction ─────────────────────────────────────────────────

def daily_bars(df30: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """The deep store's 30-minute regular-hours bars (naive-UTC index, Open/High/Low/Close/Volume) as one
    row per ET session: ``o h l c v`` on a naive date index — the study's `daily_of`, row for row."""
    if df30 is None or len(df30) == 0:
        return None
    et = pd.DatetimeIndex(df30.index).tz_localize("UTC").tz_convert("America/New_York")
    d = pd.DataFrame({"date": et.normalize().tz_localize(None),
                      "o": pd.to_numeric(df30["Open"], errors="coerce").to_numpy(float),
                      "h": pd.to_numeric(df30["High"], errors="coerce").to_numpy(float),
                      "l": pd.to_numeric(df30["Low"], errors="coerce").to_numpy(float),
                      "c": pd.to_numeric(df30["Close"], errors="coerce").to_numpy(float),
                      "v": pd.to_numeric(df30["Volume"], errors="coerce").to_numpy(float)})
    d = d[np.isfinite(d.c) & (d.c > 0)]
    if d.empty:
        return None
    g = d.groupby("date", sort=True)
    return pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "v": g.v.sum()})


def rsi2(c: np.ndarray) -> np.ndarray:
    """The study's 2-session RSI: Wilder-style averages as an EWM with alpha 1/2 over the whole series."""
    d = np.diff(c, prepend=np.nan)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ru = pd.Series(up).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    rd = pd.Series(dn).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + ru / rd)
    return np.where(rd == 0, 100.0, r)


def stats_at(daily: Optional[pd.DataFrame], through: date, min_history: int = MIN_HISTORY) -> Optional[dict]:
    """The rule's numbers at the close of the last session on or before ``through`` (nothing later is read):
    its ``session``, ``close``, the trend and exit averages, ``rsi2`` and ``dv20`` (the 20-session mean of
    close x volume); None below ``min_history`` daily bars."""
    if daily is None or daily.empty:
        return None
    d = daily[daily.index <= pd.Timestamp(through)]
    if len(d) < max(1, int(min_history)):
        return None
    c = d["c"].to_numpy(float)
    v = d["v"].to_numpy(float)
    nt, nx = int(settings.dip_long_trend_sessions), int(settings.dip_long_exit_sma_sessions)
    sma_t = float(np.mean(c[-nt:])) if len(c) >= nt else float("nan")
    sma_x = float(np.mean(c[-nx:])) if len(c) >= nx else float("nan")
    dv20 = float(np.mean((c * v)[-20:])) if len(c) >= 20 else float("nan")
    return {"session": pd.Timestamp(d.index[-1]).date(), "close": float(c[-1]), "sma_trend": sma_t,
            "sma_exit": sma_x, "rsi2": float(rsi2(c)[-1]), "dv20": dv20, "bars": int(len(c))}


def qualifies(st: Optional[dict]) -> bool:
    """The entry rule on a name's numbers at the close of D-1."""
    if not st:
        return False
    return (math.isfinite(st["dv20"]) and st["dv20"] >= float(settings.dip_long_min_dollar_volume)
            and math.isfinite(st["sma_trend"]) and st["close"] > st["sma_trend"]
            and math.isfinite(st["rsi2"]) and st["rsi2"] < float(settings.dip_long_rsi_max))


def _sec_type(ticker: str) -> Optional[str]:
    from src.data.company_names import security_type
    return security_type(ticker)


def _daily(ticker: str) -> Optional[pd.DataFrame]:
    from src.data.intraday_store import load_deep_30m
    return daily_bars(load_deep_30m(ticker))


def name_stats(ticker: str, through: date, min_history: int = MIN_HISTORY) -> Optional[dict]:
    return stats_at(_daily(ticker), through, min_history)


def _extend(tickers: Sequence[str], through: date) -> None:
    """Bring the deep store of ``tickers`` through ``through`` (the 08:30 pre-open run and the selection
    short's prepare normally have): a handful of Polygon calls on a normal morning. Fail-soft."""
    if not tickers:
        return
    try:
        from src.data.intraday_store import extend_deep_30m
        extend_deep_30m(list(tickers), workers=4, budget_seconds=120.0, min_age_days=0, today=through)
    except Exception as e:                                      # noqa: BLE001
        logger.warning(f"[dip_long] store extension for {len(tickers)} name(s) failed ({e})")


# ── the day's signals ────────────────────────────────────────────────────────

def candidates(d: date) -> Tuple[List[str], str]:
    """The names worth the exact test on ``d``: within `PREFILTER_SHARE` of the dollar-volume floor on
    Polygon's whole-market daily bars of the 20 sessions before ``d`` (the selection short's cached
    `grouped_day`) or on the day's universe file — common stocks / ADRs only. Neither available: every
    name in the deep store (slow, logged)."""
    from src.signals import sel_short
    floor = PREFILTER_SHARE * float(settings.dip_long_min_dollar_volume)
    names: set = set()
    src: List[str] = []
    frames = {}
    for s in sel_short.sessions_before(d, PREFILTER_SESSIONS):
        try:
            g = sel_short.grouped_day(s)
        except Exception as e:                                  # noqa: BLE001
            logger.debug(f"[dip_long] whole-market bars {s} unavailable ({e})")
            g = None
        if g is not None and len(g):
            frames[s] = g.drop_duplicates("ticker").set_index("ticker")
    if len(frames) >= PREFILTER_MIN_SESSIONS:
        dv = pd.DataFrame({s: g["close"] * g["volume"] for s, g in frames.items()}).mean(axis=1, skipna=True)
        names |= {sel_short._internal_symbol(t) for t, v in dv.items() if np.isfinite(v) and v >= floor}
        src.append("grouped")
    uni = sel_short.load_universe(d)
    if uni:
        names |= {t for t, v in uni.items() if v is not None and float(v) >= floor}
        src.append("universe")
    if not names:
        from src.data.intraday_store import DEEP_DIR
        names = {p.stem for p in DEEP_DIR.glob("*.pkl")}
        src.append("store")
        logger.warning(f"[dip_long] {d}: no prefilter data — testing all {len(names)} stored names")
    return sorted(t for t in names if _sec_type(t) in TYPES), "+".join(src)


def _traded_on(s: date) -> Optional[set]:
    """The names (the project's symbol form) in Polygon's whole-market daily bars of session ``s``; None when
    those bars are unavailable."""
    from src.signals import sel_short
    try:
        g = sel_short.grouped_day(s)
    except Exception:                                           # noqa: BLE001
        return None
    if g is None or not len(g):
        return None
    return {sel_short._internal_symbol(t) for t in g["ticker"].astype(str)}


def signals_path(d: date) -> Path:
    return root() / "signals" / f"{_iso(d)}.json"


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def compute_signals(d: date, extend: bool = True) -> dict:
    """The entry signals for session ``d``: every prefiltered name's numbers at the close of D-1 from its
    store series cut at D-1; a name whose store ends before D-1 is extended once, then left out
    (``stale``). Written to ``signals/<d>.json``, deepest dip first."""
    prev = prev_session(d)
    names, src = candidates(d)
    t0 = datetime.now(timezone.utc)

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
        _extend(stale, prev)
        more, stale = evaluate(stale)
        rows.update(more)
    inactive: List[str] = []
    if stale:
        # a name that did not trade on D-1 at all (a merger closed, a halt) is not a data fault
        traded = _traded_on(prev)
        if traded is not None:
            inactive = [t for t in stale if t not in traded]
            stale = [t for t in stale if t in traded]
    sig = [{"ticker": tk, **st} for tk, st in rows.items() if qualifies(st)]
    sig.sort(key=lambda r: (r["rsi2"], r["ticker"]))
    big = sorted((r for r in ({"ticker": tk, **st} for tk, st in rows.items())
                  if math.isfinite(r["dv20"]) and r["dv20"] >= float(settings.dip_long_min_dollar_volume)),
                 key=lambda r: r["rsi2"])
    out = {"day": _iso(d), "signal_session": _iso(prev), "source": src, "considered": len(names),
           "evaluated": len(rows), "eligible": len(big), "stale": stale, "inactive": inactive,
           "rule": {"min_dollar_volume": float(settings.dip_long_min_dollar_volume),
                    "rsi_max": float(settings.dip_long_rsi_max),
                    "trend_sessions": int(settings.dip_long_trend_sessions)},
           "signals": sig,
           # the closest misses among the names over the floor — journal-only
           "next": [r for r in big if not qualifies(r)][:10],
           "computed_at": t0.isoformat(timespec="seconds"),
           "seconds": round((datetime.now(timezone.utc) - t0).total_seconds(), 1)}
    _write_json(signals_path(d), out)
    logger.info(f"[dip_long] {d}: {len(sig)} signal(s) from the {prev} close among {len(big)} name(s) over "
                f"${float(settings.dip_long_min_dollar_volume) / 1e9:g}B a day ({len(names)} prefiltered by {src}"
                f"{f', {len(stale)} stale' if stale else ''}, {out['seconds']:.0f}s)"
                + (": " + ", ".join(f"{r['ticker']} RSI2 {r['rsi2']:.1f}" for r in sig) if sig else ""))
    return out


def signals_for(d: date) -> dict:
    """The day's signals — computed once a day, then read from ``signals/<d>.json``."""
    p = signals_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            logger.warning(f"[dip_long] {p} unreadable ({e}) — recomputing")
    return compute_signals(d)


# ── exits ────────────────────────────────────────────────────────────────────

def entry_day(trade: dict) -> Optional[date]:
    """The session the long was bought in."""
    raw = trade.get("dip_entry_day") or str(trade.get("entry_date") or "")[:10]
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def rebound_session(daily: Optional[pd.DataFrame], entry: date, through: date) -> Optional[dict]:
    """The FIRST session from ``entry`` through ``through`` whose close is above its exit average (the
    study's exit trigger), as ``{"session", "close", "sma_exit"}``; None when there is none. Reading every
    session since the entry, not only the last, means a tick or a day the book missed delays the exit
    rather than losing it."""
    if daily is None or daily.empty:
        return None
    d = daily[daily.index <= pd.Timestamp(through)]
    if d.empty:
        return None
    c = d["c"]
    sma = c.rolling(int(settings.dip_long_exit_sma_sessions)).mean()
    hit = d.index[(d.index >= pd.Timestamp(entry)) & (c > sma).to_numpy()]
    if len(hit) == 0:
        return None
    s = hit[0]
    return {"session": pd.Timestamp(s).date(), "close": float(c.loc[s]), "sma_exit": float(sma.loc[s])}


def exit_check(trade: dict, d: date) -> Tuple[Optional[str], dict]:
    """Does session ``d``'s open close this long? ``dip_rebound`` once a close from the entry session
    through D-1 is above its exit average (the first one is what the study sold on), ``dip_time`` once the
    long has held `dip_long_max_hold_sessions` sessions through D-1; else None. ``info`` carries the
    numbers (``stale`` when the store had no close for D-1, extended once first)."""
    e = entry_day(trade)
    prev = prev_session(d)
    info: dict = {"through": _iso(prev)}
    if e is None or prev < e:
        return None, info
    held = sessions_held(e, prev)
    info["held"] = held
    tk = str(trade["ticker"])
    daily = _daily(tk)
    last = None if daily is None or daily.empty else pd.Timestamp(daily.index[daily.index <= pd.Timestamp(prev)].max())
    if last is None or pd.isna(last) or last.date() != prev:
        _extend([tk], prev)
        daily = _daily(tk)
        last = None if daily is None or daily.empty else pd.Timestamp(daily.index[daily.index <= pd.Timestamp(prev)].max())
    if last is None or pd.isna(last) or last.date() != prev:
        info["stale"] = True
        logger.warning(f"[dip_long] {tk}: no stored close for {prev} — the rebound rule is judged on the closes "
                       f"the store has")
    hit = rebound_session(daily, e, prev)
    if hit is not None:
        info.update(rebound_session=_iso(hit["session"]), close=hit["close"], sma_exit=hit["sma_exit"])
        return "dip_rebound", info
    if held >= int(settings.dip_long_max_hold_sessions):
        return "dip_time", info
    return None, info


# ── the book's simulated cash account ────────────────────────────────────────

def funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == MECHANISM and trade.get("dip_account_shares") is not None


def comm(n: float, p: float) -> float:
    """IBKR's fixed commission: max($1, $0.005 a share), at most 1% of the order's value."""
    return min(max(1.0, 0.005 * n), 0.01 * n * p)


def account_state(trades: Sequence[dict]) -> Dict[str, float]:
    """The book's account: equity = the start balance + every funded trade's dollar P&L (shares x entry x
    the ledger's net return; an open trade at its live mark), cash = the start balance + the realized P&L -
    the open longs' cost, and the open count."""
    out = {"initial": float(settings.dip_long_account_initial), "realized": 0.0, "unrealized": 0.0,
           "cost_open": 0.0, "open": 0, "funded": 0}
    for t in trades:
        if not funded(t):
            continue
        try:
            n, e = float(t["dip_account_shares"]), float(t.get("entry_price") or 0.0)
            pnl = n * e * float(t.get("return_pct") or 0.0) / 100.0
        except (TypeError, ValueError):
            continue
        out["funded"] += 1
        if t.get("status") == "OPEN":
            out["unrealized"] += pnl
            out["cost_open"] += n * e
            out["open"] += 1
        else:
            out["realized"] += pnl
    out["equity"] = out["initial"] + out["realized"] + out["unrealized"]
    out["cash"] = out["initial"] + out["realized"] - out["cost_open"]
    return out


def size(trades: Sequence[dict], price: float) -> Tuple[int, str, Dict[str, float]]:
    """Whole shares for a new long at ``price``: floor(min(equity / slices, cash) / (price x the spread
    margin)), cut until the cost with commission fits the cash — the study's account. ``(0, why, state)``
    with ``no_slot`` (every slice open), ``no_cash`` or ``account_too_small``."""
    st = account_state(trades)
    slices = max(1, int(settings.dip_long_account_slices))
    if st["open"] >= slices:
        return 0, "no_slot", st
    budget = min(st["equity"] / slices, st["cash"])
    if not price or price <= 0 or budget <= 0:
        return 0, "no_cash", st
    n = int(math.floor(budget / (float(price) * SPREAD_MARGIN + 0.01)))
    while n > 0 and n * float(price) * SPREAD_MARGIN + comm(n, float(price)) > st["cash"]:
        n -= 1
    if n < 1:
        return 0, "account_too_small", st
    return n, "ok", st


def summary(st: Dict[str, float]) -> str:
    return (f"equity ${st['equity']:,.0f}, cash ${st['cash']:,.0f}, {int(st['open'])}/"
            f"{int(settings.dip_long_account_slices)} slices open")


# ── journals ─────────────────────────────────────────────────────────────────

def entries_path(d: date) -> Path:
    return root() / "entries" / f"{_iso(d)}.jsonl"


def exits_path(d: date) -> Path:
    return root() / "exits" / f"{_iso(d)}.jsonl"


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
                              "ticker": sig.get("ticker"), "outcome": outcome,
                              **{k: sig.get(k) for k in ("rsi2", "close", "sma_trend", "dv20")}, **kw})


def journal_exit(d: date, trade: dict, reason: str, info: dict) -> None:
    _append(exits_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "ticker": trade.get("ticker"), "reason": reason,
                            "trade": trade.get("recommendation_id"), **info})


def settled(d: date) -> Dict[str, str]:
    """The signals of ``d`` already settled for good (a missing price is retried)."""
    return {str(r.get("ticker")): str(r.get("outcome")) for r in _read_jsonl(entries_path(d))
            if r.get("outcome") in FINAL_OUTCOMES}


def trade_id(d: date, ticker: str) -> str:
    import hashlib
    return hashlib.sha1(f"dip|{_iso(d)}|{ticker}".encode("utf-8")).hexdigest()[:16]


# ── health (the email's scorer banner) ───────────────────────────────────────

def health(now: Optional[datetime] = None, trades: Optional[Sequence[dict]] = None) -> dict:
    """``problems`` for the email digest: on a market day after the entry window, no signal computation
    for today (the book did not look at the open) or a computation that left names stale; an open long
    held past its limit (an exit that did not go out)."""
    now = now or datetime.now(timezone.utc)
    d = today_et(now)
    out: dict = {"problems": [], "notes": []}
    if not is_session(d):
        return out
    et = now.astimezone(ET)
    after_window = et.hour * 60 + et.minute > 570 + float(settings.dip_long_entry_window_minutes)
    if enabled() and after_window and et.hour < 16:
        p = signals_path(d)
        if not p.exists():
            out["problems"].append("dip long: no signal computation for today's open (logs: [dip_long])")
        else:
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                if s.get("stale"):
                    out["problems"].append(f"dip long: {len(s['stale'])} name(s) had no close for "
                                           f"{s.get('signal_session')} in the store: {', '.join(s['stale'][:8])}")
                else:
                    out["notes"].append(f"dip long: {len(s.get('signals') or [])} signal(s) at the open")
            except Exception:                                   # noqa: BLE001
                out["problems"].append("dip long: today's signal file is unreadable")
    late = []
    for t in trades or []:
        if t.get("status") != "OPEN" or t.get("entry_mechanism") != MECHANISM:
            continue
        e = entry_day(t)
        if e is not None and sessions_held(e, prev_session(d)) > int(settings.dip_long_max_hold_sessions):
            late.append(str(t["ticker"]))
    if late:
        out["problems"].append(f"dip long: held past {settings.dip_long_max_hold_sessions} sessions: {', '.join(late)}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--signals", action="store_true", help="compute (and print) a day's signals")
    ap.add_argument("--status", action="store_true", help="the book's account and open longs")
    ap.add_argument("--day", default=None, help="the session the signals trade at (default: today)")
    ap.add_argument("--no-write", action="store_true", help="with --signals: print only, never write the file")
    a = ap.parse_args(argv)
    d = date.fromisoformat(a.day) if a.day else today_et()
    if a.signals:
        if a.no_write:
            global _write_json                                # noqa: PLW0603 — a dry run writes nothing

            def _write_json(p, obj):                          # type: ignore[no-redef]
                return None
        out = compute_signals(d)
        print(json.dumps({k: v for k, v in out.items() if k != "next"}, indent=1, default=str))
        return 0
    if a.status:
        from src.db import repo
        repo.set_read_only(True)
        trades = repo.load_trades()
        st = account_state(trades)
        print(summary(st))
        for t in trades:
            if t.get("entry_mechanism") == MECHANISM and t.get("status") == "OPEN":
                print(f"  {t['ticker']}: {t.get('dip_account_shares')} sh @ {t.get('entry_price')} since "
                      f"{t.get('dip_entry_day')}, now {t.get('current_price')} ({t.get('return_pct')}%)")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
