"""THE ETF DUAL-MOMENTUM BOOK (user 2026-10-08: "Improve the ETF trading strategy focussing only on ETFs alone ...
Also consider leveraged ETFs ... Don't stop until you have a working, proven algo that can trade ETFs"; the studies
PREREG46-51 in llm_trader_research/etfalgo/, `memory/etf-dual-momentum-2026-10.md`).

The rule (PREREG49's R4-DM1, Antonacci's dual momentum with a 3/6/12-month score):

* at each month's LAST session close, score each fund of `etf_dm_universe` — QQQ (Nasdaq-100), SPY (S&P 500), IWM
  (Russell 2000), TLT (20+ year Treasuries), GLD (gold) — by the mean of its 3-, 6- and 12-month price returns (63 /
  126 / 252 sessions; a fund needs 253 closes);
* hold the top fund for the next month if its 12-month return beats the 3-month T-bill's over the same 12 months
  (FRED DTB3, compounded per session); otherwise cash;
* trade on the month's first session inside `etf_dm_entry_window_minutes` of the 09:30 open (the backtests bought at
  the open; a session missed carries the month's decision to the next one); an unchanged pick is kept.

Evidence: the untouched test PREREG51 (1988-01 .. 2004-06, index proxies): +15.5 %/yr against the S&P 500 total
return's +12.1, 95% of the log growth +0.046 .. +0.234, both halves positive, worst drawdown 46% vs 47%; 2004-10 +5.7
vs +0.2; 2010-17 +6.5 vs +14.5; 2018-26 +13.5 vs +14.7 (drawdown 25% vs 34%); 1988-2026 +11.7 vs +11.3 %/yr. The same
rule on 2x / 3x funds: +21.8 / +23.3 %/yr with 78% / 93% drawdowns (1988-2004) — leveraged funds are not used.

The closes are the deep 30-minute store's regular-hours sessions (`dip_long.daily_bars`), cut at the month-end
session — never a bar of the trading session (`tests/test_no_lookahead.py`).

Sizing: the book's OWN simulated cash account, `etf_dm_account_initial` ($10,000, no deposits) + its trades' dollar
P&L; the whole account in the held fund (whole shares, the dip book's spread-and-commission margin).

Journals under `etf_dm_dir`: ``decisions/<YYYY-MM>.json`` (the month's scores and pick), ``rebalances/<day>.jsonl``.
CLI: ``python -m src.signals.etf_dm --decide [--day D] [--no-write]`` / ``--status``.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.performance.books import ETF_DM as MECHANISM
from src.signals import dip_long as DL
from src.utils import ET

LOOKBACKS = (63, 126, 252)                 # 3 / 6 / 12 months of sessions
MIN_CLOSES = 253


def enabled() -> bool:
    return bool(getattr(settings, "enable_etf_dm", False))


def root() -> Path:
    return Path(settings.etf_dm_dir)


def universe() -> Tuple[str, ...]:
    return tuple(t.strip().upper() for t in str(settings.etf_dm_universe).split(",") if t.strip())


today_et = DL.today_et
in_rth = DL.in_rth
is_session = DL.is_session
prev_session = DL.prev_session


def in_entry_window(now: Optional[datetime] = None) -> bool:
    """Regular hours, within `etf_dm_entry_window_minutes` of the 09:30 open."""
    now = now or datetime.now(timezone.utc)
    if not in_rth(now):
        return False
    et = now.astimezone(ET)
    mins = et.hour * 60 + et.minute + et.second / 60.0 - 570.0
    return 0.0 <= mins <= float(settings.etf_dm_entry_window_minutes)


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def decision_session(d: date) -> date:
    """The month-end session whose close decides the month of ``d``: the last session before the month began."""
    first = date(d.year, d.month, 1)
    return prev_session(first)


# ── data ─────────────────────────────────────────────────────────────────────

def closes(ticker: str, through: date) -> Optional[pd.Series]:
    """The fund's daily closes (the deep store's regular-hours sessions) through ``through``, never later."""
    from src.data.intraday_store import load_deep_30m
    d = DL.daily_bars(load_deep_30m(ticker))
    if d is None:
        return None
    c = d["c"]
    return c[c.index <= pd.Timestamp(through)]


def _extend(tickers: Sequence[str], through: date) -> None:
    """Bring the funds' deep store through ``through`` (the pre-open run normally has). Fail-soft."""
    if not tickers:
        return
    try:
        from src.data.intraday_store import extend_deep_30m
        extend_deep_30m(list(tickers), workers=4, budget_seconds=120.0, min_age_days=0, today=through)
    except Exception as e:                                        # noqa: BLE001
        logger.warning(f"[etf_dm] store extension for {len(tickers)} fund(s) failed ({e})")


def tbill_growth(start: date, end: date) -> float:
    """The 3-month T-bill compounded per session over (start, end]: DTB3 / 100 / 252 a session, the latest value
    published by ``end`` (ALFRED vintages: realtime_start <= end)."""
    import duckdb
    from src.data.deep import DEEP_DIR
    p = DEEP_DIR / "fred_vintages.parquet"
    r = duckdb.connect().execute(
        f"select date, value from '{p.as_posix()}' where series_id='DTB3' and value is not null "
        f"and date <= ? and realtime_start <= ? "
        "qualify row_number() over (partition by date order by realtime_start desc) = 1 order by date",
        [str(end), str(end)]).fetchdf()
    rf = pd.Series(r.value.to_numpy(float), index=pd.to_datetime(r.date))
    sess = [d for d in pd.bdate_range(pd.Timestamp(start) + pd.Timedelta(days=1), pd.Timestamp(end))
            if is_session(d.date())]
    if not sess or rf.empty:
        return 0.0
    vals = rf.reindex(rf.index.union(pd.DatetimeIndex(sess))).ffill().reindex(pd.DatetimeIndex(sess)).fillna(0.0)
    return float(np.prod(1 + vals.to_numpy() / 100.0 / 252.0) - 1)


def fund_stats(c: Optional[pd.Series], through: date) -> Optional[dict]:
    """A fund's returns over the lookbacks and its score, from closes cut at ``through`` (None: too short or the
    close of ``through`` missing)."""
    if c is None or len(c) < MIN_CLOSES or c.index[-1].date() != through:
        return None
    r = {f"r{n}": float(c.iloc[-1] / c.iloc[-1 - n] - 1) for n in LOOKBACKS}
    return {**r, "score": float(np.mean(list(r.values()))), "close": float(c.iloc[-1]),
            "start12": str(c.index[-1 - LOOKBACKS[-1]].date())}


# ── the month's decision ─────────────────────────────────────────────────────

def decision_path(d: date) -> Path:
    return root() / "decisions" / f"{month_key(d)}.json"


def compute_decision(d: date, extend: bool = True, write: bool = True) -> dict:
    """The decision for the month of session ``d``: every fund's score at the previous month's last close, the top
    one, and whether it beat the T-bill over 12 months."""
    me = decision_session(d)
    stats: Dict[str, Optional[dict]] = {tk: fund_stats(closes(tk, me), me) for tk in universe()}
    stale = [tk for tk, s in stats.items() if s is None]
    if stale and extend:
        _extend(stale, me)
        for tk in stale:
            stats[tk] = fund_stats(closes(tk, me), me)
        stale = [tk for tk, s in stats.items() if s is None]
    live = {tk: s for tk, s in stats.items() if s is not None}
    top = max(live, key=lambda tk: live[tk]["score"]) if live else None
    rf12 = tbill_growth(date.fromisoformat(live[top]["start12"]), me) if top else None
    pick = top if top and live[top]["r252"] > rf12 else None
    out = {"month": month_key(d), "decided_at_close": str(me), "funds": stats, "stale": stale, "top": top,
           "top_r12": live[top]["r252"] if top else None, "tbill_12m": rf12, "pick": pick,
           "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if write:
        DL._write_json(decision_path(d), out)
    logger.info(f"[etf_dm] {month_key(d)}: " + (f"hold {pick}" if pick else "cash")
                + (f" — top {top} scored {live[top]['score'] * 100:+.1f}% (12 months {live[top]['r252'] * 100:+.1f}% vs "
                   f"T-bills {rf12 * 100:+.1f}%)" if top else " — no fund scored")
                + (f"; stale: {', '.join(stale)}" if stale else ""))
    return out


def decision_for(d: date) -> dict:
    p = decision_path(d)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:                                    # noqa: BLE001
            logger.warning(f"[etf_dm] {p} unreadable ({e}) — recomputing")
    return compute_decision(d)


# ── the book's simulated cash account ────────────────────────────────────────

def funded(trade: dict) -> bool:
    return trade.get("entry_mechanism") == MECHANISM and trade.get("etf_dm_account_shares") is not None


def account_state(trades: Sequence[dict]) -> Dict[str, float]:
    """Equity = the start balance + every funded trade's dollar P&L (an open one at its live mark); cash = the start
    balance + the realized P&L - the open position's cost."""
    out = {"initial": float(settings.etf_dm_account_initial), "realized": 0.0, "unrealized": 0.0, "cost_open": 0.0,
           "open": 0}
    for t in trades:
        if not funded(t):
            continue
        try:
            n, e = float(t["etf_dm_account_shares"]), float(t.get("entry_price") or 0.0)
            pnl = n * e * float(t.get("return_pct") or 0.0) / 100.0
        except (TypeError, ValueError):
            continue
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
    """Whole shares of the new fund with the whole cash: floor(cash / (price x the spread margin)), cut until the
    cost with commission fits."""
    st = account_state(trades)
    if st["open"]:
        return 0, "held", st
    if not price or price <= 0 or st["cash"] <= 0:
        return 0, "no_cash", st
    n = int(math.floor(st["cash"] / (float(price) * DL.SPREAD_MARGIN + 0.01)))
    while n > 0 and n * float(price) * DL.SPREAD_MARGIN + DL.comm(n, float(price)) > st["cash"]:
        n -= 1
    if n < 1:
        return 0, "account_too_small", st
    return n, "ok", st


# ── journals ─────────────────────────────────────────────────────────────────

def rebalances_path(d: date) -> Path:
    return root() / "rebalances" / f"{d.isoformat()}.jsonl"


def journal(d: date, action: str, **kw) -> None:
    DL._append(rebalances_path(d), {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": action, **kw})


def trade_id(d: date, ticker: str) -> str:
    import hashlib
    return hashlib.sha1(f"etf_dm|{d.isoformat()}|{ticker}".encode("utf-8")).hexdigest()[:16]


# ── health (the email's scorer banner) ───────────────────────────────────────

def health(now: Optional[datetime] = None, trades: Optional[Sequence[dict]] = None) -> dict:
    """``problems``: after the month's first entry window, no decision for the month, or the book's holding not
    the decision's pick (a rotation that did not go out)."""
    now = now or datetime.now(timezone.utc)
    d = today_et(now)
    out: dict = {"problems": [], "notes": []}
    if not enabled() or not is_session(d):
        return out
    et = now.astimezone(ET)
    first = decision_session(d) == prev_session(d)
    after_window = (not first) or (et.hour * 60 + et.minute > 570 + float(settings.etf_dm_entry_window_minutes))
    if not after_window:
        return out
    p = decision_path(d)
    if not p.exists():
        out["problems"].append(f"ETF dual momentum: no decision for {month_key(d)} (logs: [etf_dm])")
        return out
    try:
        dec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        out["problems"].append("ETF dual momentum: the month's decision file is unreadable")
        return out
    held = sorted({str(t["ticker"]) for t in trades or [] if t.get("status") == "OPEN"
                   and t.get("entry_mechanism") == MECHANISM})
    want = [dec["pick"]] if dec.get("pick") else []
    if held != want:
        out["problems"].append(f"ETF dual momentum: holds {', '.join(held) or 'cash'} but {month_key(d)}'s pick is "
                               f"{dec.get('pick') or 'cash'}")
    elif first:
        out["notes"].append(f"ETF dual momentum: {month_key(d)} in {dec.get('pick') or 'cash'}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decide", action="store_true", help="compute (and print) the month's decision")
    ap.add_argument("--status", action="store_true", help="the book's account and holding")
    ap.add_argument("--day", default=None, help="a session of the month (default: today)")
    ap.add_argument("--no-write", action="store_true", help="with --decide: print only")
    a = ap.parse_args(argv)
    d = date.fromisoformat(a.day) if a.day else today_et()
    if a.decide:
        print(json.dumps(compute_decision(d, write=not a.no_write), indent=1, default=str))
        return 0
    if a.status:
        from src.db import repo
        repo.set_read_only(True)
        trades = repo.load_trades()
        st = account_state(trades)
        print(f"equity ${st['equity']:,.0f}, cash ${st['cash']:,.0f}")
        for t in trades:
            if t.get("entry_mechanism") == MECHANISM and t.get("status") == "OPEN":
                print(f"  {t['ticker']}: {t.get('etf_dm_account_shares')} sh @ {t.get('entry_price')} since "
                      f"{t.get('entry_date')}, now {t.get('current_price')} ({t.get('return_pct')}%)")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
