"""The ETF dual-momentum book as backtest pieces — the live rule (`src/signals/etf_dm.py`, PREREG46-51), so it can
trade in one account beside the other books (`account.dm_long`: the whole account in one fund, one fund at a time).

DECISION for each month, at the last session close before it began (`etf_dm.decision_session`; sessions = SPY's
daily bars): `etf_dm.fund_stats` on each fund's daily closes cut at that close — the repo's daily OHLCV cache
(`cache.load_ohlcv`, Polygon, 2006 on), because the deep 30-minute store the live book reads starts in 2021-01, short
of the 253 closes a 2021 decision needs — and the top fund is held when its 12 months beat the T-bill's
(`etf_dm.tbill_growth`: DTB3 as published by that close; ALFRED's vintages start 2005-06). The live code's own
functions decide, so the backtest and the live book cannot drift apart.

HOLDING: bought at the OPEN of the month's first session in [lo, hi] (09:30, the first 30-minute bar's open; the
first month starts at ``lo``'s open, as a book deployed mid-month buys the month's pick at its next opening window),
sold at the open of the first session of a month whose pick differs (cash: sold, nothing bought); an unchanged pick
stays one piece. A holding still open at ``end`` (default ``hi``) is sold at that session's close. Marks: the fund's
30-minute regular-hours bars from the deep store (bar END times, closes and lows — the dip adapter's convention), or
one mark a day (16:00 close, daily low) where the store holds no bars. Half-spread: the dip study's dollar-volume
classes (1 bp at $1B+ a day), on the 20-session mean of close x volume at the decision close.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtest.strategies.dip import bars_30m, half_spread_bps
from src.signals import etf_dm

FUNDS = ("QQQ", "SPY", "IWM", "TLT", "GLD")
ET = "America/New_York"


def daily(ticker: str) -> Optional[pd.DataFrame]:
    """The fund's daily bars (Open High Low Close Volume, naive date index) from the repo's daily OHLCV cache."""
    from src.data.cache import load_ohlcv
    d = load_ohlcv(ticker)
    if d is None or len(d) == 0:
        return None
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    d.index = pd.DatetimeIndex(d.index).normalize()
    return d[~d.index.duplicated(keep="last")].sort_index()


def _ns(day: pd.Timestamp, hhmm: str) -> int:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {hhmm}", tz=ET).value


def schedule(lo, hi, data: Optional[Dict[str, pd.DataFrame]] = None,
             tbill: Optional[Callable] = None, funds: Sequence[str] = FUNDS) -> List[dict]:
    """The month-by-month decisions for the months with a session in [lo, hi]: the session the month starts trading
    on (``lo`` for the first), its decision close, each fund's 3/6/12-month score and the pick (None = cash)."""
    data = data if data is not None else {f: daily(f) for f in funds}
    tbill = tbill or etf_dm.tbill_growth
    lo, hi = pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize()
    cal = data["SPY"].index if "SPY" in data and data["SPY"] is not None else data[funds[0]].index
    sess = cal[(cal >= lo) & (cal <= hi)]
    out: List[dict] = []
    for k, t in enumerate(sess):
        if k and t.month == sess[k - 1].month:
            continue
        first = pd.Timestamp(t.year, t.month, 1)
        prior = cal[cal < first]
        if not len(prior):
            continue
        me = prior[-1]
        stats = {}
        for f in funds:
            d = data.get(f)
            c = None if d is None else d["Close"][d.index <= me]
            stats[f] = etf_dm.fund_stats(c, me.date())
        live = {f: s for f, s in stats.items() if s is not None}
        top = max(live, key=lambda f: live[f]["score"]) if live else None
        rf = tbill(pd.Timestamp(live[top]["start12"]).date(), me.date()) if top else None
        pick = top if top and live[top]["r252"] > rf else None
        out.append({"trade_day": t, "decided_at": me, "pick": pick, "top": top, "tbill_12m": rf,
                    "scores": {f: (s or {}).get("score") for f, s in stats.items()}})
    return out


def _with_opens(pt, pc, pl, o_t, o_p):
    """The marks plus a 09:30 mark at each later session's open, in time order."""
    t = np.r_[np.asarray(pt, np.int64), np.asarray(o_t, np.int64)]
    o = np.argsort(t, kind="stable")
    return t[o], np.r_[np.asarray(pc, float), o_p][o], np.r_[np.asarray(pl, float), o_p][o]


def _hold(fund: str, a: pd.Timestamp, b: Optional[pd.Timestamp], end: pd.Timestamp, d: pd.DataFrame,
          bars: Optional[pd.DataFrame], dv20: float, strategy: str, decided_at: pd.Timestamp,
          open_marks: bool = False) -> Optional[dict]:
    """One holding: bought at ``a``'s open, sold at ``b``'s open (or ``end``'s close when ``b`` is None). With
    ``open_marks`` it is also marked at each later session's 09:30 open (a sale at the open priced at the open)."""
    last_day = d.index[d.index < b][-1] if b is not None else end
    ens = _ns(a, "09:30")
    xns = _ns(b, "09:30") if b is not None else _ns(end, "16:00")
    pt = pc = pl = None
    e = x = None
    if bars is not None and len(bars):
        starts = pd.DatetimeIndex(bars.index)
        et_day = starts.tz_localize("UTC").tz_convert(ET).normalize().tz_localize(None)
        held = (et_day >= a) & (et_day <= last_day)
        covered = set(et_day[held]) >= set(d.index[(d.index >= a) & (d.index <= last_day)])
        on_a = np.flatnonzero(et_day == a)
        on_b = np.flatnonzero(et_day == b) if b is not None else np.zeros(0, int)
        if covered and len(on_a) and (b is None or len(on_b)):
            e = float(bars["Open"].iloc[on_a[0]])
            x = float(bars["Open"].iloc[on_b[0]]) if b is not None else float(bars["Close"].iloc[np.flatnonzero(held)[-1]])
            pt = (starts[held] + pd.Timedelta(minutes=30)).tz_localize("UTC").asi8
            pc, pl = bars["Close"].to_numpy(float)[held], bars["Low"].to_numpy(float)[held]
            if open_marks:
                idx = np.flatnonzero(held)
                days = et_day[held]
                first = np.r_[True, days[1:] != days[:-1]] & (days != a)
                sel = idx[first]
                pt, pc, pl = _with_opens(pt, pc, pl, [_ns(et_day[k], "09:30") for k in sel],
                                         bars["Open"].to_numpy(float)[sel])
    if e is None:                                               # no 30-minute bars: one mark a day
        dd = d.loc[a:last_day]
        e = float(d.loc[a, "Open"])
        x = float(d.loc[b, "Open"]) if b is not None else float(d.loc[end, "Close"])
        pt = np.array([_ns(t, "16:00") for t in dd.index], np.int64)
        pc, pl = dd["Close"].to_numpy(float), dd["Low"].to_numpy(float)
        if open_marks:
            later = dd.index[dd.index != a]
            pt, pc, pl = _with_opens(pt, pc, pl, [_ns(t, "09:30") for t in later], dd.loc[later, "Open"].to_numpy(float))
    if not (e > 0 and x > 0):
        return None
    hs = float(half_spread_bps([dv20])[0])
    return {"strategy": strategy, "tkn": fund, "ens": int(ens), "xns": int(xns), "e": e, "x": x,
            "hs_in": hs, "hs_out": hs, "dv20": float(dv20), "pt": np.asarray(pt, np.int64),
            "pc": np.asarray(pc, float), "pl": np.asarray(pl, float), "rank": 0.0,
            "net": 100.0 * (x / e - 1) - 2 * hs / 100.0, "days": (xns - ens) / 86400e9, "asset": fund,
            "month": f"{a.year:04d}-{a.month:02d}", "open_marks": bool(open_marks),
            "rebase": True,             # a standing holding: a window / a run starting mid-hold buys it then
            "pick_day": int((pd.Timestamp(decided_at).normalize() - pd.Timestamp("1970-01-01")).days)}


def after_calls(res: dict, held: Sequence[dict], end, strategy: str = "dm",
                data: Optional[Dict[str, pd.DataFrame]] = None, bars: Optional[Callable] = None) -> List[dict]:
    """The holdings a MARGIN CALL closed before their rule did (a shared account's call closes every position), each
    re-bought at the open of the session after the call — the live book re-buys the month's pick at its next opening
    window — as a new piece to the holding's own exit. Run the account again with them until none is added
    (`simulate_rejoining`)."""
    data = data if data is not None else {}
    bars = bars or bars_30m
    end = pd.Timestamp(end).normalize()
    by_entry = {(p["tkn"], int(p["ens"])): p for p in held}
    out: List[dict] = []
    for r in res.get("positions") or []:
        if r.get("strategy") != strategy or r.get("how") != "margin_call":
            continue
        p = by_entry.get((r["tkn"], int(r["entry_ns"])))
        if p is None:
            continue
        fund = p["tkn"]
        d = data.get(fund)
        if d is None:
            d = data[fund] = daily(fund)
        call_day = pd.Timestamp(int(r["exit_ns"]), unit="ns", tz="UTC").tz_convert(ET).tz_localize(None).normalize()
        nxt = d.index[d.index > call_day]
        if not len(nxt) or _ns(nxt[0], "09:30") >= int(p["xns"]):
            continue
        xt = pd.Timestamp(int(p["xns"]), unit="ns", tz="UTC").tz_convert(ET)
        b = xt.tz_localize(None).normalize() if (xt.hour, xt.minute) == (9, 30) else None
        decided = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(p["pick_day"]))
        q = _hold(fund, nxt[0], b, end if b is None else xt.tz_localize(None).normalize(), d, bars(fund),
                  float(p["dv20"]), strategy, decided, open_marks=bool(p.get("open_marks")))
        if q is not None:
            out.append(q)
    return out


def simulate_rejoining(P: Sequence[dict], strategies, rules, end, strategy: str = "dm", max_rounds: int = 10,
                       data: Optional[Dict[str, pd.DataFrame]] = None, bars: Optional[Callable] = None, **kw) -> dict:
    """`account.simulate` with the dual-momentum book re-buying after every margin call (`after_calls`), until no
    holding is left cut short. Returns the last run, its pieces in ``res["pieces"]``."""
    from src.backtest import account as A
    P = list(P)
    for _ in range(max_rounds):
        res = A.simulate(P, strategies, rules, **kw)
        held = [p for p in P if p.get("strategy") == strategy]
        known = {(p["tkn"], int(p["ens"])) for p in held}
        extra = [q for q in after_calls(res, held, end, strategy, data, bars) if (q["tkn"], int(q["ens"])) not in known]
        if not extra:
            res["pieces"] = P
            return res
        P += extra
    res["pieces"] = P
    return res


def pieces(lo, hi, end=None, strategy: str = "dm", data: Optional[Dict[str, pd.DataFrame]] = None,
           tbill: Optional[Callable] = None, bars: Optional[Callable] = None,
           funds: Sequence[str] = FUNDS, open_marks: bool = False) -> List[dict]:
    """The account engine's pieces for the dual-momentum rule: one per holding, decisions for the months traded in
    [lo, hi], a holding still open at ``end`` (default ``hi``) sold at its close; ``open_marks``: also marked at each
    session's 09:30 open (what a sale to make room for another book's entry at the open is priced at)."""
    data = data if data is not None else {f: daily(f) for f in funds}
    bars = bars or bars_30m
    end = pd.Timestamp(end if end is not None else hi).normalize()
    plan = schedule(lo, hi, data=data, tbill=tbill, funds=funds)
    out: List[dict] = []
    cur, since, dec = None, None, None
    cache: Dict[str, Optional[pd.DataFrame]] = {}

    def close(fund, a, b, decided):
        d = data[fund]
        if fund not in cache:
            cache[fund] = bars(fund)
        dv = (d["Close"] * d["Volume"]).rolling(20).mean()
        dv20 = float(dv[dv.index <= decided].iloc[-1]) if (dv.index <= decided).any() else float("nan")
        p = _hold(fund, a, b, end, d, cache[fund], dv20 if np.isfinite(dv20) else 0.0, strategy, decided,
                  open_marks=open_marks)
        if p is not None:
            out.append(p)

    for m in plan:
        if m["pick"] == cur:
            continue
        if cur is not None:
            close(cur, since, m["trade_day"], dec)
        cur, since, dec = m["pick"], m["trade_day"], m["decided_at"]
    if cur is not None and since is not None and since <= end:
        close(cur, since, None, dec)
    return out
