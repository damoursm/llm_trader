"""THE BANK OF MODELS, JUDGED ON THE POOLED HISTORY (user directive 2026-10-14: "Instead of checking for the three
different sets (A0, A, and B), check for the full dataset (A0+A+B). Each individual model needs to give growth on that
whole set but doesn't need to have growth all the time. I.e., a model could be good only under certain market
conditions that you will need to find ... What's important is that over the years 2018-2026 it has performed well
when it was activated and didn't produce a drawback that was too high. This means that winrate and return per day are
important metrics. In an effort to not overfit, we also need to have enough trades made so that the results are
significant, reliable, and transferable to out of sample data. To make sure the combination algorithm still has good
yearly and monthly growth, we then need multiple models in our bank of models so that we cover all market
conditions.").

    python -m src.backtest.bank [--no-write] [--runs DIR --label LABEL]   -> cache/ml/bank/<day>.json

THE POOLED HISTORY = sets A0 + A + B (`periods.POOLED`): decisions 2018-01-02 .. 2026-09-24, one history. The sets
stay the simulator's construction and are reported beside the years and the contexts; no decision reads one alone.

AN ARM (`card`, `qualify`) — its own trades: the reliability cards' (`reliability.build`: each arm alone through the
account engine at the live costs, `Mimic()` seed 0; A0 from the latest ``<day>_A0`` build, A and B from the latest
``<day>`` build — `audit.card_dirs`), pooled. It QUALIFIES for the bank when, on the pooled history:
  1. ENOUGH TRADES: at least `MIN_TRADES` (100) in at least `MIN_EPISODES` (30) distinct entry months — the trades of
     one month share its market, so the months are its independent episodes;
  2. SIGNIFICANT: the 95% interval of its RETURN PER DAY lies above its BAR (a bootstrap over ENTRY MONTHS — a
     month's trades resampled together —, `B_BOOT` resamples, seed 0). THE BAR (user directive 2026-10-15: "test long
     arms against the market's average daily rise. The idea is to beat the market but I'm also ok with using 3x for
     trades we are confident the market will go up (or down)."): for a LONG arm, what the same exposure earns on an
     average day over the same span (`BENCH`, `bench_rpd`: SPY's for a stock arm; for an index arm the fund it trades
     — a 3x fund's own average day, so the leverage is never the edge, the timing is); for a SHORT arm 0 (a random
     short loses the market's rise: 0 is the stricter bar);
  3. TRANSFERABLE: its return per day beats its bar in both chronological halves of its trades (each half against the
     bar over its own span), and with ANY one calendar year left out (a model one year carries does not transfer);
  4. GROWTH: the arm alone at its live slice of a $10,000 account — its P&L realized at each exit and compounded (the
     marks inside a trade unseen) — grows over the pooled history.
Short of trades only, it is INSUFFICIENT (not proven, not refuted); short of any other check it FAILS. THE LONGEST
HISTORY — an arm whose own data reach before 2018: the dip book's daily bars from 2007 and the crash book's QQQ from
1988 (PREREG103's LONG history, carried from `audit.baseline_json()` as yearly summaries), a candidate's earlier trades
as a table — decides checks 1-3 over the whole span (the halves = before 2018 / 2018-2026), check 4 on the pooled
history, when 2018-2026 alone hold too few trades, and — THE RE-JUDGING (user 2026-10-15: "Adopt the re-judging rule")
— when the arm passes every pooled check but significance with a return per day above its bar. Its significance is
CLUSTERED like the pool's: the month bootstrap over every month of an earlier trade table, or, with yearly summaries
only, the t of the mean trade against the bar's per-trade equivalent with each earlier year and each pooled month one
cluster (the larger of that and the independent-trade standard error; at least `T_SIG`). Beside the checks: the GROSS
win rate, the return per day, the worst trade, the skew, the drawdown at its live slice, every year and every set,
and the deflated Sharpe ratio for the trials behind it (`trials.count`; nominal trials, so conservative).

WHERE IT EARNS (``context``): its trades by SPY's trend at the entry (above / below the 200-session average), SPY's
20-session realized-volatility tercile, and the six cells they cross into. A cell of at least `MIN_CELL` (30) trades
whose interval lies above 0 EARNS, below 0 LOSES (where it makes or loses money: against 0 for every arm). The
terciles are cut on the whole history: a LABEL to read results
(`reliability.regime_table`) — a filter that DECIDES uses a point-in-time definition (a fixed level, a trailing
percentile) and gets its `tests/test_no_lookahead.py` guard.

A CONTEXT FILTER (`filter_test`) — an arm kept to the contexts it earns in — is justified by the trades it removes:
at least `MIN_CELL` of them in at least `MIN_CELL_MONTHS` (10) months, their return per day's interval below 0. A
filter, threshold or parameter CHOSEN on the history is measured out of time (`walk_forward`: each year from
`FIRST_WF_YEAR` decided on the years before it only). It then passes the pooled add rule (`add_rule.pooled_verdict`).
A filter justified by RISK instead — fewer margin calls, a shallower drawdown — is the user's call.

THE COMBINATION (`combination`) — the live configuration's account draws (a daily NAV each; positions realized at
their exits otherwise, flagged), the three sets chained: the pooled growth a year, the pooled worst drawdown, the
margin-call draws, every calendar year against SPY and QQQ (price only), every MONTH (the share of positive months,
the worst month, the worst twelve months, the longest time under water) and the COVERAGE: its months grouped by the
market context at their first session — where the bank earns, and where it is idle or loses (where a new arm is
worth the most) — with each arm's share of each context's months.
"""
from __future__ import annotations

import functools
import json
import math
import pickle
import re
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest import periods
from src.backtest.account import per_day

REPO = Path(__file__).resolve().parents[2]
BANK_DIR = REPO / "cache" / "ml" / "bank"
ET = "America/New_York"
ARMS = ("vol", "etf", "fall", "dip", "crash")
POOL = periods.POOLED
POOL_LO, POOL_HI = periods.pool_span(POOL)
MIN_TRADES = 100                 # 1. an arm's trades on the pooled history (or its longest)
MIN_EPISODES = 30                # 1. ... in this many distinct entry months
MIN_CELL = 30                    # a context cell a reading or a filter rests on
MIN_CELL_MONTHS = 10             # ... in this many distinct entry months
B_BOOT, SEED = 10_000, 0
T_SIG = 1.96                     # 2. the clustered t a longest history carried as yearly summaries must reach
# 2. THE BAR of each LONG live arm: the daily cache's symbol of what the same exposure earns on an average day (a short
# arm's bar is 0; `card`'s ``bench`` names a candidate's)
BENCH = {"dip": "SPY", "crash": "TQQQ"}
FIRST_WF_YEAR = 2020             # the walk-forward's first out-of-time year (two years to decide on)
START = 10_000.0
# the live configuration's latest draws with a daily NAV each (research repository): PREREG116's Q1 — the
# configuration from the 2026-10-16 session (2026-10-14's — vol give-back 1.25, the ETF arm without the VIX-linked
# products, PREREG102's LIVE_S3 — with the dip book's daily top pick at 1/6 and at most two shorts of one name)
LIVE_RUNS = ("studies/prereg116/runs", "Q1")
RUN_FILE = re.compile(r"^(A0|A|B)_(?P<label>.+)_s(?P<seed>\d+)\.pkl$")


def live_slices() -> Dict[str, float]:
    """Each arm's live slice of the account's equity (settings)."""
    from config.settings import settings as S
    return {"vol": 1.0 / float(S.sel_short_vol_account_slices), "etf": 1.0 / float(S.sel_short_etf_account_slices),
            "fall": 1.0 / float(S.sel_short_fall_account_slices), "dip": 1.0 / float(S.dip_long_account_slices),
            "crash": 1.0 / float(S.crash_long_account_slices)}


# ── the trades ───────────────────────────────────────────────────────────────

def load_trades(cards: Optional[Mapping[str, Path]] = None) -> pd.DataFrame:
    """Every arm's trades on the pooled history, one table (`audit.uniform`'s columns + ``month``): the latest card
    build holding each set (`audit.card_dirs`)."""
    from src.backtest import audit as AU
    dirs = dict(cards) if cards is not None else AU.card_dirs()
    parts, seen = [], set()
    for st in POOL:
        d = dirs.get(st)
        if d is None or d in seen:
            continue
        seen.add(d)
        t, _ = AU.uniform(pd.read_pickle(Path(d) / "trades_base.pkl"), POOL)
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["book", "ticker", "entry_day", "net", "gross", "days", "decision", "year", "set",
                                     "month"])
    T = pd.concat(parts, ignore_index=True).drop_duplicates(["book", "ticker", "entry_day", "net", "days"])
    T["month"] = pd.DatetimeIndex(T.entry_day).to_period("M").astype(str)
    return T.sort_values("entry_day").reset_index(drop=True)


def month_boot(t: pd.DataFrame, B: int = B_BOOT, seed: int = SEED) -> np.ndarray:
    """The return per day of ``t`` (net %, days, month) under ``B`` resamples of its ENTRY MONTHS with replacement
    (a month's trades together)."""
    g = t.groupby("month").agg(s=("net", "sum"), c=("net", "size"), d=("days", "sum"))
    M = len(g)
    if M < 2:
        return np.full(B, np.nan)
    rng = np.random.default_rng(seed)
    S, C, D = g.s.to_numpy(float), g.c.to_numpy(float), g.d.to_numpy(float)
    out = np.empty(B)
    for lo in range(0, B, 1000):
        k = min(1000, B - lo)
        idx = rng.integers(0, M, size=(k, M))
        s, c, d = S[idx].sum(1), C[idx].sum(1), D[idx].sum(1)
        out[lo:lo + k] = [per_day(a / n, b / n) for a, n, b in zip(s, c, d)]
    return out


def quality(t: pd.DataFrame, boot: bool = True) -> dict:
    """An arm's quality on trades ``t``: `reliability.stats` (n, average / median trade, hold, RETURN PER DAY, the
    GROSS win rate, worst / best, the five best trades' share) + the months, the skew, the t of the mean trade
    clustered by month and the return per day's 95% interval (`month_boot`)."""
    from src.backtest import reliability as REL
    s = REL.stats(t)
    if not s.get("n"):
        return s
    net = t.net.to_numpy(float)
    s["months"] = int(t.month.nunique())
    s["skew"] = float(pd.Series(net).skew()) if len(net) > 2 else None
    g = t.groupby("month").net.agg(["sum", "count"])
    M, n, m = len(g), len(net), float(net.mean())
    if M > 1:
        resid = g["sum"].to_numpy(float) - g["count"].to_numpy(float) * m
        se = math.sqrt(float((resid ** 2).sum()) * M / (M - 1)) / n
        s["t_month"] = m / se if se > 0 else None
    if boot and M >= 2:
        d = month_boot(t)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        s["rpd_lo95"], s["rpd_hi95"] = float(lo), float(hi)
    return s


def _cell_label(s: dict) -> Optional[str]:
    if s.get("n", 0) < MIN_CELL or s.get("months", 0) < MIN_CELL_MONTHS or s.get("rpd_lo95") is None:
        return None
    return "EARNS" if s["rpd_lo95"] > 0 else ("LOSES" if s["rpd_hi95"] < 0 else "unclear")


def alone(t: pd.DataFrame, slice_: float, lo: str, hi: str, start: float = START,
          max_open: Optional[int] = None) -> dict:
    """The arm alone at its live slice of a $10,000 account: each trade sized at ``slice_`` of the equity REALIZED at
    its entry, its P&L realized at its exit, compounded (the marks inside a trade unseen). Growth a year over [lo, hi],
    the worst drawdown of the realized equity, every calendar year's growth. ``max_open``: an entry while that many
    positions are open is skipped (counted in ``skipped``) — an arm whose trades overlap heavily (PREREG112: 283 at
    once) is otherwise read at many times its equity; None = every trade taken (the live arms' rules bound them)."""
    if not len(t):
        return {"growth": None, "maxdd": None, "years": {}}
    ent = pd.DatetimeIndex(t.entry_day)
    ext = ent + pd.to_timedelta(t.days.to_numpy(float), unit="D")
    ev = sorted([(e, 1, i) for i, e in enumerate(ent)] + [(x, 0, i) for i, x in enumerate(ext)])
    eq, size, path, skipped = start, {}, [(pd.Timestamp(lo), start)], 0
    net = t.net.to_numpy(float) / 100.0
    for when, kind, i in ev:                      # an exit before an entry at the same instant (kind 0 first)
        if kind == 1:
            if max_open is not None and len(size) >= int(max_open):
                skipped += 1
                continue
            size[i] = slice_ * eq
        elif i in size:
            eq += size.pop(i) * net[i]
            path.append((when, eq))
    s = pd.Series([v for _, v in path], index=pd.DatetimeIndex([w for w, _ in path])).groupby(level=0).last()
    yrs = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
    peak = s.cummax()
    by_year = {}
    prev = start
    for y, v in s.groupby(s.index.year).last().items():
        by_year[str(y)] = float(v / prev - 1.0) * 100.0
        prev = float(v)
    return {"growth": float((s.iloc[-1] / start) ** (1.0 / yrs) - 1.0) * 100.0 if s.iloc[-1] > 0 else -100.0,
            "maxdd": float(((peak - s) / peak).max() * 100.0), "end_equity": float(s.iloc[-1]), "years": by_year,
            "skipped": int(skipped), "max_open": max_open,
            "basis": f"1/{1 / slice_:g} of the realized equity a trade, P&L at the exits, from ${start:,.0f}"
                     + (f", at most {int(max_open)} open" if max_open is not None else "")}


# ── the bar: the market's average day (user directive 2026-10-15) ───────────

def _cache_returns(sym: str) -> pd.Series:
    """``sym``'s daily price returns (%) from the daily cache (`cache/ohlcv/<sym>.json`), by session date."""
    raw = json.loads((REPO / "cache" / "ohlcv" / f"{sym}.json").read_text(encoding="utf-8"))
    idx = pd.DatetimeIndex(pd.to_datetime(raw["index"]))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    c = pd.Series(pd.DataFrame(raw["data"], columns=raw["columns"])["Close"].to_numpy(float), index=idx.normalize())
    c = c[np.isfinite(c.to_numpy()) & (c.to_numpy() > 0)].sort_index()
    c = c[~c.index.duplicated(keep="last")]
    return (c.pct_change() * 100.0).iloc[1:]


@functools.lru_cache(maxsize=None)
def _carried_bench() -> dict:
    """PREREG120's carried benchmark histories {symbol: {"days", "ret" (%), ...}} (`audit.bench_history_json()`)."""
    from src.backtest import audit as AU
    try:
        return json.loads(AU.bench_history_json().read_text(encoding="utf-8")).get("specs", {})
    except OSError:
        return {}


@functools.lru_cache(maxsize=None)
def bench_daily(spec: str) -> pd.Series:
    """The benchmark ``spec``'s daily price returns (%, by session): the daily cache's and, before the cache's first
    return, the carried history when it holds ``spec`` (PREREG120: the index stitched before its ETF, a 3x fund
    synthetic before its inception — 3x the index's day, T-bill financing, 0.95 %/yr). Shared: never mutate it."""
    h = _carried_bench().get(spec)
    old = pd.Series(np.asarray(h["ret"], float), index=pd.DatetimeIndex(pd.to_datetime(h["days"]))) if h else None
    try:
        r = _cache_returns(spec)
    except OSError:
        if old is None:
            raise
        return old
    return pd.concat([old[old.index < r.index.min()], r]) if old is not None else r


def bench_rpd(daily: pd.Series, lo, hi, exclude_years: Sequence[int] = ()) -> Optional[float]:
    """The benchmark's RETURN PER DAY (%/24-hour day, the arms' unit) over its sessions in [lo, hi], with
    ``exclude_years`` left out: its average daily return over the average calendar days a session spans (each return
    counted over the days since the session before it)."""
    idx = pd.DatetimeIndex(daily.index)
    gap = np.r_[np.nan, np.diff(idx.asi8) / 86_400e9]
    m = (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
    if len(exclude_years):
        m &= ~np.isin(idx.year, [int(y) for y in exclude_years])
    r, g = daily.to_numpy(float)[m], gap[m]
    ok = np.isfinite(r) & np.isfinite(g)
    if ok.sum() < 2:
        return None
    return per_day(float(r[ok].mean()), float(g[ok].mean()))


def _bench_of(arm: str, bench) -> Tuple[Optional[pd.Series], Optional[str]]:
    """The daily returns and the name of an arm's bar: ``bench`` (a daily cache symbol, or a series of daily returns
    in %) or the live arm's (`BENCH`); (None, None) = judged against 0."""
    spec = bench if bench is not None else BENCH.get(arm)
    if spec is None:
        return None, None
    if isinstance(spec, pd.Series):
        return spec, str(spec.name or "series")
    return bench_daily(str(spec)), str(spec)


def _bar(daily: Optional[pd.Series], lo, hi, exclude_years: Sequence[int] = ()) -> float:
    """The bar over [lo, hi]: the benchmark's return per day, 0 without a benchmark (a short arm)."""
    if daily is None:
        return 0.0
    b = bench_rpd(daily, lo, hi, exclude_years)
    return float(b) if b is not None else 0.0


def _long_part(long: Mapping) -> Optional[dict]:
    """The years BEFORE the pooled history of a carried LONG history (`by_year` summaries): n, mean, the sum of
    squared deviations, the years with trades, and per year (n, mean, days)."""
    yrs = {y: r for y, r in (long.get("by_year") or {}).items() if int(y) < int(POOL_LO[:4]) and r.get("n")}
    if not yrs:
        return None
    n = sum(int(r["n"]) for r in yrs.values())
    m = sum(int(r["n"]) * float(r["avg_trade"]) for r in yrs.values()) / n
    ss = sum((int(r["n"]) - 1) * float(r.get("sd_trade") or 0.0) ** 2 + int(r["n"]) * (float(r["avg_trade"]) - m) ** 2
             for r in yrs.values())
    return {"n": n, "mean": m, "ss": ss, "years": sorted(yrs),
            "per_year": {y: (int(r["n"]), float(r["avg_trade"]), float(r.get("avg_days") or 0.0)) for y, r in yrs.items()}}



def card(T: pd.DataFrame, arm: str, regimes: Optional[pd.DataFrame] = None, long: Optional[Mapping] = None,
         slice_: Optional[float] = None, n_trials: Optional[Tuple[int, int]] = None,
         max_open: Optional[int] = None, bench=None, long_trades: Optional[pd.DataFrame] = None) -> dict:
    """The arm's card on the pooled history (the module docstring): ``overall``, ``halves``, ``loyo`` (the average
    trade and the return per day with each year left out), ``by_year``, ``by_set``, ``context`` (trend, vol, cells),
    ``alone`` (its live slice), ``bench`` (a LONG arm's bar: over the pool, each half's span, each year left out),
    ``longest`` (with a carried LONG history of yearly summaries, or ``long_trades``: its trades before the pool, a
    table like ``T``'s), ``dsr`` (with ``n_trials``). ``bench``: the bar's daily cache symbol (or a series of daily
    returns in %) — default the live arm's (`BENCH`), none for a short arm."""
    from src.backtest import reliability as REL
    t = T[T.book == arm].sort_values("entry_day").reset_index(drop=True)
    daily, bname = _bench_of(arm, bench)
    rec: Dict[str, object] = {"arm": arm, "side": REL.SIDE.get(arm), "spec": REL.SPEC.get(arm),
                              "span": [POOL_LO, POOL_HI], "overall": quality(t)}
    if not len(t):
        return rec
    h = len(t) // 2
    a, b = t.iloc[:h], t.iloc[h:]
    rec["halves"] = {"first": quality(a, boot=False), "second": quality(b, boot=False),
                     "split": str(pd.Timestamp(b.entry_day.iloc[0]).date()) if len(b) else None}
    loyo = {}
    for y in sorted(t.year.unique()):
        r = t[t.year != y]
        if len(r):
            loyo[str(y)] = {"avg_trade": float(r.net.mean()), "rpd": per_day(float(r.net.mean()), float(r.days.mean()))}
    rec["loyo"] = loyo
    if daily is not None:
        split = rec["halves"]["split"]
        rec["bench"] = {"spec": bname, "rpd": bench_rpd(daily, POOL_LO, POOL_HI),
                        "halves": {"first": bench_rpd(daily, POOL_LO, pd.Timestamp(split) - pd.Timedelta(days=1)),
                                   "second": bench_rpd(daily, split, POOL_HI)} if split else {},
                        "loyo": {y: bench_rpd(daily, POOL_LO, POOL_HI, [int(y)]) for y in loyo}}
    rec["by_year"] = {str(y): REL.stats(g) for y, g in t.groupby("year")}
    rec["by_set"] = {s: REL.stats(g) for s, g in t.groupby("set")}
    if regimes is not None:
        r = REL.regime_of(pd.DatetimeIndex(t.entry_day), regimes)
        tr, vo = r.trend.to_numpy(object), r.vol.to_numpy(object)
        ctx: Dict[str, dict] = {}
        for name, key in (("trend", tr), ("vol", vo),
                          ("cell", np.array([f"{x}/{y}" if x is not None and y is not None else None
                                             for x, y in zip(tr, vo)], object))):
            out = {}
            for k in sorted({k for k in key if k is not None}):
                q = quality(t[key == k])
                q["label"] = _cell_label(q)
                out[str(k)] = q
            ctx[name] = out
        rec["context"] = ctx
    if slice_:
        rec["alone"] = alone(t, slice_, POOL_LO, POOL_HI, max_open=max_open)
    if long_trades is not None and len(long_trades):
        rec["longest"] = _longest_trades(t, long_trades, daily, bname)
    elif long:
        rec["longest"] = _longest(t, long, daily, bname)
    if n_trials:
        from src.backtest import audit as AU
        try:
            rec["dsr"] = {"low": AU.dsr_trades(t.net.to_numpy(float), int(n_trials[0])),
                          "high": AU.dsr_trades(t.net.to_numpy(float), int(n_trials[1]))}
        except Exception as e:                                        # noqa: BLE001 — a report, never a decision
            rec["dsr"] = {"error": str(e)}
    return rec


PRE_HI = (pd.Timestamp(POOL_LO) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")     # the last day before the pool


def _long_halves_loyo(per: Mapping[str, Tuple[int, float, float]], lo: str, daily: Optional[pd.Series]) -> dict:
    """The longest history's halves (before the pool / the pool) and every year left out, each a return per day
    against the bar over its own span; ``per`` = {year: (n, mean trade, mean days)}."""
    def mean_of(items):
        k = sum(n for n, _, _ in items)
        return (sum(n * mu for n, mu, _ in items) / k, sum(n * d for n, _, d in items) / k) if k else (None, None)

    def rpd_of(items):
        mu, d = mean_of(items)
        return per_day(mu, d) if mu is not None and d > 0 else None
    pre = [v for y, v in per.items() if int(y) < int(POOL_LO[:4])]
    post = [v for y, v in per.items() if int(y) >= int(POOL_LO[:4])]
    halves = {"before": {"n": sum(v[0] for v in pre), "avg_trade": mean_of(pre)[0], "rpd": rpd_of(pre),
                         "bar": _bar(daily, lo, PRE_HI)},
              "pooled": {"n": sum(v[0] for v in post), "avg_trade": mean_of(post)[0], "rpd": rpd_of(post),
                         "bar": _bar(daily, POOL_LO, POOL_HI)}}
    loyo_rpd = {y: rpd_of([v for yy, v in per.items() if yy != y]) for y in per}
    loyo_bar = {y: _bar(daily, lo, POOL_HI, [int(y)]) for y in per}
    ok_h = all(h["rpd"] is not None and h["rpd"] > h["bar"] for h in halves.values())
    ok_y = bool(loyo_rpd) and all(loyo_rpd[y] is not None and loyo_rpd[y] > loyo_bar[y] for y in loyo_rpd)
    return {"halves": halves, "loyo_rpd": loyo_rpd, "loyo_bar": loyo_bar, "halves_ok": ok_h, "loyo_ok": ok_y}


def _longest(t: pd.DataFrame, long: Mapping, daily: Optional[pd.Series] = None,
             bname: Optional[str] = None) -> Optional[dict]:
    """The pooled trades with a carried earlier history of YEARLY SUMMARIES: n, the mean, the episodes (the pooled
    months + each earlier year), its significance — the t of the mean trade against the bar's per-trade equivalent
    (the bar's return per day over the whole span, compounded over the arm's average hold), CLUSTERED with each earlier
    year and each pooled month one cluster (the larger of that standard error and the independent-trade one) —, the
    halves (before 2018 / 2018-2026) and every year left out against the bar, and ``checks`` 1-3 on it."""
    pre = _long_part(long)
    if pre is None:
        return None
    x = t.net.to_numpy(float)
    n1, m1, ss1 = len(x), float(x.mean()), float(((x - x.mean()) ** 2).sum())
    n0, m0, ss0 = pre["n"], pre["mean"], pre["ss"]
    n = n0 + n1
    m = (n0 * m0 + n1 * m1) / n
    ss = ss0 + ss1 + n0 * (m0 - m) ** 2 + n1 * (m1 - m) ** 2
    sd = math.sqrt(ss / (n - 1)) if n > 1 else float("nan")
    per = dict(pre["per_year"])
    for y, g in t.groupby("year"):
        per[str(y)] = (len(g), float(g.net.mean()), float(g.days.mean()))
    tot = sum(k * mu for k, mu, _ in per.values())
    loyo = {y: (tot - k * mu) / (n - k) for y, (k, mu, _) in per.items() if n - k > 0}
    days = (sum(k * d for k, _, d in per.values()) / n) if n else float("nan")
    lo = f"{min(pre['years'])}-01-01"
    bar = _bar(daily, lo, POOL_HI)
    bar_trade = (math.pow(1.0 + bar / 100.0, days) - 1.0) * 100.0 if days == days and days > 0 else 0.0
    res = [k * (mu - m) for k, mu, _ in pre["per_year"].values()]
    g = t.groupby("month").net.agg(["sum", "count"])
    res += list(g["sum"].to_numpy(float) - g["count"].to_numpy(float) * m)
    C = len(res)
    se_cl = math.sqrt(sum(r * r for r in res) * C / (C - 1)) / n if C > 1 else float("nan")
    se_iid = sd / math.sqrt(n) if sd == sd and n > 1 else float("nan")
    se = max(v for v in (se_cl, se_iid) if v == v) if (se_cl == se_cl or se_iid == se_iid) else float("nan")
    tt = (m - bar_trade) / se if se == se and se > 0 else None
    hl = _long_halves_loyo(per, lo, daily)
    episodes = int(t.month.nunique()) + len(pre["years"])
    checks = {"enough_trades": n >= MIN_TRADES and episodes >= MIN_EPISODES,
              "significant": tt is not None and tt >= T_SIG, "halves": hl["halves_ok"], "not_one_year": hl["loyo_ok"]}
    return {"n": n, "mean": m, "sd": sd, "t": tt, "t_trades_vs_0": m / se_iid if se_iid == se_iid and se_iid > 0 else None,
            "se": {"clustered": se_cl, "trades": se_iid}, "bar": bar, "bar_trade": bar_trade, "bench": bname,
            "span": [lo, POOL_HI], "rpd": per_day(m, days) if days == days and days > 0 else None,
            "episodes": episodes, "years_before": pre["years"], "halves": hl["halves"], "loyo": loyo,
            "loyo_rpd": hl["loyo_rpd"], "loyo_bar": hl["loyo_bar"], "checks": checks,
            "source": long.get("source", "PREREG103's LONG history (carried)")}


def _longest_trades(t: pd.DataFrame, earlier: pd.DataFrame, daily: Optional[pd.Series] = None,
                    bname: Optional[str] = None) -> Optional[dict]:
    """The pooled trades with the arm's EARLIER TRADES (a table: ``entry_day``, ``net``, ``gross``, ``days``; those
    entered before the pool): every trade one table — its quality with the month bootstrap over every month (the
    significance: the 95% lower bound above the bar over the whole span), the halves (before 2018 / 2018-2026) and every
    year left out against the bar, and ``checks`` 1-3 on it."""
    e = earlier[pd.DatetimeIndex(pd.to_datetime(earlier.entry_day)) < pd.Timestamp(POOL_LO)]
    if not len(e):
        return None
    cols = ["entry_day", "net", "gross", "days"]
    A = pd.concat([e[cols], t[cols]], ignore_index=True)
    A["entry_day"] = pd.DatetimeIndex(pd.to_datetime(A.entry_day)).normalize()
    A = A.sort_values("entry_day").reset_index(drop=True)
    A["year"] = A.entry_day.dt.year
    A["month"] = A.entry_day.dt.to_period("M").astype(str)
    q = quality(A)
    lo = str(A.entry_day.min().date())
    bar = _bar(daily, lo, POOL_HI)
    per = {str(y): (len(g), float(g.net.mean()), float(g.days.mean())) for y, g in A.groupby("year")}
    hl = _long_halves_loyo(per, lo, daily)
    checks = {"enough_trades": q["n"] >= MIN_TRADES and q["months"] >= MIN_EPISODES,
              "significant": q.get("rpd_lo95") is not None and q["rpd_lo95"] > bar,
              "halves": hl["halves_ok"], "not_one_year": hl["loyo_ok"]}
    return {"n": q["n"], "mean": q["avg_trade"], "rpd": q["return_per_day"], "rpd_lo95": q.get("rpd_lo95"),
            "rpd_hi95": q.get("rpd_hi95"), "t_month": q.get("t_month"), "gross_win_rate": q.get("gross_win_rate"),
            "bar": bar, "bench": bname, "span": [lo, POOL_HI], "episodes": q["months"],
            "years_before": sorted({y for y in per if int(y) < int(POOL_LO[:4])}), "halves": hl["halves"],
            "loyo": {y: float(A.net[A.year != int(y)].mean()) for y in per if (A.year != int(y)).any()},
            "loyo_rpd": hl["loyo_rpd"], "loyo_bar": hl["loyo_bar"], "checks": checks,
            "source": "the arm's earlier trades (a table)"}


def qualify(c: Mapping) -> dict:
    """The admission checks of the card ``c`` (the module docstring): {"verdict": "QUALIFIES" | "INSUFFICIENT" |
    "FAILS", "checks": {name: bool or None}, "basis": "pooled" | "longest" | "longest (re-judged)", "bar": the
    return per day its edge checks are read against (0 for a short arm), "why": [...]} (+ "pooled_checks" when the
    longest history decided)."""
    o = c.get("overall") or {}
    bn = c.get("bench") or {}
    bar = float(bn.get("rpd") or 0.0)
    hv, hb, lb = c.get("halves") or {}, bn.get("halves") or {}, bn.get("loyo") or {}
    al = c.get("alone") or {}
    growth = (al.get("growth") is not None and al["growth"] > 0) if al else None
    n, ep, rpd = o.get("n", 0), o.get("months", 0), o.get("return_per_day")
    pooled = {"enough_trades": n >= MIN_TRADES and ep >= MIN_EPISODES,
              "significant": o.get("rpd_lo95") is not None and o["rpd_lo95"] > bar,
              "halves": all((hv.get(h) or {}).get("return_per_day") is not None
                            and hv[h]["return_per_day"] > float(hb.get(h) or 0.0) for h in ("first", "second")),
              "not_one_year": bool(c.get("loyo")) and all(v["rpd"] is not None and v["rpd"] > float(lb.get(y) or 0.0)
                                                          for y, v in c["loyo"].items()),
              "growth": growth}
    lg = c.get("longest")
    rejudge = bool(lg) and [k for k, v in pooled.items() if v is False] == ["significant"] and rpd is not None \
        and rpd > bar
    use_long = bool(lg) and (not pooled["enough_trades"] or rejudge)
    against = f" (its bar: {bn.get('spec')}'s average day, {bar:+.3f} %/day)" if bn else ""
    if use_long:
        checks = dict(lg["checks"], growth=growth)
        n, ep = lg["n"], lg["episodes"]
        lb_ = lg.get("bar") or 0.0
        if lg.get("rpd_lo95") is not None:
            sig_txt = (f"on {n} trades from {lg['span'][0][:4]}: return per day {lg.get('rpd', float('nan')):+.3f} %/day, "
                       f"95% {lg['rpd_lo95']:+.3f}..{lg['rpd_hi95']:+.3f}")
        else:
            sig_txt = (f"t {lg['t']:.2f} (clustered, against {lg['bar_trade']:+.2f}% a trade) over {n} trades from "
                       f"{lg['span'][0][:4]}" if lg.get("t") is not None else "no t")
        against = f" (its bar over {lg['span'][0][:4]}-{POOL_HI[:4]}: {lg.get('bench')}'s average day, {lb_:+.3f} %/day)" \
            if lg.get("bench") else ""
    else:
        checks = pooled
        sig_txt = (f"return per day {o.get('return_per_day', float('nan')):+.3f} %/day, 95% "
                   f"{o.get('rpd_lo95', float('nan')):+.3f}..{o.get('rpd_hi95', float('nan')):+.3f}")
    why: List[str] = []
    if not checks["enough_trades"]:
        why.append(f"{n} trades in {ep} episodes (needs {MIN_TRADES} in {MIN_EPISODES})")
    if not checks["significant"]:
        why.append(f"not significant: {sig_txt}{against}")
    if not checks["halves"]:
        why.append("the edge is not in both halves of its history" + (" (against its bar)" if against else ""))
    if not checks["not_one_year"]:
        why.append("one calendar year carries it" + (" (against its bar)" if against else ""))
    if checks["growth"] is False:
        why.append(f"the arm alone at its live slice does not grow ({al.get('growth'):+.1f} %/yr)")
    failed = [k for k, v in checks.items() if v is False]
    verdict = "QUALIFIES" if not failed else ("INSUFFICIENT" if failed == ["enough_trades"] else "FAILS")
    out = {"verdict": verdict, "checks": checks,
           "basis": ("longest (re-judged)" if rejudge else "longest") if use_long else "pooled",
           "bar": bar, "bar_spec": bn.get("spec"), "why": why}
    if use_long:
        out["pooled_checks"] = pooled
    return out


# ── filters and choices ──────────────────────────────────────────────────────

def filter_test(t: pd.DataFrame, removed: np.ndarray) -> dict:
    """A context filter on an arm's trades ``t``: ``removed`` (a boolean mask) the trades it takes out. JUSTIFIED by
    its trades when they number at least `MIN_CELL` in at least `MIN_CELL_MONTHS` months and their return per day's
    interval lies below 0. Reports the kept, the removed and every trade's quality and the change in return per day."""
    removed = np.asarray(removed, bool)
    k, r = t[~removed], t[removed]
    qk, qr, qa = quality(k), quality(r), quality(t, boot=False)
    enough = qr.get("n", 0) >= MIN_CELL and qr.get("months", 0) >= MIN_CELL_MONTHS
    losing = qr.get("rpd_hi95") is not None and qr["rpd_hi95"] < 0
    return {"kept": qk, "removed": qr, "all": qa, "justified_by_trades": bool(enough and losing),
            "enough_removed": bool(enough), "removed_lose": bool(losing),
            "rpd_change": (qk.get("return_per_day", float("nan")) - qa.get("return_per_day", float("nan")))
            if qk.get("n") and qa.get("n") else None}


def walk_forward(t: pd.DataFrame, choose: Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]],
                 first_year: int = FIRST_WF_YEAR) -> dict:
    """A choice measured OUT OF TIME: for each year Y from ``first_year``, ``choose(the trades before Y)`` returns a
    rule (a function of a trade table giving the boolean mask of the trades it KEEPS), applied to Y's trades alone.
    Returns the out-of-time trades' quality, the in-sample one (the rule chosen on every year, applied to every year),
    the decay (out-of-time return per day / in-sample) and each year's choice (the rule's ``label`` when it has one)."""
    oos, chosen = [], {}
    for y in sorted(int(v) for v in t.year.unique()):
        if y < first_year:
            continue
        rule = choose(t[t.year < y].copy())
        ty = t[t.year == y]
        oos.append(ty[np.asarray(rule(ty), bool)])
        chosen[str(y)] = getattr(rule, "label", None)
    O = pd.concat(oos, ignore_index=True) if oos else t.iloc[:0]
    full = choose(t.copy())
    I = t[np.asarray(full(t), bool)]
    qo, qi = quality(O), quality(I)
    ro, ri = qo.get("return_per_day"), qi.get("return_per_day")
    return {"out_of_time": qo, "in_sample": qi, "decay": (ro / ri) if (ro is not None and ri) else None,
            "first_year": first_year, "chosen": chosen, "in_sample_choice": getattr(full, "label", None)}


# ── context activation (PREREG108) ───────────────────────────────────────────
# user 2026-10-14: "a model should understand the context and be activated if historically it performs well in it. If
# it is activated, then it can be used to trade." Each arm's ACTIVATION MAP: per market context (`context.CELLS`, read
# at the close before the decision), active or not, decided at each month's first day from the arm's trades CLOSED
# before it — its record in every context, the deactivated ones included (their shadow record).

ACTIVATION_RULES = ("shrunk", "proven", "optout")      # PREREG108's K1 (primary), K2, K3
ACTIVATION_PRIOR = 30.0          # K1: the arm's own average weighs as this many trades
ACTIVATION_BURN_DAYS = 365       # until its closed history spans a year, an arm is active everywhere
ACTIVATION_T = 1.96              # K2 / K3: the month-clustered t a cell must reach


def cell_stats(t: pd.DataFrame) -> dict:
    """Trades ``t`` (``net`` %, ``w`` weights — 1 / the draws when several draws are pooled —, ``dmonth`` the decision
    month): the weighted count n, the average net trade, the distinct months and the month-clustered t of the average."""
    if not len(t):
        return {"n": 0.0, "mean": None, "months": 0, "t": None}
    w = t.w.to_numpy(float) if "w" in t else np.ones(len(t))
    x = t.net.to_numpy(float)
    n = float(w.sum())
    m = float((w * x).sum() / n)
    g = pd.DataFrame({"s": w * x, "c": w, "k": t.dmonth.to_numpy()}).groupby("k").sum()
    M = len(g)
    tt = None
    if M > 1:
        r = g.s.to_numpy(float) - g.c.to_numpy(float) * m
        se = math.sqrt(float((r ** 2).sum()) * M / (M - 1)) / n
        tt = m / se if se > 0 else (math.copysign(math.inf, m) if m != 0 else 0.0)
    return {"n": n, "mean": m, "months": int(M), "t": tt}


def activation_map(t: pd.DataFrame, asof, rule: str = "shrunk") -> dict:
    """The arm's map at ``asof`` from its trades ``t`` (``cell``, ``net``, ``exit``, ``dmonth``, optional ``w``) CLOSED
    before it: {"burn_in": bool, "active": [cells], "cells": {cell: cell_stats + "active"}}. ``rule``: "shrunk" (K1:
    the cell's average shrunk toward the arm's own with a prior of `ACTIVATION_PRIOR` trades, active above 0), "proven"
    (K2: active only with `MIN_CELL` trades in `MIN_CELL_MONTHS` months and t >= `ACTIVATION_T`), "optout" (K3: inactive
    only with as many trades and t <= -`ACTIVATION_T`)."""
    from src.backtest import context as CX
    if rule not in ACTIVATION_RULES:
        raise ValueError(f"rule must be one of {ACTIVATION_RULES}, not {rule!r}")
    asof = pd.Timestamp(asof)
    c = t[pd.DatetimeIndex(t.exit) < asof]
    if not len(c) or (asof - pd.Timestamp(c.exit.min())).days < ACTIVATION_BURN_DAYS:
        return {"burn_in": True, "active": list(CX.CELLS), "cells": {}}
    allm = cell_stats(c)["mean"]
    cells = {}
    for cell in CX.CELLS:
        st = cell_stats(c[c.cell == cell])
        enough = st["n"] >= MIN_CELL and st["months"] >= MIN_CELL_MONTHS and st["t"] is not None
        if rule == "shrunk":
            sh = (st["n"] * (st["mean"] or 0.0) + ACTIVATION_PRIOR * allm) / (st["n"] + ACTIVATION_PRIOR)
            st["shrunk"], on = sh, sh > 0
        elif rule == "proven":
            on = bool(enough and st["t"] >= ACTIVATION_T)
        else:
            on = not (enough and st["t"] <= -ACTIVATION_T)
        st["active"] = bool(on)
        cells[cell] = st
    return {"burn_in": False, "active": [k for k, v in cells.items() if v["active"]], "cells": cells}


def schedule(t: pd.DataFrame, rule: str = "shrunk", months: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """The arm's maps month by month (``months`` "YYYY-MM", default every month of the pool): each from the trades
    closed before the month's first day."""
    months = list(months) if months is not None else [str(m) for m in pd.period_range(POOL_LO[:7], POOL_HI[:7], freq="M")]
    return {m: activation_map(t, pd.Period(m, "M").to_timestamp(how="start"), rule) for m in months}


def activated(t: pd.DataFrame, sched: Mapping[str, dict]) -> np.ndarray:
    """Which trades of ``t`` an arm takes under ``sched``: its decision month's map holds its cell (a month without a
    map, or in its burn-in, takes every trade)."""
    out = np.ones(len(t), bool)
    for i, (m, cell) in enumerate(zip(t.dmonth.to_numpy(), t.cell.to_numpy(object))):
        mp = sched.get(str(m))
        if mp is not None and not mp.get("burn_in") and cell not in set(mp["active"]):
            out[i] = False
    return out


# ── the combination ──────────────────────────────────────────────────────────

def load_runs(folder: Path, label: str, sets: Sequence[str] = POOL) -> Dict[str, Dict[int, dict]]:
    """A study's account draws {set: {seed: record}} from ``folder``'s ``<set>_<label>_s<seed>.pkl`` files."""
    out: Dict[str, Dict[int, dict]] = {}
    for p in sorted(Path(folder).glob(f"*_{label}_s*.pkl")):
        m = RUN_FILE.match(p.name)
        if not m or m.group(1) not in sets or m.group("label") != label:
            continue
        out.setdefault(m.group(1), {})[int(m.group("seed"))] = pickle.load(open(p, "rb"))
    return out


def nav_of(rec: Mapping, lo: str, start: float = START) -> Tuple[pd.Series, str]:
    """A draw's equity as a NAV (1.0 at its start): its daily NAV, else its positions' P&L realized at their exits
    (flagged "realized")."""
    nav = rec.get("nav")
    if nav:
        s = pd.Series(np.asarray(nav["values"], float), index=pd.DatetimeIndex(pd.to_datetime(nav["days"])))
        return s / float(s.iloc[0]), "nav"
    P = pd.DataFrame(rec.get("pos") or [])
    if not len(P):
        return pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp(lo)])), "realized"
    day = pd.to_datetime(P.exit_ns, unit="ns", utc=True).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize()
    eq = start + P.pnl.astype(float).groupby(day).sum().sort_index().cumsum()
    s = pd.concat([pd.Series([start], index=[pd.Timestamp(lo)]), eq]).groupby(level=0).last() / start
    return s, "realized"


def _months(s: pd.Series) -> pd.Series:
    """Monthly returns of a NAV (the first month from its first value), keyed by period."""
    me = s.groupby(s.index.to_period("M")).last()
    prev = pd.concat([pd.Series([float(s.iloc[0])], index=[me.index[0] - 1]), me.iloc[:-1]])
    return pd.Series(me.to_numpy() / prev.to_numpy() - 1.0, index=me.index)


def _chain(parts: Sequence[pd.Series]) -> pd.Series:
    """Monthly returns of consecutive sets chained: a month two sets share (a set boundary) compounds both parts."""
    out: Dict[pd.Period, float] = {}
    for p in parts:
        for k, v in p.items():
            out[k] = (1.0 + out.get(k, 0.0)) * (1.0 + float(v)) - 1.0
    return pd.Series(out).sort_index()


def _profile(m: pd.Series) -> dict:
    """One draw's monthly profile: the share of positive months, the worst / median month, the worst twelve months,
    the longest time under water (months), every calendar year's growth (%)."""
    eq = (1.0 + m).cumprod()
    peak = eq.cummax()
    under, run = 0, 0
    for v, p in zip(eq.to_numpy(), peak.to_numpy()):
        run = run + 1 if v < p - 1e-12 else 0
        under = max(under, run)
    r12 = (1.0 + m).rolling(12).apply(np.prod, raw=True) - 1.0
    years = {str(y): float((1.0 + g).prod() - 1.0) * 100.0 for y, g in m.groupby(m.index.year)}
    return {"positive_share": float((m > 0).mean()), "worst_month": float(m.min()) * 100.0,
            "median_month": float(m.median()) * 100.0, "worst_12m": float(r12.min()) * 100.0 if r12.notna().any() else None,
            "longest_under_water_months": int(under), "years": years}


def _yearly(path: Path) -> Dict[str, float]:
    from src.backtest import reliability as REL
    try:
        c = REL.spy_daily(path)["Close"]
    except Exception:                                                  # noqa: BLE001 — a reference column only
        return {}
    c = c[(c.index >= pd.Timestamp(POOL_LO) - pd.Timedelta(days=10)) & (c.index <= pd.Timestamp(POOL_HI))]
    out = {}
    for y in range(int(POOL_LO[:4]), int(POOL_HI[:4]) + 1):
        a = c[c.index < pd.Timestamp(f"{y}-01-01")]
        b = c[c.index <= pd.Timestamp(f"{y}-12-31")]
        if len(a) and len(b):
            out[str(y)] = float(b.iloc[-1] / a.iloc[-1] - 1.0) * 100.0
    return out


def combination(runs: Mapping[str, Mapping[int, dict]], regimes: Optional[pd.DataFrame] = None) -> dict:
    """The live configuration on the pooled history (the module docstring) from its draws ``runs`` {set: {seed:
    record}}: the seeds every set has, each chained through the sets."""
    w = periods.pool_weights(POOL)
    seeds = sorted(set.intersection(*[set(runs.get(s, {})) for s in POOL])) if all(runs.get(s) for s in POOL) else []
    if not seeds:
        return {"seeds": 0, "note": "no seed is drawn in every pooled set"}
    lg, dd, calls_any, prof, month_tab, basis = [], [], 0, [], [], set()
    calls_pairs = {s: sum(1 for k in seeds if runs[s][k].get("calls")) for s in POOL}
    for k in seeds:
        lg.append(sum(w[s] * 100.0 * math.log1p(float(runs[s][k]["growth"])) for s in POOL))
        dd.append(max(float(runs[s][k]["maxdd"]) for s in POOL))
        calls_any += int(any(runs[s][k].get("calls") for s in POOL))
        parts = []
        for s in POOL:
            nav, b = nav_of(runs[s][k], periods.get(s).lo)
            basis.add(b)
            parts.append(_months(nav))
        m = _chain(parts)
        prof.append(_profile(m))
        month_tab.append(m)
    lg = np.array(lg)
    M = pd.concat(month_tab, axis=1)                                    # months x seeds
    years = sorted({y for p in prof for y in p["years"]})
    out = {"seeds": len(seeds), "sets": list(POOL), "weights": w, "basis": sorted(basis),
           "growth": {"log_points": float(lg.mean()), "se": float(lg.std(ddof=1) / math.sqrt(len(lg))) if len(lg) > 1 else None,
                      "pct_per_year": float(math.expm1(lg.mean() / 100.0)) * 100.0,
                      "min_pct": float(math.expm1(lg.min() / 100.0)) * 100.0,
                      "max_pct": float(math.expm1(lg.max() / 100.0)) * 100.0},
           "maxdd": {"mean": float(np.mean(dd)), "worst": float(np.max(dd))},
           "margin_calls": {"draws_with_one": calls_any, "pairs": calls_pairs},
           "months": {k: float(np.mean([p[k] for p in prof if p[k] is not None]))
                      for k in ("positive_share", "worst_month", "median_month", "worst_12m",
                                "longest_under_water_months")},
           "worst_month_any_draw": float(min(p["worst_month"] for p in prof)),
           "years": {y: {"mean": float(np.mean([p["years"][y] for p in prof if y in p["years"]])),
                         "min": float(np.min([p["years"][y] for p in prof if y in p["years"]])),
                         "max": float(np.max([p["years"][y] for p in prof if y in p["years"]]))} for y in years},
           "spy": _yearly(Path("cache") / "ohlcv" / "SPY.json"), "qqq": _yearly(Path("cache") / "ohlcv" / "QQQ.json")}
    if regimes is not None:
        out["coverage"] = coverage(M, runs, seeds, regimes)
    return out


def coverage(M: pd.DataFrame, runs: Mapping[str, Mapping[int, dict]], seeds: Sequence[int],
             regimes: pd.DataFrame) -> dict:
    """The combination's months (``M``: months x seeds, monthly returns) grouped by the market context at each month's
    first session (`reliability.regime_of`: SPY's trend; its realized-volatility tercile — a label): per context the
    months, the average month, the share of positive months, the growth a year in those months, and each arm's share
    of their P&L (its positions' P&L realized in those months, over every draw)."""
    from src.backtest import reliability as REL
    first = [p.to_timestamp(how="start") for p in M.index]
    r = REL.regime_of(first, regimes)
    keys = {"trend": r.trend.to_numpy(object), "vol": r.vol.to_numpy(object),
            "cell": np.array([f"{a}/{b}" if a is not None and b is not None else None for a, b in zip(r.trend, r.vol)],
                             object)}
    pnl = {}                                                           # (month, arm) -> P&L summed over the draws
    for s in POOL:
        for k in seeds:
            for p in runs[s][k].get("pos") or []:
                if not p.get("exit_ns"):
                    continue
                mo = pd.Timestamp(int(p["exit_ns"]), unit="ns", tz="UTC").tz_convert(ET).tz_localize(None).to_period("M")
                key = (mo, str(p["strategy"]))
                pnl[key] = pnl.get(key, 0.0) + float(p.get("pnl", 0.0))
    arms = sorted({a for _, a in pnl})
    out = {}
    for name, key in keys.items():
        grp = {}
        for g in sorted({k for k in key if k is not None}):
            sel = key == g
            sub = M[sel]
            if not len(sub):
                continue
            lr = np.log1p(sub.to_numpy(float))
            months = set(sub.index)
            arm_p = {a: sum(v for (mo, aa), v in pnl.items() if aa == a and mo in months) for a in arms}
            tot = sum(abs(v) for v in arm_p.values())
            grp[str(g)] = {"months": int(len(sub)), "avg_month": float(sub.to_numpy().mean()) * 100.0,
                           "positive_share": float((sub.to_numpy() > 0).mean()),
                           "growth_per_year": float(math.expm1(12.0 * np.nanmean(lr))) * 100.0,
                           "arm_pnl_share": {a: (arm_p[a] / tot if tot else None) for a in arms}}
        out[name] = grp
    return out


# ── the build ────────────────────────────────────────────────────────────────

def _json_ready(o):
    if isinstance(o, dict):
        return {str(k): _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def build(day: Optional[date] = None, cards: Optional[Mapping[str, Path]] = None, runs_dir: Optional[Path] = None,
          label: Optional[str] = None, long: Optional[Mapping[str, Mapping]] = None) -> dict:
    """Every live arm's pooled card and qualification and the combination's pooled profile (the module docstring)."""
    from config.settings import settings
    from src.backtest import audit as AU
    from src.backtest import reliability as REL
    from src.backtest import trials as TR
    reg = REL.regime_table()
    T = load_trades(cards)
    if long is None:
        long = {}
        try:
            base = json.loads(AU.baseline_json().read_text(encoding="utf-8"))
            for arm in ("dip", "crash"):
                lg = (base.get("table1", {}).get(arm) or {}).get("LONG")
                if lg:
                    long[arm] = dict(lg, source=f"PREREG103 table 1 LONG ({AU.baseline_json()})")
        except OSError:
            pass
    sl = live_slices()
    arms = {}
    for arm in ARMS:
        if not (T.book == arm).any():
            continue
        try:
            c = TR.count(arm)
            nt = (max(1, int(c["low"])), max(1, int(c["high"])))
        except Exception:                                              # noqa: BLE001 — the ledger is optional
            nt = None
        cd = card(T, arm, reg, long.get(arm), sl.get(arm), nt)
        cd["qualify"] = qualify(cd)
        arms[arm] = cd
    rd = Path(runs_dir) if runs_dir is not None else Path(settings.backtest_research_dir) / LIVE_RUNS[0]
    lab = label or LIVE_RUNS[1]
    runs = load_runs(rd, lab)
    comb = combination(runs, reg) if runs else {"seeds": 0, "note": f"no draws {lab} in {rd}"}
    srcs = {s: str(p) for s, p in (dict(cards) if cards is not None else AU.card_dirs()).items() if s in POOL}
    return {"built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "registration": "user directive 2026-10-14 (the pooled evaluation)",
            "pool": {"sets": list(POOL), "span": [POOL_LO, POOL_HI], "weights": periods.pool_weights(POOL)},
            "rules": {"min_trades": MIN_TRADES, "min_episodes": MIN_EPISODES, "min_cell": MIN_CELL,
                      "min_cell_months": MIN_CELL_MONTHS, "bootstrap": {"resamples": B_BOOT, "seed": SEED,
                                                                         "over": "entry months"},
                      "bar": {"long": "the same exposure's average day (user directive 2026-10-15)", "short": 0.0,
                              "bench": dict(BENCH)},
                      "rejudge": "an arm failing only significance, its return per day above its bar, is judged on "
                                 "its longest history (user directive 2026-10-15)", "t_sig_summaries": T_SIG},
            "inputs": {"cards": srcs, "runs": f"{rd} / {lab}", "long": {a: v.get("source") for a, v in long.items()},
                       "bench_history": str(AU.bench_history_json())},
            "arms": arms, "combination": comb}


def out_path(day: Optional[date] = None, bank_dir: Optional[Path] = None) -> Path:
    return Path(bank_dir or BANK_DIR) / f"{(day or datetime.now().date()).isoformat()}.json"


def lines(b: Mapping) -> List[str]:
    """The build in a few lines (the CLI's print)."""
    out = [f"POOLED {'+'.join(b['pool']['sets'])} {b['pool']['span'][0]} .. {b['pool']['span'][1]}"]
    for arm, c in b["arms"].items():
        o, q = c["overall"], c["qualify"]
        al = c.get("alone") or {}
        bar = f"  bar {q.get('bar_spec')} {q.get('bar', 0.0):+.3f}" if q.get("bar_spec") else ""
        out.append(f"  {arm:6s} {q['verdict']:12s} n {o.get('n', 0):5d} months {o.get('months', 0):3d}  "
                   f"rpd {o.get('return_per_day', float('nan')):+.3f} [{o.get('rpd_lo95', float('nan')):+.3f}, "
                   f"{o.get('rpd_hi95', float('nan')):+.3f}]{bar}  gross win "
                   f"{o.get('gross_win_rate', float('nan')):.0%}  alone {al.get('growth', float('nan')):+.1f} %/yr "
                   f"dd {al.get('maxdd', float('nan')):.0f}%"
                   + (f"  [{q['basis']}]" if q.get("basis") != "pooled" else "")
                   + (f"  -- {'; '.join(q['why'])}" if q["why"] else ""))
        for name in ("cell",):
            for k, v in ((c.get("context") or {}).get(name) or {}).items():
                if v.get("label") in ("EARNS", "LOSES"):
                    out.append(f"         {v['label']:5s} in {k}: n {v['n']} rpd {v['return_per_day']:+.3f}")
    cb = b["combination"]
    if cb.get("seeds"):
        g = cb["growth"]
        out.append(f"  COMBINATION ({cb['seeds']} draws, {'/'.join(cb['basis'])}): {g['pct_per_year']:+.1f} %/yr "
                   f"({g['min_pct']:+.1f}..{g['max_pct']:+.1f}), worst dd {cb['maxdd']['mean']:.1f} (worst "
                   f"{cb['maxdd']['worst']:.1f}), draws with a margin call {cb['margin_calls']['draws_with_one']}, "
                   f"positive months {cb['months']['positive_share']:.0%}, worst month "
                   f"{cb['months']['worst_month']:+.1f}%, worst 12 months {cb['months']['worst_12m']:+.1f}%")
        out.append("  years: " + "  ".join(f"{y} {v['mean']:+.0f}% (SPY {cb['spy'].get(y, float('nan')):+.0f})"
                                         for y, v in cb["years"].items()))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="the bank of models on the pooled history (A0 + A + B)")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--runs", help="the folder of the combination's draws (default: the live configuration's)")
    ap.add_argument("--label", help="the draws' label (default: %s)" % LIVE_RUNS[1])
    a = ap.parse_args(argv)
    b = build(runs_dir=Path(a.runs) if a.runs else None, label=a.label)
    for ln in lines(b):
        print(ln)
    if not a.no_write:
        p = out_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_json_ready(b), indent=1), encoding="utf-8")
        print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
