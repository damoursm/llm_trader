"""THE TEST SETS BY BORROW DATA (user directive 2026-10-10: "rename or rework the test sets so that we evaluate all the
years we have the borrow fees data vs those we don't").

A short's two borrow inputs — can IBKR lend it, and at what fee — exist for some years and not others
(`src/data/deep/borrow_history.py`):

* IBKR's short-stock file (shares available AND the fee, every name, the ones delisted since too): the community
  archive 2017-10-09 .. 2024-06-23 (its last file stays in force through 2024-06-24), then our own archive from
  2026-09-25 (every tick, `data/ibkr_borrow/`).
* Between them no file exists anywhere: whether IBKR could lend a name is unknown — the simulator DRAWS it from a model
  fitted on the archive's years (`mimic.lend`) — and the fee comes from IBKR's API daily bars, which cover only the
  names still listed today (the median fee otherwise).

So every backtest of a short book, or of the account that holds them, is reported on these sets SEPARATELY, never
pooled (the dip book alone needs no borrow; inside the account it is judged with it):

* ``A`` — with IBKR's file: decisions 2021-02-01 (the deep store's first session) .. 2024-06-24.
* ``B`` — without it (lendability modeled): 2024-06-25 .. 2026-09-24.
* ``C`` — our own archive, the live forward test: 2026-09-25 onward (no backtest data yet).

A set's account starts with $10,000 at its first session and takes only the decisions inside it; a calendar year that
straddles a boundary (2024) is reported in two parts. `LEND_ARCHIVE_END` / `OWN_ARCHIVE_FROM` are the simulator's
own switch dates (`mimic.lend_all`), so the sets and the lendability model can never disagree.

SET A0 (PREREG101, registered 2026-10-14): decisions 2018-01-02 .. 2021-01-29, BEFORE the deep 30-minute store, with
IBKR's archived file (the community archive starts 2017-10-09) — built exactly as set A on its own data
(`src/backtest/a0.py`: the stores rebuilt by the same code for 2017-06 .. 2021-06). It is an EXTRA set, asked for by
name: `get("A0")`, `set_of(day, extra=True)`, `windows(end, extra=True)`, `year_parts(lo, hi, extra=True)`. Without
``extra`` every function reads A / B / C only, exactly as before (studies running on A and B are unchanged); A0 is
reported beside A and B, never pooled.

SET D — THE FORWARD HOLDOUT (PREREG106, registered 2026-10-14 13:09 ET): decisions from 2026-09-25 (the first day of
our own archive of IBKR's file, `OWN_ARCHIVE_FROM`) to the LATEST COMPLETED SESSION, computed when asked
(`latest_completed_session`: the last NYSE session whose 16:00 ET close has passed), so it grows every day. The
simulator's lendability there is our own archive — `mimic.lend_all` switches to `borrow_history.lendable_at` from
`OWN_ARCHIVE_FROM`, the set-A-like real path — and no study may tune on it. It covers C's days: C is the LIVE record
of those days, D the SIMULATION of them. So D is asked for by name (`get("D")`) or with ``forward=True``, which puts
D in C's place: `set_of(day, forward=True)`, `windows(end, forward=True)`, `year_parts(lo, hi, forward=True)`
(combinable with ``extra``; ``now`` fixes the clock). Without ``forward`` every function reads exactly as before.

SETS A2 / B2 (PREREG105, registered 2026-10-14 13:07 ET): sets A and B's decisions on stores REBUILT with set A0's
construction (the universe screened day by day by the symbol traded that day, split factors measured, warm-up from
2017-06; `src/backtest/rebuild.py`). Asked for by name only (`get("A2")`, `get("B2")`: the spans of A and B); no other
function reads them.

THE POOLED SET (user directive 2026-10-14: "Instead of checking for the three different sets (A0, A, and B), check for
the full dataset (A0+A+B)"): `POOLED` = A0 + A + B, decisions 2018-01-02 .. 2026-09-24 — ONE history on which every
model and the combination are judged (`src/backtest/bank.py`, `add_rule.pooled_verdict`). The three sets remain the
simulator's construction (each its own borrow data: IBKR's file on A0 and A, the lendability model on B) and are
still reported, as diagnostics beside the calendar years and the market contexts. Until one continuous run spans
them (PREREG105's rebuilt stores), the pooled numbers CHAIN the three sets' accounts, each from $10,000 at its first
session: a pooled rate is the years-weighted mean of the sets' log rates (`pool_weights`: each set's span over the
pool's), a pooled worst drawdown the worst of the sets'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

LEND_ARCHIVE_END = "2024-06-24"         # IBKR's archived short-stock file: its last file is in force through this day
OWN_ARCHIVE_FROM = "2026-09-25"         # our own archive of IBKR's file starts here
DATA_START = "2021-02-01"               # the deep 30-minute store's first session (the short books' backtests)
LEND_ARCHIVE_START = "2017-10-09"       # IBKR's archived short-stock file: its first file (the community archive)
A0_LO, A0_HI = "2018-01-02", "2021-01-29"   # PREREG101's set A0: the decisions before the deep store, IBKR's file in force
FORWARD_FROM = OWN_ARCHIVE_FROM         # PREREG106's set D starts where the simulator reads our own archive
ET = "America/New_York"


@dataclass(frozen=True)
class BorrowSet:
    key: str
    name: str
    lo: str                             # first decision session (inclusive)
    hi: Optional[str]                   # last decision session (inclusive); None = open-ended
    borrow: str                         # what the simulator knows about borrowing there


SETS: Tuple[BorrowSet, ...] = (
    BorrowSet("A", "with IBKR's borrow file", DATA_START, LEND_ARCHIVE_END,
              "IBKR's short-stock file: shares available and the fee for every name"),
    BorrowSet("B", "without a borrow file", (pd.Timestamp(LEND_ARCHIVE_END) + pd.Timedelta(days=1)).date().isoformat(),
              (pd.Timestamp(OWN_ARCHIVE_FROM) - pd.Timedelta(days=1)).date().isoformat(),
              "lendability drawn from a model of the archive years; IBKR's API fee for the names still listed"),
    BorrowSet("C", "our own archive (live)", OWN_ARCHIVE_FROM, None,
              "our own archive of IBKR's file: shares available and the fee"),
)
# PREREG101: the extra set before the deep store (asked for by name or with ``extra=True``; never in `SETS`)
EXTRA_SETS: Tuple[BorrowSet, ...] = (
    BorrowSet("A0", "with IBKR's borrow file, before the deep store", A0_LO, A0_HI,
              "IBKR's short-stock file (the community archive from 2017-10-09): shares available and the fee for "
              "every name, the delisted ones too"),
)
# PREREG105: sets A and B REBUILT with set A0's construction — the same decisions on their own stores
# (`src/backtest/rebuild.py`); asked for by name only (`get("A2")`, `get("B2")`): never in `SETS` / `EXTRA_SETS`, so
# no function above reads them and every existing key and default is unchanged
REBUILT_SETS: Tuple[BorrowSet, ...] = (
    BorrowSet("A2", "set A rebuilt with set A0's construction", SETS[0].lo, SETS[0].hi, SETS[0].borrow),
    BorrowSet("B2", "set B rebuilt with set A0's construction", SETS[1].lo, SETS[1].hi, SETS[1].borrow),
)
# the user's pooled evaluation set (2026-10-14): A0 + A + B, one history 2018-01-02 .. 2026-09-24
POOLED: Tuple[str, ...] = ("A0", "A", "B")


def latest_completed_session(now=None) -> str:
    """The last NYSE session whose regular close (16:00 ET) has passed at ``now`` (default: the clock; a naive ``now``
    is ET wall time) — the NYSE holiday table of `src.performance.market_calendar` (read only). A half-day counts as
    completed at 16:00 too (the table does not track early closes): a few hours late, never early."""
    from src.performance import market_calendar as MC
    t = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now)
    t = t.tz_localize(ET) if t.tzinfo is None else t.tz_convert(ET)
    d = t.date()
    if MC.is_market_day(d) and t.time() >= MC.MARKET_CLOSE_LOCAL:
        return d.isoformat()
    return MC.previous_market_day(d).isoformat()


def set_d(now=None) -> BorrowSet:
    """PREREG106's set D at ``now``: decisions from `FORWARD_FROM` through the latest completed session (its end is
    before its start until 2026-09-25's close: an empty set)."""
    return BorrowSet("D", "the forward holdout (simulated on our own archive)", FORWARD_FROM,
                     latest_completed_session(now),
                     "our own archive of IBKR's file (every tick from 2026-09-25): shares available and the fee for "
                     "every name it lists — the real path, as set A's; no study may tune on it")


def _sets(extra: bool, forward: bool = False, now=None) -> Tuple[BorrowSet, ...]:
    """The sets in time order: A / B / C, with ``extra`` A0 first, with ``forward`` D in C's place."""
    base = (tuple(s for s in SETS if s.key != "C") + (set_d(now),)) if forward else SETS
    return (EXTRA_SETS + base) if extra else base


def get(key: str, now=None) -> BorrowSet:
    """A set by its key; set D's end is the latest completed session at ``now`` (default: the clock)."""
    if key == "D":
        return set_d(now)
    for s in SETS + EXTRA_SETS + REBUILT_SETS:
        if s.key == key:
            return s
    raise KeyError(f"no borrow-data test set {key!r} (A, B, C; A0, D; A2, B2)")


def years(key: str, now=None) -> float:
    """A set's span in years: (its last decision session - its first) / 365.25 — the weight of its rates in the pool
    (`audit.set_years`' construction; set D's end is the latest completed session at ``now``)."""
    S = get(key, now=now)
    if S.hi is None:
        raise ValueError(f"set {key!r} is open-ended: it has no span")
    return (pd.Timestamp(S.hi) - pd.Timestamp(S.lo)).days / 365.25


def pool_weights(keys: Tuple[str, ...] = POOLED, now=None) -> dict:
    """Each set's share of the pooled history's years ({key: weight}, summing to 1): a pooled rate is the weighted
    mean of the sets' log rates — the growth of one account chained through them."""
    y = {k: years(k, now=now) for k in keys}
    tot = sum(y.values())
    return {k: v / tot for k, v in y.items()}


def pool_span(keys: Tuple[str, ...] = POOLED, now=None) -> Tuple[str, str]:
    """The pooled history's first and last decision sessions."""
    S = [get(k, now=now) for k in keys]
    return min(s.lo for s in S), max(s.hi for s in S if s.hi is not None)


def _key_in(d: pd.Timestamp, sets: Tuple[BorrowSet, ...]) -> Optional[str]:
    for s in sets:
        if d >= pd.Timestamp(s.lo) and (s.hi is None or d <= pd.Timestamp(s.hi)):
            return s.key
    return None


def set_of(day, extra: bool = False, *, forward: bool = False, now=None) -> Optional[str]:
    """The set a decision on ``day`` belongs to (None before the backtests' data); ``extra``: A0 too; ``forward``: D
    in C's place (None after D's last completed session)."""
    return _key_in(pd.Timestamp(day).normalize(), _sets(extra, forward, now))


def windows(data_end, extra: bool = False, *, forward: bool = False, now=None) -> List[Tuple[str, str, str]]:
    """(key, first session, last session) of each set the data reaches, its end cut at ``data_end`` (``extra``: A0
    first; ``forward``: D in C's place, cut at its last completed session too)."""
    end = pd.Timestamp(data_end).normalize()
    out = []
    for s in _sets(extra, forward, now):
        lo = pd.Timestamp(s.lo)
        if lo > end:
            continue
        hi = end if s.hi is None else min(end, pd.Timestamp(s.hi))
        if hi < lo:                                            # set D before its first completed session: empty
            continue
        out.append((s.key, lo.date().isoformat(), hi.date().isoformat()))
    return out


def year_parts(lo, hi, extra: bool = False, *, forward: bool = False, now=None) -> List[Tuple[str, str, str, str]]:
    """(label, set, first day, last day) for every calendar year in [lo, hi], a year cut where a set ends — 2024
    comes in two parts, "2024 A" and "2024 B" (``extra``: A0's boundaries too — 2021 in "2021 A0" and "2021 A";
    ``forward``: D in C's place — 2026 in "2026 B" and "2026 D")."""
    lo, hi = pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize()
    S = _sets(extra, forward, now)
    bounds = {pd.Timestamp(s.lo) for s in S if lo < pd.Timestamp(s.lo) <= hi}
    cuts = sorted({lo, hi + pd.Timedelta(days=1)}
                  | {pd.Timestamp(f"{y}-01-01") for y in range(lo.year + 1, hi.year + 1)}
                  | bounds)
    parts = [(a, b - pd.Timedelta(days=1), _key_in(a, S)) for a, b in zip(cuts[:-1], cuts[1:])]
    split = {y for y in {a.year for a, _, _ in parts} if len({k for a, _, k in parts if a.year == y}) > 1}
    return [(f"{a.year} {k}" if a.year in split else str(a.year), k, a.date().isoformat(), b.date().isoformat())
            for a, b, k in parts]


def growth(nav: pd.Series, lo, hi, annual: bool = True) -> Optional[float]:
    """The account's growth over [lo, hi] read off a daily NAV (the last value before ``lo`` to the last at/before
    ``hi``): a year's rate when ``annual`` (a part year is annualised), else the total."""
    nav = nav.sort_index()
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    before = nav[nav.index < lo]
    inside = nav[(nav.index >= lo) & (nav.index <= hi)]
    if not len(inside):
        return None
    a = float(before.iloc[-1]) if len(before) else float(inside.iloc[0])
    b = float(inside.iloc[-1])
    if a <= 0:
        return None
    tot = b / a
    if not annual:
        return tot - 1.0
    start = before.index[-1] if len(before) else inside.index[0]
    yrs = max((inside.index[-1] - start).days, 1) / 365.25
    return float(tot ** (1.0 / yrs) - 1.0)
