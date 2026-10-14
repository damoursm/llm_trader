"""The ETF short arm (the live selection short's arm `etf`) as backtest pieces.

Its picks are the research vol engine's ETF run (``VOL_TAG=_de``, 2026-10-02, recorded in
``<research>/etf/vol_events_de_arm.pkl``): on every regular-hours bar the top-1 by 30-minute ATR% among the
exchange-traded products (ETF / ETN / ETV / ETS, leveraged and inverse, the products delisted since 2021 included),
fresh against its own 20 sessions, the name's first fresh pick of the day. `EtfArm.pieces` applies the live trade rule:
a riser over 5 sessions, FINRA days to cover <= 1 and relative volume >= 1.58 (an unknown value passes), IBKR able to
lend >= $10,000 at the pick (IBKR's archived file to 2024-06-24; unknown or later = lendable, the vol adapter's
convention; the ETF arm has no same-bar fallback), entry at the next bar's close (Rule 201's delay is not modelled for
this arm), and the exits: the give-back target (half the 5-session run-up), the volatility exit in profit (the ATR%
halved), the squeeze cover (a close at 6x the entry), else the 15th session at the pick's clock.

Prices are the traded ones (the store's split-adjusted bars x the pick's factor), spreads come from the fitted model,
and the borrow is IBKR's daily fee (``realfee.RealFees``, the community archive for delisted products) x roundup(1.02 x
the prior close) from settlement, with the pick's file fee or 9.75 %/yr where IBKR has no history.
``EtfArm.pieces(lend=None, cover=None)`` reproduces the recorded 141-trade book's picks and exits (etf25.py's parity).

PREREG102 — products the arm NEVER RANKS (production's VIX block from 2026-10-14, `sel_short_etf_exclude_vix`):
``pieces(exclude=...)`` (plain symbols) reads the picks RE-RANKED without them (`excluded_events`: the recorded
construction rebuilt with those products out of the ranking by `<research>/studies/prereg102/events102.py`; on a bar
where one of them was the top-1 the next product takes the slot, fresh or not against its own history — each name's
freshness is its own history, so leaving them out of the arm's history changes nothing for the others) and never
returns a piece of an excluded product (a by-symbol guard on top). No re-ranked picks for that exclusion = an error
(a filter of the recorded picks would miss the products that take the slot).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config import settings
from src.backtest import account as A

FEE_MED = 9.75                         # %/yr: the median fee the recorded book used without an IBKR history
LEND_END = "2024-06-24"                # IBKR's archived short-stock file ends here
ET = "America/New_York"


def plain(tkn) -> str:
    """The traded symbol of an engine name (``XYZ@E`` / ``@D`` -> ``XYZ``)."""
    return str(tkn).split("@")[0]


def exclusion_key(exclude: Optional[Iterable[str]]) -> Optional[frozenset]:
    """The plain symbols of an exclusion (None when empty)."""
    if exclude is None:
        return None
    key = frozenset(plain(t) for t in exclude)
    return key or None


def events_signature(E: pd.DataFrame) -> List[int]:
    """A recorded pick file's fingerprint (rows, the sum of its panel rows and of its sessions): a re-ranked file
    built from another version of it is refused."""
    return [int(len(E)), int(np.asarray(E.index, np.int64).sum()), int(np.asarray(E["dn"], np.int64).sum())]


def excluded_events_base(period: Optional[str]) -> str:
    """The recorded pick file's stem: ``vol_events_de_arm`` (sets A / B) or ``vol_events_a0_arm`` (set A0)."""
    return "vol_events_a0_arm" if period == "A0" else "vol_events_de_arm"


def excluded_events_path(period: Optional[str], tag: str, root: Optional[str] = None) -> Path:
    """Where `events102.py` writes the picks re-ranked without the products of exclusion ``tag`` (+ its ``.json``)."""
    base = excluded_events_base(period).replace("_arm", f"_{tag}_arm")
    return Path(root or settings.backtest_research_dir) / "etf" / f"{base}.pkl"


def excluded_sidecar(period: Optional[str], exclude: Iterable[str], root: Optional[str] = None):
    """(sidecar path, its record) of the re-ranked picks for exactly this exclusion and period — the ``.json`` beside
    each ``vol_events_{de,a0}_<tag>_arm.pkl`` names its period, its excluded symbols and the recorded file's signature
    (`events_signature`) — or None."""
    key = exclusion_key(exclude)
    d = Path(root or settings.backtest_research_dir) / "etf"
    stem = excluded_events_base(period).replace("_arm", "")
    for js in sorted(d.glob(f"{stem}_*_arm.json")):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("period") != period or frozenset(meta.get("excluded") or ()) != key:
            continue
        if js.with_suffix(".pkl").exists():
            return js, meta
    return None


def excluded_events(period: Optional[str], exclude: Iterable[str], recorded: Optional[pd.DataFrame] = None,
                    root: Optional[str] = None):
    """(picks, sidecar path) of the re-ranked file for exactly this exclusion and period (`excluded_sidecar`); the
    recorded file's signature is checked against ``recorded`` when given. None found -> FileNotFoundError."""
    key = exclusion_key(exclude)
    d = Path(root or settings.backtest_research_dir) / "etf"
    hit = excluded_sidecar(period, key, root)
    if hit is not None:
        js, meta = hit
        pk = js.with_suffix(".pkl")
        if recorded is not None and list(meta.get("source_sig") or []) != events_signature(recorded):
            raise ValueError(f"{pk.name} was re-ranked from another version of the recorded picks "
                             f"({meta.get('source_sig')} vs {events_signature(recorded)}): rebuild it "
                             f"(studies/prereg102/events102.py)")
        return pd.read_pickle(pk), js
    raise FileNotFoundError(
        f"no ETF picks re-ranked without {sorted(key)} for period {period or 'A/B'} in {d} (`EtfArm.pieces(exclude=...)` "
        f"needs them: production never ranks those products) — build them: python "
        f"{Path(root or settings.backtest_research_dir) / 'studies' / 'prereg102' / 'events102.py'} work; or pass "
        f"Mimic(etf_vix=True) for the arm before 2026-10-14")


class EtfArm:
    """The ETF arm's recorded picks and the research engine's bars, spread model and real fees, loaded once.
    ``period="A0"`` (PREREG101): set A0's picks (`<research>/etf/vol_events_a0_arm.pkl`, the same construction over
    A0's panel), bars, calendar, settlement and IBKR's archived fees (`src.backtest.a0`)."""

    period: Optional[str] = None

    def __init__(self, research_dir: Optional[str] = None, period: Optional[str] = None):
        if period not in (None, "A0"):
            raise ValueError(f"EtfArm(period=...) knows set 'A0' only, not {period!r}")
        root = research_dir or settings.backtest_research_dir
        self._root = root
        self._ex: Dict[frozenset, pd.DataFrame] = {}
        for p in (os.path.join(root, "vol"), os.path.join(root, "optvol")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import costs_model as cm
        import realfee as RF
        import volengine as ve
        self.ve, self.cm = ve, cm
        if period is not None:
            from src.backtest import a0 as A0
            self.period = period
            self.rf = A0.real_fees()
            self.E = pd.read_pickle(A0.etf_events_path())
            self.cal = np.asarray(A0.calendar(), np.int64)
            self.settle = A0.settle()
            self._load = A0.load_bars
        else:
            self.rf = RF.RealFees(field="fee", archive=RF.ARCHIVE)
            self.E = pd.read_pickle(os.path.join(root, "etf", "vol_events_de_arm.pkl"))
            self.cal = np.asarray(ve.calendar(), np.int64)
            self.settle = A.default_settle()
        self._bars: Dict[str, Optional[dict]] = {}
        self._ca: Dict[str, np.ndarray] = {}

    def bars(self, tkn: str) -> Optional[dict]:
        if tkn not in self._bars:
            self._bars[tkn] = self.ve.load_bars(tkn) if self.period is None else self._load(tkn)
        return self._bars[tkn]

    def events(self, exclude: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """The arm's recorded picks; with ``exclude`` (plain symbols, PREREG102) the picks RE-RANKED without those
        products (`excluded_events`, loaded once per exclusion), never one of them."""
        key = exclusion_key(exclude)
        if key is None:
            return self.E
        cache = self.__dict__.setdefault("_ex", {})
        if key not in cache:
            E, _ = excluded_events(getattr(self, "period", None), key, recorded=self.E,
                                   root=getattr(self, "_root", None))
            cache[key] = E
        E = cache[key]
        return E[~E.tkn.map(plain).isin(key)]

    def events_source(self, exclude: Optional[Iterable[str]] = None) -> Optional[str]:
        """Which pick file `events` reads for ``exclude``: "recorded", the re-ranked file, or None (not built)."""
        key = exclusion_key(exclude)
        if key is None:
            return "recorded"
        hit = excluded_sidecar(getattr(self, "period", None), key, root=getattr(self, "_root", None))
        return None if hit is None else str(hit[0].with_suffix(".pkl"))

    def picks(self, lo, hi, max_dtc: Optional[float] = 1.0, min_rvol: float = 1.58,
              exclude: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """The arm's picks decided on sessions [lo, hi] that pass the live filters (a riser, crowding, volume);
        ``exclude``: re-ranked without those products (`events`)."""
        E = self.events(exclude)
        d0, d1 = self.ve.dnum(str(pd.Timestamp(lo).date())), self.ve.dnum(str(pd.Timestamp(hi).date()))
        # every top pick with a base: the adapter re-judges the riser test on the session calendar (the recorded
        # status read the old calendar, which counted 28 closures as sessions — PREREG63)
        E = E[(E.dn >= d0) & (E.dn <= d1) & E.status.isin(["riser", "not_riser"])]
        if max_dtc is not None:
            E = E[~(E.dtc > max_dtc + 1e-9)]
        if min_rvol:
            E = E[~(E.vratio < min_rvol)]
        return E

    def pieces(self, lo, hi, give: float = 0.5, cover: Optional[float] = 6.0, hmax: int = 15,
               lend: Optional[str] = "archive", data_end: str = "2026-09-25", strategy: str = "etf",
               max_dtc: Optional[float] = 1.0, min_rvol: float = 1.58, trend: Optional[str] = None,
               entry: str = "next", lend_after: Optional[Callable[[str, int], bool]] = None, cagap: bool = False,
               ctx: bool = False, limit: Optional[tuple] = None, through: bool = False,
               turn: Optional[tuple] = None, exclude: Optional[Iterable[str]] = None) -> List[dict]:
        """Pieces for picks on sessions [lo, hi]. ``lend``: "archive" = IBKR's archived file at the pick (to
        2024-06-24), None = the recorded book's flag (IBKR's file of 2026-09-26, the parity basis). ``trend``
        (PREREG45): "no_uptrend" skips a product whose previous session close is above its 200-session average,
        "uptrend_only" keeps only those (fewer than 200 sessions of history passes either way).
        PREREG62 (`src.backtest.mimic`, all off by default): ``entry="pick"`` enters at the pick bar's close (live:
        ~2-3 minutes after it) instead of the next bar's; ``lend_after(tkn, day)`` decides lendability after the
        archive (default: lendable); ``cagap`` skips a pick on a corporate-action gap bar
        (`sel_short.corporate_gap_flags`, live since 2026-10-05); ``ctx`` attaches each piece's exit context.
        PREREG66: ``limit=(k, W)`` turns each pick into a SELL LIMIT ORDER for the account engine
        (`Strategy(orders="fill")`) — at the pick close + k x the pick bar's 30-minute ATR, up to the cent, alive W
        sessions counting the pick's own, cancelled at the first close at/under the target; one piece per window
        session with a touch (a bar's high reaching it; ``through``: one tick past it), entered at that bar's END at
        the limit (no spread), the exits from that bar's close (`vol.short_limit_candidates`' rule).
        PREREG77 (production's ETF arm from 2026-10-12): ``turn=(k, W)`` enters each pick on THE TURN
        (`vol.turn_candidates`' rule, `_turn_piece`): at the CLOSE of the first bar within W sessions counting the
        pick's own whose close is k x the pick bar's ATR$ under the highest close since the pick, none when a close
        at/under the target comes first (that bar included); a market entry's costs; the exits from the next bar.
        Its lendability is judged when it ENTERS, as live: IBKR's archived file at the turning bar, else
        ``lend_after`` on the turn's day.
        PREREG102: ``exclude`` (plain symbols) — products the arm never ranks: the picks re-ranked without them
        (`events`; production's VIX block), never a piece of one of them."""
        ve, cal = self.ve, self.cal
        E = self.picks(lo, hi, max_dtc, min_rvol, exclude=exclude)
        if lend is None:
            E = E[E.borrowable.astype(bool)]
        lend_end = ve.dnum(LEND_END)
        end_dn = ve.dnum(data_end)
        if entry not in ("next", "pick"):
            raise ValueError(f"entry must be 'next' or 'pick', not {entry!r}")
        out = []
        for rid, r in E.iterrows():
            a = self.bars(r.tkn)
            if a is None:
                continue
            c, h, atr, ns, sday, bod = a["c"], a["h"], a["atr"], a["ns"], a["sday"], a["bod"]
            key = sday.astype(np.int64) * 100 + a["pos"]
            ckey = sday.astype(np.int64) * 100 + bod
            p = int(np.searchsorted(key, int(r.dn) * 100 + int(r.pos)))
            if p >= len(key) or key[p] != int(r.dn) * 100 + int(r.pos):
                continue
            px, clock, i0 = float(c[p]), int(bod[p]), int(np.searchsorted(cal, sday[p]))
            if i0 < 5:
                continue
            # a deadline past the data: the trade is judged to the data's end and marked there (as the vol engine)
            beyond = i0 + hmax >= len(cal) or cal[i0 + hmax] > end_dn
            j5 = int(np.searchsorted(ckey, int(cal[i0 - 5]) * 100 + clock, side="right")) - 1
            pre5 = float(c[j5]) if j5 >= 0 else np.nan
            last = int(np.searchsorted(ckey, (end_dn * 100 + 99) if beyond else (int(cal[i0 + hmax]) * 100 + clock),
                                       side="right")) - 1
            if last <= p + 1 or not (pre5 > 0) or px <= pre5:
                continue
            f = float(r.actual) / float(r.px) if r.px else 1.0
            if trend is not None:
                lastbar = np.r_[np.flatnonzero(np.diff(sday) != 0), len(sday) - 1]
                sd_, sc_ = sday[lastbar].astype(np.int64), c[lastbar]
                k = int(np.searchsorted(sd_, int(r.dn)))          # sessions strictly before the pick's
                if k >= 200:
                    up = sc_[k - 1] > float(np.mean(sc_[k - 200:k]))
                    if (trend == "no_uptrend" and up) or (trend == "uptrend_only" and not up):
                        continue
            if turn is None:                                   # judged at the pick; a turn is judged when it enters
                if lend == "archive" and int(r.dn) <= lend_end:
                    from src.data.deep.borrow_history import lendable_at
                    ok, _ = lendable_at(ve.plain(r.tkn), pd.Timestamp(int(ns[p]), unit="ns", tz="UTC"), px * f)
                    if ok is False:
                        continue
                elif lend_after is not None and int(r.dn) > lend_end and not lend_after(str(r.tkn), int(r.dn)):
                    continue
            if cagap and self._cagap(r.tkn, a)[p]:
                continue
            lvl = px - give * (px - pre5)
            if turn is not None and limit is None:
                q = self._turn_piece(r, a, p, i0, last, beyond, px, lvl, f, turn, cover, strategy, ctx)
                if q is None:
                    continue
                t_day = int(A.day_numbers([int(q["ens"])])[0])
                if lend == "archive" and t_day <= lend_end:
                    from src.data.deep.borrow_history import lendable_at
                    ok, _ = lendable_at(ve.plain(r.tkn), pd.Timestamp(int(q["ens"]), unit="ns", tz="UTC"), q["e"])
                    if ok is False:
                        continue
                elif lend_after is not None and t_day > lend_end and not lend_after(str(r.tkn), t_day):
                    continue
                out.append(q)
                continue
            if limit is not None:
                out.extend(self._limit_pieces(r, rid, a, p, i0, last, beyond, px, lvl, f, limit, through, cover,
                                              strategy, ctx))
                continue
            ie = p + 1 if entry == "next" else p
            e = float(c[ie])
            if e <= lvl:
                continue                                       # already at the target: not entered (as live)
            s2, a2 = c[ie + 1:last + 1], atr[ie + 1:last + 1]
            atr_pick = float(np.float32(atr[p]))
            hits = [np.flatnonzero(s2 <= lvl), np.flatnonzero((a2 <= 0.5 * atr_pick) & (s2 < e))]
            if cover:
                hits.append(np.flatnonzero(s2 >= cover * e))
            first = min((int(x[0]) for x in hits if len(x)), default=None)
            xi = ie + 1 + first if first is not None else last
            pt = ns[ie + 1:xi + 1].astype(np.int64)
            hs = self.cm.predict(np.array([e * f, float(c[xi]) * f]), np.array([float(r.dv20)] * 2),
                                 np.array([float(atr[ie]), float(atr[xi])]), np.array([int(bod[ie]), int(bod[xi])]),
                                 np.array([int(r.year)] * 2))
            d_in = int(A.day_numbers([int(ns[ie])])[0])
            d_out = int(A.day_numbers([int(ns[xi])])[0])
            d0 = self.settle(d_in)
            days = np.arange(d0, self.settle(d_out) + 30, dtype=np.int64)
            lastbar = np.r_[np.flatnonzero(np.diff(sday) != 0), len(sday) - 1]
            sd, sc = sday[lastbar].astype(np.int64), c[lastbar]          # each session's closing bar
            jj = np.searchsorted(sd, days) - 1                 # the last session strictly before each day
            coll = np.ceil(1.02 * sc[np.maximum(jj, 0)] * f - 1e-9)
            fb = float(r.fee_file) if np.isfinite(r.fee_file) else FEE_MED
            cum = np.cumsum(coll * self.rf.daily(r.tkn, days, fb))
            q = {"strategy": strategy, "tkn": r.tkn, "ens": int(ns[ie]), "xns": int(ns[xi]),
                 "e": e * f, "x": float(c[xi]) * f, "hs_in": float(hs[0]), "hs_out": float(hs[1]),
                 "dv20": float(r.dv20), "pt": pt, "pc": c[ie + 1:xi + 1] * f, "ph": h[ie + 1:xi + 1] * f,
                 "d0": d0, "cum": cum, "bfac": 1.0 / 100.0 / 360.0, "d_out": d_out, "pick_day": int(r.dn),
                 "year": int(r.year)}
            if ctx:
                why = (("mark" if beyond else "time") if first is None
                       else "target" if len(hits[0]) and int(hits[0][0]) == first
                       else "volnorm" if len(hits[1]) and int(hits[1][0]) == first else "cap")
                q.update(lvl=lvl * f, apk=atr_pick, pa=np.asarray(atr[ie + 1:xi + 1], float), vn=0.5,
                         cover_mult=cover, why=why, f=f)
            out.append(q)
        return out

    def _limit_pieces(self, r, rid, a: dict, p: int, i0: int, last: int, beyond: bool, px: float, lvl: float,
                      f: float, limit: tuple, through: bool, cover: Optional[float], strategy: str,
                      ctx: bool) -> List[dict]:
        """One pick's SELL limit order as engine pieces (`pieces` with ``limit``)."""
        import math
        from src.backtest.strategies.vol import BIG, window_instants
        k_, W_ = float(limit[0]), int(limit[1])
        c, h, atr, ns, sday, bod = a["c"], a["h"], a["atr"], a["ns"], a["sday"], a["bod"]
        cal = self.cal
        atr_pick = float(np.float32(atr[p]))
        off_t = math.ceil(round(px * f * (1.0 + k_ * atr_pick / 100.0) * 100.0, 6)) / 100.0
        off = off_t / f
        thr = ((0.01 if off_t >= 1.0 else 0.0001) / f) if through else 0.0
        fwd = np.arange(p + 1, last + 1)
        if not len(fwd):
            return []
        m_bar = np.searchsorted(cal, sday[fwd].astype(np.int64)) - i0
        in_win = m_bar <= W_ - 1
        canc = np.flatnonzero(in_win & (c[fwd] <= lvl))
        jc = int(canc[0]) if len(canc) else BIG
        place, end = window_instants(int(r.dn), int(ns[p]), cal, W_)
        if jc < BIG:
            end = min(end, int(ns[fwd[jc]]) + 1)                  # cancelled at the target's close
        place = [t for t in place if t < end]
        lastbar = np.r_[np.flatnonzero(np.diff(sday) != 0), len(sday) - 1]
        sd, sc = sday[lastbar].astype(np.int64), c[lastbar]
        fb = float(r.fee_file) if np.isfinite(r.fee_file) else FEE_MED
        out = []
        pos = np.arange(len(fwd))
        for m in range(W_):
            ks = np.flatnonzero(in_win & (m_bar == m) & (h[fwd] >= off + thr) & (pos <= jc))
            if not len(ks):
                continue
            jf = int(fwd[ks[0]])
            s2, a2 = c[jf:last + 1], atr[jf:last + 1]
            hits = [np.flatnonzero(s2 <= lvl), np.flatnonzero((a2 <= 0.5 * atr_pick) & (s2 < off))]
            if cover:
                hits.append(np.flatnonzero(s2 >= cover * off))
            first = min((int(x[0]) for x in hits if len(x)), default=None)
            xi = jf + first if first is not None else last
            hs = self.cm.predict(np.array([off * f, float(c[xi]) * f]), np.array([float(r.dv20)] * 2),
                                 np.array([float(atr[jf]), float(atr[xi])]), np.array([int(bod[jf]), int(bod[xi])]),
                                 np.array([int(r.year)] * 2))
            d_in = int(A.day_numbers([int(ns[jf])])[0])
            d_out = int(A.day_numbers([int(ns[xi])])[0])
            d0 = self.settle(d_in)
            days = np.arange(d0, self.settle(d_out) + 30, dtype=np.int64)
            jj = np.searchsorted(sd, days) - 1
            coll = np.ceil(1.02 * sc[np.maximum(jj, 0)] * f - 1e-9)
            cum = np.cumsum(coll * self.rf.daily(r.tkn, days, fb))
            q = {"strategy": strategy, "tkn": r.tkn, "ens": int(ns[jf]), "xns": int(ns[xi]), "e": off_t,
                 "x": float(c[xi]) * f, "hs_in": 0.0, "hs_out": float(hs[1]), "dv20": float(r.dv20),
                 "pt": ns[jf + 1:xi + 1].astype(np.int64), "pc": c[jf + 1:xi + 1] * f, "ph": h[jf + 1:xi + 1] * f,
                 "d0": d0, "cum": cum, "bfac": 1.0 / 100.0 / 360.0, "d_out": d_out, "pick_day": int(r.dn),
                 "year": int(r.year), "oid": f"{strategy}|{rid}", "ord_place": place, "ord_end": int(end),
                 "ord_lim": float(off_t), "ord_k": int(m), "limit_k": k_, "pick_px": px * f}
            if ctx:
                why = (("mark" if beyond else "time") if first is None
                       else "target" if len(hits[0]) and int(hits[0][0]) == first
                       else "volnorm" if len(hits[1]) and int(hits[1][0]) == first else "cap")
                q.update(lvl=lvl * f, apk=atr_pick, pa=np.asarray(atr[jf + 1:xi + 1], float), vn=0.5,
                         cover_mult=cover, why=why, f=f)
            out.append(q)
        return out

    def _turn_piece(self, r, a: dict, p: int, i0: int, last: int, beyond: bool, px: float, lvl: float, f: float,
                    turn: tuple, cover: Optional[float], strategy: str, ctx: bool) -> Optional[dict]:
        """One pick entered on THE TURN (`pieces` with ``turn``), or None: no turn inside the window, the target
        reached first (that bar included), or a turn on the last bar of the data."""
        k_, W_ = float(turn[0]), int(turn[1])
        c, h, atr, ns, sday, bod = a["c"], a["h"], a["atr"], a["ns"], a["sday"], a["bod"]
        cal = self.cal
        atr_pick = float(np.float32(atr[p]))
        fwd = np.arange(p + 1, last + 1)
        if not len(fwd):
            return None
        m_bar = np.searchsorted(cal, sday[fwd].astype(np.int64)) - i0
        in_win = m_bar <= W_ - 1
        runmax = np.maximum(np.maximum.accumulate(c[fwd]), px)
        trig = np.flatnonzero(in_win & (c[fwd] <= runmax - k_ * px * atr_pick / 100.0))
        canc = np.flatnonzero(in_win & (c[fwd] <= lvl))
        if not len(trig) or (len(canc) and canc[0] <= trig[0]):
            return None
        jf = int(fwd[trig[0]])
        if jf >= last:
            return None
        e = float(c[jf])
        s2, a2 = c[jf + 1:last + 1], atr[jf + 1:last + 1]
        hits = [np.flatnonzero(s2 <= lvl), np.flatnonzero((a2 <= 0.5 * atr_pick) & (s2 < e))]
        if cover:
            hits.append(np.flatnonzero(s2 >= cover * e))
        first = min((int(x[0]) for x in hits if len(x)), default=None)
        xi = jf + 1 + first if first is not None else last
        hs = self.cm.predict(np.array([e * f, float(c[xi]) * f]), np.array([float(r.dv20)] * 2),
                             np.array([float(atr[jf]), float(atr[xi])]), np.array([int(bod[jf]), int(bod[xi])]),
                             np.array([int(r.year)] * 2))
        d_in = int(A.day_numbers([int(ns[jf])])[0])
        d_out = int(A.day_numbers([int(ns[xi])])[0])
        d0 = self.settle(d_in)
        days = np.arange(d0, self.settle(d_out) + 30, dtype=np.int64)
        lastbar = np.r_[np.flatnonzero(np.diff(sday) != 0), len(sday) - 1]
        sd, sc = sday[lastbar].astype(np.int64), c[lastbar]
        jj = np.searchsorted(sd, days) - 1
        coll = np.ceil(1.02 * sc[np.maximum(jj, 0)] * f - 1e-9)
        fb = float(r.fee_file) if np.isfinite(r.fee_file) else FEE_MED
        cum = np.cumsum(coll * self.rf.daily(r.tkn, days, fb))
        q = {"strategy": strategy, "tkn": r.tkn, "ens": int(ns[jf]), "xns": int(ns[xi]), "e": e * f,
             "x": float(c[xi]) * f, "hs_in": float(hs[0]), "hs_out": float(hs[1]), "dv20": float(r.dv20),
             "pt": ns[jf + 1:xi + 1].astype(np.int64), "pc": c[jf + 1:xi + 1] * f, "ph": h[jf + 1:xi + 1] * f,
             "d0": d0, "cum": cum, "bfac": 1.0 / 100.0 / 360.0, "d_out": d_out, "pick_day": int(r.dn),
             "year": int(r.year), "turn_k": k_, "turn_W": W_, "pick_px": px * f}
        if ctx:
            why = (("mark" if beyond else "time") if first is None
                   else "target" if len(hits[0]) and int(hits[0][0]) == first
                   else "volnorm" if len(hits[1]) and int(hits[1][0]) == first else "cap")
            q.update(lvl=lvl * f, apk=atr_pick, pa=np.asarray(atr[jf + 1:xi + 1], float), vn=0.5, cover_mult=cover,
                     why=why, f=f)
        return q

    def _cagap(self, tkn: str, a: dict) -> np.ndarray:
        """The name's corporate-action gap flags per bar (the live vol / ETF arms' exclusion), memoised."""
        if tkn not in self._ca:
            from src.signals.sel_short import corporate_gap_flags
            self._ca[tkn] = corporate_gap_flags(a["sday"], a["h"], a["l"], a["c"])
        return self._ca[tkn]
