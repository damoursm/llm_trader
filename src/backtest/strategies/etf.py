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
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from src.backtest import account as A

FEE_MED = 9.75                         # %/yr: the median fee the recorded book used without an IBKR history
LEND_END = "2024-06-24"                # IBKR's archived short-stock file ends here
ET = "America/New_York"


class EtfArm:
    """The ETF arm's recorded picks and the research engine's bars, spread model and real fees, loaded once."""

    def __init__(self, research_dir: Optional[str] = None):
        root = research_dir or settings.backtest_research_dir
        for p in (os.path.join(root, "vol"), os.path.join(root, "optvol")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import costs_model as cm
        import realfee as RF
        import volengine as ve
        self.ve, self.cm = ve, cm
        self.rf = RF.RealFees(field="fee", archive=RF.ARCHIVE)
        self.E = pd.read_pickle(os.path.join(root, "etf", "vol_events_de_arm.pkl"))
        self.cal = np.asarray(ve.calendar(), np.int64)
        self.settle = A.default_settle()
        self._bars: Dict[str, Optional[dict]] = {}

    def bars(self, tkn: str) -> Optional[dict]:
        if tkn not in self._bars:
            self._bars[tkn] = self.ve.load_bars(tkn)
        return self._bars[tkn]

    def picks(self, lo, hi, max_dtc: Optional[float] = 1.0, min_rvol: float = 1.58) -> pd.DataFrame:
        """The arm's picks decided on sessions [lo, hi] that pass the live filters (a riser, crowding, volume)."""
        E = self.E
        d0, d1 = self.ve.dnum(str(pd.Timestamp(lo).date())), self.ve.dnum(str(pd.Timestamp(hi).date()))
        E = E[(E.dn >= d0) & (E.dn <= d1) & (E.status == "riser")]
        if max_dtc is not None:
            E = E[~(E.dtc > max_dtc + 1e-9)]
        if min_rvol:
            E = E[~(E.vratio < min_rvol)]
        return E

    def pieces(self, lo, hi, give: float = 0.5, cover: Optional[float] = 6.0, hmax: int = 15,
               lend: Optional[str] = "archive", data_end: str = "2026-09-25", strategy: str = "etf",
               max_dtc: Optional[float] = 1.0, min_rvol: float = 1.58, trend: Optional[str] = None) -> List[dict]:
        """Pieces for picks on sessions [lo, hi]. ``lend``: "archive" = IBKR's archived file at the pick (to
        2024-06-24), None = the recorded book's flag (IBKR's file of 2026-09-26, the parity basis). ``trend``
        (PREREG45): "no_uptrend" skips a product whose previous session close is above its 200-session average,
        "uptrend_only" keeps only those (fewer than 200 sessions of history passes either way)."""
        ve, cal = self.ve, self.cal
        E = self.picks(lo, hi, max_dtc, min_rvol)
        if lend is None:
            E = E[E.borrowable.astype(bool)]
        lend_end = ve.dnum(LEND_END)
        end_dn = ve.dnum(data_end)
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
            if i0 < 5 or i0 + hmax >= len(cal) or cal[i0 + hmax] > end_dn:
                continue
            j5 = int(np.searchsorted(ckey, int(cal[i0 - 5]) * 100 + clock, side="right")) - 1
            pre5 = float(c[j5]) if j5 >= 0 else np.nan
            last = int(np.searchsorted(ckey, int(cal[i0 + hmax]) * 100 + clock, side="right")) - 1
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
            if lend == "archive" and int(r.dn) <= lend_end:
                from src.data.deep.borrow_history import lendable_at
                ok, _ = lendable_at(ve.plain(r.tkn), pd.Timestamp(int(ns[p]), unit="ns", tz="UTC"), px * f)
                if ok is False:
                    continue
            ie = p + 1
            lvl = px - give * (px - pre5)
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
            out.append({"strategy": strategy, "tkn": r.tkn, "ens": int(ns[ie]), "xns": int(ns[xi]),
                        "e": e * f, "x": float(c[xi]) * f, "hs_in": float(hs[0]), "hs_out": float(hs[1]),
                        "dv20": float(r.dv20), "pt": pt, "pc": c[ie + 1:xi + 1] * f, "ph": h[ie + 1:xi + 1] * f,
                        "d0": d0, "cum": cum, "bfac": 1.0 / 100.0 / 360.0, "d_out": d_out, "pick_day": int(r.dn),
                        "year": int(r.year)})
        return out
