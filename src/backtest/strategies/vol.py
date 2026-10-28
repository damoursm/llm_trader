"""The vol short arm and its variants as backtest pieces — a thin wrapper over the research vol engine (the store,
evaluator and piece assembler under `settings.backtest_research_dir`/optvol, validated over the PREREG studies).

What the engine already holds: every regular-hours bar's top names by 30-minute ATR% across every stock (today's
names, the names delisted since 2021, the stocks outside the deep store; store ``store_xmc10``, ``OPTVOL_TAG=_dx``),
the live rule's filters as parameters (freshness, crowding, relative volume, top-k and the same-bar fallback, the
give-back target, the volatility exit, the time limit, the squeeze cover), IBKR's archived lendability at the bar
(2021-02 .. 2024-06; unknown later = lendable), Rule 201's passive entries, IBKR's real daily borrow fee and the
fitted spread model (or the real NBBO of each fill with ``nbbo=True``). `VolEngine.pieces` returns the account engine's
pieces for a rule over picks in [lo, hi], each stamped with its ``pick_day`` (the session the pick was made, the
study's week assignment) and its strategy name.

    eng = VolEngine()                         # loads the store once (~1 min)
    P = eng.pieces({}, "2021-02-01", "2024-06-24")            # the live rule
    P2 = eng.pieces({"max_dtc": None}, ...)                   # the live rule without the crowding filter
"""
from __future__ import annotations

import os
import pickle
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

from config import settings

# the live vol rule (the research engine's parameter names; PREREG26's LIVE + the live exits). From the 2026-10-12
# session (PREREG77, the user's choice under the add rule): the squeeze cover at 4x its entry and NO volatility exit
# (production's `sel_short_vol_cover_multiple`, `enable_sel_short_vol_volnorm_exit`). From 2026-10-14 the give-back is
# production's 1.25 (PREREG97's G1, `sel_short_vol_give_back`): every study's live draws made before that day
# (PREREG95's {A,B}_LIVE_S2, PREREG101's A0_LIVE_S2) used 1.0 — pair a new study with new live draws, or pin 1.0
#
# PREREG227's S7 (2026-10-27): every NUMBER production holds as a setting is read from it — the fallback depth
# (`sel_short_vol_fallback_ranks`), the fresh window (`sel_short_vol_own_window_days`), the run-up look-back
# (`sel_short_runup_sessions`), the crowding cut (`sel_short_max_days_to_cover`), the relative-volume cut
# (`sel_short_vol_min_rvol`), the borrow minimum (`short_borrow_min_available_usd`), the cover
# (`sel_short_vol_cover_multiple`) and the volatility exit's ratio (`sel_short_volnorm_ratio`; off for this arm, as
# production's `enable_sel_short_vol_volnorm_exit`). The rule's switches stay the simulator's own (a switch turned in
# production is a rule change: a study); the research store holds columns for the windows and look-backs it was
# built with, so another setting fails there
LIVE = {"top_k": 1, "fallback": int(settings.sel_short_vol_fallback_ranks),
        "fresh_W": int(settings.sel_short_vol_own_window_days), "fresh_k": 1, "min_atr": 0.0,
        "runup_L": int(settings.sel_short_runup_sessions), "min_runup": 0.0,
        "max_dtc": float(settings.sel_short_max_days_to_cover), "min_rvol": float(settings.sel_short_vol_min_rvol),
        "min_lend_usd": float(settings.short_borrow_min_available_usd), "max_fee": None, "max_pool_drop": None,
        "give": float(settings.sel_short_vol_give_back), "cap": float(settings.sel_short_vol_cover_multiple),
        "hmax": int(settings.sel_short_vol_max_hold_sessions),
        "volnorm_on": False, "volnorm": float(settings.sel_short_volnorm_ratio)}
# (from the 2026-10-20 session PREREG159's RPD_IS: the give-back 0.5 and the 10-session limit, production's
# `sel_short_vol_give_back` / `sel_short_arm_max_hold_sessions`; 1.25 and 15 before; from 2026-10-21 the vol arm's own
# limit, 15 sessions — PREREG173's V15, `sel_short_vol_max_hold_sessions`)
# the fall arm (the faller shorts) on the vol rule: no same-bar fallback, every pick a candidate (the faller mask is the
# adapter's), no give-back target; it keeps the 6x cover and the volatility exit — at production's
# `sel_short_fall_volnorm_ratio` (0.7 from the 2026-10-20 session, PREREG159; 0.5 before) — and its OWN time limit,
# production's `sel_short_arm_max_hold_sessions` (FALL merges over LIVE: without it the fall arm would take the vol's)
FALL = {"fallback": 0, "min_runup": -100.0, "give": None, "cap": float(settings.sel_short_cover_multiple),
        "volnorm_on": True,
        "volnorm": float(settings.sel_short_fall_volnorm_ratio), "hmax": int(settings.sel_short_arm_max_hold_sessions)}
# the vol2 arm: the vol rule without the crowding filter, the relative-volume filter and the volatility exit (and with
# the 6x cover: the vol arm's 4x is its own)
VOL2 = {"max_dtc": None, "min_rvol": 0.0, "volnorm_on": False, "cap": float(settings.sel_short_cover_multiple)}
BIG = 1 << 30
ET = "America/New_York"


def short_limit_candidates(pc: np.ndarray, ph: np.ndarray, psd: np.ndarray, pick_day: np.ndarray, cal: np.ndarray,
                           offer: np.ndarray, lvl: np.ndarray, eff: np.ndarray, W: int,
                           through: Optional[np.ndarray] = None):
    """PREREG66 — the dip book's limit entry mirrored for a short pick. Each row is a pick: its forward path's closes
    ``pc``, highs ``ph`` and sessions ``psd`` (path bar 0 = the bar after the pick; ``psd`` -1 = no bar), its session
    ``pick_day``. Its SELL limit ``offer`` rests from the pick through the end of the ``W``-th session counting the
    pick's own (``cal``: the session calendar, day numbers) and no later than the deadline bar ``eff``; it is CANCELLED
    at the first bar that closes at or under the target ``lvl`` (-inf: no target) — a touch inside that bar still
    fills, and the position then meets its target at that close. A touch is a bar whose HIGH reaches the offer
    (``through``: plus that row's tick). Returns (row, window session m, the first touching bar of that session, the
    cancel bar or BIG per pick): one candidate per (pick, window session) with a touch before the cancel."""
    n, P = pc.shape
    J = np.arange(P)[None, :]
    i0 = np.searchsorted(cal, np.asarray(pick_day, np.int64))
    m_bar = np.where(psd >= 0, np.searchsorted(cal, np.maximum(psd, 0)) - i0[:, None], BIG)
    in_win = (psd >= 0) & (m_bar <= W - 1) & (J <= np.asarray(eff)[:, None])
    with np.errstate(invalid="ignore"):
        c_hit = in_win & (pc <= np.asarray(lvl, float)[:, None])
        need = np.asarray(offer, float)[:, None] + (0.0 if through is None else np.asarray(through, float)[:, None])
        touch = in_win & (ph >= need)
    j_cancel = np.where(c_hit.any(1), c_hit.argmax(1), BIG)
    touch &= J <= j_cancel[:, None]
    rows, ms, js = [], [], []
    for m in range(int(W)):
        t = touch & (m_bar == m)
        has = np.flatnonzero(t.any(1))
        rows.append(has)
        ms.append(np.full(len(has), m, np.int64))
        js.append(t[has].argmax(1).astype(np.int64))
    rows, ms, js = np.concatenate(rows), np.concatenate(ms), np.concatenate(js)
    o = np.lexsort((ms, rows))
    return rows[o], ms[o], js[o], j_cancel


def turn_candidates(pc: np.ndarray, psd: np.ndarray, pick_day: np.ndarray, cal: np.ndarray, px: np.ndarray,
                    drop: np.ndarray, lvl: np.ndarray, eff: np.ndarray, W: int) -> np.ndarray:
    """PREREG76 — THE TURN (production's `sel_short.turn_state`, the faller shorts from 2026-10-12). Each row is a
    pick: its forward path's closes ``pc`` and sessions ``psd`` (path bar 0 = the bar after the pick; ``psd`` -1 = no
    bar), its session ``pick_day`` (``cal``: the session calendar, day numbers), its close ``px`` and the turn's size
    ``drop`` (k x its ATR$, in the path's units). The trigger is the FIRST path bar inside the window (``W`` sessions
    counting the pick's own, no later than the deadline bar ``eff``) whose close is at most the highest close since
    the pick (``px`` included) minus ``drop``; a bar closing at or under the target ``lvl`` first — that same bar
    included — cancels the pick (-inf: no target); a trigger on the deadline bar is no trade. Returns the trigger's
    path index per pick, -1 for none."""
    n, P = pc.shape
    J = np.arange(P)[None, :]
    i0 = np.searchsorted(cal, np.asarray(pick_day, np.int64))
    m_bar = np.where(psd >= 0, np.searchsorted(cal, np.maximum(psd, 0)) - i0[:, None], BIG)
    eff = np.asarray(eff, np.int64)
    in_win = (psd >= 0) & (m_bar <= W - 1) & (J <= eff[:, None])
    with np.errstate(invalid="ignore"):
        runmax = np.maximum(np.maximum.accumulate(np.where(psd >= 0, pc, -np.inf), axis=1),
                            np.asarray(px, float)[:, None])
        trig = in_win & (pc <= runmax - np.asarray(drop, float)[:, None])
        canc = in_win & (pc <= np.asarray(lvl, float)[:, None])
    j_t = np.where(trig.any(1), trig.argmax(1), BIG)
    j_c = np.where(canc.any(1), canc.argmax(1), BIG)
    return np.where((j_t < BIG) & (j_t < j_c) & (j_t < eff), j_t, -1)


def window_instants(pick_day: int, pick_end_ns: int, cal: np.ndarray, W: int) -> tuple:
    """A short limit's placement instants (right after the pick bar, then each later window session's 09:30 ET, +1 ns)
    and its natural end (16:00 ET of the W-th session counting the pick's own, +1 ns)."""
    import pandas as pd
    i0 = int(np.searchsorted(cal, int(pick_day)))
    days = [int(cal[min(i0 + m, len(cal) - 1)]) for m in range(int(W))]

    def at(day, hh, mm):
        t = pd.Timestamp("1970-01-01") + pd.Timedelta(days=day) + pd.Timedelta(hours=hh, minutes=mm)
        return pd.Timestamp(t).tz_localize(ET).value
    place = [int(pick_end_ns) + 1] + [at(d, 9, 30) + 1 for d in days[1:]]
    return place, at(days[-1], 16, 0) + 1


class VolEngine:
    """One loaded research engine: the store, IBKR's lendability arrays and real fees load once. ``period="A0"``
    (PREREG101): set A0's store and arrays instead (`src.backtest.a0`) — the same evaluator, assembler and rules over
    A0's candidates, A0's calendar for the deadlines, turns and settlement, IBKR's archived lendability and fees."""

    period: Optional[str] = None

    def __init__(self, research_dir: Optional[str] = None, tag: str = "_dx", nbbo: bool = False,
                 period: Optional[str] = None):
        if period is not None:
            self._init_period(period, research_dir, tag, nbbo)
            return
        root = research_dir or settings.backtest_research_dir
        self.dir = os.path.join(root, "optvol")
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"research engine not found at {self.dir} (settings.backtest_research_dir)")
        os.environ.setdefault("OPTVOL_TAG", tag)
        for p in (self.dir, os.path.join(root, "pylib")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import cap5k4 as C4
        import realfee as RF
        import volengine as ve
        import wf10
        import wf5 as W5
        from evaluator import WHY
        from prereg23_optuna import FIXED, Real
        from prereg26_optuna_borrow import mask
        self.logger = ve.setup()
        self.ve = ve
        C4.CAL = C4.sessions()
        self.C4, self.W5, self.wf10, self.Real, self.mask, self.FIXED = C4, W5, wf10, Real, mask, FIXED
        self.WHY = WHY
        self.B = wf10.load_borrow()
        self._B_pick = None
        self.rf = RF.RealFees(field="fee", archive=RF.ARCHIVE)
        self.nbbo = bool(nbbo)
        self.st = wf10.mm_store(model_only=not self.nbbo)
        self.asm = wf10.Assembler(self.st, pickle.load(open(wf10.CLOSES, "rb")), self.rf, cap=400_000)
        self.settle = C4.settle
        # PREREG67: the fee a short pays where IBKR holds none for its name (one per store row; None = the store's own
        # fill — IBKR's 2026-09-26 file fee, else 9.75 %/yr); `mimic.live_books` sets it from the fee model
        self.fee_fallback: Optional[np.ndarray] = None

    def _init_period(self, period: str, research_dir: Optional[str], tag: str, nbbo: bool) -> None:
        """PREREG101's set A0: the same research engine over A0's store (`src.backtest.a0`). The module-level research
        calendar (`cap5k4.CAL`) is never set here: the engine reads A0's sessions from its own ``C4`` (``CAL``,
        ``settle``) and its assembler sets `cap5k4.CAL` only while it builds pieces, so an engine of sets A / B in the
        same process is untouched."""
        if period != "A0":
            raise ValueError(f"VolEngine(period=...) knows set 'A0' only, not {period!r}")
        if nbbo:
            raise ValueError("set A0 has no cached NBBO quotes: VolEngine(period='A0') prices spreads with the model")
        from src.backtest import a0 as A0
        root = research_dir or settings.backtest_research_dir
        self.dir = os.path.join(root, "optvol")
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f"research engine not found at {self.dir} (settings.backtest_research_dir)")
        os.environ.setdefault("OPTVOL_TAG", tag)
        for p in (self.dir, os.path.join(root, "pylib")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import build  # noqa: F401  (the research engine's layout: puts the vol engine's directory on the path)
        import volengine as ve
        import wf10
        import wf5 as W5
        from evaluator import WHY
        from prereg23_optuna import FIXED, Real
        from prereg26_optuna_borrow import mask
        self.period = period
        self.logger = ve.setup()
        self.ve = ve
        self.C4 = A0.c4_proxy()
        self.W5, self.wf10, self.Real, self.mask, self.FIXED = W5, wf10, Real, mask, FIXED
        self.WHY = WHY
        self.B = A0.load_borrow("next")
        self._B_pick = A0.load_borrow("pick")
        self.rf = A0.real_fees()
        self.nbbo = False
        self.st = A0.mm_store()
        self.asm = A0.assembler(self.st, pickle.load(open(A0.closes_path(), "rb")), self.rf, cap=400_000)
        self.settle = self.C4.settle
        self.fee_fallback = None
        n = len(self.st.C)
        if len(self.B[0]) != n or len(self._B_pick[0]) != n:
            raise ValueError(f"set A0's borrow arrays do not match its store ({n} rows): rebuild them (build101.py)")
        # `mimic.lend_all`: no A0 row is after IBKR's archive, so the lend model's probabilities are never read —
        # one (unused) per row keeps its shape check; the fee fill where IBKR holds no fee: the store's own
        # (`mimic.live_books` keeps an array the engine set itself, ``_fee_pred_loaded`` None)
        self.a0_lend_p = np.zeros(n)
        self._fee_pred = self.st.C["fee"].to_numpy(float)
        self._fee_pred_loaded = None
        self.pressure_hi = A0.pressure_hi(n)

    def _real(self, lo: str, hi: str):
        """The evaluator over picks on [lo, hi], its fee fill the fee model's when one is set (`fee_fallback`)."""
        ev = self.Real(self.st, lo, hi, cut_ns=int(self.W5.END_CUT))
        tag = None if self.fee_fallback is None else id(self.fee_fallback)
        if tag != getattr(self, "_fee_tag", None):
            self.asm.cache.clear()                             # pieces carry their fee: never one built on another fill
            self._fee_tag = tag
        if self.fee_fallback is not None:
            ev.fee = np.asarray(self.fee_fallback, float)[ev.r]
        return ev

    def cfg(self, params: Dict):
        """(the evaluator's complete configuration, the full parameter set incl. the borrow keys)."""
        p = dict(LIVE)
        p.update(params or {})
        return self.wf10.cfg_of(p), p

    def lend_mask(self, params: Optional[Dict] = None, at: str = "next") -> np.ndarray:
        """Every store row's lendability under the rule's borrow keys: IBKR's archived file to 2024-06-24, unknown
        (and every later row) lendable — the default ``lend_all``. ``at``: judged at the bar after the pick ("next",
        the next-bar entry) or at the pick bar's end ("pick": the entry at the pick bar's close — PREREG63)."""
        if at == "pick":
            if self._B_pick is None:
                self._B_pick = self.wf10.load_borrow("pick")
            return self.mask(self.cfg(params or {})[1], *self._B_pick)
        return self.mask(self.cfg(params or {})[1], *self.B)

    def pieces(self, params: Dict, lo: str, hi: str, strategy: str = "vol", lend_all: Optional[np.ndarray] = None,
               extra: Optional[Dict] = None, ctx: bool = False) -> List[dict]:
        """Pieces for the rule ``LIVE`` + ``params`` over picks made on sessions [lo, hi] (their exits as far as the
        data goes). ``lend_all``: each store row's lendability (default `lend_mask`); ``extra``: evaluator settings
        over the rule's (`src.backtest.mimic`: ``entry_at_pick``, ``rule201``); ``ctx``: each piece also carries its
        exit context (``lvl`` the target in traded dollars or None, ``apk`` the pick's ATR%, ``pa`` the ATR% at each
        mark, ``vn`` the volatility exit's ratio or None, ``cover_mult``, ``why``) for `mimic.extend_short`."""
        cfg, p = self.cfg(params)
        cfg = dict(cfg, **(extra or {}))
        ev = self._real(lo, hi)
        at = "pick" if cfg.get("entry_at_pick") else "next"
        ev.lend_all = self.lend_mask(params, at) if lend_all is None else np.asarray(lend_all, bool)
        return self._pieces(ev, cfg, strategy, ctx)

    def fall_pieces(self, lo: str, hi: str, strategy: str = "fall", lend_all: Optional[np.ndarray] = None,
                    extra: Optional[Dict] = None, ctx: bool = False) -> List[dict]:
        """The faller shorts (the live arm `fall`, PREREG38's FALLERS_L / PREREG57): the vol rule's top-1 fresh pick
        when it did NOT rise over the 5 sessions before (a valid base, the close at or under it), lendable at the
        bar, no same-bar fallback, no give-back target."""
        cfg, p = self.cfg(FALL)
        cfg = dict(cfg, **(extra or {}))
        ev = self._real(lo, hi)
        import build as BLD                                  # the research engine's store layout (on sys.path)
        pre = self.st.PRE[:, BLD.LS.index(5)]
        px = self.st.C["px"].to_numpy(float)
        at = "pick" if cfg.get("entry_at_pick") else "next"
        base = self.lend_mask(FALL, at) if lend_all is None \
            else np.asarray(lend_all, bool)
        ev.lend_all = base & (pre > 0) & (px <= pre)
        return self._pieces(ev, cfg, strategy, ctx)

    def limit_pieces(self, params: Dict, lo: str, hi: str, k: float = 0.5, W: int = 5, strategy: str = "vol",
                     lend_all: Optional[np.ndarray] = None, through: bool = False, fall: bool = False,
                     ctx: bool = True) -> List[dict]:
        """PREREG66: the arm's picks (the rule ``LIVE`` + ``params``; ``fall``: the faller shorts, `fall_pieces`'
        selection) as SELL LIMIT ORDERS for the account engine (`Strategy(orders="fill")`): the limit at the pick
        close + ``k`` x the pick bar's 30-minute ATR, up to the cent, alive ``W`` sessions counting the pick's own
        (`short_limit_candidates`), lendability judged at the pick; one piece per window session with a touch —
        entered at the touching bar's END at the limit (passive: no spread), the arm's exits from that bar's close
        (the pick's target and deadline, the volatility exit in profit and the cover against the FILL); each piece
        carries ``oid`` (the pick), ``ord_place`` / ``ord_end`` (`window_instants`, ended early by a cancel at the
        target), ``ord_lim`` and ``ord_k``."""
        if fall:
            fp = dict(FALL)
            cfg, p = self.cfg(fp)
        else:
            fp = params
            cfg, p = self.cfg(params)
        cfg = dict(cfg, rule201=False, entry_at_pick=True)     # selection as at the pick; the entry is the limit's
        ev = self._real(lo, hi)
        base = self.lend_mask(fp, "pick") if lend_all is None else np.asarray(lend_all, bool)
        if fall:
            import build as BLD
            pre = self.st.PRE[:, BLD.LS.index(5)]
            px_all = self.st.C["px"].to_numpy(float)
            ev.lend_all = base & (pre > 0) & (px_all <= pre)
        else:
            ev.lend_all = base
        ev.fee_fn = self.rf.eff
        prm = self.wf10.EV.norm(cfg)
        idx, eff, marked, px, _E, lvl = ev.select(prm)
        if not len(idx):
            return []
        st, ve = self.st, self.ve
        R = ev.r[idx]
        f = (ev.actual[idx] / ev.px[idx]).astype(float)
        apk = ev.apk[idx].astype(float)
        with np.errstate(invalid="ignore"):
            off_t = np.ceil(np.round(px * f * (1.0 + float(k) * apk / 100.0) * 100.0, 6)) / 100.0
        off = off_t / f
        thr = (np.where(off_t >= 1.0, 0.01, 0.0001) / f) if through else None
        cal = np.asarray(self.C4.CAL, np.int64)
        pc, ph, pa = st.PC[R], st.PH[R], st.PA[R]
        rows, ms, js, jc = short_limit_candidates(pc, ph, st.PSD[R], ev.dn[idx], cal, off, lvl, eff, W, thr)
        if not len(rows):
            return []
        nC = len(rows)
        ar = np.arange(nC)
        J = np.arange(pc.shape[1])[None, :]
        PCc, PAc, Rr = pc[rows], pa[rows], R[rows]
        Ein, effc, lvlc, apkc = off[rows], eff[rows], lvl[rows], apk[rows]
        valid = (J >= js[:, None]) & (J <= effc[:, None])
        with np.errstate(invalid="ignore"):
            th = valid & (PCc <= lvlc[:, None])
            t_first = np.where(th.any(1), th.argmax(1), BIG)
            if prm.get("volnorm") is not None:
                vh = valid & (PAc <= float(prm["volnorm"]) * apkc[:, None]) & (PCc < Ein[:, None])
                v_first = np.where(vh.any(1), vh.argmax(1), BIG)
            else:
                v_first = np.full(nC, BIG)
            if prm.get("cap") is not None:
                ch = valid & (PCc >= float(prm["cap"]) * Ein[:, None])
                c_first = np.where(ch.any(1), ch.argmax(1), BIG)
            else:
                c_first = np.full(nC, BIG)
        first = np.minimum(np.minimum(t_first, v_first), c_first)
        hit = first < BIG
        xi = np.where(hit, first, effc)
        why = np.where(hit, np.where(first == t_first, 0, np.where(first == v_first, 1, 4)),
                       np.where(marked[rows], 3, 2))
        X = PCc[ar, xi]
        ens, xns = st.PNS[Rr, js], st.PNS[Rr, xi]
        start_day = st.PSD[Rr, js].astype(np.int64)
        cdays = st.PSD[Rr, xi].astype(np.int64) - start_day
        ix = idx[rows]
        hs_in = np.zeros(nC)                                   # a resting limit sells at its price: no spread
        hs_out = ev.halfspread(ix, xns, X, PAc[ar, xi], st.PBD[Rr, xi])
        NOT = ve.NOTIONAL
        px_, act = ev.px[ix], ev.actual[ix]

        def comm(q):
            return np.minimum(np.maximum(1.0, 0.005 * NOT / q), 0.01 * NOT) / NOT
        a_in, a_out = Ein / px_ * act, X / px_ * act
        c_in = comm(a_in) + ve.SEC_FEE + np.minimum(ve.TAF_PER_SHARE * NOT / a_in, ve.TAF_MAX) / NOT
        c_out = hs_out / 1e4 + comm(a_out)
        fee = ev.fee_fn(ev.tkn[ix], start_day, cdays, ev.fee[ix])
        net = (((Ein * (1 - c_in)) - X * (1 + c_out)) / (Ein * (1 - c_in)) - fee / 100 * cdays / 365) * 100
        T = {"idx": ix, "net": net, "days": (xns - ens) / ve.DAY_NS, "d": ev.dn[ix], "year": ev.year[ix],
             "why": why, "entry": Ein, "exit": X, "ens": ens, "xns": xns, "cdays": cdays, "hs_in": hs_in,
             "hs_out": hs_out, "passive": np.ones(nC, bool), "fill_bars": js, "model_entry": px[rows],
             "fee_eff": fee, "lvl": lvlc, "eff": effc, "ei": js}

        class _Fixed:                                          # the assembler reads the trades from ``trades``
            def __init__(self, e, t):
                self._e, self._t = e, t

            def trades(self, _prm):
                return self._t

            def __getattr__(self, a):
                return getattr(self._e, a)
        P = self.asm.pieces(_Fixed(ev, T), cfg)
        pend = ev.pick_end_ns(idx)
        out = []
        for c, q in enumerate(P):
            r = int(rows[c])
            place, end = window_instants(int(ev.dn[idx[r]]), int(pend[r]), cal, W)
            if jc[r] < BIG:
                end = min(end, int(st.PNS[R[r], jc[r]]) + 1)  # cancelled at the target's close
            place = [t for t in place if t < end]
            d = dict(q)
            ens_c = int(d["ens"])
            pt_all = np.asarray(d["pt"], np.int64)
            keep = pt_all > ens_c                              # no mark before the fill
            for key in ("pt", "pc", "ph"):
                d[key] = np.asarray(d[key])[keep]
            d.update(strategy=strategy, pick_day=int(ev.dn[idx[r]]), oid=f"{strategy}|{int(R[r])}",
                     ord_place=place, ord_end=int(end), ord_lim=float(off_t[r]), ord_k=int(ms[c]),
                     limit_k=float(k), pick_px=float(px[r] * f[r]))
            if ctx:
                xr = int(d["xi"])
                ns_ = st.PNS[R[r]][:xr + 1]
                okk = (ns_ >= 0) & (ns_ <= int(d["xns"]))     # the assembler's marks, before the fill filter
                pa_ = np.asarray(st.PA[R[r]][:xr + 1][okk], float)
                lv = float(lvl[r])
                d.update(lvl=lv * f[r] if np.isfinite(lv) else None, apk=float(apk[r]), pa=pa_[keep],
                         vn=None if prm.get("volnorm") is None else float(prm["volnorm"]),
                         cover_mult=None if prm.get("cap") is None else float(prm["cap"]),
                         why=str(self.WHY[int(why[c])]), f=float(f[r]))
            out.append(d)
        return out

    def turn_pieces(self, params: Dict, lo: str, hi: str, k: float = 0.5, W: int = 5, strategy: str = "vol",
                    lend_all: Optional[np.ndarray] = None, fall: bool = False, ctx: bool = True,
                    lend_at: Optional[Callable[[int, int, float], bool]] = None) -> List[dict]:
        """PREREG76: the arm's picks (the rule ``LIVE`` + ``params``; ``fall``: the faller shorts, `fall_pieces`'
        selection) entered on THE TURN (`turn_candidates`, k x the pick bar's ATR$, ``W`` sessions): shorted at the
        CLOSE of the triggering bar, stamped at its end, a market entry's costs; the arm's exits from the next bar
        against the entry (the target fixed at the pick, the volatility exit in profit, the cover of the entry), the
        deadline counted from the pick. Normal pieces (no order): sized when they enter. ``lend_at(store row,
        trigger end ns, traded price)``: production checks lendability when the short ENTERS — given, the pick is
        selected whatever its lendability at the pick bar and kept only when ``lend_at`` lends it at the trigger;
        None: ``lend_all`` at the pick decides (PREREG76's measurement)."""
        if fall:
            fp = dict(FALL)
            cfg, p = self.cfg(fp)
        else:
            fp = params
            cfg, p = self.cfg(params)
        cfg = dict(cfg, rule201=False, entry_at_pick=True)
        ev = self._real(lo, hi)
        base = self.lend_mask(fp, "pick") if lend_all is None else np.asarray(lend_all, bool)
        if lend_at is not None:
            base = np.ones(len(base), bool)
        if fall:
            import build as BLD
            pre = self.st.PRE[:, BLD.LS.index(5)]
            px_all = self.st.C["px"].to_numpy(float)
            ev.lend_all = base & (pre > 0) & (px_all <= pre)
        else:
            ev.lend_all = base
        ev.fee_fn = self.rf.eff
        prm = self.wf10.EV.norm(cfg)
        idx, eff, marked, px, _E, lvl = ev.select(prm)
        if not len(idx):
            return []
        st, ve = self.st, self.ve
        R = ev.r[idx]
        f = (ev.actual[idx] / ev.px[idx]).astype(float)
        apk = ev.apk[idx].astype(float)
        cal = np.asarray(self.C4.CAL, np.int64)
        jt = turn_candidates(st.PC[R], st.PSD[R], ev.dn[idx], cal, px, float(k) * px * apk / 100.0, lvl, eff, W)
        rows = np.flatnonzero(jt >= 0)
        if lend_at is not None and len(rows):
            rows = np.array([r for r in rows if lend_at(int(R[r]), int(st.PNS[R[r], jt[r]]),
                                                        float(st.PC[R[r], jt[r]] * f[r]))], np.int64)
        if not len(rows):
            return []
        js = jt[rows]
        nC = len(rows)
        ar = np.arange(nC)
        J = np.arange(st.PC.shape[1])[None, :]
        PCc, PAc, Rr = st.PC[R[rows]], st.PA[R[rows]], R[rows]
        Ein = PCc[ar, js]
        effc, lvlc, apkc = eff[rows], lvl[rows], apk[rows]
        valid = (J > js[:, None]) & (J <= effc[:, None])
        with np.errstate(invalid="ignore"):
            th = valid & (PCc <= lvlc[:, None])
            t_first = np.where(th.any(1), th.argmax(1), BIG)
            if prm.get("volnorm") is not None:
                vh = valid & (PAc <= float(prm["volnorm"]) * apkc[:, None]) & (PCc < Ein[:, None])
                v_first = np.where(vh.any(1), vh.argmax(1), BIG)
            else:
                v_first = np.full(nC, BIG)
            if prm.get("cap") is not None:
                ch = valid & (PCc >= float(prm["cap"]) * Ein[:, None])
                c_first = np.where(ch.any(1), ch.argmax(1), BIG)
            else:
                c_first = np.full(nC, BIG)
        first = np.minimum(np.minimum(t_first, v_first), c_first)
        hit = first < BIG
        xi = np.where(hit, first, effc)
        why = np.where(hit, np.where(first == t_first, 0, np.where(first == v_first, 1, 4)),
                       np.where(marked[rows], 3, 2))
        X = PCc[ar, xi]
        ens, xns = st.PNS[Rr, js], st.PNS[Rr, xi]
        start_day = st.PSD[Rr, js].astype(np.int64)
        cdays = st.PSD[Rr, xi].astype(np.int64) - start_day
        ix = idx[rows]
        hs_in = ev.halfspread(ix, ens, Ein, PAc[ar, js], st.PBD[Rr, js])
        hs_out = ev.halfspread(ix, xns, X, PAc[ar, xi], st.PBD[Rr, xi])
        NOT = ve.NOTIONAL
        px_, act = ev.px[ix], ev.actual[ix]

        def comm(q):
            return np.minimum(np.maximum(1.0, 0.005 * NOT / q), 0.01 * NOT) / NOT
        a_in, a_out = Ein / px_ * act, X / px_ * act
        c_in = hs_in / 1e4 + comm(a_in) + ve.SEC_FEE + np.minimum(ve.TAF_PER_SHARE * NOT / a_in, ve.TAF_MAX) / NOT
        c_out = hs_out / 1e4 + comm(a_out)
        fee = ev.fee_fn(ev.tkn[ix], start_day, cdays, ev.fee[ix])
        net = (((Ein * (1 - c_in)) - X * (1 + c_out)) / (Ein * (1 - c_in)) - fee / 100 * cdays / 365) * 100
        T = {"idx": ix, "net": net, "days": (xns - ens) / ve.DAY_NS, "d": ev.dn[ix], "year": ev.year[ix],
             "why": why, "entry": Ein, "exit": X, "ens": ens, "xns": xns, "cdays": cdays, "hs_in": hs_in,
             "hs_out": hs_out, "passive": np.zeros(nC, bool), "fill_bars": js, "model_entry": px[rows],
             "fee_eff": fee, "lvl": lvlc, "eff": effc, "ei": js}

        class _Fixed:                                          # the assembler reads the trades from ``trades``
            def __init__(self, e, t):
                self._e, self._t = e, t

            def trades(self, _prm):
                return self._t

            def __getattr__(self, a):
                return getattr(self._e, a)
        self.asm.cache.clear()                                 # its key cannot tell a turn entry from another entry
        P = self.asm.pieces(_Fixed(ev, T), cfg)
        self.asm.cache.clear()
        out = []
        for c, q in enumerate(P):
            r = int(rows[c])
            d = dict(q)
            keep = np.asarray(d["pt"], np.int64) > int(d["ens"])   # no mark before the entry
            for key in ("pt", "pc", "ph"):
                d[key] = np.asarray(d[key])[keep]
            d.update(strategy=strategy, pick_day=int(ev.dn[idx[r]]), turn_k=float(k), turn_W=int(W),
                     pick_px=float(px[r] * f[r]), turn_bar=int(js[c]))
            if ctx:
                xr = int(d["xi"])
                ns_ = st.PNS[R[r]][:xr + 1]
                okk = (ns_ >= 0) & (ns_ <= int(d["xns"]))
                pa_ = np.asarray(st.PA[R[r]][:xr + 1][okk], float)
                lv = float(lvl[r])
                d.update(lvl=lv * f[r] if np.isfinite(lv) else None, apk=float(apk[r]), pa=pa_[keep],
                         vn=None if prm.get("volnorm") is None else float(prm["volnorm"]),
                         cover_mult=None if prm.get("cap") is None else float(prm["cap"]),
                         why=str(self.WHY[int(why[c])]), f=float(f[r]))
            out.append(d)
        return out

    def _pieces(self, ev, cfg: Dict, strategy: str, ctx: bool) -> List[dict]:
        if self.nbbo:
            import analyze as AN
            book = ev.book(cfg)
            if len(book):
                AN.fetch_quotes(self.st, [book], self.logger)
        ev.fee_fn = self.rf.eff
        T = ev.trades(cfg)
        if T is None or not len(T["idx"]):
            return []
        P = self.asm.pieces(ev, cfg)
        idx = np.asarray(T["idx"])
        pick = np.asarray(ev.dn)[idx]
        out = []
        for j, (q, d) in enumerate(zip(P, pick)):
            r = dict(q)
            r["strategy"] = strategy
            r["pick_day"] = int(d)
            if ctx:
                i = int(idx[j])
                R, xi = int(r["R"]), int(r["xi"])
                ns = self.st.PNS[R][:xi + 1]
                f = float(ev.actual[i] / ev.px[i])
                lvl = float(T["lvl"][j]) if "lvl" in T else -np.inf
                r.update(lvl=lvl * f if np.isfinite(lvl) else None, apk=float(ev.apk[i]),
                         pa=np.asarray(self.st.PA[R][:xi + 1][ns >= 0], float),
                         vn=None if cfg.get("volnorm") is None else float(cfg["volnorm"]),
                         cover_mult=None if cfg.get("cap") is None else float(cfg["cap"]),
                         why=str(self.WHY[int(T["why"][j])]), f=f)
            out.append(r)
        return out
