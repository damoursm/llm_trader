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
from typing import Dict, List, Optional

import numpy as np

from config import settings

# the live vol rule (the research engine's parameter names; PREREG26's LIVE + the live exits)
LIVE = {"top_k": 1, "fallback": 5, "fresh_W": 20, "fresh_k": 1, "min_atr": 0.0, "runup_L": 5, "min_runup": 0.0,
        "max_dtc": 1.0, "min_rvol": 1.58, "min_lend_usd": 1e4, "max_fee": None, "max_pool_drop": None,
        "give": 1.0, "cap": 6.0, "hmax": 15, "volnorm_on": True, "volnorm": 0.5}
# the vol2 arm: the vol rule without the crowding filter, the relative-volume filter and the volatility exit
VOL2 = {"max_dtc": None, "min_rvol": 0.0, "volnorm_on": False}


class VolEngine:
    """One loaded research engine: the store, IBKR's lendability arrays and real fees load once."""

    def __init__(self, research_dir: Optional[str] = None, tag: str = "_dx", nbbo: bool = False):
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
        from prereg23_optuna import FIXED, Real
        from prereg26_optuna_borrow import mask
        self.logger = ve.setup()
        C4.CAL = C4.sessions()
        self.C4, self.W5, self.wf10, self.Real, self.mask, self.FIXED = C4, W5, wf10, Real, mask, FIXED
        self.B = wf10.load_borrow()
        self.rf = RF.RealFees(field="fee", archive=RF.ARCHIVE)
        self.nbbo = bool(nbbo)
        self.st = wf10.mm_store(model_only=not self.nbbo)
        self.asm = wf10.Assembler(self.st, pickle.load(open(wf10.CLOSES, "rb")), self.rf, cap=400_000)
        self.settle = C4.settle

    def cfg(self, params: Dict):
        """(the evaluator's complete configuration, the full parameter set incl. the borrow keys)."""
        p = dict(LIVE)
        p.update(params or {})
        return self.wf10.cfg_of(p), p

    def pieces(self, params: Dict, lo: str, hi: str, strategy: str = "vol") -> List[dict]:
        """Pieces for the rule ``LIVE`` + ``params`` over picks made on sessions [lo, hi] (their exits as far as the
        data goes)."""
        cfg, p = self.cfg(params)
        ev = self.Real(self.st, lo, hi, cut_ns=self.W5.END_CUT)
        ev.lend_all = self.mask(p, *self.B)
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
        pick = np.asarray(ev.dn)[np.asarray(T["idx"])]
        out = []
        for q, d in zip(P, pick):
            r = dict(q)
            r["strategy"] = strategy
            r["pick_day"] = int(d)
            out.append(r)
        return out
