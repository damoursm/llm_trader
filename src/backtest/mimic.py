"""The backtest made to MIMIC PRODUCTION — PREREG62 (user 2026-10-09: "Apply all of the simulator fixes. It needs to
mimic production as best as possible.").

The engine (`account.simulate`) and the adapters keep their earlier behaviour by default; `Mimic` switches on each way
the live account differed from the simulator, and `live_books` / `live_strategies` / `live_rules` assemble the live
combination — the selection short's vol, ETF and fall arms at 1/24 and the dip limit book at 1/10 in one $10,000
account (`src/performance/live_account.py`) — under it. Every switch is on by default; `Mimic.off()` runs the earlier
simulator's rules (with PREREG63's bug fixes), `Mimic().without("ext")` one fix's ablation.

* ``lend`` — after IBKR's archived short-stock file ends (2024-06-24) the simulator took every pick as lendable; now a
  stock pick is lendable on a DRAW from a model fitted on the archive's years, one probability per store row
  (``lend_model``: "v2" = PREREG67's LightGBM, 27 inputs — IBKR's API fee the session before with its 5-session
  change and 20-session max, the day's median API fee, the price, the dollar volume, FINRA's days to cover, the ATR%,
  the relative volume, the 5- and 10-session run-up, the ATR rank, Reg SHO threshold lists, days since a reverse split
  and since listing, the name's last archive record before the gap; out of sample 2023-01 .. 2024-06 AUC 0.791, log
  loss 0.516; "v1" = PREREG62's logistic, AUC 0.769) and an ETF pick at the ETF arm's archive rate
  (`ETF_LEND_RATE`). The draws are LINKED (PREREG94, user 2026-10-12): a name's across sessions (latent AR(1),
  ``lend_persist`` 0.7) and every name's on a day (a market-wide factor, ``lend_day_lam`` 0.15, ``lend_day_r`` 0.5),
  fitted on IBKR's archive (a name's lendability agrees 0.76 between picks a session apart vs 0.64 independent; the
  daily lendable count varies 1.45x the independent spread); each pick keeps its probability (`lend_draw_fn`;
  ``Mimic.borrow_v2()``: PREREG67's independent draws). The archive still decides before; our own archive from
  2026-09-25. IBKR's API publishes a fee for
  99% of the names its file could not lend, so the fee alone is no lendability proxy.
* the DRAWS (``seed``): every random decision — a set-B pick's lendability, IBKR's refusal, the house rate — is a
  uniform keyed by (decision, name, session, seed) (`draw_u`). ``draws="strat"`` (PREREG95; the DEFAULT from
  2026-10-13, user: "Apply the best decisions you're recommending"; ``draws="iid"`` = independent across seeds, every
  study before it) makes each block of ``n_draws`` (5) consecutive seeds one STRATIFIED set — each decision's
  uniforms one per stratum of the block, so a 7% refusal hits its share of the block's draws and the house rates cover
  their mix — while every draw keeps the independent draws' law exactly: the linked lend latent's INNOVATIONS are what
  is stratified (`lend_draw_fn`), so each draw is one more path of PREREG94's construction, balanced over the block
  where the innovation weighs most (a pick's lendability itself only in part). PREREG95: the same expectation as
  independent draws on both sets, no stability for the account's growth level, a rule change PAIRED on common seeds
  22-50% less noisy — what the add rule judges (`src/backtest/add_rule.py`).
* ``fee_model`` (PREREG67, not a fix of the earlier simulator's list): where IBKR holds no fee for the name — a delisted
  name after the archive, a name IBKR's API never priced — a stock short pays the fee model's MEAN prediction for its
  pick instead of the name's fee in IBKR's 2026-09-26 file or a 9.75 %/yr median. ``fee_version`` "v4" (PREREG94,
  default): LightGBM Gamma on the mean daily fee over a pick's first 21 days x a trailing level factor (IBKR's real
  API fees of the listed picks over the model's masked guess, the 30 days before; set A out of year: growth -0.69 log
  points vs IBKR's own fees, v2 +0.90); "v2": PREREG67's (log1p + Duan's smearing). ETFs keep the old fill.
* ``house`` — each stock short carries the house rates IBKR quoted for one of the 36 names it priced on 2026-10-05
  (``optvol/house_mix.json``) and the live sizing's shrink (a slice x 2.86 / rate above 2.86,
  `Strategy.shrink_budget`); IBKR refuses any opening short in `REFUSAL_RATE` of the stock picks (live's what-if,
  error 201) — the vol arm's same-bar fallback takes the next candidate, as live. ETFs keep 2.00 / 2.86 (one live ETF
  quote, 0.78 / 0.85: the defaults overstate them) and are never refused.
* ``entry`` — a short enters at its PICK bar's close, stamped at that bar's end (live: ~2-3 minutes after it), its
  first exit check at the next bar (was: the next bar's close).
* ``r201`` — Rule 201 does not delay the entry: the ledger opens at the live price (only the broker offers passively).
* ``ext`` — production's off-grid ticks (the scheduler's extended slots and the 09:30 open) judge the shorts' exits —
  the target, the 6x cover, the volatility exit in profit on the latest regular-hours ATR% — and mark every position,
  on the stored 30-minute bars 04:00-20:00 ET (`bars30m_full`, `bars30m_delisted`, ``extra_bars``): a tick reads the
  bar ending at it within its 30 minutes (19:50: the bar ending 19:30), the 09:30 tick the session's opening print.
  The overnight session has no stored prices: not modelled.
* ``costs`` — the ledger's costs: a short's leg costs what the ledger charges a leg without its own fill —
  production's resolver (`spread.resolve_leg_cost`) on a snapshot of the calibration `tracker.calibrate_sim_costs`
  installs before each tick (`LedgerCosts`), at the leg's price, session and dollar volume, the engine in ledger mode
  (`Rules(fees="ledger")`: no separate commission); the dip book's legs are broker fills the ledger charges their own
  realized cost — a resting limit filled at its price pays the commission, an opening sell ~1 bp — so they keep the
  engine's exact commission + 1 bp (`Strategy(fees="engine")`); every open position is marked net of the resolver's
  exit estimate, as the ledger marks its open trades (`Rules(mark_net=True)`, the piece's ``hs_mark``).
* ``trigger`` — margin judged at the tick marks (`Rules(trigger="close")`), not at intrabar extremes.
* ``order`` — a bar's short entries go by ticker, then arm (`sel_short.ARMS`), as live opens them: one priority for the
  short arms, each piece ranked by its ticker's alphabetical place, then its arm.
* ``cagap`` — the ETF arm skips a pick on a corporate-action gap bar (live since 2026-10-05; the recorded picks cannot
  re-rank, so that bar has no pick).

RULES production adopted (PREREG76 / PREREG77, live from the 2026-10-12 session; on by default, `Mimic.off()` without
the turns): the vol arm's squeeze cover at 4x and no volatility exit (`strategies.vol.LIVE`; the fall arm keeps 6x and
its volatility exit, `vol.FALL`), the ETF arm's whole give-back (production's `sel_short_etf_give_back`) and
``etf_turn=(k, W)`` — the ETF arm on the turn (`EtfArm.pieces(turn=...)`, its target cancelling it, lendability judged
when it enters). And:
``fall_turn=(k, W)`` — THE TURN for the faller shorts (`sel_short.turn_state`): a fall pick waits, and is shorted at
the CLOSE of the first regular-hours bar within W sessions (the pick's own counted) whose close is at least k x the
pick bar's ATR$ under the highest close since the pick (`strategies.vol.turn_candidates`, `VolEngine.turn_pieces`);
none, no trade. Lendability is judged when it ENTERS, as live: IBKR's file in force at that bar in the archive years
(and our own archive), the pick's draw in between (`turn_lend`). ``fall_turn=None``: shorted at the pick, as before.

Not a fix — a RULE under test (PREREG66, off by default): ``short_limit=(k, W)`` turns the picks of the arms in
``short_limit_arms`` into SELL LIMIT ORDERS — the dip book's limit mirrored (`strategies.vol.short_limit_candidates`):
at the pick close + k x the pick bar's 30-minute ATR, alive W sessions, cancelled at the target, the quantity fixed when
placed and each fill cut to the room (`Strategy(orders="fill")`); ``short_limit_through`` needs a bar one tick past it.

PREREG102 / PREREG104 (registered 2026-10-14; the defaults mimic production):
``etf_vix`` — the ETF arm's VIX-linked products (`sel_short.VIX_ETPS`). False (the default while production's
`sel_short_etf_exclude_vix` is on, from 2026-10-14) = production's interim block: they are never RANKED and never enter
the arm's history — the arm's picks are the ones re-ranked without them (`EtfArm.pieces(exclude=...)` reads the picks
`studies/prereg102/events102.py` rebuilt: on a bar where a VIX product was the top-1, the next product takes the slot,
fresh or not against ITS own history; every other name's freshness is unchanged, freshness being each name's own
history), and no piece of a VIX product ever comes out (a by-symbol guard on top). True = the arm as before the block
(PREREG102's E0, `Mimic.off()`). ``short_arms`` — the selection short's arms in the account (default all three; E2
leaves out "etf", F2 "fall": that arm's pieces are never built, its slice stays unused, every other book unchanged).
``fall_slices`` — the fall arm's slice count (None: production's `sim_account.arm_slices("fall")`, 12; F1 = 24).
``fall_volnorm_on`` / ``fall_volnorm`` — the fall arm's volatility exit (None: `strategies.vol.FALL`'s, production's —
on, at the ratio 0.5; F3 = off, F4 = 0.3), applied to `vol.FALL` while the books are built.
``dip_filter`` (PREREG96 Amendment 3, False = production's `enable_dip_long_filter`, off) — the dip book's orders
filtered by RP33 as production applies it (`strategies.dip.filter_keep()`: every signal scored at signal time by the
walk-forward model of its year; a filtered signal never becomes an order, never holds a slot).

TRADING HALTS (PREREG99, OFF by default — ``halts=None`` builds every book as before): ``halts="guard"`` is production's
halt guard at the short arms' entry passes (a pick halted at a pass is re-checked at the next while at most 75 minutes
old, entered at its first free pass at the 1-minute price then, else lost), ``"wait"`` / ``"wait_half"`` / ``"backup"``
/ ``"both"`` the registered variants H1a / H1b / H2 / H3 (`src/backtest/halts.py`; NYSE's halt history, Polygon's
1-minute bars). V6's extras need each store row's borrow pressure: ``eng.pressure_hi``, else V6's own NFHI.

Not modelled (no history): in-window borrow retries before 2026-09-25, late prints at the scorer's cutoff, the dip's
09:30-tick price against the official open, and a leg the broker filled (the ledger charges its own realized cost — the
resolver's averages are built from such legs).
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from statistics import NormalDist
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings
from src.backtest import account as A

ET = "America/New_York"
EPOCH = pd.Timestamp("1970-01-01")
# IBKR's archived short-stock file (the community archive) ends, our own archive of it (`borrow_history.lendable_at`)
# starts: the borrow-data test sets' boundaries too (`src/backtest/periods.py`, one source); the deep store's first
# session: a window before it is PREREG101's set A0 (`live_books`)
from src.backtest.periods import DATA_START, LEND_ARCHIVE_END, OWN_ARCHIVE_FROM  # noqa: E402
ETF_LEND_RATE = 0.827                   # the ETF arm's picks IBKR's archived file could lend, 2021-02 .. 2024-06: 62/75
REFUSAL_RATE = 0.07                     # IBKR refused any opening short in ~7% of the vol book's names (2026-10-05)
FULL = Path("cache/ml/deep/bars30m_full/parts")
DELISTED = Path("cache/ml/deep/bars30m_delisted/parts")
BAR_NS = 30 * 60 * 10**9
DAY_NS = 86400 * 10**9
# the live account's books and the PREREG57 / 61 combination's priorities and capital qualities (`Strategy.quality`:
# every short outranks the dip book, which gives way)
BOOK = {"vol": "sel_short", "etf": "sel_short", "fall": "sel_short", "dip": "dip_long", "crash": "crash_long"}
PRIORITY = {"vol": 0, "etf": 1, "fall": 2, "crash": 5, "dip": 6}
QUALITY = {"vol": 1.3, "etf": 0.7, "fall": 1.48, "crash": 0.2, "dip": 0.13}
FIXES = ("lend", "house", "entry", "r201", "ext", "costs", "trigger", "order", "cagap")
# PREREG99: production's halt guard (V0) and the registered variants H1a / H1b / H2 / H3 (`src/backtest/halts.py`)
HALT_MODES = ("guard", "wait", "wait_half", "backup", "both")
SHORT_ARMS = ("vol", "etf", "fall")                     # the selection short's arms in the live account


def production_etf_vix() -> bool:
    """The ETF arm ranks the VIX-linked products: production's switch (`sel_short_etf_exclude_vix`, on from
    2026-10-14 = they are left out = False)."""
    return not bool(settings.sel_short_etf_exclude_vix)


@dataclass(frozen=True)
class Mimic:
    """Which of production's behaviours the backtest reproduces (module docstring); ``seed`` drives the draws."""
    lend: bool = True
    house: bool = True
    entry: bool = True
    r201: bool = True
    ext: bool = True
    costs: bool = True
    trigger: bool = True
    order: bool = True
    cagap: bool = True
    seed: int = 0
    lend_model: str = "v2"                                   # PREREG67: the lendability draws' model ("v1": PREREG62's)
    lend_shift: float = 0.0                                  # PREREG68: a log-odds shift of those draws (a sensitivity)
    # PREREG94 (user 2026-10-12: "F4 fees + persistent + market-wide lend"): each name's draws linked across sessions
    # (a latent AR(1), persistence `lend_persist`) and moved together by a market-wide day factor (weight
    # `lend_day_lam`, persistence `lend_day_r`) — fitted on IBKR's archive: a name's lendability agrees between picks
    # 0.76 a session apart (independent draws 0.64), the daily lendable count varies 1.45x what independent draws give.
    # 0 / 0 = PREREG67's independent draws. Each pick's own probability is unchanged.
    lend_persist: float = 0.7
    lend_day_lam: float = 0.15
    lend_day_r: float = 0.5
    fee_model: bool = True                                   # PREREG67: the fee model where IBKR holds no fee
    fee_version: str = "v4"                                  # PREREG94's F4 ("v2": PREREG67's model)
    fall_turn: Optional[Tuple[float, int]] = (0.5, 5)        # PREREG76: (k, W) — the faller shorts enter on the turn
    etf_turn: Optional[Tuple[float, int]] = (0.5, 5)         # PREREG77: (k, W) — the ETF arm enters on the turn
    short_limit: Optional[Tuple[float, int]] = None          # PREREG66: (k, W) — the short arms' sell limits
    short_limit_arms: Tuple[str, ...] = ("vol", "etf", "fall")
    short_limit_through: bool = False
    crash: bool = True                                       # PREREG86: the crash-rebound book (production's crash_long)
    # PREREG95 (the default from 2026-10-13): "strat" = each block of `n_draws` consecutive seeds (block
    # seed // n_draws) one stratified set — every random decision spread evenly over the block's draws, each draw's law
    # unchanged; "iid" = each seed's draws independent (every study to PREREG95)
    draws: str = "strat"
    n_draws: int = 5
    # PREREG99 (None = halts not modelled, the default until tonight's paired studies finish): "guard" = production's
    # halt guard at the short arms' entry passes; "wait" / "wait_half" / "backup" / "both" = H1a / H1b / H2 / H3
    halts: Optional[str] = None
    # PREREG102: the ETF arm ranks the VIX-linked products (`sel_short.VIX_ETPS`) — False = production's block (never
    # ranked, never in the arm's history: the re-ranked picks); the default follows production's switch
    etf_vix: bool = field(default_factory=production_etf_vix)
    # PREREG102 / 104: the selection short's arms in the account (E2: no "etf", F2: no "fall" — never built)
    short_arms: Tuple[str, ...] = SHORT_ARMS
    # PREREG104: the fall arm's slice count (None: production's `sim_account.arm_slices("fall")`) and its volatility
    # exit (None: `strategies.vol.FALL`'s — production's, on at 0.5)
    fall_slices: Optional[float] = None
    fall_volnorm_on: Optional[bool] = None
    fall_volnorm: Optional[float] = None
    # PREREG96 Amendment 3 (RP33, production's construction): the dip book's orders filtered by the walk-forward
    # meta-model (`strategies.dip.filter_keep()`: every signal scored at signal time, a dropped signal never an order,
    # never a slot). False = production's `enable_dip_long_filter`, off until PREREG96's re-run decides
    dip_filter: bool = False
    # PREREG108 (context activation, user 2026-10-14: "a model should understand the context and be activated if
    # historically it performs well in it"): a schedule file (`bank.schedule` per arm: month -> its active market
    # cells, `src/backtest/context.py`); an arm takes no piece decided in a cell its month's map leaves out (a month
    # absent = active everywhere). None = production's arms, always active
    activation: Optional[str] = None

    def __post_init__(self):
        if self.entry and not self.r201:
            raise ValueError("entry at the pick bar mimics the ledger, which opens at the live price: it needs r201")
        if set(self.short_limit_arms) - {"vol", "etf", "fall"}:
            raise ValueError(f"short_limit_arms must be among vol / etf / fall, not {self.short_limit_arms}")
        if self.lend_model not in LEND_P_FILES:
            raise ValueError(f"lend_model must be one of {sorted(LEND_P_FILES)}, not {self.lend_model!r}")
        if self.fee_version not in FEE_PRED_FILES:
            raise ValueError(f"fee_version must be one of {sorted(FEE_PRED_FILES)}, not {self.fee_version!r}")
        for name in ("lend_persist", "lend_day_lam", "lend_day_r"):
            if not 0.0 <= float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), not {getattr(self, name)!r}")
        for name in ("fall_turn", "etf_turn"):
            v = getattr(self, name)
            if v is not None and not (float(v[0]) > 0 and int(v[1]) >= 1):
                raise ValueError(f"{name} must be (k > 0, W >= 1), not {v!r}")
        if self.draws not in ("iid", "strat"):
            raise ValueError(f"draws must be 'iid' or 'strat', not {self.draws!r}")
        if self.draws == "strat" and int(self.n_draws) < 2:
            raise ValueError(f"stratified draws need a block of at least 2 seeds, not {self.n_draws!r}")
        if self.halts is not None and self.halts not in HALT_MODES:
            raise ValueError(f"halts must be None or one of {HALT_MODES}, not {self.halts!r}")
        if not isinstance(self.etf_vix, bool):
            raise ValueError(f"etf_vix must be True or False, not {self.etf_vix!r}")
        if not isinstance(self.dip_filter, bool):
            raise ValueError(f"dip_filter must be True or False, not {self.dip_filter!r}")
        if self.activation is not None and not (isinstance(self.activation, str) and Path(self.activation).is_file()):
            raise ValueError(f"activation must be None or a schedule file, not {self.activation!r}")
        arms = self.short_arms
        if (not isinstance(arms, tuple) or set(arms) - set(SHORT_ARMS) or len(set(arms)) != len(arms)
                or any(not isinstance(a, str) for a in arms)):
            raise ValueError(f"short_arms must be a tuple of distinct arms among {SHORT_ARMS}, not {arms!r}")
        if self.fall_slices is not None and not (isinstance(self.fall_slices, (int, float))
                                                 and not isinstance(self.fall_slices, bool)
                                                 and math.isfinite(float(self.fall_slices))
                                                 and float(self.fall_slices) > 0):
            raise ValueError(f"fall_slices must be None or a positive number, not {self.fall_slices!r}")
        if self.fall_volnorm_on is not None and not isinstance(self.fall_volnorm_on, bool):
            raise ValueError(f"fall_volnorm_on must be None, True or False, not {self.fall_volnorm_on!r}")
        if self.fall_volnorm is not None:
            v = self.fall_volnorm
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 < float(v) <= 1.0):
                raise ValueError(f"fall_volnorm must be None or a ratio in (0, 1], not {v!r}")
            if self.fall_volnorm_on is False:
                raise ValueError("fall_volnorm sets the ratio of a volatility exit that fall_volnorm_on=False removes")

    @property
    def strat_n(self) -> Optional[int]:
        """The stratified block's size, None for independent draws."""
        return int(self.n_draws) if self.draws == "strat" else None

    def limited(self, arm: str) -> bool:
        """The arm's picks are sell limit orders (PREREG66)."""
        return self.short_limit is not None and arm in self.short_limit_arms

    def arm_on(self, arm: str) -> bool:
        """The short arm is part of the account (PREREG102 / 104: ``short_arms``)."""
        return arm in self.short_arms

    @classmethod
    def off(cls, seed: int = 0) -> "Mimic":
        """The earlier simulator."""
        return cls(**{f: False for f in FIXES}, seed=seed, lend_model="v1", fee_model=False, fall_turn=None,
                   etf_turn=None, crash=False, lend_persist=0.0, lend_day_lam=0.0, fee_version="v2", draws="iid",
                   etf_vix=True)

    @classmethod
    def borrow_v2(cls, seed: int = 0, **kw) -> "Mimic":
        """PREREG67's construction of the years without IBKR's file (the simulator to 2026-10-12): independent
        lend draws, the v2 fee model, independent seeds."""
        return cls(seed=seed, **dict({"lend_persist": 0.0, "lend_day_lam": 0.0, "fee_version": "v2", "draws": "iid"}, **kw))

    def without(self, *fix: str) -> "Mimic":
        bad = set(fix) - set(FIXES)
        if bad:
            raise KeyError(f"unknown fixes {sorted(bad)}")
        return replace(self, **{f: False for f in fix})

    def label(self) -> str:
        on = [f.name for f in fields(self) if f.name in FIXES and getattr(self, f.name)]
        return "MIMIC" if len(on) == len(FIXES) else ("OLD" if not on else "+".join(on))


# ── deterministic draws ──────────────────────────────────────────────────────

def plain(tkn) -> str:
    """The traded symbol of a backtest name (``XYZ@D`` / ``@X`` / ``@E`` -> ``XYZ``)."""
    return str(tkn).split("@")[0]


def u01(*key) -> float:
    """A uniform draw in [0, 1), fixed by ``key`` (stable across processes and runs)."""
    h = hashlib.blake2b("|".join(str(k) for k in key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 18446744073709551616.0


def draw_u(kind: str, sym, day: int, seed: int, n: Optional[int] = None) -> float:
    """The uniform of one random decision (``kind``, name, session) in draw ``seed``. Independent across seeds
    (``u01``) when ``n`` is None; else STRATIFIED over the block of ``n`` seeds holding ``seed`` (PREREG95): the
    block's n uniforms fall one per stratum [k/n, (k+1)/n) — the strata assigned to the block's draws by a
    pseudo-random permutation keyed by the decision and the block — jittered inside the stratum."""
    if not n:
        return u01(kind, sym, int(day), seed)
    b, j = divmod(int(seed), int(n))
    h = [u01(kind, sym, int(day), b, "perm", i) for i in range(int(n))]
    k = sum(1 for x in h if x < h[j])                      # draw j's stratum: its place in the block's permutation
    return (k + u01(kind, sym, int(day), seed, "jit")) / int(n)


def dnum(day) -> int:
    return int((pd.Timestamp(day) - EPOCH).days)


@lru_cache(maxsize=1)
def house_mix() -> Tuple[Tuple[str, float, float], ...]:
    """(name, maintenance, initial) house rates IBKR quoted on 2026-10-05 for the names the arms short."""
    p = Path(settings.backtest_research_dir) / "optvol" / "house_mix.json"
    return tuple((str(t), float(m), float(i)) for t, m, i in json.loads(p.read_text()))


def house_rates(tkn, day: int, seed: int, *, n: Optional[int] = None) -> Tuple[float, float]:
    """A stock short's (maintenance, initial) house rates: one measured name's, drawn per (name, session)
    (stratified over a block of ``n`` draws when given, `draw_u`)."""
    mix = house_mix()
    k = min(int(draw_u("house", plain(tkn), int(day), seed, n) * len(mix)), len(mix) - 1)
    return mix[k][1], mix[k][2]


def refused(tkn, day: int, seed: int, rate: float = REFUSAL_RATE, *, n: Optional[int] = None) -> bool:
    """IBKR refuses any opening short in the name that session (what-if error 201)."""
    return draw_u("refuse", plain(tkn), int(day), seed, n) < rate


LEND_P_FILES = {"v1": "lend_p.npy", "v2": "lend_p_v2.npy"}   # PREREG62's logistic, PREREG67's LightGBM
# one mean fee per store row: PREREG67's model; PREREG94's F4 (the mean daily fee over a pick's first 21 days, LightGBM
# Gamma, x the trailing 30-day level of IBKR's real API fees for the listed picks against the model's masked guess)
FEE_PRED_FILES = {"v2": "fee_pred_v2.npy", "v4": "fee_pred_v4.npy"}


def lend_p_path(model: str = "v2") -> Path:
    return Path(settings.backtest_research_dir) / "optvol" / "store_xmc10" / LEND_P_FILES[model]


def fee_pred_path(version: str = "v4") -> Path:
    return Path(settings.backtest_research_dir) / "optvol" / "store_xmc10" / FEE_PRED_FILES[version]


_ND = NormalDist()


def _ar1(e: np.ndarray, r: float) -> np.ndarray:
    z = np.empty(len(e))
    if not len(e):
        return z
    z[0] = e[0]
    s = math.sqrt(1.0 - r * r)
    for i in range(1, len(e)):
        z[i] = r * z[i - 1] + s * e[i]
    return z


def _gauss(u: float) -> float:
    return _ND.inv_cdf(min(max(u, 1e-15), 1.0 - 1e-15))


def lend_draw_fn(eng, mim: "Mimic") -> Callable[[str, int], float]:
    """(symbol, day) -> the uniform a pick's lendability is drawn against (lendable iff it is under the pick's
    probability). Independent per (name, session) — ``u01("lend", sym, day, seed)``, PREREG67 — when ``lend_persist``
    and ``lend_day_lam`` are 0; else PREREG94's latent: z = sqrt(lam) w(day) + sqrt(1 - lam) x(name, day), x a
    per-name AR(1) over the sessions (persistence ``lend_persist``, innovations from the same ``u01("lend", ...)``),
    w one market-wide AR(1) (``lend_day_r``, innovations ``u01("lendday", day, seed)``), u = Phi(z): uniform, so every
    pick keeps its probability. A day off the engine's session calendar draws independently. STRATIFIED draws
    (PREREG95, ``Mimic(draws="strat")``): the INNOVATIONS of both AR(1)s come from `draw_u` — one per stratum over the
    block's draws at each (name, session) — so each draw is, in law, exactly one more path of the construction above
    (PREREG95 Amendment 4: stratifying the latent's rank among the block shifted set B's growth +12 points), while the
    block's paths are balanced where the innovation weighs most."""
    r, lam, rw, seed = float(mim.lend_persist), float(mim.lend_day_lam), float(mim.lend_day_r), int(mim.seed)
    n = mim.strat_n
    if r == 0.0 and lam == 0.0:
        return lambda sym, day: draw_u("lend", sym, int(day), seed, n)
    cal_src = getattr(getattr(eng, "C4", None), "CAL", None)        # the research calendar; else the store's days
    cal = np.asarray(cal_src if cal_src is not None else np.unique(eng.st.C.dn.to_numpy()), np.int64)
    pos = {int(d): i for i, d in enumerate(cal)}

    def innov(kind: str, sym: Optional[str]) -> np.ndarray:
        if n:
            return np.array([_gauss(draw_u(kind, "*" if sym is None else sym, int(d), seed, n)) for d in cal])
        if sym is None:                                                                  # the iid draws, bit for bit
            return np.array([_gauss(u01(kind, int(d), seed)) for d in cal])
        return np.array([_gauss(u01(kind, sym, int(d), seed)) for d in cal])
    w = _ar1(innov("lendday", None), rw) if lam > 0 else None
    names: Dict[str, np.ndarray] = {}

    def draw(sym: str, day: int) -> float:
        i = pos.get(int(day))
        if i is None:
            return draw_u("lend", sym, int(day), seed, n)
        z = names.get(sym)
        if z is None:
            x = _ar1(innov("lend", sym), r)
            z = names[sym] = (math.sqrt(lam) * w + math.sqrt(1.0 - lam) * x) if w is not None else x
        return _ND.cdf(float(z[i]))
    return draw


def lend_all(eng, mim: Mimic, p_lend: Optional[np.ndarray] = None,
             draw: Optional[Callable[[str, int], float]] = None) -> np.ndarray:
    """Each store row's lendability under ``mim``: IBKR's archive to 2024-06-24 (`VolEngine.lend_mask`), a draw after
    it (`lend_draw_fn`), our own archive from 2026-09-25, IBKR's refusals removed."""
    base = eng.lend_mask(at="pick" if mim.entry else "next")
    if not (mim.lend or mim.house):
        return base
    C = eng.st.C
    out = np.asarray(base, bool).copy()
    dn = C.dn.to_numpy(np.int64)
    sym = [plain(t) for t in C.tkn.astype(str)]
    if mim.lend:
        if p_lend is None and getattr(eng, "a0_lend_p", None) is not None:
            p_lend = eng.a0_lend_p                 # PREREG101's set A0: every row is in IBKR's archive (never drawn)
        p = np.load(lend_p_path(mim.lend_model)) if p_lend is None else np.asarray(p_lend, float)
        if mim.lend_shift:                                     # the level's uncertainty: +/- log-odds on every draw
            q = np.clip(p, 1e-9, 1 - 1e-9)
            p = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + float(mim.lend_shift))))
        if len(p) != len(C):
            raise ValueError(f"lendability probabilities for {len(p)} rows, the store has {len(C)}")
        memo: Dict[Tuple[str, int], float] = {}
        draw = draw or lend_draw_fn(eng, mim)
        for i in np.flatnonzero(dn > dnum(LEND_ARCHIVE_END)):
            k = (sym[i], int(dn[i]))
            if k not in memo:
                memo[k] = draw(k[0], k[1])
            out[i] = memo[k] < p[i]
        own = np.flatnonzero(dn >= dnum(OWN_ARCHIVE_FROM))
        if len(own):
            from src.data.deep.borrow_history import lendable_at
            bod = C["bod"].to_numpy(np.int64)
            price = C["actual"].to_numpy(float)
            for i in own:
                at = pd.Timestamp(f"{(EPOCH + pd.Timedelta(days=int(dn[i]))).date()} 09:30", tz=ET) \
                    + pd.Timedelta(minutes=30 * (int(bod[i]) + 1))
                ok, _ = lendable_at(sym[i], at, float(price[i]))
                if ok is not None:
                    out[i] = bool(ok)
    if mim.house:
        memo2: Dict[Tuple[str, int], bool] = {}
        for i in np.flatnonzero(out):
            k = (sym[i], int(dn[i]))
            if k not in memo2:
                memo2[k] = refused(k[0], k[1], mim.seed, n=mim.strat_n)
            if memo2[k]:
                out[i] = False
    return out


# ── production's off-grid ticks and the stored extended-hours bars ───────────

@lru_cache(maxsize=1)
def off_ticks() -> Tuple[Tuple[int, int], ...]:
    """Production's ticks that are not a regular-hours bar close, as (minutes after midnight ET, kind): kind 0 the
    09:30 open (the opening print), kind 1 an extended-hours slot (the bar ending at it). From the scheduler's own slot
    list (`runner._session_slots`); overnight slots have no stored prices."""
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    out = []
    for t, kind in slots:
        m = t.hour * 60 + t.minute
        if kind == "extended":
            out.append((m, 1))
        elif kind == "rth" and m == 9 * 60 + 30:
            out.append((m, 0))
    return tuple(sorted(out))


class ExtBars:
    """Each name's stored 30-minute bars 04:00-20:00 ET (naive-UTC bar STARTS; `polygon_deep.bars30m_full`), from
    the deep store, the delisted store (``XYZ@D``) or ``extra_dir`` (a study's own fetch), loaded once.
    ``period="A0"`` (PREREG101): set A0's bars of every session (`src.backtest.a0.parts_dir`), every class."""

    period: Optional[str] = None

    def __init__(self, extra_dir: Optional[str] = None, period: Optional[str] = None):
        self.extra = Path(extra_dir) if extra_dir else None
        self._m: Dict[str, Optional[dict]] = {}
        if period is not None:
            if period != "A0":
                raise ValueError(f"ExtBars(period=...) knows set 'A0' only, not {period!r}")
            from src.backtest import a0 as A0
            self.period, self._a0 = period, Path(A0.parts_dir())

    def path(self, name: str) -> Optional[Path]:
        sym = plain(name)
        if self.period is not None:
            cands = [self._a0 / f"{sym}.parquet"]
        else:
            cands = [DELISTED / f"{sym}.parquet"] if str(name).endswith("@D") else [FULL / f"{sym}.parquet"]
        if self.extra is not None:
            cands.append(self.extra / f"{sym}.parquet")
        return next((p for p in cands if p.exists()), None)

    def get(self, name: str) -> Optional[dict]:
        if name not in self._m:
            self._m[name] = self._load(name)
        return self._m[name]

    def _load(self, name: str) -> Optional[dict]:
        p = self.path(name)
        if p is None:
            return None
        import duckdb
        df = duckdb.query(f"select ts, open, high, low, close, volume, session from '{p.as_posix()}' order by ts").df()
        if df.empty:
            return None
        if self.period is not None and p.parent == self._a0:      # A0's parts in the extended stores' precision:
            for k in ("open", "high", "low", "close"):                # float32 prices, whole-share volumes
                df[k] = df[k].to_numpy(float).astype(np.float32)
            df["volume"] = np.floor(df["volume"].to_numpy(float))
        start = pd.DatetimeIndex(df["ts"]).tz_localize("UTC")
        et = start.tz_convert(ET)
        return {"start": start.asi8.astype(np.int64), "end": start.asi8.astype(np.int64) + BAR_NS,
                "o": df["open"].to_numpy(float), "h": df["high"].to_numpy(float), "l": df["low"].to_numpy(float),
                "c": df["close"].to_numpy(float), "v": df["volume"].to_numpy(float),
                "rth": (df["session"] == "rth").to_numpy(),
                "day": (et.normalize().tz_localize(None) - EPOCH).days.to_numpy().astype(np.int64)}


def tick_marks(b: dict, t0: int, t1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(instant, price, high, low) of every off-grid tick strictly inside (t0, t1) that has a fresh stored price: an
    extended-hours tick the bar ending at it within its 30 minutes, the 09:30 tick the session's opening print."""
    e = np.zeros(0, np.int64), np.zeros(0), np.zeros(0), np.zeros(0)
    if b is None or not len(b["start"]) or t1 <= t0:
        return e
    i0 = int(np.searchsorted(b["start"], t0 - DAY_NS))
    i1 = int(np.searchsorted(b["start"], t1 + DAY_NS))
    days = np.unique(b["day"][i0:i1])
    if not len(days):
        return e
    ticks = off_ticks()
    mins = np.array([m for m, _ in ticks], np.int64)
    kind = np.array([k for _, k in ticks], np.int64)
    naive = (pd.to_datetime(np.repeat(days, len(mins)), unit="D")
             + pd.to_timedelta(np.tile(mins, len(days)), unit="m"))
    T = pd.DatetimeIndex(naive).tz_localize(ET).asi8.astype(np.int64)
    K = np.tile(kind, len(days))
    inside = (T > t0) & (T < t1)
    T, K = T[inside], K[inside]
    out_t, out_p, out_h, out_l = [], [], [], []
    for t, k in zip(T, K):
        if k == 0:
            j = int(np.searchsorted(b["start"], t))
            if j < len(b["start"]) and b["start"][j] == t and b["rth"][j]:
                out_t.append(t), out_p.append(b["o"][j]), out_h.append(b["o"][j]), out_l.append(b["o"][j])
        else:
            j = int(np.searchsorted(b["end"], t, side="right")) - 1
            if j >= 0 and b["end"][j] > t - BAR_NS and not b["rth"][j]:
                out_t.append(t), out_p.append(b["c"][j]), out_h.append(b["h"][j]), out_l.append(b["l"][j])
    return (np.asarray(out_t, np.int64), np.asarray(out_p, float), np.asarray(out_h, float),
            np.asarray(out_l, float))


def price_scale(p: dict, b: dict, tol: float = 0.02) -> Optional[float]:
    """The factor from the stored bars' prices to the piece's (its traded dollars): the median ratio over its
    regular-hours marks (bar closes) that the stored bars hold (a bar ending at the same instant) — its entry when it
    has none (a limit fill's or a passive entry's price is no bar close); None when none match or they disagree by more
    than ``tol`` (a split one side adjusted and the other did not)."""
    t, v = np.asarray(p["pt"], np.int64), np.asarray(p["pc"], float)
    if not len(t):
        t, v = np.array([int(p["ens"])], np.int64), np.array([float(p["e"])])
    j = np.searchsorted(b["end"], t)
    ok = (j < len(b["end"])) & (np.take(b["end"], np.minimum(j, len(b["end"]) - 1)) == t)
    if not ok.any():
        return None
    c = b["c"][j[ok]]
    r = v[ok] / np.where(c > 0, c, np.nan)
    r = r[np.isfinite(r) & (r > 0)]
    if not len(r) or r.max() / r.min() - 1.0 > tol:
        return None
    return float(np.median(r))


def _merge(p: dict, t: np.ndarray, px: np.ndarray, hi: np.ndarray, lo: np.ndarray, cut: Optional[int]) -> dict:
    """The piece with the tick marks merged into its marks (time order); with ``cut`` only marks before it."""
    q = dict(p)
    pt = np.asarray(p["pt"], np.int64)
    keep = pt < cut if cut is not None else np.ones(len(pt), bool)
    T = np.r_[pt[keep], t]
    o = np.argsort(T, kind="mergesort")
    q["pt"] = T[o]
    q["pc"] = np.r_[np.asarray(p["pc"], float)[keep], px][o]
    for k, ext in (("ph", hi), ("pl", lo)):
        if k in p:
            q[k] = np.r_[np.asarray(p[k], float)[keep], ext][o]
    if "pa" in p:                    # the ATR% at a tick: its latest regular-hours bar's (unchanged off the grid)
        pa = np.asarray(p["pa"], float)[keep]
        j = np.searchsorted(pt[keep], t, side="right") - 1
        q["pa"] = np.r_[pa, np.where(j >= 0, pa[np.maximum(j, 0)] if len(pa) else np.nan,
                                     float(p.get("apk", np.nan)))][o]
    return q


def _reborrow(q: dict, settle: Callable[[int], int]) -> None:
    d_out = int(A.day_numbers([int(q["xns"])])[0])
    q["d_out"] = d_out
    if "full_ps" in q and q.get("cum") is not None:
        last = settle(d_out) - 1
        d0, cum = int(q["d0"]), np.asarray(q["cum"], float)
        q["full_ps"] = 0.0 if last < d0 else float(cum[min(last - d0, len(cum) - 1)] * float(q["bfac"]))


def extend_short(p: dict, bars: ExtBars, settle: Callable[[int], int]) -> Tuple[dict, str]:
    """A short piece under production's off-grid ticks: the first tick inside the trade where the target (``lvl``),
    the cover (``cover_mult`` x the entry) or the volatility exit (the latest regular-hours ATR% at most ``vn`` x the
    pick's, in profit) fires becomes the exit (at that tick's price, its borrow recomputed); the ticks before the exit
    become marks. Returns (piece, outcome): "exit", "marks", "no_ticks", "no_bars", "no_scale"."""
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = price_scale(p, b)
    if s is None:
        return p, "no_scale"
    T, px, hi, lo = tick_marks(b, int(p["ens"]), int(p["xns"]))
    if not len(T):
        return p, "no_ticks"
    m, hi, lo = px * s, hi * s, lo * s
    e = float(p["e"])
    pt = np.asarray(p["pt"], np.int64)
    pa = np.asarray(p.get("pa", np.full(len(pt), np.nan)), float)
    j = np.searchsorted(pt, T, side="right") - 1
    apk = float(p.get("apk", np.nan))
    atr = np.where(j >= 0, pa[np.maximum(j, 0)] if len(pa) else np.nan, apk)
    z = np.zeros(len(T), bool)
    hit_t = m <= float(p["lvl"]) if p.get("lvl") is not None else z
    with np.errstate(invalid="ignore"):
        hit_v = (atr <= float(p["vn"]) * apk) & (m < e) if p.get("vn") is not None else z
    hit_c = m >= float(p["cover_mult"]) * e if p.get("cover_mult") else z
    hit = hit_t | hit_v | hit_c
    if not hit.any():
        return _merge(p, T, m, hi, lo, None), "marks"
    k = int(np.argmax(hit))
    q = _merge(p, T[:k], m[:k], hi[:k], lo[:k], int(T[k]))
    q["xns"], q["x"] = int(T[k]), float(m[k])
    q["why"] = "ext_" + ("target" if hit_t[k] else "volnorm" if hit_v[k] else "cap")
    _reborrow(q, settle)
    return q, "exit"


def extend_long(p: dict, bars: ExtBars) -> Tuple[dict, str]:
    """A long piece's off-grid tick marks (its exits stay the rule's)."""
    if int(p.get("ord_k", 0)) < 0:
        return p, "placeholder"
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = price_scale(p, b)
    if s is None:
        return p, "no_scale"
    T, px, hi, lo = tick_marks(b, int(p["ens"]), int(p["xns"]))
    if not len(T):
        return p, "no_ticks"
    return _merge(p, T, px * s, hi * s, lo * s, None), "marks"


# ── the ledger's costs ───────────────────────────────────────────────────────

class DailyADV:
    """Production's liquidity-class input as of a leg's day: `spread.adv_of` = the median of the last 60 daily
    close x volume — from the stored 30-minute bars (each session's volume over all its bars, its regular-hours
    close), the 60 sessions BEFORE the leg's day (no look-ahead)."""

    def __init__(self, bars: ExtBars):
        self.bars = bars
        self._m: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    def _series(self, name: str):
        if name not in self._m:
            b = self.bars.get(name)
            if b is None or "v" not in b:
                self._m[name] = None
            else:
                df = pd.DataFrame({"day": b["day"], "v": b["v"], "c": np.where(b["rth"], b["c"], np.nan)})
                g = df.groupby("day").agg(v=("v", "sum"), c=("c", "last"))
                g = g[np.isfinite(g["c"])]
                self._m[name] = (g.index.to_numpy(np.int64), (g["c"] * g["v"]).to_numpy(float))
        return self._m[name]

    def at(self, name: str, ns: int) -> Optional[float]:
        s = self._series(name)
        if s is None:
            return None
        day = int(A.day_numbers([int(ns)])[0])
        k = int(np.searchsorted(s[0], day, side="left"))
        w = s[1][max(0, k - 60):k]
        return float(np.median(w)) if len(w) else None


class LedgerCosts:
    """What the ledger charges a leg without its own fill: `spread.resolve_leg_cost` on a snapshot of the
    calibration `tracker.calibrate_sim_costs` installs (``snapshot``: the JSON a read-only snapshot wrote — the flat and
    per-session real-fill costs, the fine-session averages and the session x price band x dollar-volume band
    averages). Installs it in this process's `spread` module."""

    def __init__(self, snapshot):
        snap = json.loads(Path(snapshot).read_text()) if not isinstance(snapshot, dict) else snapshot
        from src.performance import spread
        spread.set_real_cost_override(snap.get("frac"), snap.get("by_session"))
        bucket = {}
        for k, v in (snap.get("session_bucket") or {}).items():
            a, b, c = k.split("|")
            bucket[(a, int(b), int(c))] = float(v)
        spread.set_cost_attribution(None, snap.get("session_costs") or {}, None, None, bucket)
        self.sp, self.snap = spread, snap

    def bp(self, side: str, price: float, ns: int, adv: Optional[float]) -> float:
        """The leg's one-way cost in basis points (every live trade is typed STOCK, the ETF arm's too)."""
        iso = pd.Timestamp(int(ns), unit="ns", tz="UTC").tz_convert(ET).isoformat()
        f = self.sp.resolve_leg_cost(side=side, price=float(price), asset_type="STOCK",
                                     session_fine=self.sp.session_bucket_fine(iso), ref=None, filled_qty=None,
                                     model_price=None, fill_price=None, commission=None, adv=adv)
        return float(f) * 1e4

    def apply(self, p: dict, side: int, legs: bool = True, adv: Optional["DailyADV"] = None) -> dict:
        """The piece with the ledger's costs: its entry and exit legs' (``legs``) and the exit estimate its marks are
        net of (``hs_mark``: a regular-hours leg at the entry price). The liquidity class reads production's
        dollar-volume basis as of each leg's day (``adv``: `DailyADV`; else the piece's 20-session mean)."""
        q = dict(p)
        dv = float(p["dv20"]) if p.get("dv20") is not None and np.isfinite(p["dv20"]) else None

        def a(ns):
            v = adv.at(p["tkn"], ns) if adv is not None else None
            return v if v is not None else dv
        out_side = "BUY" if side < 0 else "SELL"
        if legs:
            q["hs_in"] = self.bp("SELL" if side < 0 else "BUY", p["e"], p["ens"], a(p["ens"]))
            q["hs_out"] = self.bp(out_side, p["x"], p["xns"], a(p["xns"]))
        rth = pd.Timestamp(f"{pd.Timestamp(int(p['ens']), unit='ns', tz='UTC').tz_convert(ET).date()} 12:00", tz=ET)
        q["hs_mark"] = self.bp(out_side, p["e"], rth.value, a(p["ens"]))
        return q


# ── the live combination ─────────────────────────────────────────────────────

def rank_shorts(pieces: Sequence[dict]) -> None:
    """Live's order among a bar's short entries (`record_sel_short_trades` opens the journal's picks by ticker, then
    arm): each piece's ``rank`` = its ticker's alphabetical place x 10 + its arm's place in `sel_short.ARMS`."""
    from src.signals.sel_short import ARMS
    pos = {s: i for i, s in enumerate(sorted({plain(p["tkn"]) for p in pieces}))}
    for p in pieces:
        p["rank"] = float(pos[plain(p["tkn"])] * 10 + ARMS.index(p["strategy"]))


def live_strategies(mim: Mimic) -> List[A.Strategy]:
    """The live account's strategies (PREREG61's production-mimicking setup): the short arms each at its own slice
    (production's `sim_account.arm_slices`: vol 1/20, ETF 1/16, fall 1/12 from 2026-10-12, PREREG80; the fall arm's
    ``fall_slices`` when set, PREREG104) with IBKR's house defaults, the dip limit book at 1/10 (10 open) giving way
    whole, the latest first, de-levered the latest first; ``house`` adds the live sizing's shrink, ``order`` one
    priority for the short arms. An arm left out of ``short_arms`` keeps its strategy (no pieces: its slice unused)."""
    from src.performance.sim_account import arm_slices
    out = []
    for n in ("vol", "etf", "fall"):
        k = float(mim.fall_slices) if (n == "fall" and mim.fall_slices is not None) else arm_slices(n)
        s = A.vol_short(n, slices=k)                       # each arm its own slice (production's, PREREG80)
        out.append(replace(s, priority=0 if mim.order else PRIORITY[n], quality=QUALITY[n], book=BOOK[n],
                           house_maint=float(settings.sel_short_account_house_maint),
                           house_init=float(settings.sel_short_account_house_init),
                           shrink_budget=float(settings.sel_short_account_house_init) if mim.house else None,
                           orders="fill" if mim.limited(n) else None))
    k = int(settings.dip_long_account_slices)                  # as many open as slices (`live_account.size_long`)
    d = A.dip_long("dip", slices=float(k), max_open=k)
    out.append(replace(d, priority=PRIORITY["dip"], quality=QUALITY["dip"], preemptible=True, yield_whole=True,
                       yield_first="latest", delever_first="latest", delever_order=1, orders="fill", book=BOOK["dip"],
                       fees="engine" if mim.costs else None))
    if crash_on(mim):              # PREREG86 / 88: one long in the traded fund (TQQQ on QQQ's decisions), 1/4 of the
        rate = max(0.0, float(settings.crash_long_margin_rate or 0.0))   # equity, its fund's margin rate, selling dip
        out.append(A.Strategy("crash", 1, slices=float(settings.crash_long_account_slices), max_open=1,  # longs for
                              priority=PRIORITY["crash"], quality=QUALITY["crash"], book=BOOK["crash"],  # room
                              house_maint=rate, house_init=rate))
    return out


def crash_on(mim: Mimic) -> bool:
    """The crash-rebound book is part of the live combination: production's switch and the mimic's."""
    return bool(mim.crash) and bool(getattr(settings, "enable_crash_long", False))


def live_rules(mim: Mimic) -> A.Rules:
    return A.Rules(trigger="close" if mim.trigger else "high",
                   delever=(float(settings.live_account_delever_trigger), float(settings.live_account_delever_target)),
                   one_book_per_name=True, fees="ledger" if mim.costs else "engine", mark_net=bool(mim.costs))


def turn_lend(eng, mim: Mimic, L: np.ndarray) -> Optional[Callable[[int, int, float], bool]]:
    """PREREG76: production checks a turn's lendability when the short ENTERS — IBKR's file in force at the trigger
    bar's end where an archive holds it (IBKR's to 2024-06-24, ours from 2026-09-25; unknown = lendable, as
    `lend_all`), IBKR's refusal of the name that day after `refused` (``house``); in between, the pick's draw ``L``.
    None (the pick's lendability decides) without ``lend``."""
    if not mim.lend:
        return None
    from src.data.deep.borrow_history import lendable_at
    C = eng.st.C
    dn = C.dn.to_numpy(np.int64)
    tk = C.tkn.astype(str).to_numpy()
    lo_own, hi_arch = dnum(OWN_ARCHIVE_FROM), dnum(LEND_ARCHIVE_END)

    def ok(row: int, ns: int, price: float) -> bool:
        day = int(A.day_numbers([int(ns)])[0])
        if hi_arch < day < lo_own:
            return bool(L[row])
        sym = plain(tk[row])
        lend, _ = lendable_at(sym, pd.Timestamp(int(ns), unit="ns", tz="UTC"), float(price))
        if lend is False:
            return False
        return not (mim.house and refused(sym, int(dn[row]), mim.seed, n=mim.strat_n))
    return ok


def short_extra(mim: Mimic) -> Dict:
    """The research evaluator's settings for the short arms under ``mim``."""
    if mim.entry:
        return {"entry_at_pick": True, "rule201": False}
    return {"rule201": False} if mim.r201 else {}


def vix_etps() -> frozenset:
    """Production's VIX-linked products (`sel_short.VIX_ETPS`, plain symbols): the ones the ETF arm leaves out."""
    from src.signals.sel_short import VIX_ETPS
    return frozenset(VIX_ETPS)


def etf_exclusion(mim: Mimic) -> Optional[frozenset]:
    """The products the ETF arm never ranks under ``mim`` (None: every product, the arm before 2026-10-14)."""
    return None if mim.etf_vix else vix_etps()


def fall_exit_over(mim: Mimic) -> Dict:
    """PREREG104: the fall arm's volatility exit as ``mim`` overrides it (`strategies.vol.FALL` keys; empty: FALL's)."""
    over = {}
    if mim.fall_volnorm_on is not None:
        over["volnorm_on"] = bool(mim.fall_volnorm_on)
    if mim.fall_volnorm is not None:
        over["volnorm"] = float(mim.fall_volnorm)
    return over


@contextmanager
def fall_exit(mim: Mimic):
    """`strategies.vol.FALL` with ``mim``'s fall-exit overrides for a block (every fall build inside reads it:
    `VolEngine.fall_pieces` / `turn_pieces` / `limit_pieces` merge FALL over `vol.LIVE`), restored after; nothing
    patched without an override."""
    over = fall_exit_over(mim)
    if not over:
        yield
        return
    from src.backtest.strategies import vol as VOL
    keep = dict(VOL.FALL)
    VOL.FALL.update(over)
    try:
        yield
    finally:
        VOL.FALL.clear()
        VOL.FALL.update(keep)


@lru_cache(maxsize=8)
def _activation_schedule(path: str) -> Dict[str, Dict[str, frozenset]]:
    """A PREREG108 schedule file: {arm: {"YYYY-MM": frozenset(active cells)}}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {arm: {m: frozenset(cells) for m, cells in months.items()} for arm, months in raw["arms"].items()}


def apply_activation(books: Dict[str, List[dict]], path: str, log: Optional[dict] = None) -> Dict[str, List[dict]]:
    """PREREG108: each book's pieces decided in a market cell (`context.at` of its ``pick_day``) that its decision
    month's map leaves out are dropped — the arm is deactivated there (``log["activation_<arm>_off"]`` counts them);
    an arm or a month the schedule does not name keeps every piece."""
    from src.backtest import context as CX
    sched = _activation_schedule(str(path))
    out = dict(books)
    for arm, pcs in books.items():
        months = sched.get(arm)
        if not months or not pcs:
            continue
        days = [int(p["pick_day"]) for p in pcs]
        cells = CX.at(days)
        keep = []
        for p, d, c in zip(pcs, days, cells):
            m = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m")
            act = months.get(m)
            if act is None or c in act:
                keep.append(p)
        if log is not None:
            log[f"activation_{arm}_off"] = log.get(f"activation_{arm}_off", 0) + len(pcs) - len(keep)
        out[arm] = keep
    return out


def live_books(lo, hi, mim: Mimic, eng=None, etf=None, panel=None, bars: Optional[ExtBars] = None,
               costs: Optional[LedgerCosts] = None, p_lend: Optional[np.ndarray] = None,
               settle: Optional[Callable[[int], int]] = None, log: Optional[dict] = None,
               dip: Optional[List[dict]] = None) -> Dict[str, List[dict]]:
    """The live combination's pieces for decisions on sessions [lo, hi] under ``mim``: {"vol", "etf", "fall",
    "dip"}. ``eng`` / ``etf`` / ``panel``: a loaded `VolEngine`, `EtfArm`, dip panel (loaded when None); ``dip``: the
    dip book's order pieces for [lo, hi] already built (`dip.order_pieces`; never modified here); ``bars`` the
    extended-hours bars (``ext``), ``costs`` the ledger's costs (``costs``, required then); ``log`` collects each
    post-processing outcome's count.

    PREREG101's set A0: a window that starts before the deep store (`periods.DATA_START`), or an A0 engine, builds from
    set A0's components (`src.backtest.a0`: each one given must be A0's, one not given is loaded) and the crash book
    from A0's bars; V6, when installed on an A0 engine, must carry A0's pressure flags.

    PREREG102 / 104: the ETF arm's picks without the VIX-linked products unless ``mim.etf_vix`` (`etf_exclusion`); an
    arm left out of ``mim.short_arms`` gets no pieces (never built); the fall arm built under ``mim``'s exit
    (`fall_exit`)."""
    from src.backtest.strategies import dip as DIP
    from src.backtest.strategies.etf import EtfArm
    from src.backtest.strategies.vol import LIVE, VolEngine
    a0 = pd.Timestamp(lo) < pd.Timestamp(DATA_START) or getattr(eng, "period", None) == "A0"
    if a0:
        from src.backtest import a0 as A0
        A0.check_window(lo, hi)                                # inside A0's store (its overlap window included)
        eng, etf, panel, bars, settle = A0.check_components(eng, etf, panel, bars, settle, dip)
        A0.check_v6(eng)
    eng = eng or VolEngine()
    if mim.arm_on("etf"):
        etf = etf or EtfArm()
    settle = settle or A.default_settle()
    if mim.costs and costs is None:
        raise ValueError("Mimic(costs=True) needs the ledger's costs (LedgerCosts)")
    draw = lend_draw_fn(eng, mim) if mim.lend else None
    L = lend_all(eng, mim, p_lend, draw=draw)
    X = short_extra(mim)
    if mim.fee_model:
        # loaded once per engine and version (one array, one cache tag); an array a study set itself is kept
        if getattr(eng, "_fee_pred", None) is None or getattr(eng, "_fee_pred_loaded", None) not in (None, mim.fee_version):
            eng._fee_pred = np.load(fee_pred_path(mim.fee_version))
            eng._fee_pred_loaded = mim.fee_version
        eng.fee_fallback = eng._fee_pred
    else:
        eng.fee_fallback = None
    lim, thru = mim.short_limit, mim.short_limit_through
    with fall_exit(mim):                       # PREREG104: the fall arm's exit as `mim` sets it (vol.FALL, restored)
        if not mim.arm_on("fall"):
            fall = []                          # PREREG104's F2: never built. The vol arm is then built on an empty
            asm = getattr(eng, "asm", None)    # assembler cache, as after the fall arm's build (`turn_pieces` clears it)
            if asm is not None and hasattr(getattr(asm, "cache", None), "clear"):
                asm.cache.clear()
        elif mim.limited("fall"):
            fall = eng.limit_pieces({}, lo, hi, k=lim[0], W=lim[1], strategy="fall", lend_all=L, through=thru,
                                    fall=True)
        elif mim.fall_turn is not None:
            fall = eng.turn_pieces({}, lo, hi, k=float(mim.fall_turn[0]), W=int(mim.fall_turn[1]), strategy="fall",
                                   lend_all=L, fall=True, lend_at=turn_lend(eng, mim, L))
        else:
            fall = eng.fall_pieces(lo, hi, lend_all=L, extra=X, ctx=True)
        books = {"vol": ([] if not mim.arm_on("vol") else
                         eng.limit_pieces(dict(LIVE), lo, hi, k=lim[0], W=lim[1], strategy="vol", lend_all=L,
                                          through=thru) if mim.limited("vol")
                         else eng.pieces(dict(LIVE), lo, hi, "vol", lend_all=L, extra=X, ctx=True)),
                 "fall": fall}

        def etf_lend(tkn: str, day: int) -> bool:
            return draw(plain(tkn), int(day)) < ETF_LEND_RATE
        halt = None
        if mim.halts:                          # PREREG99: production's halt guard / a variant (src/backtest/halts.py)
            from src.backtest import halts as HALTS
            halt = HALTS.Applier(mim, eng, L)
        if mim.arm_on("etf"):
            ex = etf_exclusion(mim)            # PREREG102: production's VIX block (the re-ranked picks)
            books["etf"] = (etf.pieces if halt is None else halt.etf_pieces(etf))(
                lo, hi, give=float(getattr(settings, "sel_short_etf_give_back", 0.5)),
                entry="pick" if mim.entry else "next",
                lend_after=etf_lend if mim.lend else None, cagap=mim.cagap, ctx=True,
                limit=lim if mim.limited("etf") else None, through=thru,
                turn=None if mim.limited("etf") else mim.etf_turn, **({"exclude": ex} if ex else {}))
        else:
            books["etf"] = []                  # PREREG102's E2: never built, its slice unused
        if dip is None:
            dip = DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel if panel is not None else DIP.build_panel())
        # PREREG96's RP33 as production applies it (`enable_dip_long_filter`): a filtered signal never an order
        books["dip"] = DIP.filter_pieces(dip, DIP.filter_keep()) if mim.dip_filter else list(dip)
        if crash_on(mim):                                      # PREREG86: production's crash_long, its own pieces
            from src.backtest.strategies import crash as CRASH
            books["crash"] = CRASH.pieces(lo, hi, period="A0") if a0 else CRASH.pieces(lo, hi)
        log = log if log is not None else {}
        if halt is not None:                   # the vol and fall entries through the halt model (`log["halts"]`)
            halt.apply(books, lo, hi, X, log)
            for n in SHORT_ARMS:               # an arm left out stays out whatever the halt model rebuilt
                if not mim.arm_on(n):
                    books[n] = []
    if mim.activation:                         # PREREG108: an arm deactivated in its decision's market context
        log = log if log is not None else {}
        books = apply_activation(books, mim.activation, log)
    if mim.house:
        for n in ("vol", "fall"):
            for p in books[n]:
                p["hm"], p["hi"] = house_rates(p["tkn"], int(p["pick_day"]), mim.seed, n=mim.strat_n)
    if mim.ext:
        bars = bars or ExtBars()
        for n in ("vol", "etf", "fall"):
            out = []
            for p in books[n]:
                q, why = extend_short(p, bars, settle)
                log[f"ext_{n}_{why}"] = log.get(f"ext_{n}_{why}", 0) + 1
                out.append(q)
            books[n] = out
        out = []
        for p in books["dip"]:
            q, why = extend_long(p, bars)
            log[f"ext_dip_{why}"] = log.get(f"ext_dip_{why}", 0) + 1
            out.append(q)
        books["dip"] = out
    if mim.costs:
        adv = DailyADV(bars or ExtBars())
        for n in ("vol", "etf", "fall"):
            books[n] = [costs.apply(p, -1, adv=adv) for p in books[n]]
        books["dip"] = [costs.apply(p, 1, legs=False, adv=adv) if int(p.get("ord_k", 0)) >= 0 else p
                        for p in books["dip"]]
        if "crash" in books:                                   # market buys / sells at the open: both legs
            books["crash"] = [costs.apply(p, 1, adv=adv) for p in books["crash"]]
    if mim.order:
        rank_shorts([p for n in ("vol", "etf", "fall") for p in books[n]])
    return books
