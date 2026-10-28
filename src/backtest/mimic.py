"""The backtest made to MIMIC PRODUCTION — PREREG62 (user 2026-10-09: "Apply all of the simulator fixes. It needs to
mimic production as best as possible.").

The engine (`account.simulate`) and the adapters keep their earlier behaviour by default; `Mimic` switches on each way
the live account differed from the simulator, and `live_books` / `live_strategies` / `live_rules` assemble the live
combination — the selection short's vol / ETF / fall arms at 1/20 / 1/16 / 1/12 (at most two of one name), the dip
book's daily top pick at 1/6 (the earnings dips ranked last) and the crash book's TQQQ at 1/4 in one account
(`src/performance/live_account.py`; production's $50,000 from PREREG227's S6, ``Mimic.start``; $10,000 before) —
under it; V6's pressure extras are installed by the study harness, not here. Every switch is on by default; `Mimic.off()` runs the earlier
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
  `Strategy.shrink_budget`); IBKR refuses any opening short in a stock pick (live's what-if, error 201) BY NAME, from
  2026-10-24 (``refusal`` "model": PREREG190's refusal model on PREREG193's listing model, a versioned table of the
  probability per (name, session), one uniform per name — `is_refused`; "rate", `Mimic(**PRE193)`: `REFUSAL_RATE` of
  the picks at random) — the vol arm's same-bar fallback takes the next candidate, as live. ETFs keep 2.00 / 2.86
  (one live ETF quote, 0.78 / 0.85: the defaults overstate them) and are never refused.
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
* ``trigger`` — with ``margin="all"`` (the margin calls before PREREG174) margin judged at the tick marks
  (`Rules(trigger="close")`), not at intrabar extremes; IBKR's liquidation (``margin="ibkr"``, the default from
  2026-10-21, below) judges it on the bars' extremes whatever ``trigger`` says.
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
``dip_filter`` (PREREG96 Amendment 3; the default follows production's `enable_dip_long_filter`, ON from the
2026-10-21 session — PREREG166 / 167) — the dip book's orders filtered by RP33 as production applies it
(`strategies.dip.filter_keep()`: every signal scored at signal time by the walk-forward model of its year; a filtered
signal never becomes an order, never holds a slot), before the daily top pick.

THE THREE CHANGES OF 2026-10-20 (PREREG167; user 2026-10-20: "Deploy all three."; their defaults follow production;
``Mimic(**PRE167)`` builds the configuration before them): ``dip_filter`` (above); ``crash_pm_exit`` (PREREG166's XC,
`crash_long_pm_exit`) — the crash book's exit at the after-hours tick right after the close that triggers it (`pm_exit`,
the crash book only: PREREG121's X1 for the dip book stays ``pm_exits``); ``cap`` (PREREG165's CBg6_20,
`enable_cap_long`) — the capitulation book (`strategies/cap.py`): its limit-order pieces, its strategy (1/6, at most 3
open, capital quality 0.1 — preemptible whole, the latest first, de-levered first) and the rule that lets a dip order's
fill sell its longs (`Rules.fills_make_room`). The default's own draws reproduce PREREG167's ALL3 draw for draw.

PREREG116's Q1 (production from the 2026-10-16 session; user 2026-10-15: "Deploy Q1"): the defaults FOLLOW
PRODUCTION — ``dip_rank`` / ``dip_topk`` read `dip_long_rank` / `dip_long_daily_top` (the dip book's daily top pick by
the meta-model's walk-forward score, `strategies.dip.rank_pieces`, production's `dip_long.rank_key`), the dip slice
`dip_long_account_slices` (3 from the 2026-10-19 session; 6 before), ``dip_min_order_dv`` the pick's floor
`dip_long_order_min_dollar_volume` ($2B from the 2026-10-19 session: PREREG147's S3), ``short_name_cap``
`sel_short_max_open_per_ticker` (2) across the three short arms
(``short_name_scope`` "all"); `Mimic.off()` keeps the rules before them (every signal an order at 1/10, no cap). The
walk-forward scores (`strategies.dip.filter_scores`, PREREG96's map) run to the signals of 2026-10-06: a later signal
is unscored and ranked by its RSI2.

TRADING HALTS (PREREG99; production's guard is the DEFAULT from 2026-10-19, AUDIT151 — ``halts=None`` builds every book
as before, `Mimic.off()` / `PRE151`): ``halts="guard"`` is production's halt guard at the short arms' entry passes (a pick halted at a pass is re-checked at the next while at most 75 minutes
old, entered at its first free pass at the 1-minute price then, else lost), ``"wait"`` / ``"wait_half"`` / ``"backup"``
/ ``"both"`` the registered variants H1a / H1b / H2 / H3 (`src/backtest/halts.py`; NYSE's halt history, Polygon's
1-minute bars). V6's extras need each store row's borrow pressure: ``eng.pressure_hi``, else V6's own NFHI.

THE MARKET MODEL CORRECTED (AUDIT151, the user's go-ahead 2026-10-19: "Yes please do 1 and 2"; ON by default,
`Mimic.off()` / ``Mimic(**PRE151)`` without them; `docs/BACKTEST.md` "The market model corrected"): ``quote_costs`` —
every market leg at its real quoted half-spread and IBKR's exact fees (`src/backtest/quotes.py`); ``lend_intra`` — a
short lendable AT THE TRADE, not on the community archive's morning file (`_lend_intra`, `lendable_at_trade`,
`LEND_INTRA`); ``halts="guard"``; ``dip_open`` — the dip's open fills at IBKR's attainable 09:30:30 price
(`strategies.dip.reprice_open`); ``dividends`` (`src/backtest/dividends.py`); ``etf_reused`` False — the ETF arm never
ranks `REUSED_ETF`; and the dip panel with set A's warm-up (`strategies.dip.OWN_PANEL` / `OWN_SCORES`). Together the
deployed account's backtest 71.6 -> 68.0% a year (AUDIT151's ALLW; the default's own draws DEF151, `bank.LIVE_RUNS`).

MARGIN CALLS AS IBKR LIQUIDATES (PREREG174; the user 2026-10-21: "Research online what IBKR does exactly during margin
calls. I want the backtest to do exactly the same, and, if we can't be exact, approximate closely and be only slightly
on the safe side."; the DEFAULT from 2026-10-21, ``Mimic(**PRE174)`` / `Mimic.off()` without): ``margin="ibkr"`` —
IBKR's real-time check approximated on the bars — every position moving together from its previous mark to its bar's
adverse extreme, so no deficit inside a bar is missed —, STRICT at every hour (no Soft Edge Margin: IBKR may eliminate it
in heightened volatility; ``margin_sem`` 0.10 is its documented normal-day tolerance, a bracket), the positions
liquidated the most recently opened first in 100-share lots only until equity is back at the maintenance requirement
(IBKR's "Margin Calculations for Securities" page; ``margin_order="requirement"`` a sensitivity), each lot filled when
the deficit appears along the bar, 10% against the account (one limit-up-limit-down band), never past the bar's extreme
(``margin_fill="path"``; "extreme": every lot at the extreme, the conservative bracket), paying twice its entry
half-spread + IBKR's commission, every working order cancelled, and the end-of-day Reg T check from 15:50 ET on its
floor (equity minus Reg T's initial requirement, ``margin_regt``, at the marks — PREREG175) — `Rules(liquidation="ibkr")`,
`margin_rules`, `docs/BACKTEST.md` "IBKR's liquidation". PRODUCTION'S ACCOUNT LIQUIDATES THE SAME WAY from 2026-10-21
(the user: "production and backtest needs to liquidate like IBKR does"; `live_account.plan_liquidation` = this model,
pinned by `tests/test_live_liquidation.py`). THE MARGIN CHECK SEES WHAT PRODUCTION'S SEES (PREREG175, the default from
22:05 ET; `Mimic(**PRE175)` without): ``margin_bars`` "all" — an off-grid tick's high and low span every stored bar since
the piece's previous mark, as production's check reads every bar completed since its last — and ``ext_longs`` — every
long book marked at the off-grid ticks (the crash and capitulation books too).

THE PARITY AUDIT'S CORRECTIONS (PREREG227; the user 2026-10-27: "Do S1 to S8 now. For S6, both the simulated and live
accounts should be at $50,000."; the defaults from 2026-10-27, ``Mimic(**PRE227)`` / `Mimic.off()` without):
``open_order`` (S1, the capitulation and fear books' open fills at the attainable 09:30:30 price — the three books fill
at an open in production's order), ``crash_reenter`` (S2), ``vol_walk`` (S3, the fallback walked to the account through
the halt model's chains), ``div_all`` (S4), ``quote_all`` (S5), ``start`` (S6, production's $50,000); S7 (the numbers the
simulator held as copies read from production's settings) and S8 (set D's open-at-end capitulation and fear signals)
change nothing of sets A0 / A / B. `docs/BACKTEST.md` "The parity audit's corrections".

Not modelled (no history): in-window borrow retries before 2026-09-25, late prints at the scorer's cutoff, a leg the
broker filled (the ledger charges its own realized cost), IBKR's intraday house-rate raises on a squeezing name, the
overnight session (no stored price; production checks it at its overnight ticks).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, fields, replace
from statistics import NormalDist
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import settings
from src.backtest import account as A
from src.utils import ET as ET_TZ

ET = "America/New_York"
EPOCH = pd.Timestamp("1970-01-01")
# IBKR's archived short-stock file (the community archive) ends, our own archive of it (`borrow_history.lendable_at`)
# starts: the borrow-data test sets' boundaries too (`src/backtest/periods.py`, one source); the deep store's first
# session: a window before it is PREREG101's set A0 (`live_books`)
from src.backtest.periods import DATA_START, LEND_ARCHIVE_END, OWN_ARCHIVE_FROM  # noqa: E402
ETF_LEND_RATE = 0.827                   # the ETF arm's picks IBKR's archived file could lend, 2021-02 .. 2024-06: 62/75
REFUSAL_RATE = 0.07                     # IBKR refused any opening short in ~7% of the vol book's names (2026-10-05)
# PREREG190 / 193 — IBKR'S REFUSALS BY NAME (the user 2026-10-24: "Make the switch right away but still refit/recalibrate
# the model to make it better in the future"; `Mimic.refusal` "model", the default from 2026-10-24): each stock row of
# the short stores carries the probability that IBKR refuses an opening short in it — PREREG190's refusal model (M2c)
# on PREREG193's listing model (the rebuilt microcap rule's state and its age turned into the probability that IBKR
# lists the name; held out on IBKR's own past lists, log loss 0.223 against the rule's 0.433) — in a VERSIONED table
# built by the study (`refusal_path`), drawn against ONE uniform per name (a refusal is a lasting state). A refit is a
# new version, made the default deliberately with the bank rebuilt on it; ``Mimic(**PRE193)`` restores the random
# `REFUSAL_RATE` per (name, session), which `refused` draws (the studies before it patch that function).
REFUSAL_TABLES = {"v1": "refusal_p_v1.pkl"}
REFUSAL_TABLE = "v1"
# the last of IBKR's microcap lists each table's listing model was fitted on (the live calibration notes a refit due
# once the daily archive holds `refusal_listing_refit_sessions` lists after it)
REFUSAL_LISTS_THROUGH = {"v1": "2026-10-23"}
# AUDIT151's L_INTRA (`Mimic.lend_intra`, the default from 2026-10-19): P(still lendable at the trade | lendable on the
# community archive's MORNING file, ~07:15 ET) by that file's pool at the pick — <= $250k / $250k-1M / > $1M — then
# set B's drawn picks (a model of those morning files) and the ETF arm's set-B rate; our own per-tick archive, 16
# sessions 2026-09-25 .. 10-16 (`studies/audit151/lend_intra/estimate.json`; brackets: top 10 0.73 / 0.92 / 0.97 /
# 0.91, top 3 0.55 / 0.94 / 1.00 / 0.78). Refresh it as the archive grows.
LEND_INTRA = (0.65, 0.92, 0.98, 0.80, 0.964)
# AUDIT151's E_REUSE (`Mimic.etf_reused` False, the default from 2026-10-19): tickers the ETF arm's research universe
# holds as exchange-traded products that were STOCKS (or SPACs) when the arm picked them — their product listed later
# (Polygon's ticker details): CGRO 2023-10-02, ESSC 2025-09-29, EYES 2026-05-05, FIXX 2026-08-18, GRIN 2025-06-25
# (Grindrod Shipping in 2022), LAZR 2026-06-29 (Luminar in 2020), SPAQ 2023-01-27 (the Spartan / Fisker SPAC), STRN
# 2025-08-19. No product of them ever topped the arm after its listing: never ranked (the picks re-ranked without them,
# `etf/vol_events_{de,a0}_reuse151_arm.pkl`)
REUSED_ETF = frozenset({"CGRO", "ESSC", "EYES", "FIXX", "GRIN", "LAZR", "SPAQ", "STRN"})
# the simulator before AUDIT151's corrections became `Mimic()`'s defaults (2026-10-19): a study or a test reproducing
# an earlier construction passes ``Mimic(**PRE151)`` — with the dip panel and scores before set A's warm-up
# (`strategies.dip.OWN_PANEL_NOWARM` / `OWN_SCORES_NOWARM`) for the dip book's data
PRE151 = {"halts": None, "lend_intra": None, "etf_reused": True, "quote_costs": False, "dip_open": False,
          "dividends": False}
# the configuration before PREREG167's three changes became `Mimic()`'s defaults (production from the 2026-10-21 session;
# RPD_IS, PREREG159's RPD152 draws): a study reproducing it passes ``Mimic(**PRE167)``
PRE167 = {"dip_filter": False, "crash_pm_exit": False, "cap": False}
# the margin calls before PREREG174's IBKR liquidation became `Mimic()`'s default (2026-10-21): every position closed at
# the tick marks — a study reproducing an earlier construction passes ``Mimic(**PRE174)``
PRE174 = {"margin": "all"}
# the marks before PREREG175 became `Mimic()`'s default (2026-10-21): an off-grid tick's high and low the bar ending at
# it only, and only the dip book marked at the off-grid ticks — ``Mimic(**PRE175)`` reproduces PREREG174's draws
PRE175 = {"margin_bars": "tick", "ext_longs": ("dip",)}
# IBKR's refusals before PREREG193's switch (2026-10-24): `REFUSAL_RATE` at random per (name, session) —
# ``Mimic(**PRE193)`` reproduces every study before it (and lets a study's patched `refused` decide)
PRE193 = {"refusal": "rate"}
# the live combination before THE INSIDER BOOK (production from the 2026-10-26 session, PREREG205): ``Mimic(**PRE205)``
# reproduces every study before it (LIVE193's draws among them)
PRE205 = {"ins": False}
# the dip book's daily top pick before THE WALLSTREETBETS PREFERENCE (production from the 2026-10-26 session, PREREG209's
# KD+): ``Mimic(**PRE209)`` reproduces every study before it (LIVE205's draws among them)
PRE209 = {"dip_wsb": None}
# THE FEAR BOOK (production from the 2026-10-28 session; PREREG219-226): ``Mimic(**PRE226)`` reproduces every study
# before it (KDp's draws, `bank.LIVE_RUNS` before the deploy, among them)
PRE226 = {"fear": False}
# THE PARITY AUDIT'S CORRECTIONS S1-S6 and the $50,000 account (PREREG227, the defaults from 2026-10-27):
# ``Mimic(**PRE227)`` reproduces every study before it (E2s4's draws, `bank.LIVE_RUNS` at the fear book's deploy)
PRE227 = {"open_order": False, "crash_reenter": False, "vol_walk": False, "div_all": False, "quote_all": False,
          "start": 10_000.0}
FULL = Path("cache/ml/deep/bars30m_full/parts")
DELISTED = Path("cache/ml/deep/bars30m_delisted/parts")
BAR_NS = 30 * 60 * 10**9
DAY_NS = 86400 * 10**9
# the live account's books and the PREREG57 / 61 combination's priorities and capital qualities (`Strategy.quality`:
# every short outranks the dip book, which gives way)
BOOK = {"vol": "sel_short", "etf": "sel_short", "fall": "sel_short", "dip": "dip_long", "crash": "crash_long",
        "cap": "cap_long", "ins": "insider_long", "fear": "fear_long"}
# the book a strategy counts as for one book per name (the engine's ``Strategy.book``) where it is not its own: the
# fear book is ONE BOOK WITH THE DIP BOOK (PREREG223's M1; production's `books.SHARED_NAMES`)
NAME_BOOK = {"fear": BOOK["dip"]}
# research books, never live (no production mechanism; `BOOK` stays the live arms', pinned to the live calibration's
# map): PREREG154's QQQ sleeve
RESEARCH_BOOK = {"sleeve": "qqq_sleeve"}
PRIORITY = {"vol": 0, "etf": 1, "fall": 2, "crash": 5, "dip": 6, "cap": 7, "ins": 8, "fear": 9, "sleeve": 99}
QUALITY = {"vol": 1.3, "etf": 0.7, "fall": 1.48, "crash": 0.2, "dip": 0.13, "cap": 0.1, "ins": 0.05, "fear": 0.04,
           "sleeve": 0.0}
FIXES = ("lend", "house", "entry", "r201", "ext", "costs", "trigger", "order", "cagap")
# PREREG99: production's halt guard (V0) and the registered variants H1a / H1b / H2 / H3 (`src/backtest/halts.py`)
HALT_MODES = ("guard", "wait", "wait_half", "backup", "both")
SHORT_ARMS = ("vol", "etf", "fall")                     # the selection short's arms in the live account
# PREREG110: research books a study adds to the live combination for an add-rule test (never production's books):
# name -> {"pieces": pieces(lo, hi, period=None | "A0"), "side", "slices", "max_open", "rate", "priority", "quality",
# "book", "fees"} — filled by `register_research` in the study's process, read through `Mimic(research=(name, ...))`
RESEARCH: Dict[str, dict] = {}


def register_research(name: str, pieces: Callable[..., List[dict]], side: int = 1, slices: float = 4.0,
                      max_open: Optional[int] = 1, rate: float = 0.75, priority: int = 5,
                      quality: Optional[float] = 0.2, book: Optional[str] = None,
                      fees: Optional[str] = None, delever_order: Optional[int] = None) -> None:
    """Register a RESEARCH book (PREREG110) for `Mimic(research=...)`: ``pieces(lo, hi, period)`` builds its account
    pieces for decisions on sessions [lo, hi] (``period="A0"`` on set A0's data, like `strategies.crash.pieces`); its
    strategy holds at most ``max_open`` positions of 1/``slices`` of the equity each at the margin ``rate`` (initial =
    maintenance), entering in ``priority`` order with capital ``quality`` (a higher quality takes the dip book's room,
    as the crash book does; None: free room only); ``book`` its name for one-book-per-name (default its own);
    ``fees`` "engine" (PREREG123): its legs are fills at a known price — auction orders — that the ledger charges their
    own realized cost, so they keep the piece's half-spread + the engine's exact commission, as the dip book's legs do
    (None: the ledger resolver's leg costs, the crash book's); ``delever_order`` (PREREG124): its place in the account's
    de-leveraging line (the dip book's is 1: 0 sells it first; None: never de-levered)."""
    if name in BOOK or name in SHORT_ARMS:
        raise ValueError(f"{name!r} is one of production's books")
    if int(side) not in (1, -1):
        raise ValueError(f"side must be +1 or -1, not {side!r}")
    if fees not in (None, "engine"):
        raise ValueError(f"fees must be None or 'engine', not {fees!r}")
    RESEARCH[name] = {"pieces": pieces, "side": int(side), "slices": float(slices), "max_open": max_open,
                      "rate": float(rate), "priority": int(priority),
                      "quality": None if quality is None else float(quality),
                      "book": book or f"research_{name}", "fees": fees,
                      "delever_order": None if delever_order is None else int(delever_order)}


def research_costed(name: str, pieces: Sequence[dict], costs, adv=None, quotes=None) -> List[dict]:
    """A research book's pieces with the ledger's costs (`LedgerCosts.apply`): both legs for a book of market orders
    (PREREG110), only the marks' exit estimate for an "engine" book — its legs keep their own costs (PREREG123)."""
    r = research_spec(name)
    qk = {"quotes": quotes} if quotes is not None else {}
    return [costs.apply(p, r["side"], legs=r.get("fees") != "engine", adv=adv, **qk) for p in pieces]


def research_spec(name: str) -> dict:
    if name not in RESEARCH:
        raise KeyError(f"research book {name!r} is not registered (`register_research`) in this process")
    return RESEARCH[name]


def production_etf_vix() -> bool:
    """The ETF arm ranks the VIX-linked products: production's switch (`sel_short_etf_exclude_vix`, on from
    2026-10-14 = they are left out = False)."""
    return not bool(settings.sel_short_etf_exclude_vix)


def _prod_short_cap() -> Optional[int]:
    """Production's stacking cap (`sel_short_max_open_per_ticker`, 2 from 2026-10-16; 0 = no cap = None)."""
    c = int(getattr(settings, "sel_short_max_open_per_ticker", 0) or 0)
    return c if c > 0 else None


def _prod_dip_topk() -> Optional[int]:
    """Production's daily top pick (`dip_long_daily_top`, 1 from 2026-10-16; 0 = every signal = None)."""
    k = int(getattr(settings, "dip_long_daily_top", 0) or 0)
    return k if k > 0 else None


def _prod_dip_rank() -> str:
    """Production's ranking of the dip orders: `dip_long_rank` with the daily top pick on, else the deepest dip."""
    return str(getattr(settings, "dip_long_rank", "rsi2") or "rsi2") if _prod_dip_topk() else "rsi2"


def _prod_dip_wsb() -> Optional[Tuple[int, float, str]]:
    """Production's WallStreetBets preference in the daily top pick (PREREG209's KD+, from the 2026-10-26 session):
    (`dip_long_wsb_min_titles`, `dip_long_wsb_window_hours`, `dip_long_wsb_cut_et`) while `enable_dip_long_wsb_prefer`
    and the daily top pick are on, else None — production prefers inside `dip_long._daily_top` only."""
    if not (bool(getattr(settings, "enable_dip_long_wsb_prefer", False)) and _prod_dip_topk()):
        return None
    return (int(getattr(settings, "dip_long_wsb_min_titles", 5)), float(getattr(settings, "dip_long_wsb_window_hours",
                                                                                120.0)),
            str(getattr(settings, "dip_long_wsb_cut_et", "09:00")))


def _prod_dip_order_floor() -> Optional[float]:
    """Production's daily top pick floor (PREREG147's S3, from the 2026-10-19 session): `dip_long_order_min_dollar_volume`
    while the daily top pick is on (production applies it inside `dip_long._daily_top`), else None; 0 = None."""
    if not _prod_dip_topk():
        return None
    v = float(getattr(settings, "dip_long_order_min_dollar_volume", 0.0) or 0.0)
    return v if v > 0 else None


def _prod_dip_filter() -> bool:
    """Production's RP33 cut (`enable_dip_long_filter`, on from the 2026-10-21 session)."""
    return bool(getattr(settings, "enable_dip_long_filter", False))


def _prod_turn(arm: str) -> Optional[Tuple[float, int]]:
    """Production's turn for the fall or ETF arm (PREREG76 / 77) at its numbers (PREREG227's S7):
    (`sel_short_turn_atr`, `sel_short_turn_sessions`); ``None`` (shorted at the pick) is a study's choice."""
    return (float(settings.sel_short_turn_atr), int(settings.sel_short_turn_sessions))


def _prod_crash_pm_exit() -> bool:
    """Production's after-hours exit of the crash book (`crash_long_pm_exit`, PREREG166's XC, from 2026-10-20)."""
    return bool(getattr(settings, "crash_long_pm_exit", False))


def _prod_dip_earnings() -> Optional[int]:
    """Production's earnings-dip demotion (PREREG137's DEMOTE_E, from the 2026-10-19 session): its window in sessions
    (`dip_long_earnings_demote_sessions`) while `enable_dip_long_earnings_demote` and the daily top pick are on, else
    None — production stamps the flag inside the daily top pick only (`dip_long._daily_top`)."""
    if not (bool(getattr(settings, "enable_dip_long_earnings_demote", False)) and _prod_dip_topk()):
        return None
    return int(getattr(settings, "dip_long_earnings_demote_sessions", 3))


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
    # PREREG76 / 77: (k, W) — the faller shorts / the ETF arm enter on the turn; the defaults are production's numbers
    # (`sel_short_turn_atr` 0.5, `sel_short_turn_sessions` 5; PREREG227's S7: read, not copied)
    fall_turn: Optional[Tuple[float, int]] = field(default_factory=lambda: _prod_turn("fall"))
    etf_turn: Optional[Tuple[float, int]] = field(default_factory=lambda: _prod_turn("etf"))
    short_limit: Optional[Tuple[float, int]] = None          # PREREG66: (k, W) — the short arms' sell limits
    short_limit_arms: Tuple[str, ...] = ("vol", "etf", "fall")
    short_limit_through: bool = False
    crash: bool = True                                       # PREREG86: the crash-rebound book (production's crash_long)
    # PREREG165 / 167 (production from the 2026-10-21 session): the capitulation book (production's cap_long — built
    # while `enable_cap_long` is on, `cap_on`); PREREG166's XC: the crash book's after-hours exit (the default follows
    # production's `crash_long_pm_exit`)
    cap: bool = True
    # PREREG203 / 205 (production from the 2026-10-26 session): the insider book (production's insider_long — built while
    # `enable_insider_long` is on, `ins_on`)
    ins: bool = True
    # PREREG219-226 (production from the 2026-10-28 session): the fear book (production's fear_long — built while
    # `enable_fear_long` is on, `fear_on`): the dip signals whose implied volatility is in their own top fifth
    fear: bool = True
    crash_pm_exit: bool = field(default_factory=lambda: _prod_crash_pm_exit())
    # PREREG154 (a RESEARCH book; rejected 2026-10-19, no production book): the QQQ sleeve — the capital no book uses
    # held in QQQ above its 200-session average, as lots that give way first (`strategies/sleeve.py`)
    sleeve: bool = False
    # PREREG95 (the default from 2026-10-13): "strat" = each block of `n_draws` consecutive seeds (block
    # seed // n_draws) one stratified set — every random decision spread evenly over the block's draws, each draw's law
    # unchanged; "iid" = each seed's draws independent (every study to PREREG95)
    draws: str = "strat"
    n_draws: int = 5
    # PREREG99: "guard" = production's halt guard at the short arms' entry passes — the DEFAULT from 2026-10-19
    # (AUDIT151's H_GUARD, -0.68 +- 0.34 pp a year; None = halts not modelled, the simulator before, `Mimic.off()`);
    # "wait" / "wait_half" / "backup" / "both" = H1a / H1b / H2 / H3. The halt model covers production's construction
    # only (the fall and ETF turns, no sell limits): any other needs ``halts=None``
    halts: Optional[str] = "guard"
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
    # never a slot). The default follows production's `enable_dip_long_filter` (on from the 2026-10-21 session)
    dip_filter: bool = field(default_factory=lambda: _prod_dip_filter())
    # PREREG108 (context activation, user 2026-10-14: "a model should understand the context and be activated if
    # historically it performs well in it"): a schedule file (`bank.schedule` per arm: month -> its active market
    # cells, `src/backtest/context.py`); an arm takes no piece decided in a cell its month's map leaves out (a month
    # absent = active everywhere). None = production's arms, always active
    activation: Optional[str] = None
    # PREREG110: research books added to the live combination (`register_research`); () = production's books only
    research: Tuple[str, ...] = ()
    # PREREG113 (risk protection; production from 2026-10-16, PREREG116's Q1): at most ``short_name_cap`` open shorts
    # of one NAME — per short arm (``short_name_scope`` "arm") or across the three short arms ("all") — through the
    # engine's group cap (each short piece's group = its name). None = no cap. The defaults are production's:
    # `sel_short_max_open_per_ticker` (0 = None) counting every arm's open shorts of the name ("all")
    short_name_cap: Optional[int] = field(default_factory=lambda: _prod_short_cap())
    short_name_scope: str = "all"
    # PREREG107 (the dip book's best picks first; production from 2026-10-16, PREREG116's Q1): its orders ranked by the
    # meta-model's walk-forward score ("score") or the deepest dip first ("rsi2"), only each day's ``dip_topk`` best
    # (None: every signal), and its slice count (None: production's `dip_long_account_slices`; as many open as
    # slices). The defaults are production's: `dip_long_daily_top` (0 = None, ranked by "rsi2") and `dip_long_rank`
    dip_rank: str = field(default_factory=lambda: _prod_dip_rank())
    dip_topk: Optional[int] = field(default_factory=lambda: _prod_dip_topk())
    dip_slices: Optional[int] = None
    # PREREG121 (the overnight session, IBKR's OVERNIGHT venue from the night of 2023-07-16): "off" = production as
    # simulated today (positions held unpriced from 20:00 to 04:00); "marks" = production's overnight slots mark every
    # position (the margin check runs there); "exits" = also the short arms' exits are judged there (`night_short`)
    overnight: str = "off"
    # PREREG121's X1: the dip and crash books sell at the after-hours tick right after the close that triggers their
    # exit (16:30 ET) instead of the next open; C1: the crash book buys at its signal session's 16:30 tick
    pm_exits: bool = False
    crash_pm_entry: bool = False
    # PREREG121's E1 / E2: the dip book's limit orders also work in the extended hours ("ext") and overnight
    # ("ext+night", IBKR's OVERNIGHT venue from 2023-07-16) — `strategies.dip.order_pieces_ext`; "off" = production
    dip_ext_orders: str = "off"
    # PREREG125: the live account's de-leveraging (trigger, target) as shares of equity — None = production's
    # (`live_account_delever_trigger` / `live_account_delever_target`)
    delever: Optional[Tuple[float, float]] = None
    # PREREG136 (user 2026-10-17: "Do the full account simulation for the 'After bad news where the stock fell on its
    # filing', and the 'vol shorts after positive news rated by the local LLM' held 1, 2, 3, 5, 8, or 10 more
    # sessions"): a short's own exit — not its squeeze cover — postponed k sessions when its hold had the rule's news
    # (`news_hold_piece`); the spec "<rule>:<scope>:<k>" (`news_hold_spec`); None = production's exits
    news_hold: Optional[str] = None
    # PREREG137 Amendment 1 (exploratory): a pickle of dip ORDER ids (`strategies.dip` oids, "NAME|first session")
    # that rank after every other signal of their day in the daily top pick (`dip.rank_pieces(demote=...)`) — the
    # earnings dips; None = production's ranking (``dip_earnings``). A file given REPLACES ``dip_earnings``
    dip_demote: Optional[str] = None
    # PREREG137's DEMOTE_E as production runs it (from the 2026-10-19 session, `src/signals/dip_earnings.py`): the dip
    # orders whose company released earnings (an 8-K Item 2.02) in the window of ``dip_earnings`` sessions before the
    # signal session through D's 09:30 rank after every other order of their day (`earnings_demoted`). The default is
    # production's: `dip_long_earnings_demote_sessions` while `enable_dip_long_earnings_demote` and the daily top pick
    # are on, else None (no demotion)
    dip_earnings: Optional[int] = field(default_factory=lambda: _prod_dip_earnings())
    # PREREG142 (the meta-model refit on the repaired data): a research file of walk-forward scores in PREREG96's map
    # format (`strategies.dip.filter_scores`: oid -> score) that ranks the daily top pick instead of the default
    # (`strategies.dip.scores_path`: the own-security panels' `OWN_SCORES` from 2026-10-18, PREREG96's `FILTER_SCORES`
    # with the stored panels); None = that default
    dip_scores: Optional[str] = None
    # PREREG147's S3 as production runs it (from the 2026-10-19 session): the daily top pick only among the dip orders
    # whose signal's dv20 is at least this many dollars (the pieces' ``dv20``; every order still scored on its $1B day's
    # inputs); the default is production's (`dip_long_order_min_dollar_volume` with the daily top pick on), None = none
    dip_min_order_dv: Optional[float] = field(default_factory=lambda: _prod_dip_order_floor())
    # PREREG209's KD+ as production runs it (from the 2026-10-26 session, `src/signals/dip_wsb.py`): (min titles, window
    # hours, cut ET) — on a day where any dip order (after the RP33 cut and the floor) was named in at least min titles
    # r/wallstreetbets titles created in the window before the cut of its first session, only those flagged orders
    # compete for the daily top pick (`wsb_preferred`; the history index of the production feed's store,
    # `dip_wsb.history_index`); the default is production's, None = no preference (`PRE209`)
    dip_wsb: Optional[Tuple[int, float, str]] = field(default_factory=lambda: _prod_dip_wsb())
    # AUDIT151 (the user's go-ahead 2026-10-19: "Yes please do 1 and 2"): the backtest's model of the market corrected —
    # each ON by default, `Mimic.off()` without them; measured one by one against the deployed configuration (20 paired
    # draws a set, pp a year) and together (68.0% a year instead of 71.6%, `studies/audit151`):
    # ``lend_intra`` (L_INTRA, -4.53): a short is lendable AT THE TRADE, not on the archive's morning file — an
    # archive-lendable pick, backup, extra or turn stays lendable with its pool bucket's probability, a set-B draw with
    # the 4th, the ETF arm's set-B rate x the 5th (`LEND_INTRA`; one keyed uniform per name and session, `draw_u`
    # "intra"); None = the morning file
    lend_intra: Optional[Tuple[float, float, float, float, float]] = LEND_INTRA
    # ``etf_reused`` (E_REUSE, +0.16): True = the ETF arm also ranks `REUSED_ETF` in their stock era (the research
    # universe's defect); False = never ranked, as live ranks only what is a product at the time
    etf_reused: bool = False
    # ``quote_costs`` (C_QUOTE, -4.32): every market leg at its REAL quoted half-spread and IBKR's exact fees
    # (`src/backtest/quotes.py`, `Strategy.fees="engine"`) instead of the ledger's liquidity-bucket averages
    quote_costs: bool = True
    # ``dip_open`` (D_OPEN, +0.59): the dip book's open fills at the price IBKR could give its resting order arriving
    # 09:30:30 (`strategies.dip.reprice_open`) instead of the opening print
    dip_open: bool = True
    # ``dividends`` (DIV, +0.18): the dividends a short pays and a long receives (`src/backtest/dividends.py`)
    dividends: bool = True
    # (D_WARM, +5.40, is the dip panel's data: `strategies.dip.OWN_PANEL` / `OWN_SCORES` carry set A's warm-up)
    # PREREG174 — MARGIN CALLS AS IBKR LIQUIDATES (the user 2026-10-21: "I want the backtest to do exactly the same, and,
    # if we can't be exact, approximate closely and be only slightly on the safe side"; the default from 2026-10-21,
    # `Mimic(**PRE174)` / `Mimic.off()` without): ``margin`` "ibkr" = IBKR's liquidation (`Rules(liquidation="ibkr")`)
    # judged continuously — ``margin_check`` "bars": each position's bar up to its adverse extreme ("ticks": at the
    # tick marks, a sensitivity) —, the most recently opened positions first (``margin_order``; "requirement", the
    # largest requirement first, a sensitivity) in 100-share lots, only until equity is back at the maintenance
    # requirement, no Soft Edge Margin (``margin_sem`` 0, the safe side; IBKR's documented 10% the bracket), the
    # end-of-day Reg T floor from 15:50 ET (``margin_regt``); ``margin_fill`` "path" = the check runs along each bar
    # from the previous marks to the extremes and every lot fills when the deficit appears, ``margin_slip`` (one
    # limit-up-limit-down band, 10%) against the account, never past the bar (Amendments 2-3); "extreme" = every lot
    # at the bar's extreme (the registered model, the conservative bracket); "all" = every position closed at once
    # (the simulator before; ``trigger`` then decides where it is judged: the tick marks by default)
    margin: str = "ibkr"
    margin_check: str = "bars"
    margin_sem: float = 0.0
    margin_order: str = "recent"
    margin_regt: bool = True
    margin_fill: str = "path"
    margin_slip: float = 0.10
    # PREREG175 — THE MARGIN CHECK SEES WHAT PRODUCTION'S CHECK SEES (the user 2026-10-21: "production and backtest needs
    # to liquidate like IBKR does"; the default from 2026-10-21, `Mimic(**PRE175)` without): ``margin_bars`` "all" =
    # an off-grid tick's high and low span EVERY stored bar since the piece's previous mark (production's check reads
    # every 30-minute bar completed since its last one: the hourly slots' first half-hours, the 09:00-09:30 pre-market
    # bar, the 19:30-20:00 bar and the 04:00 bar with the night's late prints no longer fall between marks); "tick" =
    # the bar ending at the tick only (PREREG174's). ``ext_longs``: the long books marked at production's off-grid
    # ticks — production marks every book at every tick; ("dip",) = PREREG174's
    margin_bars: str = "all"
    ext_longs: Tuple[str, ...] = ("dip", "crash", "cap", "ins", "fear")
    # PREREG193 — IBKR'S REFUSALS BY NAME (the default from 2026-10-24, `Mimic(**PRE193)` without): ``refusal`` "model" =
    # the refusal table's probability per (name, session) against one uniform per name (`is_refused`;
    # ``refusal_table`` its version, `REFUSAL_TABLES`); "rate" = `REFUSAL_RATE` per (name, session) (`refused`)
    refusal: str = "model"
    refusal_table: str = REFUSAL_TABLE
    # PREREG227 — THE PARITY AUDIT'S CORRECTIONS (the user 2026-10-27: "Do S1 to S8 now. For S6, both the simulated and
    # live accounts should be at $50,000."; the defaults from 2026-10-27, ``Mimic(**PRE227)`` without):
    # ``open_order`` (S1) — the capitulation and fear books' open fills at IBKR's attainable 09:30:30 price like the dip
    # book's (`strategies.dip.reprice_open` for the three books, on the extended open-fills table), a fill the rule finds
    # no price for that session no fill (the order keeps working), one the table cannot price stamped 09:30:30: at every
    # open the three books fill in production's order — dip, capitulation, fear (`PRIORITY`) — each cut to the room left;
    # ``crash_reenter`` (S2) — a crash piece for EVERY qualifying session (`strategies.crash.trades(every=True)`), the
    # one-at-a-time rule rejecting overlaps: a buy the account could not fund, or a long a margin call sold, is bought
    # again at the next qualifying open (production); ``vol_walk`` (S3) — with the halt model (``halts``), an unlendable
    # vol pick's backups are walked until the ACCOUNT opens one (`halts.VolMachine(walk=True)`'s chains, the engine's
    # ``alt_of``: production's `tracker.fallback`), a chain member or extra skipped once the arm opened its name that
    # session; inert with ``halts=None``; ``div_all`` (S4) — every live long book's dividends (`dividends.ALL_LONG`);
    # ``quote_all`` (S5) — the insider book's legs at their quotes like the crash book's (IBKR's fees once), the
    # capitulation and fear books' like the dip book's, a fill moved to a later touch without the open's spread;
    # ``start`` (S6) — the account's starting equity: production's `sel_short_account_initial` ($50,000 from the
    # 2026-10-20 fresh start). S7 (the vol / fall / ETF arms' parameters and the turns read from production's settings)
    # and S8 (the capitulation and fear signals keep one whose exit lies past the data with `strategies.dip.OPEN_END`)
    # have no switch: they change nothing of sets A0 / A / B
    open_order: bool = True
    crash_reenter: bool = True
    vol_walk: bool = True
    div_all: bool = True
    quote_all: bool = True
    start: float = field(default_factory=lambda: float(settings.sel_short_account_initial))

    def __post_init__(self):
        if self.entry and not self.r201:
            raise ValueError("entry at the pick bar mimics the ledger, which opens at the live price: it needs r201")
        for name in ("open_order", "crash_reenter", "vol_walk", "div_all", "quote_all"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, not {getattr(self, name)!r}")
        v = self.start
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError(f"start must be a positive amount of dollars, not {v!r}")
        d = self.delever
        if d is not None and not (isinstance(d, tuple) and len(d) == 2 and 0 < float(d[1]) <= float(d[0])):
            raise ValueError(f"delever must be None or (trigger, target) with 0 < target <= trigger, not {d!r}")
        if self.news_hold is not None:
            news_hold_spec(self.news_hold)          # PREREG136: refuses a malformed spec
        if self.dip_demote is not None and not Path(self.dip_demote).exists():
            raise ValueError(f"dip_demote: no file {self.dip_demote!r}")
        if self.dip_scores is not None and not Path(self.dip_scores).exists():
            raise ValueError(f"dip_scores: no file {self.dip_scores!r}")
        w = self.dip_wsb
        if w is not None and not (isinstance(w, tuple) and len(w) == 3 and int(w[0]) >= 1 and float(w[1]) > 0
                                  and re.fullmatch(r"\d{1,2}:\d{2}", str(w[2]))):
            raise ValueError(f"dip_wsb must be None or (min titles >= 1, window hours > 0, 'HH:MM'), not {w!r}")
        e = self.dip_earnings
        if e is not None and (isinstance(e, bool) or not isinstance(e, int) or e < 1):
            raise ValueError(f"dip_earnings must be None or a whole number of sessions >= 1, not {e!r}")
        if set(self.short_limit_arms) - {"vol", "etf", "fall"}:
            raise ValueError(f"short_limit_arms must be among vol / etf / fall, not {self.short_limit_arms}")
        if self.lend_model not in LEND_P_FILES:
            raise ValueError(f"lend_model must be one of {sorted(LEND_P_FILES)}, not {self.lend_model!r}")
        if self.fee_version not in FEE_PRED_FILES:
            raise ValueError(f"fee_version must be one of {sorted(FEE_PRED_FILES)}, not {self.fee_version!r}")
        if self.refusal not in ("model", "rate"):
            raise ValueError(f"refusal must be 'model' or 'rate', not {self.refusal!r}")
        if self.refusal_table not in REFUSAL_TABLES:
            raise ValueError(f"refusal_table must be one of {sorted(REFUSAL_TABLES)}, not {self.refusal_table!r}")
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
        li = self.lend_intra
        if li is not None and not (isinstance(li, tuple) and len(li) == 5
                                   and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                           and 0.0 < float(x) <= 1.0 for x in li)):
            raise ValueError(f"lend_intra must be None or five probabilities in (0, 1], not {li!r}")
        for name in ("etf_reused", "quote_costs", "dip_open", "dividends"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, not {getattr(self, name)!r}")
        if not isinstance(self.etf_vix, bool):
            raise ValueError(f"etf_vix must be True or False, not {self.etf_vix!r}")
        for name in ("dip_filter", "cap", "crash_pm_exit", "fear"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, not {getattr(self, name)!r}")
        if self.margin not in ("ibkr", "all"):
            raise ValueError(f"margin must be 'ibkr' or 'all', not {self.margin!r}")
        if self.margin_check not in ("bars", "ticks"):
            raise ValueError(f"margin_check must be 'bars' or 'ticks', not {self.margin_check!r}")
        if self.margin_order not in ("recent", "requirement"):
            raise ValueError(f"margin_order must be 'recent' or 'requirement', not {self.margin_order!r}")
        v = self.margin_sem
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) < 1.0:
            raise ValueError(f"margin_sem must be a share of equity in [0, 1), not {v!r}")
        if not isinstance(self.margin_regt, bool):
            raise ValueError(f"margin_regt must be True or False, not {self.margin_regt!r}")
        if self.margin_fill not in ("path", "extreme"):
            raise ValueError(f"margin_fill must be 'path' or 'extreme', not {self.margin_fill!r}")
        v = self.margin_slip
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"margin_slip must be a share in [0, 1], not {v!r}")
        if self.margin_bars not in ("all", "tick"):
            raise ValueError(f"margin_bars must be 'all' or 'tick', not {self.margin_bars!r}")
        el = self.ext_longs
        if (not isinstance(el, tuple) or any(not isinstance(n, str) for n in el) or len(set(el)) != len(el)
                or set(el) - {"dip", "crash", "cap", "ins", "fear"}):
            raise ValueError(f"ext_longs must be a tuple of distinct books among dip / crash / cap / ins / fear, not "
                             f"{el!r}")
        if self.activation is not None and not (isinstance(self.activation, str) and Path(self.activation).is_file()):
            raise ValueError(f"activation must be None or a schedule file, not {self.activation!r}")
        r = self.research
        if (not isinstance(r, tuple) or any(not isinstance(n, str) for n in r) or len(set(r)) != len(r)
                or set(r) & (set(BOOK) | set(SHORT_ARMS))):
            raise ValueError(f"research must be a tuple of distinct research book names, not {r!r}")
        c = self.short_name_cap
        if c is not None and (isinstance(c, bool) or not isinstance(c, int) or c < 1):
            raise ValueError(f"short_name_cap must be None or a whole number >= 1, not {c!r}")
        if self.short_name_scope not in ("arm", "all"):
            raise ValueError(f"short_name_scope must be 'arm' or 'all', not {self.short_name_scope!r}")
        if self.dip_rank not in ("rsi2", "score"):
            raise ValueError(f"dip_rank must be 'rsi2' or 'score', not {self.dip_rank!r}")
        if self.dip_ext_orders not in ("off", "ext", "ext+night"):
            raise ValueError(f"dip_ext_orders must be 'off', 'ext' or 'ext+night', not {self.dip_ext_orders!r}")
        if self.overnight not in ("off", "marks", "exits"):
            raise ValueError(f"overnight must be 'off', 'marks' or 'exits', not {self.overnight!r}")
        for name in ("dip_topk", "dip_slices"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError(f"{name} must be None or a whole number >= 1, not {v!r}")
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
                   etf_vix=True, short_name_cap=None, dip_rank="rsi2", dip_topk=None, dip_slices=10,
                   dip_earnings=None, halts=None, lend_intra=None, etf_reused=True, quote_costs=False,
                   dip_open=False, dividends=False, dip_filter=False, crash_pm_exit=False, cap=False, margin="all",
                   refusal="rate", ins=False, dip_wsb=None, fear=False, **PRE227)

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
    """IBKR refuses any opening short in the name that session (what-if error 201) — at random, ``rate`` of the
    (name, session) draws: `Mimic(**PRE193)`'s rule."""
    return draw_u("refuse", plain(tkn), int(day), seed, n) < rate


def refusal_path(version: str = REFUSAL_TABLE) -> Path:
    """The refusal table of ``version`` (`REFUSAL_TABLES`), beside the short stores."""
    return Path(settings.backtest_research_dir) / "optvol" / "refusal" / REFUSAL_TABLES[version]


@lru_cache(maxsize=4)
def refusal_table(version: str = REFUSAL_TABLE) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], float]:
    """({symbol: (its rows' session day numbers, sorted; their probabilities)}, the stand-in probability of a name
    without a row) — the table a study built (``rows``: sym, dn, p — one row per (sym, dn); ``p_default``)."""
    obj = pd.read_pickle(refusal_path(version))
    by: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in obj["rows"].groupby("sym", sort=False):
        g = g.sort_values("dn", kind="mergesort")
        by[str(sym)] = (g["dn"].to_numpy(np.int64), g["p"].to_numpy(float))
    return by, float(obj["p_default"])


def refusal_p(sym: str, day: int, version: str = REFUSAL_TABLE) -> float:
    """The table's probability that IBKR refuses an opening short in ``sym`` at session ``day``: its own row, else its
    latest earlier row (the state persists), else the table's stand-in."""
    by, p0 = refusal_table(version)
    hit = by.get(str(sym))
    if hit is None:
        return p0
    dn, p = hit
    k = int(np.searchsorted(dn, int(day), side="right")) - 1
    return float(p[k]) if k >= 0 else p0


def is_refused(mim: "Mimic", tkn, day: int) -> bool:
    """IBKR refuses any opening short in the name that session, by ``mim.refusal``: "model" — the refusal table's
    probability against ONE uniform per name (`draw_u` "refuse", session 0: a lasting state); "rate" — `refused`."""
    if mim.refusal == "rate":
        return refused(tkn, int(day), mim.seed, n=mim.strat_n)
    sym = plain(tkn)
    return draw_u("refuse", sym, 0, mim.seed, mim.strat_n) < refusal_p(sym, int(day), mim.refusal_table)


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
                memo2[k] = is_refused(mim, k[0], k[1])
            if memo2[k]:
                out[i] = False
    if mim.lend and mim.lend_intra is not None:
        out = _lend_intra(eng, mim, out, dn, sym)
    return out


def pool_survival(pool_usd: float, q: Sequence[float]) -> float:
    """AUDIT151's L_INTRA: the probability a short the morning file lent with ``pool_usd`` of shares available is still
    lendable at the trade — ``q`` by bucket (<= $250k, $250k-1M, > $1M); no file in force (NaN): unknown, kept."""
    if not np.isfinite(pool_usd):
        return 1.0
    return q[0] if pool_usd <= 2.5e5 else (q[1] if pool_usd <= 1e6 else q[2])


def _lend_intra(eng, mim: Mimic, out: np.ndarray, dn: np.ndarray, sym: Sequence[str]) -> np.ndarray:
    """``lend_all``'s rows lendable on the MORNING file (IBKR's community archive, one file a day, ~07:15 ET) or on set
    B's draw (a model of those files), before our own per-tick archive: each kept with ``mim.lend_intra``'s
    probability — its pool bucket's on an archive day (the pool at the pick, `VolEngine.lend_mask`'s ``_B_pick``), set
    B's otherwise — against one keyed uniform per (name, session, draw) (`draw_u` "intra"): a name lost at one bar is
    lost that whole day."""
    rows = np.flatnonzero(out & (dn < dnum(OWN_ARCHIVE_FROM)))
    if not len(rows):
        return out
    eng.lend_mask(at="pick")
    B = getattr(eng, "_B_pick", None)
    usd = np.asarray(B[1], float) if B is not None else np.full(len(out), np.nan)
    qp, qb, hi_arch = tuple(mim.lend_intra[:3]), float(mim.lend_intra[3]), dnum(LEND_ARCHIVE_END)
    memo: Dict[Tuple[str, int], float] = {}
    for i in rows:
        d = int(dn[i])
        q = pool_survival(float(usd[i]), qp) if d <= hi_arch else qb
        k = (sym[i], d)
        u = memo.get(k)
        if u is None:
            u = memo[k] = draw_u("intra", k[0], d, int(mim.seed), mim.strat_n)
        if u >= q:
            out[i] = False
    return out


def lendable_at_trade(mim: Mimic) -> Callable:
    """`borrow_history.lendable_at` as ``mim`` judges a short when it ENTERS (a turn, the ETF arm's archive days): an
    archive day's file is the MORNING file, so a lendable answer stays lendable with its pool bucket's probability
    (``lend_intra``; the same keyed uniform as `lend_all`'s); without ``lend_intra`` the file's answer itself."""
    from src.data.deep import borrow_history as BH
    if mim.lend_intra is None:
        return BH.lendable_at
    qp, hi_arch = tuple(mim.lend_intra[:3]), dnum(LEND_ARCHIVE_END)

    def lendable(ticker, when, price, min_usd=None, **kw):
        ok, s = BH.lendable_at(ticker, when, price, min_usd, **kw)
        if ok is True:
            w = pd.Timestamp(when)
            d = dnum(w.tz_convert(ET).date() if w.tzinfo else w.date())
            if d <= hi_arch:
                pool = float(s.get("available") or 0.0) * float(price)
                if draw_u("intra", plain(ticker), d, int(mim.seed), mim.strat_n) >= pool_survival(pool, qp):
                    return False, s
        return ok, s
    return lendable


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


def tick_marks(b: dict, t0: int, t1: int,
               grid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(instant, price, high, low) of every off-grid tick strictly inside (t0, t1) that has a fresh stored price: an
    extended-hours tick the bar ending at it within its 30 minutes, the 09:30 tick the session's opening print.
    ``grid`` (PREREG175, ``Mimic.margin_bars`` "all"): the piece's own marks — each tick's high and low then span
    EVERY stored bar ending since the piece's previous mark (its last grid mark or the tick before, at most ``t0``
    back), its own price included, as production's margin check reads every bar completed since its last check; a tick
    at a grid mark's instant is left out (the grid mark stands). None: the bar ending at the tick (PREREG174's)."""
    out = _tick_marks(b, t0, t1)
    if grid is None or not len(out[0]):
        return out
    T, P, H, L = out
    g = np.asarray(grid, np.int64)
    keep = ~np.isin(T, g)
    T, P, H, L = T[keep], P[keep], H[keep].copy(), L[keep].copy()
    prev = int(t0)
    for k in range(len(T)):
        t = int(T[k])
        jg = int(np.searchsorted(g, t, side="left")) - 1
        start = max(prev, int(g[jg]) if jg >= 0 else int(t0), int(t0))
        i0 = int(np.searchsorted(b["end"], start, side="right"))
        i1 = int(np.searchsorted(b["end"], t, side="right"))
        if i1 > i0:
            H[k] = max(float(H[k]), float(np.max(b["h"][i0:i1])))
            L[k] = min(float(L[k]), float(np.min(b["l"][i0:i1])))
        prev = t
    return T, P, H, L


def _tick_marks(b: dict, t0: int, t1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`tick_marks` without ``grid``: each tick's own bar (PREREG174's construction)."""
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


def extend_short(p: dict, bars: ExtBars, settle: Callable[[int], int], span: bool = False) -> Tuple[dict, str]:
    """A short piece under production's off-grid ticks: the first tick inside the trade where the target (``lvl``),
    the cover (``cover_mult`` x the entry) or the volatility exit (the latest regular-hours ATR% at most ``vn`` x the
    pick's, in profit) fires becomes the exit (at that tick's price, its borrow recomputed); the ticks before the exit
    become marks. ``span`` (PREREG175): each tick's high and low span every bar since the piece's previous mark
    (`tick_marks`' ``grid``). Returns (piece, outcome): "exit", "marks", "no_ticks", "no_bars", "no_scale"."""
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = price_scale(p, b)
    if s is None:
        return p, "no_scale"
    T, px, hi, lo = tick_marks(b, int(p["ens"]), int(p["xns"]),
                               grid=np.asarray(p["pt"], np.int64) if span else None)
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


# ── PREREG121: the overnight session (IBKR's OVERNIGHT venue, from the night of 2023-07-16) ──────────────────────
OVERNIGHT_DIR = Path("cache/ml/overnight_ibkr/bars30m")
_EXT_DIP: Dict[tuple, List[dict]] = {}       # E1 / E2's dip order pieces of the last window (shared, never modified)
OVERNIGHT_FROM = pd.Timestamp("2023-07-16 20:00", tz=ET).value     # the venue's first night in IBKR's history
NIGHT_EXIT_NOTIONAL = 7_500.0     # an overnight exit needs the bar's dollar volume at least this (a short's size bound)
NIGHT_MISMATCH = 0.25             # a night whose first bar differs from the store's 19:30-20:00 close by more than this
# PRICE REGIMES (2026-10-16): one scale per name mis-prices the nights of another price regime — a corporate action one
# source adjusted and the other did not (WDC: IBKR's nights at 0.757x the store's until the Sandisk spin-off of
# 2025-02, read as a -24% dip every night). A night is kept only while the median of its name's last
# `NIGHT_REGIME_CHECKS` checked nights (its own included; point in time) is within `NIGHT_REGIME_TOL` of the scale,
# the newest of them at most `NIGHT_REGIME_DAYS` old, and its own check (when it has one) within `NIGHT_ONE_TOL`
NIGHT_REGIME_CHECKS = 5
NIGHT_REGIME_TOL = 0.03
NIGHT_REGIME_DAYS = 21
NIGHT_ONE_TOL = 0.10


@lru_cache(maxsize=1)
def night_slots() -> Tuple[int, ...]:
    """Production's overnight slots (minutes after midnight ET; `runner._session_slots`)."""
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    return tuple(sorted(t.hour * 60 + t.minute for t, kind in slots if kind == "overnight"))


TRF_04 = pd.Timestamp("2026-03-30")     # the FINRA TRF opens at 04:00 ET from this session (08:00 before)


def late_print_bar(start_ns) -> np.ndarray:
    """True for the stored 30-minute bar that carries the overnight session's LATE-REPORTED trades (PREREG121
    amendment 2): ATS trades executed 00:00-04:00 ET reach the tape when the FINRA TRF opens — 08:00 ET until
    2026-03-27, 04:00 ET from 2026-03-30 — stamped then, with the conditions of real-time extended-hours trades, so
    that bar's open, high and low are often the night's prices (its volume too). ``start_ns``: bar starts (UTC ns)."""
    st = pd.DatetimeIndex(np.atleast_1d(np.asarray(start_ns, np.int64))).tz_localize("UTC").tz_convert(ET)
    hm = st.hour * 60 + st.minute
    new = st.tz_localize(None).normalize() >= TRF_04
    return np.asarray(np.where(new, hm == 4 * 60, hm == 8 * 60), bool)


def _night_key(et: pd.DatetimeIndex) -> np.ndarray:
    """Each overnight bar's night as the day number of its evening (a 01:00 bar belongs to the evening before)."""
    return ((pd.DatetimeIndex(et).tz_localize(None) - pd.Timedelta(hours=8)).normalize().asi8
            // (86_400 * 10**9)).astype(np.int64)


class NightBars:
    """IBKR's OVERNIGHT 30-minute bars of a name (`studies/overnight24/fetch_overnight.py`), in the extended store's
    price terms: each night's first bar (20:00) against the store's 19:30-20:00 close — a name whose nights differ by
    more than `NIGHT_MISMATCH` on more than half of them is another security (None); the bars are scaled by the median
    ratio of the matching nights. Bars before the venue's first night are dropped."""

    def __init__(self, ext: ExtBars, d: Optional[str] = None):
        self.ext, self.dir = ext, Path(d) if d is not None else OVERNIGHT_DIR
        self._m: Dict[str, Optional[dict]] = {}
        self.rejected: Dict[str, str] = {}
        self.dropped: Dict[str, int] = {}          # nights left out of another price regime, per name

    def get(self, name: str) -> Optional[dict]:
        if name not in self._m:
            self._m[name] = self._load(name)
        return self._m[name]

    def _load(self, name: str) -> Optional[dict]:
        p = self.dir / f"{plain(name)}.parquet"
        if str(name).endswith("@D") or not p.exists():
            return None
        import duckdb
        df = duckdb.query(f"select ts, open, high, low, close, volume from '{p.as_posix()}' order by ts").df()
        if df.empty:
            return None
        start = pd.DatetimeIndex(df["ts"]).tz_localize("UTC").asi8.astype(np.int64)
        keep = start >= OVERNIGHT_FROM
        df, start = df[keep].reset_index(drop=True), start[keep]
        if not len(start):
            return None
        b = self.ext.get(name)
        if b is None:
            self.rejected[name] = "no extended bars to check it against"
            return None
        et = pd.DatetimeIndex(start).tz_localize("UTC").tz_convert(ET)
        first = (et.hour == 20) & (et.minute == 0)                 # each night's 20:00-20:30 bar
        j = np.searchsorted(b["end"], start[first])                 # the store's bar ending at 20:00
        ok = (j < len(b["end"])) & (np.take(b["end"], np.minimum(j, len(b["end"]) - 1)) == start[first])
        if not ok.any():
            self.rejected[name] = "no night to check against the store"
            return None
        r = df["open"].to_numpy(float)[first][ok] / b["c"][j[ok]]
        good = np.abs(r - 1.0) <= NIGHT_MISMATCH
        if good.mean() < 0.5:
            self.rejected[name] = f"another security: {int((~good).sum())} of {len(r)} nights off by > 25%"
            return None
        k = float(np.median(r[good]))
        keep = self._regime_nights(start, et, first, ok, r / k)
        if not keep.any():
            self.rejected[name] = "no night in the store's price regime"
            return None
        if not keep.all():
            self.dropped[name] = int(len(np.unique(_night_key(et[~keep]))))
        df, start = df[keep].reset_index(drop=True), start[keep]
        return {"start": start, "end": start + BAR_NS, "o": df["open"].to_numpy(float) / k,
                "h": df["high"].to_numpy(float) / k, "l": df["low"].to_numpy(float) / k,
                "c": df["close"].to_numpy(float) / k, "v": df["volume"].to_numpy(float),
                "usd": df["close"].to_numpy(float) * df["volume"].to_numpy(float)}

    @staticmethod
    def _regime_nights(start: np.ndarray, et: pd.DatetimeIndex, first: np.ndarray, ok: np.ndarray,
                       rel: np.ndarray) -> np.ndarray:
        """Which bars belong to a night in the store's price regime (`NIGHT_REGIME_*`): ``rel`` = each checked night's
        first-bar ratio over the scale, in time order; the nights are keyed by their evening's date."""
        night = _night_key(et)
        checked = _night_key(et[first][ok])
        out = np.zeros(len(start), bool)
        for n in np.unique(night):
            j = int(np.searchsorted(checked, n, side="right"))              # the checks up to this night (included)
            if j == 0:
                continue
            last = checked[max(0, j - NIGHT_REGIME_CHECKS):j]
            if (n - last[-1]) > NIGHT_REGIME_DAYS:
                continue
            if abs(float(np.median(rel[max(0, j - NIGHT_REGIME_CHECKS):j])) - 1.0) > NIGHT_REGIME_TOL:
                continue
            if checked[j - 1] == n and abs(float(rel[j - 1]) - 1.0) > NIGHT_ONE_TOL:
                continue
            out[night == n] = True
        return out


def night_marks(n: Optional[dict], t0: int, t1: int) -> Tuple[np.ndarray, ...]:
    """(instant, price, high, low, dollar volume) of every production overnight slot strictly inside (t0, t1) whose
    IBKR bar ends at it (the extended store's terms)."""
    e = tuple(np.zeros(0, np.int64 if k == 0 else float) for k in range(5))
    if n is None or not len(n["end"]) or t1 <= t0:
        return e
    i0, i1 = int(np.searchsorted(n["end"], t0, side="right")), int(np.searchsorted(n["end"], t1, side="left"))
    if i1 <= i0:
        return e
    ends = n["end"][i0:i1]
    et = pd.DatetimeIndex(ends).tz_localize("UTC").tz_convert(ET)
    mins = et.hour * 60 + et.minute
    on = np.isin(mins, np.asarray(night_slots(), np.int64))
    idx = np.arange(i0, i1)[on]
    return (n["end"][idx].astype(np.int64), n["c"][idx], n["h"][idx], n["l"][idx], n["usd"][idx])


def _short_hits(p: dict, T: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(target, volatility-exit, cover) hits of a short piece at instants ``T`` and prices ``m`` — `extend_short`'s
    rules (the latest regular-hours ATR% before each instant)."""
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
    return hit_t, hit_v, hit_c


def night_short(p: dict, nights: NightBars, ext: ExtBars, settle: Callable[[int], int], mode: str) -> Tuple[dict, str]:
    """A short piece (already through `extend_short`) under the overnight ticks: ``mode`` "marks" merges them as marks
    before its exit; "exits" also judges its exits there (`extend_short`'s rules), an exit needing the bar's dollar
    volume at least `NIGHT_EXIT_NOTIONAL` — the first hit before the current exit becomes the exit."""
    n = nights.get(p["tkn"])
    if n is None:
        return p, "no_night_bars"
    b = ext.get(p["tkn"])
    s = price_scale(p, b) if b is not None else None
    if s is None:
        return p, "no_scale"
    T, px, hi, lo, usd = night_marks(n, int(p["ens"]), int(p["xns"]))
    if not len(T):
        return p, "no_night_ticks"
    m, hi, lo = px * s, hi * s, lo * s
    if mode == "exits":
        hit_t, hit_v, hit_c = _short_hits(p, T, m)
        hit = (hit_t | hit_v | hit_c) & (usd >= NIGHT_EXIT_NOTIONAL)
        if hit.any():
            k = int(np.argmax(hit))
            q = _merge(p, T[:k], m[:k], hi[:k], lo[:k], int(T[k]))
            q["xns"], q["x"] = int(T[k]), float(m[k])
            q["why"] = "night_" + ("target" if hit_t[k] else "volnorm" if hit_v[k] else "cap")
            _reborrow(q, settle)
            return q, "night_exit"
    return _merge(p, T, m, hi, lo, None), "night_marks"


def night_long(p: dict, nights: NightBars, ext: ExtBars) -> Tuple[dict, str]:
    """A long piece's overnight tick marks (its exits stay the rule's)."""
    if int(p.get("ord_k", 0)) < 0:
        return p, "placeholder"
    n = nights.get(p["tkn"])
    if n is None:
        return p, "no_night_bars"
    b = ext.get(p["tkn"])
    s = price_scale(p, b) if b is not None else None
    if s is None:
        return p, "no_scale"
    T, px, hi, lo, _ = night_marks(n, int(p["ens"]), int(p["xns"]))
    if not len(T):
        return p, "no_night_ticks"
    return _merge(p, T, px * s, hi * s, lo * s, None), "night_marks"


# ── PREREG121: the after-hours tick acting on the close (X1, C1) ─────────────

def _session_before(b: dict, t: int) -> Optional[pd.Timestamp]:
    """The ET date of the last regular-hours bar starting before instant ``t``."""
    j = int(np.searchsorted(b["start"], int(t))) - 1
    while j >= 0 and not b["rth"][j]:
        j -= 1
    if j < 0:
        return None
    return pd.Timestamp(int(b["start"][j]), unit="ns", tz="UTC").tz_convert(ET).normalize()


def _pm_bar(b: dict, day: pd.Timestamp) -> Optional[int]:
    """The index of ``day``'s 16:00-16:30 after-hours bar (None: no print)."""
    t = pd.Timestamp(f"{day.date()} 16:00", tz=ET).value
    k = int(np.searchsorted(b["start"], t))
    return k if k < len(b["start"]) and b["start"][k] == t else None


def pm_exit(p: dict, bars: ExtBars, cost_bp: Optional[Callable] = None) -> Tuple[dict, str]:
    """X1: a long piece whose exit is a session's 09:30 open is sold instead at the AFTER-HOURS tick right after the
    close that triggered it (16:30 ET of the session before that open: the 16:00-16:30 bar's close); its marks after
    it dropped; with ``cost_bp`` (the ledger's resolver) its exit leg costs the after-hours leg. Unchanged when the
    exit is no open (a delisted name's last close), without that bar or a price scale."""
    if int(p.get("ord_k", 0)) < 0:
        return p, "placeholder"
    x_et = pd.Timestamp(int(p["xns"]), unit="ns", tz="UTC").tz_convert(ET)
    if (x_et.hour, x_et.minute) != (9, 30):
        return p, "not_an_open"
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = price_scale(p, b)
    if s is None:
        return p, "no_scale"
    day = _session_before(b, int(p["xns"]))
    k = _pm_bar(b, day) if day is not None else None
    if k is None:
        return p, "no_pm_bar"
    xns = int(b["end"][k])
    if xns <= int(p["ens"]):
        return p, "before_entry"
    z = np.zeros(0)
    q = _merge(p, np.zeros(0, np.int64), z, z, z, xns)
    q["xns"], q["x"], q["why"] = xns, float(b["c"][k]) * s, "pm_exit"
    if cost_bp is not None:
        q["hs_out"] = float(cost_bp("SELL", q["x"], xns, p.get("dv20")))
    if "days" in q:
        q["days"] = (xns - int(q["ens"])) / 86400e9
    return q, "pm_exit"


def pm_entry(p: dict, bars: ExtBars) -> Tuple[dict, str]:
    """C1: a crash piece bought at the AFTER-HOURS tick of its signal session (16:30 ET of the session before its
    entry open: the 16:00-16:30 bar's close) instead of that open. Unchanged without that bar or a price scale."""
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = price_scale(p, b)
    if s is None:
        return p, "no_scale"
    day = _session_before(b, int(p["ens"]))
    k = _pm_bar(b, day) if day is not None else None
    if k is None:
        return p, "no_pm_bar"
    q = dict(p)
    q["ens"], q["e"], q["entry_why"] = int(b["end"][k]), float(b["c"][k]) * s, "pm_entry"
    if "days" in q:
        q["days"] = (int(q["xns"]) - int(q["ens"])) / 86400e9
    return q, "pm_entry"


def extend_long(p: dict, bars: ExtBars, span: bool = False) -> Tuple[dict, str]:
    """A long piece's off-grid tick marks (its exits stay the rule's); ``span`` as `extend_short`'s."""
    if int(p.get("ord_k", 0)) < 0:
        return p, "placeholder"
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = price_scale(p, b)
    if s is None:
        return p, "no_scale"
    T, px, hi, lo = tick_marks(b, int(p["ens"]), int(p["xns"]),
                               grid=np.asarray(p["pt"], np.int64) if span else None)
    if not len(T):
        return p, "no_ticks"
    return _merge(p, T, px * s, hi * s, lo * s, None), "marks"


# ── the ledger's costs ───────────────────────────────────────────────────────

# ── PREREG136: the shorts held longer after news ─────────────────────────────────────────────────────────────────
NEWS_HOLD_DIR = Path("cache/ml/news_hold")        # the inputs (studies/prereg136/build136.py): filings, tones, names
NEWS_HOLD_RULES = {"fb": SHORT_ARMS, "lg": ("vol",)}         # the arms each rule may act on
NEWS_HOLD_SCOPES = {"vol": ("vol",), "shorts": SHORT_ARMS}
NEWS_LAG = 30 * 60 * 10**9                        # a news item acts on the marks at least 30 minutes after it


def news_hold_spec(spec: str) -> Tuple[str, Tuple[str, ...], int]:
    """(rule, arms, k) of a `Mimic.news_hold` spec "<rule>:<scope>:<k>": rule "fb" (the stock fell on its last 8-K of
    the hold) or "lg" (the local LLM reads the hold's headlines as positive; the vol arm only), scope "vol" / "shorts",
    k the sessions added (1 .. 20)."""
    parts = str(spec).split(":")
    if len(parts) != 3 or parts[0] not in NEWS_HOLD_RULES or parts[1] not in NEWS_HOLD_SCOPES:
        raise ValueError(f"news_hold must be '<fb|lg>:<vol|shorts>:<k>', not {spec!r}")
    try:
        k = int(parts[2])
    except ValueError:
        raise ValueError(f"news_hold's k must be a whole number of sessions, not {parts[2]!r}") from None
    if not 1 <= k <= 20:
        raise ValueError(f"news_hold's k must be 1 .. 20 sessions, not {k}")
    arms = tuple(a for a in NEWS_HOLD_SCOPES[parts[1]] if a in NEWS_HOLD_RULES[parts[0]])
    if not arms:
        raise ValueError(f"news_hold rule {parts[0]!r} acts on none of scope {parts[1]!r}'s arms")
    return parts[0], arms, k


class NewsInputs:
    """PREREG136's inputs (`NEWS_HOLD_DIR`), loaded once per directory: each company's 8-K acceptances (ns UTC,
    sorted; ``filings.pkl``), its LLM-rated headlines (ns, tone -1 / 0 / +1; ``llm.pkl``) and the name -> company map
    (``names.pkl``: set A0's symbols by identity, the delisted companies' segments, the single-stock funds'
    underlyings)."""

    _loaded: Dict[str, "NewsInputs"] = {}

    def __init__(self, d: Path = NEWS_HOLD_DIR):
        import pickle
        d = Path(d)
        self.filings: Dict[str, np.ndarray] = pickle.load(open(d / "filings.pkl", "rb"))
        self.llm: Dict[str, Tuple[np.ndarray, np.ndarray]] = pickle.load(open(d / "llm.pkl", "rb"))
        nm = pickle.load(open(d / "names.pkl", "rb"))
        self.a0map: Dict[str, str] = nm["a0map"]
        self.delisted: Dict[str, List[Tuple[str, int, int]]] = nm["delisted"]
        self.under: Dict[str, str] = nm["under"]

    @classmethod
    def get(cls, d: Path = NEWS_HOLD_DIR) -> "NewsInputs":
        key = str(Path(d).resolve())
        if key not in cls._loaded:
            cls._loaded[key] = cls(d)
        return cls._loaded[key]

    def company(self, tkn, ens: int, a0: bool) -> Optional[str]:
        """The company's key in the filings / news stores: today's ticker or ``XYZ_DELISTED``; None for a fund that is
        no single stock's or a name the stores do not hold."""
        sym, _, kind = str(tkn).partition("@")
        if kind == "E":
            return self.under.get(sym)
        if kind == "D":
            day = int(A.day_numbers([int(ens)])[0])
            return next((t for t, s0, s1 in self.delisted.get(sym, []) if s0 <= day <= s1), None)
        if a0 and sym in self.a0map:
            return self.a0map[sym]
        return sym


def _regular_close_marks(t: np.ndarray) -> np.ndarray:
    """Which marks are regular-hours bar closes (bar ends 10:00 .. 16:00 ET; not the 09:30 open, not an extended
    slot)."""
    et = pd.DatetimeIndex(np.asarray(t, np.int64)).tz_localize("UTC").tz_convert(ET)
    m = et.hour * 60 + et.minute
    return np.asarray((m >= 600) & (m <= 960) & (et.minute % 30 == 0))


def fell_on_filing(p: dict, filings: Optional[np.ndarray]) -> Optional[bool]:
    """PREREG136's FB news: the last 8-K accepted after the entry and at least `NEWS_LAG` before the exit — its
    reaction = the piece's first mark at least `NEWS_LAG` after the acceptance (at or before the exit) against its last
    regular-hours close mark at or before the acceptance (the entry price when none). None = no such filing (or no
    mark to read its reaction), True = the stock fell, False = it did not."""
    if filings is None or not len(filings):
        return None
    ens, xns = int(p["ens"]), int(p["xns"])
    f = np.asarray(filings, np.int64)
    f = f[(f > ens) & (f <= xns - NEWS_LAG)]
    if not len(f):
        return None
    a = int(f[-1])
    t, px = np.asarray(p["pt"], np.int64), np.asarray(p["pc"], float)
    after = np.flatnonzero((t >= a + NEWS_LAG) & (t <= xns))
    if not len(after):
        return None
    before = np.flatnonzero((t <= a) & _regular_close_marks(t))
    ref = float(px[before[-1]]) if len(before) else float(p["e"])
    return bool(float(px[after[0]]) < ref)


def llm_positive(p: dict, rated: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[bool]:
    """PREREG136's LG news: the mean LLM tone of the headlines published after the entry and at least `NEWS_LAG`
    before the exit is above 0. None = no rated headline in the hold."""
    if rated is None:
        return None
    ns, tone = np.asarray(rated[0], np.int64), np.asarray(rated[1], float)
    m = (ns > int(p["ens"])) & (ns <= int(p["xns"]) - NEWS_LAG) & np.isfinite(tone)
    if not m.any():
        return None
    return bool(tone[m].mean() > 0)


def robust_scale(p: dict, b: dict, tol: float = 0.02) -> Optional[float]:
    """`price_scale` robust to a few stray stored bars (PREREG136 Amendment 2): the median ratio of the piece's marks
    to the stored closes ending at the same instants, refused when the 10th-90th percentile spread of the ratios
    exceeds ``tol`` (the full range refuses a piece over one bad bar). Used by the news hold alone — the live
    construction's `extend_short` keeps `price_scale`."""
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
    if not len(r):
        return None
    q10, q90 = np.percentile(r, [10, 90])
    if q90 / q10 - 1.0 > tol:
        return None
    return float(np.median(r))


def news_extend(p: dict, k: int, bars: ExtBars, settle: Callable[[int], int],
                nights: Optional["NightBars"] = None) -> Tuple[dict, str]:
    """The short piece held ``k`` sessions past its exit: the new exit is the first mark at or after the same ET clock
    instant on the k-th session after the exit's day — or earlier the first mark at/over its cover multiple of the
    entry —, the marks in between (the stored regular-hours closes, the 09:30 opens, the extended slots and, with
    ``nights``, the overnight slots, in the piece's terms) merged, its borrow re-computed through the new exit day."""
    b = bars.get(p["tkn"])
    if b is None:
        return p, "no_bars"
    s = robust_scale(p, b)
    if s is None:
        return p, "no_scale"
    x0 = int(p["xns"])
    ex = pd.Timestamp(x0).tz_localize("UTC").tz_convert(ET)
    exday = int((ex.normalize().tz_localize(None) - EPOCH).days)
    rd = np.unique(b["day"][b["rth"]])
    after = rd[rd > exday]
    if len(after) < int(k):
        return p, "no_session"
    tgt = int((EPOCH + pd.Timedelta(days=int(after[int(k) - 1]), hours=ex.hour, minutes=ex.minute))
              .tz_localize(ET).value)
    hz = tgt + 2 * DAY_NS
    m = b["rth"] & (b["end"] > x0) & (b["end"] <= hz)
    T, P, H, L, K = [b["end"][m]], [b["c"][m]], [b["h"][m]], [b["l"][m]], [np.zeros(int(m.sum()), np.int8)]
    t2, p2, h2, l2 = tick_marks(b, x0, hz)
    T += [t2]; P += [p2]; H += [h2]; L += [l2]; K += [np.ones(len(t2), np.int8)]
    if nights is not None:
        n = nights.get(p["tkn"])
        if n is not None:
            t3, p3, h3, l3, _usd = night_marks(n, x0, hz)
            T += [t3]; P += [p3]; H += [h3]; L += [l3]; K += [np.full(len(t3), 2, np.int8)]
    T, P, H, L, K = (np.concatenate(v) for v in (T, P, H, L, K))
    if not len(T):
        return p, "no_marks"
    o = np.lexsort((K, T))
    T, P, H, L = T[o].astype(np.int64), P[o] * s, H[o] * s, L[o] * s
    keep = np.r_[True, T[1:] != T[:-1]]              # one price an instant (a regular close first)
    T, P, H, L = T[keep], P[keep], H[keep], L[keep]
    jt = np.flatnonzero(T >= tgt)
    if not len(jt):
        return p, "no_target_mark"
    j = int(jt[0])
    cm = p.get("cover_mult")
    why = "news_hold"
    if cm:
        hit = np.flatnonzero(P[:j + 1] >= float(cm) * float(p["e"]))
        if len(hit):
            j, why = int(hit[0]), "news_cap"
    q = _merge(p, T[:j], P[:j], H[:j], L[:j], None)
    q["xns"], q["x"], q["why"] = int(T[j]), float(P[j]), why
    q["news_held_from"] = x0
    _reborrow(q, settle)
    return q, why


def news_hold_piece(p: dict, rule: str, k: int, nin: NewsInputs, a0: bool, bars: ExtBars,
                    settle: Callable[[int], int], nights: Optional["NightBars"] = None) -> Tuple[dict, str]:
    """PREREG136: the short piece held ``k`` more sessions (`news_extend`) when its hold had the rule's news (FB:
    `fell_on_filing`, LG: `llm_positive`); a squeeze cover is never extended. Returns (piece, outcome)."""
    if "cap" in str(p.get("why", "")):
        return p, "cover"
    company = nin.company(p["tkn"], int(p["ens"]), a0)
    if company is None:
        return p, "no_company"
    news = fell_on_filing(p, nin.filings.get(company)) if rule == "fb" else llm_positive(p, nin.llm.get(company))
    if news is None:
        return p, "no_news"
    if not news:
        return p, "other_news"
    return news_extend(p, k, bars, settle, nights)


_DEMOTED: Dict[str, frozenset] = {}


def _demoted(path: Optional[str]) -> Optional[frozenset]:
    """PREREG137's demoted dip orders (a pickle of order ids), loaded once per path; None = none demoted."""
    if path is None:
        return None
    key = str(Path(path).resolve())
    if key not in _DEMOTED:
        import pickle
        _DEMOTED[key] = frozenset(str(o) for o in pickle.load(open(path, "rb")))
    return _DEMOTED[key]


_A0MAP: Dict[str, Dict[str, str]] = {}
_EARNINGS: Dict[tuple, frozenset] = {}


def _a0map(d: Path = NEWS_HOLD_DIR) -> Dict[str, str]:
    """Set A0's symbols -> today's tickers by identity (PREREG136's ``names.pkl``, `NewsInputs`' ``a0map``); {} when the
    file is missing (each symbol is then its own company key, logged once)."""
    key = str(Path(d).resolve())
    if key not in _A0MAP:
        p = Path(d) / "names.pkl"
        if p.exists():
            import pickle
            _A0MAP[key] = dict(pickle.load(open(p, "rb")).get("a0map") or {})
        else:
            from loguru import logger
            logger.warning(f"[mimic] no {p}: set A0's symbols read as today's tickers for the earnings demotion")
            _A0MAP[key] = {}
    return _A0MAP[key]


def wsb_preferred(pieces: Sequence[dict], spec: Tuple[int, float, str], log: Optional[dict] = None) -> List[dict]:
    """PREREG209's KD+ as production runs it (`dip_wsb`, `dip_long._daily_top`) on the simulator's dip order pieces
    (after the RP33 cut and the floor, before the daily top pick): on an order day (its ``oid``'s first session) where
    any order's name was in at least ``spec[0]`` r/wallstreetbets titles created in the ``spec[1]`` hours before
    ``spec[2]`` ET of that session, only those flagged orders stay (PREREG209's patched `rank_pieces`: the same pick as
    production's flagged-first order for the daily top 1). A session whose window the store does not cover flags
    nothing. ``log`` counts the days preferred and the orders left out."""
    import numpy as np
    from src.signals import dip_wsb as W
    idx = W.history_index()
    n_min, hours, cut = int(spec[0]), float(spec[1]), str(spec[2])
    hh, mm = (int(x) for x in cut.split(":"))

    def flagged(p) -> bool:
        day = pd.Timestamp(str(p.get("oid")).split("|")[-1]).date()
        c = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET_TZ).astimezone(timezone.utc)
        t1 = int(c.timestamp())
        t0 = int((c - timedelta(hours=hours)).timestamp())
        d = datetime.fromtimestamp(t0, timezone.utc).date()
        while d <= c.date():
            if d not in idx["days"]:
                return False
            d += timedelta(days=1)
        a = idx["tokens"].get(W.token_of(p["tkn"]))
        if a is None:
            return False
        return int(np.searchsorted(a, t1, "left") - np.searchsorted(a, t0, "left")) >= n_min
    fl = [flagged(p) for p in pieces]
    days = {}
    for p, f in zip(pieces, fl):
        days.setdefault(str(p.get("oid")).split("|")[-1], []).append(f)
    pref = {dd for dd, v in days.items() if any(v)}
    out = [p for p, f in zip(pieces, fl) if str(p.get("oid")).split("|")[-1] not in pref or f]
    if log is not None:
        log["wsb_days_preferred"] = len(pref)
        log["wsb_orders_left_out"] = len(pieces) - len(out)
    return out


def earnings_demoted(oids: Sequence[str], n: int, a0: bool = False) -> frozenset:
    """PREREG137's DEMOTE_E as production runs it (`src/signals/dip_earnings.py`) on the simulator's own dip orders
    (``oids``: "NAME|D"): the orders whose company released earnings — an 8-K / 8-K/A naming Item 2.02, in the deep
    store's and the delisted company store's filings (`dip_earnings.release_table`) — after 16:00 ET of the session
    ``n`` before the signal session and by D's 09:30 (`dip_earnings.window_ns`, `dip_earnings.demoted`). An order's
    company: a delisted name ``XYZ@D`` -> ``XYZ_DELISTED``; set A0's symbol (``a0``) -> today's ticker by identity
    (`_a0map`); else the name itself (the study's sim137.py mapping). Cached per (orders, n, a0)."""
    from src.signals import dip_earnings as DE
    ids = tuple(sorted({str(o) for o in oids}))
    key = (hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest(), int(n), bool(a0))
    if key not in _EARNINGS:
        a0map = _a0map() if a0 else {}
        comp: Dict[str, Optional[str]] = {}
        for oid in ids:
            name = oid.split("|")[0]
            comp[oid] = f"{name[:-2]}_DELISTED" if name.endswith("@D") else (a0map.get(name, name) if a0 else name)
        _EARNINGS[key] = frozenset(DE.demoted(comp, int(n), DE.release_table()))
    return _EARNINGS[key]


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

    def apply(self, p: dict, side: int, legs: bool = True, adv: Optional["DailyADV"] = None, quotes=None,
              quote_kw: Optional[dict] = None) -> dict:
        """The piece with the ledger's costs: its entry and exit legs' (``legs``) and the exit estimate its marks are
        net of (``hs_mark``: a regular-hours leg at the entry price). The liquidity class reads production's
        dollar-volume basis as of each leg's day (``adv``: `DailyADV`; else the piece's 20-session mean).
        ``quotes`` (AUDIT151's C_QUOTE, `src/backtest/quotes.py`): the market legs then at their REAL quoted
        half-spreads — the marks stay net of the ledger's estimate, as production marks its open trades;
        ``quote_kw``: `QuoteTable.apply`'s books (PREREG227's S5: ``limit_arms`` / ``market_arms``)."""
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
        if quotes is not None:
            q = quotes.apply(q, p, side, legs, **(quote_kw or {}))
        return q


# ── the live combination ─────────────────────────────────────────────────────

def rank_shorts(pieces: Sequence[dict]) -> None:
    """Live's order among a bar's short entries (`record_sel_short_trades` opens the journal's picks by ticker, then
    arm): each piece's ``rank`` = its ticker's alphabetical place x 10 + its arm's place in `sel_short.ARMS`."""
    from src.signals.sel_short import ARMS
    pos = {s: i for i, s in enumerate(sorted({plain(p["tkn"]) for p in pieces}))}
    for p in pieces:
        p["rank"] = float(pos[plain(p["tkn"])] * 10 + ARMS.index(p["strategy"]))
    # PREREG227's S3: one pick's fallback chain is tried in its walk's order, together, at its first member's place
    # (production walks a pick's backups right after the pick itself)
    head = {p["alt_of"]: p["rank"] for p in pieces if p.get("alt_of") is not None and int(p.get("alt_k") or 0) == 0}
    for p in pieces:
        if p.get("alt_of") is not None and p["alt_of"] in head:
            p["rank"] = float(head[p["alt_of"]]) + 1e-3 * int(p.get("alt_k") or 0)


def live_strategies(mim: Mimic) -> List[A.Strategy]:
    """The live account's strategies (PREREG61's production-mimicking setup): the short arms each at its own slice
    (production's `sim_account.arm_slices`: vol 1/20, ETF 1/16, fall 1/12 from 2026-10-12, PREREG80; the fall arm's
    ``fall_slices`` when set, PREREG104) with IBKR's house defaults and the stacking cap (``short_name_cap``), the dip
    limit book at 1/`dip_long_account_slices` (as many open; 1/6 from 2026-10-16, PREREG116's Q1 — ``dip_slices``
    overrides) giving way whole, the latest first, de-levered the latest first; ``house`` adds the live sizing's
    shrink, ``order`` one priority for the short arms. An arm left out of ``short_arms`` keeps its strategy (no
    pieces: its slice unused)."""
    from src.performance.sim_account import arm_slices
    out = []
    for n in ("vol", "etf", "fall"):
        k = float(mim.fall_slices) if (n == "fall" and mim.fall_slices is not None) else arm_slices(n)
        s = A.vol_short(n, slices=k)                       # each arm its own slice (production's, PREREG80)
        out.append(replace(s, priority=0 if mim.order else PRIORITY[n], quality=QUALITY[n], book=BOOK[n],
                           house_maint=float(settings.sel_short_account_house_maint),
                           house_init=float(settings.sel_short_account_house_init),
                           shrink_budget=float(settings.sel_short_account_house_init) if mim.house else None,
                           orders="fill" if mim.limited(n) else None,
                           # PREREG113: the stacking cap (each short piece's group is its name, `live_books`)
                           group_cap=mim.short_name_cap,
                           group_scope=None if mim.short_name_cap is None else
                           ((n,) if mim.short_name_scope == "arm" else SHORT_ARMS),
                           # AUDIT151's C_QUOTE: the legs at their real half-spreads + IBKR's exact fees on the shares
                           fees="engine" if quoted(mim) else None))
    k = int(mim.dip_slices or settings.dip_long_account_slices)   # as many open as slices (`live_account.size_long`)
    d = A.dip_long("dip", slices=float(k), max_open=k)
    out.append(replace(d, priority=PRIORITY["dip"], quality=QUALITY["dip"], preemptible=True, yield_whole=True,
                       yield_first="latest", delever_first="latest", delever_order=1, orders="fill", book=BOOK["dip"],
                       fees="engine" if mim.costs else None))
    if crash_on(mim):              # PREREG86 / 88: one long in the traded fund (TQQQ on QQQ's decisions), 1/4 of the
        rate = max(0.0, float(settings.crash_long_margin_rate or 0.0))   # equity, its fund's margin rate, selling dip
        out.append(A.Strategy("crash", 1, slices=float(settings.crash_long_account_slices), max_open=1,  # longs for
                              priority=PRIORITY["crash"], quality=QUALITY["crash"], book=BOOK["crash"],  # room
                              house_maint=rate, house_init=rate, fees="engine" if quoted(mim) else None))
    if cap_on(mim):                # PREREG165 / 167: the capitulation book — 1/6 of the equity, at most 3 open (its
        out.append(A.Strategy("cap", 1, slices=float(settings.cap_long_account_slices),       # working orders count),
                              max_open=int(settings.cap_long_max_open), priority=PRIORITY["cap"],  # capital quality
                              quality=QUALITY["cap"], preemptible=True, yield_whole=True,  # 0.1: it gives way whole,
                              yield_first="latest", delever_order=0, delever_first="latest",  # the latest first, and
                              orders="fill", book=BOOK["cap"],                                 # is de-levered first
                              fees="engine" if mim.costs else None))
    if ins_on(mim):                # PREREG203 / 205: the insider book — 1/10 of the equity, at most 5 open, Reg T,
        out.append(A.Strategy("ins", 1, slices=float(settings.insider_long_account_slices),   # 1% of the dollar volume;
                              max_open=int(settings.insider_long_max_open), priority=PRIORITY["ins"],  # capital quality
                              quality=QUALITY["ins"], preemptible=True, yield_whole=True,   # 0.05: it gives way first,
                              yield_first="latest", delever_order=0, delever_first="latest",  # whole, the latest first,
                              book=BOOK["ins"],                                  # de-levered with the capitulation book
                              fees="engine" if quoted(mim) else None))
    if mim.sleeve:                 # PREREG154: the QQQ sleeve — a lot of the capital no book uses each open, at least
        from src.backtest.strategies import sleeve as SLEEVE                                # its min share;
        out.append(A.Strategy("sleeve", 1, slices=1.0, max_open=None, priority=PRIORITY["sleeve"],
                              quality=QUALITY["sleeve"], preemptible=True, yield_whole=True,  # lots give way first,
                              yield_first="latest", delever_order=0, delever_first="latest",  # whole, latest first
                              min_frac=float(SLEEVE.MIN_FRAC), book=RESEARCH_BOOK["sleeve"],
                              fees="engine" if quoted(mim) else None))
    for name in mim.research:      # PREREG110: a study's research books (`register_research`)
        r = research_spec(name)
        out.append(A.Strategy(name, r["side"], slices=r["slices"], max_open=r["max_open"], priority=r["priority"],
                              quality=r["quality"], book=r["book"], house_maint=r["rate"], house_init=r["rate"],
                              fees="engine" if (r.get("fees") == "engine" and mim.costs) else None,
                              delever_order=r.get("delever_order")))
    if fear_on(mim):               # PREREG219-226: the fear book — 1/6 of the equity, at most 4 open (its working orders
        out.append(A.Strategy("fear", 1, slices=float(settings.fear_long_account_slices),     # count), capital quality
                              max_open=int(settings.fear_long_max_open), priority=PRIORITY["fear"],  # 0.04: it gives
                              quality=QUALITY["fear"], preemptible=True, yield_whole=True,  # way first, whole, the
                              yield_first="latest", delever_order=0, delever_first="latest",  # latest first, de-levered
                              orders="fill", book=NAME_BOOK["fear"],      # first; ONE BOOK WITH THE DIP BOOK for one book
                              fees="engine" if mim.costs else None))      # per name (PREREG223's M1)
    return out


def quoted(mim: Mimic) -> bool:
    """AUDIT151's C_QUOTE is on: the ledger's costs with every market leg at its real quoted half-spread."""
    return bool(mim.costs) and bool(mim.quote_costs)


def crash_on(mim: Mimic) -> bool:
    """The crash-rebound book is part of the live combination: production's switch and the mimic's."""
    return bool(mim.crash) and bool(getattr(settings, "enable_crash_long", False))


def cap_on(mim: Mimic) -> bool:
    """The capitulation book is part of the live combination: production's switch and the mimic's."""
    return bool(mim.cap) and bool(getattr(settings, "enable_cap_long", False))


def fear_on(mim: Mimic) -> bool:
    """The fear book is built (PREREG219-226): the Mimic's ``fear`` and production's `enable_fear_long`."""
    return bool(mim.fear) and bool(getattr(settings, "enable_fear_long", False))


def ins_on(mim: Mimic) -> bool:
    """The insider book is part of the live combination: production's switch and the mimic's."""
    return bool(mim.ins) and bool(getattr(settings, "enable_insider_long", False))


def margin_rules(mim: Mimic) -> dict:
    """The account's margin calls (PREREG174): IBKR's liquidation judged on the bars' extremes (``margin_check``
    "bars") or at the tick marks, or every position closed at once (``margin`` "all", judged where ``trigger`` says)."""
    if mim.margin == "ibkr":
        return {"trigger": "high" if mim.margin_check == "bars" else "close", "liquidation": "ibkr",
                "liq_order": mim.margin_order, "sem": float(mim.margin_sem), "regt_close": bool(mim.margin_regt),
                "liq_fill": mim.margin_fill, "liq_slip": float(mim.margin_slip)}
    return {"trigger": "close" if mim.trigger else "high"}


def live_rules(mim: Mimic) -> A.Rules:
    # PREREG154 / 165 / 205: with the QQQ sleeve, the capitulation book or the insider book a dip (or capitulation)
    # order is sized at placement as if the lower-quality lots were freed and its fill sells them for room
    # (`Rules.fills_make_room`; inert without a lower-quality preemptible book)
    return A.Rules(start=float(mim.start), **margin_rules(mim),                  # PREREG227's S6: production's start
                   delever=(tuple(float(x) for x in mim.delever) if mim.delever is not None else
                            (float(settings.live_account_delever_trigger), float(settings.live_account_delever_target))),
                   one_book_per_name=True, fees="ledger" if mim.costs else "engine", mark_net=bool(mim.costs),
                   fills_make_room=bool(mim.sleeve) or cap_on(mim) or ins_on(mim) or fear_on(mim),
                   # production's fill give-way (`live_account.fill_give_way_plan`): the fill's own size decides; on
                   # PREREG167's 60 accounts identical to the slice-capped rule (2026-10-20)
                   fill_room_exact=cap_on(mim) or ins_on(mim) or fear_on(mim))


def turn_lend(eng, mim: Mimic, L: np.ndarray) -> Optional[Callable[[int, int, float], bool]]:
    """PREREG76: production checks a turn's lendability when the short ENTERS — IBKR's file in force at the trigger
    bar's end where an archive holds it (IBKR's to 2024-06-24, ours from 2026-09-25; unknown = lendable, as
    `lend_all`), IBKR's refusal of the name that day after `refused` (``house``); in between, the pick's draw ``L``.
    None (the pick's lendability decides) without ``lend``."""
    if not mim.lend:
        return None
    lendable_at = lendable_at_trade(mim)          # AUDIT151: at the trade, not on the morning file (``lend_intra``)
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
        return not (mim.house and is_refused(mim, sym, int(dn[row])))
    return ok


def etf_rule_kw() -> Dict:
    """The ETF arm's trade rule at production's numbers (PREREG227's S7; `EtfArm.pieces`' keywords, the values it held
    as numbers): the 6x cover (`sel_short_cover_multiple`), the crowding cut (`sel_short_max_days_to_cover`) and the
    relative-volume cut (`sel_short_vol_min_rvol`) — the switches stay the simulator's (`strategies.vol.LIVE`'s rule)."""
    return {"cover": float(settings.sel_short_cover_multiple), "max_dtc": float(settings.sel_short_max_days_to_cover),
            "min_rvol": float(settings.sel_short_vol_min_rvol)}


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
    """The products the ETF arm never ranks under ``mim`` (None: every product, the arm before 2026-10-14): the
    VIX-linked ones unless ``etf_vix`` (PREREG102), the stock-era tickers `REUSED_ETF` unless ``etf_reused``
    (AUDIT151's E_REUSE) — the arm's picks re-ranked without them (`EtfArm.events`)."""
    ex = None if mim.etf_vix else vix_etps()
    if not mim.etf_reused:
        ex = frozenset((ex or frozenset()) | REUSED_ETF)
    return ex


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

        # set B's ETF picks: the archive years' rate, at the trade with AUDIT151's survival (``lend_intra``)
        etf_rate = ETF_LEND_RATE if mim.lend_intra is None else ETF_LEND_RATE * mim.lend_intra[4]

        def etf_lend(tkn: str, day: int) -> bool:
            return draw(plain(tkn), int(day)) < etf_rate
        halt = None
        if mim.halts:                          # PREREG99: production's halt guard / a variant (src/backtest/halts.py)
            from src.backtest import halts as HALTS
            halt = HALTS.Applier(mim, eng, L)
        if mim.arm_on("etf"):
            ex = etf_exclusion(mim)            # PREREG102: production's VIX block (the re-ranked picks)
            books["etf"] = (etf.pieces if halt is None else halt.etf_pieces(etf))(
                lo, hi, give=float(getattr(settings, "sel_short_etf_give_back", 0.5)),
                hmax=int(getattr(settings, "sel_short_arm_max_hold_sessions", 15)),     # 10 from 2026-10-20 (PREREG159)
                **etf_rule_kw(),                     # PREREG227's S7: the cover and the two cuts from the settings
                entry="pick" if mim.entry else "next",
                lend_after=etf_lend if mim.lend else None, cagap=mim.cagap, ctx=True,
                limit=lim if mim.limited("etf") else None, through=thru,
                turn=None if mim.limited("etf") else mim.etf_turn, **({"exclude": ex} if ex else {}),
                **({"lend_at": lendable_at_trade(mim)} if (mim.lend and mim.lend_intra is not None) else {}))
        else:
            books["etf"] = []                  # PREREG102's E2: never built, its slice unused
        if mim.dip_ext_orders != "off":                     # PREREG121's E1 / E2: orders outside regular hours
            xb = bars or ExtBars()
            pn = panel if panel is not None else DIP.build_panel()
            key = (str(lo), str(hi), mim.dip_ext_orders, id(pn), id(xb))      # the same for every draw: built once
            if key not in _EXT_DIP:
                _EXT_DIP.clear()
                _EXT_DIP[key] = DIP.order_pieces_ext(DIP.LIVE, lo, hi, panel=pn, ext=xb, nights=NightBars(xb)
                                                     if mim.dip_ext_orders == "ext+night" else None)
            dip = _EXT_DIP[key]
        elif dip is None:
            dip = DIP.order_pieces(DIP.LIVE, lo, hi, panel=panel if panel is not None else DIP.build_panel())
        # PREREG96's RP33 as production applies it (`enable_dip_long_filter`): a filtered signal never an order
        books["dip"] = DIP.filter_pieces(dip, DIP.filter_keep()) if mim.dip_filter else list(dip)
        if mim.dip_min_order_dv:                            # PREREG147's S3: the daily top pick's $2B floor
            books["dip"] = [p for p in books["dip"] if float(p.get("dv20") or 0.0) >= float(mim.dip_min_order_dv)]
        if mim.dip_rank != "rsi2" or mim.dip_topk is not None:  # PREREG107: best first / the day's top k
            # PREREG137's DEMOTE_E: production's earnings dips (a research file of order ids replaces them)
            demote = (_demoted(mim.dip_demote) if mim.dip_demote is not None else
                      earnings_demoted([p["oid"] for p in books["dip"] if p.get("oid") is not None],
                                       mim.dip_earnings, a0) if mim.dip_earnings else None)
            if mim.dip_wsb:                                     # PREREG209's KD+: a day's flagged orders only
                books["dip"] = wsb_preferred(books["dip"], mim.dip_wsb, log)
            books["dip"] = DIP.rank_pieces(books["dip"], DIP.filter_scores(mim.dip_scores), by=mim.dip_rank,
                                           topk=mim.dip_topk, demote=demote)
        if cap_on(mim):                                        # PREREG165 / 167: production's cap_long, its own pieces
            from src.backtest.strategies import cap as CAP
            books["cap"] = CAP.pieces(lo, hi, panel=panel, period="A0" if a0 else None)
        if fear_on(mim):                                       # PREREG219-226: production's fear_long, its own pieces —
            from src.backtest.strategies import fear as FEAR   # the dip book's own orders of the day left out
            dip_oids = {str(p["oid"]) for p in books["dip"] if p.get("oid") is not None}
            books["fear"] = [p for p in FEAR.pieces(lo, hi, panel=panel, period="A0" if a0 else None)
                             if str(p["oid"]) not in dip_oids]
        if crash_on(mim):                                      # PREREG86: production's crash_long, its own pieces
            from src.backtest.strategies import crash as CRASH
            ck = {"every": True} if mim.crash_reenter else {}  # PREREG227's S2: every qualifying session a candidate
            books["crash"] = CRASH.pieces(lo, hi, period="A0", **ck) if a0 else CRASH.pieces(lo, hi, **ck)
        if ins_on(mim):                                        # PREREG203 / 205: production's insider_long
            from src.backtest.strategies import insider as INS
            books["ins"] = INS.pieces(lo, hi, period="A0" if a0 else None)
        if mim.sleeve:                                         # PREREG154: the QQQ sleeve's candidate lots
            from src.backtest.strategies import sleeve as SLEEVE
            books["sleeve"] = SLEEVE.pieces(lo, hi, period="A0") if a0 else SLEEVE.pieces(lo, hi)
        for name in mim.research:                              # PREREG110: a study's research books
            books[name] = list(research_spec(name)["pieces"](lo, hi, period="A0" if a0 else None))
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
    if mim.pm_exits or mim.crash_pm_entry or mim.crash_pm_exit:   # PREREG121's X1 / C1, PREREG166's XC: the
        bars = bars or ExtBars()                                       # after-hours tick acting on the close
        log = log if log is not None else {}
        cost_bp = costs.bp if (costs is not None and mim.costs) else None
        for n in (("dip", "crash") if mim.pm_exits else ("crash",) if mim.crash_pm_exit else ()):
            if n in books:
                out = []
                for p in books[n]:
                    q, why = pm_exit(p, bars, cost_bp if n == "dip" else None)
                    log[f"pm_exit_{n}_{why}"] = log.get(f"pm_exit_{n}_{why}", 0) + 1
                    out.append(q)
                books[n] = out
        if mim.crash_pm_entry and "crash" in books:
            out = []
            for p in books["crash"]:
                q, why = pm_entry(p, bars)
                log[f"pm_entry_crash_{why}"] = log.get(f"pm_entry_crash_{why}", 0) + 1
                out.append(q)
            books["crash"] = out
    if mim.ext:
        bars = bars or ExtBars()
        span = mim.margin_bars == "all"                   # PREREG175: every bar since the previous mark
        for n in ("vol", "etf", "fall"):
            out = []
            for p in books[n]:
                q, why = extend_short(p, bars, settle, span=span)
                log[f"ext_{n}_{why}"] = log.get(f"ext_{n}_{why}", 0) + 1
                out.append(q)
            books[n] = out
        for n in mim.ext_longs:                           # PREREG175: every long book (PREREG174: the dip book's)
            if n not in books:
                continue
            out = []
            for p in books[n]:
                q, why = extend_long(p, bars, span=span)
                log[f"ext_{n}_{why}"] = log.get(f"ext_{n}_{why}", 0) + 1
                out.append(q)
            books[n] = out
    if mim.overnight != "off":                 # PREREG121: the overnight session's ticks (IBKR's venue, 2023-07-16 ->)
        bars = bars or ExtBars()
        log = log if log is not None else {}
        nights = NightBars(bars)
        for n in SHORT_ARMS:
            out = []
            for p in books[n]:
                q, why = night_short(p, nights, bars, settle, mim.overnight)
                log[f"night_{n}_{why}"] = log.get(f"night_{n}_{why}", 0) + 1
                out.append(q)
            books[n] = out
        for n in ("dip", "crash"):
            if n in books:
                out = []
                for p in books[n]:
                    q, why = night_long(p, nights, bars)
                    log[f"night_{n}_{why}"] = log.get(f"night_{n}_{why}", 0) + 1
                    out.append(q)
                books[n] = out
        log["night_rejected"] = dict(nights.rejected)
    if mim.news_hold:                          # PREREG136: the shorts held longer after news (before the costs)
        rule, arms, k = news_hold_spec(mim.news_hold)
        bars = bars or ExtBars()
        log = log if log is not None else {}
        nin = NewsInputs.get()
        nights_nh = NightBars(bars) if mim.overnight != "off" else None
        for n in arms:
            out = []
            for p in books[n]:
                q, why = news_hold_piece(p, rule, k, nin, a0, bars, settle, nights_nh)
                log[f"news_{n}_{why}"] = log.get(f"news_{n}_{why}", 0) + 1
                out.append(q)
            books[n] = out
    if mim.costs:
        adv = DailyADV(bars or ExtBars())
        qt = None
        if quoted(mim):                                        # AUDIT151's C_QUOTE: the legs' real quoted spreads
            from src.backtest import quotes as QUOTES
            qt = QUOTES.table()
        qk = {"quotes": qt} if qt is not None else {}          # a cost object without quotes keeps its own legs
        for n in ("vol", "etf", "fall"):
            books[n] = [costs.apply(p, -1, adv=adv, **qk) for p in books[n]]
        books["dip"] = [costs.apply(p, 1, legs=False, adv=adv, **qk) if int(p.get("ord_k", 0)) >= 0 else p
                        for p in books["dip"]]
        # PREREG227's S5 (``quote_all``): the capitulation and fear books' legs at their quotes as the dip book's, the
        # insider book's buys and sells at the open both at their quotes as the crash book's (IBKR's fees once, the
        # engine's: no ledger all-in cost under them)
        lk = {"quote_kw": {"limit_arms": ("dip", "cap", "fear")}} if (qt is not None and mim.quote_all) else {}
        mk = {"quote_kw": {"market_arms": ("crash", "ins")}} if (qt is not None and mim.quote_all) else {}
        if "cap" in books:                                     # limit orders, as the dip book's
            books["cap"] = [costs.apply(p, 1, legs=False, adv=adv, **qk, **lk) if int(p.get("ord_k", 0)) >= 0 else p
                            for p in books["cap"]]
        if "fear" in books:                                    # the fear book's: the same
            books["fear"] = [costs.apply(p, 1, legs=False, adv=adv, **qk, **lk) if int(p.get("ord_k", 0)) >= 0 else p
                             for p in books["fear"]]
        if "crash" in books:                                   # market buys / sells at the open: both legs
            books["crash"] = [costs.apply(p, 1, adv=adv, **qk) for p in books["crash"]]
        if "ins" in books:                                     # the insider book's: the same
            books["ins"] = [costs.apply(p, 1, adv=adv, **qk, **mk) for p in books["ins"]]
        if "sleeve" in books:                                  # PREREG154: the same, every lot at the open
            books["sleeve"] = [costs.apply(p, 1, adv=adv, **qk) for p in books["sleeve"]]
        for name in mim.research:                              # PREREG110: market orders at the open, both legs
            books[name] = research_costed(name, books[name], costs, adv, quotes=qt)
        if qt is not None:
            log = log if log is not None else {}
            for n, pcs in books.items():
                miss = sum(1 for p in pcs if p.get("q_fb"))
                if miss:
                    log[f"quote_fallback_{n}"] = miss
    if mim.order:
        rank_shorts([p for n in ("vol", "etf", "fall") for p in books[n]])
    if mim.short_name_cap is not None:         # PREREG113: each short piece's group is its traded name
        for n in SHORT_ARMS:
            for p in books[n]:
                p["group"] = plain(p["tkn"])
    # AUDIT151's D_OPEN: an open fill at IBKR's attainable price (09:30:30). PREREG227's S1 (``open_order``): the
    # capitulation and fear books' open fills too, a fill without a price that session dropped (the order keeps
    # working), one the table cannot price stamped 09:30:30 — the three then fill in production's order at the open;
    # S5 (``quote_all``): a fill moved to a later touch pays no spread
    ok = {"books": ("dip", "cap", "fear"), "none_drops": True, "stamp_unknown": True} if mim.open_order else {}
    if mim.quote_all and quoted(mim):
        ok = dict(ok, touch_free=True)
    if mim.dip_open and books.get("dip"):
        log = log if log is not None else {}
        books["dip"] = [q for q in (DIP.reprice_open(p, log, **ok) for p in books["dip"]) if q is not None]
    for n in ("cap", "fear"):                  # the capitulation and fear books' likewise (before PREREG227 a no-op:
        if mim.dip_open and books.get(n):      # `reprice_open` repriced the dip book's pieces only)
            log = log if log is not None else {}
            nl: dict = {}
            books[n] = [q for q in (DIP.reprice_open(p, nl, **ok) for p in books[n]) if q is not None]
            for k_, v_ in nl.items():
                log[f"{n}_{k_}"] = log.get(f"{n}_{k_}", 0) + v_
    if mim.dividends:                          # AUDIT151's DIV: a short pays, a long receives (after the repricing)
        from src.backtest import dividends as DIV
        log = log if log is not None else {}
        dl: List[dict] = []
        longs = DIV.ALL_LONG if mim.div_all else None          # PREREG227's S4: every live long book
        books = {k: [DIV.adjust(p, dl, longs=longs) for p in v] for k, v in books.items()}
        log["div_events"] = len(dl)
    return books
