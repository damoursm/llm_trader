# The multi-strategy backtest — runbook

User directive 2026-10-08: "Rework our simulator that consider a real environment with restrictions when trading so
that it can consider multiple strategies. The idea is to be able to reuse it to evaluate different models or
combination of models. It should be able to calculate the real fees and be able to know the available capital from
the 10000$ starting account. We should then be able to use it to get the growth ... and optimize our optuna studies."

Research tooling: nothing in production imports `src/backtest/`.

## The pieces

| Module | What it does |
|---|---|
| `src/backtest/account.py` | THE account: $10,000, no deposits, every strategy's trades in one IBKR margin account — longs and shorts, real fees, Reg T + IBKR house margin, available capital, margin calls, per-strategy attribution, both metrics (`simulate`, `metrics`) |
| `src/backtest/strategies/vol.py` | the vol short arm (and any variant of its rule) as pieces, from the research vol engine (`VolEngine`) |
| `src/backtest/strategies/dip.py` | the dip long book (and variants) as pieces, from the deep 30-minute store + the delisted names (`pieces`, `DipParams`) |
| `src/backtest/strategies/etf.py` | the ETF short arm as pieces (`EtfArm().pieces(lo, hi)`): the research ETF run's picks (`<research>/etf/vol_events_de_arm.pkl`) under the live rule — half give-back, filters, volatility exit, 6x cover, IBKR's archived lendability to 2024-06; reproduces the recorded 141-trade book exactly (`<research>/etf/bt_etf_parity.py`) |
| `src/backtest/strategies/dm.py` | the ETF dual-momentum book as pieces (`pieces(lo, hi, end=)`): the live decision code (`etf_dm.fund_stats`, `etf_dm.tbill_growth`) on the repo's daily OHLCV cache (2006 on), one piece per holding bought / sold at the month's first open, 30-minute marks from the deep store (daily before it); trades as `account.dm_long` (the whole account in one fund); all 68 monthly picks 2021-02 .. 2026-09 and the research's growth (+7.84 / +14.80 %/yr) reproduced |
| `src/backtest/study.py` | weekly Sunday walk-forward re-tuning of one or several arms TOGETHER, objective = the window account's log growth; Reality Check / argmax / Optuna picks; the chained test account |
| `src/backtest/mimic.py` | THE SIMULATOR MADE TO MIMIC PRODUCTION (PREREG62, below): `Mimic` (each fix a switch), `live_books` / `live_strategies` / `live_rules` = the live combination (vol + ETF + fall shorts at 1/24, the dip limit book at 1/10, one $10,000 account) as production runs it |

The research engine and data the vol adapter wraps live in `settings.backtest_research_dir`
(`C:\Users\mathi\PycharmProjects\llm_trader_research`: `optvol/` stores + evaluator, `vol/` the bar store, `pylib/`
optuna, `long/` the dip study, `studies/` pre-registered studies). They were moved there from the session
scratchpads on 2026-10-08; junctions at the old scratchpad paths keep the older scripts working.

## The account's rules (`account.simulate`)

* Equity = $10,000 + realized P&L + open positions marked at their latest bar (a short's accrued borrow included).
  Available capital = equity - the open positions' initial requirement.
* A new position = floor(min(equity / slices, side cap x equity - the side's open value, 1% of its 20-session dollar
  volume) / price), cut to the initial-margin room after its own costs; none under the strategy's `min_equity`
  (FINRA's $2,000 for shorts) or past its `max_open`. Entries at one instant: strategy priority, then the piece's rank.
* Margin: a short = max(Reg T, IBKR house multiple of its value) — the live rule's 2.00 maintenance / 2.86 initial;
  a long = Reg T 25% / 50%. Judged at every bar (closes, or the bar's adverse extreme with `trigger="high"`): equity
  below maintenance = a margin call that closes everything at twice the entry half-spread.
* Fees: each fill's half-spread (real NBBO or the fitted model), IBKR's fixed commission max($1, $0.005/share) capped
  at 1%, the SEC fee + FINRA TAF on every sale, a short's IBKR borrow day by day.
* Metrics (`account.metrics`): growth per year, LOG growth per year (the study objective), worst drawdown, margin
  calls; per strategy the positions, P&L, costs, average realized trade, hold, return per day, and the sum of each
  position's log(1 + P&L / equity at entry) — the per-trade objective's account form (equal to the log growth when
  positions do not overlap; with overlap the account number is the true one).
* Activity and capital in use: `simulate(..., keep_exposure=True)` also records, after every instant, the open longs'
  and shorts' value at their marks, the same per strategy (`curve_by`) and their initial margin (`curve_long` /
  `curve_short` / `curve_req`; the results are unchanged); `exposure_daily(res, sessions)` gives them per session
  (per strategy as `by:<name>` columns); `activity(res, sessions, lo, end)` counts
  the trades opened per calendar year and strategy and the sessions with a position open during regular hours
  (`holding_days`: a long sold AT the open does not hold that day).
* Capital priority (off by default; PREREG52): a `Strategy` with a `quality` whose entry finds too little room (its
  side's cap or the margin room) sells just enough of the open positions of `preemptible` strategies of lower quality
  at their mark (exit costs paid, P&L realized; `res["preemptions"]`); a preemptible strategy with `refill_min` buys
  back toward its slice at its own marks once the room it can fill is that share of equity (`res["refills"]`).
  Tested on the four books (dual momentum giving way; the dip book also giving way to the shorts): it lost 1.6-3.5
  points a year on 2021-24 and gained 11.5-14.0 on 2021-26, nothing significant — first come, first served stays.
  `yield_whole` makes a preempted position exit entirely; `reenter` re-buys a piece closed early (preempted or
  margin-called) at its next 09:30 ET mark with room, never by preempting (`res["reentries"]`). The user's rule
  (PREREG53: rank by return per day, the lowest-ranked position exits whole, dual momentum re-buys at the next open):
  +23.0 vs +24.9 %/yr on 2021-24, +39.9 vs +31.2 on 2021-26, not significant — an option, not the default.
  `strategies/dm.py` `pieces(..., open_marks=True)` also marks the fund at each session's 09:30 open (what such a
  sale at the open gets).
* Margin protection beyond the slices (off by default; PREREG57): `Strategy(room_frac=θ)` opens that strategy's
  entries only while the account's initial requirement after them stays under θ × equity (a margin buffer);
  `Rules(delever=(trigger, target))` sells, at a bar close where the maintenance requirement exceeds trigger × equity,
  the whole positions of strategies with a `delever_order` (lowest first, the latest opened first) at their marks
  until it is under target × equity, never bought back (`res["delevers"]`). Across PREREG57's grid the buffer cost
  growth (median −8.2 points a year at 50%) and de-leveraging (0.8 → 0.6) changed nothing in two configurations out
  of three and removed margin calls where it acted.
* Working limit orders (off by default; PREREG61): `Strategy(orders="reserve" | "fill")` makes that strategy's pieces
  LIMIT ORDERS — the pieces sharing an `oid`, one per session the order could fill in (`ord_k`, its touch as `ens` /
  `e`, the exit and marks of a fill there), each carrying the order's `ord_place` (placement instants), `ord_lim` and
  `ord_end`; a placeholder with `ord_k=-1` stands for an order that never touches. At each placement instant an order
  neither working nor filled is sized at its limit like an entry (deepest `rank` first; working orders count toward
  `max_open`), then fills at its first touch after placement with that quantity CUT to what the account can fund at
  that instant (equity minus the open positions' initial requirement and the other orders' reservations, within its
  side cap): the rest is not bought, and a touch that finds no room buys nothing while the order keeps working (user
  2026-10-09: a fill into less room than it needs does not earn the order's promised return). "reserve": a working
  order holds its initial requirement in the room and its value in its side; a higher-quality entry short of room
  cancels the working orders of preemptible strategies (latest placed first) before selling their positions,
  de-leveraging cancels them first. "fill": it reserves nothing. A margin call cancels both. `res["orders"]` logs every
  order's life (`placed`, `filled` with `n_order` and the room it found, `unfunded`, `cancel_giveway` /
  `cancel_delever` / `cancel_margin`, `lapsed`, `no_slot`, `no_room`). Results: studies/prereg61 (PREREG61.md). A tick's limit fills
  come BEFORE its other entries (production judges the dip book's touches at the tick's start, before the shorts'
  entries). The dip book's live limit entry is `strategies/dip.order_pieces(LIVE, lo, hi)` (the production
  `dip_long.limit_price`, placements at each window session's 09:30, the open / bar-end fills, 30-minute marks).
* Judging a new ARM (the combination's design, user 2026-10-11 — arms for different sides, horizons and markets,
  activated by their own rules, never by a regime switch; the evaluation standard, user 2026-10-14 — one pooled
  history, a bank of models): (1) standalone on the pooled history A0 + A + B (and the longest history its data
  allow): its card and its bank qualification (`bank.card` / `bank.qualify`: at least 100 trades in 30 entry months,
  the return per day's entry-month bootstrap interval above its BAR — 0 for a short arm, for a long arm the same
  exposure's average day: SPY's, or the fund it trades —, both chronological halves and every leave-one-year-out above
  it, growth at its live slice; re-judged on its longest history when it fails only significance), per year and per
  market context — where it earns, where it loses (a context it
  loses in becomes a filter in its OWN rule, justified by `bank.filter_test` and measured out of time by
  `bank.walk_forward`); (2) in the live account (`Mimic()` + V6 + the arm as one more `Strategy`, 20 stratified draws
  a pooled set, each draw's daily NAV saved): the pooled add rule (`add_rule.pooled_verdict`) with every set and every
  year reported, the months and the coverage of the market contexts (`bank.combination`); (3) before asking to
  deploy, rebuild it on production's own construction and re-run the account (`studies/prereg86/final86.py`:
  research's Polygon closes gave the crash book +0.56 / +0.51 log points, production's 16:00 bars +0.28 / +0.45).
* The crash-rebound book (PREREG86, live from 2026-10-12; traded in TQQQ from PREREG88): `strategies/crash.py`
  builds production's longs — DECIDED on QQQ's deep-store daily rows (`crash_long.stats_at` / `qualifies`, the
  open-to-open exits on QQQ's closes), HELD in the traded fund's 30-minute bars (`crash_long_trade_symbol`, TQQQ;
  `pieces(trade_sym=...)`); `Mimic(crash=True)` (the default; `Mimic.off()` leaves it out) adds them to `live_books`
  (ledger costs on both legs) and a strategy to `live_strategies` (1/4 slice, one open, quality 0.2 above the dip
  book's 0.13, the fund's `crash_long_margin_rate` 0.75 as its ``house_init`` / ``house_maint``). The deployed
  account: `studies/prereg88/final88.py` (its default reproduces the research run `account88.py`'s Q-TQQQ/4
  exactly); the TQQQ / FNGU variants: `studies/prereg88/account88.py`.
* Orders make room (off by default; PREREG86): `Rules(fills_make_room=True)` sizes a limit order of a strategy with a
  `quality` as if the lower-quality preemptible positions were freed (nothing sold at placement) and makes its fill
  sell just enough of them for the order's quantity — without it a book that yields to everything (PREREG86's
  trend sleeve) starves the dip book's limit orders. Inert when no lower-quality preemptible position exists (the
  live account reproduces final80 draw for draw).
* One book per name (off by default): `Rules(one_book_per_name=True)` with `Strategy(book=...)` skips an entry, an
  order placement or an order fill while a strategy of ANOTHER book holds the name (production: the selection short
  never shorts a name the dip book holds and the reverse; arms of one book may share a name). Use both to mimic the
  live account (studies/prereg61/final.py).
* A group cap (off by default; PREREG100's per-sector cap): `Strategy(group_cap=K, group_scope=(...))` — at most K
  open positions plus working orders in one GROUP (a piece's ``group``, e.g. its Fama-French 12 sector from
  `src/backtest/sectors.py`: `load_labels`, `stamp_groups`) among the strategies of the scope (None: the strategy
  itself). A market entry at the cap is skipped (`group_cap` in `keep_skips`); a limit order at the cap is not placed
  then (logged `group_cap`) and is tried at its next placement instant, as an order without a slot; a stack counts as
  one more position; a piece without a group is never capped nor counted. Without a cap the engine is unchanged
  (`tests/test_backtest_group_cap.py`; 400 random scenarios identical to the engine before it).
* Lack of capital: `simulate(..., keep_skips=True)` records every entry not taken in full — the shares its slice
  wanted, the shares it got (0 = skipped) and the binding limit (`margin`, `side_cap`, `volume_cap`, `house_shrink`,
  `max_open`, `min_equity`, `below_one_share`, `lendable`, `group_cap`); `capital_summary(res, pieces)` counts them
  per strategy.

## Mimicking production (`src/backtest/mimic.py`, PREREG62, 2026-10-09)

User 2026-10-09: "Apply all of the simulator fixes. It needs to mimic production as best as possible." `Mimic()` turns
on every way the live account differs from the earlier simulator (`Mimic.off()` runs the earlier one's rules — with the
PREREG63 corrections below, so earlier numbers move slightly, and PREREG62's lendability `v1` with the median fee;
`Mimic().without("ext")` one fix's ablation; ``seed`` drives the draws):

* ``lend`` — after IBKR's archived file (2024-06-24) a stock pick is lendable on a DRAW from a model fitted on the
  archive years, one probability per store row: `lend_model="v2"` (PREREG67: LightGBM on 27 point-in-time inputs —
  price, volume, ATR%, days to cover, run-up, IBKR's API fee bars, Reg SHO, the name's last archived record and its
  age; `optvol/store_xmc10/lend_p_v2.npy`; out of sample 2023-01 .. 2024-06 AUC 0.79 against 0.77 for PREREG62's
  logistic, `"v1"` = `lend_p.npy`). On unseen years it ranks well (AUC 0.76-0.80) but its LEVEL drifts 5-8 points a
  year (PREREG68), so `lend_shift` (logit) moves it for a sensitivity: ±0.37 = ±7 points of set B's lendable share.
  An ETF pick at its archive rate 0.827; our own archive from 2026-09-25. (IBKR's API fee bars are no proxy for
  lendability: IBKR publishes one for 99% of the names its file could not lend.) The draws are LINKED (PREREG94, the
  default from 2026-10-12, user: "F4 fees + persistent + market-wide lend"; `lend_draw_fn`): the pick is lendable iff
  Phi(z) < p with z = sqrt(0.15) w(day) + sqrt(0.85) x(name, day) — x a per-name AR(1) over the sessions
  (``lend_persist`` 0.7), w one market-wide AR(1) (``lend_day_lam`` 0.15, ``lend_day_r`` 0.5), each pick's own
  probability unchanged. Fitted on IBKR's archive: a name's lendability agrees 0.76 between picks a session apart
  (independent draws 0.64), the daily lendable count varies 1.45x what independent draws give (lag-1 autocorrelation
  0.35; the tightest days are the squeeze waves). ``Mimic.borrow_v2()`` = PREREG67's independent draws and v2 fees:
  the construction every study before PREREG94 ran (rerun one with it and ``fee_pred_path("v2")``).
* ``fee_model`` — a stock short with no IBKR fee (no file, no API bar: mostly names delisted since) pays the fee
  model's MEAN fee instead of the median 9.75 %/yr, fed through `VolEngine.fee_fallback`. ``fee_version="v4"`` (the
  default, PREREG94's F4; `fee_pred_v4.npy`): LightGBM Gamma (log link) on a pick's mean daily fee over its first 21
  days, × a trailing level factor — the listed top-5 picks' real IBKR API fees over the model's guess for them with
  their own API inputs masked, the 30 days before the pick, clipped to [0.5, 3] — so it follows the market's borrow
  level (set B's factor 1.14-1.47 from 2025: PREREG83's dearer borrow). ``"v2"``: PREREG67's (log1p fee + Duan
  smearing, `fee_pred_v2.npy`; on every unseen year 2019 .. 2024-06 it beat the median, PREREG68). On set A out of
  year: v4 −0.69 log points a year of growth vs IBKR's own fees, v2 +0.90. The ETF arm keeps its fees.
* What the modelled years can and cannot say (PREREG93 / 94): set B's whole construction on set A, models refit out
  of year, 20 draws — growth −0.8 log points a year vs IBKR's own data (95% −4.7 .. +3.1; v2 +0.5): growth is not
  inflated; but the lend draws deepen the worst drawdown 15 → 25% and bring a margin call in 4 draws of 20 where
  IBKR's file had none, with the lend model calibrated per pick, in the squeeze tail and along a name's run — the
  single real path against the model's spread. Read set B's drawdowns and margin calls as pessimistic, and compare
  a variant's against the live configuration's on the same draws.
* the draws (``seed``) — every random decision (a set-B pick's lendability, IBKR's refusal, the house rate) is a
  uniform keyed by (decision, name, session, seed) (`mimic.draw_u`). ``draws="iid"`` (every study before
  2026-10-13; `Mimic.off()`, `Mimic.borrow_v2()`): independent across seeds — 20 draws of the live account vary ±3.5 points of growth a draw on set A and ±18.7 on set B, a few
  squeezes lendable in some draws only deciding most of it (PREREG95's prior evidence). ``draws="strat"`` with
  ``n_draws=N`` (PREREG95; THE DEFAULT from 2026-10-13, N = 5): each block of N consecutive seeds is one STRATIFIED set — every decision's
  N uniforms one per stratum of [0, 1), so a pick with a 60% lendability is lendable in 60% of the block's draws
  and a 7% refusal hits its share; set B's linked lend latent has its INNOVATIONS stratified, so each draw keeps
  the independent draws' law exactly (PREREG95 Amendment 4: ranking the latent among the block balanced every pick
  exactly but shifted set B's growth +12 points — withdrawn). PREREG95's verdict: the same expectation as iid on
  both sets (z 0.9 / 0.2), NO level stability (the SE of a 5-draw mean of set B's growth x1.01: the spread is the
  lendability of a few extreme names, a property of the world), but a rule change PAIRED on common seeds loses
  22-50% of its noise (the faller / ETF turn off: block SD 1.45 / 1.19 vs the iid 5-draw SE 1.86 / 2.40 log
  points) — use it for decision studies; judge a set-B change on >= 20 paired draws and an interval (SE ~0.8 log
  points), never a 5-draw mean.
* ``house`` — each stock short carries the house rates of one of the 36 names IBKR quoted on 2026-10-05
  (`optvol/house_mix.json`) + the live shrink (`Strategy.shrink_budget` 2.86); 7% of stock picks refused by IBKR
  (the vol arm falls back to the next candidate). ETFs: 2.00 / 2.86, never refused.
* ``entry`` + ``r201`` — a short enters at its PICK bar's close (the evaluator's ``entry_at_pick``, the ETF adapter's
  ``entry="pick"``), Rule 201 off (the ledger opens at the live price).
* ``ext`` — the scheduler's extended slots and the 09:30 open (`off_ticks`) judge the shorts' exits and mark every
  position on the stored 04:00-20:00 bars (`ExtBars`: `bars30m_full`, `bars30m_delisted`, a study's ``extra_dir``);
  a tick reads the bar ending at it within 30 minutes, the 09:30 tick the opening print; no overnight prices.
* ``costs`` — the shorts' legs cost the ledger's resolver for a leg without its own fill (`LedgerCosts` on a
  read-only snapshot of `tracker.calibrate_sim_costs`), the engine in ledger mode; the dip's legs (broker fills
  charged their realized cost) keep the engine's exact commission + 1 bp (`Strategy(fees="engine")`); every open
  position marked net of the resolver's exit estimate (`Rules(mark_net=True)`, a piece's ``hs_mark``).
* ``trigger`` — margin at the tick marks (`Rules(trigger="close")`); ``order`` — a bar's shorts by ticker, then arm
  (`rank_shorts`); ``cagap`` — the ETF arm skips a corporate-action gap bar.

Not modelled: halts, in-window borrow retries before 2026-09-25, late prints, overnight prices, the dip's 09:30-tick
price vs the official open, a leg the broker filled (its own realized cost). Tests: `tests/test_backtest_mimic.py`.
Study: `llm_trader_research/studies/prereg62/` (`mimic_run.py`: OLD vs MIMIC vs each ablation, 5 seeds).

```python
from src.backtest import mimic as M
cs = M.LedgerCosts("<research>/studies/prereg62/cost_snapshot.json")
B = M.live_books("2021-02-01", "2026-09-25", M.Mimic(seed=0), costs=cs, bars=M.ExtBars("<extra bars dir>"))
res = A.simulate([p for v in B.values() for p in v], M.live_strategies(M.Mimic()), M.live_rules(M.Mimic()))
```

## Corrections (PREREG63, 2026-10-09)

User 2026-10-09: "Correct any bugs that are left in the simulator." Four independent reviews (engine; mimic layer +
adapters vs production's code; the research engine; metrics + walk-forward), each finding reproduced before its fix,
each fix pinned by a test that fails on the old code (`tests/test_backtest_account.py`, `test_backtest_study.py`,
`test_backtest_dip_adapter.py`, `test_backtest_mimic.py`):

* ENGINE — a piece that enters and exits at one instant (a passive fill covered within its bar, a window cut at the
  entry bar) opened after its own exit and never closed (`Book` clamps its exit 1 ns after its entry); the final
  value now values open positions as every mark does (accrued borrow, the exit estimate); per-strategy statistics
  count closed trades only, with the GROSS win rate (``net_win_rate`` beside it); a ruin keeps the exposure arrays
  aligned; no room is made for an entry the lendability check then refuses; re-entry / refill refused on a short
  (its borrow is per piece); order books' `capital_summary` reads the order log; `Rules` validated; under ledger
  rules the sizing room keeps production's fees (`sim_account.size`: commission + SEC + TAF on a short; a long the
  commission) and the P&L is production's (shares x entry x the ledger's net return, priced on the cost-adjusted
  entry); settlement skips Columbus / Veterans Day (`settlement_days`, `make_settle`); one book per name compares the
  traded symbol (``XYZ@X`` is ``XYZ``).
* WALK-FORWARD — a tie with the reference keeps the REFERENCE (M0); a limit order placed before a window's cut and
  touched after it stays a working order (a placeholder) instead of vanishing; a STANDING holding (``rebase``: the
  dual-momentum pick) is bought at a window's / a run's start (`account.rebase`, `account.from_start`) and when a
  chained week turns its arm on; `study.max_drawdown` counts a loss from the $10,000 start.
* ADAPTERS — dip: a signal needs 220 daily bars through its own close (`stats_at`), and a name that stops trading
  before its exit is sold at its last close (it vanished); ETF: a pick whose deadline lies past the data is judged to
  the data's end and marked (it was dropped), and every top pick with a base is re-judged as a riser on the session
  calendar; vol / fall: lendability judged at the PICK bar's end with the pick entry (`wf9_borrow_pick.npz`; at the
  next bar a 15:30 pick read the next morning's file); mimic: the ledger's liquidity class on production's basis
  (`DailyADV`: the 60-session median of daily close x volume before the leg's day).
* RESEARCH ENGINE — `volengine.calendar` counted 28 closures (27 holidays of 2021-23, 2025-01-09) as sessions: the
  5-session base and the 15-session deadline were a session short around them (store_xmc10's PRE / LAST rebuilt —
  `studies/prereg63/rebuild_fresh.py`, the old file kept as `fresh_prereg62.npz`; 1,680 riser / faller flips);
  `cap5k4.settle` skips the bank holidays; a pick-bar entry before a cut is kept (and marked at entry without a path
  bar); the per-trade ``net`` borrow starts at the entry session; an IBKR fee is carried at most a week; the
  same-bar fallback follows production's one-trade-per-name-per-day rule in time order (a backup the arm traded
  today is passed over for the next; an earlier backup never blocks a later top pick); a piece's marks stop at its
  exit.

## Production bugs fixed, and the simulator with them (PREREG64, 2026-10-09)

User: "Fix the bugs in production and in the simulator". Four production bugs PREREG63's reviews found, fixed in
production with failing-first tests, and the simulator's counterparts:

* **Futile give-way** — production sold every dip long for a short that still could not be funded (the shorts' own
  initial requirement past the equity); the engine's `make_room` did the same. Now both sell nothing when freeing
  every candidate would not let the entry take one more share (`live_account.give_way_plan`; the engine's `helps`),
  and the engine sells another side's position while the MARGIN ROOM (not the entry's side cap) holds it, as
  production does (`room_bound`).
* **Berkshire twice** — the universe lists BRK-B and BRK.B; production's candidates and the dip adapter's names kept
  both. Now one spelling, BRK-B (`dip_long.candidates`, `strategies/dip.eligible_names`).
* **The market calendar** — production's `market_calendar` knew holidays only from 2024 and missed 2025-01-09; now
  2015-2030 with both national days of mourning (the simulator reads AAPL's sessions since PREREG63).
* **Splits during a hold** — the ledger kept the pre-split entry and shares (`tracker.apply_splits` now rescales open
  trades; `tests/test_splits.py`). The simulator's prices are split-adjusted: nothing to change.

Re-run: `llm_trader_research/studies/prereg64/` (`rerun64.py`, the PREREG63 runs + the attribution of each fix): the
live account +37.0% a year from 2021 (was +37.3%), all of it the duplicate Berkshire orders; the give-way fix changed
nothing (under the mimic's margin the dip book never gave way for nothing).

## Rules production adopted 2026-10-10: the turn for the faller and ETF shorts, the vol arm's 4x cover without its volatility exit, the ETF arm's whole give-back (PREREG76 / PREREG77)

PREREG77 (the user's add rule — "models that are not significant but still add value and growth should be added, as
long as their drawback isn't worse" — applied to PREREG75 / 76's leads on top of the live account with the faller
turn): the pre-registered combination that adds growth with no deeper drawdown and no margin call is the vol arm's
cover at 4x (`vol.LIVE["cap"]`) + its volatility exit off (`vol.LIVE["volnorm_on"]`) + the ETF arm's whole give-back
(the mimic passes production's `sel_short_etf_give_back`) + the ETF arm on the turn (`Mimic(etf_turn=(0.5, 5))`,
`EtfArm.pieces(turn=...)`, its target cancelling it, lendability judged when it enters); the fall arm keeps the 6x cover
and its volatility exit (`vol.FALL`), vol2 the 6x cover. The live account, five draws: set A +28.3 vs +26.2 %/yr (worst
drawdown 15 vs 18%, 0.388 vs 0.458 %/day), set B +90.4 vs +58.0 (20 vs 36%, 0.468 vs 0.353 %/day), no margin call;
`Mimic()` + V6 reproduces it. `llm_trader_research/studies/prereg77/` (`run77.py`, `final77.py`, `parity77.py`).

### Each short arm its own slice (PREREG80, from the 2026-10-12 session)

`mimic.live_strategies` sizes each short arm from production's `sim_account.arm_slices` (vol 1/20, ETF 1/16, fall 1/12;
the user's choice after PREREG80's 36-sizing grid). A piece may also carry its own ``slices`` (`account.simulate`: the
sizing of a weekly walk-forward; without it, the strategy's). PREREG80, the live account, five draws: +7.1 / +15.4 pp a
year on sets A / B over 1/24 everywhere, no margin call, the worst drawdown +1.0 / +3.5 points; every short 1/16, or
the fall arm 1/8, met margin calls; the weekly walk-forward with the Reality Check left the live sizing on set A toward
bigger slices and stayed live on set B. `llm_trader_research/studies/prereg80/` (`run80.py`, `analyze80.py`,
`final80.py`).

### The faller turn (`Mimic(fall_turn=(0.5, 5))`, PREREG76)

Production waits for each faller pick's TURN (`sel_short.turn_state`, live from the 2026-10-12 session), so the mimic
does too — ON by default, `Mimic.off()` and `Mimic(fall_turn=None)` without it: the fall book comes from
`strategies.vol.VolEngine.turn_pieces` (the pure rule `vol.turn_candidates`): the pick is shorted at the CLOSE of the
first regular-hours bar, within W sessions counting its own, whose close is at least k x the pick bar's ATR$ under the
highest close since the pick (the pick close included), stamped at that bar's end with a market entry's costs, sized
then; no turn, no trade; the exits from the next bar against the entry, the deadline counted from the pick. As live,
lendability is judged when the short ENTERS (`mimic.turn_lend`): IBKR's file in force at the turning bar in the
archive years and ours from 2026-09-25 (unknown = lendable), the pick's draw in between, IBKR's refusals as for any
pick. The simulator and production find the same turning bar on the same path (`tests/test_backtest_turn.py`).
PREREG76 (five draws, the live account with V6, lendability at the pick): set A +28.0 vs +26.4 %/yr, the faller book
1.62 vs 0.90 %/day; set B +58.0 vs +58.4, 2.10 vs 1.86 %/day — not significant (p 0.37), deployed on the user's order
("even if it's not significant it is better and not hurting"). With lendability judged at the turn, as live
(PREREG77's parity check): set A +26.2 %/yr, the faller book 1.43 %/day — one faller pick in seven can no longer be
borrowed by the time it turns; the user kept it. The same study's timed limit (a model choosing the
market or the ½-ATR limit per pick) failed its forecast gate (AUC 0.52 on the arms' picks).
`llm_trader_research/studies/prereg76/`.

### The ETF arm's VIX block, the arms in the account, the fall arm's slice and exit (PREREG102 / 104, 2026-10-14)

`Mimic`'s defaults follow production. ``etf_vix`` (default `not sel_short_etf_exclude_vix`: False while the interim
block is live): False = the ETF arm never RANKS the VIX-linked products (`sel_short.VIX_ETPS`) — `EtfArm.pieces(
exclude=...)` reads the arm's picks RE-RANKED without them (`etf/vol_events_{de,a0}_novix_arm.pkl` + its `.json`,
built by `llm_trader_research/studies/prereg102/events102.py`: on a bar a VIX product topped, the next product takes the
slot, fresh or not against its own history; every unaffected pick is the recorded one; leaving them out of the arm's
history changes no other name's freshness — each name's own history — which the build checks) and a by-symbol guard
keeps every one of them out (``XYZ@E`` / ``@D`` / a deep-store name alike); a missing or stale re-ranked file is an
ERROR, never a silent filter (pin `Mimic(etf_vix=True)` for the arm before the block: PREREG102's E0, `Mimic.off()`).
``short_arms`` (default all three): an arm left out is never built, its slice stays unused, every other book is built
as before (E2 without "etf", F2 without "fall"). ``fall_slices`` (None: `sim_account.arm_slices("fall")`, 12) and
``fall_volnorm_on`` / ``fall_volnorm`` (None: `vol.FALL`'s — on at 0.5) set the fall arm's slice and volatility exit;
the exit is applied to `vol.FALL` while the books are built (`mimic.fall_exit`, restored after). Every draw a study
made before 2026-10-14 ranked the VIX products: pair a new study with new live draws, or pin `etf_vix=True`.
``dip_filter`` (PREREG96 Amendment 3; False = production's `enable_dip_long_filter`, off): the dip book's orders
filtered by RP33 as production applies it (`dip.filter_pieces(dip, dip.filter_keep())`: every signal scored at signal
time by the walk-forward model of its year; a filtered signal never becomes an order, never holds a slot).
Tests: `tests/test_backtest_prereg102_104.py`. Studies: `llm_trader_research/studies/prereg102/` (`live102.py` — the
live draws S3, E1 = F0; `account102.py`), `studies/prereg104/account104.py` and `studies/prereg96/account96_prod.py`
(DF, paired on the same live draws).

### The dip book's daily top pick and at most two shorts of one name (PREREG116's Q1, from the 2026-10-16 session)

User 2026-10-15: "Deploy Q1". `Mimic`'s defaults follow production: ``dip_rank`` / ``dip_topk`` read `dip_long_rank` /
`dip_long_daily_top` (production: "score" / 1 — each signal day only its best order by the walk-forward score,
`dip.rank_pieces(pieces, dip.filter_scores(), by="score", topk=1)`, production's `dip_long.rank_key`; 0 = None, ranked by
"rsi2"), the dip slice `dip_long_account_slices` (6, as many open; ``dip_slices`` overrides), ``short_name_cap`` /
``short_name_scope`` `sel_short_max_open_per_ticker` (2) across the three short arms ("all"). `Mimic()` is therefore
PREREG116's Q1 (`Mimic(dip_rank="score", dip_topk=1, dip_slices=6, short_name_cap=2, short_name_scope="all")` with
study102's `BASE_KW`); `Mimic.off()` keeps the rules before (every signal an order at 1/10, no cap). The walk-forward
scores (`dip.filter_scores`, PREREG96's `prod96/wf_scores_prod.pkl`) run to the signals of 2026-10-06: a later signal is
unscored and ranked by its RSI2 — production scores it with the model fit on the signals exiting before 2026-10-08.
THE LIVE REFERENCE for a paired study is now Q1's draws (`studies/prereg116/runs/{A0,A,B}_Q1_s<seed>.pkl`, 20 a set,
each with its daily NAV; `bank.LIVE_RUNS`); study102's fingerprint (`dip_long_account_slices` among its settings, the
Mimic fields off their pre-Q1 values) refuses to pair a new run with the old `LIVE_S3` draws — register a new reference
instead. Tests: `tests/test_dip_top_pick.py` (production and the simulator choose the same order; the defaults follow
the settings), `tests/test_backtest_dip_rank.py`, `tests/test_backtest_research_books.py`.

## A rule under test: the short books on sell limits (`Mimic(short_limit=...)`, PREREG66, 2026-10-09)

The dip book's limit entry mirrored for the short arms — a research option, OFF by default, not production:
`Mimic(short_limit=(k, W), short_limit_arms=(...), short_limit_through=False)` turns the named arms' picks into SELL
LIMIT ORDERS (`strategies.vol.VolEngine.limit_pieces`, `EtfArm.pieces(limit=...)`; the pure rule
`vol.short_limit_candidates`, `vol.window_instants`): at the pick close + k x the pick bar's 30-minute ATR (up to the
cent), alive W sessions counting the pick's own, cancelled at the first close at/under the target; one engine piece per
window session with a touch (a bar's HIGH reaching it), entered at that bar's END at the limit, the arm's exits from
that bar's close; the engine's `Strategy(orders="fill")` sizes the order when placed and cuts each fill to the room
(no give-way for a limit fill). Tests: `tests/test_backtest_short_limit.py`.

Verdict (k = 1/2, W = 5, five draws, 2021-26): no arm passes. The limit fills ~3 picks in 4 at 3-7.5% better prices but
misses the picks that fall at once — the shorts' best trades (+20.5% vol, +9.7% ETF, +37.0% fallers a trade at market);
vol +0.12% a day (95% -0.33 .. +0.56, not significant), ETF -0.19, fallers -0.11; every short arm on limits +31.5 vs
+37.0 %/yr from 2021. Production keeps market entries. `llm_trader_research/studies/prereg66/` (`run66.py`,
`analyze66.py`; with PREREG65's re-evaluation of the live combination and each book's contribution).

## Trading halts at the entry passes (`Mimic(halts=...)`, `src/backtest/halts.py`, PREREG99, 2026-10-13)

OFF by default (`halts=None`: every pick enters at its bar's close, as before) until the studies running tonight on
PREREG95's LIVE_S2 draws finish; PREREG99 pins it in every run. ``"guard"`` (V0) is production's halt guard
(`tracker.record_sel_short_trades`, `enable_sel_short_halt_guard`) — a correction, adopted whatever the variants show:
a vol pick, a backup, a pressure extra or a fall / ETF turn entry is checked at production's two passes per scheduler
slot (slot + 90 s, slot + 11 min; the regular and extended slots, market days) while at most 75 minutes after its
bar's end; halted at the first pass it enters at its first free pass at the close of the last Polygon 1-minute bar
ended by then (in its traded dollars), after the target check (`target_reached`) and lendability judged then (as the
turn entries); halted at every pass, no trade. The vol arm runs production's pass loop (`halts.VolMachine`: the pick,
its fallback when unlendable — a halted backup skipped and left for a later pass —, then V6's extras while the pick is
pending; a halt never triggers the fallback). The variants: ``"wait"`` / ``"wait_half"`` (H1a / H1b: a pick halted at
every window pass waits to the next session's close, entered above the target / with half the run-up left),
``"backup"`` (H2: a halt at the first pass triggers the fallback), ``"both"`` (H3). Halts: NYSE's history
(`cache/ml/halts/`, ET; a duplicate listing merged; a halt without a resume over once a post-halt 1-minute bar with
volume has ended, at the latest the next regular open). No look-ahead: a pass sees only the halts started and the bars
ended by then (`tests/test_backtest_halts.py`, checked to fail on three injected leaks). Each delayed or new piece
carries ``halt``; `log["halts"]` holds the incidence (events, scope, counts). V6's extras need each store row's borrow
pressure (``eng.pressure_hi``, else V6's own NFHI, `halts.pressure_flags`): the pass loop without halts must reproduce
the simulator's selection and the piece builder its pieces, or the build stops. Not covered: the sell-limit option, the fall / ETF arms off the turn.
The 1-minute bars are cached per (symbol, day) (`halts.configure(minute_dir=...)`, default
`<research>/studies/prereg99/minute`). Study: `llm_trader_research/studies/prereg99/` (`build99.py`, `account99.py`).

## Validation (2026-10-08)

* Vol arm: the live book (2021-02..2024-06 and 2021-26, real NBBO, both margin triggers) through `account.simulate`
  == the audited cap5k7, bit-identical in 4 of 4 runs ($14,665.034514, 109 trades; $23,137.78 with the 2025 call);
  the vol adapter reproduces the same book; the default settlement calendar equals the research one (2,062 days).
* Dip book: all 3,417 of the study's signals reproduced (+137 Berkshire class-share signals the study's type lookup
  missed); the study's account within $38 ($21,030 vs $20,992, +16.3 %/yr).
* Study driver: PREREG35's per-week live scores reproduced on 42 windows (same trades; |diff| <= 0.02 pp/yr — the
  research evaluator re-prices a cut trade's exit spread at the mark bar).
* ETF dual momentum: the adapter's monthly picks 2021-02 .. 2026-09 equal the research's R4-DM1 in 68 of 68 months,
  and its account alone the research's (+7.84 %/yr to 2024-07-05, +14.80 %/yr to 2026-09-28, worst drawdown 23.8%).
* Tests: `tests/test_backtest_account.py`, `tests/test_backtest_study.py`, `tests/test_backtest_dm.py`.

## The live record against these assumptions (`src/backtest/live_calibration.py`, from 2026-10-12)

The EOD chain's last step (`live_calibration`) writes `cache/ml/live_calibration/<day>.json` every market day: the
session, the last five sessions and the record since the live account began (2026-10-09), each beside what this
simulator assumes —

| live fact (source) | the simulator's assumption |
|---|---|
| lendable at each live pick's FIRST borrow check (`sel_short/entries/`, one decision per pick) | IBKR's archive to 2024-06-24, the lend model after |
| IBKR's refusals among the stock picks its what-if answered; its quoted house rates (`ibkr_margin`) | `REFUSAL_RATE` 7%; a name of `house_mix()` drawn |
| a short's ledger entry: minutes after its decision bar's end, its price against that bar's close — the pick bar, or the turning bar for a turn entry (reported apart) | at that bar's close, stamped at its end |
| the broker's entry legs filled, slippage against the ledger's price | every ledger trade is the account's position |
| the exits by reason and session | the regular and extended sessions' ticks |
| IBKR's charged borrow (Flex, `broker_borrow_fees`) against the formula with our inputs (`borrow_fees.verify`'s construction); IBKR's quantity against the ledger's shares | the formula with IBKR's rate (its API fee for a listed name) |
| the dip orders by status (`dip_long/orders/`) | filled by rule, cut to the room |

A live rate outside its assumption beyond its sampling error — at least 5 observations; a share's Wilson 95% range, a
median's order-statistic 95% range (none below 6 values): IBKR's refusal rate, the median delay > 5 min, the median
entry drift beyond ±50 bp, the fill share's range under 80%, the house rate outside 0.5-2x the mix, IBKR's fee or
quantity outside 0.8-1.25x — over the whole record or the last five sessions becomes a note in the email's green
scorer line. Nothing trades on it: a gap is the next study's to fix in `mimic.py`. The ledger's own stored borrow
schedule is never the comparison (it takes IBKR's charges over the formula once they arrive), and the Flex statement
lags: the record reads what the table holds. Tests: `tests/test_live_calibration.py`.

## Use

```python
from src.backtest import account as A
from src.backtest.strategies import dip
from src.backtest.strategies.vol import VolEngine

eng = VolEngine()                                     # model spreads; VolEngine(nbbo=True) for the real NBBO
V = eng.pieces({}, "2021-02-01", "2024-06-24")        # the live vol rule; e.g. {"max_dtc": None} without crowding
D = dip.pieces(dip.DipParams(), "2021-02-01", "2024-06-24")
res = A.simulate(V + D, [A.vol_short("vol"), A.dip_long("dip")], A.Rules(trigger="high"), settle=eng.settle)
m = A.metrics(res, "2021-02-01", "2024-07-15")        # m["growth"], m["log_growth"], m["by_strategy"][...]
```

A new strategy needs only an adapter that emits pieces (dicts: `strategy`, `tkn`, `ens`/`xns`, `e`/`x`,
`hs_in`/`hs_out`, `dv20`, the held bars `pt`/`pc` (+ `ph`/`pl`), a short's borrow schedule, and `pick_day` for the
weekly studies) and a `Strategy` (side, slices, limits, house rates).

Weekly studies: `study.Study` (arms with their pieces per rule, the combo grid, the reference combo, the Sundays, the
windows) -> `study.run_weeks` (parallel; Optuna with `optuna_trials`, from `pylib/`) -> `study.chained` per method ->
`study.diff_ci` vs the reference. Example: `llm_trader_research/studies/prereg44/run.py`. Pre-register before running,
and run heavy studies outside regular hours at low priority (the scheduler's ticks come first).

## The test sets by borrow data (`src/backtest/periods.py`, user directive 2026-10-10)

"Evaluate all the years we have the borrow fees data vs those we don't." Every backtest of a short book, or of the
account holding one, runs on these sets — one per kind of borrow data — and is DECIDED on their union, THE POOLED
HISTORY A0 + A + B (`periods.POOLED`, 2018-01-02 .. 2026-09-24; user directive 2026-10-14: "check for the full dataset
(A0+A+B)"; the bank and the pooled add rule below), each set reported beside:

| set | decisions | borrow inputs |
|---|---|---|
| A — with IBKR's borrow file | 2021-02-01 .. 2024-06-24 | IBKR's short-stock file (community archive 2017-10-09 .. 2024-06-23): shares available + fee, every name |
| B — without it | 2024-06-25 .. 2026-09-24 | lendability drawn from the model (`mimic.lend`, `lend_model="v2"`, linked draws: `lend_persist`, `lend_day_lam`); IBKR's API fee for the names still listed (84% of the combination's short trades), the fee model otherwise (`fee_model`, `fee_version="v4"`) |
| C — our own archive | 2026-09-25 .. | our archive of IBKR's file: the live forward test |
| A0 — before the deep store (PREREG101; in the pooled history) | 2018-01-02 .. 2021-01-29 | IBKR's short-stock file (the community archive from 2017-10-09), as set A; its own data (below) |

`periods.LEND_ARCHIVE_END` / `OWN_ARCHIVE_FROM` are the mimic's lendability switch (`mimic` imports them), so a set
cannot call a modeled day known. Each set is its own $10,000 account from its first session taking only its own
decisions (`periods.windows(data_end)`); the calendar years come in their parts (`periods.year_parts`: 2024 A / 2024
B) with their returns read off the continuous account (`periods.growth(nav, lo, hi, annual=False)`). Example:
`llm_trader_research/studies/borrow_sets/run_sets.py` + `analyze_sets.py`; with the PREREG67 models
`studies/prereg67/rerun67.py` (the live combination: set A +24.6 %/yr, set B +54.4 %/yr; set B's lendable share 7 points
lower / higher: +54.2 / +58.1; the five draws alone span +29 .. +66). Tests: `tests/test_backtest_periods.py`.

THE POOLED HISTORY chains the sets: a pooled rate is the years-weighted mean of the sets' log rates
(`periods.pool_weights()`: A0 0.353, A 0.389, B 0.258 — each set's span over the pool's, `periods.years`), the pooled
worst drawdown of a draw the worst of its sets', its months and calendar years chained through the boundaries (a
month two sets share compounds both parts: `bank.combination`). One continuous account from 2018 replaces the chain
once PREREG105's rebuilt stores (A2 / B2, A0's construction) pass their review.

### Set A0 — the years before the deep store (`src/backtest/a0.py`, PREREG101)

An EXTRA set, asked for by name (`periods.get("A0")`; `set_of` / `windows` / `year_parts` with ``extra=True``; without
it every function reads A / B / C exactly as before). Its data are rebuilt by the same research code over 2017-06 ..
2021-06 (`llm_trader_research/studies/prereg101/`: `fetch101.py` -> `build101.py` -> `parity101.py`; `MAP.md` maps
every input): the bars of every common stock / ADR and product that cleared the live floors on its day (the names dead
before 2021 included; each name keeps the research engine's class, one stitched series per symbol as every store),
the vol store `optvol/store_a0` (`build.main`), IBKR's archived lendability per row (`borrow_arrays` from 2017-10-09),
IBKR's fees (its API bar where the name's API history covers the day, else its archived file: `a0.real_fees`), the ETF
arm's picks, the dip panel, the crash book's QQQ / TQQQ bars, the extended-hours bars, and V6's pressure as
production builds it (`sel_pressure.build` over A0's sessions; its model was trained on 2017-10 .. 2024-06: in-sample
there). The components: `VolEngine(period="A0")`, `EtfArm(period="A0")`, `a0.dip_panel()` (the dip adapter reads a
panel's own ``bars_fn`` / ``eligible``), `crash.pieces(..., period="A0")`, `ExtBars(period="A0")`, `a0.settle()`.
`live_books` on a window that starts before 2021-02-01 (or with an A0 engine) takes A0's components — loading them when
none is given and refusing another period's (a default engine would build nothing there). A study runs its A0 variant
inside `a0.harness(R63._W)`, which swaps the harness's engine / ETF arm / panel / dip orders / bars / settlement for A0's,
installs A0's V6 flags in the loaded `run71` and A0's halt files in `halts` (`set_halt_paths`), all restored after.
The live configuration's 20 stratified draws: `studies/prereg101/account101.py` -> `studies/prereg95/A0_LIVE_S2_s<seed>.pkl`
(the fields of A's and B's). Tests: `tests/test_backtest_seta0.py`.


## The reliability card (`src/backtest/reliability.py`, user 2026-10-12: "Implement all 6")

One card per live arm, the same statistics for all, built from the production adapters: `python -m
src.backtest.reliability --build [--sets A,B] [--no-sensitivities] [--seed N]` → `cache/ml/reliability/<day>/`:

* `<arm>.json` — the arm's `SPEC` (mechanism, context condition, horizon, parameters), `overall`, `by_year`, `by_set`,
  `by_trend` (SPY above / below its 200-session average at the entry), `by_vol` (SPY's 20-session realized volatility
  in terciles over its history), `loses_in` (groups of ≥ 10 trades with a negative average trade), `sensitivity`
  (`costs_x2`: the pieces' leg costs doubled; `entry_delay`: `Mimic().without("entry")`, the next bar's close;
  `lend_lo` / `lend_hi`: `Mimic(lend_shift=∓0.37)`) with the change in return per day overall and by set. Each
  statistic: n, the average and median net trade, the average 24-hour hold, RETURN PER DAY (`account.per_day`), the
  GROSS win rate (the raw-price direction; `win_basis` says when only the net one was available), the worst and best
  trade and the five best trades' share of the P&L.
* The trades are the RULE'S: each arm's pieces (`mimic.live_books` under `Mimic()`, both borrow sets) through the
  account engine ALONE with a $100,000 start, so no slice is cut, nothing gives way and no margin call occurs
  (`arm_alone`); `trades_from_positions` reads an account run's positions instead (a study's compact copy carries no
  gross return), `trades_from_table` a study's long history (the dip book's signals 2007→, the crash book's 1986→).
* `bundle_arms_alone.json` — `bundle`: per arm the share of sessions with a position open, trades, P&L share, mean
  and peak notional share; the overlap matrix (the share of sessions two arms are both active); the co-activation
  headroom: the initial-margin requirement of everything open (a short at the house 2.86 unless its piece carries
  `hi`, a dip long 0.5, the crash long 0.75) against equity — its peak session, the arms active then, and
  `slice_multiple` = 1 / peak (an upper bound: the maintenance level lies lower).
* `reference.json` under `cache/ml/live_calibration/` — set B's base statistics per arm, read by the live calibration's
  `arms` section (item 6): the live arm's return per day and gross win rate against them, a gap noted from 5 trades.
* `plateau(points, tol=0.1)` — the rule for setting a parameter: of (value, metric) points on a grid, the run of
  neighbours within 10% of the best holding the best; its CENTER is the value to take; `on_peak` flags a run of one.

Studies: `python -m src.backtest.prereg <study_dir>` verifies the result files postdate the registration (and lists
what each amendment postdates); `src/analysis/stats.py` gives `pbo_cscv` (the probability of backtest overfitting by
combinatorially symmetric cross-validation over a periods × candidates matrix) and `deflated_sharpe` (the Sharpe ratio
deflated by the number of candidates tried) — report both with every multi-candidate study.


## The bank of models — the pooled history (`src/backtest/bank.py`, user directive 2026-10-14)

"Instead of checking for the three different sets (A0, A, and B), check for the full dataset (A0+A+B). Each individual
model needs to give growth on that whole set but doesn't need to have growth all the time ... we also need to have a
enough trades made so that the results are significant, reliable, and transferable to out of sample data ... we then
need multiple models in our bank of models so that we cover all market conditions." `python -m src.backtest.bank
[--no-write] [--runs DIR --label LABEL]` → `cache/ml/bank/<day>.json` (about 2 s). Tests: `tests/test_bank.py`.

* INPUTS. Each arm's trades: the reliability cards' `trades_base.pkl` (each arm alone through the account engine at the
  live costs, `Mimic()` seed 0), A0 from the latest `<day>_A0` build, A and B from the latest `<day>` build
  (`audit.card_dirs`) — rebuild them after a live change (`python -m src.backtest.reliability --build` and
  `--build --sets A0`). The daily-bar books' earlier years: PREREG103's LONG history (`table1[arm]["LONG"]["by_year"]`,
  the dip book 2007-2020 — survivorship-biased, context —, the crash book 1988-2020), only its years before 2018. The
  combination: the live configuration's draws with a daily NAV each (`LIVE_RUNS`: PREREG116's `Q1`, the
  configuration from the 2026-10-16 session; PREREG102's `LIVE_S3` for 2026-10-14's); a draw without a NAV is read from its positions' P&L realized at their exits
  (flagged ``realized``).
* `card(T, arm, regimes, long, slice_, n_trials, max_open, bench, long_trades)`: `overall` (`quality`:
  `reliability.stats` + the months, the skew,
  the month-clustered t of the mean trade and the return per day's 95% interval from `month_boot` — a bootstrap over
  ENTRY MONTHS, 10,000 resamples, seed 0), `halves` (the first and second half of its trades by time), `loyo` (the
  average trade and return per day with each calendar year left out), `by_year`, `by_set`, `context` (`trend`, `vol`,
  `cell`: SPY above / below its 200-session average and its 20-session realized-volatility tercile at the entry —
  `reliability.regime_table`, whole-history terciles, a reading label — each cell `EARNS` / `LOSES` / `unclear` from
  `MIN_CELL` 30 trades in `MIN_CELL_MONTHS` 10 months), `alone` (the arm alone at its live slice of $10,000: P&L
  realized at the exits and compounded — growth a year, the worst drawdown, every year), `bench` (a LONG arm's bar:
  its benchmark's return per day over the pool, over each half's span and with each year left out), `longest` (with
  the earlier years — `long` yearly summaries or `long_trades` a table: n, the mean, the clustered significance
  against the bar, the episodes, the halves before 2018 / 2018-2026 and leave-one-year-out against the bar, `checks`
  1-3), `dsr` (the deflated Sharpe ratio, `audit.dsr_trades`, N from `trials.count`).
* THE BAR (user directive 2026-10-15: "test long arms against the market's average daily rise. The idea is to beat
  the market but I'm also ok with using 3x for trades we are confident the market will go up (or down)."): a LONG
  arm's edge checks read its return per day against what the same exposure earns on an average day over the same
  span — `bench_rpd(daily, lo, hi)`: the benchmark's average daily price return over the average calendar days a
  session spans, the arms' own unit — SPY for a stock arm, the fund it trades for an index arm (a 3x fund's own
  average day: the leverage is never the edge, the timing is). `BENCH` = {dip: SPY, crash: TQQQ} (pinned for every
  long live arm by a test); `card(..., bench=<symbol or series>)` names a candidate's. `bench_daily(symbol)`: the daily
  cache's closes (`cache/ohlcv/<symbol>.json`), before its first session PREREG120's carried history
  (`audit.bench_history_json()`: SPY / QQQ / IWM stitched to ^GSPC / ^NDX / ^RUT before the ETF, UPRO / TQQQ / TNA
  synthetic before their inception — identical to the cache on their 2018-2026 overlap). On the pooled history SPY
  +0.038 %/day, QQQ +0.057, UPRO +0.103, TQQQ +0.150; from 1988 SPY +0.029, TQQQ +0.120. A SHORT arm's bar is 0 (a
  random short loses the market's rise: 0 is the stricter bar). Context labels and filters stay against 0.
* `qualify(card)`: QUALIFIES / INSUFFICIENT / FAILS on `enough_trades` (`MIN_TRADES` 100 in `MIN_EPISODES` 30 entry
  months), `significant` (the interval above the bar), `halves` (each half's return per day above the bar over its
  span), `not_one_year` (every leave-one-year-out above the bar without that year), `growth` (the arm alone grows);
  INSUFFICIENT = short of trades only; ``bar`` / ``bar_spec`` reported. THE LONGEST HISTORY decides checks 1-3
  (``pooled_checks`` kept beside) when the pool holds too few trades (`basis: "longest"`) and — THE RE-JUDGING, user
  2026-10-15 — when the arm fails ONLY significance on the pool with its return per day above its bar (`basis:
  "longest (re-judged)"`). Its significance is clustered: `long_trades` → the month bootstrap over every month of the
  whole table, the lower bound above the bar over the whole span; yearly summaries → the t of the mean trade against
  the bar's per-trade equivalent, each earlier year and each pooled month one cluster, the larger of that and the
  independent-trade standard error, at least `T_SIG` 1.96. Counting every trade as independent would have admitted
  PREREG110's gap-down reversal (t 2.66 on 2007-2026; the month interval −0.035 .. +0.575, PREREG119).
* `filter_test(t, removed)`: a context filter is JUSTIFIED BY ITS TRADES when it removes at least 30 trades in 10
  months whose return per day's interval lies below 0; the kept / removed / all qualities and the change in return
  per day beside. `walk_forward(t, choose, first_year=2020)`: `choose(the trades before Y)` returns a rule (a mask
  function, with an optional `label`) applied to Y's trades only — the out-of-time quality, the in-sample one and the
  decay. A filter decided on the history passes both before the pooled add rule.
* `combination(runs, regimes)`: per seed drawn in every set, the pooled growth (years-weighted log growth), the worst
  drawdown (the worst of its sets'), the margin calls (per set, and the draws with one anywhere), the MONTHS (the share
  positive, the worst and median month, the worst 12 months, the longest time under water), every calendar year
  (mean, min, max over the draws) beside SPY and QQQ (price only, `cache/ohlcv`), and `coverage`: the months by the
  context at their first session (trend, volatility, cell) — months, the average month, the share positive, the
  growth a year, each arm's share of the P&L realized in those months.

* CONTEXT ACTIVATION (PREREG108; user 2026-10-14: "a model should understand the context and be activated if
  historically it performs well in it. If it is activated, then it can be used to trade."). `src/backtest/context.py`:
  the cell in force for a decision (`context.at(days)`: the close BEFORE it — SPY above / below its 200-session average
  x its 20-session realized volatility in terciles of its own history up to that close, an expanding window from
  2006-08-14, 250 sessions at least; `table`, `cached_table`; `tests/test_context.py` checks a cut history and rewritten
  later closes change nothing). `bank.activation_map(t, asof, rule)`: an arm's cells on / off from its trades CLOSED
  before ``asof`` (columns ``cell``, ``net``, ``exit``, ``dmonth``, ``w`` = 1 / the draws when draws are pooled) —
  "shrunk" (K1: the cell's average trade shrunk toward the arm's own with a prior of `ACTIVATION_PRIOR` 30 trades, on
  above 0), "proven" (K2: 30 trades in 10 months and a month-clustered t >= 1.96), "optout" (K3: off only at t <= -1.96
  on as many); no map before a year of closed history (`ACTIVATION_BURN_DAYS`); `bank.schedule` month by month,
  `bank.activated` the trades an arm then takes. A switched-off cell keeps its shadow record (the map reads the arm's
  record in every cell), so it can come back on. `Mimic(activation=<schedule file>)` replays a schedule in the account
  (`mimic.apply_activation`: an arm's pieces decided in a cell its month's map leaves out are dropped — fallbacks and
  extras included —, counted in `log["activation_<arm>_off"]`). Studies: `studies/prereg108/` (`maps108.py` ->
  `schedule_K1/K2/K3.json` + `maps108.json`; `account108.py`, staged by the pooled rule: `study102.Study(stage40=
  "pooled")`). First stage (out of time, the live configuration's 20 draws): K1 switches the dip book off in 18% of
  its trades (+0.06 %/day against +0.19 for the kept ones; above SPY's trend at mid volatility in 2023-24) and the ETF
  arm in 17%, the vol, fall and crash books nowhere; K3 switches nothing off; K2 keeps too few trades (the vol arm 47 of
  258, the ones it drops as good as the ones it keeps).

* RESEARCH BOOKS (PREREG110): a study adds a candidate arm to the live combination without touching production's
  books — `mimic.register_research(name, pieces, side, slices, max_open, rate, priority, quality, book)` in the
  study's process (``pieces(lo, hi, period=None | "A0")`` builds the arm's account pieces, the crash book's adapter
  the template), then `Mimic(research=(name, ...))`: `live_books` builds its pieces (set A0's data on an A0 window)
  and charges both legs at the ledger's costs, `live_strategies` adds its Strategy (1/``slices`` of the equity, at
  most ``max_open`` open, the margin ``rate`` initial = maintenance, its ``priority`` and capital ``quality`` — 0.2,
  the crash book's, takes the dip book's room). Default () = production's books; a production book's name is
  refused (`tests/test_backtest_research_books.py`). `studies/prereg110/account110.py` runs the pooled add rule on it.
* STACKING CAP (PREREG113, risk protection): `Mimic(short_name_cap=K, short_name_scope="arm" | "all")` — each short
  piece's group is its traded name and each short arm gets the engine's group cap: at most K open shorts of one name
  per arm ("arm") or across the three short arms ("all"). None = production's stacking (no cap).
* THE DIP BOOK'S ORDERING (PREREG107; user 2026-10-14: "does it favor its best picks first?"):
  `Mimic(dip_rank="score")` places the dip orders by the meta-model's walk-forward score, highest first (production:
  "rsi2", the lowest RSI2 first), `dip_topk=k` keeps only each signal day's k best orders, `dip_slices=n` sets the
  dip book's slice count (as many open as slices) — `strategies.dip.rank_pieces` with PREREG96's score map
  `filter_scores()` (every order of A0, A and B scored; an unscored order ranks after the scored ones by its RSI2;
  `tests/test_backtest_dip_rank.py`).
* `bank.alone(..., max_open=K)` / `bank.card(..., max_open=K)`: the arm-alone growth check skips an entry while K
  positions are open (counted in ``skipped``) — an arm whose trades overlap heavily is otherwise read at many times
  its equity (PREREG112's insider clusters: 283 at once). None (the default) takes every trade, as the live arms'
  own rules bound them.
* The study harness's fingerprint (`studies/prereg102/study102.py`, `LATE_FIELDS`) leaves a field added after the
  live draws out while it holds its default (`activation`, `research`, `short_name_cap`, `short_name_scope`,
  `dip_rank`, `dip_topk`, `dip_slices`), so the live draws S3 stay the paired reference.

First reading (2026-10-14; cards built before the vol give-back 1.25 and the ETF arm's VIX block, the combination the
configuration of 2026-10-14):

| arm | verdict | trades / months | return per day (95%) | gross win | alone at its slice |
|---|---|---|---|---|---|
| vol | QUALIFIES | 237 / 82 | +0.590 (+0.191 .. +0.943) | 82% | +13.5 %/yr, worst drawdown 36% |
| etf | FAILS (not significant) | 192 / 79 | +0.244 (−0.433 .. +0.894) | 82% | +1.9 %/yr, 29% |
| fall | INSUFFICIENT (71 < 100) | 71 / 48 | +0.950 (+0.140 .. +2.004) | 89% | +4.0 %/yr, 11% |
| dip | QUALIFIES | 1,490 / 94 | +0.180 (+0.108 .. +0.259) | 65% | +20.6 %/yr, 16% |
| crash | QUALIFIES (longest: 118 trades 1988-2026, t 3.5) | 17 / 15 | +1.576 (+0.430 .. +3.794) | 76% | +2.8 %/yr, 2% |

Second reading (2026-10-15, the cards rebuilt under the live configuration — the vol give-back 1.25, the ETF arm's
VIX block; `cache/ml/reliability/2026-10-15*`): vol QUALIFIES (251 / 85, +0.724, +0.406 .. +1.010, 81%, alone +20.4
%/yr, 19%), ETF QUALIFIES (141 / 63, +0.487, +0.021 .. +1.076, 78%, +4.4 %/yr, 9%), fall INSUFFICIENT (78 / 50, +1.717,
+0.767 .. +2.980, 91%, +7.4 %/yr, 11%), dip and crash unchanged. No arm significantly loses in any point-in-time market
cell (`bank.filter_test` per cell). PREREG108's activation maps REJECTED (K1 +0.12 +- 0.16 pooled, 2023 carries it).

Third reading (2026-10-15, the bar and the re-judging; `cache/ml/bank/2026-10-15.json`, PREREG120): every live arm
keeps its verdict — the dip book QUALIFIES against SPY (+0.180, +0.108 .. +0.259, against +0.038; its halves +0.099 /
+0.265 against +0.030 / +0.054), the crash book QUALIFIES on its longest history against TQQQ's average day (118
trades 1988-2026, +3.22% a trade against +0.60% over its hold, clustered t 2.71; before 2018 +0.53 %/day against
+0.11, 2018-2026 +1.58 against +0.15). PREREG110's candidates (`studies/prereg120/reread120.json`): the bull-market
index dip on SPY in UPRO QUALIFIES (346 trades 1986-2026, +0.303 %/day, +0.189 .. , against UPRO's +0.071); its QQQ
version in TQQQ now FAILS by a hair (+0.276, +0.116 .. +0.427, against TQQQ's +0.118); the turn of the month fails
leave-one-year-out against its fund (no re-judging); the gap-down reversal (2007-2026: +0.263, −0.035 .. +0.575) and
the calm-bull momentum (2008-2026: +0.057, +0.005 .. +0.106, against SPY's +0.030), both re-judged, FAIL.

The combination: +48.5 %/yr (draws +39.8 .. +58.1), worst drawdown 27.8 (37.0), a margin call in 10 draws of 20 (A 1,
B 9), 78% of months positive, the worst month −16.4% (−27.2% in the worst draw), the worst 12 months +7.8% on
average, every calendar year positive on average. Coverage (growth a year in the months of each context): above
SPY's trend +65% (high volatility) / +50% (mid) / +46% (low), below it +36% (high; the vol arm 40% of the P&L, the
crash book 25%, the ETF arm −18%) — no month below the trend in low volatility, three in mid. The vol arm's 2026 in
its seed-0 card (three −300% squeezes) is that draw's lendability: across the combination's 20 draws its 2026
average trade is +3.8% (−20 .. +33).

Fourth reading (2026-10-15 evening, PREREG116's Q1 deployed for the 2026-10-16 session: the cards rebuilt under
`Mimic()`'s new defaults, the combination read from Q1's 20 draws a set; the earlier cards in
`cache/ml/reliability_archive/2026-10-15*_preQ1`): the dip book — each day's top pick — QUALIFIES on 757 trades
(+0.249 %/day, +0.142 .. +0.359, gross win 68%; A0 / A / B +0.210 / +0.155 / +0.353 against every signal's +0.152 /
+0.061 / +0.290; leave-one-year-out at least +0.23); the vol arm 249 trades (+0.718), the ETF arm 138 (+0.503,
+0.041 ..) — the two-per-name cap removes two or three trades of each; the live calibration's set-B reference for the
dip book is now +0.354 %/day, gross win 70% (+0.291, 68% before). The combination: +52.8 %/yr (draws +42.7 ..
+63.4), worst drawdown 26.4 (38.0), a margin call in 7 draws of 20 (A 1, B 6), 80% of months positive, the worst
month −16.0% (−27.2% in the worst draw), the worst 12 months +6.7% on average, every calendar year positive in
every draw; below SPY's trend in high volatility +39% a year.


## The add rule, mechanically (`src/backtest/add_rule.py`; the pooled rule from 2026-10-14)

User 2026-10-13: "Apply the best decisions you're recommending" (PREREG95); user 2026-10-14: the pooled history. A
variant of the live combination is judged PAIRED against the live configuration on the SAME seeds, `Mimic()`'s
stratified draws, on the pooled history A0 + A + B — `pooled_verdict(live, variant)`:

1. `DRAWS` = 20 paired draws per set (four blocks of five): "more draws" below 20 — unless, from 5 paired draws in
   every set, the pooled gain lies more than 3 standard errors below +0.5: then "reject" at once (futility).
2. Growth: the POOLED gain — the years-weighted mean of the sets' paired gains (`periods.pool_weights`), its standard
   error sqrt(Σ w² se²) — at least +0.5 log points a year. No set must gain on its own and no similarity between the
   sets is asked. A pooled gain within 2 standard errors of +0.5 runs 40 draws in every set and is then judged on its
   estimate. The standard error is the simulation's: the market's history is not asked for significance (user
   2026-10-10) — what the change adds must instead rest on enough trades (the bank, above).
3. Not one year: with per-year records (`year_lp(nav days, nav values)` → each draw's `"years"`), the pooled gain stays
   above 0 with ANY one calendar year left out (`loyo`); records without them answer "needs per-year growth".
4. Margin calls: no more (set, draw) pairs meet one than the live configuration's, summed over the pool.
5. Drawdown: the pooled worst drawdown — per seed the worst of its sets', averaged over the seeds every set has — at
   most 1 point deeper.
6. `risk_protection`: the growth fails, but fewer margin calls or a drawdown more than 1 point shallower, none deeper —
   the user's call (reported at futility too once every set holds 20 draws).

Inputs per set: {seed: {"growth": fraction, "maxdd": points, "calls": [...], "years": {year: log points}}} — an
account run's record and its daily NAV (save `nav` = {"days", "values"} with every draw, as PREREG102 does).
`pooled_summary(verdict)` gives one line for a study's log: the decision, the pooled gain, calls and drawdown, each
set's gain and the worst leave-one-year-out. The noise (20 paired draws): a set-B rule change's SE ~0.8 log points
(40 draws ~0.6), set A ~0.45 (PREREG95); pooled 0.1-0.35 (PREREG96's RP33 0.12, PREREG102's E1 0.22 and E2 0.32).

`verdict(live, variant)` is the TWO-SET rule every study before 2026-10-14 decided on — sets A and B, each at least
+0.5 log points and the smaller at least half the larger, margin calls and drawdown per set, futility per set — kept
unchanged to reproduce them (`tests/test_add_rule.py` pins it byte for byte). The first pooled readings
(2026-10-14): PREREG102's VIX block (E1 against E0, with the VIX-linked products) +2.15 ± 0.22 pooled but carried by
2020 alone (−0.98 without it) — RISK PROTECTION: 29 → 10 (set, draw) pairs with a margin call; dropping the ETF arm
(E2 against E1) −1.72 ± 0.32 — futility, RISK PROTECTION (calls 10 → 6, drawdown 27.8 → 23.4); PREREG96's RP33 dip
filter +1.94 ± 0.12 pooled (A0 +1.67, A +3.32, B +0.23), one more margin-call pair (10 → 11); at 40 draws a set
+1.81 ± 0.10, calls 21 → 23: REJECT stands.

The night of 2026-10-14/15 (`memory/pooled-retune-new-arms-2026-10.md`; studies PREREG107-115): PREREG109's pooled
retune (PREREG97's 21 parameters re-run under the live configuration, every trade kept) leaves every live value on its
pooled plateau; its one MOVE (the days-to-cover cap 1.5) fails the account (−0.51 ± 0.47, margin-call pairs 10 → 31:
every A0 draw called in March 2020). PREREG113's stacking caps: C1 (one short per name per arm) −3.17 ± 0.39, calls
10 → 1, drawdown 27.8 → 23.5; C2 (two per name across the short arms) −0.01 ± 0.24, calls 10 → 8, drawdown 25.9 —
both RISK PROTECTION. PREREG107's dip ordering: O1 (best first) +2.92 but calls 10 → 12 (REJECT); O3 (the day's top
3) ADD +2.95 ± 0.18; O4 (the day's top 1 at a 1/5 slice) ADD +5.00 ± 0.34 — exploratory, then CONFIRMED on 2010-2017
by PREREG115 (the dip book alone +14.9 vs +11.2 %/yr, 7 of 8 years; O3 and RP33 not). PREREG114's packages: J1 (O1 +
C2) ADD +2.78, J2 (RP33 + C2) ADD +1.88, J3 (O4 + C2) ADD +4.52 ± 0.32 (calls 10 → 7, drawdown 27.8 → 26.6; pooled
growth 48.5 → 55.4 %/yr). New arms: PREREG110's bull-market index dip (N3) qualifies at the trade level on QQQ and
SPY (`strategies/index_dip.py`); its account test and the other candidates' failures are in the memory note.
