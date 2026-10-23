# The multi-strategy backtest — runbook

User directive 2026-10-08: "Rework our simulator that consider a real environment with restrictions when trading so
that it can consider multiple strategies. The idea is to be able to reuse it to evaluate different models or
combination of models. It should be able to calculate the real fees and be able to know the available capital from
the 10000$ starting account. We should then be able to use it to get the growth ... and optimize our optuna studies."

Research tooling with TWO production readers — changes to them reach the scheduler: `live_calibration` (the EOD
chain's daily record of the live trades against the simulator's assumptions, and its line in the scorer banner) and,
from 2026-10-17, `mimic` (the news-hold shadow measures its would-be exits with `mimic.news_extend`; `src/signals/
news_hold.py`), which pulls in `account` and `periods`. The research state — what was tested, what is live, the open
leads — is `docs/RESEARCH.md`.

## The pieces

| Module | What it does |
|---|---|
| `src/backtest/account.py` | THE account: $10,000, no deposits, every strategy's trades in one IBKR margin account — longs and shorts, real fees, Reg T + IBKR house margin, available capital, margin calls, per-strategy attribution, both metrics (`simulate`, `metrics`) |
| `src/backtest/strategies/vol.py` | the vol short arm (and any variant of its rule) as pieces, from the research vol engine (`VolEngine`) |
| `src/backtest/strategies/dip.py` | the dip long book (and variants) as pieces, from the deep 30-minute store + the delisted names (`pieces`, `DipParams`) |
| `src/backtest/strategies/etf.py` | the ETF short arm as pieces (`EtfArm().pieces(lo, hi)`): the research ETF run's picks (`<research>/etf/vol_events_de_arm.pkl`) — its DEFAULTS (half give-back, filters, volatility exit, 6x cover, IBKR's archived lendability to 2024-06) reproduce the recorded 141-trade book exactly (`<research>/etf/bt_etf_parity.py`); `mimic.live_books` passes production's rule (the whole give-back, the turn, no VIX-linked product) |
| `src/backtest/strategies/dm.py` | the ETF dual-momentum book as pieces (`pieces(lo, hi, end=)`): the live decision code (`etf_dm.fund_stats`, `etf_dm.tbill_growth`) on the repo's daily OHLCV cache (2006 on), one piece per holding bought / sold at the month's first open, 30-minute marks from the deep store (daily before it); trades as `account.dm_long` (the whole account in one fund); all 68 monthly picks 2021-02 .. 2026-09 and the research's growth (+7.84 / +14.80 %/yr) reproduced |
| `src/backtest/study.py` | weekly Sunday walk-forward re-tuning of one or several arms TOGETHER, objective = the window account's log growth; Reality Check / argmax / Optuna picks; the chained test account |
| `src/backtest/mimic.py` | THE SIMULATOR MADE TO MIMIC PRODUCTION (PREREG62, below): `Mimic` (each fix a switch; its defaults FOLLOW production's settings), `live_books` / `live_strategies` / `live_rules` = the live combination as production runs it — the vol / ETF / fall shorts at 1/20 / 1/16 / 1/12 (at most two of one name), the dip book's daily top pick by score at 1/6 with the earnings dips ranked last, the crash book's TQQQ at 1/4 (75% margin), one $10,000 account. V6's pressure extras are installed on top by the study harness (`run71` via study102's `R75.Applied({})`), NOT by `live_books`: `Mimic()` alone, and so the reliability cards, leave them out |
| `src/backtest/strategies/crash.py` | the crash-rebound book as pieces: production's `crash_long` decisions on QQQ, held in TQQQ (`Mimic(crash=True)`) |
| `src/backtest/bank.py`, `reliability.py`, `add_rule.py`, `periods.py`, `context.py` | THE EVALUATION STANDARD: the bank's qualification and the combination's pooled profile; each arm's card (its `SPEC` names its settings; the live calibration's reference carries each arm's settings fingerprint and the simulator's, and is STALE for the live score once a deploy changes either: rebuild the cards after every deploy — `--build`, `--build --sets A0`; `bank.load_runs` names every book by its live arm, PREREG165 / 167's "capb" as "cap"); the pooled add rule; the borrow-data sets (A0, A, B, C, D); the market contexts (CLAUDE.md's standing rules) |
| `src/backtest/a0.py`, `halts.py`, `quotes.py`, `dividends.py`, `audit.py`, `trials.py`, `prereg.py`, `live_calibration.py` | set A0's components (PREREG101); the halt models (PREREG99; production's guard ON by default from 2026-10-19); the legs' real quoted spreads and the dividends (AUDIT151, ON by default); the overfitting audit and the trial ledger (PREREG106, run by hand); the pre-registration checker; the live record against the simulator (a production reader) |

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
  below maintenance = a margin call that closes everything at twice the entry half-spread (`Rules(liquidation="all")`,
  the audited engine's) — or, with `Rules(liquidation="ibkr")` (PREREG174; `Mimic()`'s default), IBKR's liquidation:
  the most recently opened positions first, in 100-share lots, only until equity is back at the requirement
  (below, "IBKR's liquidation at a margin call").
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
  the return per day's entry-month bootstrap interval above its BAR — 0 for a short arm, for a long arm the
  market's (SPY's) average day, whatever it holds (PREREG123) —, both chronological halves and every leave-one-year-out above
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
  book's 0.13, the fund's `crash_long_margin_rate` 0.75 as its ``house_init`` / ``house_maint``). The account as
  deployed on 2026-10-12: `studies/prereg88/final88.py` (its default reproduces the research run `account88.py`'s
  Q-TQQQ/4 exactly; today's deployed draws are PREREG137's DMTE, `bank.LIVE_RUNS`); the TQQQ / FNGU variants:
  `studies/prereg88/account88.py`. The extended-hours marks (``ext``) leave the crash book alone: only the shorts
  and the dip book are marked at the extended-hours ticks.
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
* ``trigger`` — margin at the tick marks (`Rules(trigger="close")`) with the earlier margin calls (``margin="all"``;
  IBKR's liquidation, the default from 2026-10-21, judges it on the bars' extremes); ``order`` — a bar's shorts by ticker, then arm
  (`rank_shorts`); ``cagap`` — the ETF arm skips a corporate-action gap bar.

Not modelled by default (overnight prices exist as an option, `overnight=`, off): in-window borrow retries before
2026-09-25, late prints, overnight prices, a leg the broker filled (its own realized cost). Margin in real time is
approximated on the bars from 2026-10-21 (PREREG174, below; before it the engine judged it at the tick marks:
AUDIT151's M_TRIGGER bracket, -4.6 pp with 15 call draws of 20 at each bar's extreme with everything closed). From
2026-10-19 the halt guard, the legs' real quoted spreads, lendability at the trade, the dip's attainable
open fills and the dividends ARE modelled — the next section. Tests: `tests/test_backtest_mimic.py`.
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
`Mimic()` + V6 reproduced it on 2026-10-10 (today's defaults differ: the vol give-back 1.25, the VIX block, Q1, the
earnings demotion, F4 fees with linked lend draws, stratified draws). `llm_trader_research/studies/prereg77/` (`run77.py`, `final77.py`, `parity77.py`).

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
``short_name_scope`` `sel_short_max_open_per_ticker` (2) across the three short arms ("all"). `Mimic()` WAS
PREREG116's Q1 (`Mimic(dip_rank="score", dip_topk=1, dip_slices=6, short_name_cap=2, short_name_scope="all")` with
study102's `BASE_KW`) until 2026-10-17; it is now Q1 plus the earnings-dip demotion (`dip_earnings=3`, next section —
PREREG137's DMTE); `Mimic.off()` keeps the rules before (every signal an order at 1/10, no cap). The walk-forward
scores (`dip.filter_scores`, PREREG96's `prod96/wf_scores_prod.pkl`) run to the signals of 2026-10-06: a later signal is
unscored and ranked by its RSI2 — production scores it with the model fit on the signals exiting before 2026-10-08.
Q1's draws (`studies/prereg116/runs/{A0,A,B}_Q1_s<seed>.pkl`, 20 a set, each with its daily NAV) were the paired
reference for the 2026-10-16 session's configuration; after the 10-16 store repairs their set-B draws no longer
reproduce, so PREREG136 re-ran them on today's construction as L136 (`live136_config.json`). From the 2026-10-19
session the live configuration adds the earnings-dip demotion and THE LIVE REFERENCE is PREREG137's DMTE draws (next
section; `bank.LIVE_RUNS`). study102's fingerprint (`dip_long_account_slices` among its settings, the Mimic fields off
their pre-Q1 values, `LATE_FIELDS`) refuses to pair a new run with draws of another construction — register a new
reference instead. Tests: `tests/test_dip_top_pick.py` (production and the simulator choose the same order; the defaults follow
the settings), `tests/test_backtest_dip_rank.py`, `tests/test_backtest_research_books.py`.

### The earnings-dip demotion (PREREG137's DEMOTE_E, from the 2026-10-19 session)

User 2026-10-17: "Fully deploy the earnings dip demotion in live". `Mimic.dip_earnings` follows production: the window in
sessions (`dip_long_earnings_demote_sessions`, 3) while `enable_dip_long_earnings_demote` and the daily top pick are
on, else None. `mimic.earnings_demoted(oids, n, a0)` flags the dip orders whose company released earnings (an 8-K /
8-K/A naming Item 2.02) in production's window (`dip_earnings.window_ns`: after 16:00 ET of the session n before the
signal session, by D's 09:30) on the deep store's and the delisted company store's filings (`dip_earnings.release_table`;
a delisted ``XYZ@D`` reads ``XYZ_DELISTED``, set A0's symbol its ticker by identity, PREREG136's ``names.pkl``), and
`dip.rank_pieces(demote=...)` ranks them after every other order of their day (their rank + 1e7: they also fill last).
``dip_demote`` (a research pickle of order ids) replaces it when given. PARITY (2026-10-17): the flags equal
PREREG137's `sim137.py` on all 4,730 orders of A0, A and B (386 flagged; the 1- and 5-session windows 193 / 515), and
`Mimic()` reproduces the study's DMTE draws exactly (A s0 / s1, B s0, A0 s0 / s1: growth, drawdown, calls, the daily
NAV) — the DMTE draws (`studies/prereg116/runs/{A0,A,B}_DMTE_s<seed>.pkl`, 20 a set) are the deployed configuration's,
`bank.LIVE_RUNS`. A study reproducing Q1 or L136 must pin `dip_earnings=None` (study102's `LATE_FIELDS` keeps it out of
the fingerprint then). The acceptance instants: the deep store's are right for today's companies (185 of 186 sampled
against EDGAR's index pages), the delisted store's mostly 4-5 h early (Eastern time read as UTC) — on the index pages'
instants the 386 flags are unchanged. ALSO FIXED 2026-10-17: `rank_pieces(by="rsi2", topk=k)` chose each day's k best
by the SCORE (only the fill order read the RSI2); it now picks the deepest dip, production's "rsi2" — only PREREG137
Amendment 3's RS1 / RS1D ran it. Tests: `tests/test_dip_earnings.py` (the ranking by score and by RSI2, with the
demotion, is production's; the simulator's construction is production's `check`; the defaults follow the settings).

## The market model corrected (AUDIT151, the defaults from 2026-10-19)

User 2026-10-18: "Can you validate the backtesting and live rules and models? I want to be 100% sure there aren't
mistakes that could make the backtesting not representative of the live." — then, 2026-10-19: "Yes please do 1 and 2"
(the corrections as defaults). The audit (`llm_trader_research/studies/audit151/`, report
https://claude.ai/artifact/8YXe8DrUXiAMskzY6wCuLz) found the rules and models implemented the same live and in the
simulator, and seven mistakes in the simulator's model of the market, each measured against the deployed configuration
(20 paired draws a set, pp a year against 71.6%). All ON in `Mimic()`; `Mimic.off()` and `mimic.PRE151` (with
`strategies.dip.OWN_PANEL_NOWARM` / `OWN_SCORES_NOWARM`) are the construction before:

| switch | what | pp |
|---|---|---|
| ``quote_costs`` (`src/backtest/quotes.py`) | every market leg at Polygon's REAL quoted half-spread at its instant (a 09:30:00 leg read at 09:30:30), capped at 400 bp, + IBKR's exact commission / SEC fee / TAF on the shares (`Strategy.fees="engine"`): the shorts' legs, the crash book's, the dip book's exits and open fills (a touched resting limit pays none); the marks stay net of the ledger's estimate (``hs_mark``). The ledger's buckets it replaces (~23 bp all-in) were calibrated on calm names: real quotes average 51 / 83 bp at the vol arm's entries / exits. NOT the evaluator's fitted spread model (``VolEngine(nbbo=False)``), ~4x too wide | -4.32 |
| ``lend_intra`` (`mimic._lend_intra`, `lendable_at_trade`, `LEND_INTRA`) | lendability AT THE TRADE: the community archive's file is the MORNING file (~07:15 ET); a lendable pick, backup, extra or turn stays lendable with its morning pool's probability (<= $250k 0.65, $250k-1M 0.92, > $1M 0.98), a set-B draw 0.80, the ETF arm's set-B rate x 0.964 — our own per-tick archive, 16 sessions (refresh it as the archive grows); one keyed uniform per (name, session, draw) | -4.53 (brackets -3.53 / -5.29) |
| ``halts="guard"`` | production's halt guard at the short arms' entry passes (PREREG99's V0, below) | -0.68 |
| ``dip_open`` (`strategies.dip.reprice_open`, `OPEN_FILLS`) | a dip open fill at the price IBKR could give its resting order arriving 09:30:30 — the arrival trades' median at/under the limit, else the limit at a later touch (that 30-minute bar's end) — from Polygon's minute bars and trades for EVERY dip order piece that fills at an open in A0 / A / B (5,468 (name, day, limit) rows; `studies/audit151/dip_open/extend_fills.py`); a piece the table lacks keeps the print (``log["dip_open_unknown"]``) | +0.59 |
| ``dividends`` (`src/backtest/dividends.py`) | a short pays, a long receives every ex-date held through (entered before 20:00 ET of the session before it): the exit price and the marks from that instant raised by the cash; AUDIT151's frozen table (`build_table` makes a new one) | +0.18 |
| ``etf_reused`` False (`REUSED_ETF`) | the ETF arm never ranks the 8 tickers its research universe holds as products but which were stocks / SPACs when it picked them (GRIN, LAZR, SPAQ, ...): the picks re-ranked without them (`etf/vol_events_{de,a0}_reuse151_arm.pkl`) | +0.16 |
| the dip panel (`OWN_PANEL` / `OWN_SCORES`) | set A's dip book with its warm-up: the A / B panel started 2021-01-04, so the first set-A signal fell in November 2021; the 223 names it shares with the A0 panel carry their 2016-2020 rows, and the 2021 signals their walk-forward scores | +5.40 |

All together: 68.0% a year (-3.59 +- 0.85 pp), margin-call draws 7 = 7, the worst drawdown 31.1%. THE DEFAULT's own
draws (label DEF151, `studies/audit151/runs/`, `default151.py`) are the bank's reference (`bank.LIVE_RUNS`). PARITY
(2026-10-19): set B seed 0 reproduces ALLW's stored run exactly; on sets A and A0 every short, ETF and crash piece
equals runner151's patched construction re-run the same day, and the account results differ only where (i) the
default's open-fill table is keyed by (name, day, LIMIT) and covers every order — runner151's looked up 108 rows by
(name, day), so two orders of one name opening the same day shared one row and an order outside S3's 108 kept the
print — and (ii) IBKR's fee history was refreshed after ALLW's runs (`fee_api_daily.parquet` rebuilt 2026-10-18
23:59: IFBD@X's April 2024 fees). A study's legs outside the quote store pay their arm's AUDIT151 median and are
flagged (``q_fb``; ``log["quote_fallback_<arm>"]``): list the traded ones (`quotes.traded_misses`) and fetch them
(`python -m src.backtest.quotes --fetch legs.csv`) before deciding. A study built before 2026-10-19 refuses to run
under the new defaults (study102's fingerprint, `LATE_FIELDS`); to reproduce it pin `study102.PRE151_KW` and
`study102.pre151_panel()`. Tests: `tests/test_backtest_audit151.py`.

## The deployed parameters from 2026-10-20 (PREREG159's RPD_IS)

The user's deploy of PREREG152's return-per-day tuning on all nine years (2026-10-20, with the live account's fresh
start at $50,000): `Mimic()`'s defaults follow it, every value read from production's settings —
`strategies.vol.LIVE` (the vol give-back `sel_short_vol_give_back` 0.5, the time limit `hmax` =
`sel_short_arm_max_hold_sessions` 10, which the fall arm inherits and the shadow vol2 too — FROM 2026-10-21 the vol
arm's own `sel_short_vol_max_hold_sessions` 15, PREREG173's V15, the shadow vol2 too, while `strategies.vol.FALL` pins
the fall arm's `hmax` to `sel_short_arm_max_hold_sessions` 10: a default build reproduces V15's draws exactly,
`studies/prereg173/verify173.py`), `strategies.vol.FALL`'s
volatility exit at `sel_short_fall_volnorm_ratio` 0.7, the ETF arm's `give` (`sel_short_etf_give_back` 0.5) and `hmax`
(`mimic.live_books`), `strategies.dip.LIVE`'s exits (`dip_long_exit_sma_sessions` 3, `dip_long_max_hold_sessions` 5;
`DipParams()` keeps 5 / 10), the dip order's life (`dip_long_limit_sessions` 3), the slices (fall 8, dip 2 = at most 2
open, crash 3), the crash book through production's `crash_long` (the 250-session trend, the 3-session exit average, 5
sessions) and the de-leveraging (0.7 / 0.5). A default build reproduces PREREG159's draws — PREREG152's harness with
RPD_IS's overrides — EXACTLY (`studies/prereg159/parity_default.py`, seed 0 of A0, A and B: every record field and the
daily NAV), and the bank's reference is those draws (`bank.LIVE_RUNS` = `studies/prereg159/runs`, RPD152). The studies
before 2026-10-20 pin their own configuration (study102's `check_config` refuses a vol give-back other than 1.25;
PREREG152's harness applies its overrides over the defaults). `mimic.BOOK` holds the live arms only (the live
calibration's map, a test pins them equal); research books such as PREREG154's sleeve take theirs from
`mimic.RESEARCH_BOOK`.

A NEW LONG BOOK ON THE DEPLOYED CONFIGURATION (PREREG161-164, 2026-10-20): the dip book at 1/2 with 2 open takes the long
side's whole cap (`Rules.gross_cap` 1.0 x equity), so a research long book in the dip book's place (quality 0.13) blocks
its fills and measures that, not itself (PREREG164's E1_4 -16.2 pp a year, PREREG163's W3_4 -21.6); test a new long book
also in a GIVE-WAY place (quality 0.1, preemptible whole, de-levered first, `Rules.fills_make_room`: PREREG162's gold
book). The research harness `llm_trader_research/studies/prereg162/acct_rpd.py` runs RPD_IS's cached books (PREREG160's
`books/`) plus one book with its own `account.Strategy`, with the gate (without the book every draw equals PREREG159's
RPD152, or PREREG160's NOSLEEVE for draws 20-39) — the template for the next arm. The give-way place is where
PREREG165's capitulation book adds (+3.4 pp a year at 1/6, three open; in the crash book's place it takes the dip book's
room and loses on A0), its trades realizing ~0.3 %/day of their stage-1 0.68 (sold to make room). A whole-market,
survivorship-free daily panel for 2005-2026 (`studies/prereg165/panel165.py`: Polygon's grouped daily bars, the
delisted included, today's and dated security types, series cut at 30-day gaps) tests a daily-bar rule before 2018.

## The three changes from the 2026-10-21 session (PREREG167's ALL3)

The user's deploy of PREREG167 ("Deploy all three.", 2026-10-20): `Mimic()`'s defaults follow production's three
switches — ``dip_filter`` (`enable_dip_long_filter`: the dip book's orders filtered by RP33's walk-forward keep map,
`strategies.dip.filter_pieces`, before the $2B floor and the daily top pick), ``crash_pm_exit``
(`crash_long_pm_exit`: `mimic.pm_exit` on the crash book alone — every 09:30 exit moved to the 16:00-16:30 bar's close of
the session before, the crash book's leg costs left as they were; PREREG121's ``pm_exits`` still moves the dip book's
too) and ``cap`` (with `enable_cap_long`: THE CAPITULATION BOOK, `src/backtest/strategies/cap.py` — PREREG165's
inverted-trend dip signals kept on the sessions after a breadth of at least `cap_long_breadth_min`, read from PREREG165's
series `studies/prereg165/data/breadth165.pkl`, as the dip book's limit-order pieces (`strategies.dip.order_pieces` with
``signals_fn``) on the dip panel — set A0's for A0 —, costed as the dip book's (legs at their fills, the attainable open
price, the dividends); its strategy ``cap``: 1/`cap_long_account_slices`, ``max_open`` `cap_long_max_open`, priority 7,
quality 0.1, preemptible whole, the latest first, ``delever_order`` 0, ``orders="fill"``, book ``cap_long``; and
`Rules.fills_make_room` on, so a dip order is sized as if its lots were freed and its fill sells them — with
`Rules.fill_room_exact`, production's rule (`live_account.fill_give_way_plan`): the fill's own size, the room and its
side's cap without the slice, decides how many lots go; the slice-capped rule could sell a lot that added no share when
the order's quantity exceeded its slice at the fill price — identical on PREREG167's 60 accounts). `Mimic.off()`
has none of them; ``Mimic(**mimic.PRE167)`` builds the configuration before (RPD_IS). PARITY (2026-10-20, the session
scratchpad's `parity167.py`): the default `Mimic()` builds PREREG167's dip (811 + 319 pieces), crash (12 + 3) and
capitulation (746 + 141) pieces identically and reproduces its 60 accounts (20 draws of A0, A and B) exactly — growth,
the daily NAV, the margin calls —; the bank's reference is those draws (`bank.LIVE_RUNS` = `studies/prereg167/runs`,
ALL3). The research harness of PREREG152-167 (`studies/prereg153/joint153.py`, `prereg162/acct_rpd.py`) pins
`dip_filter=False` (study102's `BASE_KW`) and builds new books with the other two switches as production sets them: a
study on it that must reproduce RPD_IS passes ``PRE167``.

## IBKR's liquidation at a margin call (PREREG174 / 175, the default from 2026-10-21; production's too)

The user, 2026-10-21: "Research online what IBKR does exactly during margin calls. I want the backtest to do exactly the
same, and, if we can't be exact, approximate closely and be only slightly on the safe side." WHAT IBKR DOES (its "Margin
Calculations for Securities in Margin Accounts" page — the Reg T calculations; its knowledge base; the customer
agreement): excess liquidity = equity with loan value - the maintenance requirement (IBKR's, the larger of Reg T and its
house rate), checked in REAL TIME; below zero IBKR liquidates, without a call or notice, with MARKET orders, "to bring the
Excess Liquidity balance up to at least zero" — "our system is designed to liquidate Customer's most recent transactions
in a minimum of 100 share increments" — at the normal commissions; its SOFT EDGE MARGIN tolerates a deficit up to 10% of
the net liquidation value from the start of the trading day until 15 minutes before the close (then, and whenever SEM
does not apply, 100% of the requirement; once breached, the full requirement; IBKR "may eliminate SEM completely in
times of heightened volatility"); its end-of-day Reg T check wants SMA >= 0 (15:50-17:20 ET on a day with a position
change; SMA is at least equity minus the Reg T initial requirement). Which positions go is IBKR's discretion (a client's
order is not binding); whether it liquidates US stocks outside regular hours is not documented.

THE MODEL (`Rules(liquidation="ibkr", liq_fill="path")`, `Mimic(margin="ibkr", margin_fill="path")`, `mimic.margin_rules`;
tests `tests/test_backtest_ibkr_liquidation.py`):
* the check runs CONTINUOUSLY along each bar: every position moves together from its previous mark to its bar's
  adverse extreme (a short to its high, a long to its low) — no deficit inside a bar is missed, all extremes at once is
  the safe side —, on the regular-hours bars and, at each extended-hours tick, EVERY stored bar since the previous mark
  (PREREG175's ``margin_bars="all"``, the default from 22:05 ET: the hourly slots' first half-hours, 09:00-09:30,
  19:30-20:00 and the 04:00 bar no longer fall between marks; ``"tick"`` = the bar ending at the tick, PREREG174's), every
  long book marked at those ticks (``ext_longs``: the crash and capitulation books too, as production marks every book);
* STRICT at every hour (no SEM — the safe side: IBKR may eliminate it in heightened volatility, the account's margin
  events are squeezes, and IBKR's intraday house-rate raises on a squeezing name are not modelled); ``margin_sem=0.10``
  is SEM as documented on a normal day (09:30-15:45 ET, a bar ending at 16:00 at the full requirement, lost for the
  session once breached);
* at each crossing the open positions are liquidated the MOST RECENTLY OPENED first, in 100-SHARE LOTS — the fewest
  lots that bring equity up to the requirement there, the whole position when it holds fewer — and the path goes on;
  each lot fills 10% against the account from its price at the crossing (one limit-up-limit-down band: a market order
  in a squeeze meets about one pause), never past the bar's extreme, paying twice its entry half-spread + IBKR's
  commission (+ the SEC fee and TAF on a long's sale); every working order cancelled; liquidated shares are never
  bought back, a position partly liquidated keeps its rule;
* the end-of-day Reg T check at the first instant at/after 15:50 ET on its floor (equity minus Reg T's initial
  requirement; ``margin_regt``), at the instant's MARKS (PREREG175: production's check reads the live prices; it never
  fired at the extremes in PREREG174's 720 runs, so no run changed); ruin when nothing is left; ``res["liquidations"]``
  every lot, ``res["call_kinds"]``.
Switches: ``margin_fill="extreme"`` (every lot at the bar's extreme: the registered model, the conservative bracket),
``margin_check="ticks"`` (at the tick marks only), ``margin_order="requirement"`` (the largest requirement first),
``Mimic(**mimic.PRE174)`` / `Mimic.off()` (every position closed at once: the margin calls before). The reliability
cards keep the margin rule before it (`reliability.card_rules`: the arm alone at $100k never meets a call there, so a
card holds the rule's trades).

ON THE LIVE CONFIGURATION (V15, 20 draws a set, `studies/prereg174`): 76.6% a year with everything closed at the tick
marks -> 75.7% (-0.96 +- 0.20 pp), margin events in 13 draws of 20 (15 set-draw pairs) instead of 2, the drawdown (25.8%,
34.2% in the worst draw), the months and the worst 12 months (+9.3%) unchanged. In each event IBKR's rule is gentler (a
lot or two instead of everything: on the two old events +0.02 pp) but the real-time check catches squeezes the tick
marks miss — 11 of the 15 pairs are ONE after-hours bar, ICCT's de-SPAC squeeze of 2023-08-30 (a short entered that
morning at $7.30; the 17:30-18:00 ET bar to $42.00; 100 shares covered near $35.7). The bracket: at the ticks +0.02 pp,
every lot at the bar's top -1.12, everything closed at the bars -3.88 (AUDIT151's M_TRIGGER construction). SEM's 10%
removes 2 pairs (+0.04 pp); the largest requirement first -0.09 pp. PREREG173's V15 against ALL3 under it: +2.87 pp,
margin-event pairs 16 -> 15 — the add rule's ADD (its REJECT was one call pair under the old rule).

PRODUCTION LIQUIDATES THE SAME WAY (from 2026-10-21; the user: "production and backtest needs to liquidate like IBKR
does"; `live_account.plan_liquidation` = this model for the ledger's account, `margin_watch` = what each check reads —
the 30-minute bars completed since the last check and the live price —, `tracker.monitor_live_account` executes it,
`live_account_liquidation` "all" the rule before). PARITY: on the engine's own scenarios the plan's lots and fills equal
the engine's (`tests/test_live_liquidation.py`). PREREG175 ran the live configuration on the marks production's check
sees (every bar between the marks, every book marked at the ticks): 75.8% a year against 75.7% (+0.18 +- 0.02 pp),
margin events 17 in 14 set-draw pairs (12 draws) against 18 in 15 (13), the drawdown unchanged — the newly checked bars
meet no deficit the old marks missed; the long books' extra marks move a few give-ways and de-leveraging sales. The
bank's draws are these (`bank.LIVE_RUNS` = `studies/prereg175/runs`, NEWMARKS); ``Mimic(**PRE175)`` reproduces
PREREG174's.

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

``"guard"`` is the DEFAULT from 2026-10-19 (AUDIT151's H_GUARD: -0.68 +- 0.34 pp a year on the deployed
configuration; before it `halts=None`, every pick at its bar's close, and the gap was known: PREREG128 measured -0.44 log
points on Q1). `Mimic.off()` / `PRE151` keep halts unmodelled; the halt model covers production's construction only
(the fall and ETF turns, no sell limits) — any other construction needs ``halts=None``. The reliability cards read set
A0's halt files while A0's books are built (`reliability._halt_files`), and the pass loop adds V6's pressure extras only
when V6 is installed (`halts.v6_installed`: the engine's evaluator is ``Real6``) — the cards' plain evaluator has none,
though set A0's engine carries its pressure flags (2026-10-19: the A0 cards failed the loop's own selection check). ``"guard"`` (V0) is production's halt guard
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

## The 24-hour session (PREREG121, 2026-10-15; research options, all OFF by default)

User 2026-10-15: "We should optimize our trading on all sessions (regular, extended hours and overnight)." `Mimic()`
models the regular session and production's EXTENDED ticks (04:00-20:00, Polygon's 30-minute bars); the OVERNIGHT
session (20:00-04:00) holds positions unpriced. Polygon holds no overnight prints; IBKR's OVERNIGHT venue keeps 30-minute
TRADES and BID_ASK history from 2023-07-16/17 for listed names (none for delisted ones), fetched by
`llm_trader_research/studies/overnight24/fetch_overnight.py` into `cache/ml/overnight_ibkr/{bars30m,bidask1h}/<SYM>.parquet`
(one request at a time: two at once pushed IBKR's answers past ib_async's default 60-second limit, and a cancelled request
must never read as an empty history; status `status_v2.json`). The options:

* ``overnight`` "marks" / "exits" (default "off"): production's 8 overnight slots (`night_slots`: 20:30, 21:30, 22:30,
  23:30, 01:00, 02:00, 03:00, 03:30 ET) on the venue's nights from 2023-07-16 20:00 (`OVERNIGHT_FROM`), each reading the
  IBKR bar ENDING at it (`NightBars`, `night_marks`). `NightBars` checks each name against the extended store — a night's
  20:00 bar open against the store's 19:30-20:00 close; more than half the nights off by > 25% (`NIGHT_MISMATCH`) =
  another security, left out (`log["night_rejected"]`) — and scales the bars by the median ratio. "marks" marks every
  position there (the margin check runs on them); "exits" also judges the SHORT arms' exits there (`night_short`:
  `extend_short`'s target / volatility-exit / cover rules, the bar's dollar volume at least `NIGHT_EXIT_NOTIONAL` $7,500,
  exit `why` "night_*", the overnight leg cost the ledger's resolver charges). The long books' exits stay the rule's
  (`night_long`: marks only).
* ``pm_exits`` (X1): the dip and crash books sell at 16:30 (the 16:00-16:30 bar's close) of the session whose close
  triggered their exit instead of the next open (`pm_exit`; the dip leg at the after-hours cost — DEFECT: with
  `fees="engine"` the engine adds its commission and sale fees on top of the ledger's all-in after-hours cost, so the
  leg is charged twice; this and four other defects of these off-by-default options bias PREREG121 / 123's overnight
  verdicts, not re-run — `memory/data-audit-2026-10-16.md`, `docs/RESEARCH.md` section 5).
* ``crash_pm_entry`` (C1): the crash book buys TQQQ at its signal day's 16:30 instead of the next open (`pm_entry`).
* ``dip_ext_orders`` "ext" / "ext+night" (E1 / E2): the dip book's limit orders also work 04:00-20:00 from the signal
  day's 16:30 (and, "ext+night", in the overnight session) — `strategies.dip.order_pieces_ext`: a fill at the first bar
  opening at/under the limit (at its open) or reaching it (at the limit, the bar's end), outside regular hours only with
  the bar's dollar volume at least $15,000; a fill before D's open counts its exit from D's close.

Verdicts by the pooled add rule against Q1 (`llm_trader_research/studies/prereg121/`, `account121.py`, `verdicts121.json`):
X1 REJECT (-5.26 log points a year: holding to the open earns more than the after-hours leg saves), C1 REJECT (-0.60), E1
REJECT (-0.40), and corrected for the late prints below E1c REJECT (-0.51; set B +4.1, A0 -3.3, A -1.0 — 2025 alone
gains); N0 (overnight marks) +0.00: the same margin calls on the same days, no call hides in the unpriced nights; N1
(overnight exits) REJECT +0.03 (the window 2023-07 -> alone +0.07; overnight targets come out worse than the later
exit, the one avoided call a knife edge in one draw; at zero overnight cost +0.04 more, at IBKR's quoted overnight
spread +0.01); E2 (the dip orders on the overnight venue too) REJECT -0.34 (the window +1.91; +0.49 over E1c's: the
venue adds little to orders that lose before 2021 and in 2024). The 3x add-ons of PREREG123 / 124 — TQQQ / UPRO held
overnight, the turn of the month — fail the account on margin calls and worse months; PREREG122's overnight dip (M3) fails the bank (its 2024 'edge' was a data artifact — WDC's price regime, below; 2025-26 lose) (`research_costed`, `fees="engine"`,
`delever_order` below). The overnight runs read a FROZEN copy of the fetch's bars (`cache/ml/overnight_ibkr/bars30m_<name>/` with a
manifest, `overnight24/freeze_night.py`; `account121.NIGHT_DATA` points `mimic.OVERNIGHT_DIR` at it), never the folder
the fetch is writing. PRICE REGIMES (2026-10-16): `NightBars` scales a name's IBKR nights to the store by one ratio and DROPS the nights of another price regime (a corporate action one source adjusted and the other did not: WDC's nights sat at 0.757x the store's until the Sandisk spin-off, read as a -24% dip every night) — a night is kept only while the median of the name's last 5 checked nights is within 3% of the scale, the newest at most 21 days old, its own check within 10% (`NIGHT_REGIME_*`, `NightBars.dropped`); N0 / N1 / E2 re-run on it unchanged (N0r / N1r / E2r), M3's 2024 'edge' vanished. Tests: `tests/test_backtest_overnight.py`.

LATE PRINTS (amendment 2): overnight ATS trades executed 00:00-04:00 ET reach the tape when the FINRA TRF opens — 08:00
ET until 2026-03-27, 04:00 ET from 2026-03-30 — stamped then with the conditions of real-time extended trades (Form T),
so Polygon's 08:00-08:30 bar (before) / 04:00-04:30 bar (after) carries the night's prices in its open, high and low and
its volume (`mimic.late_print_bar`). A rule reading a pre-market bar's open / high / low must leave that bar out or read
its close only (`order_pieces_ext` does). The live configuration reads only closes at its ticks (0.4% of its short exits
at the 08:30 tick, consistent with the next open).

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

THE LIVE CONFIGURATION is `mimic.live_books` / `live_strategies` / `live_rules` under `Mimic()` (with V6 installed by
the study harness, `studies/prereg102/study102.py`'s `Runner`); start every account study there. The example below is
the engine's raw interface — the engine factories' defaults (`A.vol_short` / `A.dip_long`: 24 and 10 slices, the dip
book buying every signal at the open) are the rules of 2026-10-08, not today's:

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
| B — without it | 2024-06-25 .. 2026-09-24 | lendability drawn from the model (`mimic.lend_all` / `lend_draw_fn`, `lend_model="v2"`, linked draws: `lend_persist`, `lend_day_lam`); IBKR's API fee for the names still listed (84% of the combination's short trades), the fee model otherwise (`fee_model`, `fee_version="v4"`) |
| C — our own archive | 2026-09-25 .. | our archive of IBKR's file: the live forward test |
| D — the forward holdout (PREREG106) | C's days, simulated | our own archive of IBKR's file (`periods.get("D")`, `forward=True`); no study tunes on it |
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
PREREG101's draws of the 2026-10-13 configuration (LIVE_S2): `studies/prereg101/account101.py` ->
`studies/prereg95/A0_LIVE_S2_s<seed>.pkl` (the fields of A's and B's); the deployed configuration's A0 draws are
`studies/prereg116/runs/A0_DMTE_s<seed>.pkl` (`bank.LIVE_RUNS`). Tests: `tests/test_backtest_seta0.py`.


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
  (`arm_alone`; its margin judged as before PREREG174 — at the tick marks, `reliability.card_rules` — so a card holds
  the rule's trades, never a liquidation's lot); `trades_from_positions` reads an account run's positions instead (a study's compact copy carries no
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
  combination: the live configuration's draws with a daily NAV each (`LIVE_RUNS`: PREREG137's `DMTE`, the
  configuration from the 2026-10-19 session; PREREG116's `Q1` for the 2026-10-16 session's; PREREG102's `LIVE_S3` for 2026-10-14's); a draw without a NAV is read from its positions' P&L realized at their exits
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
  session spans, the arms' own unit. FROM 2026-10-16 (PREREG123; user: "if we pick a 3x ETF position in the right
  direction, this should be considered a good pick. So, it's not only about beating the market, it's also about
  following it with 3x leverage."): SPY for EVERY long arm, a stock, a 1x fund or a 3x fund — the leverage is part of
  the arm's return (until then an index arm's bar was the fund it trades: a 3x fund's own average day); an inverse
  fund's long is a down bet, bar 0. `BENCH` = {dip: SPY, crash: SPY} (pinned for every long live arm by a test);
  `card(..., bench=<symbol or series>)` names a candidate's. Under it TQQQ / UPRO held every session QUALIFY too
  (`studies/prereg123/reread123.json`'s references): for a 3x arm the account test decides, and its return per day
  beside the fund held every day tells timing from leverage. `bench_daily(symbol)`: the daily
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
  the crash book's, takes the dip book's room; None takes free room only). Default () = production's books; a
  production book's name is refused (`tests/test_backtest_research_books.py`). `studies/prereg110/account110.py` runs
  the pooled add rule on it. ``fees="engine"`` (PREREG123): the book's legs are fills at a known price (auction
  orders: a market-on-close buy at the official close, an opening-cross sell at the official open) that the ledger
  charges their own cost, so `research_costed` leaves the piece's half-spread and the engine adds IBKR's exact
  commission, as for the dip book's legs — the ledger resolver's class average (~20 bp a leg on TQQQ: the legacy
  pipeline's $1 minimum commission on ~$730 orders) is not what such a fill pays. ``delever_order`` (PREREG124): the
  book's place in the account's de-leveraging line (0 = sold before the dip book's 1; None = never de-levered).
* DE-LEVERAGING THRESHOLDS (PREREG125): `Mimic(delever=(trigger, target))` — the account sells the de-leverable books
  above ``trigger`` x equity in maintenance down to ``target``; None = production's `live_account_delever_trigger` /
  `live_account_delever_target` (0.8 / 0.6). A field added after the stored draws: study102's `LATE_FIELDS` keeps it
  out of the fingerprint at its default.
* NEWS HOLD (PREREG136, research): `Mimic(news_hold="<fb|lg>:<vol|shorts>:<k>")` — a short's own exit (never its
  squeeze cover) postponed to the same ET clock tick k sessions later, or earlier at its cover multiple, when its hold
  had the rule's news: "fb" = the company's last 8-K during the hold (at least 30 minutes before the exit) made the
  stock fall (its first mark 30 minutes after the acceptance under the last regular close before it); "lg" = the
  local LLM's mean tone of the hold's Polygon headlines is positive (the vol arm only; 2021+). `news_hold_piece`,
  `news_extend` (the extension's marks from `ExtBars`, its scale `robust_scale`, its borrow `_reborrow`), inputs
  `cache/ml/news_hold/` (`NewsInputs`: filings, rated headlines, the name -> company map; `studies/prereg136/
  build136.py`), applied after the overnight step and before the costs; None = production's exits
  (`tests/test_backtest_news_hold.py`). A field added after the stored draws: `LATE_FIELDS` keeps it out of the
  fingerprint at its default.
* DIP DEMOTION (PREREG137 Amendment 1, research, exploratory): `Mimic(dip_demote="<pickle of order ids>")` — the
  listed dip orders (`strategies.dip` oids "NAME|first session") rank after every other order of their day in the
  daily top pick (`dip.rank_pieces(demote=...)`; among themselves by the same key; a day with only demoted orders still
  buys its best); used with the earnings dips (an 8-K Item 2.02 accepted from 16:00 ET three sessions before the signal
  session to D's 09:30 open, `studies/prereg137/sim137.py`). None = production's ranking (``dip_earnings``, the
  earnings-dip demotion deployed from the 2026-10-19 session, above); a file given replaces it; `LATE_FIELDS` keeps it
  out of the fingerprint at its default.
* STACKING CAP (PREREG113, risk protection): `Mimic(short_name_cap=K, short_name_scope="arm" | "all")` — each short
  piece's group is its traded name and each short arm gets the engine's group cap: at most K open shorts of one name
  per arm ("arm") or across the three short arms ("all"). None = no cap; the DEFAULT is production's
  `sel_short_max_open_per_ticker` (2) across the three arms ("all").
* THE DIP BOOK'S ORDERING (PREREG107; user 2026-10-14: "does it favor its best picks first?"):
  `Mimic(dip_rank="score")` places the dip orders by the meta-model's walk-forward score, highest first (production:
  "score" with the daily top 1 — `Mimic()`'s defaults; "rsi2" = the deepest dip first, the rule to 2026-10-15),
  `dip_topk=k` keeps only each signal day's k best orders, `dip_slices=n` sets the
  dip book's slice count (as many open as slices) — `strategies.dip.rank_pieces` with PREREG96's score map
  `filter_scores()` (every order of A0, A and B scored; an unscored order ranks after the scored ones by its RSI2;
  `tests/test_backtest_dip_rank.py`).
* THE SCORE MAP (PREREG142, research): `Mimic(dip_scores="<pickle>")` — the walk-forward scores that rank the daily
  top pick read from that file (PREREG96's format: ``oid``, ``score``; `strategies.dip.filter_scores(path)`) instead of
  the default (`strategies.dip.scores_path()`); None = that default. PREREG142 paired its refit's map
  (`wf_scores_new.pkl`) with the deployed model's rebuilt on the same inputs (`wf_scores_old.pkl`) on the corrected dip
  panels (`panel142.pkl`, `panel142_a0.pkl`, put in the harness by `account142.py`'s Runner). `LATE_FIELDS` keeps it
  out of the fingerprint at its default.
* THE $2B FLOOR AT 1/3 (PREREG147's S3, production from the 2026-10-19 session): `Mimic(dip_min_order_dv=...)` keeps only the dip orders whose signal's dv20 is at least that many dollars before the daily top pick (`live_books`, after the RP33 filter, before `rank_pieces`) — the default is production's `dip_long_order_min_dollar_volume` ($2B) while the daily top pick is on, None otherwise; ``dip_slices`` None reads production's `dip_long_account_slices` (3 from the 2026-10-19 session). `Mimic()` reproduces PREREG147's S3 draws exactly (`prereg147/default_parity147.py`); `bank.LIVE_RUNS` reads them. A study reproducing the configuration before it pins `dip_min_order_dv=None, dip_slices=6` (PREREG142's R142_OLD); study102's fingerprint (it holds `dip_long_account_slices`) now refuses the older `live136_config.json` — a study of the deployed configuration starts from `prereg147/live147_config.json`.
* THE OWN-SECURITY DIP PANELS (2026-10-18, the reused-ticker fix; user: "Yes please apply the fix"): the DEFAULTS of
  `strategies.dip.build_panel()` and `a0.dip_panel()` are PREREG142's corrected panels (`OWN_PANEL` / `OWN_PANEL_A0`
  under `backtest_research_dir`: every series cut to its own security, every renamed company's earlier tickers joined —
  production's `src/signals/dip_security.py`, parity on 255 of the 256 changed names) and the default score map their
  signals' walk-forward scores (`OWN_SCORES` = PREREG142's `wf_scores_old.pkl`: model96's yearly models on the
  corrected inputs; `filter_keep(research=True)` still reads PREREG96's file, the only one with its keep flags). The
  plain study102 harness under the deployed configuration reproduces PREREG142's `R142_OLD` draws exactly (sets A0
  and A, every field and the daily NAV: `studies/prereg142/default_parity.py`), so `Mimic()` IS the corrected live
  account (+56.7% a year pooled; the bank's `LIVE_RUNS`). `strategies.dip.OWN_DEFAULT = False` (or `build_panel(own=False)`,
  `a0.dip_panel(own=False)`) restores the stored panels (`cache/backtest/dip_daily.pkl`, `dip_daily_a0.pkl`) and
  PREREG96's scores — every study before 2026-10-18 (PREREG137's `DMTE` draws: +56.1%). `build_panel(force=True)` rebuilds
  only the stored construction; the corrected one is PREREG142's build (`studies/prereg142`: spans142.py raw / events /
  spans -> bars142.py -> panel142.py).
* `bank.alone(..., max_open=K)` / `bank.card(..., max_open=K)`: the arm-alone growth check skips an entry while K
  positions are open (counted in ``skipped``) — an arm whose trades overlap heavily is otherwise read at many times
  its equity (PREREG112's insider clusters: 283 at once). None (the default) takes every trade, as the live arms'
  own rules bound them.
* The study harness's fingerprint (`studies/prereg102/study102.py`: built from `Mimic(seed=0, **BASE_KW)`, never from
  a label's config) leaves out a field added after the reference draws while it holds its "off" value, `LATE_FIELDS`:
  `activation` None, `research` (), `short_name_cap` None, `short_name_scope` "arm", `delever` None, `dip_rank` "rsi2",
  `dip_topk` None, `dip_slices` None, `overnight` "off", `pm_exits` / `crash_pm_entry` False, `dip_ext_orders` "off",
  `news_hold` None, `dip_demote` None, `dip_earnings` None, `dip_scores` None. Today's defaults are NOT those values (Q1's fields and
  `dip_earnings` 3 are on), so a study pairing with L136 (`prereg136/live136_config.json`) pins `dip_earnings=None` in
  its `BASE_KW` (`prereg137/account137c.py`); pairing with the deployed DMTE draws needs a new registered config.

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

Fourth reading (2026-10-16, every long arm against SPY's average day; `cache/ml/bank/2026-10-16.json`, PREREG123): the
crash book QUALIFIES on 1988-2026 against SPY (+3.22% a trade against SPY's +0.14% over its hold, clustered t 3.18);
every other live arm unchanged. Re-read (`studies/prereg123/reread123.json`), longest histories: now QUALIFY the index
dip in TQQQ (+0.276, +0.116 .., against +0.028), the turn of the month in TQQQ (+0.188, +0.050 .., re-judged) and
PREREG122's overnight holds — TQQQ on QQQ's uptrend nights (+0.112, +0.066 ..), TQQQ every night (+0.092, +0.041 ..),
UPRO on SPY's uptrend nights (+0.074, +0.047 ..); TQQQ / UPRO held every session QUALIFY too (+0.120 / +0.073). The
index dips and the turn of the month on QQQ earn more a day than the fund held every day; the overnight holds about
the same. The account tests (`studies/prereg123/verdicts123.json`) are the decision.

The combination of 2026-10-14 (PREREG102's LIVE_S3, `cache/ml/bank/2026-10-14.json`): +48.5 %/yr (draws +39.8 .. +58.1), worst drawdown 27.8 (37.0), a margin call in 10 draws of 20 (A 1,
B 9), 78% of months positive, the worst month −16.4% (−27.2% in the worst draw), the worst 12 months +7.8% on
average, every calendar year positive on average. Coverage (growth a year in the months of each context): above
SPY's trend +65% (high volatility) / +50% (mid) / +46% (low), below it +36% (high; the vol arm 40% of the P&L, the
crash book 25%, the ETF arm −18%) — no month below the trend in low volatility, three in mid. The vol arm's 2026 in
its seed-0 card (three −300% squeezes) is that draw's lendability: across the combination's 20 draws its 2026
average trade is +3.8% (−20 .. +33).

Fourth reading, b (2026-10-15 evening, PREREG116's Q1 deployed for the 2026-10-16 session: the cards rebuilt under
`Mimic()`'s new defaults, the combination read from Q1's 20 draws a set; the earlier cards in
`cache/ml/reliability_archive/2026-10-15*_preQ1`): the dip book — each day's top pick — QUALIFIES on 757 trades
(+0.249 %/day, +0.142 .. +0.359, gross win 68%; A0 / A / B +0.210 / +0.155 / +0.353 against every signal's +0.152 /
+0.061 / +0.290; leave-one-year-out at least +0.23); the vol arm 249 trades (+0.718), the ETF arm 138 (+0.503,
+0.041 ..) — the two-per-name cap removes two or three trades of each; the live calibration's set-B reference for the
dip book is now +0.354 %/day, gross win 70% (+0.291, 68% before). The combination: +52.8 %/yr (draws +42.7 ..
+63.4), worst drawdown 26.4 (38.0), a margin call in 7 draws of 20 (A 1, B 6), 80% of months positive, the worst
month −16.0% (−27.2% in the worst draw), the worst 12 months +6.7% on average, every calendar year positive in
every draw; below SPY's trend in high volatility +39% a year.

Fifth reading (2026-10-17 evening, PREREG137's earnings-dip demotion deployed for the 2026-10-19 session: the cards
rebuilt under `Mimic()`'s new defaults — `cache/ml/reliability/2026-10-17*`, Q1's in `2026-10-15*`, the live
calibration's reference before it in `reliability_archive/reference_2026-10-17_preDMTE.json` —, the combination read
from PREREG137's DMTE draws, `bank.LIVE_RUNS`): the dip book QUALIFIES on 760 trades (+0.275 %/day, +0.165 .. +0.386,
gross win 68%; A0 / A / B +0.237 / +0.176 / +0.382 against Q1's top pick +0.210 / +0.155 / +0.353), earning above
SPY's trend in every volatility tercile and below it in high volatility; the set-B reference for the dip book +0.383
%/day, gross win 71% (+0.354, 70% before); the short arms as before (the fall arm +1.702 %/day after the 10-16 data
repairs). The combination: +56.1 %/yr (draws +46.3 .. +65.9), worst drawdown 26.4 (37.7), a margin call in 7 draws of
20 (A 1, B 6), 81% of months positive, the worst month −16.2% (−27.9% in the worst draw), the worst 12 months +8.1% on
average, every calendar year positive in every draw; below SPY's trend in high volatility +38% a year.


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

## Research books without a production twin: the QQQ sleeve (PREREG154, rejected 2026-10-19; PREREG160, 2026-10-20)

`Mimic(sleeve=True)` (off by default; no production book) adds `strategies/sleeve.py`'s candidate LOTS — QQQ bought
at each 09:30 open while its previous close is above its 200-session average (the deep store's daily rows, set A0's
rows prepended for set A's warm-up), held to the open of the first session that is not, sold at the open after a
set's last decision — as the strategy "sleeve": the whole equity as its slice, at least `sleeve.MIN_FRAC` (5%) of the
equity a lot (the engine's `Strategy.min_frac`: a smaller market position is skipped, logged ``min_frac``), quality 0
(it gives way first, whole, the latest first), de-levered first, priority 99 (after every book at the same instant),
with `Rules.fills_make_room` (a dip order sized as if its lots were freed). The study's cushioned variants add the
engine's ``room_frac`` to the strategy. The registered sleeve cost 11.2 points of yearly growth (67.8 -> 56.6%; margin calls in 15 draws of 20 against 7);
a cushion of 30% added +3.2 points as a lone peak. On the configuration deployed 2026-10-20 (PREREG160, `studies/prereg160/run160.py`:
joint153's harness with RPD_IS's choices, its own book cache, the gate exact against PREREG159's draws) the registered
sleeve cost -8.5 points (66.4 -> 58.0%) with no added margin call, and the best cushion,
20%, added +3.1 as a lone peak (`memory/qqq-sleeve-2026-10.md`). Tests: `tests/test_backtest_sleeve.py`.
