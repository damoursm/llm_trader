---
name: evaluate
description: House standard for evaluating any model, signal, gate or exit rule in llm_trader — TWO KINDS of test sets: the NEWS test sets for signals and models, each reported separately, never pooled (news set 1 history 2026-06-17..09-27, news set 2 live all-source news from 2026-09-28), and the BORROW-DATA sets for strategy backtests, DECIDED on their union — the POOLED HISTORY A0 + A + B, 2018-01-02..2026-09-24 (set A0 2018-01-02..2021-01-29 and set A 2021-02-01..2024-06-24 with IBKR's short-stock file, set B 2024-06-25..2026-09-24 without it; set C our own archive from 2026-09-25, set D the forward holdout — src/backtest/periods.py), every set reported beside; model metrics = IC to the next H/L pivot, its day-clustered t, and the top/bottom 5%, 3% and 1% next-pivot returns (src/analysis/eval_metrics.py), plus the selection objective for models trained on it; gates/exits = counterfactual excess return per decision; the LIVE STRATEGY (the $10,000 live account) = return per day, the GROSS win rate AND growth per year on the pooled history (no deposits, the simulator that mimics production, src/backtest/mimic.py) with its years and months; each ARM qualified for the bank on the pooled history (>= 100 trades in 30 entry months, the return per day's entry-month interval above its bar — 0 for a short arm, the same exposure's average day for a long one: SPY, or the fund it trades —, both halves and every leave-one-year-out above it, growth at its slice; an arm failing only significance re-judged on its longest history, clustered — src/backtest/bank.py); a change by the pooled add rule (add_rule.pooled_verdict); label = the next H/L pivot on 30-minute bars from the tick, unresolved rows at the last close; simulated/panel rows, never the trade ledger.
argument-hint: "[what to evaluate, e.g. 'the ml_exit upgrade' or 'Gate 1c']"
---

Evaluate: $ARGUMENTS

**The standing rule.** Evaluate on the test sets (§1) with the metric set
(§2), against the next H/L pivot (§3), on simulated/panel rows (§4), long and
short. Signals and models use the NEWS test sets (§1a), each reported
separately, never pooled; strategy backtests — the live books, the account, any
short rule — run on the BORROW-DATA sets and are DECIDED on their union, the
POOLED HISTORY 2018-2026 (§1b), every set reported beside.

## 1. The test sets

### 1a. The news test sets (signals, models, deciders on the shadow pipeline)

| set | signal dates | what it is | news features |
|---|---|---|---|
| **News set 1 — history** | 2026-06-17 → 2026-09-27 | OHLCV + news since news collection began (first news pull 2026-06-16 08:00 UTC; first scored run 2026-06-17) | the per-source history rebuilt with today's news code: `news_replay` rows `pool_spec='src:<group>'`, read with `src.analysis.news_history.source_feature_frame()` |
| **News set 2 — live all-source news** | 2026-09-28 → | the live pipeline running the ALL-SOURCE news ingestion (every per-ticker feed asks about every scored name; live since `news_coverage.ALL_SOURCE_SINCE`) | the LIVE values the pipeline persisted — no reconstruction: the `signals` news columns, the archive (`news_articles` + `news_article_feeds` rebuild any single feed's pool) and `sentiment_digests`; the technical-indicator vectors from the live capture, `src.analysis.live_features.load("30m" | "daily")` |

### 1b. The borrow-data sets and the POOLED HISTORY (strategy backtests — user directives 2026-10-10 and 2026-10-14)

A short's borrow inputs exist for some years and not others: "evaluate all the years we have the borrow fees data vs
those we don't". Defined once in `src/backtest/periods.py` (`SETS`, `set_of`, `windows`, `year_parts`, `growth`);
their boundaries ARE the simulator's lendability switch (`mimic.lend_all`).

| set | decisions | what the simulator knows about borrowing |
|---|---|---|
| **A0 — with IBKR's borrow file, before the deep store** | 2018-01-02 → 2021-01-29 | IBKR's archived file (the community archive from 2017-10-09), as set A; its own stores (`src/backtest/a0.py`, PREREG101: the names screened day by day by the symbol traded then) |
| **A — with IBKR's borrow file** | 2021-02-01 → 2024-06-24 | IBKR's short-stock file (the community archive 2017-10 → 2024-06): shares available AND the fee for every name, the ones delisted since too |
| **B — without a borrow file** | 2024-06-25 → 2026-09-24 | no file exists: lendability DRAWN from a model of the archive years (`mimic.lend`, PREREG67's `v2`; its level drifts 5–8 points a year on unseen years — when set B decides, also report `Mimic(lend_shift=±0.37)`, ±7 points); the draws LINKED as IBKR's file is — a name's across sessions, all names on a day (PREREG94: `lend_persist` 0.7, `lend_day_lam` 0.15); the fee from IBKR's API daily bars for the names still listed (84% of the live combination's short trades there), PREREG94's F4 fee model otherwise (`fee_version="v4"`, the trailing level of the listed names' real fees; `Mimic.borrow_v2()` = the construction before 2026-10-12). On set A this construction moves growth −0.8 log points a year but deepens the drawdown 15 → 25% with margin calls in 4 of 20 draws: set B's drawdowns and margin calls are pessimistic (PREREG93 / 94) |
| **C — our own archive (live)** | 2026-09-25 → | our archive of IBKR's file (`data/ibkr_borrow/`): the live forward test |
| **D — the forward holdout** | 2026-09-25 → the latest completed session | C's days SIMULATED on our own archive (PREREG106, `periods.get("D")`): never tuned on |

- DECIDE ON THE POOLED HISTORY (user directive 2026-10-14: "Instead of checking for the three different sets
  (A0, A, and B), check for the full dataset (A0+A+B)"): A0 + A + B, 2018-01-02 → 2026-09-24 (`periods.POOLED`).
  Each set is still its own $10,000 account from its first session (a trade decided inside it may exit after its
  end), chained into the pooled numbers: a pooled rate = the years-weighted mean of the sets' log rates
  (`periods.pool_weights`), a pooled drawdown the worst of the sets'. Every set is reported beside — a diagnostic
  (where it earns, where it costs), never a separate gate.
- Calendar years in their parts (`year_parts`): 2021 is "2021 A0" + "2021 A", 2024 "2024 A" + "2024 B"; the chained
  account's years and months compound the parts (`bank.combination`).
- State with every result how much of each set's trading had the data (the coverage per year part); set B's
  lendability is modelled — its drawdowns and margin calls are pessimistic.
- The dip book alone borrows nothing: a long-only study may use its full history; inside the account it is judged
  on the pooled history with the shorts.

- News sets: split with `eval_metrics.split_test_sets` / `by_test_set` (dates in `TEST_SETS`).
  The sets are disjoint.
- **News set 1 coverage — state it with every result.** `bundle` / `events` / `polygon`
  from 2026-06-17; `finnhub` / `google` / `all` from 2026-07-03 only; RSS and press
  wires never. Do NOT use the live `signals` news columns for news set 1 — they span
  six prompt eras and are masked by the news epoch. The history built on
  2026-09-24 ends 2026-09-23: extend it through 09-27 before using the whole set
  (`python -m src.analysis.news_history --acquire finnhub`, `--acquire google`,
  then `--score --loop`).
- **News set 2 accrues forward.** Fewer than ~10 settled days is a sanity read, never
  a verdict; state the settled share. Its news features differ from news set 1's in COVERAGE by design (news set 1's history
  mirrors the old ingestion: pre-fetch names only, Finnhub 60, Google 150) — a model
  trained on news set 1 meets richer digests on news set 2; say so beside any news-set-2 result.
- **In-sample.** A model fitted on any row inside a set is in-sample there:
  train strictly before the set's first day, or walk forward inside it (§5).
  `ml_ohlcv`'s live artifact trains through 2026-06-01, so both sets are out of
  sample for it; the panel-trained stackers and `ml_exit` are not unless refit
  walk-forward.
- **Reading them together.** A change ships when it clears the bar (§5) on news set 1
  and news set 2 does not contradict it. An effect on news set 1 only suggests a
  reconstruction artifact (the Google and Finnhub rebuilds recover ~60% of live's
  articles); on news set 2 only, either new-ingestion value or too few days — say
  which, with the day count.

## 2. The metrics

**Models / rankers** — anything emitting a cross-sectional score (methods, ML
models, the combine, confidence). The MODEL METRIC SET, computed only with
`src/analysis/eval_metrics.py`:

```python
from src.analysis.eval_metrics import by_test_set
res = by_test_set(df, "score", label="fwd_ret_pivot")   # df: signal_date, ticker, score, label
```

1. **IC** — mean per-day Spearman correlation of the score with the signed
   next-H/L-pivot return, over the day's scored cross-section;
2. **t** — its day-clustered t (one observation per signal date);
3. **top 5% / bottom 5% returns**,
4. **top 3% / bottom 3% returns**,
5. **top 1% / bottom 1% returns** — each day the `ceil(p × n)` highest- and
   lowest-scored names and the mean of their signed next-pivot return, averaged
   over days, each tail with its own day-clustered t and split halves.

- The top tail is the long book (positive is good); the bottom tail is the short
  book (NEGATIVE is good — its oriented return is the negation). The day's
  universe mean over the same rows is reported beside the tails as the baseline,
  never as a metric.
- Rank only the rows the model actually scores: a method that abstains with 0.0
  is passed its rows WITH a view, or its zeros become a tied block mid-ranking.
  Tail ties break on the ticker. A day with fewer than 20 scored, labelled rows
  is skipped (`MIN_DAY_ROWS`).
- At 1% a tail is 1–5 names a day (~1 on a single news source): its t is the
  noisiest of the set — read it beside the 3% and 5% tails.
- IC and its t are the number a change must move; the tails say whether the move
  reaches the names a rule trades (the LIVE selection short: the top-1 fresh pick
  per bar; the SHADOW rank rule: the top/bottom 3 per side; the staged cutover
  rules: top/bottom 5%). A U-shaped payoff has a flat IC and good tails, so both
  are always reported.

**The selection objective** — for models trained on it (the next models, user
directive 2026-09-25), reported beside the model metric set, PER SIDE:

```python
from src.analysis.eval_metrics import selection_by_test_set
res = selection_by_test_set(df, "score", "long")   # and "short"; df: EVERY run's rows +
                                                   # run_id, fwd_ret_pivot, bars_ahead
```

Per run the top (long) or bottom (short) name, kept only when its score is also
a new extreme against its own last 30 trading days (the live freshness rule; a
name with under 10 prior scores is kept), one entry per name per day, each
earning its next-pivot return divided by the sessions to the pivot
(`bars_ahead / 13`, floored at one day — `bars_ahead` comes from `label_rows`).
Report the mean return per day with its day-clustered t and halves, entries per
day, the share of days with an entry, and the all-names baseline. The pick is
made on the score alone; a pick without a label stays an unlabeled entry.

**Deciders** — anything binary (gates, exits, sizing tilts, whole-algorithm
A/Bs): mean oriented next-pivot return per decision vs the MATCHED
COUNTERFACTUAL, day-clustered t, net of calibrated costs when the branches trade
different amounts. Kept vs dropped with the side-mix benchmark (`gate_funnel`:
`value` = keep_exc − drop_exc); hold-matched control for exits
(`exit_policy_sim` EXCESS — never raw `mean_ret`); paired same-day arm
difference (`arm_eval`, the Tier-2 walk-forward backtest). A model that serves a
decision is judged twice: its score as a model, the rule built on it as a
decider.

**The live strategy (the $10,000 live account since 2026-10-09: the selection short's
vol, ETF and fall arms + the dip long book)** is a decider whose trades end at
IMPLEMENTABLE exits, not at the pivot. Judge it on TWO metrics, both reported every
time, on the POOLED HISTORY (§1b), with each arm's GROSS win rate beside (user directives
2026-10-04: "growth per year and return per day are the new metrics to follow";
2026-10-07: "Change the $5000+$1000/14days to $10000. We'll use this number from now on.";
2026-10-14: "winrate and return per day are important metrics"):

1. **Return per day** = `(1 + mean net trade)^(1 / mean 24-hour hold) − 1` of the
   whole rule and of each book, with the return per trade beside it (a bootstrap by
   pick day); a book INSIDE the account is judged on its return per day first, the
   account's growth as the guard;
2. **Growth per year** = a **$10,000 account, no deposits** (so money- and time-weighted
   agree), compounding through THE ENGINE that mimics production (`src/backtest/
   mimic.py` + `account.simulate`, runbook `docs/BACKTEST.md`; 20 stratified draws per
   set of IBKR's lendability, house rates and refusals): the live slices (vol 1/20, ETF
   1/16, fall 1/12, dip 1/10, crash 1/4),
   IBKR's house margin with forced buy-backs, the dip book giving way and de-levered,
   IBKR's daily borrow, the FINRA $2,000 minimum, shorts ≤ 100% of the account and ≤ 1%
   of the stock's dollar volume, the ledger's costs, the extended-hours ticks. Report
   the worst drawdown and the margin calls beside it, every calendar year against SPY
   and QQQ and the MONTHS (the share positive, the worst month, the worst 12 months,
   the longest time under water) — `bank.combination`. Differences: `study.diff_ci`
   (21-day block bootstrap of the daily log-return difference);
3. **An arm's place in the bank** (`src/backtest/bank.py`, user directive 2026-10-14:
   "Each individual model needs to give growth on that whole set but doesn't need to
   have growth all the time ... enough trades made so that the results are
   significant, reliable, and transferable"): on the pooled history its own trades —
   at least 100 in at least 30 entry months, the return per day's 95% interval (an
   entry-month bootstrap) above its BAR, its return per day above the bar in both
   chronological halves and with any one calendar year left out, growth at its live
   slice; INSUFFICIENT when only short of trades. THE BAR (user 2026-10-15: "test long
   arms against the market's average daily rise. The idea is to beat the market"): 0
   for a short arm; for a long arm the same exposure's average day over the same span
   — SPY's for a stock arm, the fund it trades for an index arm (a 3x fund's own
   average day: the leverage is never the edge) — `bank.BENCH`, `bank.bench_rpd`. THE
   RE-JUDGING (user 2026-10-15): an arm failing ONLY significance, its return per day
   above its bar, is judged on its longest history, significance clustered by month
   (never trade by trade). Its context cells (SPY trend x volatility
   tercile at the entry, >= 30 trades) say where it earns and where it loses; a filter
   keeping it out of a losing context is justified by >= 30 significantly losing
   removed trades (`bank.filter_test`) and measured out of time (`bank.walk_forward`).
   CONTEXT ACTIVATION (user 2026-10-14: "a model should understand the context and be
   activated if historically it performs well in it"): a model trades only in the
   market cells its walk-forward map switches on (`src/backtest/context.py`, point in
   time; `bank.activation_map` / `schedule` / `activated`; PREREG108's rules) and is
   judged on its ACTIVATED trades — no trades in a context it is off in is no fault;
   a map reaches production only through the pooled add rule
   (`Mimic(activation=<schedule>)`).

The two can disagree (the vol arm's 100% give-back raised growth and lowered return per
day): say so and let the user choose. Every trade is net of the costs production's
ledger charges (`mimic.LedgerCosts`), IBKR's commissions, SEC/TAF on a short sale and
IBKR's borrow fee day by day. Engine and conventions:
`memory/metrics-growth-and-return-per-day.md`. Survivorship-free always (the names
delisted since 2021, every common stock outside the store with the 400-bar floor).
Parameter searches run as the weekly walk-forward (`memory/optuna-weekly-walkforward-rule.md`).

Not used as metrics: AUC, hit %, precision@k (a rank statistic minus magnitude,
or skill-shaped noise), and the LIVE LEDGER's compound NAV (a monitor — ~410 days to
detect 0.10%/day). The REPLAYED account's growth per year above is a metric for the
live strategy.

## 3. The label — the next H/L pivot on 30-minute bars

The signed % move from the row's OWN tick price to the extreme of the NEXT
resolved pivot: zigzag over 30-minute regular-hours bars (swing highs on bar
HIGHS, lows on bar LOWS, confirmed at `pivot_min_move_pct`), search starting at
the first bar that begins at or after the tick (the tick's own bar excluded). It
may resolve later the same session or days out — never a fixed horizon. Report
the definition in force (`pivot_target.pivot_basis()`, e.g. `hl1@30m`) and never
compare levels across definitions.

- **Settled labels**: the panel's `fwd_ret_pivot`
  (`signal_panel.build_panel(horizons=(5,), dedupe="last")`, settled rows only).
- **The unresolved tail too** (the evaluation standard) — `label_rows` from this
  skill folder:

  ```python
  import sys; sys.path.insert(0, ".claude/skills/evaluate")
  from live_labels_intraday import label_rows            # rows: ticker, generated_at, price
  lab = label_rows(df)                                   # bars=None reads the production 30-min cache
  df["y"], df["settled"], df["same_day"] = lab.target_pct, lab.resolved, lab.same_day
  df["bars_ahead"] = lab.bars_ahead                      # bars to the pivot (selection objective)
  ```

  An unresolved row is marked at the LAST VISIBLE CLOSE, never at the running
  leg's extreme (measured: +39% magnitude and 14% sign flips, flattering
  hold-friendly models). Warm the cache first for a large read
  (`python -m src.data.backfill --with-30m --skip-daily`); a ticker with no
  30-minute history has no label.
- **Mechanical check, every run**: no resolved row's pivot bar starts at or
  before its tick.
- **Orientation**: the label is MARKET-signed; for anything position-relative
  (exits, held positions) multiply by the position's direction (+1 long / −1
  short).
- **Provisional labels are for evaluation only** — never a training label or a
  calibration input.

## 4. The population — simulated/panel rows, never the ledger

- Models and signals → the `signals` panel, one row per ticker per day (the
  day's LAST run, `dedupe="last"`); news set 1's news features from
  `news_history.source_feature_frame()`, joined on (run_id, ticker).
- Exits → `ml_exit_dataset.build_exit_dataset` (simulated held positions;
  `entry_directional` marks the real held population) and `exit_policy_sim`.
- Whole-decision policies → `simulated_trades`, `exit_policy_sim`, `policy_eval`.
- Restrict to the Gate-4 tradeable pool (price ≥ $5, 20-day dollar volume ≥ $5M)
  unless the question is about observe-only names.
- Never the trade ledger: a gate's dropped cohort does not exist there, and it
  holds ~11 attributed trades per method against ~9,000 panel views.

## 5. Evidence

- Per-day statistics with a DAY-CLUSTERED t; a statistic pooled across days is
  not the statistic.
- PAIRED contrasts to compare variants: same rows, same days, per-day
  difference, t on the difference.
- Split halves: report both; halves of opposite sign are noise, whatever the t.
- Walk-forward for anything fitted: train only on rows whose label had printed
  before the fold's cutoff (each row's settle date, `end_date_pivot`, is the
  embargo).
- AN ARM'S RELIABILITY (user 2026-10-12, six rules; CLAUDE.md Standing rules): specify
  the mechanism, its context condition and horizon, few parameters at the middle of a
  plateau (`reliability.plateau`); build its CARD (`python -m src.backtest.reliability
  --build`: return per day and gross win rate by year, set and regime, where it loses,
  sensitivity to doubled costs / a one-bar delay / lendability ±0.37); pre-register and
  check it (`python -m src.backtest.prereg <study_dir>`); report the candidates tried
  with `stats.pbo_cscv` and `stats.deflated_sharpe`; a per-trade quality model only where
  the arm has thousands of trades (the dip book); the bundle's overlap and headroom
  (`reliability.bundle`); the live score against the card's reference (the calibration
  record's `arms`).
- The house bar, PRE-REGISTERED before the run: paired t ≥ 2 AND same-sign
  halves, judged on each test set as §1 requires (a strategy: on the pooled history,
  its halves chronological; report each set beside — an effect on set B only may be
  the lendability model's, not the market's: say so).
- The DEPLOY rule is separate from the house bar (user 2026-10-10; the pooled history
  2026-10-14): a variant the house bar does not call proven is still ADDED when its
  POOLED gain — the years-weighted mean of the sets' paired gains — is at least +0.5
  log points a year (no set must gain alone), no ONE calendar year carries it (the
  pooled gain stays above 0 with any year left out: save every draw's daily NAV), no
  more (set, draw) pairs meet a margin call than the live configuration's on the same
  seeds and the pooled worst drawdown is at most 1 point deeper — judged PAIRED on
  common seeds, 20 stratified draws per set (40 in every set within 2 SE of +0.5; 3 SE
  below +0.5 after 5 draws rejects at once), mechanically by
  `add_rule.pooled_verdict`. What it adds must rest on enough trades: an arm
  qualified for the bank, a filter justified by its removed trades (or by risk:
  fewer margin calls, a shallower drawdown — the user's call). Report each set's and
  each year's gain in log points a year (set B's higher growth inflates percentage
  points); PREREG81: rule tweaks keep 9–23% of their in-sample gain out of sample.
- A new ARM for the live combination (user 2026-10-11: arms that cover different
  parts of the market, activated by the market, not by us): report its trades
  per year and per regime (SPY above / below its 200-day, volatility), its return
  per day and per trade WHEN ACTIVE (the long history where data allows), and
  the years it loses — its own rule must keep it out of the markets that hurt it;
  then its bank qualification (`bank.qualify`) and the combination WITH it on the
  pooled history — its years against SPY and QQQ, its months, its coverage of the
  market contexts (`bank.combination`; an arm that earns where the bank is idle or
  loses is worth the most). Never a switch that turns an arm on or off by hand: a
  context the arm loses in goes into its OWN rule, a point-in-time filter its trades
  justify.
- State n, the day count, the settled share and the label definition. Fewer
  than ~10 days is a sanity read.
- Long and short: for a model the top tails ARE its long side and the bottom
  tails its short side; for a decider report ALL / LONG / SHORT.

## 6. Traps

- **Pooling across a boundary** — the 2026-09-28 news switch, an artifact
  retrain, a config flip: split at the boundary; a pooled number measures
  neither treatment.
- **Epoch masks** — a scorer whose output changed is NaN before its epoch; "no
  skill" may be "no data". Check coverage first.
- **`combine_source` resolves per ticker** — filter on the row's own value, never
  group by run.
- **`signal_date` is ET, `generated_at` is UTC** — join through `signals`, never
  by date arithmetic.
- **In-sample fitted layers** — a curve fitted on the window being evaluated
  (rank shaping, a calibration) makes the result in-sample; use the as-of history
  (`shapes_for_date`) or say so.
- **Re-score as production serves** — the same bar and the same features
  available at serving time; check the rank correlation with the persisted live
  column before trusting an offline IC.
- **Build serve-time inputs the way serving sees the store** — a session snapshot
  built at 08:30 has no bar of its own session in the 30-minute grid; a fixture
  or a rebuild after the fact that holds the session's bars hid a bug that left
  every price-dependent deep feature missing on ~90% of live names (fixed
  2026-09-26, `deep_features.RTH.with_session`).
- **Position vs clock bar** — the `ml30` arrays' `bar` is the bar's POSITION in
  its session, live runs are keyed by the CLOCK bar; they differ for a name that
  skipped a bar earlier that session. Compare per-bar scores only where the two
  agree.
- **Absolute IC levels are not comparable across harnesses** — only paired
  contrasts within one.

## 7. Output

Lead with the verdict and the number behind it. Then, for EACH test set:

| model | days | IC | t | top 5% | bot 5% | top 3% | bot 3% | top 1% | bot 1% | baseline |
|---|---|---|---|---|---|---|---|---|---|---|

with each tail's t beside it (for a decider: the counterfactual excess ALL / LONG
/ SHORT). For a change to the LIVE STRATEGY, both metrics on the POOLED HISTORY first
(the decision row), then each set (diagnostic rows), then the calendar years in their
parts and the months:

| rule | history | trades | return/trade | return/day (95%) | gross win | growth/yr (20 draws) | worst drawdown | margin calls |
|---|---|---|---|---|---|---|---|---|

For an ARM, its bank card (`python -m src.backtest.bank`):

| arm | verdict (basis) | trades / months | return/day (95%) | bar | gross win | halves | worst year left out | alone at its slice | earns / loses in |
|---|---|---|---|---|---|---|---|---|---|

For a model trained on the selection objective, add per side:

| model | side | days | entries/day | days with entry | return/day | t | halves | baseline |
|---|---|---|---|---|---|---|---|---| Then the caveats that would change the reading: day count, settled
share, boundaries, news set 1's coverage, the borrow-data set's coverage, in-sample fitting. If a bar was
pre-registered, say whether it was met on each set — if missed, the change does
not ship.
