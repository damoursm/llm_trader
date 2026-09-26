---
name: evaluate
description: House standard for evaluating any model, signal, gate or exit rule in llm_trader — TWO test sets reported separately (set 1 history 2026-06-17..09-27, set 2 live all-source news from 2026-09-28); model metrics = IC to the next H/L pivot, its day-clustered t, and the top/bottom 5%, 3% and 1% next-pivot returns (src/analysis/eval_metrics.py), plus the selection objective (per-side return per day of the top/bottom name per run after the own-history rule) for models trained on it; gates/exits = counterfactual excess return per decision; label = the next H/L pivot on 30-minute bars from the tick, unresolved rows at the last close; simulated/panel rows, never the trade ledger.
argument-hint: "[what to evaluate, e.g. 'the ml_exit upgrade' or 'Gate 1c']"
---

Evaluate: $ARGUMENTS

**The standing rule.** Evaluate on the TWO test sets (§1) with the metric set
(§2), against the next H/L pivot (§3), on simulated/panel rows (§4), long and
short — each test set reported separately, never pooled.

## 1. The two test sets — report each separately, never pool

| set | signal dates | what it is | news features |
|---|---|---|---|
| **Set 1 — history** | 2026-06-17 → 2026-09-27 | OHLCV + news since news collection began (first news pull 2026-06-16 08:00 UTC; first scored run 2026-06-17) | the per-source history rebuilt with today's news code: `news_replay` rows `pool_spec='src:<group>'`, read with `src.analysis.news_history.source_feature_frame()` |
| **Set 2 — live all-source news** | 2026-09-28 → | the live pipeline running the ALL-SOURCE news ingestion (every per-ticker feed asks about every scored name; live since `news_coverage.ALL_SOURCE_SINCE`) | the LIVE values the pipeline persisted — no reconstruction: the `signals` news columns, the archive (`news_articles` + `news_article_feeds` rebuild any single feed's pool) and `sentiment_digests`; the technical-indicator vectors from the live capture, `src.analysis.live_features.load("30m" | "daily")` |

- Split with `eval_metrics.split_test_sets` / `by_test_set` (dates in `TEST_SETS`).
  The sets are disjoint.
- **Set 1 coverage — state it with every result.** `bundle` / `events` / `polygon`
  from 2026-06-17; `finnhub` / `google` / `all` from 2026-07-03 only; RSS and press
  wires never. Do NOT use the live `signals` news columns for set 1 — they span
  six prompt eras and are masked by the news epoch. The history built on
  2026-09-24 ends 2026-09-23: extend it through 09-27 before using the whole set
  (`python -m src.analysis.news_history --acquire finnhub`, `--acquire google`,
  then `--score --loop`).
- **Set 2 accrues forward.** Fewer than ~10 settled days is a sanity read, never
  a verdict; state the settled share. Its news features differ from set 1's in COVERAGE by design (set 1's history
  mirrors the old ingestion: pre-fetch names only, Finnhub 60, Google 150) — a model
  trained on set 1 meets richer digests on set 2; say so beside any set-2 news result.
- **In-sample.** A model fitted on any row inside a set is in-sample there:
  train strictly before the set's first day, or walk forward inside it (§5).
  `ml_ohlcv`'s live artifact trains through 2026-06-01, so both sets are out of
  sample for it; the panel-trained stackers and `ml_exit` are not unless refit
  walk-forward.
- **Reading them together.** A change ships when it clears the bar (§5) on set 1
  and set 2 does not contradict it. An effect on set 1 only suggests a
  reconstruction artifact (the Google and Finnhub rebuilds recover ~60% of live's
  articles); on set 2 only, either new-ingestion value or too few days — say
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

**The live strategy (the selection short, since 2026-09-28)** is a decider whose
trade ends at an IMPLEMENTABLE exit, not at the pivot: judge a change to it by the
return per trade and per day (`(1 + mean)^(1/days) − 1`) of the whole rule —
entry at the pick bar's close (or the delayed entry being tested), cover at half
the 5-session run-up checked at 30-minute closes, else 15 sessions — net of the
REAL costs: the NBBO half-spread at entry and exit (`nbbo_backfill.quote_at`),
IBKR fixed commissions, SEC/TAF on the short sale, and the borrow fee from
`data/ibkr_borrow` at entry (`ibkr_borrow.borrow_at`). Always beside the
volatility-matched control (`sel_models.vol_control`): top-1 picks are the most
volatile names and volatile names drift. Pick parameters out of sample (choose on
one date half, score on the other). Harnesses: scratchpad `short_eval.py`,
`short_exit_search.py`, `short_exit_holds.py`, `short_entry_delay.py`
(`memory/selection-model-exits-2026-09.md`, `memory/sel-short-deploy-2026-09.md`).

Not used as metrics: AUC, hit %, precision@k (a rank statistic minus magnitude,
or skill-shaped noise), and compound NAV (a monitor — ~410 days to detect
0.10%/day).

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
  day's LAST run, `dedupe="last"`); set 1's news features from
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
- The house bar, PRE-REGISTERED before the run: paired t ≥ 2 AND same-sign
  halves, judged on each test set as §1 requires.
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
/ SHORT). For a model trained on the selection objective, add per side:

| model | side | days | entries/day | days with entry | return/day | t | halves | baseline |
|---|---|---|---|---|---|---|---|---| Then the caveats that would change the reading: day count, settled
share, boundaries, set 1's news coverage, in-sample fitting. If a bar was
pre-registered, say whether it was met on each set — if missed, the change does
not ship.
