---
name: evaluate
description: House standard for evaluating any signal, model, gate or exit rule in llm_trader — the next H/L pivot as the label, found on 30-minute bars from the tick's own time and price so it may resolve later the SAME session, unresolved pivots marked at the last visible close, simulated trades, split long and short; headline metrics = per-day pivot IC for rankers, counterfactual excess return per decision for gates/exits.
argument-hint: "[what to evaluate, e.g. 'the ml_exit upgrade' or 'Gate 1c']"
---

Evaluate: $ARGUMENTS

**The standing rule.** When evaluating, use the H/L pivot targets. The label is
ALWAYS the next H/L pivot — it may confirm later the SAME day as the tick (but
only strictly AFTER the tick) or many days later, and it is never truncated to a
fixed horizon or to the day. If the pivot has not resolved yet, use the LAST
CLOSING PRICE as the resolution. Also, use the simulated trades and evaluate for
both long and short.

Everything below is how to satisfy that rule correctly in this codebase.

## 1. The label — the next H/L pivot on 30-MINUTE bars, from the tick, last CLOSE for the unresolved tail

The target is the signed % move from the row's **own tick price** to the **next
pivot extreme** on the H/L basis (swing highs on each bar's HIGH, lows on its LOW,
confirmed by `pivot_min_move_pct`), with the zigzag run over **30-minute
regular-hours bars** and the search starting at the first bar whose START is at
or after the tick's timestamp. The bar containing the tick is excluded (it cannot
be split). The next pivot may therefore come **later the same session** or weeks
out — the horizon is whatever the pivot takes, never a fixed number of days.
Unresolved rows are **marked at the LAST VISIBLE CLOSE**, never the running leg's
extreme.

```python
import sys; sys.path.insert(0, ".claude/skills/evaluate")
from live_labels_intraday import label_rows          # rows: ticker / generated_at / price
lab = label_rows(df, bars30m)                        # bars30m: {ticker: 30-min OHLCV, naive-UTC index, RTH only}
df["y_mkt"], df["settled"], df["same_day"] = lab.target_pct, lab.resolved, lab.same_day
```

**This is the PRODUCTION label — the only one** (2026-09-14; daily retired 2026-09-16): the
skill's `live_labels_intraday.py` is a thin wrapper over
`src/analysis/pivot_target.intraday_pivot_targets` — the same function the
signals panel (`fwd_ret_pivot`), the sim/directional panels
(`simulated_trades._pivot_fwd_for_row`), the exit dataset, the tracker's live
targets and every dashboard surface read through `pivot_rows.pivot_fwd_row`, and
that trains `ml_ohlcv` (`session_close_labels` over the deep store
`cache/ml/bars30m_deep`, each deep row anchored at its session close).
`bars=None` reads the production 30-minute cache (`cache/ohlcv_30m/`, Polygon
aggregates, RTH-filtered, capped at `intraday_30m_max_bars` = 260 sessions — the
deep 2021→ history is `cache/ml/bars30m_deep/<TK>.pkl`, never the tick cache); warm it for the panel universe with
`python -m src.data.backfill --with-30m --skip-daily` before a large read, since
a tick only refreshes the names it touches. Pass `bars=` only to evaluate on a
scratch fetch; build such a frame with `.to_numpy()` columns — constructing it
from JSON-indexed Series against a new DatetimeIndex silently yields all-NaN
bars, and the labeler refuses such a series rather than scoring it as "no
pivots". Where the skill and the panel differ is only the UNRESOLVED tail: the
panel writes settled rows only (a training label never sees a provisional
value); the skill marks that tail at the last visible close for evaluation.

**Why daily bars were wrong for this.** With the zigzag on daily H/L and the
anchor at the day's close, the first candidate bar after any anchor is
*tomorrow*: a swing later the same session was not excluded by a rule, it was
unrepresentable. Whether a model's FEATURES are daily or intraday does not
change the label — it is the next pivot in the subsequent bars either way.
Report the same-session share and the median bars-to-pivot beside every result.

**Which marks, which threshold.** The 30-minute marks basis and the
confirmation threshold are settings (`pivot_label_basis` hl|close,
`pivot_min_move_pct`), the labeler follows them, and every trained artifact is
stamped with the fingerprint in force (`pivot_target.pivot_basis()`, `hl1@30m`).
**Decided 2026-09-16 (user): H/L marks on 30-minute bars for ALL labels —
ml_ohlcv, the entry stackers, ml_exit — and the daily H/L label is
decommissioned** (there is no fallback; a ticker with no 30-minute history has
no label). The 2026-09-15 five-label comparison
(`memory/pivot-label-verdict-2026-09.md`) is the record behind the threshold
question: at 1% the 30-minute H/L label's median move is 1.3% and a third of it
is the intra-bar extreme; at 2–2.5% it matches the old daily label's swing and
carries more power. Too many pivots is a threshold question — raise
`pivot_min_move_pct` — never a reason to go back to daily bars. Report the
label definition in force beside every number; do not compare levels across
definitions — only paired contrasts within one.

**Re-scoring `ml_ohlcv` offline.** Production scores an intraday tick on the
PREVIOUS session's completed bar with the six cross-sectional ranks absent. An
offline score must reproduce that (previous row, `_XRANK_SOURCES` NaN) — checked
by its rank correlation with the persisted `ml_ohlcv` column (0.996; the same-day
row scores −0.006). Same-day features report an IC the served model never had
(+0.17–0.27 vs +0.05 measured 2026-09-15).

**Mechanical check, every run.** Assert that no resolved row's pivot bar starts
at or before its tick; the harness must refuse to publish otherwise.

Do **not** use the running leg's extreme as the provisional resolution. That is
the best price the open leg happened to reach, which nothing guarantees was
capturable, and it systematically inflates the unresolved tail — measured
2026-09-13 on 5,101 panel rows: 12.6% unresolved, on which the extreme basis ran
**39% larger in magnitude** than the close (close = 72% of extreme) with **13.9%
sign flips**. It flatters hold-friendly models most: the frozen `ml_exit`
artifact's IC fell +0.128 → +0.072 (out of significance) on the switch, because a
hold-conviction model graded on the best price a held position reached is graded
on the one number an open position cannot bank. History:
`memory/decile-ledger-validity-2026-09.md`.

**Orientation — the double sign flip.** The label above is MARKET-signed. For
anything position-relative (exits, held positions, a directional call), multiply
by the position's direction sign (+1 long / −1 short): positive then means "the
move went the position's way". Getting this wrong silently inverts shorts.

**Hard guardrail:** provisional labels are for EVALUATION and MONITORING ONLY.
Never let them reach a calibration, a training label, or anything that decides
live trades — they keep extending until confirmation, so fitting on them fits a
moving target. Training always uses settled labels plus each row's own settle
date (`end_date_pivot` / `end_date_pv`) as the walk-forward embargo.

## 2. The metric — two functionals of one quantity (decided 2026-08-29)

Every layer of the system is one of two shapes, and each shape has ONE headline
metric. Both are functionals of the same quantity — the oriented pivot move — so
the layers' objectives telescope instead of fighting.

**RANKERS** (anything emitting a cross-sectional score: methods, ML models, the
combine, confidence) → **mean per-day Spearman IC vs the signed pivot label**,
with its day-clustered t. This is the number a model or method change must move.

- Not AUC: on a binarized label AUC IS a rank statistic minus the label's
  magnitude — measured day-level corr with IC +0.905 (1,988 method-days) — and
  it replicated worse in every split tested. It adds nothing and discards the
  magnitudes that sizing and rank-shaping consume.
- Not precision@k / hit%: the weakest future-money predictors measured, and hit%
  can replicate WITHOUT predicting money — side composition and drift persist
  (the ~48.6% market-relative baseline problem), which is skill-shaped noise.
- Measured 2026-08-29 (49 days, 61 methods, Gate-4 pool, live labels).
  Method-ranking self-replication across contiguous / odd-even / well-covered
  splits: IC .48/.65/.59 · tail spread .47/.64/.52 · AUC .37/.60/.58 ·
  hit .38/.22/.63 · mean oriented return .32/.48/.37. IC is the only candidate
  stable across all three designs, and its half-1 reading predicted half-2
  realized return nearly as well as the return-unit metrics predicted
  themselves.

**DECIDERS** (anything binary: gates, exits, sizing tilts, the LLM layer,
follow-through selection, whole-algorithm A/Bs) → **mean oriented pivot return
per decision vs the matched counterfactual**, day-clustered t, NET of the
calibrated costs whenever the compared branches trade different amounts or
sessions.

- The counterfactual is not optional: kept-vs-dropped with the side-mix
  benchmark (`gate_funnel`'s `value` = keep_exc − drop_exc), hold-matched
  control (`exit_policy_sim`'s EXCESS), paired same-day arm difference
  (`arm_eval`, the Tier-2 wf backtest). Raw oriented means are incomparable
  across side mixes, and an exit rule's raw return mostly measures its holding
  period.
- IC is a category error here: a binary decision has no cross-section, so its
  "IC" degenerates to a rescaled mean difference (the funnel's gate-IC column
  reads ±0.01 noise while the excess column carries the same information in
  %-per-decision units that ADD UP and compare directly to the cost stack).

A model is evaluated as a ranker even when it serves a decision: ml_exit's
conviction → ranker IC on the oriented remaining move; the CLOSE RULE built on
it → decider excess. When one number must summarize the whole algorithm, it is
the decider metric on the end-to-end book vs its counterfactual; compound NAV is
the monitor, not the target (at ~1.0%/day NAV sd, detecting a 0.10%/day
improvement needs ~410 days — nothing is decidable there).

Demoted to diagnostics, never optimization targets: decile/payoff curves (the
SHAPE input to rank_shaping — a U or hump is invisible to IC and must be fixed
by transforming the score, after which IC applies again), Brier/calibration (a
repair step via isotonic, needed because conviction feeds sizing), fixed-horizon
ICs (monitoring + holding-period machinery), gross win rate (the reporting
convention).

## 3. The population — simulated trades, not the ledger

Use the simulated/panel surfaces, because the real trade ledger only contains
what the gates let through (selection bias) and is far smaller:

- **Entry / signal quality** → the `signals` panel (`signal_panel.build_panel`);
  one row per (ticker, day) with every method score.
- **Exit rules** → `ml_exit_dataset.build_exit_dataset` (simulated held positions,
  one row per position-day, with `entry_directional` marking the real held
  population) and `exit_panel` for the live held book's activation events.
- **Whole-decision policies** → `simulated_trades`, `exit_policy_sim` (its headline
  is EXCESS vs a hold-matched control, never raw `mean_ret`), `policy_eval`.

Restrict to the **Gate-4 tradeable population** (price ≥ $5, 20-day dollar volume
≥ $5M from the OHLCV cache) unless the question is explicitly about observe-only
names — otherwise the result is dominated by names the system would never trade.

The real ledger is structurally unusable for both metric families: a gate's
dropped cohort never exists there (no counterfactual), and its power is hopeless
for rankers — measured 2026-08-29: median 9,176 panel views per method vs median
11 attributed real trades per method, and at per-trade gross return sd 7.04%
even a 0.5%/trade edge needs ~800 closed trades pooled. The ledger's roles are
execution-cost calibration (the constants inside the decider metric's NET
adjustment), broker reconciliation, and the NAV monitor — never the evaluation
sample.

## 4. Split long and short — always

Report ALL, LONG and SHORT for every headline number. The sides behave
differently in this system (the funnel repeatedly measures the SELL side as the
healthier one), and a pooled number can hide one side degrading. When the pooled
statistic mixes sign-flipped sides, say so — the pooled oriented IC is a
different quantity from the per-side raw-frame IC, and confusing the two has
already produced one false "worse" conclusion.

Also split by side any time you change selection or sizing: a change can be
side-neutral on scale and still bias side COMPOSITION (that is what the Gate 1c
side floor exists to fix).

## 5. Statistics that count as evidence here

- **Per-day Spearman IC** with a **day-clustered t** (mean over days ÷ SE of the
  daily series). Pooled IC across days is not the statistic — the system ranks
  within a day.
- **PAIRED contrasts** when comparing two variants: same rows, same folds, per-day
  difference, t on the difference. Unpaired means with no t are not evidence.
- **Split-half sign check.** Report both halves. A result whose halves flip sign
  is noise, whatever the full-window t says — this has killed several otherwise
  significant-looking findings (every volatility-conditioning attempt so far).
- **Walk-forward** for anything model-shaped: train only on rows whose label had
  printed before the fold cutoff (the settle-date embargo).
- **Pre-register the bar** before running when the result may ship: the house
  default is paired t ≥ 2 AND same-sign halves.
- State n, day count, and the settled/provisional mix. Thin day counts (< ~10)
  are a sanity read, not a verdict.

## 6. Traps that have produced wrong answers here

- `combine_source` resolves **per ticker** — an "ML run" contains weighted
  fail-soft rows. Filter on the row's own `combine_source`, never group by run.
- **Epoch masks**: a scorer whose output changed is NaN before its epoch. A column
  reading "no skill" may just be masked — check coverage before concluding.
- **Era boundaries**: an artifact retrain or config flip splits the data into two
  treatments. Split the analysis at the latest boundary; pooling measures neither.
- `signal_date` is ET, `generated_at` is UTC — join through the `signals` table,
  never by naive date arithmetic.
- Any curve fitted on the same window you are evaluating (rank shaping, a
  calibration) makes the result in-sample — use the as-of history
  (`shapes_for_date`) or say plainly that the number is not a skill estimate.
- Absolute IC levels are **not** comparable across different harnesses; paired
  contrasts within one harness are. Don't compare across scripts.

## 7. Output

Lead with the verdict and the number that supports it. Give the per-side table,
then the caveats that would change the reading (day count, provisional share,
era mixing, selection). If a pre-registered bar was set, say explicitly whether
it was met — and if it was missed, say the result does not ship.
