---
name: evaluate
description: House standard for evaluating any signal, model, gate or exit rule in llm_trader — H/L pivot targets with last-price resolution for unresolved pivots, simulated trades, split long and short.
argument-hint: "[what to evaluate, e.g. 'the ml_exit upgrade' or 'Gate 1c']"
---

Evaluate: $ARGUMENTS

**The standing rule.** When evaluating, use the H/L pivot targets and, if recent
pivots are unresolved, use the last price as the resolution (next pivot price).
Also, use the simulated trades and evaluate for both long and short.

Everything below is how to satisfy that rule correctly in this codebase.

## 1. The label — H/L pivots, last price for the unresolved tail

The target is the signed % move from a bar's close to the **next pivot extreme**
on the H/L basis (swing highs on each bar's HIGH, lows on its LOW, confirmed by
`pivot_min_move_pct`). Settled pivots come from the panel
(`fwd_ret_pivot` / `_pivot_targets`). Recent bars have no confirmed pivot yet —
do **not** drop them, and do not wait for settlement: the running leg extreme,
extended by the latest price, stands in as the resolution.

```python
import sys; sys.path.insert(0, ".claude/skills/evaluate")
from live_labels import label_frame

lab = label_frame(sorted(df["ticker"].unique()))       # omit asof = current prices
df["y_mkt"]   = [lab.get((t, d), (float("nan"), False))[0] for t, d in zip(df["ticker"], df["signal_date"])]
df["settled"] = [lab.get((t, d), (float("nan"), False))[1] for t, d in zip(df["ticker"], df["signal_date"])]
```

Report the settled/provisional split alongside every result. Pass `asof="YYYY-MM-DD"`
for a point-in-time view. This proxy is validated (within-day rank corr with the
settled label 0.963–1.000, sign agreement ~100%).

**Orientation — the double sign flip.** The label above is MARKET-signed. For
anything position-relative (exits, held positions, a directional call), multiply
by the position's direction sign (+1 long / −1 short): positive then means "the
move went the position's way". Getting this wrong silently inverts shorts.

**Hard guardrail:** provisional labels are for EVALUATION and MONITORING ONLY.
Never let them reach a calibration, a training label, or anything that decides
live trades — they keep extending until confirmation, so fitting on them fits a
moving target. Training always uses settled labels plus each row's own settle
date (`end_date_pivot` / `end_date_pv`) as the walk-forward embargo.

## 2. The population — simulated trades, not the ledger

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

## 3. Split long and short — always

Report ALL, LONG and SHORT for every headline number. The sides behave
differently in this system (the funnel repeatedly measures the SELL side as the
healthier one), and a pooled number can hide one side degrading. When the pooled
statistic mixes sign-flipped sides, say so — the pooled oriented IC is a
different quantity from the per-side raw-frame IC, and confusing the two has
already produced one false "worse" conclusion.

Also split by side any time you change selection or sizing: a change can be
side-neutral on scale and still bias side COMPOSITION (that is what the Gate 1c
side floor exists to fix).

## 4. Statistics that count as evidence here

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

## 5. Traps that have produced wrong answers here

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

## 6. Output

Lead with the verdict and the number that supports it. Give the per-side table,
then the caveats that would change the reading (day count, provisional share,
era mixing, selection). If a pre-registered bar was set, say explicitly whether
it was met — and if it was missed, say the result does not ship.
