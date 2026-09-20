# ml_ohlcv as the sole combine — cutover runbook

**Status: STAGED, NOT LIVE.** Every flag below defaults OFF and the code is inert
until they are flipped. `tests/test_rank_entry_rules.py::test_defaults_reproduce_the_shipped_rank_rule`
pins that the selector is byte-identical to the shipped rank rule with the
defaults in force.

> **⚠ ON HOLD since 2026-09-19.** The exit study (`scratchpad/exit_day.py`,
> `verify_drift.py`, report `exit_report.html`) found that the +1.5 to +2.2%
> per decision this cutover was selected on is the pivot label read with
> hindsight, not collectable return: on the same entries every implementable
> exit has a net at or below zero (best −0.01%/entry, none of 45 rules positive;
> next-day close −0.34%),
> and the independent daily-cache check shows no drift after the first session.
> Retrains at 1%/3%/5% and on fixed-horizon targets do not change the verdict.
> The machinery below stays tested and inert; do not flip it on the label
> evidence. Details: `memory/pivot-label-hindsight-2026-09.md`.
>
> **PARTIALLY SUPERSEDED 2026-09-19 23:45 UTC.** The user directed that the
> **1% refit** be deployed instead, and it was: `refit_D30_t1.pkl` is now
> `cache/ml/ml_ohlcv_model.pkl`. That covers step 2.1 (the artifact) and makes
> the 30-minute serving path LIVE, but with a **different model** and, critically,
> **WITHOUT step 2.2** — `PIVOT_MIN_MOVE_PCT` stays **1.0**, because the 1% refit
> carries the same `hl1@30m` basis production already runs. That removes this
> runbook's single largest risk (`ml_exit` and the stackers silently serving a
> 1%-label model under a 5% regime). **Steps 2.2 and 3's entry-rule flags remain
> OFF and unexecuted**; the five staged pieces below are still staged. If the 5%
> model is ever revisited, sections 2.2 onward apply unchanged.
> Deployment record: `memory/ml-ohlcv-30m-rows-deploy-2026-09.md`.

User directive, 2026-09-18: *"Deploy to production the retrained ml_ohlcv model
'Gap cluster ≥ 10×, at a new 30-day high or no history' and have it be the live
combine. We will review and retrain the ML stackers but until then, let's have
the ml_ohlcv 100% of the weight of the live combine and decision making process.
Also, I'm ok with the low volume trading this implies. Deploy the 'Top 5% /
bottom 5%, at a new 30-day high or no history' version as shadow."*

---

## 1. What is staged

| piece | file | flag |
|---|---|---|
| 30-minute serving path for `ml_ohlcv` | `src/signals/ml_model.py` (`_score_30m`), `src/analysis/ml_dataset.py` (`hlc_30m`, `ticker_feature_frame(hlc=)`) | none — dispatches on the ARTIFACT's own `feature_bars` stamp |
| ml_ohlcv as the whole combine | `src/signals/aggregator.py` | `ML_OHLCV_SOLE_COMBINE` |
| Gap-cluster entry rule (**6x**, was 10x) | `src/signals/rank_entry.py` | `RANK_ENTRY_RULE=gap_cluster`, `RANK_ENTRY_SCORE=ml_ohlcv`, `RANK_ENTRY_GAP_MULTIPLE` |
| Own-history freshness filter | `src/signals/score_history.py` | `ENABLE_RANK_ENTRY_FRESHNESS` |
| Union routing: both rules decide every run | `rank_entry.active_rules` | `RANK_ENTRY_UNION`, `RANK_ENTRY_RULE_B` |
| A/B split as the alternative to union | `rank_entry.resolve_arm` | `RANK_ENTRY_AB_SHARE` |
| Shadow arm (records the rule that did NOT decide) | `src/pipeline.py`, `rank_entry.build_shadow_recommendations` | `ENABLE_RANK_ENTRY_SHADOW` |

Nothing else changes: the other 27 methods and both stackers keep scoring and
persisting, so the panel, the IC machinery and every calibration keep accruing.
They leave the COMBINE, not the system, and one flag puts them back.

## 2. Prerequisites, in order

**2.1 The artifact.** Install the measured artifact, do not refit first:

```bash
cp "<scratchpad>/models5/refit_E30.pkl" cache/ml/ml_ohlcv_model.pkl     # back up the old one first
```

`refit_E30.pkl` is stamped `pivot_basis=hl5@30m`, `feature_bars=30m`,
`train_max_date=2026-06-01`. That early cutoff is deliberate and is the reason to
ship **this** artifact rather than a fresh one: every number behind the decision
(+3.82%/entry for the gap rule, +2.57% for the top-5% rule, per-day IC +0.229)
was produced by this exact model on 2026-06-17..09-04, which is strictly out of
sample for it. A refit would be a different model with no measured record.

⚠ **Install it WITH step 2.2, not before.** With `PIVOT_MIN_MOVE_PCT` still 1.0
the basis guard does return `BASIS_STALE`, so the artifact cannot serve wrong
scores — but the method then reads 0.0 for every ticker, and `ml_ohlcv` is one
of the 31 signed stacker features, so the live combine would spend the gap
consuming a zeroed column. The artifact and the threshold move in the same
restart.

**2.2 The label threshold.** `.env`: `PIVOT_MIN_MOVE_PCT=5.0`.

This is the one-way part. Read it before flipping:

- `pivot_basis()` becomes `hl5@30m`; the installed artifact starts scoring.
- The 9 leg features change meaning, because they share the zigzag threshold.
- **`ml_exit` and both stackers carry NO `pivot_basis` stamp**, so they keep
  serving models fitted to the 1% label with no warning. The stackers are out of
  the decision path in sole-combine mode, but **`ml_exit` still closes every
  position**. That is the largest un-addressed risk in this cutover and it is
  deliberate: adding an abstain check there would disable the exit timer
  entirely, which is worse. Retrain `ml_exit` on the 5% label as the first
  follow-up.
- Every evaluation surface switches to the 5% label, which is the standing
  directive, and unsettled rows rise because a 5% swing resolves less often
  inside `MAX_PIVOT_BARS_30M`.

**2.3 Deep store.** `python -m src.data.intraday_store --extend` (~3,400 Polygon
calls; never beside an RTH tick). Serving reads the deep store plus the tick
cache tail, so a stale store costs depth, not correctness.

## 3. The cutover

Run it in a quiet window. The overnight venue trades Sun–Thu nights, so the long
window is **Friday after 19:50 ET until Sunday 20:00 ET**.

```bash
# 1. stop the scheduler — the SUPERVISE parent first, or it respawns the child
#    (find both: Get-CimInstance Win32_Process | ? {$_.CommandLine -match 'main.py'})
# 2. back up, then install
cp cache/ml/ml_ohlcv_model.pkl cache/ml/ml_ohlcv_model.hl1.$(date +%F).bak.pkl
cp "<scratchpad>/models5/refit_E30.pkl" cache/ml/ml_ohlcv_model.pkl
# 3. .env
#    PIVOT_MIN_MOVE_PCT=5.0
#    ML_OHLCV_SOLE_COMBINE=true
#    RANK_ENTRY_RULE=gap_cluster
#    RANK_ENTRY_GAP_MULTIPLE=6.0      # 6x, not 10x — see 'the multiple' below
#    RANK_ENTRY_SCORE=ml_ohlcv
#    ENABLE_RANK_ENTRY_FRESHNESS=true
#    RANK_ENTRY_UNION=true            # BOTH rules decide every run (overrides the A/B share)
#    RANK_ENTRY_RULE_B=top_pct
#    RANK_ENTRY_SHADOW_RULE=auto      # records the arm that did not decide
#    ENABLE_RANK_ENTRY_SHADOW=true
#    (BROKER_MAX_POSITIONS=200 is already set in .env)
# 4. register the scorer epoch for ml_ohlcv at the ACTUAL restart instant
#    (src/signals/method_epochs.py) — its output changes categorically: new
#    label, new feature resolution, intraday refresh instead of one value a day
# 5. restart
powershell -File scripts/restart_all.ps1
```

## 4. Verify on the first run

| check | expected |
|---|---|
| `[ml_ohlcv]` warnings | no `BASIS_STALE` line |
| `signals.ml_ohlcv` | non-zero for most of the pool; spread roughly ±0.24 |
| `signals.combine_source` | `ml_ohlcv` on every row |
| `[rank_entry] rule=gap_cluster score=ml_ohlcv` | present; ~6 picks a run at 6x (it was 0–2 at 10x) |
| `[freshness] ml_ohlcv: standings for N tickers` | N in the thousands |
| `[rank_entry:shadow]` | picks recorded, ~4 a side |
| tick duration | +10–20 s (30-minute features cost ~120 ms/ticker, pooled) |

**The gap multiple is 6x, not the 10x this rule was first measured at** (user
directive 2026-09-18). At 10x the long side took 272 entries at +1.86 (t 2.4)
while 6x takes 1,435 at +2.27 (t 8.4) — it dominates on every axis for longs;
shorts post a higher +4.04 at 10x but on 405 entries at t 1.8 against 1,869 at
+2.08 (t 6.0), i.e. 2.4x the total return at a fixed size. The 10x headline was
also inflated by the truncated freshness window (+5.54 over the first 30 study
days against +2.43 once the look-back was full), and **10x is the one cell that
failed the out-of-sample holdout**: +0.34 (t 0.4) against 6x's +0.62 (t 3.5).

Volume consequence, checked before shipping: the gap arm goes from 10.4 to
**50.8 entries/day** and the union with the top-5% arm from 54.2 to **67.0/day**,
which at the ledger's realised 2.30-day mean hold is **~154 concurrent positions
against the cap of 200**. It also makes the two A/B arms comparable in size,
50.8 against 52.2 a day, which is what the arm comparison needed.

**Expect the freshness filter to be inert for the first trading day.** A
standing is only counted from the ml_ohlcv scorer epoch, because the score's
SCALE moves with each retrain (the daily 1% artifact averages |0.0375| against
this one's |0.097|) and a window straddling that boundary would read nearly
every name as a new high — the filter would look like it was working while
selecting nothing. So on day one no name has a standing, every name is kept, and
the live rule is the bare gap cluster, which measured +2.13%/entry against
+3.82% filtered. It self-heals once a day of post-epoch runs has accrued. Cutting
over on a Friday evening means the first standings exist for Tuesday.

### ⚠ The A/B changes the book's SIZE, and the position cap binds

The two rules differ about fivefold in volume. Measured entries per day: gap
cluster **10.4**, top-5% cut **52.2**. A 50/50 split is therefore **~31 entries a
day against ~10 on the gap cluster alone**, roughly tripling the book.

At a median hold of about 3 days that implies **~93 concurrent positions**
against `BROKER_MAX_POSITIONS` = **100**, so the book would run at the cap. That
matters beyond a refused order: **the SIM ledger has no position cap at all**, so
once the broker cap binds the two diverge silently and every sim-vs-broker
comparison on this era becomes unreadable.

**Decided 2026-09-18 (user): share 0.5, and `BROKER_MAX_POSITIONS` raised
100 → 200.** Checked before raising rather than after: at ~991k CAD equity and a
2,000 CAD base order, 200 positions is ~40% of equity and ~61% at the 1.5x
sizing ceiling, against a `broker_max_gross_exposure_pct` of 1.0. So COUNT is
genuinely the binding constraint, notional is not, and the real risk stop is
untouched. The raise is already in `.env`; it takes effect at the cutover
restart. The cap also no longer binds quietly: `reconcile.sync` now logs **CRITICAL once per sync** the
first time the POSITION cap refuses an entry, naming the divergence explicitly.
The gross-exposure cap keeps its INFO line, because that one is the intended risk
stop doing its job. Per-trade evidence was already there (`broker_status`
= `SKIPPED_CAP: …`); what was missing was anyone noticing.

So the expected operating state is: the book settles around ~93 positions with
headroom to 200, and if it ever reaches the cap you get a CRITICAL line the same
tick. If that starts firing, the choice is then an informed one — raise again,
drop the share to 0.25, or read only the arm-segmented panel. The arm is stamped per run (`rank-gap10f` /
`rank-top5f`) on `runs.llm_synthesis_provider`, every recommendation and every
new trade, so the PANEL comparison stays valid even while the ledger does not.

### Dry run, 2026-09-18 10:00 ET (production code, real cross-section)

Run `2026-09-18_140023`, 353 tradeable names, scored through `_score_30m` with
`refit_E30.pkl`: **353/353 `OK`** in 36 s, range −0.229..+0.259, mean |score|
0.105, which matches the 0.097 the study measured. Arm A selected **1 name** (MU,
BUY, +0.259, top of 353); arm B selected **36** (18 a side, 5% of 353).
Standings: 0 tickers, exactly as the epoch guard predicts today.

A/B assignment over all **2,079** real run ids in the database splits **51.8%**
to arm B (z +1.64, not distinguishable from 50/50), it is stable under replay,
and over the last 200 runs the shadow never once duplicated the live arm.

## 5. Rollback

`ML_OHLCV_SOLE_COMBINE=false`, `RANK_ENTRY_RULE=topk`,
`RANK_ENTRY_SCORE=combined`, `ENABLE_RANK_ENTRY_FRESHNESS=false`, restore the
backed-up artifact, `PIVOT_MIN_MOVE_PCT=1.0`, restart. The panel keeps every
row from the interval, stamped `combine_source='ml_ohlcv'`, so the era stays
segmentable.

## 6. Known gaps, in priority order

1. **`ml_exit` serves a 1%-label model** against 5%-label entries (see 2.2).
2. **No production 30-minute TRAINING path.** `train_and_persist_pivot` still
   builds one row per ticker-day from the daily parquet, so this artifact can
   only be reproduced by the scratchpad script that built it. The weekly retrain
   is held (`ENABLE_EOD_ML_TRAIN=false`), so nothing silently overwrites it with
   a daily-row model — but a retrain cannot be run until this is built.
3. **The top-10 truncation** in `pipeline.py` cuts recommendations before the
   gates. The shipped rules cannot reach it; a `CRITICAL` line now fires if a
   future rule does.
4. **The confidence divisor is static** (`ml_ohlcv_raw_confidence_scale`).
   `ml_scale` solves its value from ML-source panel rows, of which there are none
   on day one. Revisit once the era has history.
5. **The rules are not held out.** They were chosen on 2026-06-17..09-04. A
   split-half test put the cell rank correlation between the two halves at
   +0.405, and cells of the gap cluster's kind gave back about two thirds of
   their first-half edge.

## 7. Pre-cutover verification — done, and what it found

**Passed.** The 30-minute serving path scores 353/353 names OK at 121 ms each with
the right distribution; live serving reproduces the offline evaluation vector to
4.8e-5, which is exactly its own 4-dp rounding; the basis guard abstains on a
mismatch; the A/B hash splits 51.8% over all 2,079 real run ids, is stable under
replay, and never lets the shadow duplicate the live arm; defaults leave the
selector byte-identical.

**Found and FIXED — the absolute twin.** The first version of the sole-combine
branch also wrote the ml_ohlcv score into `_abs_buy`/`_abs_sell`, i.e. into
`combined_score_abs`. That column is what `ml_exit` reads as `ex_combine`, and it
exists to be basis-invariant so the exit model's feature does not step when the
combine changes shape. |combined_score_abs| averages **0.20–0.25** against this
score's **0.07–0.11**, so it would have shrunk that feature ~2.5x on the model
that closes EVERY position, silently. The branch now leaves the absolute camps
alone, as the ml-stacker arm already does. Pinned by
`test_sole_combine_leaves_the_absolute_twin_alone`.

**Found and NOT fixable by code — the A/B cannot settle the rule question.**
Per-day net over the study window, on the 56 days both rules fired:

| | net/entry | sd across days | days fired |
|---|---|---|---|
| arm A, gap cluster | +3.82% | **12.88** | 56 / 65 |
| arm B, top 5% | +2.57% | 2.50 | 65 / 65 |
| paired difference | +1.62 | 12.52 | t **+0.97** |

Arm A's per-day variance is five times arm B's because it takes so few names
that one position sets the day. Reaching |t| = 2 at the observed +1.62 gap needs
**~237 trading days (~11 months)**, and ~950 days if the true gap is half of it.
Splitting the book 50/50 halves each arm's sample again.

**Resolved 2026-09-18 by UNION ROUTING** (user): both rules decide every run and
the book trades the union, so each accrues at 100% of its natural rate instead of
the 50% a split would give it. Measured over 1,773 runs: the gap cluster takes
0.68 names/run, the top-5% cut 8.20, and **59.5%** of the cluster's picks are
also in the top 5% — it searches the top 20% of ranks, so its boundary can fall
outside a 5% cut. The union is 8.48/run, **only 3.4% more than arm B alone**, so
the book's size is set by the top-5% rule either way: ~54 entries/day and ~162
concurrent positions against the raised cap of 200.

The 237-day figure above already assumed both rules running at full rate, so
union does not shorten it — it prevents the split from making it twice as long.
The horizon is driven by arm A's per-day variance, which is a property of taking
0.68 names a run. So: read the traded book as an **operational** test, and take
the per-rule comparison from `engine_recommendations`, where each rule is written
as its own LIVE arm every run (`rank_gap_cluster`, `rank_top_pct`) and joined to
the panel. `python -m src.analysis.engine_eval` on the disagreement subset is the
right surface, never the traded P&L — a name both rules picked is ONE position
and the ledger cannot attribute it.

What IS answerable sooner is each rule against zero, which is the question that
actually gates keeping them: arm A already carries t +2.2 over 65 days.

## 8. Still unverified before the cutover

1. **No end-to-end pipeline run with the flags ON.** The pieces are unit-tested
   and the selector is proven on real data, but `build_signals` has never
   executed with `ML_OHLCV_SOLE_COMBINE=true`. Worth one `python main.py` in a
   sandbox before the restart.
2. **Confidence and sizing under the new combine.** `raw_confidence` becomes
   |ml_ohlcv| / 0.20, and confidence drives the sizing ramp. The resulting
   distribution of position sizes has not been looked at.
3. **Gate attrition.** The study applied no gates; live picks still face Gates
   2/3/4/4b/5. Gate 5 (anti-chase) is the one to watch, because a top-of-run
   scorer correlates with a recent run-up by construction.
4. **`direction` on gap-cluster picks.** The rule bypasses the band, so a pick
   can carry `direction=NEUTRAL` while its action is BUY. Nothing is known to
   break, but no gate has been exercised with that combination.
5. **Exit behaviour against a 5% target.** Entries move to a 5% pivot while the
   exits keep their tuning; if `ml_exit` closes before the pivot resolves, the
   realised return is not the measured one.
6. **First-tick cost after the threshold flip**, when the pivot memo rescans.
