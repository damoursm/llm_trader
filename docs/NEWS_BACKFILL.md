# News backfill — regenerating historical news features

Runbook for `src/analysis/news_replay.py`. Written 2026-09-07/08 after the
pilot; **the full backfill RAN overnight 2026-09-11 → 09-12** (record below).
Read this before running it again — several of the constraints below are not
obvious and were each found the expensive way.

## THE DATASET (2026-09-12) — always filter on both clauses

```sql
SELECT ... FROM news_replay
WHERE pool_spec = 'union168h' AND replayed_at > '2026-09-12'
```

**26,388 rows / 70 ticks / 70 days (2026-06-17 → 2026-09-11) / 2,691 tickers**,
one tick per day (the last of each, matching `build_panel(dedupe="last")`), no
duplicate ticker-days. Scored under the current news logic: prompt
v7dir-2026-09-10, catalyst cap `analyst`@0.03, logprob expectation ON,
passing-mention 0.75, cluster mode hybrid@0.25, engine `local/qwen3:8b`,
`--union-hours 168`.

`pool_spec` alone is **not** enough. The table is keyed
`(run_id, ticker, pool_spec)` so pool shapes can coexist and be compared, and
1,031 rows from the pilot and the sweep (`faithful`, `union24h`, `union72h`) are
deliberately kept — including 168h rows scored by the PREVIOUS prompt, which is
what `replayed_at` separates. Query on one clause and you mix prompt eras or pool
shapes inside a single column.

Feature coverage (share of rows with a non-zero value): `news` 45.5%,
`news_unpriced` 45.4%, `news_bull_fresh` 31.4%, `catalyst_tilt` 25.6%,
`sent_velocity` 22.8%, `news_quiet` 13.1%, `news_bear_fresh` 12.8%,
`news_shock` 13.7% (1.2% before the baseline-basis fix below).

## What it does

Rebuilds a past tick's article pool from the archive, re-runs **today's** news
pipeline over it, and writes the whole news family to the `news_replay` table:
`news`, `news_raw_score`, `sent_velocity`, `news_shock`, `news_quiet`,
`news_bull_fresh`, `news_bear_fresh`, `catalyst_tilt`, `news_unpriced`,
`news_unpriced_all`, plus `news_catalyst`, `news_recency_mass`,
`news_article_count` — every method on the `method_epochs.NEWS_FAMILY` list.

"Replay" means the same thing it means in `src/analysis/replay.py`: **what
today's code would have produced given the data available then** — not a
re-enactment of the code that ran that day. That is what makes a replayed value
comparable with a live one instead of with a retired scorer.

## Why it exists

The news-family scorer epoch masks every news column before it, so the stacker training set has ~0% news coverage until forward history
accrues. The archive (`cache/news_*.json`, 1,132 hourly bundles back to
2026-06-16) holds the raw articles, so the history is regenerable — imperfectly.

## Fidelity — what you are getting, measured

418 paired ticker-runs, engine and prompt held constant (local Qwen v6 against
its own live verdicts), tune/test split, held-out estimate:

| | value |
|---|---|
| Spearman vs the live verdict | **0.35** |
| sign agreement | **73%** |
| magnitude vs live | **0.40x** |
| abstention | 21% vs live ~10% |

The 0.40x decomposes into two independent ~0.6 factors that compound and are
**not** tunable: raw CONVICTION 0.62 (flat across every pool shape — the archive
is aggregator/listicle content the v6 prompt tiers down into the LEAN band) and
evidence MASS 0.56-0.63 (a fresh article weighs ~1.0, a 4-day-old one ~0.025).
The articles are both older and weaker. No window setting escapes it; a sweep of
0h / 24h / 72h / 168h confirmed it.

**Counter-intuitive result worth remembering: the MORE faithful the
reconstruction, the WORSE it matches live.** A strictly faithful pool (only what
the tick held) scores rho 0.24; the deliberately unfaithful 7-day union scores
0.35, because it substitutes *something* for the ~55% of each digest that came
from feeds nobody persisted. Faithfulness and fidelity point in opposite
directions here.

### What this means for training

The 0.40x scale gap **stops mattering** once the news features are consumed as
within-run ranks (`stacker_news_basis="rank"`, the default since 2026-09-07): a
rank is invariant to scale, so a backfilled row and a live row are on one basis.
That was the whole reason for the rank transform, and it is what makes the
backfill usable at all.

What remains is the ORDER disagreement — rho 0.35 — which is genuine noise, not a
shift. Judge whether it helps the way anything else is judged here: retrain with
and without the backfilled rows and read the paired walk-forward pivot IC at the
house bar (`.claude/skills/evaluate`).

## 2026-09-08 UPDATE — what the backfill now regenerates

Three production changes landed after the fidelity numbers below were measured,
and the backfill inherits all three because a replay runs TODAY's code:

1. ~~**Source-tier filter** (`enable_source_tier_filter`, ON)~~ — **REVERTED
   2026-09-09.** It was shipped on one day of n=60 and failed its paired
   re-test: -0.047 over 12 days, -0.026 over 50, t -0.55 with opposite-sign
   halves. Its epoch boundary was removed with it. A backfill run today does
   NOT inherit it, which voids the "shared generator" argument below.
2. **Qwen at 100% of live sentiment** (`sentiment_local_share=1.0`), DeepSeek
   demoted to shadow. Train and serve now share one generator, which is what the
   backfill needs to be worth running at all.
3. **Rank basis for the news stacker features** (`stacker_news_basis="rank"`),
   which is what makes the backfill's ~0.40x magnitude gap irrelevant — a rank
   is scale-invariant. Only the ORDER disagreement (rho ~0.35) survives.

**That consequence no longer holds — read this before running anything.**

## 2026-09-11 UPDATE — the fidelity numbers describe a scorer that no longer exists

The 09-08 argument for running this was "the backfill and live production share a
generator by construction". That was true for about two days. Since it was
written the sentiment scorer has changed **categorically four times**, and the
news-family epoch has moved with each one:

| change | effect on the verdict |
|---|---|
| passing-mention abstention (09-10 14:30) | ~12% of digests now abstain before scoring |
| logprob expectation (09-10 19:50) | the value is the expectation over the score's digit tokens, not the argmax — 19 values on a 0.05 grid became ~75 off it |
| prompt **v7dir** (09-11 02:40) | the model names the direction first; **11% of signs move**, mean abs score 0.367 -> 0.422 |
| catalyst-class cap (09-11 06:30) | `analyst` verdicts clipped to +/-0.10 |
| source-tier filter REVERTED (09-09) | the digest pool is different again |

A replay runs TODAY's code, so every replayed row is internally consistent — that
part still works. What is void is the **measurement**: rho 0.35 / 73% sign / 0.40x
magnitude were measured with v6 replay against v6 live verdicts, and none of those
three generators is the current one. **Re-measure before quoting those numbers or
deciding on them** (`--fidelity`, plus the `--paired-only` run over v6-era ticks
that was already outstanding for the filter and never happened).

Note also what re-measuring now compares: today's replay (v7dir + logprob + cap)
against live verdicts scored by v4/v5/v6, because everything before 09-11 06:30
predates the current scorer. That is a scorer-era difference on top of the pool
difference, and it will read as replay infidelity if it is not accounted for.

**And the training question is NOT settled** — CLAUDE.md states the rule flatly:
replayed values must never become stacker features, because a linear stacker would
learn a weight on a diluted column and apply it to sharper live values. The
counter-argument in "What this means for training" below is that a within-run RANK
is scale-invariant, so the 0.40x gap cannot reach the model. Both are partly right,
and the honest resolution is that the rank transform kills the MAGNITUDE gap but
not the other two:

* **order noise** — rho ~0.35 is the disagreement that survives a rank transform,
  and it is most of the column;
* **population** — a rank is invariant to scale but NOT to which tickers are in
  the cross-section, and replayed rows abstain at 21% against live's ~10%, so the
  ranked set itself differs.

So: no replayed row reaches a stacker until the paired retrain test this document
already specifies has been run and read at the house bar. That test needs live
post-epoch rows to compare against, and there are currently one day's worth.

## 2026-09-12 — four fidelity gaps closed before the run

An audit of "does the replay actually mimic live" found four places it did not.
All are fixed and pinned by `tests/test_news_replay.py`; none was visible in the
output, which is why they are listed here rather than left to be discovered.

1. **The clustering corpus was never installed.** `news_cluster_mode` is hybrid,
   and the content-aware modes merge on IDF from the TICK'S WHOLE POOL.
   `cluster_articles` degrades to the bare TIME partition when `corpus_size() ==
   0` — safe live, but here it meant a backfill labelled "today's code" while
   clustering under the rule today's code replaced. Now installed once per tick
   from `pool`, mirroring `build_signals`. NEVER per ticker from its own digest:
   within-digest IDF zeroes exactly the shared-story terms the merge depends on.

2. **Two weighted methods were not regenerated at all.** `news_quiet` (0.10) and
   `news_bull_fresh` (0.08) shipped after this module was written, so a backfill
   would have produced a news family missing its two newest voters — invisibly,
   since the columns would simply not exist rather than read wrong.

3. **The provider shortcut could not fire.** Live skips the LLM when enough fresh
   articles carry provider sentiment, and that path serves **~42 of ~130 scored
   tickers per tick (~32%)**. It is bypassed for a forced engine — correct for a
   hold review, which forces one to RE-JUDGE, wrong for the replay, which forces
   one to PIN. Worse, the replay never mapped Polygon's `insights` onto
   `provider_insights` even though `get_ticker_news_history` returns them, so the
   shortcut was unreachable regardless. Both fixed: `analyse_sentiment` takes
   `allow_provider`, and the replay reconstructs the raw labels exactly as
   `provider_news.fetch_polygon_news` does. This also removes calls the live tick
   never made — a third of the estimated cost.

4. **Replayed digests polluted the live store.** `_record_digest` ran
   unconditionally, and `news_replay` keeps no `digest_id`, so a full run would
   have written ~34k orphaned rows into `sentiment_digests` — unjoinable to the
   rows they came from, stamped with whatever run id happened to be current, and
   counted against the 180-day retention. `store_digest=False` for the replay.

**Still NOT mimicked, and structural:** the pool itself. Only the
yfinance/NewsAPI leg was ever archived (~600 of a ~2,400-article tick pool);
`news_articles` captures the whole pool only from 2026-09-11 16:50 onward.

## Prerequisites (both already in place — do not remove them)

1. **`as_of`** (`sentiment._clock`). The scorer measured article age against
   `datetime.now()`, so any tick older than a week discarded its whole digest and
   abstained: 144 tickers "scored" in 13 seconds, silently. `as_of` is threaded
   through `_recency_weight`, `attention_mass`, `recent_cluster`,
   `analyse_sentiment` (freshness cut, top-20 sort **and** the digest age labels
   the model reads), `_provider_sentiment_score`, `compute_sentiment_velocity`
   and `compute_news_priced_in`. `as_of=None` is the live path — pinned by
   `tests/test_sentiment_as_of.py`.
2. **`digest_articles`** — one shared definition of "the digest" (fresh,
   recency-sorted, capped at 20). Without it the replay recorded the pre-cut
   relevant set (493 articles for NDAQ) while the live shadow row records the
   post-cut list, and every digest-size comparison was apples-to-oranges.

## Running it

```bash
# PRE-FLIGHT: two ticks at the era extremes, small slices (~2 min of GPU)
python -m src.analysis.news_replay --only-runs 2026-06-17_235026 --limit-tickers 10 --redo --union-hours 168
python -m src.analysis.news_replay --only-runs 2026-07-30_033029 --limit-tickers 10 --redo --union-hours 168

# FULL RUN: 65 ticks (the LAST tick of each day), ~19.3k calls
python -m src.analysis.news_replay --days 120 --union-hours 168

# VALIDATION
python -m src.analysis.news_replay --fidelity --pool-spec union168h
```

Frozen config: `--union-hours 168`, engine `local`, one tick per day, tag
re-confirmation on, Polygon as-of on. **Do not re-tune it** without a fresh
held-out split — six configurations have already been measured on the same rows.

Resumable: a finished `(run_id, pool_spec)` is skipped, so a kill costs at most
one tick. `--budget-seconds` bounds a single tick. `--paired-only` replays just
the tickers that have a live verdict to compare against (for fidelity work, not
for a real backfill).

## SCHEDULING — this is the constraint that bites

**Run it in the weekend window: Friday 19:50 ET -> Sunday 20:00 ET.**

The local box serves Qwen at 100% of live sentiment *on the critical path*
(`sentiment_local_share=1.0`). Ticks were measured at 1137-1640 s against the
**2700 s tick watchdog**, and the backfill needs ~12-22 h of the same GPU. Adding
that contention on a weekday is how a tick reaches the watchdog and the scheduler
gets killed. The only genuinely idle window is Friday after the last extended
tick (no overnight session Friday or Saturday night) until the overnight venue
reopens Sunday 20:00 ET. Nothing else contends: the Saturday 08:00 ET ML retrain
slot is HELD, and the 02:00 ET auto-refactor is CPU, not GPU.

Timing: ~2.2 s/call uncontended (~14 h), ~3.5 s/call contended (~22 h). The
per-tick `catalyst_tilt` refit adds ~66 s x 65 ticks (~72 min, ~5%) — memoisable
per signal_date if that ever matters.

## No time travel — the three holes, each closed and tested

1. **Pool**: only bundles STAMPED at or before the tick, only articles PUBLISHED
   at or before it, Polygon queried with `published_utc.lte`. The guard runs on
   the RAW load, before the recency window trims anything — a guard placed after
   a filter can never fire.
2. **Derived layers**: the tick runs inside `analysis_asof(signal_date)`, and the
   attention baseline is computed by `baselines_as_of` because
   `news_shock.load_attention_baselines` hardcodes `CURRENT_DATE` and
   `analysis_asof` does not reach its own SQL. That baseline reads only signal
   dates strictly BEFORE the tick, on either basis, and `--repair-shock`
   restates the same rule because it sees the whole window at once.
3. **Prices**: `replay.visible_history` gives the bars the tick could see,
   including the forming-bar rule (the signal-date bar exists only for a tick at
   or after the 16:00 ET close).

Pinned by `tests/test_news_replay.py` (18 probes).

## Known limits — expected, not bugs

* **RSS wires and per-ticker Google News are unrecoverable.** Both were
  age-capped at 24h at fetch time and never persisted. They were ~55% of a live
  digest. `macro_news_*.json` exists but is empty.
* **`news_shock` needs the POOL-CONSISTENT baseline** (fixed 2026-09-12). It is
  a RATIO of today's evidence mass to the ticker's own normal, so both sides must
  come from the same generator. The first full backfill divided a REPLAYED
  numerator by the LIVE `signals` series, and a replayed pool carries ~0.19x the
  live mass (older articles, and only the yfinance/NewsAPI leg survives) — so the
  ratio sat structurally below 1, `clip(log2(ratio)/3, 0, 1)` returned exactly 0,
  and the feature read non-zero on **1.2%** of rows against **13.7%** on the
  consistent basis. `baselines_as_of(signal_date, pool_spec)` now builds the
  baseline from `news_replay`'s own stored mass for that pool shape, with NO
  fallback to the live series (a baseline that mixes generators is the defect,
  not the cure) — so a pool with too little replayed history abstains instead.
  It still needs `news_shock_min_days` (5) covered days, so early ticks read 0
  by design.
* **Re-run the repair after any backfill**: the baseline a tick reads depends on
  which OTHER ticks have already been replayed, which makes a resumable run
  order-dependent. `--repair-shock` recomputes the column once the whole window
  exists (maximal coverage, order-independent) from the `news` and
  `news_recency_mass` already on each row — no LLM call, and it never writes to
  `signals`:

  ```bash
  python -m src.analysis.news_replay --repair-shock --pool-spec union168h       --since 2026-09-12            # dry run: reports rows / changed / coverage
  python -m src.analysis.news_replay --repair-shock --pool-spec union168h       --since 2026-09-12 --apply    # write the values back
  ```
* **`catalyst_tilt` is 0 for early ticks** — under `analysis_asof` there is no
  event history to fit, and the calibration fail-softs to abstain.
* **Do not re-fetch the missing feeds.** Anything fetched today returns articles
  the tick could not have seen. Polygon is the one exception, because the live
  pipeline made that exact call with the same endpoint, ordering and limit.

## NOT BUILT: the consumption path

### The trap to design around: one generator per signal_date

The rank transform normalises ACROSS days. It does **not** protect against mixing
generators WITHIN a day. If one `signal_date` held live values for some tickers
and replayed values for others, ranking that cross-section would compare a live
score against a replayed one — and since replayed values run ~0.40x live scale,
the replayed tickers would rank systematically lower. That looks like signal and
is pure provenance artifact.

Today the split is clean by accident, not by construction: the replay covers the
epoch-masked days before the current news-family epoch and live covers after. The consumption path
must ENFORCE it — one generator per `signal_date`, and a loud refusal (not a
silent pass) on a mixed day.

Rows land in `news_replay` and **nothing reads them**. A test pins that this
module can never write `signals_replay`. To use them for training you still need
to merge them into the panel at dataset-build time (as the `signals_replay`
restore already does for OHLCV methods), and let restored cells SKIP the scorer
epoch mask — that is the whole point, since a value produced by today's scorer is
comparable with today's. Stamp provenance so a training row's news origin stays
queryable.

## Decision record

`memory/news-backfill-fidelity-2026-09.md` (the measurements and why the faithful
reconstruction lost), `memory/engine-scale-mismatch-2026-09.md` (why ranks),
`memory/stacker-feature-set-2026-09.md` (the retrain checklist).
