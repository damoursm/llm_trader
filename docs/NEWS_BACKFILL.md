# News backfill — regenerating historical news features

Runbook for `src/analysis/news_replay.py`. Written 2026-09-07/08 after the
pilot; **the full backfill RAN overnight 2026-09-11 → 09-12** (record below).
Read this before running it again — several of the constraints below are not
obvious and were each found the expensive way.

## ONE-YEAR BACKFILL — Finnhub + events at 08:30 ET, the liquid universe (2026-09-25)

`src/analysis/news_finnhub_backfill.py`. Model features, not a fidelity study:
every session 2025-10-06 → 2026-09-24, every name the `sel30` arrays call
tradeable that day (~1,985/day, 2,492 names), each name's OWN feed as of 08:30
ET, scored alone through the current news family. Rows: `news_replay`,
`pool_spec` `pre:finnhub` / `pre:events`, run ids `pre-<date>-b<band>`
(5 bands of 500 names, most liquid first), `replay_version`
`finnhub-preopen-v1` / `events-preopen-v1`. Research only — nothing reads them.

```bash
python -m src.analysis.news_finnhub_backfill --plan                    # once (done 2026-09-25)
# three long-running processes, each resumable and safe to kill/restart:
python -m src.analysis.news_finnhub_backfill --score --source events --window any
python -m src.analysis.news_finnhub_backfill --acquire --window weekend   # Sat 01:00 → Sun 19:55 ET
python -m src.analysis.news_finnhub_backfill --score --source finnhub --window any
# progress
python -m src.analysis.news_finnhub_backfill --status --source events
python -m src.analysis.news_finnhub_backfill --status --source finnhub   # includes the pull frontier
```

Logs: `logs/finnhub_backfill_{acquire,score-events,score-finnhub}.log` (the
module's own lines + warnings) and `logs/finnhub_backfill_<mode>.out` (stderr).
Launched detached (WMI `Win32_Process.Create`, BelowNormal priority) so they
outlive the session that started them; they do NOT survive a reboot — relaunch
the same commands, everything resumes.

**The run in progress (from 2026-09-25) goes EVENTS FIRST** (user directive: the
models retrain once the events data is ready). Only the events scorer and the
Finnhub pull run; the Finnhub SCORER is started by a detached watcher,
`cache/news_hist/finnhub_pre/watch_events_then_finnhub.ps1` (log
`logs/finnhub_backfill_watch.log`), once the events scorer's `.out` ends with its
final `{"stored"...}` summary — and it relaunches the events scorer (up to 5
times) if that process dies before finishing. The watcher does not survive a
reboot either; relaunch it with the processes.

**`catalyst_tilt` is 0 on every row dated before 2026-06-16.** Its calibration
fits the live news-event dataset, which starts with news collection, so for an
earlier as-of it has no events and abstains; the scorer's log shows it as
`[catalyst_tilt] calibration failed (method abstains): 'DataFrame' object has no
attribute 'news'` once per day-band (117 by 2026-03-25). Expected, not a defect —
but a model trained on these rows sees that column switch on in June.

Constraints, each found the hard way:

* **Finnhub's history is a ROLLING ~360-day window** (first served day
  2025-09-30 on 2026-09-25). The pull goes oldest week first; a week not pulled
  before it rolls off is gone for good.
* **One key, two users.** The live refresher (50/min) and the pull (55/min)
  cannot share the 60/min free tier, so the pull runs only in the weekend window
  AND while `cache/finnhub_refresher_state.json` reports idle.
* **Never score a Finnhub day before the pull's frontier passes it**
  (`progress.json` `week_done`). The scorer waits on it; a hand-run
  `score_day_band` would store the day-band as COMPLETE with most names missing,
  and the resume logic would never revisit it.
* **The LLM is the whole cost.** Measured on the first events day-bands (489
  names): 346-347 LLM calls, ~820 s of verdicts, 11 s of feature pass — the
  local server's ~0.45 calls/s is the ceiling. Event articles are re-stamped
  daily, so the backfill keys its verdict cache on the RENDERED prompt
  (`_prompt_key`) and reuses a verdict whenever a name's digest reads the same as
  the day before: 37% of band-0 digests repeat day to day, and the third day-band
  took 236 calls instead of ~346 (-32%). The cache is in memory, so a restart
  costs one day of reuse. Estimated whole run: events ~200k calls, Finnhub
  ~330k — ~2.5-3 weeks of wall clock running continuously between live ticks,
  the top-500 band of each source first (~2 days for events band 0).
* **The tick-phase gate reads the scheduler log incrementally** (`LivePhase`):
  `news_replay.live_tick_phase` reads a fixed 3 MB tail and an RTH tick logs more
  than that at DEBUG, so mid-tick it answers "no tick" (found 2026-09-25 — that
  defect still affects `news_history --score --loop` and `news_replay --tick-aware`).

## PER-SOURCE HISTORY — every feed, one feature group each (2026-09-23)

`src/analysis/news_history.py`. The June–September news history rebuilt from
EVERY recoverable feed, each feed scored ALONE through the current news family,
so a model (or an evaluation) can tell which source reads better and which one
matches live. Rows: `news_replay`, `pool_spec = 'src:<group>'`,
`replay_version = 'news-sources-v1'`, groups `events`, `polygon`, `bundle`,
`finnhub`, `google`, `all`. Research only — nothing reads them.

```bash
# the three jobs (resumable; each can be killed and restarted at any time)
python -m src.analysis.news_history --acquire finnhub      # ~1.7k requests, ~40 min
python -m src.analysis.news_history --acquire google       # ~10-14k requests, ~8 h (30/min)
python -m src.analysis.news_history --score --loop         # scores each group as its legs land
# progress
python -m src.analysis.news_history --status
# LLM-free article-level fidelity on the archive era (after those weeks are fetched)
python -m src.analysis.news_history --calibrate --only-runs <archive-era runs>
# when every group is stored: news_shock on the order-independent baseline, then the report
python -m src.analysis.news_history --repair-shock --apply
python -m src.analysis.news_history --report
```

Logs: `logs/news_history_{finnhub,google,score}.log`. Raw provider answers:
`cache/news_hist/raw/{finnhub,google}/<TICKER>/` (one JSON per week, plus
per-day / per-run re-asks), per-run leg counts `cache/news_hist/runs/<run>.json`.

**What each group is.** `bundle` = the hourly yfinance + NewsAPI file the tick
used (exact). `polygon` = live's market-wide call with `published_utc.lte`
(exact, but for late-ingested articles). `events` = 8-K (deep store SEC filings,
accepted by the tick, live's `_build_article`), analyst / EPS / short interest /
ticker events (the day's cached BUILT articles — every tick of that ET day read
that file), Quiver contracts / lobbying / dark pool (the day's cached raw
payloads, re-built by live's own builders with that day as `date.today()`,
cache-only). `finnhub` / `google` = re-asked from the providers' history with
live's rules. `all` = every leg in live's merge order (bundle, 8-K, analyst,
ticker events, EPS, short, Polygon, Finnhub, Google, Quiver), URL-deduped.

**How the providers are asked — measured, not assumed:**
* Finnhub filters by UTC DATE and returns only the NEWEST ~250 items of a window
  (AAPL, 9 days: 249 items, the oldest 3 days gone). Live's `from=D-3, to=D` (D
  = the ET date) therefore never saw items after 20:00 ET of D, and the rule
  reproduces that. One request per (ticker, ISO week); a week at the cap is
  re-asked per ET day with live's exact request.
* Google News date-bounded search is relevance-THINNED, not capped: a 3-day
  window returned 100 items for "apple" stock in June (55 on 06-18) where a
  9-day window returned 68 (16 on 06-18). So the returned count is not a
  saturation signal. One request per (ticker, week, query); a query returning 30+
  items is re-asked per run with the 3-day window the pilot validated. Undated
  entries are dropped (live stamps them with fetch time).
* **Coverage is live's, and live's is narrow.** Step 1 fetches on the universe
  BEFORE the smart-money / macro-discovery / cointegration-peer additions, which
  more than double it (median 128 pre-fetch names of 388 scored,
  `signals.universe_source`). Google, Finnhub, the 8-K scan, Polygon's universe
  filter and the Quiver builders ran only on that set, and within it Finnhub on
  the FIRST 60 and Google on the first 150 of live's Step-0 order (watchlist,
  sector ETFs, trending, commodities, factor ETFs, then discovered names —
  rebuilt with the discovered names alphabetical). Before 2026-07-03 there is no
  `universe_source`, so the `finnhub` / `google` / `all` groups are not built for
  those 10 runs. Found by `--calibrate`: with every scored name asked, only 61%
  of the Polygon as-of set and 28% of the 8-Ks were in live's pool; restricted
  to the pre-fetch universe, 95% and 100%.

**Sharing the box with live.** A live tick takes ~35 min overnight but holds the
news feeds only in Step 1 (~3 min) and the LLM only while it scores sentiment
(~6 min, between `Signal weights [` and `rank pool:` in the scheduler log). The
fetchers pause during the first, the scorer during the second
(`news_replay.live_tick_phase`); all jobs run at below-normal priority. Google
stops after repeated non-200 answers (with a 30-min cool-off before each retry)
and holds off 30 min whenever the LIVE Google leg logs a non-200 tally or an
empty 20+-ticker sweep — Google carries ~37% of live digest articles, so the
history must never cost the live leg its quota.

**Order.** Both fetchers take the ARCHIVE-ERA weeks first (from 2026-09-11, where
`--calibrate` can check each rebuilt leg against what live actually held), then
the rest oldest-first. The scorer takes runs oldest-first; `news_shock` depends
on which earlier runs exist, so run `--repair-shock` per group at the end.

**Throughput.** Verdicts are pre-filled on the LLM server's two slots and the
features computed single-threaded (`analysis_asof` is thread-local — a worker
thread would compute the derived family with NO cutoff): ~1.8 s per verdict.

**A group is stored whole or not at all.** An engine failure returns 0.0, the
same value as an abstention, so any failed call leaves the group undone for the
next pass.

**PRE-REGISTERED READ (written 2026-09-23, before any per-source row was scored).**
Quality, per group: per-day Spearman IC of `news` vs the signed pivot label over
rows with a view, day-clustered t, split halves; the same against the 5-session
close (the realizable drift — the pivot label reads with hindsight). Source vs
source: the paired per-day IC difference on rows where BOTH hold a view
(`--report` → `paired_pivot`, per test set). House bar: t >= 2 AND same-sign
halves, on BOTH labels, for anything to count as "better". Prior: the pooled
`news` method measured about -0.03 IC over the same months, so no single source
is expected to clear the bar alone. Fidelity: each group vs live `news` on runs
since 2026-09-12 (rank correlation, sign, direction counting no-view), and
`--calibrate`'s article recall for the two re-asked feeds (the 2026-09-23 pilot:
Google 61%, Finnhub 67%). A source counts as "matching live" only through the
first; the second explains why.

**RESULTS (2026-09-24, the complete build).** 80 runs (06-17..09-23) for
`bundle`/`events`/`polygon` (30,126 rows each), 70 runs (from 07-03) for
`finnhub`/`google`/`all` (27,154 rows each); 12,645 Google and 1,653 Finnhub
requests, every one answered; 6 unreadable model answers stored as 0.0 (as live
records them). `news_shock` repaired per group afterwards.

| group | views | IC pivot (t) | halves | IC 5-session (t) | halves |
|---|---|---|---|---|---|
| events | 15.2% | -0.023 (-1.20) | -0.008 / -0.038 | **-0.085 (-4.46)** | -0.125 / -0.047 |
| polygon | 11.2% | -0.002 (-0.10) | -0.001 / -0.003 | **-0.070 (-4.35)** | -0.121 / -0.020 |
| bundle | 26.3% | +0.028 (+1.72) | +0.041 / +0.015 | +0.002 (+0.16) | -0.017 / +0.021 |
| finnhub | 11.5% | **+0.052 (+2.06)** | +0.088 / +0.015 | +0.043 (+1.72) | +0.073 / +0.014 |
| google | 22.1% | **+0.044 (+2.77)** | +0.049 / +0.040 | -0.013 (-0.79) | -0.030 / +0.002 |
| all | 35.7% | +0.018 (+1.40) | +0.007 / +0.028 | -0.034 (-2.72) | -0.071 / +0.002 |

Against the pre-registered bar (t >= 2 AND same-sign halves on BOTH labels) no
single source clears it, as expected; Finnhub is closest (pivot t 2.06, 5-session
t 1.72, same-sign halves on both). The source-vs-source bar IS cleared: on the
same names, `events` reads worse than `bundle` (pivot -0.065 t -2.04 / 5d -0.089
t -2.94), `finnhub` (-0.173 t -3.49 / -0.152 t -2.81) and `google` (-0.097 t -2.84
/ -0.103 t -2.88), halves same-sign throughout. The three article feeds (bundle,
Finnhub, Google) are indistinguishable from one another (paired |t| < 1.4). The
pooled `all` is DILUTED by the two structured/provider sources: +0.018 on the
pivot, -0.034 (t -2.72) at 5 sessions.

POST-HOC (not pre-registered — leads, not results): Polygon's negative is its
PROVIDER-LABEL path (61% of its views; 5-session IC -0.091, t -4.32; the same
articles LLM-scored read -0.014, t -0.45) — the path live takes for ~a third of
scored names (`enable_provider_sentiment`). The events negative is the EPS
surprise (5d IC -0.097, t -3.82; pivot -0.062, t -2.22) and analyst-rating items
(5d -0.149, t -4.05; pivot -0.074, t -2.04) — the analyst reading agrees with the
live `analyst` catalyst cap, and says the SIGN, not just the size, is contrary.

Since 2026-09-24 `--report` reports the house MODEL METRIC SET per test set (IC to the
next H/L pivot, its t, top/bottom 5%, 3% and 1% returns — `src/analysis/eval_metrics.py`,
evaluate skill §1/§2); the table above predates that switch.

Coverage by month is flat (Google 21% / 21% / 25% of names with a view, Jul-Sep),
so Google's thinning with age shows in article counts, not in view rates.

Fidelity to live, runs since 2026-09-12 (live = the current scorer): `all` vs live
`news` rho **+0.725**, sign 93.8% where both have a view, direction counting
no-view 73.9%, live views kept 87.6% — the best pre-archive reconstruction yet
(the `hist` pilot +0.53, bundle+Polygon +0.42) against live's own repeatability
(+0.875 / 95.4%). Single sources against the POOLED live verdict are not a
like-for-like fidelity measure (sign agreement where both have a view: Polygon
91.5%, events 79.8%, Finnhub 75.9%, bundle 73.6%, Google 73.6%); the per-source
fidelity is the article-level `--calibrate` above.

**Live defect found while building (not fixed):** all Quiver gov-contract
articles share one URL, as do lobbying and ticker-event articles, and
`pipeline._dedupe_by_url` keeps the first — so ~1 of ~1,400–3,500 contract
articles per tick reaches the pool. The `events` and `all` groups mirror it.

## AFTER A NEWS-SCORER CHANGE — re-score the archive (2026-09-23)

A categorical news-scorer change moves the news-family epoch, and the epoch
masks every earlier row — six resets between 09-04 and 09-12 left the stackers
~8 trading days of news history. Since 2026-09-11 ~16:50 UTC that is no longer
forced: re-score the archived runs under the new logic and the panel restores
them instead of masking them.

```bash
# 1. certify (no model call): does the archive reproduce each run's live digests?
python -m src.analysis.news_replay --certify --days 14
# 2. re-score the last run of each day since the archive began, in an idle window
python -m src.analysis.news_replay --source archive --days 14 --tick-aware
# 3. compare re-scores with live on runs AFTER the new epoch (live = current code there)
python -m src.analysis.news_replay --compare archive --only-runs R1,R2
```

**What makes it faithful.** For every ticker the live scorer scored, the input is
the EXACT digest it read — `sentiment_digests` (from 2026-09-08) holds its
articles and its text, and the prompt carries that text verbatim
(`sentiment._digest_text`'s override, taken only when the new selection yields
the same `news_digest_id`). The archived pool (`news_articles`, a run's pool =
`first_seen_at <= run <= last_seen_at`, Polygon insights re-attached by URL)
supplies the derived family's inputs. A ticker live read NO digest for
(abstained, found nothing, took the provider shortcut) keeps live's own verdict:
the pool cannot say whether the run's true pool held a digest for it, and scoring
its guess invented 59 views live never had — keeping live's verdict took the
re-score's views that match live from 84.6% to 94.6% (direction agreement
counting "no view", 93.3% → 97.4%). Measured on four post-epoch runs (09-15 → 09-18):

| | measured |
|---|---|
| live digests with their exact input | **100%** (356/356); 355 selected identically under today's code |
| pool rebuild alone reproducing a digest | 37–48% (first-sighting titles/tags; the archive cannot say whether an article sat in a given run's pool) |
| re-score vs live, raw verdict on live digests | rho **+0.87–0.90**, sign **95–97%**, within 0.02 **~51%**, exact ~11% |
| **live vs live** (853 digests re-scored live > 3.5 h apart) | rho **+0.875**, sign **95.4%**, within 0.02 **52.5%**, exact 26.8% |
| two re-scores of the same text | rho +0.97, sign 99.3%, exact 77.9% |
| attention mass / article count vs live | rho +0.97 / +0.97 |
| derived family vs live | shock +0.93, unpriced +0.87, bull_fresh +0.76, catalyst_tilt +0.69, bear_fresh +0.56, quiet +0.44 (rare) |

**A re-score agrees with live exactly as well as live agrees with itself.** The
local model is not deterministic under the live server's batched load, so a live
score is one draw; a re-score is another draw on the identical input. Magnitudes
run ~1.1x live.

**The restore** (`news_replay.restore_news_rescored`, `enable_news_rescore_restore`
ON, called by `build_panel` before the epoch mask): only `RESTORABLE_POOL_SPECS`
(`archive`) rows, only those stamped with the news epoch IN FORCE (`news_epoch`),
only onto pre-epoch rows, joined run-exact. Restored cells escape the panel mask;
the `news_rescored` flag lets `ml_stacker.add_news_features` spare them too. A
later scorer change retires every earlier re-score automatically (its stamp no
longer matches) — run the re-score again under the new logic. Reconstructions
(`faithful`, `union*h`, `hist-*`) can never reach the panel.

**Before the archive: the historical-provider pilot (2026-09-23) — measured, not
worth a June–September rebuild.** `--source hist` adds Finnhub's history (free
tier: one rolling year, 250 per call) and Google News date-bounded search
(`after:`/`before:`) to the bundle + Polygon legs, each with its fetcher's live
rules. On the same four post-archive ticks, against live:

| rebuild | digest match | `news` rho | sign | magnitude | mass rho |
|---|---|---|---|---|---|
| archive (exact re-score) | 100% | +0.864 | 97% | 1.10x | +0.974 |
| **hist: + Finnhub + Google** | 12% | **+0.530** | 89% | **0.98x** | +0.852 |
| old: bundle + Polygon | 8% | +0.420 | 85% | 0.81x | +0.707 |

Article recall of the historical fetches vs the live pool: Google 61% by URL
(64% by title) while returning ~60% more articles, Finnhub 67%; no throttling
over ~1,100 Google requests. **And Google's date-bounded search thins with age**:
the same 15 large-cap queries returned 13.7 in-window items each for a June tick
against 25.0 for a September one, and recall fell 71% → 51% between a 5-day-old
and an 8-day-old tick. So a June–September rebuild would land BELOW the pilot's
+0.53 — still a different quantity from a live score (live vs live is +0.875),
and the old reconstruction at ~+0.33–0.42 already measured worthless to the
stackers and harmful to `ml_exit` (`memory/ml-retrain-replay-news-2026-09.md`). Live's
Google (150) and Finnhub (60) ticker caps follow a per-tick discovery order that
was never stored, which a pre-archive rebuild would also have to approximate.
The pilot's rows stay in `news_replay` (`hist-liveset`, and the `faithful` arm on
the same ticks); fetches are cached under `cache/news_hist/`.

**Limits.** Runs before the archive began are refused (their pool does not
exist). The derived family and the non-digest tickers rest on the rebuilt pool,
so they are close to live, not identical. Run it in an idle window: ~100–120
model calls per run (only live-scored digests and the pool's extra digests go to
the model), and `--tick-aware` pauses whenever a live tick runs. Replay runs use
their own verdict cache (`cache/sentiment_llm_replay.json`), never the live one.

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
