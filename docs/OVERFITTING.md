# Overfitting: what we measure and how to read it (PREREG106)

PREREG106 (registered 2026-10-14 13:09 ET; user: "I want us to be able to evaluate over fitting") builds a
MEASUREMENT layer. It decides nothing, gates nothing and trades nothing. The gating rules at the end are proposals
until the user adopts them.

Since 2026-10-14 the EVALUATION STANDARD itself decides on overfitting's main symptoms (user directive: "check for the
full dataset (A0+A+B) ... we also need to have a enough trades made so that the results are significant, reliable,
and transferable to out of sample data"; CLAUDE.md Standing rules). `src/backtest/bank.py` admits an arm to the bank
only with at least 100 trades in 30 entry months on the pooled 2018-2026 history, the 95% interval of its return per
day (an entry-month bootstrap) above ITS BAR (0 for a short arm; SPY's average day for a long arm, whatever it holds —
PREREG123), above the bar in both chronological halves and with any one calendar year left out, and the arm alone
growing at its live slice; an arm failing only significance is re-judged on its longest history (PREREG119 / 120). A
filter is admitted only on enough significantly losing removed trades, a choice made on the history only with its
out-of-time walk-forward; and `add_rule.pooled_verdict` refuses a gain that one calendar year carries. The tools
below stay reports beside those decisions.

Why: PREREG101 / 103 found that every study since PREREG44 had decided on the same two sets (A, B). On the first
untouched set (A0), the live account kept about half its in-sample return per day (decay 0.47), and two of five books
failed out of sample. The trial counts behind each model had to be rebuilt by reading 106 PREREG files.

## The tools

| tool | where | what it measures |
|---|---|---|
| Trial ledger | `src/backtest/trials.py` → `cache/ml/trials/ledger.jsonl` | Every candidate a study evaluates, recorded once: study, model, variant, sets, metrics, decision, kind. `count(model)` = PREREG103's seed (low / high) + the recorded trials. `set_uses()` = how many recorded decisions each set carried. IN PRACTICE (2026-10-17): the ledger holds only PREREG103's 6 seed rows — no study since has called `trials.record`, so `count()` is the seed and `set_uses()` zero; studies pass N by hand (`n_trials=`). |
| Deflated verdict | `add_rule.pooled_verdict(..., n_trials= or model=, kind=)` → `v["overfitting"]` (the pooled gain under `["sets"]["pooled"]`; the two-set `add_rule.verdict` keeps the same report for the studies before 2026-10-14) | The paired gain as a probability, deflated for the N trials behind it, plus the gain shrunk by PREREG81's measured retention. The add-rule decision beside it is unchanged. |
| Set D | `periods.get("D")`, `periods.*(..., forward=True)` | The forward holdout: decisions from 2026-09-25 through the latest completed session (computed when asked). No study may tune on it. |
| Recurring audit | `src/backtest/audit.py` → `cache/ml/audit/<day>.json`, `audit.health()` | PREREG103's evaluation rerun from saved outputs. Gives each model's out-of-sample decay and label per set, and notes any label that moved down since the last audit. Run once (2026-10-14), by hand; not wired (below); its combination row still reads PREREG95's LIVE_S2 draws. |
| The bank (DECIDES) | `src/backtest/bank.py` → `cache/ml/bank/<day>.json` | Each arm on the pooled history: the trade count and its episodes, the entry-month bootstrap interval of its return per day, the chronological halves, leave-one-year-out, growth at its live slice, where it earns and loses (the context cells), the deflated Sharpe ratio; `filter_test` (a filter's removed trades) and `walk_forward` (a choice out of time); the combination's years, months and coverage. |
| Already there | `stats.pbo_cscv`, `stats.deflated_sharpe`, `reliability.plateau`, the reliability cards, `python -m src.backtest.prereg` | PBO over a family of variants, the deflated Sharpe ratio, the parameter plateau, each arm's card, and the registration-before-results check. |
| Sharpe inference (PREREG183) | `src/analysis/stats.py`: `sharpe_variance`, `sharpe_psr`, `min_track_record`, `track_record_for_power`, `critical_sharpe`, `sharpe_power`, `pfdr`, `ofdr`, `fdr_critical_sharpe`, `deflated_sharpe_k`, `sharpe_moments`, `ols_hac`, `ols_cluster` | López de Prado, Lipton & Zoonekynd's Sharpe inference (SSRN 5520741): the estimate's variance under skew, fat tails and AR(1) autocorrelation, the PSR under the null, how long a record must run, power, the planned and observed false discovery rates, the best-of-K test with the maximum's variance; the robust regressions for market exposure. |
| A card on its months (PREREG183) | `bank.monthly_unit_pnl`, `bank.dsr_months` → every card's ``dsr_months`` | The deflated Sharpe on the arm's monthly unit-bet P&L (its episodes), beside the per-trade one that counts clustered trades as independent. |
| The live record's horizon (PREREG183) | `bank.track_record` → the bank's ``combination.track_record`` | How many months the live account needs to show a profit, and to show a shortfall, at the backtest's Sharpe and at half of it. |
| A change against the market's noise (PREREG183) | `add_rule.market_noise`, `pooled_verdict(..., market=True)` | The paired monthly gain's Newey-West t, its p with ρ / skew / kurtosis, deflated for K tries, the observed FDR at a prior, and the chance a no-effect change shows the +0.5 gain. |
| The add rule's own record (PREREG184) | `llm_trader_research/studies/prereg184/h184.py` (index, pairs, parity, wf, loyo, lp) | Every recorded verdict matched to the exact saved draws it compared; each deployment pair re-decided at each year-end on the years before it and scored on the next year: the hit rate of its adds against its refusals, the retention of an add's in-sample gain, stricter bars (the gain, the market-noise t) on the same decisions, study and year bootstraps. |
| The forward check (PREREG185; production from 2026-10-23) | `src/backtest/forward_check.py` (the EOD chain's `forward_check` step -> `cache/ml/forward_check/<day>.json`, notes in the email; the research version `llm_trader_research/studies/prereg185/h185.py all <ledger copy>` -> `trades185.json`, `band185.json`) | Every live trade rebuilt by the simulator's rules on Polygon's as-traded bars and paired with the ledger's and IBKR's: entry, exit at the same tick, exit timing, costs, the net trade; the live account in the backtest's tracking band (and a held-out one). Research only. |

## How a study uses them

```python
from src.backtest import add_rule, trials

# 1. record every candidate BEFORE judging it (idempotent per study / variant / kind; a later decision is a revision)
trials.record("PREREG107", "vol", "G2: give back 1.5", ["A", "B"], {"gain_A": ..., "gain_B": ...}, None, "rule")
# 2. the POOLED add rule, with the overfitting report beside its unchanged decision ({set: {seed: record}} each)
v = add_rule.pooled_verdict(live, variant, model="vol", kind="rule")      # or n_trials=5 (PREREG137 passed it by hand)
print(add_rule.pooled_summary(v))   # "... | overfitting report (N ...) (rule retention): pooled z ... p ..., shrunk ..."
# 3. record the outcome (appends a revision of the same trial)
trials.record("PREREG107", "vol", "G2: give back 1.5", ["A", "B"], {...}, v["decision"], "rule")
```

Kinds with a measured retention: `"rule"` (a rule tweak or parameter change) and `"sizing"`. Any other kind
(`"mechanism"`, ...) is recorded but gets no shrink. The sets a trial may name: A0, A, B, C, D, LONG.

## Reading the numbers

* **z, p (deflated).** z = gain / se, where se is the paired standard error across the draws.
  z0(N) = (1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)) is the expected best z among N skill-less trials (z0 = 3.17 at
  N = 753, 4.34 at 80,141), and p = Φ(z − z0). It is reported at the ledger's low and high N.
  **Caveat:** se is the SIMULATION's error (the lendability draws), not the market's. A clear gain gets z > 10, so p
  ≈ 1 at any N. p therefore only flags gains that are not even distinct from draw noise once selection is counted.
  For the market's own noise, pass a history-resampled se (a block bootstrap of the paired difference) to
  `add_rule.deflated_gain`.
* **shrunk.** gain × retention: rule 0.15 (band 0.09–0.23), sizing 0.5 (0.47–0.50), from PREREG81's CSCV halvings.
  This is the expected out-of-sample effect, and the prediction set D should confirm. `clears_min_gain` asks whether
  it still reaches +0.5 log points; for a rule tweak that needs an in-sample gain of about +3.3.
* **Audit labels.** These are PREREG103's registered labels. ROBUST: the OOS 95% interval of return per day is above 0
  and decay ≥ 0.5. WEAKER OUT OF SAMPLE: above 0, decay < 0.5. FAILS OUT OF SAMPLE: the interval includes 0. Set D is
  labelled only from 10 trades (the other sets keep PREREG103's rule). A move down (ROBUST → WEAKER → FAILS) since the
  previous audit — or since PREREG103, for the first one — is a note for the email's green scorer line, never an
  alarm. `--parity` reproduces PREREG103's table 3 and table 1 exactly from its own inputs.
* **The ledger's counts are nominal.** Correlated variants are far fewer independent trials, so a deflation by them
  is conservative. The seed carries no per-set history, so `set_uses` starts counting at the ledger's first record.

## Measured: the optimism of a joint tuning (PREREG153, 2026-10-20)

A tuning is judged OUT OF SAMPLE (CLAUDE.md Standing rules, Evaluation: "EVERY TUNING"). PREREG153 measured what that
costs for the live combination's 32 parameters (25 rules, 7 sizing; PREREG152's values), tuned TOGETHER by coordinate
ascent on the full account (draws 0-1), once on all nine years 2018-2026 and once per year held out, then each year
scored at its own fold's settings and the years stitched into one account (seeds 0-4, the bank's pooled figure):

| | growth search | return-per-day search | live settings |
|---|---|---|---|
| in sample (tuned on all years) | 82.1% a year | 67.8% (return per day 0.77 %/day) | 67.3% (in sample too) |
| held out (each year tuned without it) | **57.1%** | **57.8%** (return per day 0.63 %/day) | — |
| optimism (in sample minus held out) | +24.9 ± 5.1 pp | +10.0 ± 1.3 pp | — |

PREREG152 (the same folds, each rule parameter tuned ALONE on its own arms' trades, the sizing on the account): by
return per day 66.0% in sample -> **61.0%** held out (optimism +5.0 pp; return per day 0.78 -> 0.79 %/day: kept whole; no margin call in 5 draws), by growth 59.0% -> 35.7% (calls in every draw). The steadiest
objective is return per day chosen on each arm's own trades — many trades, little noise; the least steady, growth.

THE TRAINING PROTOCOL (the user's, 2026-10-21; CLAUDE.md's standing rule of that name; PREREG172 / 176): each book's
holding parameters searched together on the book alone, by its return per day (R x growth for the books with a very
good return per day), then the sizing for growth searched jointly on the account with the margin events and the
drawdown guarded, every choice cross-validated by year. Its held-out track: PREREG172 (the margin calls before PREREG174)
R x G -1.6 +- 2.7 pp against R alone; PREREG176 (IBKR's liquidation) **71.0%** a year against the live configuration's
76.0% on the same draws (-4.9 pp), while its in-sample choice gained +2.1 pp a year (40 draws) carried by 2025. Each
book's own return per day rose in sample and fell held out in the account (0.761 -> 0.621 %/day): the per-book
objective overfits too, mostly through the books whose parameters barely matter.

How to read it (paired by draw: per draw the pooled growth, then the mean difference). The growth search's in-sample
gain over live (+15.0 pp) became -10.0 ± 4.4 pp held out: re-tuning these parameters on eight years does not help the ninth. The return-per-day search kept about
48% of its return-per-day gain held out (0.50 -> 0.63 %/day against 0.77 in sample, in line with
PREREG81's sizing retention of ~0.5) but grew 9.5 ± 1.5 pp a year less than live. The risk limits
were fitted too: a search that checks margin calls on two draws meets more of them on new draws (the growth search's
all-years settings: 1 of the 2 searched draws, as live; 3 of 3 new draws, live 1). Live's own settings came from many
studies on the same history, so its 67% is in sample as well; the held-out tracks (57-61% a year, growth tuned one at a time aside) put the cautious
expectation between the high 50s and about 60%. Even they are optimistic: only the parameters were held out (the arms, their designs and the dip model
were chosen on this history), and a fold trains on later years.

The harness (`llm_trader_research/studies/prereg153/`): `joint153.py parity` (the gate: the live configuration's cached
books must reproduce the reference draws exactly), `joint153.py coord` (every fold's ascent in lockstep, one parameter
a round, then the final draws) with `joint153.py work <id>` workers (claim files; IDLE priority; one in market hours),
`analyze153.py` (the registered readings), `report153.py` (the report's tables), `extra153.py` (return per day held
out, the folds' training gains, the calls by draw). A future tuning reuses it with its own grids and folds; a
walk-forward (each year tuned on the years before it only) is the stricter form.

## Sharpe-ratio inference (PREREG183, 2026-10-22)

The user, after a survey of the public Claude trading skills (none worth installing): "Copy the formulas to improve
our own skills." The formulas are López de Prado, Lipton & Zoonekynd's ("Sharpe Ratio Inference: A New Standard for
Reporting and Decision-Making", SSRN 5520741), ported from their reference code's formulas into `src/analysis/stats.py`
(no scipy) and pinned to that code's own test values. With ρ the returns' first-order autocorrelation, γ3 the
skewness, γ4 the plain kurtosis, a = (1+ρ)/(1-ρ), b = (1+ρ+ρ²)/(1-ρ²), c = (1+ρ²)/(1-ρ²):

* V[SR] = (a - b γ3 SR + c (γ4-1)/4 SR²) / T, times the variance of the maximum of K normals for the best of K;
* PSR = Φ((SR - SR0) / sqrt(V[SR0])): the variance UNDER THE NULL (the old `probabilistic_sharpe` takes it at the
  estimate with T - 1 and no ρ — kept for the studies before);
* MinTRL = V[SR0](1) (z / (SR - SR0))² (an even chance); `track_record_for_power` for any power;
* pFDR = 1 / (1 + (1-β) P[H1] / (α P[H0])) reads a DECISION RULE; oFDR = p0 P[H0] / (p0 P[H0] + p1 P[H1]) reads an
  OBSERVATION (its alternative is an assumption: state it).

What they said about our own record on 2026-10-22 (`llm_trader_research/studies/prereg183/`):

* **The live record's horizon** (NEWMARKS, 40 draws a set; monthly Sharpe 0.61, skew +1.5, kurtosis 8.5, ρ +0.17):
  at the backtest's own Sharpe, 10 months for an even chance to show the account makes money and 21 for 80% power;
  at half of it (PREREG103's out-of-sample decay), 41 and 85 months; 80% power to show the live Sharpe is HALF the
  backtest's needs 67 months (44% at two years). The daily returns give the same picture (8 / 18; 33 / 71; 63). The
  live record (3 sessions on 2026-10-22) has a power of about 5%: nothing yet.
* **The cards on their months.** The per-trade deflated Sharpe counts every trade as an observation. On the arms'
  monthly unit-bet P&L (105 months, the paper's best-of-K test) the vol arm's deflated Sharpe at its 753 trials is
  0.10 (0.56 per trade; the paper's test on the trades alone 0.69: its trades cluster in squeeze months), the
  capitulation book's 0.69 (0.73; 0.89), the fall arm's 0.87, the ETF arm's 0.99, the crash book's 0.98 (12
  trials), the dip book's 1.00. The bank's month bootstrap — the gate — already counted months; the deflated Sharpe
  is reported beside it.
* **Skill or market** (each trade on SPY's move over its own window, clustered by month): the dip book's average
  trade +2.22% is +2.00% alpha (t 6.2) and +0.22% market (β 1.26); the capitulation book +1.85% alpha (t 3.0, β 1.07,
  the market fell -0.25% during its trades); the crash book has no alpha (β 3.95, R² 0.96: its edge is WHEN it holds
  3x the market, by design); the shorts' alphas +5.6 to +9.7% a trade. The account: +5.1% a month of alpha (t 5.4),
  monthly β -0.06 — market-neutral over months; daily β 0.47 the same day, -0.16 the next.
* **The dip meta-model's folds purged** (rows whose label window overlaps a test year, and a 14-day embargo): the RP33
  cut 0.2520 -> 0.2459; 5 of the 5,439 signals of 2018-2026 change side. The out-of-fold scores themselves move (10%
  of the training rows' keep decisions flip, Spearman 0.93): the per-signal score is noisy, the cut is not.
* **The deployed changes against the MARKET's noise** (the paired monthly log difference, the mean over 20 draws a
  set): DMTE +2.1 log points a year, t 1.85, p 0.02 (deflated for 10 tries 0.14); S3 +8.8, t 1.76, p 0.05 (0.47);
  RPD152 -0.8, t -0.13 (risk protection); ALL3 +4.2, t 1.63, p 0.05 (0.44); V15 +1.8, t 1.73, p 0.06 (0.51). At the
  research log's deploy rate as the prior (32 of 147 studies, 22%) and an alternative as large as observed, each has a
  12-30% probability of being noise — about one of the four gains expected false. A no-effect change shows the add
  rule's +0.5 gain 32-47% of the time on this history: the +0.5 bar guards against the draws' noise, not the market's.

How to read them: these are REPORTS. The bank's gates and the add rule decide as before; the readings say how much
the history can prove. The months, not the trades, are the unit; a change's market t is the number to quote beside
its draws' t; the live record is uninformative for about a year at best.

## The add rule out of time and the forward check (PREREG184 / 185, 2026-10-22)

The user, on the plan for being more sure the growth transfers to live: "Do 1 and 5".

THE ADD RULE'S OWN RECORD (PREREG184). 303 verdict entries of the studies' verdict files were matched exactly to the
draws they compared (214 pairs; the rule rebuilt from the draws reproduces all 305 decided entries). The 169
deployment pairs (34 studies) were re-decided at each year-end from 2019 on the calendar years before it and scored
on the next year, each study counted once:

* The rule's adds gained +0.67 pp a year the following year, its refusals +1.66 (in log points +0.33 against +0.46;
  without 2026, whose part-year is annualized, +0.07 against +1.39 pp). An add kept 0.11 of its in-sample gain. Over
  every decision the next-year gain rises with the in-sample one (slope 0.40): large effects persist, the margin near
  +0.5 is the market's noise.
* A market-noise floor (t >= 2 on the in-sample months, beside the rule) passed 31 decisions instead of 161; 75%
  gained, +6.19 pp against +0.67 (p 0.06 by study; +2.1 log points, p 0.05), the same total. The gain bars alone
  (+1, +3.3 log points) do not raise the hit rate.
* The rule's risk tests: the one-pair margin-call count refused changes that gained +12.5 pp out of time with no
  more calls; the drawdown test refused changes that went 1.3 points deeper (as designed). The count was replaced by
  the rate on 2026-10-23 (the user's call, below).
* Upper bounds: every candidate's design saw all the years.

THE FORWARD CHECK (PREREG185). The 16 live trades from 2026-09-25 (14 dips, two shorts) rebuilt by the simulator's
rules on the same prices: all 14 closed trades left on the simulator's own tick; the dip touch fills equal to the
cent; the 12 closed dips -0.12% a trade live minus the simulator (one episode: the fresh start closed nine); their
09:30 exits -0.17% under the opening print the simulator books (AUDIT151 priced the open ENTRIES at the attainable
09:30:30 price, never the exits); the quote model charged a thin small cap 4.4% where IBKR's fills were favourable.
The live account after three sessions sits at the 4th percentile of the backtest's three-session windows (the META
concentration). No execution gap beyond noise yet; the tool reruns on any later ledger copy.

Proposed (the user's call, not applied): a market-noise floor for deploying. ADOPTED 2026-10-23: the margin-call
criterion as a rate with its sampling error (`add_rule.margin_events`: the one-sided exact binomial on the (set,
draw) pairs where one side alone meets a margin event, p < 0.05; the user: "We shouldn't judge the performance based
on number of margin calls anymore, just the performance in rates (return per day and growth)"). The forward check is wired into the EOD chain and the email from 2026-10-23 (the user: "Do both
for number 4"); the user chose a SECOND ACCOUNT under the market-noise floor instead of tightening the live rule
(PREREG186; PREREG189: the dip book, the fall arm and the crash book together at their live rules pass it).

## Proposed gates under the pooled standard (NOT applied; PREREG106, for the user to adopt or not)

The pooled standard (2026-10-14) settles two of the four first proposed: G-A0 (A0 confirms A and B) is moot — A0 is
part of the one history every decision reads —, and a parameter chosen on the history is already measured out of time
(`bank.walk_forward`). What remains to adopt:

* **G-K.** One primary candidate per study: with k variants tried, the pooled add rule's +0.5 bar applies to the
  registered primary only. The others are exploratory and need a fresh study (or set D) to confirm.
* **G-PLATEAU.** A parameter change (not a new mechanism) needs its plateau (`reliability.plateau`) on the pooled
  history, and its walk-forward choice inside that plateau in most years.
* **G-FORWARD.** A deployed change is reviewed on set D once it has accrued its registered sample. If it falls below
  its prediction's lower band (the shrunk band), it is offered for removal.

## Not wired yet (the hooks, for when the user adopts them)

The audit runs only by hand (`python -m src.backtest.audit`, about 1 s). To run it every market day:

1. `config/settings.py`, beside `enable_eod_live_calibration`: `enable_eod_overfit_audit: bool = False`.
2. `src/scheduler/eod.py`: append `"overfit_audit"` to `STEPS` (after `"live_calibration"`), add
   `"overfit_audit": settings.enable_eod_overfit_audit` to `_enabled`, and register the step in `STEP_FNS`:

   ```python
   def _overfit_audit(day) -> Tuple[object, str]:
       # PREREG106: PREREG103's evaluation rerun from saved outputs; a degraded out-of-sample label is a scorer note
       from src.backtest import audit as AU
       p = AU.write(_day(day))
       notes = json.loads(p.read_text(encoding="utf-8")).get("notes") or []
       return {"path": str(p), "notes": notes}, ("; ".join(notes) if notes else "no label degraded")
   ```
3. `src/pipeline.py`, after the `live_calibration` block of the scorer health:

   ```python
   try:                                  # PREREG106's audit: an out-of-sample label that degraded (notes)
       from src.backtest import audit as _audit_h
       sel_health["notes"] = list(sel_health.get("notes") or []) + _audit_h.health()["notes"]
   except Exception as _au_e:                                 # noqa: BLE001
       logger.warning(f"[audit] health read failed (fail-soft): {_au_e}")
   ```
4. `tests/conftest.py`, beside `live_calibration_dir`: `monkeypatch.setattr(audit, "AUDIT_DIR", tmp_path / "audit")`
   (so the email's banner never reads production's audits in a test). Also update `src/backtest/__init__.py`'s
   docstring: production would then import three of its modules (it already imports `live_calibration` and, through
   the news-hold shadow, `mimic`).

Set D in the simulator RUNS from 2026-10-23 (PREREG188: PREREG105's stores extended, the gaps below closed — the ETF arm's lendability and the fees on our own archive, the evaluator's data end moved, D's dip scores, breadth, quotes and open fills; it reproduces production's decisions and its account from the fresh start, the one divergence IBKR's refusals by name). The map as it stood before. On D's days the vol and fall arms already read lendability from our own archive
(`mimic.lend_all` from `OWN_ARCHIVE_FROM`, `turn_lend`). Three gaps remain:

* **Data.** The research engine's stores end at the 2026-09-25 close (`optvol/wf5.py` `END_CUT`, `build.py`
  `HI` / `DATA_END`, `cap5k4.py` `END`, `EtfArm.pieces(data_end=...)`, PREREG62's extended-hours bars). D needs them
  extended through the latest session, for example by PREREG105's continuous construction. Then
  `python -m src.backtest.reliability --build --sets D` writes `cache/ml/reliability/<day>_D/`, which the audit reads.
* **ETF lendability.** The ETF arm still DRAWS lendability after 2024-06-24 (`lend_after`). In `EtfArm.pieces`, the
  archive branch should also take our own archive's days: `own_from = ve.dnum(OWN_ARCHIVE_FROM)` next to
  `lend_end = ve.dnum(LEND_END)`. Then `if lend == "archive" and (int(r.dn) <= lend_end or int(r.dn) >= own_from):` at
  the pick, and `... and (t_day <= lend_end or t_day >= own_from):` at the turn. A / B / A0 are unchanged; their days
  are below `own_from`.
* **Fees.** The research engine's `RealFees` reads IBKR's API parts (refreshed to about 2026-10-06 for most names),
  carries each one 7 days, then falls back to the fee model. In `optvol/realfee.py` `RealFees.series`, our own
  archive's daily closes (`cache/ml/deep/borrow_daily.parquet`, read once; `borrow_history.fee_daily`'s
  `own_archive` source) should win on their days from 2026-09-25.

The map of 2026-10-22 (PREREG185 [R8]): besides the three, the dip RP33 scores and the halts end 2026-10-06, the
capitulation breadth and the dip panel 2026-10-16, and D's legs have no quotes. The shortest route is PREREG105's AB2
stores, which already know set D (data to 2026-10-13): `fetch105.py extend <day>` and `work` (~10.9k Polygon
requests), `build105.py work` (~2 h), AB2's ETF events without the stock-era tickers, a D runner on live105's with the
vol / fall `END_CUT` moved, then a parity check of AB2 against the default construction on set B before any reading.
Trade-by-trade parity with live holds only from 2026-10-21 (the configuration changed six times inside D).
