# Overfitting: what we measure and how to read it (PREREG106)

PREREG106 (registered 2026-10-14 13:09 ET; user: "I want us to be able to evaluate over fitting") builds a
MEASUREMENT layer. It decides nothing, gates nothing and trades nothing. The gating rules at the end are proposals
until the user adopts them.

Since 2026-10-14 the EVALUATION STANDARD itself decides on overfitting's main symptoms (user directive: "check for the
full dataset (A0+A+B) ... we also need to have a enough trades made so that the results are significant, reliable,
and transferable to out of sample data"; CLAUDE.md Standing rules). `src/backtest/bank.py` admits an arm to the bank
only with at least 100 trades in 30 entry months on the pooled 2018-2026 history, the 95% interval of its return per
day (an entry-month bootstrap) above 0, its edge in both chronological halves and with any one calendar year left
out; a filter only on enough significantly losing removed trades, a choice made on the history only with its
out-of-time walk-forward; and `add_rule.pooled_verdict` refuses a gain that one calendar year carries. The tools
below stay reports beside those decisions.

Why: PREREG101 / 103 found that every study since PREREG44 had decided on the same two sets (A, B). On the first
untouched set (A0), the live account kept about half its in-sample return per day (decay 0.47), and two of five books
failed out of sample. The trial counts behind each model had to be rebuilt by reading 106 PREREG files.

## The tools

| tool | where | what it measures |
|---|---|---|
| Trial ledger | `src/backtest/trials.py` → `cache/ml/trials/ledger.jsonl` | Every candidate a study evaluates, recorded once: study, model, variant, sets, metrics, decision, kind. `count(model)` = PREREG103's seed (low / high) + the recorded trials. `set_uses()` = how many recorded decisions each set carried. |
| Deflated verdict | `add_rule.verdict(..., n_trials= or model=, kind=)` → `v["overfitting"]` | Each set's paired gain as a probability, deflated for the N trials behind it, plus the gain shrunk by PREREG81's measured retention. The add-rule decision beside it is unchanged. |
| Set D | `periods.get("D")`, `periods.*(..., forward=True)` | The forward holdout: decisions from 2026-09-25 through the latest completed session (computed when asked). No study may tune on it. |
| Recurring audit | `src/backtest/audit.py` → `cache/ml/audit/<day>.json`, `audit.health()` | PREREG103's evaluation rerun from saved outputs. Gives each model's out-of-sample decay and label per set, and notes any label that moved down since the last audit. |
| The bank (DECIDES) | `src/backtest/bank.py` → `cache/ml/bank/<day>.json` | Each arm on the pooled history: the trade count and its episodes, the entry-month bootstrap interval of its return per day, the chronological halves, leave-one-year-out, growth at its live slice, where it earns and loses (the context cells), the deflated Sharpe ratio; `filter_test` (a filter's removed trades) and `walk_forward` (a choice out of time); the combination's years, months and coverage. |
| Already there | `stats.pbo_cscv`, `stats.deflated_sharpe`, `reliability.plateau`, the reliability cards, `python -m src.backtest.prereg` | PBO over a family of variants, the deflated Sharpe ratio, the parameter plateau, each arm's card, and the registration-before-results check. |

## How a study uses them

```python
from src.backtest import add_rule, trials

# 1. record every candidate BEFORE judging it (idempotent per study / variant / kind; a later decision is a revision)
trials.record("PREREG107", "vol", "G2: give back 1.5", ["A", "B"], {"gain_A": ..., "gain_B": ...}, None, "rule")
# 2. the add rule, with the overfitting report beside its unchanged decision
v = add_rule.verdict(live, variant, model="vol", kind="rule")
print(add_rule.summary(v))   # "... | overfitting report (N 754..80142) (rule retention): A z 3.10 p 0.471..0.107, shrunk ..."
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
   docstring: production would then import two of its modules.

Set D in the simulator. On D's days the vol and fall arms already read lendability from our own archive
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
