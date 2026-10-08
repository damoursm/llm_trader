# The multi-strategy backtest — runbook

User directive 2026-10-08: "Rework our simulator that consider a real environment with restrictions when trading so
that it can consider multiple strategies. The idea is to be able to reuse it to evaluate different models or
combination of models. It should be able to calculate the real fees and be able to know the available capital from
the 10000$ starting account. We should then be able to use it to get the growth ... and optimize our optuna studies."

Research tooling: nothing in production imports `src/backtest/`.

## The pieces

| Module | What it does |
|---|---|
| `src/backtest/account.py` | THE account: $10,000, no deposits, every strategy's trades in one IBKR margin account — longs and shorts, real fees, Reg T + IBKR house margin, available capital, margin calls, per-strategy attribution, both metrics (`simulate`, `metrics`) |
| `src/backtest/strategies/vol.py` | the vol short arm (and any variant of its rule) as pieces, from the research vol engine (`VolEngine`) |
| `src/backtest/strategies/dip.py` | the dip long book (and variants) as pieces, from the deep 30-minute store + the delisted names (`pieces`, `DipParams`) |
| `src/backtest/strategies/etf.py` | the ETF short arm as pieces (`EtfArm().pieces(lo, hi)`): the research ETF run's picks (`<research>/etf/vol_events_de_arm.pkl`) under the live rule — half give-back, filters, volatility exit, 6x cover, IBKR's archived lendability to 2024-06; reproduces the recorded 141-trade book exactly (`<research>/etf/bt_etf_parity.py`) |
| `src/backtest/study.py` | weekly Sunday walk-forward re-tuning of one or several arms TOGETHER, objective = the window account's log growth; Reality Check / argmax / Optuna picks; the chained test account |

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
  below maintenance = a margin call that closes everything at twice the entry half-spread.
* Fees: each fill's half-spread (real NBBO or the fitted model), IBKR's fixed commission max($1, $0.005/share) capped
  at 1%, the SEC fee + FINRA TAF on every sale, a short's IBKR borrow day by day.
* Metrics (`account.metrics`): growth per year, LOG growth per year (the study objective), worst drawdown, margin
  calls; per strategy the positions, P&L, costs, average realized trade, hold, return per day, and the sum of each
  position's log(1 + P&L / equity at entry) — the per-trade objective's account form (equal to the log growth when
  positions do not overlap; with overlap the account number is the true one).
* Activity and capital in use: `simulate(..., keep_exposure=True)` also records, after every instant, the open longs'
  and shorts' value at their marks and their initial margin (`curve_long` / `curve_short` / `curve_req`; the results
  are unchanged); `exposure_daily(res, sessions)` gives them per session; `activity(res, sessions, lo, end)` counts
  the trades opened per calendar year and strategy and the sessions with a position open during regular hours
  (`holding_days`: a long sold AT the open does not hold that day).
* Lack of capital: `simulate(..., keep_skips=True)` records every entry not taken in full — the shares its slice
  wanted, the shares it got (0 = skipped) and the binding limit (`margin`, `side_cap`, `volume_cap`, `house_shrink`,
  `max_open`, `min_equity`, `below_one_share`, `lendable`); `capital_summary(res, pieces)` counts them per strategy.

## Validation (2026-10-08)

* Vol arm: the live book (2021-02..2024-06 and 2021-26, real NBBO, both margin triggers) through `account.simulate`
  == the audited cap5k7, bit-identical in 4 of 4 runs ($14,665.034514, 109 trades; $23,137.78 with the 2025 call);
  the vol adapter reproduces the same book; the default settlement calendar equals the research one (2,062 days).
* Dip book: all 3,417 of the study's signals reproduced (+137 Berkshire class-share signals the study's type lookup
  missed); the study's account within $38 ($21,030 vs $20,992, +16.3 %/yr).
* Study driver: PREREG35's per-week live scores reproduced on 42 windows (same trades; |diff| <= 0.02 pp/yr — the
  research evaluator re-prices a cut trade's exit spread at the mark bar).
* Tests: `tests/test_backtest_account.py`, `tests/test_backtest_study.py`.

## Use

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
