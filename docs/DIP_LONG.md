# The dip long book — runbook

The second live strategy (from the 2026-10-09 open; user directive 2026-10-08: "Deploy the mega-cap dip
buying strategy to live production"). Since the 2026-10-12 session it enters through a ½-ATR LIMIT ORDER (user
directive 2026-10-09: "deploy the Order holds nothing, fills into the room left for the 1/2 ATR limit in live for the
dip buying model. Remember that the simulator needs to mimic production as much as possible."). Code
`src/signals/dip_long.py` ("THE LIMIT ENTRY"), ledger steps in `src/performance/tracker.py` (`record_dip_long_trades`
→ `_place_dip_orders`, `monitor_dip_long_orders`, `monitor_dip_long_positions`, `cancel_dip_orders`), the broker's
resting orders in `src/broker/reconcile.py` (`_sync_dip_orders`), tests `tests/test_dip_long.py`,
`tests/test_dip_limit.py` and the dip guards in `tests/test_no_lookahead.py`. The studies:
`memory/long-dip-megacaps-2026-10.md` (the rule), `memory/dip-delayed-entry-2026-10.md` (the limit entry,
PREREG59-61), report https://claude.ai/artifact/3eY5F8UWDZTg2tpEqcpGz6 (sections 8-10).

## The rule

| | |
|---|---|
| Universe | Polygon type CS / ADRC, 20-session mean of close x regular-hours volume >= $1B (`dip_long_min_dollar_volume`); each name once in the project's form — the universe file lists Berkshire as BRK-B AND BRK.B, the candidates keep BRK-B (PREREG64: it was bought twice in the simulator's book) |
| Signal (close of D-1) | close above its 200-session average, 2-session RSI under 10 |
| Order | a BUY limit at that close minus `dip_long_limit_atr` (0.5) x its 14-session average true range, floored to the cent (`dip_long.limit_price`), alive sessions D .. D+4 (`dip_long_limit_sessions`); `dip_long_entry_mode=open` restores the original buy at D's open |
| Placed | at each window session's opening tick, deepest dip (lowest RSI) first; the quantity is fixed when first placed: 1/10 of THE LIVE ACCOUNT's equity at the limit price, within its margin room at Reg T's 50% and the longs' value at most the equity; working orders count toward the 10 dip slots and RESERVE NOTHING; another book's name, no slot, no room: tried again at the next session's opening tick |
| Filled | at the session's OPEN when it opens at/under the limit, else at the END of the first completed 30-minute regular-hours bar whose low reaches it (Polygon's bars); AT MOST what the live account's room allows at that tick (`live_account.fill_cap`) — the rest is not bought; no room: nothing, and the order waits for the next session (`unfunded`) |
| Lapses | after 16:00 of its last session; a margin call cancels the working orders (placed again at a later opening tick) |
| Splits | a split executed after the order's day rescales a pending or working order like a position, from 04:00 ET of its date (`dip_long.apply_order_splits`: close, ATR, trend average and limit x the factor, the limit floored to the cent, the quantity / it in whole shares; event `split`); an open long is rescaled by the ledger (`tracker.apply_splits`, PREREG64) |
| Sell | at the open after the first close above the 5-session average since the FILL session, or after 10 sessions |
| Gives way | a short — or, from 2026-10-12, the crash-rebound book's TQQQ long (`live_account.crash_give_way_plan`) — the shared room cannot fund in full sells the book's latest longs WHOLE, as many as needed (`dip_gave_way`, never bought back) — none when selling them all would not fund one more share of it (PREREG64); at a tick where the maintenance requirement exceeds 80% of equity the latest longs are sold until it is at most 60% (`dip_delever`); a margin call closes everything (`dip_margin_call`) |

The daily bars are built from the deep 30-minute store exactly as the study built them (`daily_bars`); nothing of
session D is read for D's limit, and a touch is judged only on bars that have ENDED by the tick.

## The simulator mimics it

The backtest engine runs the same orders: `src/backtest/strategies/dip.order_pieces` (the same `limit_price`, the
placement instants, one piece per touched session with the open / bar-end fill and 30-minute marks) with
`Strategy(orders="fill")` (a fill cut to the room at that instant; no room: the order keeps working), a tick's fills
before its other entries, and `Rules(one_book_per_name=True)` (the books never share a name). PREREG61 amendment 2,
the live combination 2021-26: +58.6 %/yr from 2021 vs +55.0 buying at the open (+2.3 log points, not significant),
the dip book's return per day 0.184 vs 0.127%; February 2021 - July 2024 lower (+23.6 vs +26.0).

## The meta-model filter (PREREG96's RP33) — BUILT 2026-10-14, OFF until the user deploys it

User 2026-10-14: "I'm ok with deploying PREREG96's dip meta-model filters RP33 and CP33 if they gain in total on all
three sets" — RP33's add-rule gains: set A +2.39, set B -1.06, set A0 +1.13 log points a year (+2.46 in total; CP33
+0.29, and RP33 is ahead on every set). Code `src/signals/dip_filter.py`, switch `enable_dip_long_filter` (False),
model `dip_long_filter_model`, tests `tests/test_dip_filter.py` + the filter guards in `tests/test_no_lookahead.py`,
parity `llm_trader_research/studies/prereg96/prod96/` (`parity96.py` -> `PARITY96.md`, `parity96.json`).

| | |
|---|---|
| Model | PREREG96's "R": a LightGBM regressor (native Booster) of a signal's NET return under the live rule (the 1/2-ATR order's first fill, the live exit, the simulator's dip costs), model96's parameters exactly (300 trees, 31 leaves, learning rate 0.03, feature / bagging fraction 0.8, 50 rows a leaf, seed 0); `cache/ml/dip_long/filter/model.txt` + `model.json` (the features in order, the fit window, the cut and how it was set, the sha256 of the model and of its training table — checked at every load) |
| Fit | the 4,992 FILLED signals of PREREG96's table (2007-07 .. 2026-09: the daily cache before 2021-10, survivors only; the deep store after, delisted names included) whose exit falls before 2026-10-08 — every exit the table holds (its panel ends 2026-10-07); the walk-forward's construction (a signal still open at the cut is not a training row). Alternative beside it: `model_wf2026.txt` = the study's own 2026 walk-forward model (exits before 2026-01-01, cut +0.2279), the model whose 2026 decisions the add rule measured |
| Policy P33 | KEPT when its score >= the cut, +0.2520 (net %): the 33rd percentile of the training rows' out-of-fold scores (the signal years in five folds, `years[i::5]`, each scored by a model fit on the other years) |
| Inputs (D-1's close) | the name's daily bars (`daily_bars` of its store series): RSI2, distance to its 200-session average, to its 20-session high and highest close, ATR14 %, 1 / 5 / 20-session returns, 20-session realized volatility, log10 of its 20-session dollar volume, the limit's distance (1/2 ATR, floored to the cent); `breadth` = the day's signal count; `days_since_prev` = days since the name's previous signal (the rule's last firing in its store series, else the table's last pre-2021 signal kept in the JSON, else missing); SPY's distance to its 200-session average, 5- and 20-session returns, 20-session volatility — from the daily OHLCV cache (`cache/ohlcv/SPY.json`, Polygon's daily bars: the official close, the table's own source; refreshed after each close by the EOD chain), else the deep store's SPY (the last regular-hours trade: 17 of 2,370 walk-forward keep decisions 2021-26 differ with it) |
| When | at the day's signal computation (`dip_long.compute_signals`, the first tick of the opening window): every signal is scored; a dropped one NEVER becomes an order. It takes effect at the next computation — a day already computed keeps its orders, whichever way the switch is turned |
| Journal | `signals/<day>.json`: the `filter` block (the model, the cut, the breadth, SPY's source and session, the counts), every signal's `filter_score` / `filter_keep` / `filter_inputs` / `filter_prev_signal`, the dropped ones in `filtered`; `orders/<day>.json`: each dropped one with status `filtered` and its score (never placed, never rested at IBKR, never filled, takes no slot); the kept orders carry `filter_score` and their trades `dip_filter_score` |
| Fail-open | no model, a checksum that is not the JSON's, or no SPY close for D-1 in either source: EVERY signal is kept (the book as without the filter) and the scorer banner shows a problem; a signal whose own inputs raise is kept unscored (`filter_error`) |
| Off | `ENABLE_DIP_LONG_FILTER=false` (the default): nothing is imported or scored, the signal and order files are byte-for-byte as before |
| Refit | yearly, the walk-forward's recipe: rebuild PREREG96's table (`studies/prereg96/data96.py`) after the year's last exit, then `python -m src.signals.dip_filter --fit --table <signals96.pkl> --cut <YYYY-01-01>`; the JSON records the cut, the rows and the checksums |
| Simulator | `strategies/dip.order_pieces(..., keep=dip.filter_keep())` or `dip.filter_pieces(pieces, dip.filter_keep())`: each signal's WALK-FORWARD score (the model of its year) on production's inputs, every signal scored as production scores it (`filter_keep(research=True)`: the study's draws, where only the filled signals were filtered) — `prod96/wf_scores_prod.pkl` |

Evidence (PREREG96, `memory/arm-reliability-program-2026-10.md`): the return regressor ranks the dip signals out of
sample every deep-store year (Spearman IC +0.11 .. +0.27, mean +0.17, t 6.2); the kept signals' return per day 0.141 ->
0.199 %/day on set A and 0.318 -> 0.372 on set B; the live account under the add rule's paired stratified draws:
set A +2.39 log points a year (20 draws), set B -1.06 (20; rejected there by futility: in set B's bull market every
dip paid and the book's growth was its volume), set A0 +1.13 (33); no margin-call draw added on any set, the worst
drawdown 15.1 -> 15.1 / 27.2 -> 27.9 / 29.3 -> 27.0%. PARITY (2026-10-14,
`prod96/PARITY96.md`): production's fit reproduces model96's 16 walk-forward models exactly (scores identical, 4,635 of
4,635 keep flags, the cuts); production's inputs equal data96's on all 3,481 deep-store signals but for `breadth` on 15
rows (production counts 7 firings the study's table left out at its edges: 6 still open when the table was built, one
on a delisted name's last session); the keep decisions on production's inputs equal the study's on 2,370 of 2,370
filled signals (and on A0's 900 of 900). Refits move the decisions: consecutive walk-forward models agree on 82-89% of
a year's filled signals out of sample; the 2026-10-08 refit agrees with the 2026 walk-forward model on 74% of 2026's
(462 of 624, signals it trained on).

```bash
python -m src.signals.dip_filter --status                    # the model, its fit window, its cut, the switch
python -m src.signals.dip_filter --score --day 2026-10-14    # what it would keep of a day's signals (writes nothing)
```

## A normal day

1. 08:30 the pre-open run and 09:00 the selection short's prepare extend the deep 30-minute store through the
   previous session, and cache Polygon's whole-market daily bars the prefilter reads.
2. 09:30 tick, start: `monitor_dip_long_positions` sells every long whose exit is due (log `[dip_long] SELL`).
3. 09:30 tick, live path: `record_dip_long_trades` computes the day's signals once (`signals/<day>.json`, ~2 s),
   writes them as orders (`orders/<day>.json`), places the pending orders and arms the carried ones (log
   `[dip_long] LIMIT`), and fills at the open those whose session opened under the limit (log `[dip_long] BUY`).
   The broker sync then rests each working order at IBKR (`[broker] dip limit ... resting`, refs `dipw-...`).
4. Every later tick, at its start: `monitor_dip_long_orders` fills the orders whose limit a bar reached (log
   `[dip_long] BUY ... limit order filled`), before the shorts' entries; each sync resizes the resting IBKR orders to
   the room left (cancel + re-place under the next ref). The long a fill opens is linked to its resting order.
5. The first tick at/after 16:00 judges the session's last bar, then lapses the orders whose window ended.

## Checks

```bash
python -m src.signals.dip_long --status                       # the account and the open longs
python -m src.signals.dip_long --signals --day 2026-10-09 --no-write   # what the book would order (prints only)
```

Files under `cache/ml/dip_long/`: `signals/<day>.json` (the computation: `signals` with `atr14` and `limit`, the
closest misses in `next`, `stale` = names that traded on D-1 but the store lacks, `inactive` = names that did not
trade), `orders/<day>.json` (each order's life: `placed` / `armed` / `filled` / `unfunded` / `held` / `no_slot` /
`no_room` / `cancel_margin` / `lapsed`, its quantity, the IBKR order resting for it), `entries/<day>.jsonl` (each
fill), `exits/<day>.jsonl` (each sale with the close and average that fired it).

## When something is wrong

* **Scorer banner "dip long: no signal computation for today's open"** — the 09:30-10:30 ticks never reached
  the entry step (scheduler down, an exception: grep `[dip_long]` in the day's log). The day's pending orders are
  neither placed nor armed today; they are tried again at the next session's opening tick. Exits are not affected.
* **"names had no close for D-1 in the store"** — the pre-open extension failed for those names; the entry
  step tried to extend them itself. Run `python -m src.data.intraday_store --extend --tickers A,B` out of
  hours if it repeats.
* **A fill the paper account did not get** — the bars reached the limit but IBKR's resting order did not fill
  (queue): the linked entry leg is cancelled by the tick-scoped lifetime and resent at the market; the trade log
  shows the slippage. **A resting order IBKR filled without a bar touch** — no ledger long owns it: the drift pass
  flattens it.
* **"held past 10 sessions"** — an exit did not go out (no live price for days?): check the trade's
  `dip_sessions_held` and the exits journal.
* **"the meta-model filter did not score today's signals"** (the filter on) — the day's every signal became an order,
  as without the filter. The reason is in the problem and in the signal file's `filter.error`: the model file missing
  or not the one its JSON's sha256 names (`python -m src.signals.dip_filter --status`), or no SPY close for D-1 in
  the daily cache nor the deep store (the EOD chain's cache warm and the pre-open run both failed). **"computed without
  it"** — the day's signals were computed before the switch went on: the filter starts at the next session.
* **To stop new longs**: `ENABLE_DIP_LONG=false` in `.env` and restart the scheduler: no order fills, the resting
  IBKR orders are cancelled at the next sync, the orders lapse. The open longs still leave on their rule.
  `DIP_LONG_ENTRY_MODE=open` returns to buying at the open.
