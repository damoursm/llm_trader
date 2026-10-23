# The capitulation book (`cap_long`) — runbook

Live from the 2026-10-21 session (user 2026-10-20: "the last lead, buy large stocks together on market-wide
capitulation days, is a good lead even if one year carries it because it's a buy and we expect most years to be good
for buying. Plus, it lifts the weak months", then, on PREREG167's report: "Deploy all three."). Code
`src/signals/cap_long.py`; ledger steps `tracker.record_cap_long_trades` (→ `_place_cap_orders`) /
`monitor_cap_long_orders` / `monitor_cap_long_positions` / `cancel_cap_orders`; sizing `live_account.size_cap` and the
give-way order (`_give_way_longs`, `fill_give_way_plan`); the broker's resting orders `reconcile._sync_book_orders`
(refs `capw-…`); simulator `src/backtest/strategies/cap.py` (`Mimic(cap=True)`, the default while `enable_cap_long`
is on). Tests `tests/test_cap_long.py` + `tests/test_no_lookahead.py`. Evidence: `memory/rpd-review-capitulation-2026-10-20.md`,
`llm_trader_research/studies/prereg165/` (the rule, its out-of-sample test, the account), `prereg167/` (with the dip
book's meta-model cut and the crash book's after-hours exit).

## The rule

- **The breadth** (`cap_long.breadth_at`): at the close of session S, the share of the 100 most-traded common stocks /
  ADRs (`cap_long_breadth_top`; ranked by their 20-session mean close × volume in Polygon's whole-market daily bars,
  each with at least 220 bars in its run — a gap of more than 30 calendar days starts a new run) that closed UNDER their
  own 200-session average with a 2-session RSI under 10. The bars are Polygon's unadjusted ones (the borrow-pressure
  model's cache, `sel_pressure.grouped_ohlc`, 300 sessions through S); the closes are brought into S's terms by the
  effective split set (`intraday_store.split_rows`). A session missing from the cache is an error, never a guess: the
  breadth is not kept and the book places nothing that day (a scorer-banner problem after the opening window).
- **A capitulation session**: B(D−1) ≥ `cap_long_breadth_min` (0.20). Then EVERY common stock / ADR that at the
  close of D−1 traded ≥ `cap_long_min_dollar_volume` ($2B) a day, closed under its `cap_long_trend_sessions` (200)
  average with a 2-session RSI under `cap_long_rsi_max` (10), on 220 daily bars or more (the dip book's construction
  from the deep 30-minute store, each name from its own security) becomes a BUY LIMIT ORDER at that close minus
  `cap_long_limit_atr` (½) × its 14-session ATR, floored to the cent, alive sessions D … D+2
  (`cap_long_limit_sessions` 3). Every signal is an order — no daily top pick, no meta-model, no earnings demotion —
  the deepest dip (the lowest RSI2) first when they compete.
- **The order's life** is the dip book's (`docs/DIP_LONG.md`): placed at each window session's opening tick (its
  quantity fixed then), filled at the session's open when it opened at/under the limit, else at the END of the first
  completed 30-minute regular-hours bar whose low reaches it — at most what the live account's room allows then
  (`live_account.fill_cap`); a touch with no room buys nothing (`unfunded`, retried the next session); lapsed after
  its last session's 16:00; a split executed after the order's day rescales it (`dip_long.apply_order_splits`).
- **The exit**: sell at the open after the first close above its `cap_long_exit_sma_sessions` (3) average since the
  fill session (its own close counts), or after `cap_long_max_hold_sessions` (5) sessions (`cap_long.exit_check`;
  reasons `cap_rebound` / `cap_time`).
- **Size**: each order 1/`cap_long_account_slices` (6) of the live account's equity at its limit, within the long
  side's cap (the longs' value at most the equity) and the margin room at Reg T's 50%; at most `cap_long_max_open` (3)
  of its longs open, its working orders counted (`live_account.size_cap`). The trade stamps `cap_account_shares` (what
  the broker orders), `cap_breadth`, `cap_close` / `cap_sma_trend` / `cap_rsi2` / `cap_dv20`, `cap_limit`,
  `cap_fill_kind`, `cap_order_qty`, `cap_fill_cap`.
- **Its capital gives way FIRST** (the backtest's capital quality 0.1, under the dip book's 0.13 and the crash book's
  0.2): a short, the crash book's buy or a DIP ORDER's fill short of room sells its latest longs WHOLE before any dip
  long (exit reason `cap_gave_way`; `_give_way_longs`, `fill_give_way_plan`); a dip order is sized as if its longs were
  sold (`size_long`; at IBKR it rests at that size, `fill_cap(free_cap=True)`); de-leveraging sells its longs first
  (`cap_delever`); a margin call liquidates them as IBKR does — the most recently opened positions first in 100-share
  lots until the account is back at its requirement (`cap_margin_call`; from 2026-10-21) — and cancels its working
  orders.
- **One book per name**: never a name another book holds (an order is `held` and retried), and no other book on its
  names.

## When it runs

- PRE-MARKET (04:00–09:30 ET ticks of a session day, the end-of-tick block): the day's breadth is computed and kept
  (`cap_long.prepare` → `breadth/<D−1>.json`, a few seconds: 300 cached whole-market files).
- THE OPENING TICK (within `cap_long_entry_window_minutes` of 09:30, the live path's second half right after the dip
  book's orders): the day's gate and signals (`signals/<D>.json`), the orders placed / armed, the open fills; the broker
  syncs at once.
- EVERY REGULAR-HOURS TICK: the exits at the start (after the crash book's), the touches after the dip book's
  (`monitor_cap_long_orders`), the lapses; the end-of-tick block repeats them. The broker rests each armed order as a
  DAY limit BUY sized to the room (`capw-<ticker>-<day>-<n>`), owned by the orphan sweep; a position bought by a
  resting order before the ledger links it is not drift.

## Where things are

- `cache/ml/cap_long/breadth/<S>.json` — the breadth at S's close: the top names (their dv20, close, 200-session
  average, RSI2, oversold flag), the oversold list, the window read.
- `cache/ml/cap_long/signals/<D>.json` — the day's gate (`breadth`, `theta`, `gate`) and, on a capitulation session,
  every signal with its numbers and limit (the candidates, stale and inactive names, the own-security block).
- `cache/ml/cap_long/orders/<D>.json` — each order's life (the dip book's schema: placed / armed / filled / unfunded /
  held / no_slot / account_full / lapsed / cancel_margin / split; its resting IBKR order).
- `cache/ml/cap_long/entries/<D>.jsonl`, `exits/<D>.jsonl` — fills; exits and the account's sales (`cap_gave_way`,
  `cap_delever`, `cap_margin_call`).

## Checks

- `python -m src.signals.cap_long --breadth [--day D]` — the breadth deciding D (nothing written).
- `python -m src.signals.cap_long --parity --days 60` — production's breadth against PREREG165's research series:
  2026-10-20, 59 of 60 sessions equal, every gate equal; the one difference (2026-09-28: WBD in the research's top 100)
  is a name delisted since — today's type table no longer lists it, the table of that day did.
- Problems ride the email's scorer banner (`cap_long.health`): no gate decided by the end of the opening window, a long
  held past its limit; a capitulation session is a note.
- Switches: `enable_cap_long` (off: no new orders, the working ones neither filled nor rested; open longs still leave
  on their rule); `enable_live_account` (off: the book places nothing).

## The evidence

- PREREG165 (registered 2026-10-20 14:26 ET; Amendments 1–2): the breadth on a fixed-size universe (the 100
  most-traded) is stable across eras, unlike a count of signals. At θ = 20%: 364 trades 2018–2026 (pooled), +0.683
  %/day (95% 0.23 … 1.27); UNTOUCHED 2006–2017 on a survivorship-free whole-market panel (Polygon's grouped bars,
  `prereg165/panel165.py`) 149 trades, +1.79 %/day (1.00 … 2.54). In the deployed configuration (RPD_IS, 20 paired
  draws a set) the give-way place at 1/6 × 3 (CBg6_20): +3.38 pp a year, margin-call draws 1 → 1, the worst drawdown
  unchanged, the weak market context (below SPY's trend in high volatility) +21.1 → +28.0% a year; the crash place
  (quality 0.2, not preemptible) lost on set A0. Its plateau check FAILED on the low side (θ 15%: +1.02 pp) — the
  registered θ* (30%) failed the bank's 30-month count, so 20% was chosen by Amendment 1 (exploratory).
- PREREG167 (with the dip book's meta-model cut and the crash book's after-hours exit): +7.19 ± 0.13 pp a year
  (66.5 → 73.7%), ADD; this book's own increment on top of the other two +3.57 pp (A0 −0.02, A +4.97, B +0.59); 2022
  carries much of it — a crisis buy (the user: acceptable for a buy). It lowers the account's per-trade return per day
  (0.745 → 0.704 alone): its trades are slower than the shorts', and it is judged by its own trades (+0.68 %/day, far
  above SPY's +0.038 %/day) with the account's growth as the guard.
- ITS CARD (2026-10-21; `reliability.SPEC` "cap", `bank.BENCH` SPY, `cache/ml/reliability/2026-10-21*`): the arm alone through the
  engine (at most 3 open) on the pooled history 135 trades in 40 months, +0.39 %/day (95% +0.036 … +0.85 against SPY's
  +0.038: significance missed by 0.002, every other check passed), gross win 67%, the worst trade −31% (March 2020);
  RE-JUDGED on its longest history with the untouched 2006–2017 trades above (`bank.LONG_TRADES`): 284 trades in 72
  months, +1.14 %/day (+0.59 … +1.70 against +0.030) — QUALIFIES; deflated Sharpe 0.73 over 25 trials. The live
  calibration scores it from the 2026-10-21 session against set B's reference (19 trades, +0.81 %/day).
- The simulator's default `Mimic()` reproduces PREREG167's ALL3 exactly (the book's 887 pieces; 60 of 60 accounts:
  growth, the daily NAV, the margin calls; 2026-10-20).

## Known gaps

- The engine's give-way for a dip fill follows production's (`Rules.fill_room_exact`, on with the book in `Mimic()`:
  the fill's own size decides, `live_account.fill_give_way_plan`); the slice-capped rule before it could sell a lot
  that added no share when the order's quantity exceeded its slice at the fill price — on PREREG167's 60 accounts the
  two are identical (it never happened in 2018-2026).
- Production places its orders after the dip book's open fills of the same tick (the engine places both books' orders
  first): a capitulation order the dip fills leave no room for is not placed that session (`account_full`), where the
  engine places it and its fill finds no room (`unfunded`); the fills' sizes agree.
- The breadth of a PAST session read today uses today's type table (a name delisted since is missing); live it reads
  the table of its day.
