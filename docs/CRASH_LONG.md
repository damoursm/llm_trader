# The crash-rebound book (`crash_long`) — runbook

Live from the 2026-10-12 session (user 2026-10-11: "Crash book only", "Deploy it anyway", then after PREREG88:
"deploy the QQQ decisions, trade TQQQ to replace the QQQ decisions, trade QQQ"). Code `src/signals/crash_long.py`;
ledger steps `tracker.record_crash_long_trades` / `monitor_crash_long_positions`; sizing `live_account.size_crash` /
`crash_give_way_plan`; simulator `src/backtest/strategies/crash.py` (`Mimic(crash=True)`, the default). Evidence:
`memory/crash-book-leveraged-2026-10.md` (PREREG88), `memory/four-new-models-2026-10.md` (PREREG86, the rule),
`llm_trader_research/studies/prereg86/`, `prereg88/`.

## The rule — decided on QQQ, traded in TQQQ

- At the close of D−1, QQQ (`crash_long_symbol`, the DECISION symbol) below its 250-session average (200 before the
  2026-10-20 session, PREREG159)
  (`crash_long_trend_sessions`) with a 2-session RSI under 5 (`crash_long_rsi_max`) → buy TQQQ
  (`crash_long_trade_symbol`) at D's regular-hours open (the first tick within `crash_long_entry_window_minutes` of
  09:30).
- Sell TQQQ at the open after QQQ's first close above its 3-session average since the entry session
  (`crash_long_exit_sma_sessions`), or after 5 sessions held (`crash_long_max_hold_sessions`) — 5 / 10 before the
  2026-10-20 session. Both decisions read
  QQQ's closes, never TQQQ's (the trade's `crash_decision_symbol`).
- THE AFTER-HOURS EXIT (PREREG166's XC; user 2026-10-20: "Deploy all three."; from the 2026-10-20 evening,
  `crash_long_pm_exit`): the same exit is taken at the first after-hours tick at/after 16:30 ET of the session whose
  close triggers it — QQQ's close of that session above its 3-session average, or the long's 5th session — instead of
  the next open (`crash_long.pm_exit_check`, `tracker.monitor_crash_long_pm_exit`, in the live path and the
  end-of-tick block of every tick from 16:30 to 20:00 ET). The session's own close is Polygon's last regular-hours
  30-minute bar of QQQ (the store holds the sessions before it until the next pre-open run), the store's construction;
  TQQQ is sold at the after-hours price fetched then (the broker's extended-hours LMT, `outsideRth`); the trade stamps
  `crash_pm_exit`. A tick that cannot read the close retries at the next one; every tick missed until 20:00 leaves the
  sale to the next open's exit, as before. The simulator: `Mimic.crash_pm_exit` (the default follows the switch;
  `mimic.pm_exit` on the crash book alone, the 16:00-16:30 bar's close).
- One long at a time; 1/3 of the live account's equity in TQQQ (`crash_long_account_slices`; 1/4 before the 2026-10-20
  session) within the long side's
  cap and the margin room at the fund's requirement, `crash_long_margin_rate` 0.75 (FINRA's 3 × 25% for a 3x fund,
  initial = maintenance; never under Reg T's 50% / 25%). The capitulation book's latest longs, then the dip book's, are
  sold whole when that is what makes room (exit reasons `cap_gave_way` / `dip_gave_way`; from the 2026-10-21 session);
  a margin call liquidates it as IBKR does — in 100-share lots when it is among the most recently opened positions,
  only until the account is back at its requirement (`crash_margin_call`; `tracker.monitor_live_account`, from
  2026-10-21) —; de-leveraging leaves it.
  Another book holding TQQQ (an ETF-arm short, say) settles the day's signal `held`; while this book holds TQQQ the
  selection short never shorts it (`already_open`).
- WHEN IT RUNS (regular-hours ticks, and the after-hours ticks 16:30-20:00 ET for its after-hours exit): the exit at
  the START of every tick (`monitor_crash_long_positions` at the open, `monitor_crash_long_pm_exit` after hours; TQQQ's
  price fetched now; no price = retried next tick); the buy in the opening window, after the dip book's order fills and
  before the shorts' entries (`record_crash_long_trades`); all again in the end-of-tick block. The book has no account
  of its own: with `enable_live_account` off it buys nothing.
- Measured (THE BANK, `docs/RESEARCH.md`): QUALIFIES on its longest history against SPY's average day — 118 trades
  1988-2026, +3.2% a trade against SPY's +0.14% over the same holds (clustered t 3.2); in the pooled 2018-2026 history
  only 17 trades. On the 1988-2020 history its exit average (5) is a ONE-POINT PEAK (3: +0.66, 5: +2.10, 7: +0.65,
  10: +0.80) and its RSI cut (5) lies just off that history's plateau, which is [3] alone (+2.45) — a reading
  disclosed outside PREREG97's registered rule (its Amendment 1), not a decision: the first parameters to revisit if the
  book is re-studied.

## Where things are

- `cache/ml/crash_long/signals/<day>.json` — the day's decision (QQQ's numbers, `trade_symbol`; computed at the first
  opening-window tick; a store missing D−1's close is retried, never kept).
- `cache/ml/crash_long/entries/<day>.jsonl` — keyed by the decision symbol (`ticker` QQQ, `trade_symbol` TQQQ):
  `opened`, `holding`, `held`, `no_slot`, `account_full`, `account_too_small` settle the day (no retry later in the
  window); only `no_price` (and a decision not yet computed) is retried.
- `cache/ml/crash_long/exits/<day>.jsonl` — `crash_rebound` (with QQQ's close and its exit average), `crash_time`
  (`decision_symbol` and the sessions held), `crash_margin_call` (the price and the return only).
- The trade (ticker TQQQ) carries `crash_decision_symbol`, `crash_signal_day`, `crash_entry_day`, QQQ's
  `crash_close` / `crash_sma_trend` / `crash_rsi2` (a TQQQ split never rescales them; `tracker.SPLIT_PRICE_FIELDS`
  leaves them out), `crash_margin_rate`, `crash_account_shares` (what the broker orders),
  `crash_account_notional` / `_equity` / `_slices`, `crash_gave_way`, `crash_sessions_held`.

## Checks

- Today's numbers without writing the day's file:
  `python -c "from src.signals import crash_long as c, dip_long as d; t=d.today_et(); print(c.stats_at(c._daily(), d.prev_session(t)))"`.
- Problems ride the email's scorer banner (`crash_long.health`): no decision by the end of the window, a long held past
  its limit.
- Switches: `enable_crash_long` (off: no new longs; an open one still leaves on its rule); `enable_live_account` (off:
  no crash buys either — the book is sized only from the live account; an open long still leaves on its rule);
  `crash_long_trade_symbol` (QQQ restores trading the decision symbol itself; then `crash_long_margin_rate` 0 for
  Reg T).

## The risk to keep in view

At 1/4 in TQQQ a crash trade carries the market exposure of three quarters of the account. Over 1988–2020 the worst
trade lost 36% (October 1987, synthetic TQQQ) — about 9% of the account in one trade; 2021–2026 holds no such day.

## Research beside it (not production)

* THE AFTER-HOURS EXIT (PREREG166's XC, 2026-10-20; the return-per-day review; DEPLOYED the same evening, the rule
  above): selling TQQQ at the 16:30 after-hours tick of the session whose close triggers the exit (instead of the next
  09:30 open) doubles the book's return per day on RPD_IS (2.27 -> 4.66 %/day: the shorter hold and, in 2018-2026, an
  average -1.76% overnight after the trigger close) and adds +1.83 pp a year (every set; the standard add rule's ADD;
  with the dip book's meta-model cut +3.62 pp, with the capitulation book too +7.19 pp, PREREG167). On 2010-2017 (16
  trades, TQQQ's real after-hours bars) its return per day is higher too (1.31 -> 1.52) but its average trade lower
  (-0.65 pp: the overnight averaged +0.59% there) — not confirmed per trade; the live record will tell. The default
  `Mimic()` reproduces PREREG166's XC crash pieces exactly.

* THE OTHER 3x BOOKS ON THE DEPLOYED CONFIGURATION (PREREG161, 2026-10-20): the bull-market index dips in TQQQ / UPRO,
  the turn of the month in TQQQ and TQQQ / UPRO overnight — all qualify for the bank under the market's bar, none adds
  to RPD_IS (-6.6 to +0.8 pp a year; the two closest, the turn of the month and the uptrend nights in TQQQ, under +0.5
  log points and carried by 2020). The crash book stays the account's only 3x book.
* LARGE-CAP CAPITULATION BREADTH (PREREG163 Amendment 1, exploratory): buying every $2B+ dip under its own trend on a
  session with at least 5 such signals adds +1.95 pp a year in a give-way place — the stock-level twin of this book's
  index capitulation (11 of this book's 15 decision sessions of 2018-2026 are among its 72 sessions with 5 or more such
  signals, 7 of those 11 in 2022) — but the gain is 2022's alone and the threshold a lone peak. Re-built on a
  fixed-size universe's BREADTH (PREREG165: the share of the 100 most-traded stocks oversold under their trend, at least
  20%) it held out of sample on 2006-2017 and is DEPLOYED from the 2026-10-21 session as THE CAPITULATION BOOK
  (`docs/CAP_LONG.md`): a separate book whose capital gives way before the dip book's, under this one's (quality 0.2) —
  this book's buy sells its latest longs first when room is short.
