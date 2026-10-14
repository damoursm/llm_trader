# The crash-rebound book (`crash_long`) — runbook

Live from the 2026-10-12 session (user 2026-10-11: "Crash book only", "Deploy it anyway", then after PREREG88:
"deploy the QQQ decisions, trade TQQQ to replace the QQQ decisions, trade QQQ"). Code `src/signals/crash_long.py`;
ledger steps `tracker.record_crash_long_trades` / `monitor_crash_long_positions`; sizing `live_account.size_crash` /
`crash_give_way_plan`; simulator `src/backtest/strategies/crash.py` (`Mimic(crash=True)`, the default). Evidence:
`memory/crash-book-leveraged-2026-10.md` (PREREG88), `memory/four-new-models-2026-10.md` (PREREG86, the rule),
`llm_trader_research/studies/prereg86/`, `prereg88/`.

## The rule — decided on QQQ, traded in TQQQ

- At the close of D−1, QQQ (`crash_long_symbol`, the DECISION symbol) below its 200-session average
  (`crash_long_trend_sessions`) with a 2-session RSI under 5 (`crash_long_rsi_max`) → buy TQQQ
  (`crash_long_trade_symbol`) at D's regular-hours open (the first tick within `crash_long_entry_window_minutes` of
  09:30).
- Sell TQQQ at the open after QQQ's first close above its 5-session average since the entry session
  (`crash_long_exit_sma_sessions`), or after 10 sessions held (`crash_long_max_hold_sessions`). Both decisions read
  QQQ's closes, never TQQQ's (the trade's `crash_decision_symbol`).
- One long at a time; 1/4 of the live account's equity in TQQQ (`crash_long_account_slices`) within the long side's
  cap and the margin room at the fund's requirement, `crash_long_margin_rate` 0.75 (FINRA's 3 × 25% for a 3x fund,
  initial = maintenance; never under Reg T's 50% / 25%). The dip book's latest longs are sold whole when that is what
  makes room (exit reason `dip_gave_way`); a margin call closes it (`crash_margin_call`); de-leveraging leaves it.
  Another book holding TQQQ (an ETF-arm short, say) settles the day's signal `held`.

## Where things are

- `cache/ml/crash_long/signals/<day>.json` — the day's decision (QQQ's numbers, `trade_symbol`; computed at the first
  opening-window tick; a store missing D−1's close is retried, never kept).
- `cache/ml/crash_long/entries/<day>.jsonl` — keyed by the decision symbol (`ticker` QQQ, `trade_symbol` TQQQ):
  `opened`, `holding`, `held`, `no_slot`, `account_full`, `account_too_small`, `no_price` (retried).
- `cache/ml/crash_long/exits/<day>.jsonl` — `crash_rebound`, `crash_time`, `crash_margin_call` (with
  `decision_symbol` and QQQ's close / exit average).
- The trade (ticker TQQQ) carries `crash_decision_symbol`, `crash_signal_day`, `crash_entry_day`, QQQ's
  `crash_close` / `crash_sma_trend` / `crash_rsi2` (a TQQQ split never rescales them; `tracker.SPLIT_PRICE_FIELDS`
  leaves them out), `crash_margin_rate`, `crash_account_shares` (what the broker orders),
  `crash_account_notional` / `_equity` / `_slices`, `crash_gave_way`, `crash_sessions_held`.

## Checks

- Today's numbers without writing the day's file:
  `python -c "from src.signals import crash_long as c, dip_long as d; t=d.today_et(); print(c.stats_at(c._daily(), d.prev_session(t)))"`.
- Problems ride the email's scorer banner (`crash_long.health`): no decision by the end of the window, a long held past
  its limit.
- Switches: `enable_crash_long` (off: no new longs; an open one still leaves on its rule); `crash_long_trade_symbol`
  (QQQ restores trading the decision symbol itself; then `crash_long_margin_rate` 0 for Reg T).

## The risk to keep in view

At 1/4 in TQQQ a crash trade carries the market exposure of three quarters of the account. Over 1988–2020 the worst
trade lost 36% (October 1987, synthetic TQQQ) — about 9% of the account in one trade; 2021–2026 holds no such day.
