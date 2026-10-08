# The dip long book — runbook

The second live strategy (from the 2026-10-09 open; user directive 2026-10-08: "Deploy the mega-cap dip
buying strategy to live production"). Code `src/signals/dip_long.py`, ledger steps in `src/performance/tracker.py`
(`record_dip_long_trades`, `monitor_dip_long_positions`), tests `tests/test_dip_long.py` and the dip guards in
`tests/test_no_lookahead.py`. The study: `memory/long-dip-megacaps-2026-10.md`, report
https://claude.ai/artifact/3DfPZz6KzJiW5r8znx3nVz.

## The rule

| | |
|---|---|
| Universe | Polygon type CS / ADRC, 20-session mean of close x regular-hours volume >= $1B (`dip_long_min_dollar_volume`) |
| Signal (close of D-1) | close above its 200-session average, 2-session RSI under 10 |
| Buy | at session D's open (the 09:30 tick), deepest dip (lowest RSI) first |
| Sell | at the open after the first close above the 5-session average since the entry, or after 10 sessions |
| Size | 1/10 of the book's own $10,000 cash account (equity), at most its cash, 10 longs at most |

The daily bars are built from the deep 30-minute store exactly as the study built them (`daily_bars`), and
nothing of session D is read for D's decisions.

## A normal day

1. 08:30 the pre-open run and 09:00 the selection short's prepare extend the deep 30-minute store through the
   previous session, and cache Polygon's whole-market daily bars the prefilter reads.
2. 09:30 tick, start: `monitor_dip_long_positions` sells every long whose exit is due (log `[dip_long] SELL`).
3. 09:30 tick, live path: `record_dip_long_trades` computes the day's signals once (`signals/<day>.json`, ~2 s)
   and buys them (log `[dip_long] BUY`), then a broker sync sends the orders before the scorer wait
   (`[live] dip_long orders synced`).
4. Until 10:30 a later tick retries a signal that had no live price; after that the day's unsettled signals lapse.

## Checks

```bash
python -m src.signals.dip_long --status                       # the account and the open longs
python -m src.signals.dip_long --signals --day 2026-10-09 --no-write   # what the book would buy (prints only)
```

Files under `cache/ml/dip_long/`: `signals/<day>.json` (the computation: `signals`, the closest misses in
`next`, `stale` = names that traded on D-1 but the store lacks, `inactive` = names that did not trade),
`entries/<day>.jsonl` (each signal's outcome), `exits/<day>.jsonl` (each sale with the close and average that
fired it).

## When something is wrong

* **Scorer banner "dip long: no signal computation for today's open"** — the 09:30-10:30 ticks never reached
  the entry step (scheduler down, an exception: grep `[dip_long]` in the day's log). The day's signals lapse;
  nothing to repair. Exits are not affected (they are judged at every regular-hours tick).
* **"names had no close for D-1 in the store"** — the pre-open extension failed for those names; the entry
  step tried to extend them itself. Run `python -m src.data.intraday_store --extend --tickers A,B` out of
  hours if it repeats.
* **"held past 10 sessions"** — an exit did not go out (no live price for days?): check the trade's
  `dip_sessions_held` and the exits journal.
* **To stop new longs**: `ENABLE_DIP_LONG=false` in `.env` and restart the scheduler. The open longs still
  leave on their rule (`monitor_dip_long_positions` runs whatever the switch says).
