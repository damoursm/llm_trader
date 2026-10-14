# The ETF dual-momentum book — runbook

Built 2026-10-08 (user: "Improve the ETF trading strategy focussing only on ETFs alone ... Also consider leveraged
ETFs ... Don't stop until you have a working, proven algo that can trade ETFs"). **OFF until the user orders the
deployment** (`enable_etf_dm=False`). Code `src/signals/etf_dm.py`, ledger step `tracker.rebalance_etf_dm`, tests
`tests/test_etf_dm.py` and the ETF guards in `tests/test_no_lookahead.py`. The study: PREREG46-51 in
`llm_trader_research/etfalgo/`, `memory/etf-dual-momentum-2026-10.md`, report
https://claude.ai/artifact/VVxQwuoEKnek7ipjN3ruxp.

## The rule (PREREG49's R4-DM1, Antonacci's dual momentum)

| | |
|---|---|
| Funds | QQQ, SPY, IWM, TLT, GLD (`etf_dm_universe`) — 1x funds; leveraged funds measured and rejected (78-94% drawdowns) |
| Score (each month's last close) | the mean of the fund's 3-, 6- and 12-month price returns (63 / 126 / 252 sessions; 253 closes needed) |
| Hold | the top fund for the next month if its 12-month return beats the 3-month T-bill's over the same 12 months (FRED DTB3 as published by that close); otherwise cash |
| Trade | in the opening window (`etf_dm_entry_window_minutes`, 60) of the month's first session: sell the old fund, buy the new one; an unchanged pick is kept; a missed session carries the month's decision to the next one |
| Size | the book's OWN simulated cash account, `etf_dm_account_initial` $10,000, no deposits; the whole account in the fund (whole shares) |

The closes are the deep 30-minute store's regular-hours sessions (`dip_long.daily_bars`), cut at the month-end
session. Parity: on every month 2022-02 .. 2026-10 (57 months; the store starts 2021-01) the live code picks exactly
what the research picked.

## Evidence (both metrics; return per day is low by construction — one trade lasts ~100 days)

| Period | Growth per year | S&P 500 with dividends | Worst drawdown (vs S&P) | Return per day |
|---|---|---|---|---|
| 1988-01 .. 2004-06, untouched test (PREREG51: PASS) | +15.5% | +12.1% | 46% (47%) | 0.063% |
| 2004-07 .. 2010-02 | +5.7% | +0.2% | 28% (55%) | 0.020% |
| 2010-02 .. 2017-12 | +6.5% | +14.5% | 29% (19%) | 0.022% |
| 2018-01 .. 2026-10 | +13.5% | +14.7% | 25% (34%) | 0.049% |
| 1988-01 .. 2026-10, the whole record | +11.7% | +11.3% | 46% (55%) | — |

PREREG51's 95% interval of the log growth: +0.046 .. +0.234 a year; both halves positive; no margin call. It trades
~3 times a year and holds a fund 85-95% of the time. It is a defensive rotation: it beat the market by leaving equities
in the 2000-02 and 2008 bear markets (2008: +21.5% vs -36.8%) and trailed it through most of the 2009-2024 bull market.

## A normal month

1. Before the month's first session, the 08:30 pre-open run extends the deep 30-minute store through the previous
   session (the month-end close).
2. First session, 09:30 tick, live path: `rebalance_etf_dm` computes the month's decision once
   (`decisions/<YYYY-MM>.json`, log `[etf_dm] <month>: hold X` or `cash`), sells the held fund when it is not the
   pick (log `[etf_dm] SELL`), buys the pick with the whole account (log `[etf_dm] BUY`), and the broker syncs right
   away.
3. The rest of the month: every session's opening window re-checks that the book holds the pick (a sale or a buy
   that had no price, a session the scheduler missed); otherwise nothing.

## Checks

```bash
python -m src.signals.etf_dm --decide --no-write          # this month's scores and pick (prints only)
python -m src.signals.etf_dm --decide --day 2026-11-02 --no-write
python -m src.signals.etf_dm --status                     # the account and the holding
```

Files under `cache/ml/etf_dm/`: `decisions/<YYYY-MM>.json` (each fund's 3/6/12-month returns and score, the top
fund, its 12 months against the T-bill's, the pick; `stale` = funds whose month-end close the store lacked),
`rebalances/<day>.jsonl` (each step: `sold`, `bought`, `no_price`, `held` = another book holds the fund,
`no_cash`, `account_too_small`).

## When something is wrong

* **Scorer banner "ETF dual momentum: no decision for <month>"** — the opening-window ticks never reached the step
  (scheduler down, an exception: grep `[etf_dm]` in the day's log). The next session's window retries.
* **"holds X but <month>'s pick is Y"** — a rotation did not go out (no live price, the fund held by another book,
  a broker refusal): read `rebalances/<day>.jsonl` and the trade's broker legs.
* **A fund `stale` in the decision** — the store lacked its month-end close and the step's own extension failed;
  the decision scored the others. Run `python -m src.data.intraday_store --extend --tickers QQQ,SPY,IWM,TLT,GLD`
  out of hours, delete the month's decision file and let the next window recompute it.
* **To deploy**: `ENABLE_ETF_DM=true` in `.env`, then restart the scheduler (`scripts/restart_all.ps1`, or kill the
  `--schedule` child just before a slot). The first opening window after that buys the current month's pick.
* **To stop new buys**: `ENABLE_ETF_DM=false` and restart. A held fund is still SOLD when a later month's pick
  differs (the rule's sale runs whatever the switch says); to leave at once, close the trade by hand.
