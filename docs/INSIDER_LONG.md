# The insider book (`insider_long`) — runbook

Live from the 2026-10-26 session (the user, 2026-10-24, on PREREG204's report: "Ok. Deploy the Opportunistic, SPY 20%+
down to the live combine."). Code `src/signals/insider_long.py`; ledger steps `tracker.record_insider_long_trades` /
`monitor_insider_long_positions`; sizing `live_account.size_insider` and the give-way order (`_give_way_longs`,
`fill_give_way_plan`, `_delever_longs`); simulator `src/backtest/strategies/insider.py` (`Mimic(ins=True)`, the default
while `enable_insider_long` is on; `Mimic(**mimic.PRE205)` without). Tests `tests/test_insider_long.py` +
`tests/test_no_lookahead.py`. Evidence: `memory/insider-books-account-prereg203-2026-10.md`, `memory/capital-place-prereg204-2026-10.md`,
`llm_trader_research/studies/prereg201/` – `prereg205/`.

## The rule (PREREG203's O1_20)

- **The gate**: SPY's close at D−1 at least `insider_long_spy_drawdown` (20%) under its highest close of the
  `insider_long_spy_high_sessions` (252) sessions through D−1 (at least `insider_long_spy_min_sessions`, 200, read).
  The closes are Polygon's whole-market daily bars (`sel_pressure.grouped_ohlc`'s cache — SPY never split, so they are
  the official closes, equal close for close to the OHLCV cache's the study read). A session missing from the cache is
  an error, never a guess: no decision that day (retried each tick; a scorer-banner problem after the window). The
  book is DORMANT outside a deep sell-off: 2018–2026 held 76 gate sessions (2018-12-26, 2020-03-13 – 04-08, 2022-06-14 –
  2023-01-04 with gaps); 2008-09 held 257.
- **The events**: the deep store's insider tables — the SEC's quarterly bulk set (`form345.parquet`) and its live tail
  read from EDGAR (`form345_live.parquet`, fetched through yesterday by the 08:30 pre-open run and the nightly refresh).
  Every Form 4 / 4/A row of an officer or director with code P (acquired, a price above 0), a non-derivative title (no
  option, warrant, right, note, preferred, unit, debenture, bond), a well-formed issuer symbol (BRK-B / BRK/B → BRK.B),
  a filing and a trade date, shares above 0; a transaction reported twice (an amendment repeating its original) counted
  once, at its first report in the files' order; only rows filed BEFORE D read. An EVENT is one company's purchases
  filed on one date (their shares × price summed: ``notional``).
- **The classes** (Cohen, Malloy & Pomorski, point in time): an insider who traded (bought or sold, any company) in each
  of the three calendar years before the purchase's trade year — by the filings made before the purchase's — is
  ROUTINE when one of those trades fell in the purchase's month in all three years, OPPORTUNISTIC otherwise; without
  trades in all three years, neither. An event qualifies when at least one of its rows is opportunistic and none
  routine.
- **The filing window**: the events whose company's first session after the filing is D — filing dates from the
  session before D through D's eve (a Monday reads Friday's, Saturday's and Sunday's; after Good Friday, Thursday's and
  Friday's). The pre-open run must have fetched the window's filing days (`insider_long.filings_fresh`: each weekday
  covered by the bulk table or the live tail, a missing one excused as EDGAR's holiday when a later day's rows exist or
  it is a federal holiday); else, the gate on, no decision is kept and the step retries.
- **The stock**: from the whole-market daily bars as traded (the 252 sessions before D): its run of bars since its last
  gap of more than 30 calendar days holds at least `insider_long_min_history` (61) bars (and its last bar is at most 30
  days before D), its close at D−1 at least `insider_long_min_price` ($5), the mean close × volume of its last 20 bars
  at least `insider_long_min_dollar_volume` ($5M).
- **The trade**: bought at D's open (the opening tick's live price), the largest purchase first; sold at the open after
  `insider_long_hold_sessions` (60) sessions held — the entry session the first (`insider_long.exit_check`, reason
  `ins_time`). The study sold at the 60th session's close: production sells one overnight later, the other long books'
  time exit (on the study's 301 entries 2018–2026: +26.11% against +26.67% gross a trade).
- **Size**: each long 1/`insider_long_account_slices` (10) of the live account's equity within the long side's cap (the
  longs' value at most the equity) and the margin room at Reg T's 50%, at most 1% of the stock's dollar volume, at most
  `insider_long_max_open` (5) open (`live_account.size_insider`; outcomes `no_slot` / `account_full` /
  `account_too_small` / `volume_cap`, settled for the day). The trade stamps `ins_account_shares` (what the broker
  orders), `ins_filing_date`, `ins_notional`, `ins_buyers`, `ins_n_opp`, `ins_close`, `ins_dv20`, `ins_rank`,
  `ins_spy_drawdown`. A second event of a name the book holds opens another long (the study's account stacked them).
- **Its capital gives way FIRST** (the backtest's quality 0.05, under the capitulation book's 0.1 and the dip book's
  0.13): a short, the crash book's buy, a dip or a capitulation order's fill short of room sells its latest longs WHOLE
  before any other book's (`ins_gave_way`); the dip and capitulation orders are sized as if they were sold (at IBKR
  they rest at that size); de-leveraging sells them together with the capitulation book's longs, the latest first
  (`ins_delever`; the engine's ``delever_order`` 0 for both); a margin call liquidates them as IBKR does — the most
  recently opened positions first in 100-share lots (`ins_margin_call`).
- **One book per name**: never a name another book holds (`held`), and no other book on its names.

## When it runs

- PRE-MARKET from 09:00 ET (the end-of-tick block): the day's decision computed and kept (`insider_long.prepare` →
  `signals/<D>.json`: ~3 s to read SPY's 252 closes; on a gate session the events and the stocks' numbers, ~5 s).
- THE OPENING TICK (within `insider_long_entry_window_minutes` of 09:30, the live path's first half right after the
  crash book's buy): the buys; a missing price is retried at the next tick within the window. The broker syncs at once.
- EVERY REGULAR-HOURS TICK: the exits at the start (after the capitulation book's); the end-of-tick block repeats the
  steps.

## Where things are

- `cache/ml/insider_long/signals/<D>.json` — the day's gate (SPY's close, its high and session, the drawdown) and, on a
  gate session, the filing window and its freshness, the events, the signals in rank order with their numbers, the
  rejected names by reason.
- `cache/ml/insider_long/entries/<D>.jsonl`, `exits/<D>.jsonl` — every signal the entry step settled (opened / held /
  no_slot / account_full / account_too_small / volume_cap; no_price retried); exits and the account's sales
  (`ins_time`, `ins_gave_way`, `ins_delever`, `ins_margin_call`).

## Checks

- `python -m src.signals.insider_long --gate [--day D]` — the gate deciding D (nothing written).
- `python -m src.signals.insider_long --signals --day 2020-03-16 --no-write` — a past gate session: the gate on (SPY
  20.4% under its high), 260 events, 31 signals = PREREG203's 31 pieces of that session.
- `python -m src.signals.insider_long --parity [--lo 2018-01-02 --hi 2026-09-24]` — production's decisions against
  PREREG203's O1_20 entries on every gate session (fetches the whole-market bars it lacks).
- Problems ride the email's scorer banner (`insider_long.health`): no decision by the end of the opening window, a long
  held past its limit; a gate session is a note.
- Switches: `enable_insider_long` (off: no buy; open longs still leave on their rule); `enable_live_account` (off: the
  book buys nothing). The second account never trades it unless its books name `ins`.

## The evidence

- PREREG201 – 204 (`memory/insider-purchases-prereg201-2026-10.md`, `insider-selloff-prereg202-2026-10.md`,
  `insider-books-account-prereg203-2026-10.md`, `capital-place-prereg204-2026-10.md`): the opportunistic insiders' edge faded after 2008
  except in deep sell-offs; with SPY 20%+ under its high the rule's 640 signals 2007–2026 (23 entry months) average
  +24.0% over 60 sessions against SPY's +15.1% over the same days (+8.9 pp, 95% +1.6 … +14.0) — a 3x S&P fund +45.3%;
  in the live account as the lowest-quality book +3.68 pp a year on 40 draws a set (the add rule's ADD, 2020 and 2022
  carrying it); its place chosen out of sample (giving way first) held out +0.84 log points a year (4 of 9 years: a rule
  that trades in three years cannot win five). The arm FAILS the bank's checks (PREREG205 [R3]: 359 trades in 18 months with 2007–2017, one year carrying it; deflated Sharpe 0.00 on its months over 45 trials) — deployed
  as the user's call.
- PREREG205 (the deployed construction): see `docs/RESEARCH.md` and `llm_trader_research/studies/prereg205/`.

## Known gaps

- The stocks outside the deep 30-minute store have no off-grid marks in the simulator (daily closes, lows and opens
  only) and no dividends; production marks them at every tick.
- A filing day the pre-open run did not fetch is not traded that session (no inline fetch: a full day of Form 4s takes
  minutes of EDGAR requests).
- The research's de-duplication saw every filing at once; production sees only those filed before the session (an
  amendment filed later can never move an earlier event — the parity on 2018–2026 is exact).
