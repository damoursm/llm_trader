# The fear book (`fear_long`) — runbook

Live from the 2026-10-28 session (the user, 2026-10-27, after PREREG224-226: "Deploy to production the $1B, 1/6 each,
4 slots."; the code deployed between ticks of the 2026-10-27 session — `fear_long_first_session` 2026-10-28: before it
no fetch, no order, no banner check of the day). Code `src/signals/fear_long.py`; ledger steps `tracker.record_fear_long_trades` (→ `_place_fear_orders`) /
`monitor_fear_long_orders` / `monitor_fear_long_positions` / `cancel_fear_orders`; sizing `live_account.size_fear` and
the give-way order (`_give_way_longs`, `_delever_longs`); the one-book rule with the dip book `books.SHARED_NAMES`
(`tracker._dip_other_book` / `_fear_other_book`, the broker's `_resting_books` ``own``); the broker's resting orders
`reconcile._sync_book_orders` (refs `fearw-…`); simulator `src/backtest/strategies/fear.py` (`Mimic(fear=True)`, the
default while `enable_fear_long` is on). Tests `tests/test_fear_long.py` + `tests/test_no_lookahead.py`. Evidence:
`memory/fear-book-capital-prereg223-2026-10.md`, `memory/fear-floors-prereg224-2026-10.md`,
`memory/fear-expansion-prereg225-2026-10.md`, `memory/fear-sizing-prereg226-2026-10.md`;
`llm_trader_research/studies/prereg219/` … `prereg226/`, the deploy's parity `studies/deploy_fear/`.

## The rule

- **FEAR** (`fear_long.ivp_at`, PREREG219's percentile exactly): for a signal whose signal session is S (the close the
  dip rule reads), IVP = the percentile of the name's 30-day implied volatility closing S — IBKR's
  OPTION_IMPLIED_VOLATILITY daily bars — among its closes of the `fear_long_ivp_window` (252) sessions through S (at
  least `fear_long_ivp_min_values` (126) values; the last bar within 5 days of S; the window no longer than 252 × 1.6
  sessions in calendar time; ties count half). FEAR = IVP ≥ `fear_long_ivp_min` (0.8).
- **The signals** (`fear_long.compute_signals`): every signal of the dip book's day (`dip_long.signals_for`: its kept
  pick, its not-top and its meta-model-filtered signals alike — a common stock / ADR that closed above its 200-session
  average with a 2-session RSI under 10 trading $1B+ a day) with dv20 ≥ `fear_long_min_dollar_volume` ($1B) that is
  FEAR and is NOT the dip book's own order of the day (its daily top pick, `fear_long.dip_pick`). Each signal's fate is
  journaled in `signals/<D>.json`: `fear` (an order), `calm`, `no_iv` (no value in the day's table: never ordered),
  `dip_pick`, `below_floor`. Computed once, when the day's table is final (done, stopped, or its window over) and the
  dip book has made its own order (`fear_long.dip_ready`).
- **The order's life** is the capitulation book's (`docs/CAP_LONG.md`): a BUY LIMIT ORDER at the close minus
  `fear_long_limit_atr` (½) × the 14-session ATR, alive `fear_long_limit_sessions` (3) sessions, placed at each window
  session's opening tick IVP highest first (ties: the deepest RSI2), filled at the open when the session opened at/under
  the limit, else at the end of the first completed 30-minute bar whose low reaches it, at most what the room allows
  (`unfunded` otherwise, retried the next session), lapsed after its last session's 16:00, rescaled by a split.
- **The exit**: sell at the open after the first close above its `fear_long_exit_sma_sessions` (3) average since the
  fill session, or after `fear_long_max_hold_sessions` (5) sessions (`fear_long.exit_check`; `fear_rebound` /
  `fear_time`).
- **Size**: each order 1/`fear_long_account_slices` (6) of the live account's equity at its limit, within the long
  side's cap and the margin room at Reg T's 50%; at most `fear_long_max_open` (4) of its longs open, its working orders
  counted (`live_account.size_fear`). The trade stamps `fear_account_shares` (what the broker orders), `fear_ivp` /
  `fear_iv`, `fear_close` / `fear_sma_trend` / `fear_rsi2` / `fear_dv20`, `fear_limit`, `fear_fill_kind`,
  `fear_order_qty`, `fear_fill_cap`.
- **Its capital gives way FIRST** (the backtest's quality 0.04, under the insider book's 0.05): a short, the crash
  book's buy, a dip, capitulation or insider entry short of room sells its latest longs whole before any other long
  (`fear_gave_way`); every other book's order is sized as if they were sold; de-leveraging sells them with the
  capitulation and insider books' longs, the latest first (`fear_delever`); a margin call liquidates them as IBKR does
  (`fear_margin_call`) and cancels its working orders.
- **One book with the dip book** (PREREG223's M1): the dip book may buy a name the fear book holds and the fear book a
  name the dip book holds — two long trades in one symbol, netted at IBKR, each ordering and closing its own shares
  (`reconcile._owned_by_open`); the capitulation, insider and crash books and the shorts still keep one book per name.
  Up to ~71% of the equity can then sit in one name (the dip book's 1/2 and the fear book's 1/6).

## The implied volatility

- **The store** (`fear_long_iv_dir`, `data/ibkr_iv/`): one gzip JSON file a name, rows [date, open, high, low, close]
  of IBKR's daily implied volatility. Seeded on 2026-10-27 with PREREG219's history (223 names, 2006-11 → 2026-10-26;
  `python -m src.signals.fear_long --seed <research store>`); every fetch merges into it (a fetched bar replaces a
  stored one of its date: the store holds IBKR's latest values, as the seed does). **Never delete** — IBKR serves
  listed names only and the simulator reads it.
- **Its journal** (P7 of the parity audit of 2026-10-27; `fear_long.merge_rows`): before a fetch merges an answer,
  every stored bar the answer CHANGES is written to `data/ibkr_iv/revisions/<D>.jsonl` — ``revised`` with the four
  values the store held and IBKR's new ones, ``absent`` for a stored bar inside the answer's span that IBKR no longer
  serves (kept in the store) — and the day's table counts them per name (``answered`` / ``revised`` / ``absent``, the
  day's ``revisions``). The EOD live calibration reads it (``iv_revisions``: the bars re-sent, revised and no longer
  served since the book's first session, the largest close move, the FEAR calls a revision flips — the percentile at
  the revised bar's own session with the replaced close against the store's), with a note on the session's
  revisions and on every flipped call. The 20:04 ET fetch test of 2026-10-27 re-read a month of 4 names: no bar
  revised. The simulator never reads the live store while it builds the bank: each card build freezes a copy of it
  (`docs/BACKTEST.md`, "The implied-volatility store").
- **The pre-market fetch** (`fear_long.prepare` → `fetch`): from `fear_long_iv_start_et` (08:30) to
  `fear_long_iv_stop_et` (09:27) ET of a session, a tick launches a DETACHED `python -m src.signals.fear_long --fetch
  --day D` (log `logs/fear_iv_<D>.log`, lock `cache/ml/fear_long/fetch.lock`; never waited for; not launched again
  once the day's table is done). It computes the day's candidates — the dip rule's names at S's close from the deep
  store, loosely (dv20 ≥ 90% of the floor, the close above 98% of its trend, RSI2 under the rule's + 3: a superset of
  the 09:30 signals; names whose store does not reach S yet are waited for, re-checked every minute) — and asks IBKR
  for each one, the deepest dip first: 2 years for a name the store lacks, a month to extend one it holds, on its own
  client (`ibkr_client_id` + `fear_long_iv_client_offset` 68, read-only), one request every
  `fear_long_iv_gap_seconds` (12), each bounded by `fear_long_iv_timeout_seconds` (60), held from 2 minutes before to 3
  after each :00 / :30. Each answer: merged into the store, its IVP at S written to `ivp/<D>.json` at once (``ok`` /
  ``no_contract`` / ``no_data`` / ``no_value`` / ``error``). It stops at the window's end, on a lost connection or
  after three timeouts in a row (``stopped``).
- **IBKR's answers**: the fetch hooks IBKR's error stream (PREREG219's fetch): an empty answer with IBKR's own error
  (162, no data) is ``no_data`` and not asked again that day; a timeout — or an empty answer without a word from IBKR
  — is ``error``, asked again while the window lasts (three in a row stop the fetch).
- **Fail-safe**: a signal without a value is not ordered; no table at all = no fear order that day (a scorer-banner
  problem after the opening window). A signal the fetch failed or never asked is a problem; one IBKR has no implied
  volatility for (``no_contract`` / ``no_data`` / ``no_value``: too short a history) a note. The fear book never
  guesses an implied volatility.

## When it runs

- PRE-MARKET (08:30–09:27 ET): the fetch launched by the live path's first half and the end-of-tick block.
- THE OPENING TICK (within `fear_long_entry_window_minutes` of 09:30, the live path's second half right after the dip
  and capitulation books' orders): the day's fear signals, the orders placed / armed, the open fills; the broker syncs
  at once.
- EVERY REGULAR-HOURS TICK: the exits at the start (after the insider book's), the touches after the capitulation
  book's; the end-of-tick block repeats them.

## Files (`fear_long_dir` = `cache/ml/fear_long/`)

`ivp/<D>.json` (the day's implied-volatility table: the candidates, each name's answer and IVP — with the bars it
answered, added, revised and no longer served —, the stale names, the status, the day's ``revisions``),
`signals/<D>.json` (the fear signals and every dip signal's fate), `orders/<D>.json` (the dip book's schema),
`entries/<D>.jsonl`, `exits/<D>.jsonl`, `fetch.lock`; beside the store, `data/ibkr_iv/revisions/<D>.jsonl` (the
journal: never delete).

## Checks

- `python -m src.signals.fear_long --status` — the scorer banner's problems and notes.
- `python -m src.signals.fear_long --ivp AAPL --day 2026-10-28` — a name's IVP at the close before the day, from the
  store.
- `python -m src.signals.fear_long --signals --day D` — the day's fear signals PREVIEWED (nothing written); `--write`
  writes `signals/<D>.json`, which the opening tick decides once — never before the day's table is final.
- `python -m src.signals.fear_long --fetch --day D --until HH:MM` — a fetch by hand (only in a window where the gateway
  may take a request: never 23:40–00:50 ET, away from the :00 / :30 slots).
- The second account never trades it (`second_account_books` holds no ``fear``).

## Evidence (PREREG221 / 223 / 224 / 225 / 226)

The fearful $1B+ dips at 4 open: 416 trades in 77 months on 2018-2026, +0.490 %/day (95% +0.289 … +0.718 against SPY's
+0.038); the $2B+ half's fear premium holds on 2007-2017 (+1.51 pp a trade over the calm dips, p 0.001), the $1-2B
half's only on 2018-2026 (+0.23 pp on 2007-2017, p 0.39). In the live account (40 draws a set, PREREG225 / 226)
88.21 → 93.04% a year (+4.83 ± 0.27 pp, every set, without its best year +1.98 log points), the worst 12 months +8.0 →
+7.9%, the worst draw's drawdown 34.7 → 36.3%, the largest one-name long exposure 71% of equity. Bigger fear positions
add nothing (PREREG226); the dip book itself stays at its $2B floor (PREREG224: at $1B −29 pp a year).
