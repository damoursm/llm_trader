# The dip long book — runbook

**From the 2026-10-20 session — PREREG159's RPD_IS (user 2026-10-20: "Can you deploy the config of the 'Held out: return per day, one at a time' to the live prod?") and THE FRESH START (every live position closed at the 09:30 tick, the live account restarting at $50,000; CLAUDE.md "THE LIVE ACCOUNT").** Each long is 1/2 of the equity with at most 2 open (`DIP_LONG_ACCOUNT_SLICES` 2), its limit order lives 3 sessions (`DIP_LONG_LIMIT_SESSIONS`), and it sells at the open after the first close above its 3-session average (`DIP_LONG_EXIT_SMA_SESSIONS`) or after 5 sessions (`DIP_LONG_MAX_HOLD_SESSIONS`); de-leveraging runs at 70% → 50% of the equity. Where a value below differs, this line wins.

**From the 2026-10-26 session — PREREG209's KD+ (user 2026-10-25: "Build KD+ for production and deploy it.").** THE WALLSTREETBETS PREFERENCE (`ENABLE_DIP_LONG_WSB_PREFER` true; the section below): when any of the day's kept $2B+ signals was named in 5+ r/wallstreetbets titles in the 120 hours before 09:00 ET, the pick is the best-ranked of those flagged signals; else the pick as before.

**From the 2026-10-21 session — PREREG166 / 167 (user 2026-10-20: "Deploy all three.").** THE META-MODEL'S CUT (RP33, `ENABLE_DIP_LONG_FILTER` true) drops each day's signals scoring under its cut BEFORE the daily top pick, which picks among the kept $2B+ signals (the filter section below). THE CAPITULATION BOOK (`docs/CAP_LONG.md`) shares the account: its longs give way to this book's fills — a dip order is sized as if they were sold and its fill sells them, the latest first, whole (`cap_gave_way`) — and a name it holds is another book's (`held`).

The second live strategy (from the 2026-10-09 open; user directive 2026-10-08: "Deploy the mega-cap dip
buying strategy to live production"). Since the 2026-10-12 session it enters through a ½-ATR LIMIT ORDER (user
directive 2026-10-09: "deploy the Order holds nothing, fills into the room left for the 1/2 ATR limit in live for the
dip buying model. Remember that the simulator needs to mimic production as much as possible."). Code
`src/signals/dip_long.py` ("THE LIMIT ENTRY"), ledger steps in `src/performance/tracker.py` (`record_dip_long_trades`
→ `_place_dip_orders`, `monitor_dip_long_orders`, `monitor_dip_long_positions`, `cancel_dip_orders`), the broker's
resting orders in `src/broker/reconcile.py` (`_sync_dip_orders`), tests `tests/test_dip_long.py`,
`tests/test_dip_limit.py` and the dip guards in `tests/test_no_lookahead.py`. The studies:
`memory/long-dip-megacaps-2026-10.md` (the rule), `memory/dip-delayed-entry-2026-10.md` (the limit entry,
PREREG59-61), report https://claude.ai/artifact/3eY5F8UWDZTg2tpEqcpGz6 (sections 8-10). From the 2026-10-16 session only
each day's TOP-SCORED signal becomes an order, at a 1/6 slice (THE DAILY TOP PICK below; PREREG116's Q1, user
2026-10-15: "Deploy Q1"; tests `tests/test_dip_top_pick.py`). From the 2026-10-19 session a dip that is its company's
own earnings reaction ranks after the day's other signals (THE EARNINGS-DIP DEMOTION below; PREREG137's DEMOTE_E, user
2026-10-17: "Fully deploy the earnings dip demotion in live"; `src/signals/dip_earnings.py`, tests
`tests/test_dip_earnings.py`). From the same session each name's history is read from its OWN security — a reused
ticker's older security cut, a renamed company's earlier tickers joined (THE OWN-SECURITY SERIES below; user
2026-10-18: "Yes please apply the fix"; `src/signals/dip_security.py`, tests `tests/test_dip_security.py`).

## The rule

| | |
|---|---|
| Universe | Polygon type CS / ADRC, 20-session mean of close x regular-hours volume >= $1B (`dip_long_min_dollar_volume`); each name once in the project's form — the universe file lists Berkshire as BRK-B AND BRK.B, the candidates keep BRK-B (PREREG64: it was bought twice in the simulator's book) |
| Signal (close of D-1) | close above its 200-session average, 2-session RSI under 10 |
| Order | a BUY limit at that close minus `dip_long_limit_atr` (0.5) x its 14-session average true range, floored to the cent (`dip_long.limit_price`), alive sessions D .. D+2 (`dip_long_limit_sessions` 3 from the 2026-10-20 session; D .. D+4 before); `dip_long_entry_mode=open` restores the original buy at D's open |
| Ordered | only the day's TOP PICK (`dip_long_daily_top` 1) AMONG THE SIGNALS TRADING $2B+ A DAY (`dip_long_order_min_dollar_volume`, from the 2026-10-19 session: THE $2B FLOOR, below): the best of them by the meta-model's score (below) — an earnings dip after every other signal (THE EARNINGS-DIP DEMOTION, below), from the 2026-10-26 session a WallStreetBets-flagged signal before every unflagged one (THE WALLSTREETBETS PREFERENCE, below) —, the others journaled `not_top` (under the floor: `below_order_floor`); `dip_long_daily_top=0` orders every signal (the rule to 2026-10-15; no demotion, no floor then) |
| Placed | at each window session's opening tick, best-ranked first (`dip_long.rank_key`: by score; the deepest dip first with the rule off); the quantity is fixed when first placed: 1/3 (`dip_long_account_slices`, from the 2026-10-19 session; 1/6 from 2026-10-16, 1/10 before) of THE LIVE ACCOUNT's equity at the limit price, within its margin room at Reg T's 50% and the longs' value at most the equity; working orders count toward the 3 dip slots and RESERVE NOTHING; another book's name, no slot, no room: tried again at the next session's opening tick; an order of a day before the daily top pick is never placed (`superseded`) |
| Filled | at the session's OPEN when it opens at/under the limit — only for an order armed by 09:35 ET (`dip_long.placed_on_time`; one placed later judges only the bars that END after it) —, else at the END of the first completed 30-minute regular-hours bar whose low reaches it (Polygon's bars); AT MOST what the live account's room allows at that tick (`live_account.fill_cap`) — the rest is not bought; no room: nothing, and the order waits for the next session (`unfunded`) |
| Lapses | after 16:00 of its last session; a margin call cancels the working orders (placed again at a later opening tick) |
| Splits | a split executed after the order's day rescales a pending or working order like a position, from 04:00 ET of its date (`dip_long.apply_order_splits`: close, ATR, trend average and limit x the factor, the limit floored to the cent, the quantity / it in whole shares; event `split`); an open long is rescaled by the ledger (`tracker.apply_splits`, PREREG64) |
| Sell | at the open after the first close above the 3-session average since the FILL session, or after 5 sessions (from the 2026-10-20 session; 5 / 10 before) |
| Gives way | a short — or, from 2026-10-12, the crash-rebound book's TQQQ long (`live_account.crash_give_way_plan`) — the shared room cannot fund in full sells the book's latest longs WHOLE, as many as needed (`dip_gave_way`, never bought back) — none when selling them all would not fund one more share of it (PREREG64); at a tick where the maintenance requirement exceeds 80% of equity the latest longs are sold until it is at most 60% (`dip_delever`); a margin call closes everything (`dip_margin_call`) |

The daily bars are built from the deep 30-minute store exactly as the study built them (`daily_bars`); nothing of
session D is read for D's limit, and a touch is judged only on bars that have ENDED by the tick.

## THE OWN-SECURITY SERIES — LIVE from the 2026-10-19 session

User 2026-10-18: "Yes please apply the fix" (AUDIT127, PREREG142). The deep store keeps whatever traded under a
ticker: a REUSED ticker stitched an older security in front of the new one — SpaceX (SPCX, listed 2026-06-12, ~90
sessions) was a signal on 2026-10-15 and 10-16 on a SPAC ETF's 200-session average (score rank 16 of 17: not bought) —
and a RENAMED company's history sits under its earlier ticker (EchoStar as SATS until 2026-06-23). Before the rule reads
a name, `dip_long._daily` rebuilds its stored bars from its own security (`src/signals/dip_security.py`, PREREG142's
construction ported function for function; tests `tests/test_dip_security.py` + its guard in `tests/test_no_lookahead.py`,
mutation-checked); every reader of the book's daily bars goes through it — the signal, the meta-model's 17 inputs (so
`breadth` and `days_since_prev` too), the order's ATR, the exit.

| | |
|---|---|
| Runs | Polygon's ticker events (`/vX/reference/tickers/{t}/events`): the `ticker_change` entries in date order, one ticker's consecutive entries merged; without events, the ticker details' list date (`cache/ml/deep/ticker_details.parquet`) as the start of one run; a name without a run of its own ticker (GOOGL, BRK-B) is read as stored |
| Cut | the stored bars before the security's first run are another security's when the series BREAKS there: a gap of more than 30 calendar days, or the close moving more than x1.5 AND the 20-session median dollar volume more than x5, either way (an event date with no break — C's reverse split, LIN, GOOG — cuts nothing) |
| Join | a run under ANOTHER ticker overlapping the store's window (from 2021-01-04): that ticker's regular-hours 30-minute bars over the run, UNADJUSTED, in today's terms by the security's own splits (each counted inside its ticker's run; only splits executed by today — the store's terms); genuine with bars on at least 20 sessions; on overlapping days closes within 2% = the same security (stored kept), more = another security used the name (the stored bars inside the run cut) — believed only when the run's last close is within 25% of the next close; every junction within 25% across at most 30 days, a failing one drops everything before it |
| Live | `compute_signals`: a candidate never answered is asked at the day's computation, and a day's SIGNAL answered a day ago or earlier is asked again and re-evaluated (both within `dip_long_security_budget_seconds` 15; one request a name, no retry); after the computation the candidates answered more than `dip_long_security_max_age_days` (7) ago are refreshed in a background thread (single-flight); a joined run's bars are fetched once (`runs/`; an empty answer retried after 7 days) |
| Cache | `cache/ml/dip_long/security/events.json` (each name's answer and when; a failed answer never replaces a good one: `last_failure`), `runs/<TICKER>_<start>_<end>.m.pkl` (+ `index.json`); seeded 2026-10-18 from PREREG142's fetches of 2026-10-17 (1,913 names, 72 runs: `python -m src.signals.dip_security --seed`) |
| Fail-open | a construction that raises reads the name as stored (as before the fix), journaled — a scorer-banner PROBLEM when it is the day's pick; Polygon not answering keeps the name's last answer, else its list date (a note for a signal) |
| Journal | `signals/<day>.json`: the `security` block (asked_new, rechecked, failed, skipped, runs_fetched, the seconds; `changed` — each candidate rebuilt: how, its first session, the cut, the runs joined; `errors`); a rebuilt signal carries `own_security` |
| Parity | production's construction against PREREG142's corrected panel (`python -m src.signals.dip_security --parity`): identical daily bars on 255 of the 256 names the study changed or planned; RBC differs only by the study's duplicate drop, which also removed RBC Bearings' own ROLL history (production keeps it; never a $1B name). Not live: the study's duplicate drop (a joined run's days removed from the earlier ticker's own series — that series no longer trades) and its daily-cache foreign segments (before 2021) |
| Effect | 2026-10-16 recomputed: SPCX no longer a signal (16 of the 17), the pick unchanged (LITE); BNY (BK joined, a closed-end fund cut), COHR, META (FB joined), P (PSTG joined) and SPCX rebuilt among the candidates. The simulator's default panels are this construction's (PREREG142's, `strategies.dip.OWN_DEFAULT`): the deployed configuration on them +56.7% a year against +56.1% on the stored panels (the correction measured with the deployed model: +0.6 points) |
| Off | `ENABLE_DIP_LONG_OWN_SECURITY=false`: the stored series, nothing asked |

```bash
python -m src.signals.dip_security --name SPCX       # one name's construction now (the cached answers)
python -m src.signals.dip_security --refresh SPCX,ECHO   # ask Polygon now (events + the joined runs)
python -m src.signals.dip_security --parity          # against PREREG142's corrected panel
```

## The simulator mimics it

The backtest engine runs the same orders on the same series — its default dip panels are the own-security ones
(PREREG142's corrected panels, `strategies.dip.OWN_DEFAULT`, below) — : `src/backtest/strategies/dip.order_pieces`
(the same `limit_price`, the placement instants, one piece per touched session with the open / bar-end fill and
30-minute marks) with
`Strategy(orders="fill")` (a fill cut to the room at that instant; no room: the order keeps working), a tick's fills
before its other entries, and `Rules(one_book_per_name=True)` (the books never share a name). PREREG61 amendment 2,
the live combination 2021-26: +58.6 %/yr from 2021 vs +55.0 buying at the open (+2.3 log points, not significant),
the dip book's return per day 0.184 vs 0.127%; February 2021 - July 2024 lower (+23.6 vs +26.0).

## THE DAILY TOP PICK (PREREG116's Q1) — LIVE from the 2026-10-16 session

User 2026-10-14: "does it favor its best picks first? Would it be better the buy the top 1 pick even if we have more
slots available?" (PREREG107), then 2026-10-15: "Deploy Q1". Settings `dip_long_daily_top` (1), `dip_long_rank`
("score"), `dip_long_account_slices` (6); code `dip_long.compute_signals` → `_daily_top` / `rank_key`,
`dip_filter.score_signals`, `tracker._place_dip_orders` (`dip_long.placeable`, `not_top_order`); tests
`tests/test_dip_top_pick.py` + the top pick's guard in `tests/test_no_lookahead.py`; evidence
`memory/pooled-retune-new-arms-2026-10.md`.

| | |
|---|---|
| Score | the meta-model below (the model file `dip_long_filter_model`, fit on the signals exiting before 2026-10-08) scores every signal of the day on the closes through D-1 — from the 2026-10-21 session its RP33 CUT drops the signals under it first (`enable_dip_long_filter` on; PREREG166 / 167) and the pick is the best of the kept signals; before, every signal competed |
| Pick | the highest score; a signal whose inputs raise ranks after every scored one by its RSI2; the order id (ticker + "|") breaks a tie — the simulator's `strategies.dip.rank_pieces` (`test_the_daily_top_pick_is_the_simulators_choice`) |
| The others | journaled in `signals/<day>.json` `not_top` and in `orders/<day>.json` with status `not_top` (never placed, never rested at IBKR, no slot) |
| Transition | an order of a day before the rule (no `dip_top`) is never placed: `superseded` at its next opening tick; a working one is armed as before |
| Fail-open | no model, a checksum that is not the JSON's, or no SPY close for D-1: the day is unscored and the DEEPEST DIP is the pick (the simulator's rule for an unscored order); the `top` block's `error` raises a scorer-banner problem |
| Why the slice and the pick go together | the 1/5 slice WITHOUT the daily top-1 met margin calls in 27 (set, draw) pairs against the live 10 (PREREG117): never raise the slice with `dip_long_daily_top` 0 |
| Evidence | PREREG107's O4 (top-1 at 1/5) +5.00 +- 0.34 log points a year but a PEAK (PREREG116: neighbours +2.8..+3.5); Q1 = top-1 at 1/6 + at most two shorts of one name: +2.89 +- 0.29 (ADD), margin-call pairs 10 -> 7, drawdown 27.8 -> 26.4, the live account +52.8 %/yr; the 2010-2017 replay +12.8 vs +11.2 %/yr; about 1.7 dip trades a week against 3.3 |
| Journal | `signals/<day>.json`: the `top` block (the rule, by, top, breadth, scored, the model and its cut, SPY's source, `picked`, `error`), each signal's `top_rank` / `filter_score` / `filter_inputs`; the trade stamps `dip_top_rank` and `dip_filter_score` |
| Simulator | `Mimic()`'s defaults follow production (`dip_rank` / `dip_topk` from `dip_long_rank` / `dip_long_daily_top`; the walk-forward scores `strategies.dip.filter_scores` run to the signals of 2026-10-06 — a later one is unscored) |

## THE $2B FLOOR AT 1/3 (PREREG147's S3) — LIVE from the 2026-10-19 session

User 2026-10-18: "Deploy the $2B+ stocks only, 1/3 per position" (after PREREG146 / 147: "can you verify if big stocks
get better results?", "What about trading 2B+ stocks but using bigger sized position?"). Settings
`dip_long_order_min_dollar_volume` ($2B) and `dip_long_account_slices` (3); code `dip_long._daily_top` / `order_floor`
(the floor), `live_account.size_long` (the slice and the cap, unchanged code); the simulator's `Mimic.dip_min_order_dv`
(production's floor by default) and `dip_slices` (None = production's 3); tests `tests/test_dip_top_pick.py` (the floored
pick, a day with none over the floor, the simulator's choice, the defaults).

| | |
|---|---|
| Pick | the day's top pick by score (the earnings dips last) among the signals whose 20-session mean dollar volume at D-1 (the rule's `dv20`) is at least $2B; ranks (`top_rank`) are given among them, the pick rank 1 |
| Scored as before | every $1B signal is still computed, journaled and scored — the meta-model's `breadth` counts all of them (its training table's construction); a signal under the floor is `not_top` with `below_order_floor` and no rank; a day without a $2B+ signal orders nothing (log `no signal at or above the $2B order floor`) |
| Size | each long 1/2 of the live account's equity at its limit, at most 2 open, from the 2026-10-20 session (1/3, 3 open, from 2026-10-19; working orders count); the longs' value at most the equity, Reg T's 50%, the give-way and the de-leveraging unchanged |
| Journal | the `top` block's `order_floor` and `below_floor` (the tickers under it); each signal's `below_order_floor`; the email note counts them |
| Transition | the longs bought before the rule (9 open on 2026-10-18, at 1/10 and 1/6) keep their exits; with 3 slots the first order under the rule waits until fewer than 3 dip longs are open (the 10-09 longs leave by the open of 2026-10-23 at the latest); a pending order of a day before the rule (LITE, 2026-10-16, $4.2B a day) is placed when a slot frees within its window |
| Evidence | PREREG147 (20 paired draws a set, the corrected panels, against the $1B pick at 1/6): +71.6 vs +56.8% a year (+14.8 +- 0.5 pp; A0 +12.17, A +4.81, B +11.10 log points; every leave-one-year-out positive, without 2020 +7.19), margin-call pairs 7 -> 7, the worst drawdown 26.4 -> 30.3% (41.2 vs 37.7 in the worst draw) — the add rule's only failed test, the user's call; the $2B+ dips earn ~2x a trade; the book trades ~0.9 a week (1.7 before) at ~0.38 %/day; the bigger slice without the floor does not pay ($1B at 1/5 / 1/4 +2.1 / -0.5); the rule's own ADD was the tier-sized pick (1/3 at $2B+, 1/6 below: +9.8 pp, calls 6, drawdown 27.1; PREREG148 / 149) — offered, not chosen. Exploratory (36 trials on the same history); the 2010-2017 replay is the open out-of-sample check |
| Parity | the default `Mimic()` (study102's plain harness, the deployed settings) reproduces PREREG147's S3 runs exactly on sets A0 and A, every field and the daily NAV (`prereg147/default_parity147.py`); `bank.LIVE_RUNS` = S3's draws |
| Off | `DIP_LONG_ORDER_MIN_DOLLAR_VOLUME=0` and `DIP_LONG_ACCOUNT_SLICES=6`: the $1B pick at 1/6 (the rule to 2026-10-18) |

## THE EARNINGS-DIP DEMOTION (PREREG137's DEMOTE_E) — LIVE from the 2026-10-19 session

User 2026-10-17: "Fully deploy the earnings dip demotion in live". Code `src/signals/dip_earnings.py` (called from
`dip_long._daily_top`; `dip_long.rank_key`), settings `enable_dip_long_earnings_demote` (True),
`dip_long_earnings_demote_sessions` (3), `dip_long_earnings_edgar` (True); tests `tests/test_dip_earnings.py` + its guard
in `tests/test_no_lookahead.py`; evidence `memory/dip-earnings-dips-2026-10.md`, report
https://claude.ai/artifact/9gVkGD4XAdMnhRhAgsQNPn.

| | |
|---|---|
| Earnings dip | the signal's company filed an 8-K or 8-K/A naming Item 2.02 (its earnings release) ACCEPTED after 16:00 ET of the session 3 before the signal session (D-1) and at or before 09:30 ET of D — the reactions of D-3, D-2, D-1 and the night before D (`dip_earnings.window`; PREREG137's E) |
| Rank | after every other signal of its day, the earnings dips among themselves by the same key (score, then RSI2): the day's best other dip is the pick; a day of earnings dips only buys its best one (`rank_key`'s first element; the simulator's `strategies.dip.rank_pieces(demote=...)`, `test_the_ranking_with_the_demotion_is_the_simulators`) |
| Orders | each order carries `earnings_dip` / `earnings_status` / `earnings_release` (a not-top one too); an earnings dip's order ranks after the others when orders compete for a slot or fills for the room; nothing else changes (the slice, the limit, the exit) |
| Source 1: the store | the name's part of the deep store's `sec_filings` (`cache/ml/deep/sec_filings/parts/<T>.parquet`: the nightly full pass ~00:00 ET, the pre-open's pass over EDGAR's live feed ~08:33) — the study's own data; a flag there needs no request |
| Source 2: EDGAR now | for a signal the store does not flag, the company's submissions JSON (its CIK from the part; ONE request a name at the opening tick — no retry, no Retry-After wait, at most 10 s, never past the day's 45 s, best-ranked first: `dip_earnings._get`): catches a release accepted after the pre-open pass (4 of the study's 551 came at 09:00-09:30). EDGAR's JSON instants DRIFT once a filing is days old (+4 h, +8 h: every sampled October release on 2026-10-17), so a release it answers within 12 h of the window is re-timed from its filing index page's "Accepted" (`retimed`, `accepted_at`; `acceptance_source` "index", else "json") |
| Statuses | `flagged` (a release in the window) · `clear` (EDGAR's current filings hold none) · `store_only` (the store holds none; EDGAR off, silent or out of budget) · `no_filings` (the store holds no part for the name — a listing newer than its last universe pass: never an earnings dip, as the study's unmapped names; an ADR whose issuer files 6-Ks has a part without 8-Ks and reads `store_only` / `clear`) · `unknown` (the part could not be read) |
| Fail-open | `unknown` is NOT demoted (the ranking as without the rule): a scorer-banner PROBLEM when it is the day's pick, nothing when it ranks below it; `store_only` for the pick with EDGAR on is a note; the whole check raising demotes nothing (`top.earnings.error`, a problem) |
| Journal | `signals/<day>.json`: `top.earnings` (the rule, the window, `flagged`, each signal's status, `unknown`, EDGAR's requests and failures, the store's last pass, the seconds); each signal's `earnings_dip`, `earnings_status`, `earnings_release` (accession, form, items, acceptance, source); `top.picked` carries the pick's; the trade stamps `dip_earnings_dip` and `dip_earnings_release` (the accession) |
| Evidence | EXPLORATORY (chosen after PREREG137's top-pick table of the same years): the live account on the production-mimicking simulator, 20 paired draws a set, the pooled add rule ADD: +52.8 -> +56.1% a year (+3.25 +- 0.20 pp; A0 +1.81, A +1.51, B +3.40 log points), 8 of 9 years (2019 -3.2), margin calls 7 -> 7 draws, drawdown 26.4 -> 26.4%, the dip book's return per day A .147 -> .171, B .317 -> .346, A0 .169 -> .196 %/day, its trades as many; the window's plateau 1 / 3 / 5 sessions +2.3 / +3.3 / +2.9 pp (5: one more margin-call draw); set B's lendability +-0.37: +3.27 / +3.41; NOT confirmed on the $200M-$1B tier (PREREG140); the overfitting report's shrunk gain +0.31 log points |
| Parity | production's flags (`mimic.earnings_demoted` on this module's window) equal the study's on all 4,730 simulator orders of sets A0, A and B — 386 flagged (the 1- and 5-session windows: 193 / 515) — and the default `Mimic()` reproduces the study's DMTE draws exactly (growth, drawdown, calls, the daily NAV); on the index pages' instants the flags are the same 386 |
| Off | `ENABLE_DIP_LONG_EARNINGS_DEMOTE=false`: nothing is read or asked, the signal files are as before; it takes effect at the next computation (a day already computed keeps its pick) |

## THE WALLSTREETBETS PREFERENCE (PREREG209's KD+) — LIVE from the 2026-10-26 session

User 2026-10-25: "Build KD+ for production and deploy it." Code `src/signals/dip_wsb.py` (called from
`dip_long._daily_top`), settings `enable_dip_long_wsb_prefer` (True), `dip_long_wsb_min_titles` (5),
`dip_long_wsb_window_hours` (120), `dip_long_wsb_cut_et` ("09:00"), `dip_long_wsb_max_feed_lag_minutes` (60); tests
`tests/test_dip_wsb.py` + its guard in `tests/test_no_lookahead.py`; the study `llm_trader_research/studies/prereg209/`,
evidence `memory/wsb-attention-prereg209-2026-10.md`, report https://claude.ai/artifact/3UfKEYNQ81TJJSh11UpXqg.

| | |
|---|---|
| Flag | the signal's stock named in at least 5 r/wallstreetbets submission titles CREATED in the 120 hours before 09:00 ET of D (`dip_wsb.window`: [cut − 120 h, cut)) — the production feed's store (`data/reddit_wsb/<UTC day>.jsonl.gz`, every 30 minutes, `docs/DATA_FEEDS.md`; seeded from 2018-01-01) |
| Counting | PREREG208's (`feeds.wsb._RX`: an upper-case whole word of 1-5 letters or the cashtag $TICKER; `wsb.WORDS` and every single letter only as cashtags) + PREREG209's eight cashtag-only tokens (RH, TD, EDIT, IMO, PSA, EV, TV, DOW: Robinhood, TD Ameritrade, "EDIT:", ...); a post once per name; a class share's root (BRK-B → BRK) |
| Pick | the eligible signals (kept by RP33, at or above the $2B floor) in their ranking's order (score, the earnings dips last) with the FLAGGED ones first (`dip_wsb.prefer`): the pick is the best-ranked flagged signal; a day without one picks as before; a flagged signal under the floor is never the pick |
| Fail-open | nothing is preferred (the pick as without the rule) when the window is not in the store — a UTC day it touches has no file (`missing_days`), its newest post before the cut is more than 60 minutes older than the cut (`stale`: the feed stopped) — when the signals are computed before the cut (`early`) or the check raises (`error`): a NOTE in the scorer banner, never a problem. The feed guards the window (from the 2026-10-26 evening, after that morning's archive outage left the first live day stale): a failing archive retried, asked every minute from 08:40 to 09:27 ET on a market day, and its gap filled from Reddit's own API once the user's app is configured — none yet (`docs/DATA_FEEDS.md`, "When the archive fails") |
| Journal | `signals/<day>.json`: `top.wsb` (the rule, the window, its status, the days read and missing, the store's newest post, `titles` per signal, `flagged`, `preferred`, `pick_without` = the pick without the rule); each signal's `wsb_titles` / `wsb_flagged`; `top.picked` carries the pick's |
| Simulator | `Mimic.dip_wsb` (production's settings by default; `mimic.PRE209` = none): on an order day where any order (after the RP33 cut and the floor) is flagged, only the flagged orders compete for the daily top pick (`mimic.wsb_preferred`, the same pick as production's for the daily top 1), the flags from the store's history index (`dip_wsb.history_index`) |
| Parity | the default `Mimic()` rebuilds PREREG209's KD+ long books piece for piece (A / B 813 dip pieces, A0 316); production's file counts equal the history index's on the last 60 signal files (109 of 109, `python -m src.signals.dip_wsb --parity`) |
| Evidence | PREREG209 (pre-registered; 40 draws a set, pooled A0 + A + B, against LIVE205): +88.1% a year against +78.8% (+9.3 ± 0.3 pp, +5.08 log points; A0 +4.0, A +5.6, B +26.7 pp; every year left out at least +3.0 log points), margin-event pairs 37 → 36, months positive 76.1 → 76.6%, the worst 12 months +9.1 → +8.0%, the average worst drawdown 25.7 → 27.3% — the add rule's only failed test (the worst draw's 35.7 → 34.7%): the user's call; 2023 −5.5 and 2026 −6.5 log points; the gain in rising markets of middling volatility (SPY above its trend, mid volatility: 75.9 → 107.8% a year); about one order in seven changes, mostly to the forum's large caps (NVDA, AMD, TSLA, AAPL, MU) — the dip book's own trades +0.61 against +0.54 %/day |
| Off | `ENABLE_DIP_LONG_WSB_PREFER=false`: nothing is read, the pick as before; it takes effect at the next computation (a day already computed keeps its pick) |

## The meta-model filter (PREREG96's RP33) — BUILT 2026-10-14, ON FROM THE 2026-10-21 SESSION (before the daily top pick)

DEPLOYED (user 2026-10-20: "Deploy all three.", on PREREG167): on the deployed configuration (RPD_IS) the cut before the
daily top pick raises the book's return per day 0.44 -> 0.53 %/day and the account +1.79 pp a year alone (PREREG166;
the return-per-day rule's ADD, the standard rule's REJECT: 2024 carries it), +3.62 pp with the crash book's after-hours
exit and +7.19 pp with the capitulation book too (PREREG167, 66.5 -> 73.7% a year, margin-call draws 1 -> 1). THE EARLIER
POOLED VERDICT (PREREG109 Annex B, on the configuration of 2026-10-14 without the daily top pick): REJECTED — +1.94 log
points a year at 20 draws but one more margin-call pair (10 -> 11), +1.81 +- 0.10 at 40 draws with margin-call pairs
21 -> 23; its model's score has ranked THE DAILY TOP PICK (above) since 2026-10-16.

User 2026-10-14: "I'm ok with deploying PREREG96's dip meta-model filters RP33 and CP33 if they gain in total on all
three sets" — RP33's add-rule gains: set A +2.39, set B -1.06, set A0 +1.13 log points a year (+2.46 in total; CP33
+0.29, and RP33 is ahead on every set). Code `src/signals/dip_filter.py`, switch `enable_dip_long_filter` (True from the 2026-10-21 session),
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
| Off | `ENABLE_DIP_LONG_FILTER=false` (the default to 2026-10-20; true from the 2026-10-21 session): no cut and no `filter` / `filtered` journal; the daily top pick still scores every signal with the same model (`filter_score`, `filter_inputs`, the `top` block) |
| Refit | yearly, the walk-forward's recipe, THEN A PAIRED ACCOUNT TEST before it replaces the deployed model, read across SEVERAL SEEDS (PREREG142: a refit reshuffles about one daily top pick in five, and the refit on the repaired table lost 1.6 points a year; PREREG143: the recipe's seed alone moves a fit's growth by ~3 points a year — seeds 0-4 of the deployed recipe 53.8-56.8%, the deployed fit the best of its five — so one fit against another measures mostly the seed lottery; the deployed model stayed). Rebuild the table with PREREG142's construction (`studies/prereg142/`: `spans142.py raw / events / spans` -> `bars142.py` -> `panel142.py` -> `data142.py` — today's stores, every reused ticker cut to its own security, every renamed company's earlier tickers joined; data96's code underneath), fit `python -m src.signals.dip_filter --fit --table <signals142.pkl> --cut <YYYY-MM-DD> --out <a staged file>`, score the walk-forward years (`model142.py scores`) and run the pooled add rule against the deployed model on the corrected panels (`account142.py`, `Mimic(dip_scores=...)`); only a pass replaces `model.txt` (the JSON records the cut, the rows and the checksums) |
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

THE REFIT ON THE REPAIRED DATA (PREREG142, 2026-10-17, `llm_trader_research/studies/prereg142/`, user: "Ok rebuild
and refit the 11 points model"): the table rebuilt from today's stores (7,545 signals) with every reused ticker cut to
its own security (41 fake signals out: SNOW, META 2022, COIN, SPCX, GPRO, GM, AI — they had averaged +4.3% net) and
every renamed company's earlier tickers joined (211 in: Facebook's years under META, Square's under XYZ, ...). The
refit's walk-forward models against the deployed ones, both on the corrected data, the pooled add rule (20 paired
draws a set): REJECTED, -1.05 +- 0.11 log points a year (55.1% against 56.8% a year; set A +1.66, B -3.83, A0 -1.99),
margin-call pairs 7 -> 6. Its ranking is no better: Spearman IC 0.194 vs 0.197 on 2022-2026 (0.178 vs 0.154 on
2011-2020), the daily top pick +1.57 vs +1.54% a trade on 2021-2026. The two models' scores rank-correlate 0.93 yet
pick a different top on 19-23% of the multi-signal days. The deployed `model.txt` stays; the candidate is kept in the
study (`model_r142.*`). The correction itself, with the deployed model: +0.6 points a year (the fake SPCX trades of
July 2026 lost, the fake META rebound of October 2022 won).

THE SEARCH FOR A BETTER MODEL (PREREG143 / 144, 2026-10-17, user: "Apply any fix needed to fit the model correctly. A
refit should in theory provide better results. Find the best model possible."): ten recipes walk-forward on the
corrected table — ten-seed ensembles, LambdaRank with one query per signal day, the within-day demeaned / winsorized /
fill-aware labels, a smaller model (15 leaves, 100 rows a leaf), three dip-day inputs (the dip day's gap, its volume
against its 20-day mean, consecutive down closes), the volatility heuristic (the highest ATR14% first) — and the same
recipe trained on the rule's dips at $500M / $250M a day (2.7x / 6x the rows). On the day's pick's value (1,269
multi-signal days 2011-2026) none beats the deployed model (best +0.045% a pick, interval -0.15 .. +0.26; LambdaRank
-0.35; PBO 0.47); in the account every one loses: the corrected-table ensemble 55.3% a year against 56.8%, the old
table's 54.2%, the smaller model 54.0%, the volatility heuristic 47.0%, the wider data 47.3 / 48.4%. Five seeds of the
deployed recipe: 56.8 (the deployed fit), 56.0, 56.2, 54.3, 53.8% on its own data; 55.1, 55.7, 53.1, 55.0, 56.3% on
the corrected data. The deployed `model.txt` stays.

ONE MODEL PER STOCK SIZE (PREREG145, 2026-10-18, user: "Try retraining by medium stocks and large stocks and then each
model needs to serve their corresponding stock size."): the deployed model kept on the $1B dips, a second model (the
deployed recipe, 10 seeds, walk-forward) fit ONLY on the medium tier's dips ($250M - $1B, or $500M - $1B) scoring that
tier, the book's pool widened to it — one daily top pick over both tiers, or the $1B dips first. REJECTED, every form:
50.5 / 47.9% a year ($250M / $500M floors, one ranking), 51.7 / 50.1% ($1B dips first), against 56.8%; margin-call pairs
7 -> 26 / 8 / 24 / 15, every set-A0 call on 2020-03-16 (the wider pool had filled the slots with volatile medium dips
before the March 2020 crash). The medium tier's dips earn +0.36% a filled trade unranked, its best pick on a day
without a $1B dip ~0, and the medium-only model ranks them no better than the deployed model applied to them; the
deployed model ranking both tiers (a diagnostic) made +2.4 pp, carried by 2025 alone, with margin calls 7 -> 25. Set
A0's medium names needed their 30-minute bars from 2016-11-01 (the A0 store held them from 2017-06-01; refetched for
the same security only, `studies/prereg145/a0ext145.py` -> `panel145_a0.pkl`, reusable).

BIGGER STOCKS (PREREG146, 2026-10-18, user: "if small and medium stocks got worse results, can you verify if big stocks
get better results?"): per trade they do — the filled dips at $1-2B a day earn +1.0% a trade (0.16 %/day), at $2-3B
+2.2%, $3-5B +2.3%, $5B+ +1.9% (0.32-0.41 %/day; every bigger tier's gain over $1-2B with its 95% interval above 0, in
2007-2020 and 2021-2026 alike) — but the book is not better for trading only them. REJECTED, every form, on the
corrected panels against the live configuration: the pool cut to $2B+ +0.7 points a year (57.1 vs 56.4% at 40 draws;
+0.43 log points under the rule's +0.5, carried by 2019, margin-call pairs 15 -> 16, the drawdown 1.8 points deeper),
$3B+ -6.9, $5B+ -11.5; the bigger dips ranked first -1.5 / -4.7 / -2.7 ($2B / $3B / $5B); a model fit only on the $2B+ /
$3B+ dips -4.2 / -8.6 (its rank correlation with the outcome is about the deployed model's — 0.199 vs 0.189 at $2B,
0.194 vs 0.199 at $3B — yet its daily picks did worse in the account). The restricted books' dips earn
about twice as much a day (0.37-0.42 against 0.16-0.36 %/day by set) but trade 25-60% less, the freed slots mostly in
cash; and on the days a bigger stock dips beside the model's smaller pick, the model's pick earned more (+1.62 vs
+1.23% a pick at $2B).

NEW INFORMATION AND THE MARKET CONTEXT (PREREG170, 2026-10-20; PREREG181, 2026-10-22, user: "Do task 2 also"): ten-seed
walk-forward ensembles of the deployed recipe given more inputs, each against the same construction without them — the
companies' own SEC XBRL numbers, the sector fund's trend and move, Wikipedia attention (170), and the market context the
17 inputs lack (181): SPY against its 50-session average and the 50 against the 200 (PREREG179's grind state), its
20-session volatility's percentile in its own history, and the breadth of the 100 most-traded stocks (the share oversold
under their trend, the share under it). On the book's whole daily decision — each year's RP33 cut from the model's own
out-of-fold scores, then the $2B+ top pick, 1,182 sessions 2011-2026 — none decides better (the grind -0.03% of a
position a day, the regime +0.01, the breadth -0.04, all five -0.08 with its 95% interval under 0); the deployed recipe
stays ahead of every re-fit. The market's state already enters through the four SPY inputs, and a day-level input
moves every signal of its day together: it can change the cut, not the pick.

BIGGER POSITIONS FOR THE BETTER TRADES (PREREG147-149, 2026-10-18; user: "What about trading 2B+ stocks but using bigger
sized position?", then "we want higher size for better trades"; the user then chose the $2B+ book at 1/3 — DEPLOYED from
the 2026-10-19 session, THE $2B FLOOR AT 1/3 above): on the corrected panels against
the live configuration (56.8% a year), 20 paired draws a set —
* the $2B+ pool alone at 1/5 / 1/4 / 1/3 a position (as many open): +5.4 / +10.0 / +14.8 pp a year, margin-call pairs
  8 / 8 / 7 against 7, the worst drawdown 28.4 / 29.1 / 30.3 against 26.4 (1/3 fails only the drawdown: a larger gain
  for a deeper one, the user's call); the $1B pool at 1/5 / 1/4 +2.1 / -0.5 — the size pays only on the better trades;
* the $2B+ book at 1/3 with its remaining capital in $1-2B dips — on the days without a $2B+ dip, or the live pick
  sized by tier — loses against the $2B+ book alone (-0.9 to -4.5 pp: the $1-2B trades earn about nothing outside set B);
* THE LIVE PICK SIZED BY TIER (T3_6: the daily top pick unchanged, 1/3 of the equity when it trades $2B+ a day, 1/6
  below, at most 6 open) ADDS by the pooled rule: 66.6% a year (+9.8 pp), margin-call pairs 7 -> 6, the worst drawdown
  26.4 -> 27.1%, every year's leave-one-out positive (without 2020 +3.96 log points); its neighbours hold (1/2 at $2B+
  +9.8 pp with the drawdown 3.3 deeper, 1/4 +6.4, 1/8 below $2B +9.4, the line at $1.5B / $3B +5.1 / +7.0). Exploratory:
  chosen after PREREG146-148 on the same history (36 trials). A deployment would size each order by its dv20 (production
  code + a `Mimic` option + their parity test).

THE RULE UNDER THE STOCK'S OWN TREND (PREREG163, 2026-10-20; user: "Work on your next candidates 1 to 4. Make sure to
explore all"): the same signal ($2B+, RSI2 < 10), order and exits with the stock's close UNDER its 200-session average —
the dip book's mirror for a weak market, where its own signals are few — fails: the daily pick by the deepest RSI2
earns +0.08 to +0.12 %/day on 2018-2026 (95% intervals reaching -0.4) in the weak cell, any below-trend cell or every
context, and in the account at the dip book's place it costs -5.0 to -36.2 pp a year (the drawdown 26 -> 33-45%; 2020's
picks -15% each). Every signal's order earns more (+0.27 %/day in every context, 962 trades) — on MARKET-WIDE
capitulation days (sessions with many such signals: +3.2% a trade at 5 or more, -0.7% on single-signal days in the weak
cell). Amendment 1 (exploratory) bought every such order on sessions with at least K signals in a give-way place: at
K 5 +1.95 pp a year, but 2022 alone (without it -0.75 log points) and a lone peak (K 3 +0.68, K 8 -0.02) — not
recommended. The longer history 2007-2017 holds only survivors (the 2008 failures are missing), so it cannot decide.
THE META-MODEL'S CUT ON THE DAILY PICK (PREREG166's RP33, 2026-10-20; the return-per-day review): the RP33 cut
(`enable_dip_long_filter`) applied before the daily top pick on RPD_IS raises the book's return per day 0.44 ->
0.53 %/day (337 trades a draw against 381) and the account +1.79 pp a year (calls 1 -> 1, drawdown +0.8) — the
return-per-day rule's ADD, the standard rule's REJECT (2024 carries it); with the crash book's after-hours exit
(U) +3.62 pp, the standard rule's ADD; with the capitulation book too (PREREG167) +7.19 pp: DEPLOYED from the
2026-10-21 session (the user: "Deploy all three."). A deeper limit (1.0 ATR) or a
one-session order raises the book's return per day further (0.72 / 0.57) but costs growth and a margin call.
A new long book on the deployed account must not take this book's room: at its place (quality 0.13) every one tried on
2026-10-20 crowded the 1/2 slices (PREREG164's pre-earnings book -16.2 pp a year).

```bash
python -m src.signals.dip_filter --status                    # the model, its fit window, its cut, the switch
python -m src.signals.dip_filter --score --day 2026-10-14    # what it would keep of a day's signals (writes nothing)
```

## A normal day

1. 08:30 the pre-open run and 09:00 the selection short's prepare extend the deep 30-minute store through the
   previous session, and cache Polygon's whole-market daily bars the prefilter reads.
2. 09:30 tick, start: `monitor_dip_long_positions` sells every long whose exit is due (log `[dip_long] SELL`).
3. 09:30 tick, live path: `record_dip_long_trades` computes the day's signals once (`signals/<day>.json`, ~2 s, plus
   ~0.3 s a signal for EDGAR's earnings check: log `[dip_long] ... is an earnings dip` when one is ranked last; each
   candidate read from its own security — Polygon asked about a new candidate or a day-old signal, ~0.2 s a name: log
   `[dip_long] own security: N candidate(s) rebuilt`),
   writes them as orders (`orders/<day>.json`), places the pending orders and arms the carried ones (log
   `[dip_long] LIMIT`), and fills at the open those whose session opened under the limit (log `[dip_long] BUY`).
   The broker sync then rests each working order at IBKR (`[broker] dip limit ... resting`, refs `dipw-...`).
4. Every later tick, at its start: `monitor_dip_long_orders` fills the orders whose limit a bar reached (log
   `[dip_long] BUY ... limit order filled`), before the shorts' entries; each sync resizes the resting IBKR orders to
   the room left (cancel + re-place under the next ref). The long a fill opens — a touch's or an open fill's, also
   when IBKR did not confirm a cancel of that order — is linked to its resting order (from the 2026-10-19 session,
   AUDIT151: before it an open fill was never linked, so IBKR's execution of the resting order was never recorded
   and the entry sync logged `POSITION_ALREADY_AT_TARGET`). An OPEN fill's ledger entry price then becomes IBKR's
   own average fill once the fill refresh reads it (log `[broker] dip open fill ... the ledger books IBKR's`;
   the opening print kept as `dip_open_print`): the order reaches IBKR ~30 s after the open, so the print was never
   attainable.
5. The first tick at/after 16:00 judges the session's last bar, then lapses the orders whose window ended.

## Checks

```bash
python -m src.signals.dip_long --status                       # the open longs, and the book's OWN cash account (used only with enable_live_account off — THE LIVE ACCOUNT that sizes it has no CLI)
python -m src.signals.dip_long --signals --day 2026-10-09 --no-write   # what the book would order: writes no signal file, but may extend the deep store for stale names and asks EDGAR (the earnings check)
```

Files under `cache/ml/dip_long/`: `signals/<day>.json` (the computation: `signals` with `atr14` and `limit`, the
closest misses in `next`, `stale` = names that traded on D-1 but the store lacks, `inactive` = names that did not
trade), `orders/<day>.json` (each order's life: `placed` / `armed` / `filled` / `unfunded` / `held` / `no_slot` /
`account_full` / `account_too_small` / `no_price` / `cancel_margin` / `lapsed` / `split`, and the never-placed
`not_top` / `superseded` / `filtered`; its quantity, the IBKR order resting for it), `entries/<day>.jsonl` (each
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
* **"the meta-model filter did not score today's signals"** (the filter on) — the filter kept every signal; the daily
  top pick still ordered only the day's best (the deepest dip when the signals stayed unscored). The reason is in the problem and in the signal file's `filter.error`: the model file missing
  or not the one its JSON's sha256 names (`python -m src.signals.dip_filter --status`), or no SPY close for D-1 in
  the daily cache nor the deep store (the EOD chain's cache warm and the pre-open run both failed). **"computed without
  it"** — the day's signals were computed before the switch went on: the filter starts at the next session.
* **"today's pick X's filings could not be read"** — the pick's part of the store (`cache/ml/deep/sec_filings/parts/`)
  could not be read: the pick was not checked for an earnings release (not demoted). `top.earnings` in the day's signal
  file holds the error. **"the earnings-dip check failed"** — the check raised as a whole: no signal was demoted that
  day. **"EDGAR did not answer for today's pick"** (a note) — the store's filings decided (complete through its last
  pass, `store_through`). `python -m src.signals.dip_long --signals --day D --no-write` re-runs the day's check now.
* **To stop new longs**: `ENABLE_DIP_LONG=false` in `.env` and restart the scheduler: no order fills, the resting
  IBKR orders are cancelled at the next sync, the orders lapse. The open longs still leave on their rule.
  `DIP_LONG_ENTRY_MODE=open` returns to buying at the open.
