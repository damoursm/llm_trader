# Selection short — runbook for the live strategy

**Status: LIVE from 2026-09-28 — it makes every trade, with TWO arms** (user
directive 2026-09-26: "Deploy 'Short the most volatile name' ... alongside the
current short model ... over time we'll evaluate which one is the best"): the
MODEL's pick and the most-volatile-name RULE's pick, each through the same trade
rule — which, from 2026-09-28, shorts only names whose short interest is under
one day of volume (user directive 2026-09-27). Everything else (the rank rule,
follow-through, the LLM synthesis) is computed and persisted as SHADOW.
Code `src/signals/sel_short.py` (scorer), `src/performance/tracker.py`
(`record_sel_short_trades`, `monitor_sel_short_positions`,
`flatten_legacy_positions`), wiring in `src/pipeline.py`, settings `sel_short_*`,
tests `tests/test_sel_short.py`. What is in production and why: CLAUDE.md, top
section. Evidence: `memory/selection-model-exits-2026-09.md`; deployment record:
`memory/sel-short-deploy-2026-09.md`.

## 1. What it does

- **Model (V2 from 2026-10-02).** `fx_v2_le2026-04-30` (user directive 2026-10-02:
  "Deploy to live production the V2 model arm"): the same tail regressor trained on
  today's names AND the companies delisted since 2021, with v1's 146 inputs + 15
  (`src/signals/sel_v2.py`: run-up, own-ATR margin, relative volume; reverse splits
  and SEC distress / dilution filings at 08:30 ET; the bar's ATR% / run-up ranks and
  sector-relative values) + the 6 base cross-sectional ranks, the 8 yfinance
  earnings / analyst inputs BLANKED. Its cross-sectional inputs need every name, so a
  bar is scored in one place (`score_v2`). 2025-26 survivorship-free: +1.20 %/day vs
  v1's +1.00 (+0.20, 95% -0.28..+0.67; May-Sep 2026 -0.51). v1 below is the rollback.
- **v1.** `sel_models` `tailreg_long_400`: a LightGBM tail regressor fit on
  sessions <= 2026-04-30, 146 features — the 85 base OHLCV/leg features of the
  30-minute bar (`ml_model.features_30m_from_hlc`, the training function) plus 61
  per-ticker deep features from the pre-open session snapshot
  (`deep_features.serving_vector`).
- **Universe.** The 3,430 deep-store names whose 20-session mean regular-hours
  dollar volume is >= $5M (fixed once a day, ~2,150), scored at price >= $5.
- **Rule.** Every regular-hours 30-minute bar, the TOP-1 score is the candidate.
  It is taken only when FRESH (above its own highest score over the previous 30
  sessions, or fewer than 10 scores) and it is the name's first fresh pick that
  day — `eval_metrics.selection_entries`, pinned by a parity test.
- **The vol arm** (`enable_sel_short_vol`). The same run applies the SAME rule to a
  second score: each name's 30-minute ATR% (`sel_short_vol_feature` = `atr_pct_14`,
  a base feature the scorer computes anyway; no deep snapshot needed). Its own
  freshness history is `scores_vol/`, judged over `sel_short_vol_own_window_days`
  (20) sessions (user directive 2026-09-27; the model keeps 30); "first pick of the
  day" counts its own decisions only. Measured with live-faithful mechanics
  (scratchpad `rank_params.py`: every bar, the $5 floor on the traded price, one
  position per ticker, NBBO costs; return per day = the average trade's return over
  its average 24-hour holding period): 72 trades Jan–Sep 2026, +17.9%/trade over
  9.7 days, 1.71 %/day (95% 0.88–2.62), against the model's 139 at +1.1%/trade,
  0.11 %/day (−0.77–0.87); the model's median trade is +9.4%, its average dragged
  down by five squeezes (−106% to −350%). The live run is the comparison.
- **The vol arm's added stocks** (user directives 2026-10-05: "New listings + test the
  rest", then "Add them to live and backtest the results are good";
  `enable_sel_short_vol_added_stocks`). The model's name list is frozen at its install,
  and ~1,000 liquid common stocks sat outside the deep store — and outside the backtest.
  Each `--prepare` screens Polygon's whole-market daily bars (`grouped/`) for common
  stocks / ADRs (Polygon type CS / ADRC) outside the deep store that averaged ≥ $5M a
  day over ≥ 10 of the 20 sessions before the day, whatever their listing date
  (`screen_listings`; nothing of the day itself is read — `tests/test_no_lookahead.py`).
  `add_listings` fetches each one's 30-minute history into the deep store, refreshes the
  deep universe (its short interest and days to cover arrive with the next deep
  refresh), records it in `vol_listings.json` and seeds its `scores_vol/` history with
  the bars the live scorer would have scored (400 bars visible, the price and
  dollar-volume floors, existing day files only) — without that history the name would
  read as fresh on its first live bar. The vol arm ranks them beside the model's names;
  the model arm never (no model score), the ETF arm never (not a product): the arms
  route by NAME (`arm_rows`), never by status. 103 recent listings added 2026-10-05
  10:35 ET (CBRS, HONA, SKHY, GLND…), ranked from that day's 11:00 bar; the established
  ones (Microchip, Cheniere, BP…) added after that day's close, ranked from the 10-06
  prepare with their days to cover known. Evidence: the pre-registered backtest with
  every common stock outside the store (2,729 names, Feb 2021 – Sep 2026,
  `memory/vol-arm-every-stock-2026-10.md`) kept the rule's edge per trade (+2.15 %/day
  on their 133 trades) and brought two squeeze margin calls (AMTD Digital 2022, SMX
  2025) — verdict "keep out"; with the live 400-bar floor on both sides growth went
  +150% → +169% a year with no margin call, not significant. A stock's ATR% waits for
  400 bars (~31 sessions), as for every live name (47 of the live rule's 397 backtest
  trades came earlier than live could take them).
- **The thin stocks — DECOMMISSIONED 2026-10-08** (user: "Decommission the Thin arm"; `enable_sel_short_thin` False: no thin screen, universe or picks; the text below describes the code kept for a re-test) (`enable_sel_short_thin`, user directive 2026-10-07: "Add the thin
  stocks to the live vol arm"; live from the 2026-10-08 prepare). Common stocks / ADRs at
  $5+ trading $1M (`sel_short_thin_min_dollar_volume`) up to the $5M floor a day (20-session
  mean, fixed at the prepare) — the names the vol arm's floor leaves out. The prepare screens
  the whole market for thin common stocks outside the deep store (`screen_listings(band="thin")`,
  sessions before the day only) and adds them like the added stocks (`add_listings(arm="thin")`,
  recorded with `"band": "thin"`), then writes `universe_thin/<day>.json` (products excluded).
  The run scores them beside the universe as VOL_ONLY rows (no model vector under the vol
  floor) flagged `thin` by name; arm `thin` ranks ONLY those rows with the vol rule — its own
  freshness history `scores_thin/`, the same filters, same-bar fallback, give-back and exits —
  so a thin name never takes a liquid pick's slot nor enters the vol history. FREE CAPITAL
  ONLY: the entry step settles every other arm's picks first and funds a thin short only while
  the margin room of `sel_short_thin_reserve_slices` (2) live slices stays free (`thin_reserve`).
  Evidence (PREREG32, 2021-02 .. 2024-06, one account): final growth +27.2 vs +10.7 %/yr,
  drawdown 46% vs 12%, time-weighted +3.2 pp/yr (98.33% -28.5..+24.2) — not significant;
  2021-26 +37.1 vs +28.3 %/yr with 6 margin calls. Deployed on the user's order.
- **The vol2 arm** (`enable_sel_short_vol2`, user directive 2026-10-07: "Deploy it as another
  version that will also make real trades in the paper account"; live from the 2026-10-08
  session). The vol rule WITHOUT the crowding filter, the relative-volume filter and the
  volatility exit: the vol arm's universe, ranking (30-minute ATR%), freshness against its own
  copy of the vol history (`scores_vol2/`, seeded from `scores_vol/` on 2026-10-07; new vol
  listings are backfilled into both), first fresh pick of the day, riser rule, same-bar
  fallback (its backups unfiltered too), whole give-back target, 15-session limit and 6x cover;
  funded by the same simulated account. Its pick is usually the vol arm's own: when both arms
  short a name, each opens its own trade and its own IBKR order. Evidence (PREREG37,
  2021-02 .. 2024-06, one $10,000 account): +20.7 vs +11.7 %/yr (n.s.), return per day 0.36 vs
  1.44 %, 6 vs 1 margin calls over the 2021-26 context. Trades carry `sel_arm="vol2"`.
- **Several arms on one name** (user directive 2026-10-07: "if multiple arms pick the same
  stock, still order for each one of them"): every arm's pick opens its own trade and sends its
  own order at IBKR (its own reference, its own share count), whether the other arm's order is
  working or already filled; the fallback's one-trade-per-name-per-day rule counts the arm's
  OWN trades only.
- **The ETF arm** (`enable_sel_short_etf`, user directive 2026-10-02: "Add the ETF
  vol as a new arm"; live from the 2026-10-05 prepare). The vol arm's rule on the
  EXCHANGE-TRADED PRODUCTS alone (ETF / ETN / ETV / ETS, leveraged and inverse
  included; `sel_short.etf_names`): top-1 ATR%, fresh against its own
  `scores_etf/` over `sel_short_etf_own_window_days` (20), first fresh pick of the
  day, then the trade rule with the relative-volume filter. `prepare` adds the
  2,012 products the deep store gained on 2026-10-02 (status `VOL_ONLY`: ATR% and
  days to cover, never a model score); the model arm keeps the model's names, the
  vol arm those plus its added stocks. Single-stock funds journal `underlying` / `underlying_side` /
  `underlying_dtc` (journal-only). Measured 2021-26 with delisted products: 141
  trades, +0.78 %/day (95% −0.03…+1.58), the worst tail 2x long single-stock funds
  (HIMZ −238%). The live run is the comparison.
- **Trade.** SHORT it only when it ROSE over the 5 sessions before the pick (the
  close vs the last close at or before the same bar 5 sessions earlier, whatever its
  age; a base from before a gap in the history — a reused ticker stitching two
  securities, an IPO on a reused symbol, a long halt — is flagged `pre5_stale`,
  journal-only), its short interest is
  under one day of volume (the short-interest filter, next bullet) and IBKR can
  lend it (>= $10k available, any fee). Flat size: `sel_short_size_multiplier` (1.0) × the
  base order (2,000 CAD). No gate cascade, no sizing chain. Both arms trade through
  this. There is NO one-position-per-ticker limit (user directive 2026-09-28:
  "Remove the one position per ticker rule"): a repeat pick of a name the book
  already shorts opens ANOTHER trade (`sel_stack_n`), netted into one position at
  IBKR — the broker sync sizes each entry and cover by what each open trade owns;
  a name two arms pick on the same bar opens ONE TRADE PER ARM, each with its own
  target, the model's first (user directive 2026-10-04: "Two trades, one per arm";
  until then one trade stamped `sel_arm = "model+vol"`); a name another book holds is
  never stacked on.
  `SEL_SHORT_MAX_OPEN_PER_TICKER` = 1 restores the old rule (a repeat pick noted on
  the open trade, `sel_also`). Measured on 2025 + Jan–Sep 2026: model 0.87 → 1.04
  %/day, vol 1.99 → 2.22, not significant; up to 3–4 shorts stacked on one name.
  Every trade carries `sel_arm` (`model` / `vol` / `etf`; `model+vol` only on trades
  opened before 2026-10-04).
- **The short-interest filter** (user directive 2026-09-27: "Add in live production
  the 'Under one day of volume' filter"; `enable_sel_short_dtc_filter`, both arms).
  The pick's FINRA days to cover — short interest ÷ average daily volume, floored at
  1.00, read from the session snapshot (`dp_si_dtc`, known at the 08:30 ET cutoff) —
  must be at most `sel_short_max_days_to_cover` (1.0). Above it the scorer journals
  the pick `crowded` (with its target and deadline, so it can be followed) and the
  tracker never sees it; it still counts as the name's pick of the day, as in the
  evaluation. An unknown value passes, with a warning. Every trade stamps
  `sel_days_to_cover`. Measured on the live-faithful trades (scratchpad
  `gross_net_dtc.py`, the filter at the trade step; return per day net, 24 h basis):

  | Arm | Without: trades, %/day | With: trades, %/day | Gain, 95% range |
  |---|---|---|---|
  | vol, Jan–Apr / May–Sep | 19, 1.22 / 53, 1.92 | 13, 1.94 / 40, 2.59 | +0.72 (−0.55, +3.20) / +0.67 (−0.15, +1.80) |
  | model, Jan–Apr / May–Sep | 67, −0.63 / 72, 1.01 | 20, −0.46 / 36, 3.28 | +0.17 (−1.26, +1.73) / +2.27 (+0.53, +5.55) |
  | vol, Jan–Sep | 72, 1.71 | 53, 2.41 | +0.70 (−0.02, +1.66) |
  | model, Jan–Sep | 139, 0.11 | 56, 1.41 | +1.30 (+0.23, +2.80) |

  On the record: the filter was chosen AFTER seeing these trades (7 traits × 2 arms
  looked at), the model's gain is all May–September, and losses beyond −50% came at
  the same rate with and without it — it selects better ordinary trades, it does not
  stop squeezes. The live trades are its test.
- **The relative-volume filter** (user directive 2026-10-01: "implement relative volume
  filter to prod"; `enable_sel_short_vol_rvol_filter`, VOL ARM ONLY). The pick bar's
  volume ÷ the mean volume of the name's previous 260 regular-hours 30-minute bars
  (`sel_short.rvol_at`; 20 sessions, ≥ 20 bars needed) must be at least
  `sel_short_vol_min_rvol` (1.58 = the 20th percentile of the live rule's trades
  2021–26; no outcome used to set it). Below it the scorer journals the pick
  `low_rvol` (target and deadline kept) and the tracker never sees it; it still counts
  as the name's pick of the day. An unknown value passes. Every pick of both arms
  journals `rvol`; the model arm is not filtered (never tested). Measured on the vol
  engine (scratchpad `rvol_table.py`; return per day net, 24 h basis; 2026 = picks to
  early September; the delisted names with their FINRA days to cover by symbol, 2026-10-01):

  | Year | With delisted names: live → filtered (trades, %/day) | Today's names only: live → filtered |
  |---|---|---|
  | 2021 | 97, 2.19 → 90, 2.16 | 69, 1.45 → 60, 2.22 |
  | 2022 | 73, 3.05 → 62, 2.86 | 39, 3.36 → 33, 3.44 |
  | 2023 | 79, 1.38 → 58, 1.64 | 43, 1.00 → 33, 1.02 |
  | 2024 | 81, 2.10 → 61, 2.67 | 59, 2.74 → 46, 2.77 |
  | 2025 | 88, 2.33 → 66, 2.24 | 60, 1.60 → 45, 1.55 |
  | 2026 | 78, 2.21 → 53, 2.92 | 72, 2.80 → 47, 3.21 |
  | **2021–26** | **496, 2.18 → 390, 2.40 (+0.22, 95% −0.03…+0.54)** | **342, 2.09 → 264, 2.34 (+0.26, −0.06…+0.66)** |

  On the record: relative volume was FOUND by searching (it drove the re-fitted
  confidence that filtered live best, `memory/relax-confidence-study-2026-10.md`), so
  these years cannot test it; the 20% it removes earned ~1.4 %/day. The in-sample gain is
  FRAGILE: this book's own 20th percentile is 1.52, and between 1.52 and 1.58 sit six
  trades, one a −167% squeeze — at 1.52 the gain is +0.08 (−0.12…+0.32). A cut set each
  year from the previous years only gives +0.46 (+0.07…+0.97) over 2022–26. The live
  picks are its test — compare the journal's `low_rvol` picks (followed to their target /
  deadline) with the traded ones. Other cuts and the model arm (2026-10-01, delisted
  names included): stricter cuts (25–40% of trades) reach 2.40–2.55 %/day against 2.40
  at the deployed cut (none significant) for 5–24% fewer trades, and choosing the cut
  from past years does not beat a fixed 20% out of sample — keep 20%. The MODEL arm gains nothing at the same cut
  (+0.04 %/day, 95% −0.20…+0.32) — it stays unfiltered (`memory/relax-confidence-study-2026-10.md`).
- **Exit.** Cover when the price has given back the arm's share of the 5-session run-up
  (`target = close − share × (close − close 5 sessions before)`, `sel_short.give_back`):
  HALF for the model and ETF arms (`SEL_SHORT_GIVE_BACK` 0.5), the WHOLE run-up for the VOL
  arm (`SEL_SHORT_VOL_GIVE_BACK` 1.0: the target is the close 5 sessions before; user
  directive 2026-10-04 evening, "Vol arm only" once the per-arm numbers were in). Evidence,
  Feb 2021 – Sep 2026, audited compounding engine at 8 slices: the vol arm's growth +105% →
  +200% a year (log +0.38, 95% +0.21…+0.54, 6 of 6 years) while its return per day falls
  2.88 → 2.19; at 100% the model arm fell to 0.30 %/day from 1.20 and lost money at every
  slice count, and the ETF arm's growth did not change (0.78 → 0.53 %/day), so both keep half
  (`memory/vol-give-back-slices-2026-10.md`). Judged at EVERY tick in
  every session (user directive 2026-09-27: "We should always have the possibility to
  enter or exit at any time"), on a mark no older than `sel_short_mark_max_age_minutes`
  (45); otherwise at the first tick at/after the same bar 15 sessions later. No stop
  besides the vol arm's squeeze cover (below).
  (The evaluation judged RTH 30-minute closes: covers outside regular hours are new.)
- **Volatility-normalised exit** (user directive 2026-09-28: "Implement the Cover when
  volatility halves (in profit) for the two live models in production";
  `enable_sel_short_volnorm_exit`, both arms). A short that is IN PROFIT (a fresh mark
  below the entry price) is also covered once the name's 30-minute ATR% at the latest
  completed regular-hours bar has fallen to `sel_short_volnorm_ratio` (0.5) of its value
  at the pick bar (`sel_volnorm`; the target is checked first). The pick's ATR% is
  journaled (`atr_pct`) and stamped on the trade (`sel_atr_pct`); the live value is built
  exactly like it (`sel_short.live_atr`: the deep store + that session's Polygon bars,
  the training feature function) for any held name — also one that fell out of the
  day's universe — and only for trades in profit. The cover stamps `sel_exit_atr_pct` and
  `sel_exit_atr_bar`. Measured on 2025 + Jan–Sep 2026 (live-faithful, net, with the
  filter): vol 1.67 → 2.08 %/day (+0.41, 95% −0.17…+1.14; +0.49 in 2025, +0.32 in 2026),
  model 0.85 → 0.91 (+0.05, −0.20…+0.37) — not significant; the live trades judge it.
  Every other adaptive exit tested (20 rules: trails after the target, re-anchoring,
  break-even, scale-outs, longer holds) was worse or flat (`memory/short-adaptive-exits-2026-09.md`).
- **The squeeze cover, every arm** (user directives 2026-10-05: "have it tuned so that we
  don't have margin calls while still maximizing growth", then the cover for every arm;
  `enable_sel_short_squeeze_cover`, `sel_short_cover_multiple` 6.0; tuned on the vol arm, the
  model and ETF arms take it untested). A short whose fresh mark reaches 6× its entry
  price is bought back (`sel_cover`, judged at every tick like the target), the entry
  carried through every split executed after the entry day and on/before today
  (`intraday_store.split_factor_between` on the deep `splits` family: these names
  reverse-split often, and a 1-for-10 mid-hold is not a 10-fold squeeze; unreadable
  split data falls back to the raw entry). Why: slices limit what a short weighs when it
  opens, not after a squeeze — alone in the account a short at 1/k under IBKR's 200%
  maintenance is margin-called once it rises (k + 1)/3-fold (8.3 at 24 slices); on every
  common stock 2021–26 AMTD Digital (~13-fold in two sessions, 2022) and SMX (~10-fold,
  Dec 2025) went past any slice count's limit. Tuned with the slices under IBKR's house
  margin (pre-registered PREREG12, 7 cover levels × 10 slice counts, both universes, six
  starts, calls at closes and highs, the safe setup whose safer neighbours are safe too,
  the best mean growth over the 2021–25 starts): 6× with 24 slices, 1 of 447 trades
  covered, return per day 2.23 → 2.19 (`memory/ibkr-house-margin-2026-10.md`).
- **The simulated account every arm is sized from** (user directives 2026-10-05: "Have the
  account based sizing considering the simulated 5000$+1000$ every two weeks", then every arm;
  `src/performance/sim_account.py`, `enable_sel_short_account_sizing`, `sel_short_account_*`).
  The paper account holds ~CAD 991k; the plan is a real account of $10,000, no deposits (user
  directive 2026-10-07). That account is replayed from the ledger: its equity is the
  $10,000 plus every funded trade's
  dollar P&L (shares × entry × the ledger's net return, spreads, commissions and borrow
  included; open trades at their mark). A new short (every arm) gets the audited replay engine's size:
  floor(min(equity / 24, equity − the open shorts' value, 1% of the stock's 20-session dollar
  volume) / price) shares, within the initial-margin room, none while equity is under FINRA's
  $2,000; a pick no share fits is journaled `account_full`, `volume_cap`, `account_too_small`
  or `account_below_minimum`. The trade stamps `sel_account_shares` / `sel_account_notional` /
  `sel_account_equity` / `sel_account_slices`, and the broker orders exactly that count, also
  when it re-anchors a resend (`reconcile._entry_qty`). Equity below the open shorts'
  maintenance is a margin call: every funded short is bought back (`sel_margin_call`, a
  CRITICAL log), as a broker would. The three arms share the account's equity and room (the
  slices were tuned on the vol arm alone).
- **IBKR's margin, and the what-if at entry** (user 2026-10-05: "Start with 1", then "Do both,
  deploy 24 slices and build the what-if"). The account's margin is the larger of Reg T's
  tiers (maintenance 30% or $5 a share at $5 and above, 100% or $2.50 a share below) and
  IBKR's HOUSE rate, a multiple of the short's value: IBKR's what-if quotes on 2026-10-05 put
  the volatile names the arms short at a median 2.00 maintenance / 2.86 initial, one name in
  seven at 3.8–41, and refused any opening short in 13 names. For every funded pick the entry
  step asks IBKR once per name per pass (`sim_account.ibkr_margin` → `IBKRBroker.what_if_short`:
  a ~$500 DAY limit SELL, priced by IBKR, never transmitted; `enable_sel_short_ibkr_whatif`,
  bounded by `sel_short_whatif_timeout_seconds` 10 s):
  - a REFUSAL (error 201 "No Trading Permission": risk-management close-only, or "No Opening
    Trades: Small Cap, Subject to Compliance Restriction") skips the pick as `ibkr_refused`
    — before, its order was refused and resent for six ticks;
  - the quoted RATES size the pick (an initial rate above the default 2.86 shrinks it by
    2.86 / rate, so no short ties up more initial margin than a default one) and stay with the
    trade for its life: `sel_house_maint` / `sel_house_init` / `sel_house_source`
    (`ibkr_whatif` | `default`), the raw answer `sel_ibkr_whatif`, journaled `ibkr_margin` in
    `entries/` and carried into `tradelog/`;
  - NO ANSWER (no IBKR session, a timeout, an unset figure) = the defaults
    `sel_short_account_house_maint` 2.00 / `_init` 2.86.
  NEVER IN BULK: two bulk probes (66 and ~100 what-ifs in a minute) each coincided with the
  gateway losing IBKR (20:07, a refused re-login and a 10-minute outage; 20:27, 13 s); runs of
  5–43 names spaced 2 s apart did not. Evidence: 24 slices with the 6× cover at 2.00 / 2.86
  (PREREG12): no margin call from any start date at closes or highs, mean money-weighted
  growth +57% a year over the 2021–25 starts (+56% money-weighted / +55% time-weighted from
  Feb 2021, worst drawdown 27%), where 10 slices were margin-called 255 times (+35%); with the
  rates drawn from the 36 measured names, a call in every draw without the shrink and none in
  200 draws with it — the shrink was designed after that check (`memory/ibkr-house-margin-2026-10.md`).
- **Corporate-action gaps** (user 2026-10-05: spin-offs are in neither the split nor the
  dividend data). CTVA's 2026-10-01 spin-off (77.65 → 14.44 at the open) read as a ~40%
  ATR% decaying over days: the most volatile name of every bar, never fresh, never a riser —
  the vol arm took nothing for three sessions. `sel_short.corporate_gap_flags` flags a bar
  when, within 5 sessions, a session OPENED at least 40% under the previous close (its first
  bar's high) and the 30-minute ATR without session-opening gaps is under 40% of the full
  one: a level shift, not volatility (a crash that keeps trading wildly stays ranked). The vol
  and ETF arms leave flagged bars out of their ranking and their freshness history (`arm_rows`;
  each scored name carries `ca_gap` / `ca_gap_pct`, the backfill applies it too) and the
  decision journals who was left out (`ca_gap_excluded`). Such a name is no riser for the run-up
  window anyway, so the rule only frees the slot. Backtest (every stock, live floor, 6× cover,
  10 slices, 2021–26): 437 → 447 trades, return per day 2.25 → 2.18 (95% −0.19…+0.02), growth
  +149 → +150% a year — neutral; 490 names flagged, 1,032 eligible bars. Bars written to the
  history before the fix keep CTVA unfresh for 20 sessions.
- **Metrics (user directives 2026-10-04).** Every change is judged on two numbers,
  both reported: RETURN PER DAY = (1 + average net trade)^(1 / average 24-hour hold) − 1,
  and GROWTH PER YEAR = a $10,000 account, no deposits (from 2026-10-07),
  compounding through the audited replay engine (each short a stated share of the
  account, IBKR borrow, Reg T margin with forced buy-backs, the $2,000 minimum, ≤ 100% of
  the account and ≤ 1% of daily dollar volume) — without deposits money-weighted = time-
  weighted: the account's yearly growth. The two metrics can disagree; the user chooses
  (`memory/metrics-growth-and-return-per-day.md`).
- **Performance (live-faithful re-test, 2026-09-27).** Same trade rule, every bar a
  run, the $5 floor on the traded price, one position per ticker, NBBO costs, today's
  borrow fees; return per day = the average trade over its average 24-hour hold:

  | Arm | Trades Jan–Sep | Per trade | Per day (95% range) | Jan–Apr / May–Sep per day | Worst |
  |---|---|---|---|---|---|
  | model, no filter | 139 | +1.1% (median +9.4%) | 0.11% (−0.73…0.89) | −0.63 / 1.01 | −350% |
  | model, filter | 56 | +11.4% (median +15.1%) | 1.41% (0.16…3.03) | −0.46 / 3.28 | −122% |
  | vol, no filter | 72 | +17.9% (median +25.1%) | 1.71% (0.90…2.60) | 1.22 / 1.92 | −150% |
  | vol, filter | 53 | +24.0% (median +27.0%) | 2.41% (1.64…3.43) | 1.94 / 2.59 | −89% |

  The deploy-time evidence for the model (+11.3%/trade, +1.40 %/day on 156 trades) came
  from looser mechanics (3 bars a session in Jan–Apr, the split-adjusted $5 floor, no
  one-position rule, session days) and overstated it. Missing 1–2 ticks and retrying
  (entry or exit) moves return per day by −0.13…+0.18, every 95% range including 0;
  delayed exits are the costlier side (scratchpad `missed_ticks.py`).

## 2. A market day

| ET | What runs | Where to look |
|---|---|---|
| 08:30 | Deep pre-open run (subprocess, on its own timer — never behind the 08:30 tick): the fast families in concurrent lanes — short volume / interest in one market-wide sweep each, SEC filings re-read only for the companies EDGAR's live feed shows filing since the nightly full pass — with the 30-minute store extended beside them, then today's session snapshot (the same rows as the old one-after-another run, which took 47 min + 3 min + 3 min) | `logs/deep_preopen_console.log`, `cache/ml/deep/snapshot/<date>.parquet` |
| 09:00 tick | `sel_short --prepare`: the store extension (usually the one that does the work — whichever of the two daily extenders runs second finds nothing to fetch), the day's universe, a coverage check of the snapshot if it already exists | `cache/ml/sel_short/prepare/<date>.json` |
| 09:30 tick | no bar has closed yet: marks and exits only. The one-shot legacy flatten ran at this tick on 2026-09-28 (09:44: the 22 legacy trades closed, IBKR flat once their exits filled); its marker `cache/legacy_flatten_done.json` keeps it from running again | scheduler log `[sel_short] legacy flatten: closed N` |
| 10:00 → 16:00 ticks | `sel_short --run --day D --bars k…` for every completed bar not yet run, oldest first (bar 0 = 09:30–10:00 … bar 12 = 15:30–16:00): split check, fetch, score ~2,150 names (~2–3 min a bar), journal the decision. The tick's LIVE PATH (start of the tick, while the shadow pipeline fetches) waits for it, opens the pending short and the broker sync sends it | `logs/sel_short.log`, `runs/<date>_<k>.json`, `picks/<date>.jsonl` |
| every tick | `monitor_sel_short_positions`: `sel_target` / `sel_cover` (vol) / `sel_volnorm` / `sel_time` covers, in every session (the live ATR% is fetched only for trades in profit) | scheduler log, the ledger |
| every tick | `sel_short.health()`: the email's scorer banner — red with the problems + a 🔔 SCORER subject tag (a completed bar unscored 10 min after it ended, a crashed bar, a `thin_run`, a scorer exit ≠ 0, a pre-open snapshot missing or defective that no run has rebuilt yet, or one whose rebuild failed); otherwise a green line (runs today, snapshot coverage) whose notes carry what the scorer fixed itself (a defective snapshot it rebuilt) | the email |
| after 17:00 | nothing launches | |

The scorer runs as a SUBPROCESS launched at the start of each tick. The tick's
trading path runs FIRST (`enable_live_path_first`): marks and exits before
anything else, then — on the main thread while the shadow pipeline's Steps 1-3
fetch — it waits for the scorer (at most `sel_short_wait_seconds`, 480 s; never
for a `--prepare`), opens the picks and syncs the broker. The end-of-tick pass
re-judges exits, takes a late scorer's picks and syncs again. A pick the tick could not take (the run was slow, no live
price) is taken by a later tick while it is <= `sel_short_entry_max_age_minutes`
(75) old; after that it is marked `expired`.

## 3. Files (`cache/ml/sel_short/`)

| File | Content |
|---|---|
| `model.txt`, `model.json` | the booster and its feature order: V2 (`--install-v2`; model.json lists `extra_features` and the blanked `masked` inputs) |
| `model_v1.txt`, `model_v1.json` | v1 (`tailreg_long_400`), kept for a rollback |
| `extras/<date>.json` | V2's per-session inputs (reverse splits, SEC distress / dilution filings at 08:30 ET) for the day's names — written by the prepare, else the first run |
| `universe/<date>.json` | the day's names and their 20-session $ volume |
| `prepare/<date>.json` | the prepare record (extension counters, universe size, snapshot coverage) |
| `scores/<date>.pkl` | every bar's V2 scores for every scored name — the FRESHNESS history the rule reads over 30 sessions (rebuilt with V2 for 2026-08-21 → 10-02 by `--backfill --arms model`) |
| `scores_v1/<date>.pkl` | v1's history (archived at the V2 install; never delete — the rollback reads it) |
| `scores_vol/<date>.pkl` | the vol arm's history: every bar's ATR% (seeded from the arrays' own column, 09-15 → 09-25 backfilled) |
| `scores_etf/<date>.pkl` | the ETF arm's history: every bar's ATR% of the exchange-traded products (seeded 2026-10-02 for the 20 sessions 09-04 → 10-02, `--backfill --arms etf --tickers … --merge`) |
| `scores_thin/<date>.pkl` | the thin stocks' history: every bar's ATR% of the names in that day's thin universe (seeded 2026-10-07 for the 20 sessions before 10-08, `--backfill --arms thin`) |
| `scores_vol2/<date>.pkl` | the vol2 arm's history: the vol arm's (copied from `scores_vol/` on 2026-10-07, then appended by each run) |
| `universe_thin/<date>.json` | the day's thin stocks (common stocks at $5+ trading $1-5M a day) and their 20-session dollar volume |
| `etf_underlying.json` | single-stock fund -> underlying and side (431 funds; journal-only) |
| `picks/<date>.jsonl` | one decision per bar PER ARM (`"arm": "model"` / `"vol"`): `short`, `crowded` (a short the short-interest filter refused — target and deadline kept, never traded), `not_fresh`, `not_first_today`, `not_a_riser`, `thin_run` (< 20 names scored); every pick carries `days_to_cover` (null = unknown), `pre5_stale` (the run-up's base close is from before a gap in the name's history; journal-only) and, JOURNAL-ONLY (2026-09-29, for a blind test of a confidence score — nothing decides on them), its candidate confidence components: `run_mean` / `run_std` / `z_in_run` (the pick against the bar's cross-section), `runner_up` / `runner_up_score` / `gap2_z` (the margin over the bar's second name), `own_margin_z` (the freshness margin; null under 10 priors), `other_arm_score` / `other_arm_rank` / `other_arm_n` (the other arm's view of the same name) and `dv20`; each VOL pick also journals its `confidence` (the frozen rule `src/signals/sel_short_conf_vol_v1.json`, stamped `sel_confidence` on the trade) — journal-only: a bottom-third filter on it lowered return per day on 2021–24, years it never saw (`memory/vol-arm-anatomy-2026-09.md`) |
| `vol_listings.json` | the vol arm's added listings: ticker → added day, Polygon type, listing date, name (never delete — the vol arm ranks exactly these beside the model's names) |
| `grouped/<date>.pkl`, `ticker_details.json` | the listings screen's inputs: Polygon's whole-market daily bars per session, Polygon's reference record per candidate (an unknown symbol is asked again after 20 h) |
| `entries/<date>.jsonl` | every pick the entry step settled (2026-10-05): `outcome` (`opened`, `no_borrow_retry` — IBKR cannot lend it now; journaled at every check, the pick stays pending inside its 75-minute window (2026-10-07) —, `no_borrow` (the retry off), `no_borrow_fallback` (the vol pick settled by the same bar's lendable backup, PREREG16), `halted_retry` (halted at the entry tick: re-checked next tick), `traded_today` (a backup already traded that day), `ibkr_refused`, `borrow_fee`, `target_reached`, `already_open`, `expired`), the live `price`, the pick's `ssr`, and IBKR's borrow row at that moment (`borrow_checked`, `borrow_listed`, `borrow_available`, `borrow_fee_pct`, `borrow_file_ts`) — for the picks NOT taken too |
| `tradelog/<date>.jsonl` | each `short` end to end (2026-10-05): the pick (target, days to cover, relative volume, ATR%, the short-sale restriction `ssr` + its inputs), the entry step, the ledger trade (entry, exit, reason, return) and the broker's legs `broker_entry` / `broker_exit` (status, filled / requested, fill price, commission, attempts, refusals and their errors, the book and limit at the first submit, first fill time, seconds to fill, slippage against the ledger in bp — positive = worse) + `broker_return_pct` (the fills' gross return when both legs filled); the prepare rewrites the last 25 sessions |
| `picks/<date>.consumed.json` | what the tracker did with each `short` (vol keys end in `|vol`; a fallback trade's key is its OWN bar-and-ticker key): `opened`, `already_open`, `target_reached`, `no_borrow` (retry off), `no_borrow_fallback`, `ibkr_refused`, `traded_today`, `expired` — an unlendable or halted pick is NOT consumed while its window lasts |
| `runs/<date>_<bar>.failed.json` | a bar whose run crashed (count + error); retried by the next launch, skipped after 2 |
| `launches/<date>.jsonl` | each scorer launch's arguments, exit code and duration (the email reads it) |
| `snapshot/<date>.json` | what the day's first snapshot check found (`found` coverage, `rebuilt`, final `coverage`, `error`) — a missing/defective pre-open snapshot shows in the email |
| `runs/<date>_<bar>.json` | the run record: the model's decision (+ the vol arm's under `arms`), status counts (`OK` / `NO_BAR` / `FETCH_FAILED` / `NO_SNAPSHOT` / `NO_DATA` / `ERROR`), n scored, `n_days_to_cover` (names that carried one), `snapshot_coverage`, seconds |
| `busy.lock` | held by a running scorer, touched every 30 s; stale after 180 s |

Never delete `scores/`, `scores_vol/`, `scores_etf/`, `scores_thin/` or `scores_vol2/`: without them every name reads
as "no history" and passes the freshness rule. Seeding history for NEW names uses
`--backfill --tickers … --merge` (without `--merge` a backfill rewrites whole day files).

## 4. Is it working?

```powershell
Get-Content logs\sel_short.log -Tail 20            # one line per run: "<day> bar <k>: model <decision> <ticker> | vol <decision> <ticker> (scored n/N, s)"; a crowded pick adds "(days to cover x)", a low_rvol one "(relative volume x)"
Get-Content cache\ml\sel_short\picks\<date>.jsonl  # every bar's decision
Get-Content cache\ml\sel_short\picks\<date>.consumed.json
Select-String logs\llm_trader_<date>.log -Pattern "\[sel_short\]"   # launches, OPEN SELL, covers, the flatten
```

Healthy: 13 runs a day (bar 0 at ~10:03 … bar 12 at ~16:03), `n_scored` ~1,950–2,000,
`n_days_to_cover` close to the universe (all 2,147 names carried one on the
2026-09-25 snapshot; only ~5% of them sit at the 1.00 floor, but ~40% of the
model's picks and ~70% of the vol arm's did), `NO_BAR` a few dozen (names that did
not trade that bar), `FETCH_FAILED` ~0, and about one opened short every other
session (the evaluation with the filter, both arms as one book — they often pick the
same names: 0.6 per session, held ~7.7 days, so ~3 open on average, 5 at the 90th
percentile, 7 at most — ~$4.6k of entry value at ~$1,414 each, ~$10k at the peak;
without the filter ~9 open, ~$13k; scratchpad `capital_filter.py`). A `crowded`
decision is the filter working, not a fault; so is a vol `low_rvol` (about one vol
short in five, by construction of the cut). `n_rvol` (the run's names carrying a
relative volume) should be close to `n_scored`.

The session snapshot must carry the training features. Its coverage (share of the
universe with a previous close) should be ~0.98:

```bash
python -c "from datetime import date; from src.signals import sel_short as s; d=date.today(); print(s.snapshot_coverage(d, sorted(s.load_universe(d) or {})))"
```

Below 0.8 the scorer rebuilds it before scoring (`ensure_snapshot`, ~3 min); a
coverage still low after that means the 30-minute store is behind the previous
session for most names — run `python -m src.data.intraday_store --extend`, then
`python -m src.analysis.deep_features --snapshot <date>`.

## 5. Commands

```bash
python -m src.signals.sel_short --prepare --day 2026-09-29 [--no-extend]
python -m src.signals.sel_short --run --day 2026-09-29 --bar 3      # score one bar; the next tick opens its pick if <= 75 min old
python -m src.signals.sel_short --backfill --day 2026-09-15 --until 2026-09-25   # rebuild score days of both arms (~1 min/day, ~2,000 names)
python -m src.signals.sel_short --backfill --arms vol --day ... --until ...     # one arm's history only
python -m src.signals.sel_short --seed-vol                           # the vol arm's history from the evaluation arrays (keeps existing days)
python -m src.signals.sel_short --install                            # a NEW model only: copies the booster, seeds the history
python -m src.signals.sel_short --install-v2 cache/ml/sel/final/fx_v2_le2026-04-30.txt   # V2 (v1 -> model_v1.*, scores -> scores_v1/); then --backfill --arms model for 30 sessions
python -m src.analysis.deep_features --snapshot 2026-09-29           # rebuild a session snapshot
python -m src.signals.sel_short --listings --dry --day 2026-10-06    # the vol arm's added-stocks screen (no change); without --dry it adds them
python -m src.signals.sel_short --thin-listings --dry --day 2026-10-08   # the thin stocks' screen ($1-5M a day outside the store); without --dry it adds them
python -m src.signals.sel_short --trade-log --day 2026-10-05 [--until ...]   # rewrite tradelog/<day>.jsonl (picks, entry step, ledger, broker fills)
```

Run manual commands only when no scorer holds `busy.lock` (or it is older than
180 s): the tick's `launch` refuses to start while a live scorer holds it, but a
manual run does not check. A day missing from `scores/` inside the 30-session
window is logged by `--prepare` with the exact `--backfill` command to fill it.

## 6. Switches

| Setting | Effect |
|---|---|
| `ENABLE_SEL_SHORT=false` | no scorer, no new entries; OPEN selection shorts keep their two exits (the monitor does not read the flag) |
| `ENABLE_SEL_SHORT_VOL=false` | the vol arm stops journaling picks (the model trades alone); its open trades keep their exits |
| `ENABLE_SEL_SHORT_ETF=false` / `SEL_SHORT_ETF_OWN_WINDOW_DAYS` | the ETF arm off (its products leave the next prepare's universe; open trades keep their exits) / its freshness window (20) |
| `ENABLE_SEL_SHORT_THIN=false` / `SEL_SHORT_THIN_MIN_DOLLAR_VOLUME` / `SEL_SHORT_THIN_RESERVE_SLICES` | the vol arm's thin stocks off (no thin screen, no thin universe, no thin picks; open thin trades keep their exits) / the band's lower floor ($1M) / the live slices of margin a thin short must leave free (2) |
| `ENABLE_SEL_SHORT_VOL2=false` | the vol2 arm off (no vol2 picks; its open trades keep their exits) |
| `SEL_SHORT_OWN_WINDOW_DAYS` / `SEL_SHORT_VOL_OWN_WINDOW_DAYS` | each arm's freshness window in sessions (30 / 20); read by the scorer at each launch |
| `ENABLE_SEL_SHORT_DTC_FILTER=false` | the short-interest filter off: crowded picks trade again (both arms); picks still record `days_to_cover` |
| `ENABLE_SEL_SHORT_VOL_RVOL_FILTER=false` / `SEL_SHORT_VOL_MIN_RVOL` | the vol arm's relative-volume filter off (low_rvol picks trade again; picks still record `rvol`) / its cut (1.58 x the usual 30-min volume); read by the scorer at each launch |
| `SEL_SHORT_MAX_OPEN_PER_TICKER` | open shorts allowed per name (0 = no limit, live; 1 = the old one-position rule) |
| `ENABLE_SEL_SHORT_BORROW_RETRY=false` | an unlendable pick is given up at its first check (`no_borrow`) instead of being re-checked every tick inside its window (2026-10-07) |
| `ENABLE_SEL_SHORT_VOL_FALLBACK=false` / `SEL_SHORT_VOL_FALLBACK_RANKS` | the vol arm's same-bar fallback off (an unlendable or IBKR-refused vol pick is simply not traded) / how deep it reaches (5 = ranks 2-5, live from 2026-10-07; 3 = PREREG16's tested depth; 10 worse — PREREG21); read by the scorer at each launch and by the entry step |
| `ENABLE_SEL_SHORT_HALT_GUARD=false` | names halted at the entry tick are entered anyway, and held shorts are no longer stamped `sel_halted` (NYSE's current-halt list, `src/data/trade_halts.py`) |
| `ENABLE_SEL_SHORT_BORROW_WATCH=false` | held shorts are no longer stamped with IBKR's borrow state (`sel_borrow_state`; journal-only — covering when the pool empties was refused, PREREG19) |
| `BROKER_BUY_IN_CONFIRM_SYNCS` | syncs running on which IBKR must hold fewer short shares than the open trades own before the buy-in alert (3) |
| `SEL_SHORT_MARK_MAX_AGE_MINUTES` | the target and the volatility exit are judged only on a mark this fresh (45) |
| `ENABLE_SEL_SHORT_VOLNORM_EXIT=false` / `SEL_SHORT_VOLNORM_RATIO` | the volatility-normalised exit off / its threshold (0.5 = the ATR% halved since the pick) |
| `ENABLE_LIVE_PATH_FIRST=false` | the old tick order: marks, exits, entries and ONE broker sync at the END of the tick, after the shadow pipeline (entries ~10–40 min after the bar instead of ~2.5 min into the tick) |
| `BROKER_REFUSED_RESENDS_PER_TICK` / `BROKER_REFUSED_MAX_TICKS` | a refused order is resent this many times in the same tick (2); an entry refused on this many ticks is given up (6) — exits never give up |
| `SEL_SHORT_MAX_DAYS_TO_COVER` | the filter's threshold in days of volume (1.0 = FINRA's floor, "under one day of volume"); read by the scorer at each launch |
| `SEL_SHORT_SIZE_MULTIPLIER` | position size vs the base order (1.0) |
| `SEL_SHORT_GIVE_BACK` / `SEL_SHORT_MAX_HOLD_SESSIONS` / `SEL_SHORT_RUNUP_SESSIONS` | the evaluated rule's parameters (0.5 / 15 / 5) — changing them leaves the evidence behind |
| `ENABLE_LEGACY_ENTRIES=true` | the rank rule trades again, beside the selection short; while both `ENABLE_SEL_SHORT` and legacy entries are on, unfilled legacy entries are sent normally again |
| `ENABLE_FOLLOW_THROUGH_TRADING=true` | follow-through trades again |
| `LEGACY_FLATTEN_AFTER` | the one-shot flatten instant; empty = never. It runs once per value (marker file) |

A setting change needs a scheduler restart (`scripts/restart_all.ps1`); the
scorer subprocess reads settings at each launch.

## 7. When something goes wrong

| Symptom | What happened | What the system does |
|---|---|---|
| a traceback in `logs/sel_short.log`, a `runs/<date>_<bar>.failed.json` | the scorer crashed on that bar | the bars after it still run; the next launch retries it (skipped after 2 failures); the email shows it |
| `history RESET (split effective …)` in a log | the split data records a split after the name's history was last adjusted | the whole history is refetched before any inference; a reset during the day also rebuilds the day's snapshot |
| `stored bars rescaled … but the split data records no split … NOT reset` | the stored scale changed with no split on record (a data correction, or a split not in the data yet) | appended as before, never reset on the price change alone (user directive 2026-09-28) — check the name |
| `refused (…) — resending in this tick` / `REFUSED_GAVE_UP` | IBKR refused the order (not shortable, precautions, overnight venue) | resent in the tick, then each tick; an entry is given up after 6 refused ticks |
| `still running after 480s` in the scheduler log | a slow run | the tick trades without it; the next tick takes its pick if <= 75 min old |
| `thin_run` decisions | < 20 names scored (a Polygon outage reads as `NO_BAR` for every name) | no trade that bar |
| `another scorer holds busy.lock` | a previous scorer still running (a long prepare) | this tick launches nothing; the next one retries |
| `session snapshot … covers only N%` | the snapshot was built on a store that was behind | rebuilt before scoring |
| `no scored name carries days to cover` | no session snapshot (the model arm cannot score either) | the filter cannot judge: the vol arm's picks pass it, each warned `days to cover unknown` |
| `<arm> pick <TICKER>: days to cover unknown` | the name has no FINRA short interest in the store | the pick passes the filter (as evaluated) |
| `no scored name carries a relative volume` | the run's series carried no volume (a data fault) | the vol arm's relative-volume filter cannot judge: its picks pass it |
| `no live price — retrying next tick` | no price for the pick | retried until the pick expires |
| `COVER <T> — volatility halved in profit: 30-min ATR x% … vs y% at the pick` | the volatility-normalised exit fired | the trade closes `sel_volnorm`; the exit order goes out in the same tick's sync |
| `COVER <T> — squeeze: mark … at/above 6x the entry … (split-adjusted level …)` (WARNING) | the vol arm's squeeze cover fired | the trade closes `sel_cover` (stamped `sel_cover_level`); the buy-back goes out in the same tick's sync |
| `live ATR% failed (fail-soft, no volatility exit this pass)` | the bars for the held names could not be fetched or built | only the target and the deadline are judged that pass; the next pass retries |
| a short never fills | 69% of RTH short entries fill on the first attempt (25% extended, 9% overnight) | killed at the settle deadline, re-anchored and resent every tick while the ledger row is open |
| 🔔 BROKER banner: `IB Gateway NOT logged in to IBKR since …` | the gateway's IBKR login failed: IBKR's nightly re-login demand that IBC could not complete, or a manual login to this paper account elsewhere (TWS, Client Portal, mobile) | nothing can trade until it logs in. IBKR's re-login demand: the sync restarts the gateway itself (`BROKER_GATEWAY_RELOGIN_RESTART`; the banner says so) — restart the `IBC Gateway` task only if it stays red. A login elsewhere: close that session, then restart the task (never automatic — it would log out your manual session) |
| V2 must be rolled back | — | with no scorer running (`busy.lock` absent or stale): move `scores/` to `scores_v2/`, `scores_v1/` to `scores/`, copy `model_v1.txt` / `model_v1.json` over `model.txt` / `model.json` — the next run scores with v1 |
| `V2 session inputs failed` | the day's per-session inputs could not be computed | each worker reads them per name (slower); the next run retries the file |
| `borrow` snapshot older than 120 min | the IBKR FTP download failed | the borrow check is skipped (warned); the broker still refuses a short it cannot locate |

## 8. Known gaps

- Entries execute once the scorer finishes, ~2.5 min after the tick starts (the
  live path; 2:20–2:36 on the 2026-09-28 afternoon ticks, longer when a tick scores
  two bars; before 2026-09-28 they waited for the shadow pipeline, 10–40 min). A
  1–2 tick delay measured harmless either way (§1).
- Bar-12 picks (the 16:00 close) enter in the post-market, where short fills are poor.
- Rule 201 (fixed 2026-10-05, user: "The broker doesn't handle Rule 201 when the
  short-sale restriction is on"): a short entry under the short-sale price test — the
  pick's `ssr` (read off regular-hours bars like the backtest's flag; a pre-market
  trigger is missed) or a trigger since (today's low ≥ 10% under the previous close on
  Polygon's snapshot, `polygon_client.get_day_low_prev_close`) — is offered ONE TICK
  ABOVE THE NATIONAL BEST BID, the lowest price the rule allows, and rests until the
  next tick instead of the settle pass's 30-second kill; the next tick re-anchors it at
  the new bid, and a refused one is resent at the bid + 1 tick in the same tick
  (`reconcile._rule201_in_force` / `_rule201_limit` / `_bid_now`, `enable_broker_rule201`;
  the leg is stamped `broker_ssr` / `broker_ssr_source`). The backtest models them the
  same way since 2026-10-05 (user: "Implement rule 201 for the backtest"; the optvol
  evaluator's `rule201`, inputs `ssr.npz` per store from `ssr_store.py`: the offer placed
  at each bar's close fills during the first later bar whose high reaches it, at the
  offer, no spread crossed, unless the exit fired first — then no trade). Live setting,
  2021–26: 63% of the vol book's entries are under the test; every one fills, one bar
  later on average; return per day 2.18 → 2.19, growth +150% a year either way, no
  margin call; a stricter fill (a trade one tick THROUGH the offer) gives the same.
  `tradelog/` measures the live fills. The paper account may not enforce the rule.
- The backtest judged every pick's borrow against ONE IBKR file (2026-09-26) and took
  the delisted names as always borrowable; live checks the file at entry. Every pick's
  borrow row at entry, taken or not, is in `entries/` (09-25 → 10-02, 25% of the
  market's biggest risers were not in IBKR's file at all).
- A bar scored late (its tick ran long, or the scheduler was down) keeps its pick
  only while it is <= 75 minutes old; its scores still fill the freshness history.
- Covers outside regular hours (any-time exits) were not in the evaluation, and fill on
  thinner books; the shadow system's daily and 30-minute TICK caches do not yet reset a
  history on a split (only the deep 30-minute store, which is all the strategy reads).
- The evaluation grouped a run by the bar's POSITION in the session; live groups by
  the CLOCK bar. They differ only for a name that did not trade in an earlier bar.
- The evidence is ~200 trades, May–September (January to mid-May was ~0 for every
  exit), with TODAY's borrow fees applied to past trades.
- The short-interest filter was chosen in sample (see §1); FINRA reports short
  interest twice a month, so a pick's days to cover can be a few weeks old.
- Comparing the arms: the ledger holds the union; from 2026-10-04 every arm's pick is
  its own trade (a name two arms pick on one bar = two trades, each with its arm's
  target; earlier ones were ONE trade, `sel_arm = "model+vol"`), a later pick of a held
  name is its own stacked trade (`sel_stack_n` > 1). Each arm's full record is its
  own journal (`picks/`, `arm`), which can be simulated independently (scratchpad
  `short_models.py` harness) beside the live fills.
