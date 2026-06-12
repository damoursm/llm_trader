# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
python main.py                # Run once, console output only
python main.py --email        # Run once and send email report
python main.py --schedule     # Start the poll-loop runner (RTH every 30 min 9:30–16:00 ET + extended trading ticks 4:00–19:55; NYSE market days; emails per scheduler_email_every_tick — every slot when on, else only the 16:00 close)
python main.py --dashboard    # Launch the read-only monitoring dashboard (Plotly Dash via waitress)
```

Logs go to `logs/llm_trader_YYYY-MM-DD.log` (7-day rotation). The HTML report (when `ENABLE_CHARTS=true`) is saved to `logs/report_YYYY-MM-DD_HHMM.html`.

To force a fresh run ignoring caches, delete `cache/news_*.json` and/or `cache/snapshots_*.json`. The OHLCV cache (`cache/ohlcv/`) and the DuckDB database (`data/llm_trader.db` — trades, recommendations, run history) should never be deleted casually.

**Intraday operation (no unclosed-bar look-ahead).** The scheduled runner ticks **every 30 min, 9:30–16:00 ET** on NYSE market days (weekends AND holidays skipped via `market_calendar.is_market_day`). The pipeline marks and fills on **live** prices (`_fetch_price` → `fast_info.last_price`); the daily OHLCV feeding the indicators is **completed-bars-only** — `market_data._drop_forming_bar` removes the still-forming current-session bar until the 16:00 close, so no indicator or NAV walk ever reads an unclosed daily bar. Open-position NAV is **marked to the live price each tick** (`daily_nav._build_marks` uses `current_price` as the open-trade end anchor). A **Hybrid** intraday-timing overlay (`src/signals/intraday_timing.py`) affects entry/exit *timing* only: it defers an entry whose 30-min momentum strongly opposes it (`enable_intraday_timing`, default on) and can close on a hard intraday reversal (`enable_intraday_exit`, opt-in → `exit_reason = "intraday_reversal"`); during extended sessions it fetches `prepost=True` bars so the gate reads the live extended tape, not yesterday's last RTH bar. It never changes the daily-signal direction.

**Extended hours (Phase 1 — TRADING, on by default).** `extended_hours_mode` = `off | observe | trade` (default **`trade`**). The runner ticks through the FULL extended day per `extended_windows` (default `04:00-07:00@60,07:00-09:30,16:00-17:30,18:00-19:00@60,19:55-19:55` ET — liquid shoulders every 30 min, thin dead zones hourly via the `@MM` per-window cadence suffix; a window with equal endpoints is a SINGLE slot, and the **last after-hours tick is 19:55, not 20:00**: the pipeline takes ~4 min from tick to order submission, so a 20:00 slot's orders reached IBKR after the session close and could never fill same-day). In **trade** mode extended slots are FULL trading ticks (`_tick_plan` in `runner.py`): ledger entries/exits/marks and broker paper orders happen off-hours too; in **observe** mode (Phase 0) they run `run_pipeline(observe_only=True)` — full analysis + DuckDB persistence but zero ledger/broker mutation. **Email gating:** with `scheduler_email_every_tick` on, EVERY slot emails — RTH and extended alike; with it off, only the 16:00 closing RTH slot sends the daily report. Either way the scheduler passes `email_if_configured=False` so its per-slot decision is authoritative — manual `python main.py` runs keep the email-if-configured convenience.

Session classification (`market_calendar.current_session` → `rth | extended | overnight`) is fixed once per run and drives the **extended signal profile**: (1) aggregator weight overlay — options-derived methods (`put_call`, `max_pain`, `oi_skew`, `iv_expr`) ×`extended_stale_options_weight_mult` (0.5 — chains are frozen at the RTH close) and `news`/`sent_velocity`/`ext_gap` ×`extended_news_weight_mult` (1.25); (2) `extended_confidence_bump` (+6 pp) on the macro-regime actionable threshold; (3) the **ext_gap scorer** (`src/signals/extended_session.py`) — live extended print vs last completed close in ATR units, tanh-scaled with a deadband, always 0.0 during RTH, fail-closed on a stale OHLCV cache; (4) a session-context block in the Claude prompt (thin books, frozen options data, news/gap = live information). Snapshot prices are genuinely extended off-hours (Polygon `lastTrade` includes extended prints; the yfinance fallback pulls 1-min `prepost` bars).

**Extended trade lifecycle:** `_execution_iso()` returns *now* during RTH and during extended sessions in trade mode (extended fills are real); overnight/weekend/observe-mode timestamps still snap to the next regular open. Every entry stamps `entry_session` (+`extended_size_multiplier` applied — off-RTH entries are sized ×0.5 by default on top of the confidence tier and correlation haircut); every close stamps `exit_session`. Costs are session-aware end to end: spread multipliers ×4 extended / ×10 overnight (`spread._session_spread_multiplier`), per-leg `entry_session`/`exit_session` in `_pct_return`, the daily-NAV anchors, `_normalize_closed_returns`, `_flip_trade`, and the live M2M (`update_open_trades` charges the *current* session's exit cost; the NAV open-anchor uses `_session_of_mark`). **Broker off-RTH:** `reconcile.sync()` forces a marketable **LMT + `outsideRth=True`** outside regular hours (IBKR rejects MKT there); RTH keeps the configured `broker_order_type`. An order submitted overnight rests until the 04:00 pre-market open.

**Session-split evaluation:** every perf surface accepts `session=` — the dashboard's Method Performance and Returns tabs each carry an RTH/Extended/Overnight toggle, trade tables show a `Session` column, the email/dashboard performance breakdown adds per-session rows (group `session`, shown once any non-RTH trade exists), and `_compute_llm_perf` pseudo-trades charge the extended entry leg.

## Architecture

### Pipeline flow (`src/pipeline.py`)

`run_pipeline()` is the single entry point. Steps execute in sequence:

```
0.  Universe construction   — Static watchlist + dynamic discovery (trending, screener, macro→holdings, catalyst/analyst/earnings expansion), then a liquidity gate filters untradeable microcaps.
1.  News & sentiment feeds  — RSS, NewsAPI, 8-K filings, Google Trends, Reddit, analyst ratings, EPS surprises, short interest. All normalised to NewsArticle.
2.  Market snapshots        — Polygon batch primary, yfinance fallback for indices/futures/OTC.
3.  Smart-money + macro     — Insider/congressional trades, options flow, SEC filings (13D/G, Form 144, 13F, Form 4), plus market-context fetchers (FRED, COT, VIX, MOVE, DIX, breadth, credit, bond/global macro, sector rotation, business cycle, etc.).
3X. Regime & timing layers  — Macro Regime Filter, Market Mode, Catalyst Timing — gate BUYs, adjust thresholds, and tweak aggregator weights.
3#. Per-ticker scorers      — News sentiment, sentiment velocity, technical, trend strength, momentum, money flow, pattern recognition, IV rank/expression, PEAD, cointegration, cross-sectional rank, extended-session gap (off-hours runs only). Each returns a score in [-1, +1].
4.  Signal aggregation      — aggregator.py combines per-ticker scores into a TickerSignal; sector_pairs.py finds market-neutral opposites.
5.  Recommendations         — claude_analyst.py → BUY/SELL/HOLD/WATCH; DeepSeek auto-fallback; rule-based last resort.
6.  Performance tracking    — tracker.py + daily_nav.py: paper trades, M2M updates, path-faithful daily compound.
7.  Charts + email          — charts/, email_sender.py.
```

For deeper detail on any step (modules, signal scoring, weights, config flags), see the module file or `memory/project_signals.md`.

### Data flow design

**Two tiers of data:**
- **Per-ticker signals** (news, sentiment velocity, technical, trend strength, insider): scored `[-1.0, +1.0]`, combined in `aggregator.py` with configurable weights into a `TickerSignal`. News sentiment is the *level*; sentiment velocity is the *rate of change* (Δ recent−prior tone) — distinct methods, distinct weights. Trend strength (ADX/DMI + Donchian) measures trend *quality* and is distinct from price momentum (return *size*). Convergence multiplier (1.25× when ≥2 methods agree, 0.60× when only 1 does) prevents single-source BUY/SELL calls.
- **Macro context** (FRED, COT): not per-ticker. Passed as structured context blocks directly into the Claude prompt. Claude applies regime overlays (recession → avoid POSITION longs; EXTREME_LONG COT → cap BUY conviction on that asset).
- **Macro Regime Filter** (`macro_regime.py`): hard pre-filter applied after Claude runs. Reads VIX, MOVE, bond internals, global macro, FRED, breadth, credit, and the Dark Pool Index (DIX) contexts; computes a weighted composite regime (PANIC|RISK_OFF|CAUTION|NEUTRAL|RISK_ON) that gates actionable signals by raising the confidence threshold and blocking BUY entries during PANIC/RISK_OFF. This is the top-down overlay that prevents buying into market crashes.
- **Market Mode Switching** (`market_mode.py`): adjusts the aggregator's signal weight profile based on market structure. TRENDING (low VIX, healthy breadth) → up-weight `tech`/`news` (momentum), down-weight `vwap`/`put_call`. CHOPPY (high VIX, mixed breadth) → up-weight `vwap`/`put_call` (mean-reversion/contrarian), down-weight `tech`. Computed from VIX, breadth, highs/lows, and McClellan before `build_signals()` runs.
- **Catalyst Timing** (`catalyst_timing.py`): event-driven post-processing layer applied after recommendations are ranked. Three mechanisms: (1) Earnings Blackout removes tickers within 2 days of earnings from the actionable set — with a **post-release exemption**: a ticker whose actual EPS is already in the PEAD context (`days_since_report ≤ 1`) is NOT blocked, since the binary event has resolved and the PEAD/gap-reaction entry is exactly the post-earnings trade the system wants; (2) OpEx Amplifier boosts `max_pain` weight in the aggregator during OpEx week; (3) 8-K + Insider Buy WATCH elevation promotes or injects WATCH recommendations when material catalyst + insider conviction coincide.

**8-K filings, Google Trends, Reddit sentiment, analyst ratings, EPS surprises, and short interest are treated as news** — all return `List[NewsArticle]` and are scored by the same DeepSeek sentiment pipeline as RSS articles. Google Trends articles describe search interest spikes/drops; Reddit articles summarise mention count and upvote-weighted sentiment across r/wallstreetbets, r/stocks, and r/investing. Short interest articles surface squeeze setups, bearish positioning builds, and short covering signals.

**Smart money signals** (insider trades, options flow, SEC filings) all return `List[InsiderTrade]` and are combined into a single `insider_score` per ticker in the aggregator.

**Method attribution** (`tracker.py`): every new trade stores the raw method scores (`news`, `sent_velocity`, `tech`, `insider`, `put_call`, `max_pain`, `oi_skew`, `vwap`, `pattern`, `momentum`, `sector_momentum`, `money_flow`, `trend_strength`, `pead`, `iv_rank`, `iv_expr`, `coint`, `cross_sectional`, `ext_gap`) at entry time plus `methods_agreeing` and `dominant_method`. `get_performance_for_email()` returns:
- `performance_table`: unified list of breakdown rows (columns: trades, win_rate, compound_return, avg_return, wtd_avg_return, best, worst) grouped as: **total** (All Trades) → **asset** (Stocks only / ETFs only / Commodities only) → **direction** (Longs only BUY / Shorts only SELL) → **session** (RTH / Extended / Overnight by entry session — shown once any non-RTH trade exists) → **method** (per signal method, ≥2 attributed trades required). Rows are rendered in the email "Performance Breakdown" section with visual dividers between groups. **Includes open trades at their current M2M `return_pct`** (maintained each run by `update_open_trades()`) — treated as hypothetical exits so every live position is reflected in the breakdown.
- `trades_svg`: inline SVG (820×336px dark-theme) with equity curve (top panel) and per-trade return bars (bottom panel). Embedded directly in the email via `{{ perf.trades_svg | safe }}` — no kaleido/Plotly dependency.
- `portfolio_metrics`: dict from `compute_portfolio_metrics()` — delegates to `daily_nav.compute_compound_return()` (see [Performance calculation](#performance-calculation) below) for a path-faithful daily-walked compound over real OHLCV closes. For every calendar day with any active position the engine forms the capital-weighted average daily return across active trades (`Σ(r·w)/Σ(w)` with `w = position_size_multiplier`) and compounds sequentially. Keys: `compound_inception` (every trade ever), `return_1w`/`return_2w`/`return_1m` (trades with `entry_date >= today − N days`; open trades included at the live M2M end-anchor). Overwrites `stats["compound_return"]` so the headline number matches the tiles. Displayed in the email Portfolio Performance card as time-window tiles.
- `method_order_by_winrate`: list of method keys sorted descending by solo win rate (no-data methods last). Used to order the solo performance and method eval tables in the email — best-performing signals appear first.
- `attributed_count`: count of trades with `methods_agreeing` populated (used to decide whether method rows appear).
- `solo_method_perf`: dict from `compute_solo_method_performance()` — for each closed trade with stored method_scores, asks "what direction would this method alone have signalled?" Same direction as actual → actual return; opposite → negated return; |score| < 0.10 (no view) → skipped. Rendered in email section 4b as a standalone table (trades, win_rate, compound_return, avg_return, best, worst per method). Distinct from `performance_table` method rows which only count trades where the method agreed with the aggregated direction.
- `llm_perf`: dict from `_compute_llm_perf(window_days, session)` — per-LLM-engine stats over **every recommended trade, executed or not**: `{"synthesis": {model_id: segment_stats}, "sentiment": {model_id: segment_stats}}`. The real ledger is too small (and gate-selection-biased) to compare engines, so each engine is scored on its full recommendation stream: every BUY/SELL it produced (actionable or not), deduped to the engine's last call per (ticker, day), anchored at the recommendation-time snapshot price (`signals.price`; legacy recs fall back to the rec-day cached close), marked at the latest cached close through the same cost model as real trades (a brand-new call therefore starts ≈ −cost, like a real position). Synthesis attribution comes from `recommendations.llm_provider` (exact model id on new rows; legacy provider strings mapped), sentiment from the run's dominant scorer. The per-run 50/50 A/B flip (`LLM_AB_ANTHROPIC_SHARE`) supplies both engines with comparable samples. New trades are additionally stamped `llm_synthesis_model` / `llm_sentiment_model` as provenance. Rendered as highlighted extra rows of the dashboard's Method Performance model-evaluation table; `rule-based (no LLM)` appears as its own baseline row.
- `method_eval_stats`: dict from `compute_method_eval_stats()` — per-method directional accuracy and conviction calibration. For each method: overall directional_accuracy (% correct directions), avg_return_correct, avg_return_wrong, and conviction_bands (Low 0.10–0.35 / Medium 0.35–0.65 / High 0.65+) each with trades, accuracy, avg_return. Rendered in email section 4c as per-method cards. A well-calibrated signal shows rising accuracy Low→High; flat/declining accuracy suggests the method adds noise rather than genuine directional insight.
Legacy trades without `methods_agreeing` are excluded from method rows but included in total/asset/direction rows.

### Performance calculation

Performance numbers come from two engines that work side-by-side, both rooted in the *real* observed prices stored on each trade — never interpolated, never synthesised.

**Per-trade `return_pct` (buy-and-hold, cost-aware) — `tracker._pct_return()`**

```
cost_in  = _one_side_cost(entry_price, asset_type)            # half-spread + commission
cost_out = _one_side_cost(exit_or_current_price, asset_type)
BUY  : eff_entry = entry × (1 + cost_in);   eff_exit = exit × (1 − cost_out)
       return_pct = (eff_exit − eff_entry) / eff_entry × 100
SELL : eff_entry = entry × (1 − cost_in);   eff_exit = exit × (1 + cost_out)
       return_pct = (eff_entry − eff_exit) / eff_entry × 100
```

The one-way cost (`spread._one_side_cost`) is **half-spread + commission**. Bid-ask half-spread is price-tiered and asset-type-aware (see `_dynamic_half_spread` in `src/performance/spread.py` for the May-2026-calibrated tiers). The commission term (`_commission_fraction`) is a deliberate **fee ceiling, not a best estimate** — results must err conservative: the pricier all-in plan is the default (`ibkr_fixed`: max(\$1.00, \$0.005/share); `ibkr_tiered`: max(\$0.35, \$0.0035/share), commission only; `none`: spread-only), capped at 1% of trade value, then × `commission_buffer` (default **1.5**, applied after the cap) to cover SEC/TAF sell-side fees, venue/clearing fees under tiered, surcharges, and schedule drift — ~41 bp round trip at the assumed `commission_notional_usd` (≈730 USD — a deliberate constant, not live FX, so the math stays deterministic; the 1.0× base notional is also the worst case, since larger positions see a smaller min-commission floor in % terms) vs ~27 bp actual. Unrecognized model names fall back to `ibkr_fixed`, not the cheaper plan. Calibrate the buffer against actual fills (`broker_orders.commission` in DuckDB) once paper data accumulates. At this system's order sizes the commission minimum dominates spread on liquid large caps. Closed trades use `exit_price`; open trades use the live `current_price` (refreshed each tick by `update_open_trades()`, timestamped via `current_price_datetime`). For open trades this is a "what if you closed right now" mark — same convention as the email's M2M display. `_normalize_closed_returns()` recomputes this each pipeline tick from the stored entry/exit prices so every closed trade's `return_pct` reflects the current cost model.

**Path-faithful daily compound — `daily_nav.compute_compound_return(trades)`**

Per-trade walk:
1. Load OHLCV closes for the ticker from `cache/ohlcv/<TICKER>.json` (force-refreshed for open-trade tickers via `tracker._refresh_open_trade_ohlcv()` so today−1's close exists).
2. Build chronological marks: `(entry_date, eff_entry)` → every cached close strictly between → `(exit_date or today, eff_exit)`. The entry/exit anchors apply the same `_one_side_cost` (half-spread + commission) as `_pct_return`, so both engines charge identical costs; intermediate marks are raw closes, **no cost applied** — those days the position is marked-to-market, not traded.
3. For each adjacent pair compute `r_d = sign × (mark_d − mark_{d−1}) / mark_{d−1}` with `sign = +1` for BUY, `−1` for SELL. The short formula is path-faithful (depends on adjacent days only), not the buy-and-hold formula — so a volatile short's daily compound legitimately differs from its `return_pct` (volatility decay for daily-rebalanced shorts).

Portfolio aggregation:
1. Collect every `(date, daily_return, weight)` tuple from every trade (`weight = position_size_multiplier`).
2. Group by date; per-day portfolio return = `sum(r·w) / sum(w)` over active positions.
3. Compound sequentially: `compound = ∏(1 + day_return) − 1`.

This is **100% deterministic**: same DuckDB trades (`data/llm_trader.db`) + same `cache/ohlcv/*.json` produces bit-identical output. Time-window variants (`return_1w`/`return_2w`/`return_1m`) just filter trades by `entry_date >= today − N days` before aggregating.

**Summary statistics — `tracker._compute_segment_stats()`**

For any slice of trades (All / Stocks / ETFs / Commodities / Longs / Shorts / per-method):

| Metric | Formula | Notes |
|---|---|---|
| `trades` | `len(trades)` | |
| `win_rate` | `100 × count(t.return_pct > 0) / len(trades)` | Strictly positive on the spread-adjusted `return_pct`, so a flat round trip is a loss by the spread cost. |
| `compound_return` | `daily_nav.compute_compound_return(trades)` | Path-faithful daily walk (the engine above). |
| `avg_return` | `mean(t.return_pct)` | Equal-weighted across trades. |
| `wtd_avg_return` | `Σ(t.return_pct · t.position_size_multiplier) / Σ(t.position_size_multiplier)` | Capital-weighted. |
| `best` / `worst` | `max(t.return_pct)` / `min(t.return_pct)` | |

Open trades' `return_pct` is the live M2M from `update_open_trades()` so they contribute to every metric (treated as hypothetical exits). The compound metric uses the daily walk; everything else uses the per-trade `return_pct`. These can diverge for shorts — by design, since path-faithful compound and buy-and-hold are different (both correct) measures.

**Hypothetical solo & eval stats — `tracker._flip_trade()` / `_hypothetical_trades_for_method()`**

For "what if only this method had decided" simulations:
- If the method's sign matches the actual trade's action → use the trade verbatim (real walk, real `return_pct`).
- If it disagrees → `_flip_trade()` returns a same-ticker/same-dates trade dict with `action`/`direction` inverted and `return_pct` re-derived via `_pct_return` for the flipped action. The daily-NAV engine then walks the **real OHLCV closes** in the opposite direction — no negation-of-stored-number tricks.

Conviction bands (`_eval_stats`): Low `0.10–0.35`, Medium `0.35–0.65`, High `0.65+` on `|method_score|`. Each band reports `trades`, `accuracy` (% with `return_pct > 0`), `avg_return`, and `compound_return` (also via the daily-walked engine). A well-calibrated method shows rising accuracy Low → Medium → High.

**Trade lifecycle — `tracker.py`**

| Function | When | What it does |
|---|---|---|
| `record_new_trades()` | After Claude produces recommendations | Opens a trade per actionable BUY/SELL; stamps `entry_datetime` + `entry_price` + `entry_session` together; applies the correlation-aware sizing haircut (continuous scale-down vs same-direction open peers, hard skip when correlated-exposure cap is hit) and the `extended_size_multiplier` haircut (×0.5 default) for off-RTH entries; stores all 19 raw method scores + `methods_agreeing` + `dominant_method`. |
| `close_trades_on_signal_reversal()` | Before opening new trades, if today's actionable signal reverses an open position | Closes with `exit_datetime = current_price_datetime` (most recent live mark) — no second fetch. |
| `_refresh_open_trade_ohlcv()` | First step of `update_open_trades()` | Force-refreshes OHLCV cache for every open-trade ticker (bypasses 3-day TTL) so the daily walk has a real close for every day held. |
| `_normalize_closed_returns()` | Second step of `update_open_trades()` | Idempotently rederives `return_pct` for every closed trade through the current spread model. |
| `update_open_trades()` | Each pipeline tick | Refreshes `current_price` + `current_price_datetime` + `return_pct` for every OPEN trade. **No time cap** — positions are never closed on age. |
| `monitor_open_positions()` | Each tick, after `update_open_trades` | The primary exit: closes a position when its thesis deteriorates — `macro_regime_exit` (long in PANIC/RISK_OFF), `signal_flipped`, `signal_decay` (entry vs today strength), or `confidence_loss` (today's confidence below an entry-relative floor). Toggle with `enable_signal_decay_exits`. |

### Database (DuckDB)

`data/llm_trader.db` is the **single source of truth** for trades, recommendations, and run history — it replaces the old `cache/trades.json` / `cache/hypothetical_trades.json` ledgers (now import-only). All access goes through `src/db/`:

- **`connection.py`** — `connect(read_only=False)` yields a short-lived DuckDB handle and closes it on exit. Read-write opens `ensure_schema()` first; read-only opens require the file to already exist (raise `FileNotFoundError` otherwise).
- **`schema.py`** — idempotent `CREATE TABLE IF NOT EXISTS` for eight tables: `runs`, `run_sources`, `recommendations`, `trades`, `hypothetical_trades`, `broker_reconciles`, `broker_orders`, `signals`. Each trade / hypothetical row stores the **full dict in a JSON `data` column** (so `tracker.py` round-trips byte-identical dicts, exactly as the old JSON files did) alongside projected scalar columns used for SQL and the dashboard. New tables appear automatically on the next write connection; new *columns* on an existing table need a one-time `ALTER TABLE`.
- **`signals` table (the learning panel)** — every run persists the FULL per-ticker signal cross-section (one row per scored ticker, not just the top-10 recommendations): all 19 method scores as DOUBLE columns (mirroring `tracker._ALL_METHODS` via `schema.SIGNAL_METHOD_COLUMNS` — a drift test keeps them in sync) plus `combined_score`, `confidence`, `direction`, `n_methods_agreeing`, `dominant_method`, snapshot `price`, and the full score dict as JSON. Purpose: the trade ledger only teaches what the gates let through (selection bias); this panel, joined against forward returns from `cache/ohlcv/`, is the unbiased dataset for information-coefficient analysis and threshold/weight tuning. News/options inputs can't be reconstructed historically, so forward collection is the only path to a backtest-quality dataset. Analyse with `python -m src.analysis.signal_panel` (`--horizons 1,5,10 --days 90`): dedupes to the last run per (day, ticker), computes per-method Spearman IC + directional hit rate per horizon (zero scores = "no view" excluded, `--min-n` floor before an IC is reported).
- **`repo.py`** — the typed read/write API: `load_trades()` / `save_trades()` (full-replace, matching the old whole-file rewrite semantics), `insert_run()`, `insert_run_sources()`, `insert_recommendations()`, and `fetch_df()` (DataFrame reads for the dashboard). `set_read_only(True)` flips every read path to open read-only — the dashboard sets this so it never takes the write lock.
- **`migrate.py`** — one-time `python -m src.db.migrate` imports the legacy JSON ledgers; idempotent (skips populated tables unless `--force`). As a safety net, `tracker._load_trades()` self-seeds the DB from `cache/trades.json` if the DB is empty, so the cutover never loses history even if migration is skipped.

**Concurrency model:** DuckDB allows a single read-write handle OR many read-only handles across processes. The pipeline is the **sole writer** and holds the write lock only momentarily (open → write → close); at the end of every run it persists `runs` + `run_sources` + `recommendations` (`pipeline.py`, wrapped in try/except so a DB hiccup never aborts the run). The dashboard connects **read-only** and retries with exponential backoff across the brief window the writer holds the lock.

### Monitoring dashboard (`dashboard/`)

A read-only Plotly Dash app for inspecting the database. Launch with `python main.py --dashboard` (host/port from `settings.dashboard_host` / `dashboard_port`, default `127.0.0.1:8050`).

- **`app.py`** — three tabs: **Recommendations & Rationale** (per-run data sources + recommendations), **Method Performance** (solo win-rate bars, per-method table, and an *LLM models used* breakdown — the exact synthesis & sentiment models per run, including DeepSeek / rule-based fallbacks), **Returns** (KPI tiles + equity curve + open/closed trades, with a **Simulated ⇄ IBKR toggle**: *Simulated (model)* is the strategy ledger — every decision at its decision price through the modeled cost stack; *IBKR (actual fills)* re-anchors each broker-backed trade at its real average fill prices with the commissions actually charged, no modeled costs — dollar-P&L-led tiles, Shares/Notional columns, and a ledger-CLOSED trade whose exit hasn't filled shown as still OPEN; built by `src/performance/broker_view.py`, served by `data.broker_trades()`. The simulated view's trade tables additionally carry **IBKR entry / IBKR exit** columns (`_ibkr_leg_disp`: ✓ filled · ⏳ working/partial/pending · ↻ re-anchoring · ✕ cancelled · ✗ rejected · – never sent) so each sim trade shows whether its orders really executed, and every trade table has a **Held** column — wall-clock holding time as `2d 5h` / `6h` / `45m` (`_held_disp`: entry → exit for closed trades, entry → now for open ones; legacy date-only rows fall back to `Nd` trading days)). Each tab's content is embedded as that `dcc.Tab`'s `children`, so the Tabs component swaps content **client-side** with no `tabs.value`→callback round trip (that round trip proved unreliable in the browser — every tab showed the first-rendered one). `app.layout` is the `serve_layout` **function**, rebuilt per page load so a long-running dashboard always reflects the latest run; the selected tab is persisted across reloads and `_safe()` wraps each tab body so a data hiccup shows a message instead of a blank page. Tables are sortable (click a header; shift-click for multi-sort), carry a per-column filter row, use human-readable headers, format numeric columns (kept numeric so they sort correctly), and render timestamps in Eastern time (`_fmt_et`). Every column header, KPI tile, and section heading carries a hover tooltip (`tooltip_header` / HTML `title`) explaining the metric. `run()` serves through **waitress** — a production-grade, multi-threaded, cross-platform WSGI server (the right choice on Windows; gunicorn doesn't run there) — wrapped in an **auto-restart supervisor loop** so a crash self-heals; it falls back to the Dash dev server only if waitress isn't installed.
- **`data.py`** — read-only accessors over `repo.fetch_df()`. Sets `repo.set_read_only(True)` at import, wraps every read in exponential-backoff retry (`_retry`, ~11 s budget) for the daily write-lock window, and caches the heavy `get_performance_for_email()` result for 60 s so tab switches stay responsive.
- **`figures.py`** — Plotly figures (method win-rate bars; equity curve via `src.charts.builder`).

### Model routing

| Task | Engines (A/B-routed) | Last resort |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V4-Flash (`deepseek-v4-flash`, non-thinking) ⇄ Claude Haiku 4.5 | score 0.0 (`none`) |
| Final synthesis / BUY/SELL/HOLD/WATCH | `ANALYST_MODEL` Claude (default: `claude-haiku-4-5-20251001`) ⇄ DeepSeek V4-Flash (`deepseek-v4-flash`) | rule-based `_fallback_recommendations()` |

**A/B routing (`LLM_AB_ANTHROPIC_SHARE`, default 0.5):** each run flips a coin per role — synthesis and sentiment independently — to pick which provider goes FIRST; the other stays as the error fallback (credits exhausted, rate limit, connection failure, missing key). 0.5 gives both engines comparable recommendation samples for the dashboard's per-LLM evaluation rows; 1.0 = always-Anthropic-first (legacy behavior), 0.0 = always-DeepSeek-first. The engine that actually answered is recorded per run (`runs.llm_synthesis_provider` / `llm_sentiment_provider`), per recommendation (`recommendations.llm_provider` — exact model id on new rows; legacy rows hold the provider string), and stamped on each new trade (`llm_synthesis_model` / `llm_sentiment_model`).

Available analyst models (set in `.env`):
- `claude-haiku-4-5-20251001` — fastest, cheapest; 8 192 output tokens
- `claude-sonnet-4-6` — higher quality; 64 000 output tokens
- `claude-opus-4-7` — highest quality; 32 000 output tokens

**Claude API calls use streaming** (`client.messages.stream`) to avoid SDK timeout errors on large prompts (40+ tickers can take >10 minutes non-streaming). Text chunks are accumulated into a single string before JSON parsing.

**Engine fallback** (`_call_claude_analyst()` / `_call_deepseek_analyst()` in `claude_analyst.py`): whichever engine the A/B flip did NOT pick first is tried automatically when the primary fails — any `anthropic.APIStatusError` (HTTP 400/401/402/403/429/5xx — covers credits exhausted, bad key, payment required, rate limit, server errors), `anthropic.APIConnectionError`, or DeepSeek/OpenAI-SDK error. Both engines receive the identical prompt (DeepSeek via the OpenAI-compatible streaming API, non-thinking mode; requires `DEEPSEEK_API_KEY`). If both LLMs fail, falls back to rule-based `_fallback_recommendations()`. The source model name is logged at INFO level (`via <model>`). Parsed responses are **deduped per ticker** (first occurrence wins, `_dedupe_recommendations`) — LLMs occasionally repeat tickers (DeepSeek 2026-06-11) and every downstream consumer assumes one recommendation each; `record_new_trades` independently adds batch-opened tickers to its `already_open` guard so a duplicate can never open twin trades.

### Caching strategy

| Cache | Key | TTL | Location |
|---|---|---|---|
| News + 8-K | `YYYY-MM-DD_HH` | 1 hour | `cache/news_*.json` |
| Snapshots | `YYYY-MM-DD_HH` | 1 hour | `cache/snapshots_*.json` |
| Reddit sentiment | `YYYY-MM-DD_HH` | 1 hour | `cache/reddit_*.json` |
| Google Trends | `YYYY-MM-DD` | 1 day | `cache/trends_*.json` |
| IPO pipeline | `YYYY-MM-DD` | 1 day | `cache/ipo_*.json` |
| VIX & term structure | `YYYY-MM-DD` | 1 day | `cache/vix_*.json` |
| Credit market (HYG/SPY) | `YYYY-MM-DD` | 1 day | `cache/credit_*.json` |
| Put/call ratio | `YYYY-MM-DD` | 1 day | `cache/put_call_*.json` |
| Analyst ratings | `YYYY-MM-DD` | 1 day | `cache/analyst_ratings_*.json` |
| Earnings surprises | `YYYY-MM-DD` | 1 day | `cache/earnings_surprises_*.json` |
| Earnings calendar | `YYYY-MM-DD` | 1 day | `cache/earnings_calendar_*.json` |
| Earnings discovery (Section E) | `YYYY-MM-DD` | 1 day | `cache/earnings_discovery_*.json` |
| Analyst discovery (Section E) | `YYYY-MM-DD` | 1 day | `cache/analyst_discovery_*.json` |
| PEAD signals | `YYYY-MM-DD` | 1 day | `cache/pead_*.json` |
| Short interest | `YYYY-MM-DD` | 1 day | `cache/short_interest_*.json` |
| Market breadth | `YYYY-MM-DD` | 1 day | `cache/breadth_*.json` |
| New 52w highs/lows | `YYYY-MM-DD` | 1 day | `cache/highs_lows_*.json` |
| McClellan Oscillator | `YYYY-MM-DD` | 1 day | `cache/mcclellan_*.json` |
| Macro Surprise Index | `YYYY-MM-DD` | 1 day | `cache/macro_surprise_*.json` |
| Fed Rate Expectations | `YYYY-MM-DD` | 1 day | `cache/fedwatch_*.json` |
| Revision Momentum | `YYYY-MM-DD` | 1 day | `cache/revision_momentum_*.json` |
| Earnings Whisper | `YYYY-MM-DD` | 1 day | `cache/whisper_*.json` |
| Bond market internals | `YYYY-MM-DD` | 1 day | `cache/bond_internals_*.json` |
| MOVE Index | `YYYY-MM-DD` | 1 day | `cache/move_*.json` |
| Dark Pool Index (DIX) | `YYYY-MM-DD` | 1 day | `cache/dix_*.json` |
| Global macro (DXY + Cu/Au) | `YYYY-MM-DD` | 1 day | `cache/global_macro_*.json` |
| Sector rotation (Ebb & Flow) | `YYYY-MM-DD` | 1 day | `cache/sector_rotation_*.json` |
| Rotation Drivers (rate cycle) | `YYYY-MM-DD` | 1 day | `cache/rotation_drivers_*.json` |
| OHLCV (charts) | per ticker | incremental | `cache/ohlcv/*.json` |
| COT positioning | ISO week | 1 week | `cache/cot_YYYY_WW.json` |
| Insider cluster watchlist | ticker | 10 days | `cache/cluster_watchlist.json` |
| Pattern libraries | per ticker | 7 days | `cache/patterns/<TICKER>.json` |
| Trades · hypotheticals · runs · recs | — | permanent | **DuckDB** (`data/llm_trader.db`) |

**Important:** The news cache includes 8-K articles only on the run that fetches them (8-K fetch always runs fresh and appends to whichever articles list is active — from cache or live). COT data is cached by ISO week, so the first run each week downloads from CFTC; subsequent runs within the week are instant. The DuckDB database (`data/llm_trader.db`) is **not** a cache — it is the permanent single source of truth (see [Database (DuckDB)](#database-duckdb)), never auto-expired; the legacy `cache/trades.json` is import-only.

### Adding a new data source

Three patterns depending on what the source produces:

1. **Per-ticker signal** (like sentiment/technical/insider): implement `[-1.0, +1.0]` scorer, add weight in `aggregator.py`, add `enable_*` flag in `settings.py`. Also add the method to `tracker._ALL_METHODS` **and** `schema.SIGNAL_METHOD_COLUMNS` (signals-table column; `ALTER TABLE signals ADD COLUMN <m> DOUBLE` on an existing DB) — `tests/test_db_signals.py` fails if the two drift.
2. **Smart money signal** (like options flow, SEC filings): return `List[InsiderTrade]` from a new module, call from pipeline in the Step 3 block, extend `smart_money` list.
3. **Macro context** (like FRED, COT): return a new Pydantic model, pass to `generate_recommendations()` as an optional parameter, build a prompt block + instruction section in `claude_analyst.py`, render in email HTML template.

### Key constraints

**Polygon.io client** (`src/data/polygon_client.py`): wraps the Polygon REST API. `get_snapshots_batch()` fetches all tickers in two calls (snapshots + prev-close). `get_bars()` fetches OHLCV history. Falls back gracefully when `POLYGON_API_KEY` is absent — `is_available()` returns False and `market_data.py` routes everything through yfinance instead. Free Polygon tier covers all US equity/ETF tickers; indices (^VIX etc.) and futures (GC=F etc.) must use yfinance directly.

**SEC EDGAR rate limit:** 10 requests/second. All EDGAR calls use `_REQUEST_DELAY = 0.12–0.15s`. The `_ticker_index` (name→ticker) and `_ticker_cik` (ticker→CIK) maps are module-level globals populated once per process — never re-fetch them in a loop.

**yfinance rate limits:** When a 429 is returned, `market_data.py` applies exponential backoff (60s → 120s → 240s). After 3 failures the loop stops early. Options chain scanning (`options_flow.py`) is silent-fail per ticker.

**CFTC downloads:** The COT ZIP files are ~5–10 MB each. Both current and previous year are fetched if the current year has insufficient history. Always check the cache first.

**Actionable signal threshold:** Baseline is `confidence ≥ 0.78` AND `sources_agreeing ≥ 2`. The Macro Regime Filter adjusts this dynamically: RISK_ON → 72%, CAUTION → 80%, RISK_OFF → 82%, PANIC → 88%. Runs outside RTH add `extended_confidence_bump` (+6 pp, capped at 95%) on top of the regime threshold — thin extended books demand more conviction. BUY entries are additionally blocked entirely during PANIC and RISK_OFF regimes. A single strong signal source never produces a BUY/SELL regardless of score.

### Email HTML template

The HTML is a Jinja2 template string (`HTML_TEMPLATE`) embedded directly in `email_sender.py` — there is no separate `.html` file. When adding new sections, follow the existing comment convention (`<!-- ══ N — SECTION NAME ══ -->`) and pass new variables through the `Template(...).render(...)` call.

**Chart embedding:**
- Per-ticker OHLCV charts (when `ENABLE_CHARTS=true`): base64 PNG embedded as `cid:chart_<TICKER>` MIME parts.
- Signal overview chart (when `ENABLE_CHARTS=true`): base64 PNG embedded as `cid:overview_chart`.
- Equity curve + trade bars: **inline SVG** generated by `_build_trades_svg()` in `tracker.py`, embedded via `{{ perf.trades_svg | safe }}` — no kaleido or Plotly required.

**Email sections (current order):**
1. Aggregated Recommendations (BUY/SELL)
2. Portfolio Performance (stats + time-window tiles: 1w/2w/1m dollar-weighted returns + since-inception compound + inline SVG equity curve)
3. Performance Breakdown (unified table: total/asset/direction/method rows; methods ordered by solo win rate descending)
4. Trade Details (BUY/SELL only; prices formatted via `fmt_price()` for sub-penny precision)
5. Monitor List (HOLD/WATCH)
6. Analyst Ratings, EPS Surprises, Macro Regime Filter, Market Mode, Catalyst Timing, FRED, COT, VIX, McClellan, Highs/Lows, Breadth, Macro Surprise, FedWatch, Credit, MOVE, Dark Pool Index (DIX), Bond Internals, Global Macro, Seasonality, OpEx, Put/Call, Short Interest, GEX, Revision Momentum, Earnings Whisper, IPO Pipeline, Reddit, Sector Pairs, Cointegration Pairs, Smart Money Signals

### Broker integration (paper-first live execution)

`src/broker/` adds a real **execution leg** alongside the internal NAV simulator — a "shadow & reconcile" design for moving toward live trading. Gated by `settings.broker_mode`; **`off` (default) makes zero broker calls** (behavior unchanged).

- **Modes:** `off` | `dry_run` (log intended orders, submit nothing) | `ibkr_paper` | `ibkr_live`. Paper vs live is the configured `ibkr_port` (IB Gateway 4002/4001); the factory refuses to start `ibkr_paper` on a known live port.
- **Broker = Interactive Brokers** via `ib_async` → IB Gateway. It's the only API broker that lets a Canadian resident paper *and* live trade US securities (CIRO blocks API orders only for Canadian-listed names; the tradeable universe is 100% US stocks/ETFs). `ib_async` is imported lazily, so the package imports fine without it (off/dry_run paths).
- **Single hook point — `reconcile.sync()`** runs once per tick in `pipeline.run_pipeline()` right after `record_new_trades` (the ledger is final there). Idempotent and self-healing, in order: (0) **fill refresh** — orders submitted on an earlier tick that hadn't reached a terminal state inside `submit_order`'s ~12 s poll (queued overnight, partial fill, late commission report) are repaired from today's executions via `Broker.get_fills()` (IBKR `reqExecutions`, keyed by `orderRef`); (0b) **tick-scoped order lifetime** (`broker_tick_scoped_orders`, default on) — an order lives exactly one tick: anything submitted before this sync and still unfilled is cancelled and its leg cleared, so the entry/exit pass below resubmits THIS tick re-anchored at the current mark under a fresh `-rN` client_ref — entries resubmit only when the trade survived this tick's signal pass (`monitor_open_positions` runs first; a closed trade's entry dies via cancel-on-close), so no order ever works the book on a previous tick's decision or price (`broker_unfilled_cancel_minutes` is the age fallback when tick-scoping is off; partial fills are left working; a cancel that loses the race to a fill is left for the fill-refresh pass). **Dead/expired orders are recovered**: every off-RTH submission is a DAY LMT that IBKR expires at the 20:00 session close — when the cancel finds nothing working and the ref has no fills today, the leg is cleared as `EXPIRED` and re-sent instead of resting as a zombie `Submitted` forever; entry legs are only cleared when no broker position backs them (day-scoped fills feed can miss a prior-day fill — resubmitting against a held position would double it), and a dead exit with nothing held is stamped terminal (an exit sized from the recorded entry fill would open a fresh short); (0c) **cancel-on-close** — a trade the ledger CLOSED while its entry order was still working gets that order cancelled immediately (`_cancel_entries_for_closed`): this was the orphan factory — nothing else would ever cancel it, and a later fill would create a position with no open trade behind it; any partially-filled part is flattened by the exit pass; (1) internal OPEN trades without a `broker_order_id` → sized entry submitted; (2) internal CLOSED trades still holding a broker position → close submitted, **sized from the actual held position when the positions feed has a row** (sign-checked — a wrong-sign holding is never blind-traded; falls back to the recorded fill qty when the feed has no row) so lagging fill counts leave no residue; (3) **drift** — broker positions the ledger cannot explain. In-flight closes are exempt (a CLOSED trade's working exit order, or an exit submitted this tick — the positions snapshot predates it). True orphans are handled per `broker_drift_action`: **`"flatten"` (default)** submits a price-capped marketable LMT at a fresh live quote (never adopted into the ledger — an adopted trade the strategy never signalled would contaminate every metric), re-cancelled + re-anchored each tick while it rests; a **same-side flatten that already FILLED within the last ~90 min blocks a resubmit** (`_recent_flatten_fill` — the positions snapshot can lag a fill, and a duplicate would FLIP the position, not flatten it; opposite-side fills never block, so an over-flatten correction goes through; unreadable fills fail closed to `flatten_skipped`); `"report"` is the legacy surface-only behavior; **`ibkr_live` always refuses auto-flatten** (downgraded to report — auto-selling unexpected real-money positions is a human decision). `IBKRBroker.get_positions()` does a blocking `reqPositions()` refresh before reading — `ib.positions()` is a subscription cache that only advances while the event loop runs, so on the long-lived singleton connection it could be a full tick stale by sync time. Every broker call is exception-safe — a failure logs and degrades to report-only, never breaking the run. The internal NAV sim stays the source of truth for analytics. **`tracker.py` is not modified.**
- **Submission reliability (`_submit_with_retry`)**: every submit's broker answer is verified. **Transient** failures (connection drop, timeout, pacing — classified by `_is_transient_failure`) retry up to `broker_submit_retries` (default 2) with `broker_retry_wait_seconds` (default 5 s) pauses — a deliberately SHORT window so a delayed fill stays anchored to the tick's model price; **hard rejects** (insufficient funds, permissions, invalid contract) never retry. Before each retry a **duplicate guard** (`_known_order_result` via `Broker.get_open_orders()` + `get_fills()`) checks whether the errored attempt actually reached the broker — if so the existing order is adopted, never resubmitted (orderRef is a tag; IBKR does not dedupe, so a blind retry would double the position). Every retry is converted to a **marketable LMT capped at `broker_limit_cap_bps`** from the model price (even when `broker_order_type=MKT`), so the worst acceptable fill never drifts past the cap no matter when the retry lands. Failed attempts persist as `SUBMIT_FAILED` rows in `broker_orders`; stale cancels as `STALE_CANCEL` rows; per-run `retries` / `stale_cancels` counters surface in the email's broker-health line.
- **Order types:** `broker_order_type` = `MKT` (default) or `LMT` — a **marketable limit** at the model price ± the **session's cap** in the adverse direction (rounded away from the model to a valid tick): `broker_limit_cap_bps` (20) during RTH, `broker_limit_cap_bps_extended` (80 — the extended-book spread runs ~4× RTH, matching the sim's 4× cost multiplier; a 20 bp cap sits inside the extended spread and can never fill) off-hours. The cap bounds the worst acceptable fill; an order whose cap is exceeded rests unfilled until the tick-scoped pass re-anchors it. The design goal is **fill in the decision tick at the current spread** rather than later at a drifted price. **Off-RTH ticks override the configured type**: IBKR rejects MKT outside regular hours, so extended-session submissions are always marketable LMT with `outsideRth=True` (`OrderRequest.outside_rth`); an order submitted overnight rests until the 04:00 pre-market open. Class-share tickers are normalised both ways (`to_ib_symbol`/`from_ib_symbol` in `ibkr.py`: yfinance `BRK-B` ↔ IBKR `BRK B`).
- **Execution measurement (persisted):** slippage is recorded for **entries and exits** as **cost-normalized bps** (positive = adverse for either side), alongside actual IBKR commissions captured from fills. Each `sync()` report — one summary row plus one event row per order submission / fill repair — is persisted to DuckDB (`broker_reconciles` + `broker_orders` tables, `repo.insert_broker_report`, called from `_persist_run`), so the paper phase accumulates a durable slippage/reject/drift record instead of losing it after each email.
- **Sizing** (`sizing.py`): two modes via `broker_sizing_mode` — `notional` (default: fixed `broker_base_notional` in `broker_base_notional_ccy`, e.g. 1000 CAD, converted to a USD share budget via live FX in `fx.py`) or `equity_pct` (% of account equity). Both × the existing 1.0/1.5/2.0× confidence multiplier, with max-positions and max-gross-exposure caps (all USD-normalised, since US securities are USD-priced).
- **Reuse, not rebuild:** `broker_*` fields ride along in the DuckDB JSON `data` column (no migration); `recommendation_id` is the idempotent IBKR `orderRef` — and the reconciler enforces **one broker order per client_ref, ever**: IBKR does not dedupe orderRef, so a second ledger trade sharing a ref is skipped and durably marked `DUPLICATE_REF_NOT_SUBMITTED` (entry and exit passes share the per-tick ref set; a twin exit sized from the full held position would flip the book short); a broker-health verdict (`_assess_broker_health` in `pipeline.py`) mirrors `_assess_llm_health` → CRITICAL log + email banner (`broker_health` var) + `🔔 BROKER` subject tag on drift/rejects, plus a green "N entries / M exits / fills repaired / avg slippage" line when healthy.
- **Ops:** IB Gateway must run alongside the scheduler and needs a ~daily re-login (use IBKR auto-restart + IBController). Before enabling: `python -m src.broker.smoketest` (connectivity), then `--order` for a 1-share round trip.

Phased rollout: `dry_run` → `ibkr_paper` + reconcile (run weeks; measure slippage / tracking-error / rejects) → flip to `ibkr_live` (port 4001) with a capital cap. Plan: `~/.claude/plans/snuggly-munching-piglet.md`.

### Configuration

All settings live in `config/settings.py` as a `pydantic-settings` `BaseSettings` class. Every field maps directly to an environment variable (uppercase). Add new fields with defaults there; they become available as `settings.field_name` everywhere. Never read `.env` directly — always go through `settings`.
