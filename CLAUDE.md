# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
python main.py                # Run once, console output only
python main.py --email        # Run once and send email report
python main.py --schedule     # Start APScheduler (every 30 min, 9:30–16:00 ET, Mon–Fri; emails at the 16:00 close)
python main.py --dashboard    # Launch the read-only monitoring dashboard (Plotly Dash via waitress)
```

Logs go to `logs/llm_trader_YYYY-MM-DD.log` (7-day rotation). The HTML report (when `ENABLE_CHARTS=true`) is saved to `logs/report_YYYY-MM-DD_HHMM.html`.

To force a fresh run ignoring caches, delete `cache/news_*.json` and/or `cache/snapshots_*.json`. The OHLCV cache (`cache/ohlcv/`) and the DuckDB database (`data/llm_trader.db` — trades, recommendations, run history) should never be deleted casually.

**Intraday operation (no unclosed-bar look-ahead).** The scheduled runner ticks **every 30 min, 9:30–16:00 ET** (Mon–Fri). The pipeline marks and fills on **live** prices (`_fetch_price` → `fast_info.last_price`); the daily OHLCV feeding the indicators is **completed-bars-only** — `market_data._drop_forming_bar` removes the still-forming current-session bar until the 16:00 close, so no indicator or NAV walk ever reads an unclosed daily bar. Open-position NAV is **marked to the live price each tick** (`daily_nav._build_marks` uses `current_price` as the open-trade end anchor). A **Hybrid** intraday-timing overlay (`src/signals/intraday_timing.py`) affects entry/exit *timing* only: it defers an entry whose 30-min momentum strongly opposes it (`enable_intraday_timing`, default on) and can close on a hard intraday reversal (`enable_intraday_exit`, opt-in → `exit_reason = "intraday_reversal"`). It never changes the daily-signal direction.

## Architecture

### Pipeline flow (`src/pipeline.py`)

`run_pipeline()` is the single entry point. Steps execute in sequence:

```
0.  Universe construction   — Static watchlist + dynamic discovery (trending, screener, macro→holdings, catalyst/analyst/earnings expansion), then a liquidity gate filters untradeable microcaps.
1.  News & sentiment feeds  — RSS, NewsAPI, 8-K filings, Google Trends, Reddit, analyst ratings, EPS surprises, short interest. All normalised to NewsArticle.
2.  Market snapshots        — Polygon batch primary, yfinance fallback for indices/futures/OTC.
3.  Smart-money + macro     — Insider/congressional trades, options flow, SEC filings (13D/G, Form 144, 13F, Form 4), plus market-context fetchers (FRED, COT, VIX, MOVE, DIX, breadth, credit, bond/global macro, sector rotation, business cycle, etc.).
3X. Regime & timing layers  — Macro Regime Filter, Market Mode, Catalyst Timing — gate BUYs, adjust thresholds, and tweak aggregator weights.
3#. Per-ticker scorers      — News sentiment, sentiment velocity, technical, trend strength, momentum, money flow, pattern recognition, IV rank/expression, PEAD, cointegration, cross-sectional rank. Each returns a score in [-1, +1].
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
- **Catalyst Timing** (`catalyst_timing.py`): event-driven post-processing layer applied after recommendations are ranked. Three mechanisms: (1) Earnings Blackout removes tickers within 2 days of earnings from the actionable set; (2) OpEx Amplifier boosts `max_pain` weight in the aggregator during OpEx week; (3) 8-K + Insider Buy WATCH elevation promotes or injects WATCH recommendations when material catalyst + insider conviction coincide.

**8-K filings, Google Trends, Reddit sentiment, analyst ratings, EPS surprises, and short interest are treated as news** — all return `List[NewsArticle]` and are scored by the same DeepSeek sentiment pipeline as RSS articles. Google Trends articles describe search interest spikes/drops; Reddit articles summarise mention count and upvote-weighted sentiment across r/wallstreetbets, r/stocks, and r/investing. Short interest articles surface squeeze setups, bearish positioning builds, and short covering signals.

**Smart money signals** (insider trades, options flow, SEC filings) all return `List[InsiderTrade]` and are combined into a single `insider_score` per ticker in the aggregator.

**Method attribution** (`tracker.py`): every new trade stores the raw method scores (`news`, `sent_velocity`, `tech`, `insider`, `put_call`, `max_pain`, `oi_skew`, `vwap`, `pattern`, `momentum`, `money_flow`, `trend_strength`, `pead`, `iv_rank`, `iv_expr`, `coint`, `cross_sectional`) at entry time plus `methods_agreeing` and `dominant_method`. `get_performance_for_email()` returns:
- `performance_table`: unified list of breakdown rows (columns: trades, win_rate, compound_return, avg_return, wtd_avg_return, best, worst) grouped as: **total** (All Trades) → **asset** (Stocks only / ETFs only / Commodities only) → **direction** (Longs only BUY / Shorts only SELL) → **method** (per signal method, ≥2 attributed trades required). Rows are rendered in the email "Performance Breakdown" section with visual dividers between groups. **Includes open trades at their current M2M `return_pct`** (maintained each run by `update_open_trades()`) — treated as hypothetical exits so every live position is reflected in the breakdown.
- `trades_svg`: inline SVG (820×336px dark-theme) with equity curve (top panel) and per-trade return bars (bottom panel). Embedded directly in the email via `{{ perf.trades_svg | safe }}` — no kaleido/Plotly dependency.
- `portfolio_metrics`: dict from `compute_portfolio_metrics()` — delegates to `daily_nav.compute_compound_return()` (see [Performance calculation](#performance-calculation) below) for a path-faithful daily-walked compound over real OHLCV closes. For every calendar day with any active position the engine forms the capital-weighted average daily return across active trades (`Σ(r·w)/Σ(w)` with `w = position_size_multiplier`) and compounds sequentially. Keys: `compound_inception` (every trade ever), `return_1w`/`return_2w`/`return_1m` (trades with `entry_date >= today − N days`; open trades included at the live M2M end-anchor). Overwrites `stats["compound_return"]` so the headline number matches the tiles. Displayed in the email Portfolio Performance card as time-window tiles.
- `method_order_by_winrate`: list of method keys sorted descending by solo win rate (no-data methods last). Used to order the solo performance and method eval tables in the email — best-performing signals appear first.
- `attributed_count`: count of trades with `methods_agreeing` populated (used to decide whether method rows appear).
- `solo_method_perf`: dict from `compute_solo_method_performance()` — for each closed trade with stored method_scores, asks "what direction would this method alone have signalled?" Same direction as actual → actual return; opposite → negated return; |score| < 0.10 (no view) → skipped. Rendered in email section 4b as a standalone table (trades, win_rate, compound_return, avg_return, best, worst per method). Distinct from `performance_table` method rows which only count trades where the method agreed with the aggregated direction.
- `method_eval_stats`: dict from `compute_method_eval_stats()` — per-method directional accuracy and conviction calibration. For each method: overall directional_accuracy (% correct directions), avg_return_correct, avg_return_wrong, and conviction_bands (Low 0.10–0.35 / Medium 0.35–0.65 / High 0.65+) each with trades, accuracy, avg_return. Rendered in email section 4c as per-method cards. A well-calibrated signal shows rising accuracy Low→High; flat/declining accuracy suggests the method adds noise rather than genuine directional insight.
Legacy trades without `methods_agreeing` are excluded from method rows but included in total/asset/direction rows.

### Performance calculation

Performance numbers come from two engines that work side-by-side, both rooted in the *real* observed prices stored on each trade — never interpolated, never synthesised.

**Per-trade `return_pct` (buy-and-hold, spread-aware) — `tracker._pct_return()`**

```
half_in  = _dynamic_half_spread(entry_price, asset_type)
half_out = _dynamic_half_spread(exit_or_current_price, asset_type)
BUY  : eff_entry = entry × (1 + half_in);   eff_exit = exit × (1 − half_out)
       return_pct = (eff_exit − eff_entry) / eff_entry × 100
SELL : eff_entry = entry × (1 − half_in);   eff_exit = exit × (1 + half_out)
       return_pct = (eff_entry − eff_exit) / eff_entry × 100
```

Bid-ask half-spread is **price-tiered and asset-type-aware** (ETF 1 bp; large-cap stock ≥ \$50 → 2 bp; \$10–\$50 → 4 bp; \$1–\$10 → 12.5 bp; \$0.10–\$1 → 37.5 bp; \$0.01–\$0.10 → 100 bp; sub-penny → 250 bp; commodity ≥ \$100 → 1.5 bp, < \$100 → 3 bp). Closed trades use `exit_price`; open trades use the live `current_price` (refreshed each tick by `update_open_trades()`, timestamped via `current_price_datetime`). For open trades this is a "what if you closed right now" mark — same convention as the email's M2M display. `_normalize_closed_returns()` recomputes this each pipeline tick from the stored entry/exit prices so every closed trade's `return_pct` reflects the current spread model.

**Path-faithful daily compound — `daily_nav.compute_compound_return(trades)`**

Per-trade walk:
1. Load OHLCV closes for the ticker from `cache/ohlcv/<TICKER>.json` (force-refreshed for open-trade tickers via `tracker._refresh_open_trade_ohlcv()` so today−1's close exists).
2. Build chronological marks: `(entry_date, eff_entry)` → every cached close strictly between → `(exit_date or today, eff_exit)`. Intermediate marks are raw closes, **no spread applied** — those days the position is marked-to-market, not traded.
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
| `record_new_trades()` | After Claude produces recommendations | Opens a trade per actionable BUY/SELL; stamps `entry_datetime` + `entry_price` together; applies the correlation-aware sizing haircut (continuous scale-down vs same-direction open peers, hard skip when correlated-exposure cap is hit); stores 17 raw method scores + `methods_agreeing` + `dominant_method`. |
| `close_trades_on_signal_reversal()` | Before opening new trades, if today's actionable signal reverses an open position | Closes with `exit_datetime = current_price_datetime` (most recent live mark) — no second fetch. |
| `_refresh_open_trade_ohlcv()` | First step of `update_open_trades()` | Force-refreshes OHLCV cache for every open-trade ticker (bypasses 3-day TTL) so the daily walk has a real close for every day held. |
| `_normalize_closed_returns()` | Second step of `update_open_trades()` | Idempotently rederives `return_pct` for every closed trade through the current spread model. |
| `update_open_trades()` | Each pipeline tick | Refreshes `current_price` + `current_price_datetime` + `return_pct` for every OPEN trade. **No time cap** — positions are never closed on age. |
| `monitor_open_positions()` | Each tick, after `update_open_trades` | The primary exit: closes a position when its thesis deteriorates — `macro_regime_exit` (long in PANIC/RISK_OFF), `signal_flipped`, `signal_decay` (entry vs today strength), or `confidence_loss` (today's confidence below an entry-relative floor). Toggle with `enable_signal_decay_exits`. |

### Database (DuckDB)

`data/llm_trader.db` is the **single source of truth** for trades, recommendations, and run history — it replaces the old `cache/trades.json` / `cache/hypothetical_trades.json` ledgers (now import-only). All access goes through `src/db/`:

- **`connection.py`** — `connect(read_only=False)` yields a short-lived DuckDB handle and closes it on exit. Read-write opens `ensure_schema()` first; read-only opens require the file to already exist (raise `FileNotFoundError` otherwise).
- **`schema.py`** — idempotent `CREATE TABLE IF NOT EXISTS` for five tables: `runs`, `run_sources`, `recommendations`, `trades`, `hypothetical_trades`. Each trade / hypothetical row stores the **full dict in a JSON `data` column** (so `tracker.py` round-trips byte-identical dicts, exactly as the old JSON files did) alongside projected scalar columns used for SQL and the dashboard.
- **`repo.py`** — the typed read/write API: `load_trades()` / `save_trades()` (full-replace, matching the old whole-file rewrite semantics), `insert_run()`, `insert_run_sources()`, `insert_recommendations()`, and `fetch_df()` (DataFrame reads for the dashboard). `set_read_only(True)` flips every read path to open read-only — the dashboard sets this so it never takes the write lock.
- **`migrate.py`** — one-time `python -m src.db.migrate` imports the legacy JSON ledgers; idempotent (skips populated tables unless `--force`). As a safety net, `tracker._load_trades()` self-seeds the DB from `cache/trades.json` if the DB is empty, so the cutover never loses history even if migration is skipped.

**Concurrency model:** DuckDB allows a single read-write handle OR many read-only handles across processes. The pipeline is the **sole writer** and holds the write lock only momentarily (open → write → close); at the end of every run it persists `runs` + `run_sources` + `recommendations` (`pipeline.py`, wrapped in try/except so a DB hiccup never aborts the run). The dashboard connects **read-only** and retries with exponential backoff across the brief window the writer holds the lock.

### Monitoring dashboard (`dashboard/`)

A read-only Plotly Dash app for inspecting the database. Launch with `python main.py --dashboard` (host/port from `settings.dashboard_host` / `dashboard_port`, default `127.0.0.1:8050`).

- **`app.py`** — three tabs: **Recommendations & Rationale** (per-run data sources + recommendations), **Method Performance** (solo win-rate bars, per-method table, and an *LLM models used* breakdown — the exact synthesis & sentiment models per run, including DeepSeek / rule-based fallbacks), **Returns** (KPI tiles + equity curve + open/closed trades). Each tab's content is embedded as that `dcc.Tab`'s `children`, so the Tabs component swaps content **client-side** with no `tabs.value`→callback round trip (that round trip proved unreliable in the browser — every tab showed the first-rendered one). `app.layout` is the `serve_layout` **function**, rebuilt per page load so a long-running dashboard always reflects the latest run; the selected tab is persisted across reloads and `_safe()` wraps each tab body so a data hiccup shows a message instead of a blank page. Tables are sortable (click a header; shift-click for multi-sort), carry a per-column filter row, use human-readable headers, format numeric columns (kept numeric so they sort correctly), and render timestamps in Eastern time (`_fmt_et`). Every column header, KPI tile, and section heading carries a hover tooltip (`tooltip_header` / HTML `title`) explaining the metric. `run()` serves through **waitress** — a production-grade, multi-threaded, cross-platform WSGI server (the right choice on Windows; gunicorn doesn't run there) — wrapped in an **auto-restart supervisor loop** so a crash self-heals; it falls back to the Dash dev server only if waitress isn't installed.
- **`data.py`** — read-only accessors over `repo.fetch_df()`. Sets `repo.set_read_only(True)` at import, wraps every read in exponential-backoff retry (`_retry`, ~11 s budget) for the daily write-lock window, and caches the heavy `get_performance_for_email()` result for 60 s so tab switches stay responsive.
- **`figures.py`** — Plotly figures (method win-rate bars; equity curve via `src.charts.builder`).

### Model routing

| Task | Model | Fallback |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku 4.5 |
| Final synthesis / BUY/SELL/HOLD/WATCH | Configurable via `ANALYST_MODEL` (default: `claude-haiku-4-5-20251001`) | DeepSeek V3 (`deepseek-chat`) on any Claude API error, then rule-based `_fallback_recommendations()` |

Available analyst models (set in `.env`):
- `claude-haiku-4-5-20251001` — fastest, cheapest; 8 192 output tokens
- `claude-sonnet-4-6` — higher quality; 64 000 output tokens
- `claude-opus-4-7` — highest quality; 32 000 output tokens

**Claude API calls use streaming** (`client.messages.stream`) to avoid SDK timeout errors on large prompts (40+ tickers can take >10 minutes non-streaming). Text chunks are accumulated into a single string before JSON parsing.

**DeepSeek analyst fallback** (`_call_deepseek_analyst()` in `claude_analyst.py`): triggered automatically when Claude raises `anthropic.APIStatusError` (HTTP 400/401/402/403/429/5xx — covers credits exhausted, bad key, payment required, rate limit, server errors) or `anthropic.APIConnectionError`. Uses `deepseek-chat` (DeepSeek V3) with the identical prompt via the OpenAI-compatible streaming API. Requires `DEEPSEEK_API_KEY`. If DeepSeek also fails, falls back to rule-based `_fallback_recommendations()`. The source model name is logged at INFO level (`via <model>`).

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

1. **Per-ticker signal** (like sentiment/technical/insider): implement `[-1.0, +1.0]` scorer, add weight in `aggregator.py`, add `enable_*` flag in `settings.py`.
2. **Smart money signal** (like options flow, SEC filings): return `List[InsiderTrade]` from a new module, call from pipeline in the Step 3 block, extend `smart_money` list.
3. **Macro context** (like FRED, COT): return a new Pydantic model, pass to `generate_recommendations()` as an optional parameter, build a prompt block + instruction section in `claude_analyst.py`, render in email HTML template.

### Key constraints

**Polygon.io client** (`src/data/polygon_client.py`): wraps the Polygon REST API. `get_snapshots_batch()` fetches all tickers in two calls (snapshots + prev-close). `get_bars()` fetches OHLCV history. Falls back gracefully when `POLYGON_API_KEY` is absent — `is_available()` returns False and `market_data.py` routes everything through yfinance instead. Free Polygon tier covers all US equity/ETF tickers; indices (^VIX etc.) and futures (GC=F etc.) must use yfinance directly.

**SEC EDGAR rate limit:** 10 requests/second. All EDGAR calls use `_REQUEST_DELAY = 0.12–0.15s`. The `_ticker_index` (name→ticker) and `_ticker_cik` (ticker→CIK) maps are module-level globals populated once per process — never re-fetch them in a loop.

**yfinance rate limits:** When a 429 is returned, `market_data.py` applies exponential backoff (60s → 120s → 240s). After 3 failures the loop stops early. Options chain scanning (`options_flow.py`) is silent-fail per ticker.

**CFTC downloads:** The COT ZIP files are ~5–10 MB each. Both current and previous year are fetched if the current year has insufficient history. Always check the cache first.

**Actionable signal threshold:** Baseline is `confidence ≥ 0.78` AND `sources_agreeing ≥ 2`. The Macro Regime Filter adjusts this dynamically: RISK_ON → 72%, CAUTION → 80%, RISK_OFF → 82%, PANIC → 88%. BUY entries are additionally blocked entirely during PANIC and RISK_OFF regimes. A single strong signal source never produces a BUY/SELL regardless of score.

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

### Configuration

All settings live in `config/settings.py` as a `pydantic-settings` `BaseSettings` class. Every field maps directly to an environment variable (uppercase). Add new fields with defaults there; they become available as `settings.field_name` everywhere. Never read `.env` directly — always go through `settings`.
