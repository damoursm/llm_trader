# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

```bash
python main.py                # Run once, console output only
python main.py --email        # Run once and send email report
python main.py --schedule     # Start APScheduler (8:00 AM ET, Mon–Fri, sends email)
```

Logs go to `logs/llm_trader_YYYY-MM-DD.log` (7-day rotation). The HTML report (when `ENABLE_CHARTS=true`) is saved to `logs/report_YYYY-MM-DD_HHMM.html`.

To force a fresh run ignoring caches, delete `cache/news_*.json` and/or `cache/snapshots_*.json`. The OHLCV cache (`cache/ohlcv/`) and trades ledger (`cache/trades.json`) should never be deleted casually.

## Architecture

### Pipeline flow (`src/pipeline.py`)

`run_pipeline()` is the single entry point. Steps execute in sequence:

```
0. Ticker discovery      — trending.py → expands universe beyond static watchlist
1. News fetch            — news_fetcher.py → RSS + NewsAPI, cached hourly
1b. SEC 8-K filings      — eight_k.py → always fresh, appended to articles list
1c. Google Trends        — google_trends.py → search interest spikes, cached daily
1d. Reddit sentiment     — reddit_sentiment.py → WSB/stocks/investing, cached hourly
1e. Analyst ratings      — analyst_ratings.py → upgrades/downgrades/PT changes, cached daily
1f. EPS surprises        — earnings.py → recent beat/miss articles, cached daily
1g. Short interest       — short_interest.py → FINRA Reg SHO + yfinance squeeze/covering, cached daily
2. Market snapshots      — market_data.py → polygon_client.py (single batch REST call) primary, yfinance fallback for anything Polygon doesn't cover (indices, futures, OTC), cached hourly
3a. Insider/pol trades   — insider_trades.py → House/Senate watchers + EDGAR Form 4
3b. Options flow         — options_flow.py → yfinance sweep detection
3c. SEC EDGAR filings    — sec_filings.py → 13D/13G, Form 144, 13F superinvestors
3d. FRED macro context   — fred.py → yield curve, CPI, unemployment, credit spreads
3e. CFTC COT             — cot.py → weekly futures positioning, cached by ISO week
3f. IPO pipeline         — ipo_pipeline.py → S-1/S-11 sector demand signal, cached daily
3g. VIX & term structure  — vix.py → ^VIX/^VXN/^VVIX/^VIX3M/^VXMT, cached daily
3h. Put/call ratio        — put_call.py → CBOE market-wide + per-ticker extremes, cached daily
3i. Earnings calendar     — earnings.py → upcoming dates + IV warning, cached daily
3j. Credit market         — credit.py → HYG vs SPY divergence, leading indicator, cached daily
3k. Market breadth        — breadth.py → % of sector ETFs above 200d SMA, thrust detection, cached daily
3l. McClellan Oscillator  — mcclellan.py → EMA19−EMA39 of NYSE A/D, Summation Index, zero-cross, cached daily
3m. New 52-week highs/lows — highs_lows.py → HL Spread = %near-highs − %near-lows, divergence detection, cached daily
3n. Macro Surprise Index  — macro_surprise.py → CESI-style: actual FRED vs trailing 3-period avg, composite z-score, cached daily
3o. Fed Rate Expectations — fedwatch.py → T-bill spread proxy for CME FedWatch: implied cuts/hikes bp, P(cut/hold/hike) at next FOMC, rate-trend shift, cached daily
3p. Revision Momentum     — revision_momentum.py → analyst PT/rating trend: recent (0-30d) vs prior (31-60d) upgrade/downgrade delta, momentum score per ticker, cached daily
3q. Earnings Whisper      — earnings_whisper.py → implied whisper = consensus × (1 + avg_historical_beat%); beat-rate, eps_trend direction, net revisions; cached daily
3r. Insider cluster       — aggregator.py → _detect_insider_cluster(): ≥3 different insiders buying within 5 days → 1.75× amplifier on insider_score; cluster_detected/size stored on TickerSignal; cluster_watchlist.py → persists detections to cache/cluster_watchlist.json and injects active tickers into the analysis universe for 10 days (lead-indicator window; clusters historically precede moves by 5–20 days)
3s. OpEx calendar         — opex.py → compute_opex_context(): pure date math; 3rd-Friday detection, Triple Witching flag, POST_OPEX window; amplifies/discounts max_pain_score confidence in Claude prompt
3t. Seasonality calendar  — seasonality.py → compute_seasonality_context(): pure date math; monthly historical biases (April strongest, September weakest, Sell in May), end-of-month/quarter rebalancing windows, January effect (small-cap IWM), quarterly window dressing; STRONG_TAILWIND→STRONG_HEADWIND composite signal
3u. Bond market internals — bond_internals.py → 10Y-3M yield curve (recession predictor), TLT/IEF/TIP/LQD 1/4/8-week momentum, real yield, IG credit; bond-equity divergence (TLT vs SPY 5d return — EQUITY_CATCHUP_LIKELY when bonds rally hard while equities hold); RISK_OFF→RISK_ON composite regime; cached daily
3v. MOVE Index            — move.py → ICE BofA Treasury implied vol (^MOVE primary, VXTLT fallback); CALM→PANIC signal, 5d spike detection, MOVE/VIX ratio divergence; cached daily
3w. Global macro          — global_macro.py → DXY strength (DX-Y.NYB: strong dollar = headwind for EM/commodities/multinationals) + Copper/Gold ratio (HG=F/GC=F: Dr. Copper growth barometer) + Oil/Bond divergence (CL=F vs TLT: co-rally = POLICY_PIVOT_SIGNAL; oil up + bonds down = STAGFLATION_RISK; oil down + bonds up = GROWTH_FEAR_RISK_OFF); RISK_OFF→RISK_ON composite; cached daily
3x. Macro Regime Filter   — macro_regime.py → compute_macro_regime(): composite of VIX+MOVE+bond+global+FRED+breadth+credit into PANIC|RISK_OFF|CAUTION|NEUTRAL|RISK_ON; gates BUY entries and adjusts actionable threshold (PANIC=88%, RISK_OFF=82%, CAUTION=80%, NEUTRAL=78%, RISK_ON=72%); no cache (instant, pure computation from already-fetched contexts)
3y. Market Mode Switching — market_mode.py → compute_market_mode(): VIX+breadth+HL+McClellan composite into TRENDING|NEUTRAL|CHOPPY; dynamically adjusts aggregator signal weights: TRENDING up-weights tech/news (momentum bias), down-weights vwap/put_call; CHOPPY up-weights vwap/put_call (mean-reversion bias), down-weights tech; passed to build_signals() as weight_profile override; no cache (instant)
3z. Catalyst Timing      — catalyst_timing.py → three event-driven guards applied after recommendations are ranked: (1) Earnings Blackout: tickers within 2 days of earnings are removed from actionable BUY/SELL set (IV crush/gap risk); (2) OpEx Max-Pain Amplifier: during OpEx week, max_pain weight in aggregator boosted 0.12→0.20; Triple Witching (Mar/Jun/Sep/Dec) → 0.28; (3) 8-K + Insider Buy WATCH elevation: when both a recent 8-K filing and an insider purchase coincide for the same ticker, the ticker is auto-elevated from HOLD→WATCH or injected as a new WATCH if not yet in the top-10; no cache (instant, uses already-fetched contexts)
3A. Pattern Recognition  — pattern_recognition.py → compute_pattern_score(): detects 8 classical chart patterns (double_bottom, inv_head_shoulders, ascending_triangle, bull_flag, double_top, head_shoulders, descending_triangle, bear_flag) using local extrema on the last 60 bars; scores each detection by its historical win rate on that specific ticker (cold path: fetches 2y of history, scans 40-bar sliding windows with step=5, records 5d/10d forward returns — library cached 7 days in cache/patterns/<TICKER>.json; warm path: instant lookup + current detection); score = (win_rate − 0.5) × 2 × inherent_direction ∈ [−1, +1]; falls back to weak prior (±0.25) when fewer than 3 historical occurrences exist; base weight 0.18 in aggregator
3B. Sector Rotation      — sector_rotation.py → fetch_sector_rotation_context(): "Ebb and Flow" mechanism — tracks where money is entering and exiting across all 11 SPDR sector ETFs; computes excess return vs SPY (1w/1m/3m), volume ratio (5d/20d), and a cross-sectional z-score rotation_score ∈ [-1,+1]; volume modifier amplifies conviction; classifies STRONG_INFLOW/INFLOW/NEUTRAL/OUTFLOW/STRONG_OUTFLOW per sector; derives cyclical vs defensive avg spread → RISK_ON/NEUTRAL/RISK_OFF regime; surfaces explicit rotation pairs (e.g., "XLK → XLP") and top 3 inflow/outflow sectors; cached daily in cache/sector_rotation_YYYY-MM-DD.json
3C. Rotation Drivers     — rotation_drivers.py → fetch_rotation_drivers_context(): rate-cycle phase from FRED DFF trajectory (3m/12m bp change) + CPIAUCSL inflation trend; real rate = DFF − CPI; classifies EARLY_TIGHTENING|PEAK_TIGHTENING|TIGHTENING_PAUSE|PIVOT_IMMINENT|EASING_CYCLE|NEUTRAL; maps each phase to favoured/avoid asset lists (e.g., PIVOT_IMMINENT → favour TLT/XLRE/XLU, avoid XLE); Claude instruction 27c applies ±0.03–0.04 confidence adjustments per phase; optionally enhanced by FedWatch implied_cuts_12m_bp for PIVOT_IMMINENT detection; cached daily in cache/rotation_drivers_YYYY-MM-DD.json; requires FRED_API_KEY
3D. Business Cycle Rotation — business_cycle_rotation.py → compute_business_cycle_context(): Fidelity-style structural economic cycle phase derived from already-fetched FRED macro context (regime, yield_curve_signal, inflation_signal, unemployment_trend — no new API calls); phases: EARLY_EXPANSION|MID_EXPANSION|LATE_EXPANSION|LATE_CYCLE|CONTRACTION|UNKNOWN; maps each phase to per-sector cycle scores ∈ [-1,+1] (EARLY: XLF/XLRE/XLY lead; MID: XLK/XLI lead; LATE: XLE/XLB lead; LATE_CYCLE/CONTRACTION: XLV/XLP/XLU lead); convergence check vs Ebb-and-Flow real-time inflows; Claude instruction 27d applies ±0.03 confidence adjustments per cycle phase and surfaces sector-leadership guidance; no cache (instant, pure synthesis); ENABLE_BUSINESS_CYCLE_ROTATION=true
3E. Price Momentum (Perceived Value) — price_momentum.py → compute_price_momentum_score(): 1m (21-bar) and 2m (42-bar) returns normalised against the ticker's own trailing 252-bar return distribution (rolling std of 21-bar returns); z_composite = 0.6×z_1m + 0.4×z_2m; volume-confirmation adjustment ±0.10 (uptrend+rising vol → +0.10; downtrend+rising vol → -0.10; thin vol → -0.05×sign); tanh(z/1.5) + vol_adj clamped to [-1,+1]; weight 0.18 in aggregator; OHLCV cache-first (load_ohlcv() → works with ENABLE_FETCH_DATA=false); falls back to get_history(18mo); minimum 50 bars; tracked in performance attribution under "Technical" category; email section 5x: Price Momentum Leaderboard (top 5 chased / bottom 5 sold); ENABLE_PRICE_MOMENTUM=true
3F. Money Flow Indicators — money_flow.py → compute_money_flow_score(): three complementary volume-based indicators combined into a single accumulation/distribution score; MFI 14-period (volume-weighted RSI: <20=accumulation zone, >80=distribution zone; contrarian score = tanh((50−MFI)/20)); CMF 20-period (Chaikin Money Flow: positive=buyers in control, negative=sellers; score = tanh(CMF/0.15)); OBV 21-bar linear-regression slope z-score (rising=sustained buying, score = tanh(obv_z)); composite = 0.40×mfi_score + 0.40×cmf_score + 0.20×obv_score, final = tanh(composite/0.6) clamped to [-1,+1]; weight 0.15 in aggregator; OHLCV cache-first (works with ENABLE_FETCH_DATA=false); falls back to get_history(18mo); minimum 30 bars; tracked in performance attribution under "Technical" category; email section 5z: Money Flow Leaderboard (top 5 accumulation / bottom 5 distribution); ENABLE_MONEY_FLOW=true
4a. Sector Pairs         — sector_pairs.py → find_sector_pairs(): scans _SECTOR_MAP for stocks and their sector ETF having opposing non-NEUTRAL directions (both ≥35% confidence); forms market-neutral LONG/SHORT pair trades that isolate idiosyncratic alpha from sector beta; ETF_BULL_STOCK_BEAR or ETF_BEAR_STOCK_BULL setups; reported in email but not tracked in trades.json (pair-trade accounting differs from single-leg); no cache (instant)
4. Signal aggregation    — aggregator.py → per-ticker combined score
5. Recommendations       — claude_analyst.py → Claude generates BUY/SELL/HOLD/WATCH
6. Performance tracking  — tracker.py → paper trades in cache/trades.json; confidence-scaled position sizing (1×/1.5×/2×) with per-sector cap of 3×; weighted avg return reported alongside equal-weight avg; method attribution: at entry each trade stores the 10 raw method scores + methods_agreeing list + dominant_method; get_performance_for_email() returns performance_table (unified breakdown rows), trades_svg (inline SVG equity curve + per-trade bars), attributed_count, and solo_method_perf (hypothetical per-method standalone simulation)
7. Charts + email        — charts/, email_sender.py
```

### Data flow design

**Two tiers of data:**
- **Per-ticker signals** (news, technical, insider): scored `[-1.0, +1.0]`, combined in `aggregator.py` with configurable weights into a `TickerSignal`. Convergence multiplier (1.25× when ≥2 methods agree, 0.60× when only 1 does) prevents single-source BUY/SELL calls.
- **Macro context** (FRED, COT): not per-ticker. Passed as structured context blocks directly into the Claude prompt. Claude applies regime overlays (recession → avoid POSITION longs; EXTREME_LONG COT → cap BUY conviction on that asset).
- **Macro Regime Filter** (`macro_regime.py`): hard pre-filter applied after Claude runs. Reads VIX, MOVE, bond internals, global macro, FRED, breadth, and credit contexts; computes a weighted composite regime (PANIC|RISK_OFF|CAUTION|NEUTRAL|RISK_ON) that gates actionable signals by raising the confidence threshold and blocking BUY entries during PANIC/RISK_OFF. This is the top-down overlay that prevents buying into market crashes.
- **Market Mode Switching** (`market_mode.py`): adjusts the aggregator's signal weight profile based on market structure. TRENDING (low VIX, healthy breadth) → up-weight `tech`/`news` (momentum), down-weight `vwap`/`put_call`. CHOPPY (high VIX, mixed breadth) → up-weight `vwap`/`put_call` (mean-reversion/contrarian), down-weight `tech`. Computed from VIX, breadth, highs/lows, and McClellan before `build_signals()` runs.
- **Catalyst Timing** (`catalyst_timing.py`): event-driven post-processing layer applied after recommendations are ranked. Three mechanisms: (1) Earnings Blackout removes tickers within 2 days of earnings from the actionable set; (2) OpEx Amplifier boosts `max_pain` weight in the aggregator during OpEx week; (3) 8-K + Insider Buy WATCH elevation promotes or injects WATCH recommendations when material catalyst + insider conviction coincide.

**8-K filings, Google Trends, Reddit sentiment, analyst ratings, EPS surprises, and short interest are treated as news** — all return `List[NewsArticle]` and are scored by the same DeepSeek sentiment pipeline as RSS articles. Google Trends articles describe search interest spikes/drops; Reddit articles summarise mention count and upvote-weighted sentiment across r/wallstreetbets, r/stocks, and r/investing. Short interest articles surface squeeze setups, bearish positioning builds, and short covering signals.

**Smart money signals** (insider trades, options flow, SEC filings) all return `List[InsiderTrade]` and are combined into a single `insider_score` per ticker in the aggregator.

**Method attribution** (`tracker.py`): every new trade stores the 10 raw method scores (`news`, `tech`, `insider`, `put_call`, `max_pain`, `oi_skew`, `vwap`, `pattern`, `momentum`, `money_flow`) at entry time plus `methods_agreeing` and `dominant_method`. `get_performance_for_email()` returns:
- `performance_table`: unified list of breakdown rows (columns: trades, win_rate, compound_return, avg_return, wtd_avg_return, best, worst) grouped as: **total** (All Trades) → **asset** (Stocks only / ETFs only / Commodities only) → **direction** (Longs only BUY / Shorts only SELL) → **method** (per signal method, ≥2 attributed trades required). Rows are rendered in the email "Performance Breakdown" section with visual dividers between groups.
- `trades_svg`: inline SVG (820×336px dark-theme) with equity curve (top panel) and per-trade return bars (bottom panel). Embedded directly in the email via `{{ perf.trades_svg | safe }}` — no kaleido/Plotly dependency.
- `attributed_count`: count of trades with `methods_agreeing` populated (used to decide whether method rows appear).
- `solo_method_perf`: dict from `compute_solo_method_performance()` — for each closed trade with stored method_scores, asks "what direction would this method alone have signalled?" Same direction as actual → actual return; opposite → negated return; |score| < 0.10 (no view) → skipped. Rendered in email section 4b as a standalone table (trades, win_rate, compound_return, avg_return, best, worst per method). Distinct from `performance_table` method rows which only count trades where the method agreed with the aggregated direction.
- `method_eval_stats`: dict from `compute_method_eval_stats()` — per-method directional accuracy and conviction calibration. For each method: overall directional_accuracy (% correct directions), avg_return_correct, avg_return_wrong, and conviction_bands (Low 0.10–0.35 / Medium 0.35–0.65 / High 0.65+) each with trades, accuracy, avg_return. Rendered in email section 4c as per-method cards. A well-calibrated signal shows rising accuracy Low→High; flat/declining accuracy suggests the method adds noise rather than genuine directional insight.
Legacy trades without `methods_agreeing` are excluded from method rows but included in total/asset/direction rows.

### Model routing

| Task | Model |
|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`), Haiku 4.5 fallback |
| Final synthesis / BUY/SELL/HOLD/WATCH | Configurable via `ANALYST_MODEL` (default: `claude-haiku-4-5-20251001`) |

Available analyst models (set in `.env`):
- `claude-haiku-4-5-20251001` — fastest, cheapest; 8 192 output tokens
- `claude-sonnet-4-6` — higher quality; 64 000 output tokens
- `claude-opus-4-7` — highest quality; 32 000 output tokens

**Claude API calls use streaming** (`client.messages.stream`) to avoid SDK timeout errors on large prompts (40+ tickers can take >10 minutes non-streaming). Text chunks are accumulated into a single string before JSON parsing.

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
| Global macro (DXY + Cu/Au) | `YYYY-MM-DD` | 1 day | `cache/global_macro_*.json` |
| Sector rotation (Ebb & Flow) | `YYYY-MM-DD` | 1 day | `cache/sector_rotation_*.json` |
| Rotation Drivers (rate cycle) | `YYYY-MM-DD` | 1 day | `cache/rotation_drivers_*.json` |
| OHLCV (charts) | per ticker | incremental | `cache/ohlcv/*.json` |
| COT positioning | ISO week | 1 week | `cache/cot_YYYY_WW.json` |
| Insider cluster watchlist | ticker | 10 days | `cache/cluster_watchlist.json` |
| Pattern libraries | per ticker | 7 days | `cache/patterns/<TICKER>.json` |
| Trades ledger | — | permanent | `cache/trades.json` |

**Important:** The news cache includes 8-K articles only on the run that fetches them (8-K fetch always runs fresh and appends to whichever articles list is active — from cache or live). COT data is cached by ISO week, so the first run each week downloads from CFTC; subsequent runs within the week are instant.

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
2. Portfolio Performance (stats + inline SVG equity curve)
3. Performance Breakdown (unified table: total/asset/direction/method rows)
4. Trade Details (BUY/SELL only)
5. Monitor List (HOLD/WATCH)
6. Analyst Ratings, EPS Surprises, Macro Regime Filter, Market Mode, Catalyst Timing, FRED, COT, VIX, McClellan, Highs/Lows, Breadth, Macro Surprise, FedWatch, Credit, MOVE, Bond Internals, Global Macro, Seasonality, OpEx, Put/Call, Short Interest, GEX, Revision Momentum, Earnings Whisper, IPO Pipeline, Reddit, Sector Pairs, Smart Money Signals

### Configuration

All settings live in `config/settings.py` as a `pydantic-settings` `BaseSettings` class. Every field maps directly to an environment variable (uppercase). Add new fields with defaults there; they become available as `settings.field_name` everywhere. Never read `.env` directly — always go through `settings`.
