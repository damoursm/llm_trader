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
2. Market snapshots      — market_data.py → yfinance, cached hourly
3a. Insider/pol trades   — insider_trades.py → House/Senate watchers + EDGAR Form 4
3b. Options flow         — options_flow.py → yfinance sweep detection
3c. SEC EDGAR filings    — sec_filings.py → 13D/13G, Form 144, 13F superinvestors
3d. FRED macro context   — fred.py → yield curve, CPI, unemployment, credit spreads
3e. CFTC COT             — cot.py → weekly futures positioning, cached by ISO week
3f. IPO pipeline         — ipo_pipeline.py → S-1/S-11 sector demand signal, cached daily
3g. VIX & term structure  — vix.py → ^VIX/^VXN/^VVIX/^VIX3M/^VXMT, cached daily
3h. Put/call ratio        — put_call.py → CBOE market-wide + per-ticker extremes, cached daily
3i. Earnings calendar     — earnings.py → upcoming dates + IV warning, cached daily
4. Signal aggregation    — aggregator.py → per-ticker combined score
5. Recommendations       — claude_analyst.py → Claude generates BUY/SELL/HOLD/WATCH
6. Performance tracking  — tracker.py → paper trades in cache/trades.json
7. Charts + email        — charts/, email_sender.py
```

### Data flow design

**Two tiers of data:**
- **Per-ticker signals** (news, technical, insider): scored `[-1.0, +1.0]`, combined in `aggregator.py` with configurable weights into a `TickerSignal`. Convergence multiplier (1.25× when ≥2 methods agree, 0.60× when only 1 does) prevents single-source BUY/SELL calls.
- **Macro context** (FRED, COT): not per-ticker. Passed as structured context blocks directly into the Claude prompt. Claude applies regime overlays (recession → avoid POSITION longs; EXTREME_LONG COT → cap BUY conviction on that asset).

**8-K filings, Google Trends, Reddit sentiment, analyst ratings, EPS surprises, and short interest are treated as news** — all return `List[NewsArticle]` and are scored by the same DeepSeek sentiment pipeline as RSS articles. Google Trends articles describe search interest spikes/drops; Reddit articles summarise mention count and upvote-weighted sentiment across r/wallstreetbets, r/stocks, and r/investing. Short interest articles surface squeeze setups, bearish positioning builds, and short covering signals.

**Smart money signals** (insider trades, options flow, SEC filings) all return `List[InsiderTrade]` and are combined into a single `insider_score` per ticker in the aggregator.

### Model routing

| Task | Model |
|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`), Haiku fallback |
| Final synthesis / BUY/SELL/HOLD/WATCH | Configurable via `ANALYST_MODEL` (default: `claude-haiku-4-5-20251001`) |

To use Sonnet for higher quality: set `ANALYST_MODEL=claude-sonnet-4-6` in `.env`.

### Caching strategy

| Cache | Key | TTL | Location |
|---|---|---|---|
| News + 8-K | `YYYY-MM-DD_HH` | 1 hour | `cache/news_*.json` |
| Snapshots | `YYYY-MM-DD_HH` | 1 hour | `cache/snapshots_*.json` |
| Reddit sentiment | `YYYY-MM-DD_HH` | 1 hour | `cache/reddit_*.json` |
| Google Trends | `YYYY-MM-DD` | 1 day | `cache/trends_*.json` |
| IPO pipeline | `YYYY-MM-DD` | 1 day | `cache/ipo_*.json` |
| VIX & term structure | `YYYY-MM-DD` | 1 day | `cache/vix_*.json` |
| Put/call ratio | `YYYY-MM-DD` | 1 day | `cache/put_call_*.json` |
| Analyst ratings | `YYYY-MM-DD` | 1 day | `cache/analyst_ratings_*.json` |
| Earnings surprises | `YYYY-MM-DD` | 1 day | `cache/earnings_surprises_*.json` |
| Earnings calendar | `YYYY-MM-DD` | 1 day | `cache/earnings_calendar_*.json` |
| Short interest | `YYYY-MM-DD` | 1 day | `cache/short_interest_*.json` |
| OHLCV (charts) | per ticker | incremental | `cache/ohlcv/*.json` |
| COT positioning | ISO week | 1 week | `cache/cot_YYYY_WW.json` |
| Trades ledger | — | permanent | `cache/trades.json` |

**Important:** The news cache includes 8-K articles only on the run that fetches them (8-K fetch always runs fresh and appends to whichever articles list is active — from cache or live). COT data is cached by ISO week, so the first run each week downloads from CFTC; subsequent runs within the week are instant.

### Adding a new data source

Three patterns depending on what the source produces:

1. **Per-ticker signal** (like sentiment/technical/insider): implement `[-1.0, +1.0]` scorer, add weight in `aggregator.py`, add `enable_*` flag in `settings.py`.
2. **Smart money signal** (like options flow, SEC filings): return `List[InsiderTrade]` from a new module, call from pipeline in the Step 3 block, extend `smart_money` list.
3. **Macro context** (like FRED, COT): return a new Pydantic model, pass to `generate_recommendations()` as an optional parameter, build a prompt block + instruction section in `claude_analyst.py`, render in email HTML template.

### Key constraints

**SEC EDGAR rate limit:** 10 requests/second. All EDGAR calls use `_REQUEST_DELAY = 0.12–0.15s`. The `_ticker_index` (name→ticker) and `_ticker_cik` (ticker→CIK) maps are module-level globals populated once per process — never re-fetch them in a loop.

**yfinance rate limits:** When a 429 is returned, `market_data.py` applies exponential backoff (60s → 120s → 240s). After 3 failures the loop stops early. Options chain scanning (`options_flow.py`) is silent-fail per ticker.

**CFTC downloads:** The COT ZIP files are ~5–10 MB each. Both current and previous year are fetched if the current year has insufficient history. Always check the cache first.

**Actionable signal threshold:** Only `BUY` and `SELL` with `confidence ≥ 0.78` AND `sources_agreeing ≥ 2` are considered actionable. A single strong signal source never produces a BUY/SELL regardless of score.

### Email HTML template

The HTML is a Jinja2 template string (`HTML_TEMPLATE`) embedded directly in `email_sender.py` — there is no separate `.html` file. When adding new sections, follow the existing comment convention (`<!-- ══ N — SECTION NAME ══ -->`) and pass new variables through the `Template(...).render(...)` call. All chart images are embedded as base64 inline (`cid:` references in the MIME structure).

### Configuration

All settings live in `config/settings.py` as a `pydantic-settings` `BaseSettings` class. Every field maps directly to an environment variable (uppercase). Add new fields with defaults there; they become available as `settings.field_name` everywhere. Never read `.env` directly — always go through `settings`.
