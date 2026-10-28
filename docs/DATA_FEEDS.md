# The data feeds (`src/data/feeds/`) — runbook

Live from 2026-10-25 (the user: "Can you make sure we can fetch those new data sources on production as needed, like
daily or every 30min?"). The free sources PREREG208 tested, archived by production on the cadence each one changes.
PREREG208 (2026-10-25) found that none of the three marks the spike shorts that keep squeezing; PREREG209 found that the
WallStreetBets titles mark the dip book's better dips: FROM THE 2026-10-26 SESSION THE DIP BOOK's daily top pick reads
the WSB store (its WallStreetBets preference, PREREG209's KD+ — `src/signals/dip_wsb.py`, docs/DIP_LONG.md: the titles
created in the 120 hours before 09:00 ET; a store that does not cover the window, or whose newest post is more than 60
minutes older than the cut, prefers nothing). The OCC's and the FDA's are read by no live book; a decision to use one
goes through a study and the pooled add rule.

## What is fetched, and when

| Source | What | Cadence | Store |
|---|---|---|---|
| r/wallstreetbets (the Arctic Shift archive) | every submission: id, time, title, score, comment count, flair — never the author | every run (30 min): what is new since the last post seen, the last `feeds_wsb_overlap_minutes` (120) re-read for late captures; a failing archive retried, asked every minute around the dip book's 09:00 cut, its gap filled from Reddit's own API once an app is configured (below: When the archive fails) | `data/reddit_wsb/<UTC day>.jsonl.gz` |
| The OCC's volume query | each session's options volume by underlying x account type (C customer, F firm, M market maker) x put / call x exchange | every run: each completed session not archived yet; one the OCC has not published asked again after `feeds_occ_retry_minutes` (60) | `data/occ_volume/<day>.csv.gz` (raw, **never delete**: the OCC serves only its last 2 years) + `agg/<day>.csv.gz` |
| openFDA's Drugs@FDA | every application's submissions and approvals | the export date checked every `feeds_fda_check_hours` (24); a new export downloaded | `data/openfda/drugsfda_<export>.json.zip` (the newest 4: each export repeats the whole history), `approvals.csv` |
| EDGAR's full-text search (`efts.QUERIES`; from the 2026-10-26 evening) | 8-Ks naming PDUFA — each document kept, the FDA target dates its text gives extracted — and three red flags, the hits only: going-concern doubts (10-K / 10-Q / 8-K), Item 4.02 non-reliance (8-K), Wells notices; each hit's accession, form, file date, CIKs, display names (the ticker), SIC codes, 8-K items and `seen_utc` | every `feeds_efts_check_hours` (6), each query from its last window's end minus `feeds_efts_overlap_days` (3) to today; never inside production's SEC lanes (`feeds_sec_quiet_et`: 23:30-00:45 and 08:15-09:45 ET), one SEC request a second (`feeds_sec_pace_seconds`) | `data/edgar_fts/<query>/hits/<YYYY-MM>.jsonl`, `pdufa/docs/<adsh>__<file>.gz`, `pdufa/dates.jsonl` |
| Cboe's daily options statistics (from the 2026-10-26 evening) | each session's call / put volume and open interest by product (all, index, exchange-traded products, equity, VIX, SPX + SPXW, ...) and Cboe's 24 put/call ratios | each completed session of the last `feeds_cboe_history_days` (30) not archived; one Cboe has not posted yet (it answers 403 until the evening) asked again after `feeds_cboe_retry_minutes` (60) | `data/cboe_options/daily/<day>.json.gz` (raw), `ratios.csv`, `volumes.csv`; `history.csv` (Cboe's CSVs 2006-11-01 – 2019-10-04) |
| The SEC's Financial Statement and Notes data sets (from the 2026-10-26 evening) | each 10-Q / 10-K's Rule 10b5-1 / non-Rule trading-arrangement facts (Reg S-K Item 408(a), from 2023-Q3: adopted / terminated, the person and title, the dates, the duration, the shares) with the dimension rows naming the person and the filing's acceptance instant | the list page every `feeds_plans_check_hours` (24); each listed file from `feeds_plans_start` (2026_10) not held downloaded and parsed once, one a run; never inside production's SEC lanes | `data/edgar_notes/raw/<name>_notes.zip`, `parsed/<name>.{facts,dims,subs}.jsonl.gz` |
| GDELT 1.0's daily Global Knowledge Graph (from the 2026-10-26 evening) | each UTC day's article count and tone sum for every news theme (ECON_STOCKMARKET, ECON_INFLATION, CRISISLEX_*, ...); the 17-65 MB file itself is not kept | each day from `feeds_gdelt_start` (2026-10-19) once posted (~07:00 UTC the next day), checked against GDELT's MD5 list, at most `feeds_gdelt_max_days` (3) a run (about 45 s a day) | `data/gdelt/agg/<day>.csv.gz`, `days.csv` |

The scheduler's own timer (`runner._start_feeds_timer`, 30 s poll) launches `python -m src.data.feeds --run` every
`feeds_poll_minutes` (30), 10-25 minutes into a half hour — never in a :00 / :30 slot's first minutes, where a tick's
live path runs — as a SUBPROCESS with its output in `logs/feeds_console.log` (overwritten each run; the scheduler's
log keeps each run's `feeds:` summary line), one run at a time across processes (`cache/feeds/feeds.lock`), killed
with its process tree past `feeds_budget_seconds` (900) + 300. The timer runs on weekends and holidays too (posts do
not stop); the OCC's sessions follow the NYSE calendar.

Each source's state is `cache/feeds/state.json` (``last_try`` / ``last_ok`` / ``last_error`` / ``last_msg``; the WSB
resume point ``after_utc`` / ``after_id``, its outage and fill keys — ``archive_error`` / ``archive_fail_since`` /
``archive_fail_after_utc`` / ``archive_last_ok``, ``fill_next_id`` / ``fill_reach_utc`` / ``fill_newest_utc`` /
``fill_last_calls`` / ``fill_error`` / ``fill_last_ok``, ``rereads``; the OCC's retry stamps, the FDA's export date). A
source that raises is recorded and the others still run; the WSB source raises only when the archive failed and
nothing filled its gap.

## When the archive fails (from the 2026-10-26 evening)

The user 2026-10-26: "Try to make the KD+ data availability more robust, whether by retrying or using the app under my
Reddit account." That morning Arctic Shift answered 422 / 500 / 530 from 08:10 ET: the dip book's WallStreetBets window
was stale at its 09:00 cut and KD+ stood down (the pick as without it). `src/data/feeds/wsb.py`, settings
`feeds_wsb_*` and `enable_feeds_wsb_reddit_fill`:

- **Retries.** A failed archive request keeps the pages it already served and is asked again `feeds_wsb_retries` (2)
  times, `feeds_wsb_retry_seconds` (30) apart and doubling, within the run's budget.
- **The pre-cut window.** On NYSE days from `feeds_wsb_precut_lead_minutes` (20) before `dip_long_wsb_cut_et` (09:00)
  to `feeds_wsb_precut_tail_minutes` (27) after it — 08:40 – 09:27 ET, the 08:40 and 09:10 runs — a failing archive is
  asked every `feeds_wsb_precut_retry_seconds` (60) until it answers or the run's budget ends: an archive back by about
  09:25 still makes the window fresh for the 09:30 signal computation.
- **The gap fill, from Reddit's own API** (an app registered under the user's Reddit account: `REDDIT_CLIENT_ID` /
  `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` in `.env`; application-only OAuth, no password): when the archive failed
  after its retries — or, inside the pre-cut window, the store's newest post is more than
  `dip_long_wsb_max_feed_lag_minutes` (60) old (the dip book's own staleness test). Reddit numbers every post in one
  base-36 sequence (about 1.45 million a day in October 2026), so every id from the newest one stored to the newest on
  `/r/all/new` is read through `/api/info`, 100 a call, oldest first — about 600 calls an hour of outage —, at most
  `feeds_wsb_reddit_calls_per_minute` (80; Reddit's free limit is 100) and `feeds_wsb_reddit_max_calls` (1,000) a run,
  stopping `feeds_wsb_reddit_margin_seconds` (90) before the budget; a longer gap continues at the next run
  (``fill_next_id``). r/wallstreetbets' posts are kept — the ones its moderators removed too, as the archive captures
  them (the subreddit's own listing and RSS hold about a sixth: 91 of the archive's 553 posts of 2026-10-21 .. 26) —
  stamped ``src`` "reddit".
- **The archive wins.** Its copy of a post always replaces a filled one and a filled copy never replaces its; its resume
  point never moves past its own posts, so it refetches the outage when it answers again, and the outage's span is read
  from it once more `feeds_wsb_reread_hours` (6) later (posts it captures late; ``rereads``). `dip_wsb` counts a filled
  post like the archive's, and its freshness test reads the store's newest post from either source.
- **Health.** From the scheduler's next restart (`feeds.health` runs in the scheduler), an archive outage is a note
  naming how its gap is filled: no Reddit app configured / Reddit's API failed too / filled through HH:MM UTC.
- **STATUS 2026-10-26: NO APP IS CONFIGURED.** The `.env` credentials of an earlier app are refused (401), and Reddit
  closed self-service API keys in November 2025 (its Responsible Builder Policy): a new app needs Reddit's approval
  first — until then its create-app form reloads with no error —, requested through that form's "register to use the
  API" link from the account that will own the app. Until it exists only the retries and the pre-cut loop run. Not yet
  verified (they need the app): that `/api/info` serves the removed posts' titles, and the fill's match to the archive
  over an hour both cover.
- **Tests.** `tests/test_data_feeds.py` (the retries keep the served pages, the fill by id with the archive's copy
  winning, a filled copy never replacing the archive's, the pre-cut loop, the fill when the archive answers but is
  behind, refused credentials and the rate limit, the health note) and `tests/test_dip_wsb.py` (filled posts count and
  keep the window fresh); the test baseline pins the retries and the pre-cut window to 0 and the fill off
  (`tests/conftest.py`).

## The free-data census's feeds (from the 2026-10-26 evening)

The user 2026-10-26, after the census of free sources (`memory/free-data-sources-census-2026-10-26.md`, report
https://claude.ai/artifact/EVRZC5mmhZuTHe3HmWTd94): "Continue exploring the new data sources and make sure we can pull it
in live for our production." `src/data/feeds/efts.py`, `cboe.py`, `plans.py`, `gdelt.py` (SEC requests through
`sec.py`), each a source of the same runner, with its own cadence (`skip_reason`: a source's cadence and the SEC's quiet
windows skip it for a run). No live book reads them: each is studied first — PREREG210 the PDUFA dates (a pre-decision
run-up long book; the vol and fall shorts' squeeze risk), PREREG213 the 10b5-1 plans, PREREG214 the red-flag filings
and the dip book; Cboe's put/call and GDELT's themes are the crisis books' context (the capitulation, crash and insider
books: a panic's put buying against a grind, a shock against a credit slide).

- Point in time: an EDGAR hit's ``seen_utc`` is when this feed first saw it (the history imported from research has
  none: its file date, or the filing index's acceptance instant, stands in); Cboe's session D and GDELT's day D are
  posted in the evening of D / about 03:00 ET of D+1 — a decision on D+1 reads them; a 10b5-1 fact is known at its
  filing's ``accepted``, and a monthly notes file is listed some weeks after its month (the research history covers
  it; live, a decision reads the files listed by then).
- Seeding (each under the feeds lock: `python -m src.data.feeds --import-efts pdufa DIR`, `--import-plans DIR`,
  `--import-cboe-history`, `--backfill-cboe START`, `--backfill-gdelt START END`): Cboe's history 2006-11-01 – today
  (the CSVs and every daily file from 2019-10-07) seeded 2026-10-26 evening; the PDUFA 8-Ks (4,744 filings 2003-01-02
  – 2026-10-22, 3,272 with an exact date) and the notes files (21 files 2023-Q3 – 2026-09: 344,148 facts in 56,846
  filings, equal to the research parse) imported the same evening from the census's research downloads
  (`llm_trader_research/data/biotech_calendars/pdufa/`, `.../edgar_text/fsn/`); GDELT's history (2015-02-18 on, about
  156 GB to stream, 9-26 hours) waits for the user's go-ahead.
- Readers: `efts.read_dates("pdufa")` / `efts.hits(query)`, `cboe.ratios(name, before=D)`, `plans.read("facts" |
  "dims" | "subs")`, `gdelt.shares(day, themes)`.
- Health: a census feed with no good fetch in 30 h (EDGAR, Cboe), 72 h (the notes files) or 36 h (GDELT) is a note,
  from the scheduler's next restart.

## Checks

- `python -m src.data.feeds --status` — each source's state and the health notes.
- `python -m src.data.feeds --run [--only wsb,occ,fda]` — one run by hand (the lock keeps it from overlapping the
  timer's).
- Health: a stale feed is a NOTE in the email's scorer banner (`feeds.health`): WSB no good fetch in 3 h (an Arctic
  Shift outage also a note naming how its gap is filled, above), the OCC in 30 h or a completed session of the last
  three missing after a day's grace, the FDA in 72 h. Never an alarm.
- Switches: `enable_data_feeds` (the timer and the runner), `enable_feeds_wsb` / `_occ` / `_fda` / `_efts` / `_cboe`
  / `_plans` / `_gdelt` (each source).

## Seeding and readers

- History: `python -m src.data.feeds --import-wsb DIR`, `--import-occ DIR`, `--import-fda ZIP EXPORT_DATE` merge
  PREREG208's research downloads (`llm_trader_research/data/reddit_wsb/raw`, `.../occ_volume/raw`,
  `.../openfda/drug-drugsfda-0001-of-0001.json.zip`) into the stores; a WSB day merges by post id, an OCC session
  already archived is kept. Seeded 2026-10-25: WSB every UTC day from 2018-01-01 (3,219 days through 2026-10-24, 2.6M
  posts, ~100 MB), the OCC every session from 2024-10-28, the FDA's export of 2026-10-23.
- Readers (point in time): `dip_wsb.window_counts(d, tickers)` — the dip book's: the titles naming each created in
  the 120 hours before 09:00 ET of session d (PREREG209's counting), with the window's status; `wsb.mentions(ticker,
  start, end)` — the titles naming it over UTC days (an upper-case whole
  word or a cashtag; an English or WSB word and a single letter only as a cashtag; None when a day is missing);
  `occ.sessions(before, n)` + `occ.volume(underlying, days, actype, porc)` — the sessions before a decision day (a
  closure's near-empty file is not a session); `fda.read(start, end)` — the original NDA / BLA approvals.

## The sources' limits

- Arctic Shift is a free community archive (its title and count searches time out on a subreddit this large; plain
  time windows work, 100 posts a request). Requests are sequential and paced (`feeds_pace_seconds`). The archive has
  its own holes — 2019-10-14 – 19 (0 – 34 posts a day; asked again, still empty), 2018-03-11 and 2018-04-08 (under a
  quarter of their 29-day median) —, which `wsb.mentions` reads as days without titles.
- Reddit itself (2026-10-26): its keyless JSON answers 403 from this box, its listings and RSS hold only the posts
  moderators kept (about a sixth of the archive's), PullPush answers 429; its API needs an app Reddit has approved
  (above), application-only OAuth at 100 calls a minute free.
- The OCC answers only its last 2 years ("Report date cannot be prior to 2 years"): the archive is the only history.
  A session it has not published yet answers "No record(s) found" or "Report date cannot be greater than" its latest
  (the second first seen at 16:10 ET 2026-10-26, a failure until fixed that evening): both are asked again later.
- Not available on our keys (2026-10-25): options prices / volume per contract (Polygon's options plan), Quiver's
  WallStreetBets dataset (sold on no plan any more; its lists end 2025-03). Earnings-call transcripts do answer on our
  Alpha Vantage key (EARNINGS_CALL_TRANSCRIPT, 2010 on, no publication time; the census of 2026-10-26,
  `memory/free-data-sources-census-2026-10-26.md`) — read by no feed.
