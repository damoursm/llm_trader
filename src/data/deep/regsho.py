"""REG SHO THRESHOLD LISTS — the daily lists of securities with persistent fails
to deliver (SEC Regulation SHO, Rule 203(c)(6): aggregate fails >= 10,000 shares
and >= 0.5% of shares outstanding for five consecutive settlement days). Each
listing market publishes the list for the securities it lists; a name on it is
hard to deliver, so a proxy for hard-to-borrow that reaches back years — the
borrow files themselves do not (`borrow.py`; user directive 2026-10-02: "Download
and add to new features the Reg SHO threshold lists").

Sources (free, one file per market per trading day, 2021 and earlier):

    nasdaq  ftp://ftp.nasdaqtrader.com/symboldirectory/regsho/nasdaqth{YYYYMMDD}.txt
            ~23:00 ET on D; history from 2005. The same file over https
            (www.nasdaqtrader.com/dynamic/symdir/regsho/) sits behind an Imperva
            bot shield that answers a script with a JavaScript challenge after a
            few requests (2026-10-02) — FTP does not
    nyse / amex / arca
            https://www.nyse.com/api/regulatory/threshold-securities/download
            ?selectedDate={DD-Mon-YYYY}&market={NYSE | NYSE American | NYSE Arca}
            ~22:00 ET on D. CAUTION: an unpublished day or a holiday answers 200
            with the header and a made-up trailer, exactly like an empty list —
            so only COMPLETED trading days are ever asked (`completed_days`)
    bzx     https://cdn.cboe.com/resources/us/equities/market-statistics/
            reg-sho-threshold/bzx_equities_reg_sho_threshold_{YYYYMMDD}.txt
            ~03:00 ET on D+1 (Cboe lists most of its ETFs on BZX); 403 until then

Every file is the list FOR date D (its trailer carries D). The last publisher
(Cboe) posts by ~03:05 ET on D+1, so the list of D is known from D+1 — before the
08:30 ET session cutoff of the next session, which the pre-open refresh fetches
(`deep_features.LAG_DAYS["regsho"]`).

Rows: ``date`` (D, ISO), ``market``, ``symbol`` (our spelling: '.', '/' and ' '
become '-'), ``symbol_raw``, ``name``. An empty list is a valid answer (the
manifest records the key done with 0 rows); ``calendar()`` turns the manifest
into the per-market list dates, which tells "not on the list" from "no list".

    python -m src.data.deep regsho [--since 2021-01-04]
"""
from __future__ import annotations

import io
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from src.data import deep as _deep
from src.data.deep import GENERIC_HEADERS, Manifest, RateLimiter, family_dir, write_parquet

FAMILY = "regsho"
MARKETS = ("nasdaq", "nyse", "amex", "arca", "bzx")
START = date(2021, 1, 4)
_NYSE_MARKET = {"nyse": "NYSE", "amex": "NYSE American", "arca": "NYSE Arca"}
# one keep-alive session per host, a browser agent, a back-off on an HTML page
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36 (research; daily Reg SHO lists)",
            "Accept-Encoding": "gzip, deflate"}
# nyse.com (Cloudflare) answered a 2.5 requests/s burst with 429 and Retry-After 2,538 s
# (2026-10-02) — one request per 2 s, and a Retry-After is honoured in full
_LIMITERS = {"nasdaq": RateLimiter(0.3), "nyse": RateLimiter(2.0), "bzx": RateLimiter(0.3)}
# the longest server-asked wait one fetch sits through; past it the key fails and the
# next run retries it. The backfill honours an hour; the refresh (`refresh_regsho`)
# lowers it so a rate limit can never hold the 08:30 pre-open run
MAX_WAIT_S = 3600.0
_SESSIONS: dict = {}
_SESSION_LOCK = threading.Lock()
_FTP_HOST, _FTP_DIR = "ftp.nasdaqtrader.com", "symboldirectory/regsho"
_FTP: dict = {"conn": None}
_FTP_LOCK = threading.Lock()
_ET = ZoneInfo("America/New_York")
_TRAILER = re.compile(r"^\d{14}$")


def key_of(d: date, market: str) -> str:
    return f"{d:%Y%m%d}_{market}"


def parse_key(key: str):
    ds, market = key.split("_", 1)
    return datetime.strptime(ds, "%Y%m%d").date(), market


def url_of(d: date, market: str) -> str:
    if market == "nasdaq":
        return f"ftp://{_FTP_HOST}/{_FTP_DIR}/nasdaqth{d:%Y%m%d}.txt"
    if market == "bzx":
        return ("https://cdn.cboe.com/resources/us/equities/market-statistics/reg-sho-threshold/"
                f"bzx_equities_reg_sho_threshold_{d:%Y%m%d}.txt")
    if market in _NYSE_MARKET:
        return ("https://www.nyse.com/api/regulatory/threshold-securities/download?selectedDate="
                f"{d.strftime('%d-%b-%Y')}&market={_NYSE_MARKET[market].replace(' ', '%20')}")
    raise ValueError(f"unknown market {market!r}")


def normalise(symbol: str) -> str:
    """The exchange's spelling -> ours (`BRK.B`, `BRK/B`, `BRK B` -> `BRK-B`)."""
    s = str(symbol).strip().upper()
    return re.sub(r"[./ ]+", "-", s)


def parse_list(text: str, d: date, market: str) -> pd.DataFrame:
    """One market's file -> rows. Header first, data `SYM|NAME|...`, an all-digit
    trailer last; a row whose threshold flag is present and not 'Y' is dropped."""
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if not lines or not lines[0].lower().startswith("symbol"):
        raise RuntimeError(f"no header ({(lines or [''])[0][:60]!r})")
    head = [h.strip().lower() for h in lines[0].split("|")]
    flag_at = next((i for i, h in enumerate(head) if "threshold flag" in h), None)
    rows: List[dict] = []
    for ln in lines[1:]:
        if _TRAILER.match(ln):
            continue
        parts = ln.split("|")
        if not parts[0].strip():
            continue
        if flag_at is not None and len(parts) > flag_at and parts[flag_at].strip().upper() not in ("Y", ""):
            continue
        rows.append({"date": d.isoformat(), "market": market, "symbol": normalise(parts[0]),
                     "symbol_raw": parts[0].strip(), "name": parts[1].strip() if len(parts) > 1 else ""})
    return pd.DataFrame(rows, columns=["date", "market", "symbol", "symbol_raw", "name"])


def _ftp_text(name: str) -> str:
    """One file from Nasdaq's FTP over a single shared connection (reconnected
    on any error). FileNotFoundError when the file is not there (yet)."""
    import ftplib
    import time
    with _FTP_LOCK:
        last: Optional[Exception] = None
        for attempt in range(4):
            _LIMITERS["nasdaq"].wait()
            try:
                if _FTP["conn"] is None:
                    c = ftplib.FTP(_FTP_HOST, timeout=60)
                    c.login()
                    _FTP["conn"] = c
                buf = io.BytesIO()
                _FTP["conn"].retrbinary(f"RETR {_FTP_DIR}/{name}", buf.write)
                return buf.getvalue().decode("latin-1")
            except ftplib.error_perm as e:
                if str(e)[:3] == "550":
                    raise FileNotFoundError(name) from e
                last = e
            except Exception as e:                       # noqa: BLE001
                last = e
            try:
                if _FTP["conn"] is not None:
                    _FTP["conn"].close()
            except Exception:                            # noqa: BLE001
                pass
            _FTP["conn"] = None
            time.sleep(5.0 * (attempt + 1))
        raise RuntimeError(f"FTP gave up: {last}")


def fetch(key: str) -> pd.DataFrame:
    """One (day, market) list. Raises when the file is not there (yet) — the
    manifest then retries it; returns an EMPTY frame for a published empty list."""
    import time
    import requests
    d, market = parse_key(key)
    if market == "nasdaq":
        try:
            return parse_list(_ftp_text(f"nasdaqth{d:%Y%m%d}.txt"), d, market)
        except FileNotFoundError as e:
            raise RuntimeError("not on the FTP (not published)") from e
    host = "nyse" if market in _NYSE_MARKET else market
    lim = _LIMITERS[host]
    with _SESSION_LOCK:
        sess = _SESSIONS.get(host)
        if sess is None:
            sess = _SESSIONS[host] = requests.Session()
            sess.headers.update(_HEADERS)
    last: Optional[Exception] = None
    for attempt in range(6):
        lim.wait()
        try:
            r = sess.get(url_of(d, market), timeout=60, allow_redirects=False)
        except Exception as e:                           # noqa: BLE001
            last = e
            time.sleep(5.0 * (attempt + 1))
            continue
        if r.status_code in (301, 302, 303, 307, 308, 403, 404):
            raise RuntimeError(f"HTTP {r.status_code} (not published)")
        if r.status_code == 429 or r.status_code >= 500:
            last = RuntimeError(f"HTTP {r.status_code}")
            try:
                wait = float(r.headers.get("Retry-After") or 0)
            except ValueError:
                wait = 0.0
            pause = max(wait + 5.0, 30.0 * 2 ** attempt)
            if pause > MAX_WAIT_S:
                raise RuntimeError(f"HTTP {r.status_code}, asked to wait {pause:.0f}s (> {MAX_WAIT_S:.0f}s)")
            time.sleep(pause)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        if r.text.lstrip()[:1] == "<":                  # an HTML block / error page, not the list
            last = RuntimeError("HTML page instead of the list")
            time.sleep(30.0 * (attempt + 1))
            continue
        return parse_list(r.text, d, market)
    raise RuntimeError(f"gave up: {last}")


_REF_NAMES = ("AAPL", "MSFT", "AMZN")


def store_sessions() -> List[date]:
    """The NYSE sessions the deep 30-minute store actually traded (regular-hours
    bars of a few names that never miss one) — the historical calendar.
    `market_calendar.is_market_day` only knows holidays from 2024 (2026-10-02:
    it called 2021's MLK Day, 2022's Juneteenth, 2023's Good Friday and the
    2025-01-09 day of mourning sessions), and NYSE answers a holiday with an
    empty list that would break every streak."""
    from src.data.intraday_store import DEEP_DIR as BARS_DIR
    days: set = set()
    for tk in _REF_NAMES:
        p = Path(BARS_DIR) / f"{tk}.pkl"
        if not p.exists():
            continue
        idx = pd.DatetimeIndex(pd.read_pickle(p).index)
        et = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(_ET)
        mins = et.hour * 60 + et.minute
        days |= set(et[(mins >= 570) & (mins < 960)].date)
    return sorted(days)


def completed_days(start: date = START, now: Optional[datetime] = None) -> List[date]:
    """Trading days whose lists every market has published by ``now``: up to the
    session before today, once it is 06:00 ET (Cboe posts D's list ~03:05 ET on
    D+1). Never today — NYSE answers an unpublished day like an empty list.
    Sessions come from the store (`store_sessions`); days after the store's last
    session from `market_calendar` (it knows the current holidays)."""
    from src.performance.market_calendar import is_market_day
    now = now or datetime.now(_ET)
    now = now.astimezone(_ET)
    last = now.date() - timedelta(days=1)
    if now.hour < 6:
        last -= timedelta(days=1)
    known = [d for d in store_sessions() if start <= d <= last]
    out = list(known)
    d = (known[-1] + timedelta(days=1)) if known else start
    while d <= last:
        if is_market_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def keys_for(days: Sequence[date], markets: Sequence[str] = MARKETS) -> List[str]:
    return [key_of(d, m) for d in days for m in markets]


def calendar() -> pd.DataFrame:
    """(date, market, rows) for every list fetched — empty lists included."""
    man = Manifest(FAMILY)
    rows = []
    for k, v in man.done.items():
        try:
            d, m = parse_key(k)
        except ValueError:
            continue
        rows.append({"date": d.isoformat(), "market": m, "rows": int(v.get("rows", 0))})
    return pd.DataFrame(rows, columns=["date", "market", "rows"]).sort_values(["date", "market"]).reset_index(drop=True)


def consolidate_all() -> dict:
    """``regsho.parquet`` (every row) + ``regsho_calendar.parquet`` (every list
    fetched, empty ones included) — what `deep_features.MarketTables` reads."""
    from src.data.deep import consolidate
    n = consolidate(FAMILY)
    cal = calendar()
    m = write_parquet(cal, _deep.DEEP_DIR / "regsho_calendar.parquet")
    logger.info(f"[deep.regsho] calendar: {m:,} lists over {cal['date'].nunique() if len(cal) else 0} days")
    return {"rows": n, "lists": m}


def run(since: date = START, workers: int = 3, budget_seconds: float = 0.0,
        now: Optional[datetime] = None, markets: Sequence[str] = MARKETS) -> dict:
    """Fetch every completed trading day's lists not yet held (the manifest
    resumes), then consolidate. Days already done are never re-fetched: a
    published list does not change. ``markets`` splits a long backfill into one
    process per host (each host has its own pace)."""
    from src.data.deep import run_keys
    days = completed_days(since, now)
    r = run_keys(FAMILY, keys_for(days, markets), fetch, workers=workers, budget_seconds=budget_seconds,
                 retry_failed=True, empty_is_done=True)
    r["consolidated"] = consolidate_all()
    return r
