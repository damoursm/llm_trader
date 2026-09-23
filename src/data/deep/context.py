"""Market-context families — one table each, whole history in one fetch, no
manifest (idempotent overwrite).

* ``market_daily``   — yfinance daily OHLCV for volatility indices, rates,
  dollar, commodities, broad and sector ETFs (VIX from 1990, MOVE 2002, …).
* ``fred_vintages``  — ALFRED: every vintage of every observation for a macro
  series set (``realtime_start`` = the day that value became public, so a
  feature can use the FIRST print, not today's revised number).
* ``fama_french``    — Ken French daily Mkt-RF / SMB / HML / RF / Mom.
* ``cboe_putcall``   — CBOE total put/call ratio history (the CSV the live
  put_call feed reads once for its last row).
* ``dix``            — SqueezeMetrics DIX / GEX history (same: full CSV, last row used live).
* ``cot_tff``        — CFTC Traders in Financial Futures, one part per year.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.data.deep import GENERIC_HEADERS, RateLimiter, http_get

MARKET_SYMBOLS: List[str] = [
    "^VIX", "^VIX9D", "^VIX3M", "^VIX6M", "^VVIX", "^VXN", "^SKEW", "^MOVE", "^TICK",
    "^TNX", "^IRX", "^FVX", "^TYX", "^GSPC", "^NDX", "^RUT", "^DJI",
    "DX-Y.NYB", "GC=F", "CL=F", "HG=F", "SI=F", "NG=F", "ZN=F", "ES=F", "BTC-USD",
    "SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "SHY", "HYG", "LQD", "JNK",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
    "GLD", "SLV", "USO", "UUP", "EEM", "EFA", "VNQ",
]

FRED_SERIES: List[str] = [
    "DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M", "DFF", "EFFR", "SOFR",
    "DFEDTARU", "DFEDTARL", "DTB3", "DTB6", "DTB1YR", "DPRIME", "MORTGAGE30US",
    "BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLH0A0HYM2EY", "NFCI", "ANFCI", "STLFSI4",
    "VIXCLS", "DTWEXBGS", "DEXUSEU", "DEXJPUS", "DCOILWTICO", "T5YIE", "T10YIE", "DFII10",
    "UNRATE", "PAYEMS", "ICSA", "CCSA", "CPIAUCSL", "CPILFESL", "PCEPILFE", "INDPRO",
    "RSAFS", "UMCSENT", "HOUST", "PERMIT", "M2SL", "WALCL", "RRPONTSYD", "TOTRESNS",
    "WTREGEN", "USREC", "GDPC1", "A191RL1Q225SBEA",
]

_FRED_LIMITER = RateLimiter(0.6)     # FRED allows ~120 req/min


# ── yfinance daily series ────────────────────────────────────────────────────

def market_daily(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    import yfinance as yf
    symbols = list(symbols or MARKET_SYMBOLS)
    raw = yf.download(symbols, period="max", auto_adjust=False, group_by="ticker",
                      threads=True, progress=False)
    frames = []
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym not in raw.columns.get_level_values(0):
                continue
            d = raw[sym].dropna(how="all").copy()
            if d.empty:
                continue
            d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
            d.index.name = "date"
            d = d.reset_index()
            d.insert(0, "symbol", sym)
            frames.append(d)
    else:
        d = raw.dropna(how="all").copy()
        d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
        d.index.name = "date"
        d = d.reset_index()
        d.insert(0, "symbol", symbols[0])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None) if getattr(df["date"].dt, "tz", None) is not None else pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.date.astype(str)
    for c in df.columns:
        if c not in ("symbol", "date"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── FRED / ALFRED vintages ───────────────────────────────────────────────────

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fred_window(series_id: str, api_key: str, rt_start: str, rt_end: str):
    """All observations of a series over one real-time window, paginated.
    Returns (rows, status) — status 400 with 'vintage dates' means the window
    holds more than ALFRED's 2,000-vintage cap and must be split."""
    rows: List[dict] = []
    offset = 0
    while True:
        r = http_get(_FRED_URL, params={"series_id": series_id, "api_key": api_key,
                                        "file_type": "json", "realtime_start": rt_start,
                                        "realtime_end": rt_end, "limit": 100000, "offset": offset},
                     headers=GENERIC_HEADERS, timeout=120, limiter=_FRED_LIMITER)
        if r is None:
            return rows, None
        if r.status_code != 200:
            msg = ""
            try:
                msg = str((r.json() or {}).get("error_message", ""))
            except Exception:                            # noqa: BLE001
                pass
            return rows, (r.status_code, msg)
        j = r.json()
        obs = j.get("observations") or []
        rows.extend(obs)
        offset += len(obs)
        if not obs or offset >= int(j.get("count") or 0):
            return rows, 200


def fred_series_vintages(series_id: str, api_key: str) -> pd.DataFrame:
    """Every vintage of every observation. ALFRED caps a request at 2,000
    vintage DATES, which daily series exceed, so those are fetched in yearly
    real-time windows. Inside a window ALFRED clamps ``realtime_start`` to the
    window's start for values first published earlier, so the windows are
    merged by (date, value) keeping the EARLIEST realtime_start — the true
    first-print date lands in exactly one window, unclamped. A series with no
    ALFRED history at all (DFF, T10Y3M, …: derived or never-revised daily
    series) is fetched plain and stamped ``realtime_start = date`` with
    ``vintage = False`` — the feature builder lags those by a day."""
    from datetime import date as _date
    rows, st = _fred_window(series_id, api_key, "1776-07-04", "9999-12-31")
    vintage = True
    if isinstance(st, tuple) and st[0] == 400 and "vintage dates" in st[1]:
        rows = []
        y0, y1 = 1990, _date.today().year
        for y in range(y0, y1 + 1):
            end = "9999-12-31" if y == y1 else f"{y}-12-31"
            start = "1776-07-04" if y == y0 else f"{y}-01-01"
            part, st2 = _fred_window(series_id, api_key, start, end)
            if st2 != 200:
                logger.warning(f"[deep.context] FRED {series_id} window {y}: HTTP {st2}")
                continue
            rows.extend(part)
    elif isinstance(st, tuple) and st[0] == 400 and "not exist in ALFRED" in st[1]:
        vintage = False
        plain = _fred_plain(series_id, api_key)
        rows = [{"date": d, "value": v, "realtime_start": d, "realtime_end": "9999-12-31"}
                for d, v in zip(plain["date"], plain["value"])]
    elif st != 200:
        logger.warning(f"[deep.context] FRED {series_id}: HTTP {st}")
    if not rows:
        return pd.DataFrame()
    df = merge_vintages(pd.DataFrame(rows))
    df["vintage"] = bool(vintage)
    # ALFRED's first vintage of a series can start long after the series does
    # (BAMLH0A0HYM2 vintages begin 2023-09 while FRED serves it from 1996), so
    # observations older than the first vintage are backfilled from plain FRED
    # and flagged vintage=False, realtime_start = the observation date.
    if vintage and len(df):
        first = str(df["date"].min())
        plain = _fred_plain(series_id, api_key, observation_end=first)
        plain = plain[plain["date"] < first]
        if len(plain):
            plain = plain.assign(realtime_start=plain["date"], realtime_end="9999-12-31", vintage=False)
            df = pd.concat([plain[df.columns], df], ignore_index=True)
    df.insert(0, "series_id", series_id)
    return df[["series_id", "date", "realtime_start", "realtime_end", "value", "vintage"]].sort_values(
        ["date", "realtime_start"]).reset_index(drop=True)


def fred_series_update(series_id: str, api_key: str, since: str) -> pd.DataFrame:
    """The vintages of a series valid anywhere in the real-time window
    ``[since, today]`` — the nightly refresh's unit, which
    ``refresh.merge_vintage_update`` folds into the stored table (that merge
    is what discards the window's clamped ``realtime_start`` on values first
    printed before ``since``). A window that still exceeds ALFRED's
    2,000-vintage cap (a very long gap) falls back to the full fetch; a series
    with no ALFRED history returns its plain observations dated >= ``since``
    stamped ``vintage = False``."""
    rows, st = _fred_window(series_id, api_key, since, "9999-12-31")
    vintage = True
    if isinstance(st, tuple) and st[0] == 400 and "vintage dates" in st[1]:
        return fred_series_vintages(series_id, api_key)
    if isinstance(st, tuple) and st[0] == 400 and "not exist in ALFRED" in st[1]:
        vintage = False
        plain = _fred_plain(series_id, api_key)
        plain = plain[plain["date"] >= since]
        rows = [{"date": d, "value": v, "realtime_start": d, "realtime_end": "9999-12-31"}
                for d, v in zip(plain["date"], plain["value"])]
    elif st != 200:
        logger.warning(f"[deep.context] FRED {series_id} update window {since}: HTTP {st}")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = merge_vintages(pd.DataFrame(rows))
    df["vintage"] = bool(vintage)
    df.insert(0, "series_id", series_id)
    return df[["series_id", "date", "realtime_start", "realtime_end", "value", "vintage"]]


def _fred_plain(series_id: str, api_key: str, observation_end: Optional[str] = None) -> pd.DataFrame:
    """Today's FRED view of a series (no vintages), optionally up to a date."""
    rows: List[dict] = []
    offset = 0
    while True:
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
                  "limit": 100000, "offset": offset}
        if observation_end:
            params["observation_end"] = observation_end
        r = http_get(_FRED_URL, params=params, headers=GENERIC_HEADERS, timeout=120, limiter=_FRED_LIMITER)
        if r is None or r.status_code != 200:
            break
        j = r.json()
        obs = j.get("observations") or []
        rows.extend(obs)
        offset += len(obs)
        if not obs or offset >= int(j.get("count") or 0):
            break
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    return df


def merge_vintages(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse window-clamped duplicates: one row per (date, value) with the
    earliest realtime_start and the latest realtime_end. Pure."""
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    out = (df.groupby(["date", "value"], dropna=False, as_index=False)
             .agg(realtime_start=("realtime_start", "min"), realtime_end=("realtime_end", "max")))
    return out.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def fred_vintages(series: Optional[List[str]] = None) -> pd.DataFrame:
    from config.settings import settings
    key = settings.fred_api_key
    if not key:
        logger.warning("[deep.context] no FRED key configured")
        return pd.DataFrame()
    frames = []
    for sid in series or FRED_SERIES:
        d = fred_series_vintages(sid, key)
        if len(d):
            frames.append(d)
            logger.info(f"[deep.context] FRED {sid}: {len(d):,} vintage rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Ken French factors ───────────────────────────────────────────────────────

_FF_DAILY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
_FF_MOM = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def parse_ff_csv(text: str, names: List[str]) -> pd.DataFrame:
    """Rows 'YYYYMMDD,v1,v2,...' from a Ken French CSV, header block skipped,
    stopping at the first blank line after the data starts."""
    rows = []
    started = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\d{8}\s*,", s):
            started = True
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= len(names) + 1:
                rows.append(parts[: len(names) + 1])
        elif started and not s:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date"] + names)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date.astype(str)
    for n in names:
        df[n] = pd.to_numeric(df[n], errors="coerce") / 100.0     # percent -> fraction
    return df


def _zip_first_text(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = z.namelist()[0]
        return z.read(name).decode("utf-8", "replace")


def fama_french() -> pd.DataFrame:
    r1 = http_get(_FF_DAILY, headers=GENERIC_HEADERS, timeout=120)
    r2 = http_get(_FF_MOM, headers=GENERIC_HEADERS, timeout=120)
    if r1 is None or r1.status_code != 200:
        return pd.DataFrame()
    ff = parse_ff_csv(_zip_first_text(r1.content), ["mkt_rf", "smb", "hml", "rf"])
    if r2 is not None and r2.status_code == 200:
        mom = parse_ff_csv(_zip_first_text(r2.content), ["mom"])
        if len(mom):
            ff = ff.merge(mom, on="date", how="left")
    return ff


# ── CBOE put/call, DIX ───────────────────────────────────────────────────────

def cboe_putcall() -> pd.DataFrame:
    """The legacy CBOE datahouse CSV; the URL serves the site's HTML shell as of
    2026-09-19, so this returns empty until CBOE's new data path is wired."""
    url = "https://www.cboe.com/publish/scheduledtask/mktdata/datahouse/putcallratio.csv"
    r = http_get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    if r is None or r.status_code != 200:
        return pd.DataFrame()
    lines = r.text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.upper().startswith("DATE")), None)
    if start is None:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    df = df.rename(columns={"p_c_ratio": "pc_ratio"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["date"] != "NaT"].reset_index(drop=True)


def dix() -> pd.DataFrame:
    r = http_get("https://squeezemetrics.com/monitor/static/DIX.csv",
                 headers={"User-Agent": "llm-trader/1.0"}, timeout=60)
    if r is None or r.status_code != 200:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── CFTC Traders in Financial Futures ────────────────────────────────────────

_TFF_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"


def cot_tff_year(year: str) -> pd.DataFrame:
    r = http_get(_TFF_URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"HTTP {getattr(r, 'status_code', None)}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".txt", ".csv")))
        df = pd.read_csv(io.BytesIO(z.read(name)), dtype=str, encoding="utf-8",
                         encoding_errors="replace", on_bad_lines="skip", low_memory=False)
    df.columns = [re.sub(r"[^A-Za-z0-9]+", "_", str(c).strip()).strip("_") for c in df.columns]
    date_col = next((c for c in df.columns if c.lower().startswith("report_date_as")), None)
    if date_col:
        df["report_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
    for c in df.columns:
        if c not in ("report_date",) and df[c].str.match(r"^-?\d+(\.\d+)?$", na=False).mean() > 0.9:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["year"] = str(year)
    return df
