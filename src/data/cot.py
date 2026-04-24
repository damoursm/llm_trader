"""
CFTC Commitment of Traders (COT) — weekly futures positioning data.

Two reports are downloaded (no API key required — all public):
  Disaggregated Futures Only — physical commodities (gold, silver, oil, gas, copper…)
    Speculator proxy: Managed Money (hedge funds)
    URL: https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip

  Traders in Financial Futures (TFF) — equity index futures (S&P 500, Nasdaq)
    Speculator proxy: Leveraged Money (hedge funds)
    URL: https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip

Signal logic:
  - Net speculator position = managed/leveraged longs − shorts
  - Net % of open interest = net / OI × 100
  - Percentile of current net% within the last 52 weekly observations
  - Contrarian at extremes (≥80th pct = EXTREME_LONG → bearish; ≤20th = EXTREME_SHORT → bullish)
  - Momentum in the middle (60-79th → bullish trend; 20-39th → bearish trend)

Results are cached by ISO week number — re-runs within the same week skip the download.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from src.models import COTSignal, COTContext

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("cache")
_TIMEOUT   = 30

_DISAGG_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
_TFF_URL    = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"

# (CFTC market name substring, related ETF tickers, friendly label)
_DISAGG_CONTRACTS: List[Tuple[str, List[str], str]] = [
    ("GOLD",        ["GLD", "IAU", "GDX"],  "Gold"),
    ("SILVER",      ["SLV"],                "Silver"),
    ("CRUDE OIL",   ["USO"],                "Crude Oil"),
    ("NATURAL GAS", ["UNG"],                "Natural Gas"),
    ("COPPER",      ["CPER"],               "Copper"),
    ("PLATINUM",    ["PPLT"],               "Platinum"),
    ("PALLADIUM",   ["PALL"],               "Palladium"),
]

_TFF_CONTRACTS: List[Tuple[str, List[str], str]] = [
    ("S&P 500",  ["SPY"],  "S&P 500"),
    ("NASDAQ",   ["QQQ"],  "Nasdaq 100"),
]

# Disaggregated uses "Managed Money"; TFF uses "Leveraged Money"
_DISAGG_LONG  = "M_Money_Positions_Long_All"
_DISAGG_SHORT = "M_Money_Positions_Short_All"
_TFF_LONG     = "Lev_Money_Positions_Long_All"
_TFF_SHORT    = "Lev_Money_Positions_Short_All"
_OI_COL       = "Open_Interest_All"
_DATE_COL     = "As_of_Date_In_Form_YYMMDD"
_NAME_COL     = "Market_and_Exchange_Names"

# How many weekly observations to use for percentile computation
_HISTORY_WEEKS = 52


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(year: int, week: int) -> Path:
    return _CACHE_DIR / f"cot_{year}_{week:02d}.json"


def _load_cache(year: int, week: int) -> Optional[List[dict]]:
    path = _cache_path(year, week)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _save_cache(year: int, week: int, data: List[dict]) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    _cache_path(year, week).write_text(json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# Download and parse
# ---------------------------------------------------------------------------

def _download_zip(url: str) -> Optional[bytes]:
    """Download a ZIP file, return raw bytes or None on failure."""
    try:
        resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        logger.warning(f"[cot] Download failed {url}: {e}")
        return None


def _parse_csv_from_zip(zip_bytes: bytes) -> List[Dict[str, str]]:
    """Extract the first .txt file from a ZIP and parse it as CSV."""
    rows: List[Dict[str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txt_names:
                logger.warning("[cot] No .txt file found in ZIP")
                return rows
            # The annual file is the largest one
            fname = max(txt_names, key=lambda n: zf.getinfo(n).file_size)
            raw = zf.read(fname).decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(raw))
            for row in reader:
                rows.append(dict(row))
    except Exception as e:
        logger.warning(f"[cot] ZIP parse error: {e}")
    return rows


def _parse_date(yymmdd: str) -> Optional[date]:
    """
    Parse CFTC date fields which come in YYMMDD (e.g. '260404' → 2026-04-04)
    or sometimes YYYY-MM-DD format.
    """
    s = yymmdd.strip()
    for fmt in ("%y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _safe_int(val: str) -> int:
    try:
        return int(val.replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def _extract_series(
    rows: List[Dict[str, str]],
    keyword: str,
    long_col: str,
    short_col: str,
) -> List[Tuple[date, float, float, float]]:
    """
    Filter rows by market name keyword and return sorted (date, net_pct, long, short) tuples.
    net_pct = (longs - shorts) / open_interest × 100
    Sorted oldest-first.
    """
    keyword_upper = keyword.upper()
    series: List[Tuple[date, float, float, float]] = []
    seen_dates: set = set()

    for row in rows:
        name = row.get(_NAME_COL, "").upper()
        if keyword_upper not in name:
            continue

        dt = _parse_date(row.get(_DATE_COL, ""))
        if dt is None or dt in seen_dates:
            continue
        seen_dates.add(dt)

        long_pos  = _safe_int(row.get(long_col,  "0"))
        short_pos = _safe_int(row.get(short_col, "0"))
        oi        = _safe_int(row.get(_OI_COL,   "0"))
        if oi <= 0:
            continue

        net_pct = (long_pos - short_pos) / oi * 100
        series.append((dt, net_pct, long_pos, short_pos))

    series.sort(key=lambda x: x[0])   # oldest first
    return series


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _classify(percentile: float, wow_change: float) -> Tuple[str, str]:
    """
    Return (signal_label, direction) based on 52-week percentile.
    Extremes are contrarian; middle is momentum.
    """
    if percentile >= 80:
        return "EXTREME_LONG", "BEARISH"    # specs max long → contrarian sell
    if percentile >= 60:
        return "BULLISH_TREND", "BULLISH"
    if percentile >= 40:
        return "NEUTRAL", "NEUTRAL"
    if percentile >= 20:
        return "BEARISH_TREND", "BEARISH"
    return "EXTREME_SHORT", "BULLISH"       # specs max short → contrarian buy


def _build_signal(
    keyword: str,
    tickers: List[str],
    label: str,
    series: List[Tuple[date, float, float, float]],
) -> Optional[COTSignal]:
    """Build a COTSignal from a historical series of (date, net_pct, long, short) tuples."""
    if not series:
        return None

    window = series[-_HISTORY_WEEKS:]   # last 52 weeks (or whatever is available)
    current_date, current_net, current_long, current_short = window[-1]

    net_values  = [x[1] for x in window]
    min_52w     = min(net_values)
    max_52w     = max(net_values)
    spread      = max_52w - min_52w

    if spread < 0.5:                    # near-zero range → uninformative
        percentile = 50.0
    else:
        percentile = (current_net - min_52w) / spread * 100

    # Week-over-week change (use second-to-last if available)
    wow = current_net - window[-2][1] if len(window) >= 2 else 0.0

    signal, direction = _classify(percentile, wow)

    # Human-readable summary
    trend_arrow = "▲" if wow > 0.5 else ("▼" if wow < -0.5 else "→")
    pct_label   = f"{percentile:.0f}th percentile of {len(window)}-week range"
    summary = (
        f"Net speculator position: {current_net:+.1f}% of OI "
        f"({trend_arrow} {wow:+.1f}% WoW). "
        f"{pct_label}. Signal: {signal}."
    )

    return COTSignal(
        contract          = label,
        tickers           = tickers,
        report_date       = current_date,
        net_speculator_pct= round(current_net, 2),
        net_change_wow    = round(wow, 2),
        percentile_52w    = round(percentile, 1),
        signal            = signal,
        direction         = direction,
        summary           = summary,
    )


# ---------------------------------------------------------------------------
# Download strategy: current year + previous year for full 52w history
# ---------------------------------------------------------------------------

def _fetch_rows(url_template: str, year: int) -> List[Dict[str, str]]:
    """Download and parse COT rows for a given year."""
    url   = url_template.format(year=year)
    data  = _download_zip(url)
    if not data:
        return []
    return _parse_csv_from_zip(data)


def _get_all_rows(url_template: str) -> List[Dict[str, str]]:
    """
    Download the current year's COT file. Also fetch the previous year if
    the current year has fewer than _HISTORY_WEEKS observations for any contract,
    which happens early in the calendar year.
    """
    today      = date.today()
    curr_rows  = _fetch_rows(url_template, today.year)
    # Quick heuristic: if fewer than HISTORY_WEEKS total rows, also fetch last year
    if len(curr_rows) < _HISTORY_WEEKS * 3:   # 3 contracts minimum × weeks
        prev_rows = _fetch_rows(url_template, today.year - 1)
        return prev_rows + curr_rows           # chronological order
    return curr_rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_cot_context() -> Optional[COTContext]:
    """
    Fetch CFTC COT data for tracked commodity and financial futures.
    Results are cached by ISO week — re-runs within the same week are instant.
    Returns None if all downloads fail.
    """
    today     = date.today()
    iso_year, iso_week, _ = today.isocalendar()

    # Check cache
    cached = _load_cache(iso_year, iso_week)
    if cached is not None:
        logger.info(f"[cot] Using cached COT data (week {iso_year}-W{iso_week:02d})")
        try:
            signals = [COTSignal(**s) for s in cached]
            return _build_context(signals, today)
        except Exception as e:
            logger.warning(f"[cot] Cache parse error: {e}")

    logger.info("[cot] Downloading CFTC COT reports...")
    signals: List[COTSignal] = []

    # 1. Disaggregated report — physical commodities
    disagg_rows = _get_all_rows(_DISAGG_URL)
    if disagg_rows:
        for keyword, tickers, label in _DISAGG_CONTRACTS:
            series = _extract_series(disagg_rows, keyword, _DISAGG_LONG, _DISAGG_SHORT)
            sig    = _build_signal(keyword, tickers, label, series)
            if sig:
                signals.append(sig)
                logger.info(
                    f"[cot] {label}: {sig.signal} ({sig.percentile_52w:.0f}th pct) "
                    f"net={sig.net_speculator_pct:+.1f}% WoW={sig.net_change_wow:+.1f}%"
                )
    else:
        logger.warning("[cot] Disaggregated report unavailable — skipping commodities")

    # 2. TFF report — financial futures (S&P 500, Nasdaq)
    tff_rows = _get_all_rows(_TFF_URL)
    if tff_rows:
        for keyword, tickers, label in _TFF_CONTRACTS:
            series = _extract_series(tff_rows, keyword, _TFF_LONG, _TFF_SHORT)
            sig    = _build_signal(keyword, tickers, label, series)
            if sig:
                signals.append(sig)
                logger.info(
                    f"[cot] {label}: {sig.signal} ({sig.percentile_52w:.0f}th pct) "
                    f"net={sig.net_speculator_pct:+.1f}% WoW={sig.net_change_wow:+.1f}%"
                )
    else:
        logger.warning("[cot] TFF report unavailable — skipping financial futures")

    if not signals:
        return None

    # Save cache
    _save_cache(iso_year, iso_week, [s.model_dump() for s in signals])
    return _build_context(signals, today)


def _build_context(signals: List[COTSignal], report_date: date) -> COTContext:
    extremes = [s for s in signals if s.signal in ("EXTREME_LONG", "EXTREME_SHORT")]
    trends   = [s for s in signals if s.signal in ("BULLISH_TREND", "BEARISH_TREND")]

    parts = []
    if extremes:
        names = ", ".join(s.contract for s in extremes)
        parts.append(f"Extreme speculator positioning in: {names}.")
    if trends:
        bull = [s.contract for s in trends if s.direction == "BULLISH"]
        bear = [s.contract for s in trends if s.direction == "BEARISH"]
        if bull:
            parts.append(f"Specs adding longs in: {', '.join(bull)}.")
        if bear:
            parts.append(f"Specs reducing longs / adding shorts in: {', '.join(bear)}.")
    if not parts:
        parts.append("Speculator positioning broadly neutral — no extreme readings.")

    summary = " ".join(parts)
    return COTContext(signals=signals, report_date=report_date, summary=summary)
