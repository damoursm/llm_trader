"""Macro → Discovery loop — pull top holdings of the favored sector/factor ETFs.

The pipeline already identifies *favored regimes/sectors* but discovery ignored them.
This closes the loop: it reads the macro modules and auto-pulls the **top holdings of the
favored ETFs** into the analysis universe, biasing stock selection toward where macro
money is flowing.

Favored ETFs are drawn from three sources already computed each run:
  • Sector rotation  — ``top_inflow`` SPDR sectors (real-time cross-sector money flow).
  • Business cycle   — ``top_cycle_leaders`` (structural phase leadership).
  • DIX regime       — a factor tilt: bullish (accumulation) → MTUM (momentum);
                       bearish (distribution) → USMV (low-vol/defensive).

Top holdings are fetched from yfinance ``funds_data`` (cached daily in
``cache/etf_holdings_YYYY-MM-DD.json``) with a static SPDR fallback when the live
lookup is unavailable. The constituents are injected into the universe and then receive
the full signal stack — so the macro view actually drives which single names get analysed.

Fail-graceful: any error yields an empty context; discovery never blocks the run.
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from config import settings
from src.models import MacroDiscoveryContext

_CACHE_DIR = Path("cache")

# Static fallback: top holdings of the 11 SPDR sector ETFs (cap-weighted, stable for
# months). Used only when the live yfinance holdings lookup fails. Factor ETFs are
# omitted (their holdings rotate; rely on the live lookup for those).
_STATIC_HOLDINGS: Dict[str, List[str]] = {
    "XLK":  ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "CRM", "CSCO", "AMD", "ACN", "ADBE"],
    "XLF":  ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "AXP", "SPGI", "MS"],
    "XLE":  ["XOM", "CVX", "COP", "WMB", "EOG", "OKE", "KMI", "SLB", "PSX", "MPC"],
    "XLV":  ["LLY", "JNJ", "ABBV", "UNH", "MRK", "TMO", "ABT", "ISRG", "AMGN", "DHR"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "BKNG", "LOW", "TJX", "NKE", "SBUX", "ORLY"],
    "XLP":  ["COST", "WMT", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB"],
    "XLI":  ["GE", "CAT", "RTX", "HON", "UNP", "ETN", "BA", "DE", "LMT", "ADP"],
    "XLB":  ["LIN", "SHW", "ECL", "FCX", "APD", "NEM", "CTVA", "DOW", "MLM", "NUE"],
    "XLU":  ["NEE", "SO", "DUK", "CEG", "AEP", "D", "EXC", "SRE", "XEL", "PEG"],
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "DLR", "O", "CCI", "CBRE"],
    "XLC":  ["META", "GOOGL", "GOOG", "NFLX", "DIS", "TMUS", "T", "VZ", "CMCSA", "CHTR"],
}


# ─────────────────────────────────────────────────────────────────────────────
# ETF holdings (daily-cached)
# ─────────────────────────────────────────────────────────────────────────────

def _holdings_cache_path(today: date) -> Path:
    return _CACHE_DIR / f"etf_holdings_{today.isoformat()}.json"


def _load_holdings_cache(today: date) -> Dict[str, List[str]]:
    p = _holdings_cache_path(today)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_holdings_cache(today: date, cache: Dict[str, List[str]]) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    try:
        _holdings_cache_path(today).write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as e:
        logger.debug(f"[macro_disc] holdings cache save failed: {e}")


def _clean_symbol(sym: str) -> str:
    return (sym or "").strip().upper()


def _fetch_holdings_yf(etf: str) -> List[str]:
    """Live top-holdings via yfinance funds_data. Returns [] on any failure."""
    try:
        import yfinance as yf
        th = yf.Ticker(etf).funds_data.top_holdings
        if th is None or th.empty:
            return []
        out = []
        for sym in list(th.index):
            s = _clean_symbol(str(sym))
            # Keep plain equity tickers; skip cash/derivative placeholders
            if s and 1 <= len(s) <= 6 and all(c.isalnum() or c in ".-" for c in s):
                out.append(s)
        return out
    except Exception as e:
        logger.debug(f"[macro_disc] yfinance holdings failed for {etf}: {e}")
        return []


def _get_etf_holdings(etf: str, top_n: int, cache: Dict[str, List[str]]) -> List[str]:
    """Top holdings for an ETF — daily cache → yfinance → static SPDR fallback."""
    etf = _clean_symbol(etf)
    if etf in cache:
        return cache[etf][:top_n]
    holds = _fetch_holdings_yf(etf) or _STATIC_HOLDINGS.get(etf, [])
    cache[etf] = holds
    return holds[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Favored-ETF selection
# ─────────────────────────────────────────────────────────────────────────────

def _dix_factor_etf(dix_context) -> Optional[tuple]:
    """Map the DIX regime to a favored factor ETF. Returns (etf, reason) or None."""
    if dix_context is None or getattr(dix_context, "direction", "NEUTRAL") == "NEUTRAL":
        return None
    d = dix_context.direction
    sig = getattr(dix_context, "signal", "")
    if d == "BULLISH":
        return ("MTUM", f"DIX {sig} → momentum tilt")
    if d == "BEARISH":
        return ("USMV", f"DIX {sig} → low-vol/defensive tilt")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_macro_discovery(
    sector_rotation_context=None,
    business_cycle_context=None,
    dix_context=None,
    today: Optional[date] = None,
) -> MacroDiscoveryContext:
    """Pull top holdings of the favored sector/factor ETFs. Fail-graceful."""
    today = today or date.today()
    empty = MacroDiscoveryContext(report_date=today, summary="Macro discovery produced no candidates.")
    if not settings.enable_macro_discovery:
        return MacroDiscoveryContext(report_date=today, summary="Macro discovery disabled.")

    try:
        n_top = max(1, settings.macro_discovery_top_sectors)
        favored: List[str] = []
        reasons: Dict[str, str] = {}

        def _add(etf: str, reason: str):
            etf = _clean_symbol(etf)
            if not etf:
                return
            if etf not in reasons:
                favored.append(etf)
                reasons[etf] = reason

        # 1. Sector-rotation inflows (real-time money flow)
        if sector_rotation_context is not None:
            for etf in (getattr(sector_rotation_context, "top_inflow", []) or [])[:n_top]:
                _add(etf, "sector inflow")

        # 2. Business-cycle phase leaders (structural)
        if business_cycle_context is not None:
            phase = getattr(business_cycle_context, "cycle_phase", "")
            for etf in (getattr(business_cycle_context, "top_cycle_leaders", []) or [])[:n_top]:
                _add(etf, f"{phase} leader" if phase else "cycle leader")

        # 3. DIX regime → factor tilt
        factor = _dix_factor_etf(dix_context)
        if factor:
            _add(factor[0], factor[1])

        if not favored:
            logger.info("[macro_disc] No favored ETFs identified — skipping")
            return empty

        # Pull top holdings of each favored ETF
        cache = _load_holdings_cache(today)
        by_etf: Dict[str, List[str]] = {}
        ordered: List[str] = []
        for etf in favored:
            holds = _get_etf_holdings(etf, settings.macro_discovery_holdings_per_etf, cache)
            holds = [h for h in holds if h and h != etf]
            if holds:
                by_etf[etf] = holds
                ordered.extend(holds)
        _save_holdings_cache(today, cache)

        # Dedup preserving order, cap
        seen = set()
        tickers: List[str] = []
        for t in ordered:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
        tickers = tickers[: settings.macro_discovery_max]

        if not tickers:
            return empty

        fav_str = ", ".join(f"{e}({reasons[e]})" for e in favored if e in by_etf)
        summary = (
            f"{len(tickers)} candidate(s) from {len(by_etf)} favored ETF(s): {fav_str}"
        )
        logger.info(f"[macro_disc] {summary}")
        return MacroDiscoveryContext(
            favored_etfs=[e for e in favored if e in by_etf],
            etf_reasons={e: reasons[e] for e in favored if e in by_etf},
            by_etf=by_etf,
            tickers=tickers,
            report_date=today,
            summary=summary,
        )
    except Exception as e:
        logger.warning(f"[macro_disc] failed: {e}")
        return empty
