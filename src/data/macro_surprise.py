"""
Macro Economic Surprise Index — CESI-style breadth of economic beats vs. misses.

Inspired by the Citigroup Economic Surprise Index: compares the most recent FRED data
release for each indicator to its own recent trend (trailing 3-month average). When data
consistently beats its own trend, economic momentum is accelerating → risk-on tailwind.
When data consistently misses, momentum is decelerating → defensive positioning.

Indicators monitored (all from FRED, free API key):
  PAYEMS    — Nonfarm Payroll Employment (MoM change, thousands)       weight=1.5
  INDPRO    — Industrial Production Index (MoM % change)               weight=1.0
  RSAFS     — Advance Retail Sales (MoM % change)                      weight=1.5
  UMCSENT   — University of Michigan Consumer Sentiment (level)         weight=1.0
  CPIAUCSL  — CPI All Urban (MoM % change; lower = positive surprise)  weight=1.5
  UNRATE    — Unemployment Rate (level; lower = positive surprise)      weight=1.5

Surprise computation per indicator:
  actual   = most recent reading (or MoM change)
  expected = average of prior 3 periods
  surprise = actual − expected
  z_score  = surprise / trailing_std (∈ [-3, +3])

For CPI and UNRATE the sign is flipped (lower value = positive market surprise).

Overall Surprise Score = weighted mean of sign-adjusted z-scores / 3  → ∈ [-1, +1]

Signal thresholds:
  > +0.40   STRONG_BEAT  → BULLISH  (economy accelerating well above trend)
  +0.15–0.40 MILD_BEAT   → BULLISH  (modest positive momentum)
  −0.15–0.15 NEUTRAL     → NEUTRAL  (in line with recent trend)
  −0.40– −0.15 MILD_MISS → BEARISH  (modest negative momentum)
  < −0.40   STRONG_MISS  → BEARISH  (economy decelerating well below trend)
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from config import settings
from src.models import MacroSurpriseContext, MacroSurpriseIndicator

_BASE    = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = 15
CACHE_DIR = Path("cache")

# ── Indicator registry ────────────────────────────────────────────────────────
# type: "mom_abs" | "mom_pct" | "level"
# bullish_up: True → higher value = positive surprise; False → lower = positive
_INDICATORS = [
    {"id": "PAYEMS",   "name": "Nonfarm Payrolls",      "type": "mom_abs",  "bullish_up": True,  "weight": 1.5, "unit": "k jobs"},
    {"id": "INDPRO",   "name": "Industrial Production", "type": "mom_pct",  "bullish_up": True,  "weight": 1.0, "unit": "% MoM"},
    {"id": "RSAFS",    "name": "Retail Sales",          "type": "mom_pct",  "bullish_up": True,  "weight": 1.5, "unit": "% MoM"},
    {"id": "UMCSENT",  "name": "Consumer Sentiment",    "type": "level",    "bullish_up": True,  "weight": 1.0, "unit": "index"},
    {"id": "CPIAUCSL", "name": "CPI Inflation",         "type": "mom_pct",  "bullish_up": False, "weight": 1.5, "unit": "% MoM"},
    {"id": "UNRATE",   "name": "Unemployment",          "type": "level",    "bullish_up": False, "weight": 1.5, "unit": "%"},
]

_N_PRIOR     = 3    # trailing periods used as "expectation" baseline
_N_FETCH     = 9    # observations fetched (need _N_PRIOR + 1 + 5 for std)
_SCORE_SCALE = 3.0  # divisor to map from z-score range to [-1, +1]

_SIG_STRONG_BEAT = 0.40
_SIG_MILD_BEAT   = 0.15
_SIG_MILD_MISS   = -0.15
_SIG_STRONG_MISS = -0.40


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    return CACHE_DIR / f"macro_surprise_{date.today().isoformat()}.json"


def _load_cache() -> Optional[MacroSurpriseContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        ctx = MacroSurpriseContext.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        logger.info(
            f"[macro_surprise] Loaded from cache — score={ctx.score:+.2f} "
            f"({ctx.signal}) {ctx.beats}B/{ctx.in_line}N/{ctx.misses}M"
        )
        return ctx
    except Exception as e:
        logger.warning(f"[macro_surprise] Cache load failed: {e}")
        return None


def _save_cache(ctx: MacroSurpriseContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[macro_surprise] Cache save failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FRED fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(series_id: str, api_key: str) -> list[dict]:
    try:
        resp = httpx.get(
            _BASE,
            params={
                "series_id":  series_id,
                "api_key":    api_key,
                "file_type":  "json",
                "sort_order": "desc",
                "limit":      _N_FETCH,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception as e:
        logger.warning(f"[macro_surprise] FRED {series_id} fetch failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Surprise arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def _raw_values(obs: list[dict]) -> list[float]:
    out = []
    for o in obs:
        v = o.get("value", ".")
        if v != ".":
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _to_metric(values: list[float], metric_type: str) -> Optional[list[float]]:
    """Convert raw levels to the metric we use for surprise (newest-first)."""
    if not values:
        return None
    if metric_type == "level":
        return values
    if metric_type == "mom_abs":
        if len(values) < 2:
            return None
        return [values[i] - values[i + 1] for i in range(len(values) - 1)]
    if metric_type == "mom_pct":
        if len(values) < 2:
            return None
        result = []
        for i in range(len(values) - 1):
            denom = abs(values[i + 1])
            if denom > 0.001:
                result.append((values[i] - values[i + 1]) / denom * 100)
        return result if result else None
    return None


def _compute_surprise(
    metrics: list[float],
) -> Optional[tuple[float, float, float, float]]:
    """
    Given a newest-first metric series, compute:
      (actual, expected, surprise, z_score)

    Returns None if there is insufficient data.
    """
    if len(metrics) < _N_PRIOR + 1:
        return None

    actual   = metrics[0]
    prior    = metrics[1 : 1 + _N_PRIOR]
    expected = sum(prior) / len(prior)
    surprise = actual - expected

    # Trailing std using all available prior data (at least 2 points)
    all_prior = metrics[1:]
    if len(all_prior) >= 2:
        mean_p  = sum(all_prior) / len(all_prior)
        std     = math.sqrt(sum((x - mean_p) ** 2 for x in all_prior) / len(all_prior))
    else:
        std = max(abs(expected) * 0.10, 0.10)

    z = surprise / std if std > 0.001 else 0.0
    z = max(-3.0, min(3.0, z))
    return actual, expected, surprise, z


def _classify_indicator(z_signed: float) -> str:
    if z_signed >  0.5:  return "BEAT"
    if z_signed < -0.5:  return "MISS"
    return "IN_LINE"


def _classify_overall(score: float) -> tuple[str, str]:
    if score >=  _SIG_STRONG_BEAT: return "STRONG_BEAT",  "BULLISH"
    if score >=  _SIG_MILD_BEAT:   return "MILD_BEAT",    "BULLISH"
    if score >   _SIG_MILD_MISS:   return "NEUTRAL",      "NEUTRAL"
    if score >   _SIG_STRONG_MISS: return "MILD_MISS",    "BEARISH"
    return                                "STRONG_MISS",  "BEARISH"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_macro_surprise_context() -> Optional[MacroSurpriseContext]:
    """
    Fetch 6 FRED series and compute per-indicator and composite surprise scores.
    Returns MacroSurpriseContext or None if FRED key is missing or all fetches fail.
    Cached daily.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    api_key = settings.fred_api_key
    if not api_key:
        logger.debug("[macro_surprise] FRED_API_KEY not set — skipping")
        return None

    logger.info("[macro_surprise] Fetching FRED series for surprise computation...")

    # Fetch all series concurrently (httpx is synchronous here; pipeline thread pool handles parallelism)
    raw: dict[str, list[dict]] = {}
    for ind in _INDICATORS:
        raw[ind["id"]] = _fetch(ind["id"], api_key)

    if not any(raw.values()):
        logger.warning("[macro_surprise] All FRED fetches returned empty")
        return None

    # ── Compute per-indicator surprises ───────────────────────────────────────
    components: list[MacroSurpriseIndicator] = []
    weighted_z_sum = 0.0
    weight_sum     = 0.0

    for ind in _INDICATORS:
        obs = raw.get(ind["id"], [])
        if not obs:
            continue

        values  = _raw_values(obs)
        metrics = _to_metric(values, ind["type"])
        if not metrics:
            continue

        result = _compute_surprise(metrics)
        if result is None:
            continue

        actual, expected, surprise, z = result

        # Sign-adjust: if lower = bullish, flip z so positive z = positive surprise
        z_signed = z if ind["bullish_up"] else -z

        # Release date of most recent non-missing observation
        release_date = next(
            (o["date"] for o in obs if o.get("value", ".") != "."), "N/A"
        )

        indicator = MacroSurpriseIndicator(
            series_id=ind["id"],
            name=ind["name"],
            unit=ind["unit"],
            actual=round(actual, 3),
            expected=round(expected, 3),
            surprise=round(surprise, 3),
            z_score=round(z_signed, 2),
            signal=_classify_indicator(z_signed),
            release_date=release_date,
        )
        components.append(indicator)

        # Weighted composite
        w = ind["weight"]
        weighted_z_sum += w * z_signed
        weight_sum     += w

    if not components or weight_sum == 0:
        logger.warning("[macro_surprise] No indicators computed — skipping")
        return None

    raw_score = weighted_z_sum / weight_sum
    score     = round(max(-1.0, min(1.0, raw_score / _SCORE_SCALE)), 3)
    signal, direction = _classify_overall(score)

    beats   = sum(1 for c in components if c.signal == "BEAT")
    misses  = sum(1 for c in components if c.signal == "MISS")
    in_line = sum(1 for c in components if c.signal == "IN_LINE")

    # ── Summary ───────────────────────────────────────────────────────────────
    signal_desc = {
        "STRONG_BEAT":  "Economic data significantly ahead of recent trend — cyclical tailwind for risk assets.",
        "MILD_BEAT":    "Economic data modestly above trend — mild positive momentum.",
        "NEUTRAL":      "Data broadly in line with recent trend — no directional macro surprise.",
        "MILD_MISS":    "Data modestly below recent trend — mild headwind; consider defensives.",
        "STRONG_MISS":  "Data significantly below trend — decelerating economy; shift toward defensives.",
    }.get(signal, "")

    beat_names = [c.name for c in components if c.signal == "BEAT"]
    miss_names = [c.name for c in components if c.signal == "MISS"]

    beat_str = f" Beats: {', '.join(beat_names)}." if beat_names else ""
    miss_str = f" Misses: {', '.join(miss_names)}." if miss_names else ""

    summary = (
        f"Macro Surprise Score: {score:+.2f} ({signal}). "
        f"{beats} beats, {in_line} in-line, {misses} misses across {len(components)} indicators. "
        f"{signal_desc}{beat_str}{miss_str}"
    )

    ctx = MacroSurpriseContext(
        score=score,
        signal=signal,
        direction=direction,
        indicators=components,
        beats=beats,
        misses=misses,
        in_line=in_line,
        report_date=date.today(),
        summary=summary,
    )
    _save_cache(ctx)

    logger.info(
        f"[macro_surprise] Score={score:+.2f} ({signal}) | "
        f"{beats}B / {in_line}N / {misses}M | {direction}"
    )
    return ctx
