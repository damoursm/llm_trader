"""
Estimate Revision Momentum — analyst PT/rating revision trend detector.

Compares analyst upgrades/downgrades/price-target changes over two consecutive 30-day windows:
  - Recent  (0–30 days ago)
  - Prior   (31–60 days ago)

Rising revisions in the recent window vs. the prior window = earnings momentum factor.
Falling revisions or rising downgrades = analyst consensus deteriorating.

Source: yfinance ticker.upgrades_downgrades (free, no key required).
Cached daily to cache/revision_momentum_YYYY-MM-DD.json.
"""

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import yfinance as yf
from loguru import logger

from config import settings
from src.models import RevisionMomentumContext, TickerRevisionData

CACHE_DIR      = Path("cache")
_REQUEST_DELAY = 0.70   # 2× analyst_ratings.py to reduce rate-limit collisions when both run concurrently
_LOOKBACK_DAYS = 60     # total window; each half = 30 days

_PT_RAISE = frozenset({"Raises", "Initiated"})
_PT_CUT   = frozenset({"Lowers"})


def _cache_path() -> Path:
    return CACHE_DIR / f"revision_momentum_{date.today().isoformat()}.json"


def _load_cache() -> Optional[RevisionMomentumContext]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ctx  = RevisionMomentumContext.model_validate(data)
        logger.info(f"[revision] Loaded cached revision momentum ({len(ctx.tickers)} tickers)")
        return ctx
    except Exception as e:
        logger.warning(f"[revision] Cache load failed: {e}")
        return None


def _save_cache(ctx: RevisionMomentumContext) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        _cache_path().write_text(
            json.dumps(ctx.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"[revision] Cache save failed: {e}")


def _window_stats(rows) -> dict:
    """Aggregate upgrade/downgrade/PT-raise/PT-cut counts and mean PT for a row subset."""
    upgrades   = int((rows["Action"] == "up").sum())
    downgrades = int((rows["Action"] == "down").sum())
    maintained = rows[rows["Action"].isin({"main", "reit"})]
    pt_raises  = int(maintained["priceTargetAction"].isin(_PT_RAISE).sum()) if "priceTargetAction" in maintained.columns else 0
    pt_cuts    = int(maintained["priceTargetAction"].isin(_PT_CUT).sum())   if "priceTargetAction" in maintained.columns else 0
    pts = []
    if "currentPriceTarget" in rows.columns:
        for val in rows["currentPriceTarget"].dropna():
            try:
                v = float(val)
                if v > 0:
                    pts.append(v)
            except (ValueError, TypeError):
                pass
    avg_pt = round(sum(pts) / len(pts), 2) if pts else None
    return dict(upgrades=upgrades, downgrades=downgrades,
                pt_raises=pt_raises, pt_cuts=pt_cuts, avg_pt=avg_pt)


def _compute_momentum(recent: dict, prior: dict) -> tuple[float, str]:
    """
    Compute revision momentum score ∈ [-1, +1].

    Logic:
        bull = upgrades × 2 + pt_raises × 1   (bulls scored by conviction weight)
        bear = downgrades × 2 + pt_cuts × 1

        net_recent = bull_recent − bear_recent
        net_prior  = bull_prior  − bear_prior
        momentum   = net_recent − net_prior   (direction of analyst consensus change)

    Score is normalised by total activity to prevent scale bias, then
    stretched by ×3 to spread the effective output range.
    """
    bull_r = recent["upgrades"] * 2 + recent["pt_raises"]
    bear_r = recent["downgrades"] * 2 + recent["pt_cuts"]
    bull_p = prior["upgrades"] * 2 + prior["pt_raises"]
    bear_p = prior["downgrades"] * 2 + prior["pt_cuts"]

    total = (recent["upgrades"] + recent["downgrades"] + recent["pt_raises"] + recent["pt_cuts"] +
             prior["upgrades"]  + prior["downgrades"]  + prior["pt_raises"]  + prior["pt_cuts"])

    if total == 0:
        return 0.0, "STABLE"

    raw = ((bull_r - bear_r) - (bull_p - bear_p)) / max(1, total) * 3
    score = round(max(-1.0, min(1.0, raw)), 3)

    if score >= 0.25:
        direction = "IMPROVING"
    elif score <= -0.25:
        direction = "DETERIORATING"
    else:
        direction = "STABLE"

    return score, direction


def fetch_revision_momentum_context(tickers: List[str]) -> Optional[RevisionMomentumContext]:
    """
    Compute analyst estimate revision momentum across the watchlist.

    Returns None if no tickers had any analyst activity in the 60-day window.
    Caches the result for the current calendar day.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    if not settings.enable_fetch_data:
        logger.debug("[revision] ENABLE_FETCH_DATA=false — skipping yfinance fetch")
        return None

    now           = datetime.now(timezone.utc)
    cutoff_recent = now - timedelta(days=30)
    cutoff_prior  = now - timedelta(days=_LOOKBACK_DAYS)

    ticker_data: List[TickerRevisionData] = []

    for sym in tickers:
        try:
            ud = yf.Ticker(sym).upgrades_downgrades
            if ud is None or ud.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            idx = ud.index
            if idx.tzinfo is None:
                idx = idx.tz_localize("UTC")

            recent_rows = ud[idx >= cutoff_recent]
            prior_rows  = ud[(idx >= cutoff_prior) & (idx < cutoff_recent)]

            if recent_rows.empty and prior_rows.empty:
                time.sleep(_REQUEST_DELAY)
                continue

            recent_stats = _window_stats(recent_rows)
            prior_stats  = _window_stats(prior_rows)
            score, direction = _compute_momentum(recent_stats, prior_stats)

            pt_change_pct: Optional[float] = None
            if (recent_stats["avg_pt"] and prior_stats["avg_pt"] and prior_stats["avg_pt"] > 0):
                pt_change_pct = round(
                    (recent_stats["avg_pt"] - prior_stats["avg_pt"]) / prior_stats["avg_pt"] * 100, 1
                )

            n_firms = int(ud[idx >= cutoff_prior]["Firm"].nunique()) if "Firm" in ud.columns else 0

            ticker_data.append(TickerRevisionData(
                ticker=sym,
                recent_upgrades=recent_stats["upgrades"],
                recent_downgrades=recent_stats["downgrades"],
                recent_pt_raises=recent_stats["pt_raises"],
                recent_pt_cuts=recent_stats["pt_cuts"],
                prior_upgrades=prior_stats["upgrades"],
                prior_downgrades=prior_stats["downgrades"],
                prior_pt_raises=prior_stats["pt_raises"],
                prior_pt_cuts=prior_stats["pt_cuts"],
                momentum_score=score,
                direction=direction,
                avg_pt_current=recent_stats["avg_pt"],
                avg_pt_prior=prior_stats["avg_pt"],
                pt_change_pct=pt_change_pct,
                n_firms=n_firms,
            ))
            logger.debug(
                f"[revision] {sym}: score={score:+.3f} ({direction}) | "
                f"recent {recent_stats['upgrades']}↑{recent_stats['downgrades']}↓ | "
                f"prior {prior_stats['upgrades']}↑{prior_stats['downgrades']}↓"
            )
            time.sleep(_REQUEST_DELAY)

        except Exception as e:
            logger.debug(f"[revision] {sym} failed: {e}")
            time.sleep(_REQUEST_DELAY)

    if not ticker_data:
        logger.info("[revision] No analyst revision data found")
        return None

    scores        = [t.momentum_score for t in ticker_data]
    breadth_score = round(sum(scores) / len(scores), 3)

    if breadth_score >= 0.35:
        signal    = "STRONG_IMPROVING"
        direction = "BULLISH"
    elif breadth_score >= 0.10:
        signal    = "IMPROVING"
        direction = "BULLISH"
    elif breadth_score <= -0.35:
        signal    = "STRONG_DETERIORATING"
        direction = "BEARISH"
    elif breadth_score <= -0.10:
        signal    = "DETERIORATING"
        direction = "BEARISH"
    else:
        signal    = "NEUTRAL"
        direction = "NEUTRAL"

    sorted_tickers    = sorted(ticker_data, key=lambda x: x.momentum_score, reverse=True)
    top_improving     = [t.ticker for t in sorted_tickers if t.momentum_score >= 0.10][:5]
    top_deteriorating = [t.ticker for t in reversed(sorted_tickers) if t.momentum_score <= -0.10][:5]

    n_improving     = sum(1 for t in ticker_data if t.direction == "IMPROVING")
    n_stable        = sum(1 for t in ticker_data if t.direction == "STABLE")
    n_deteriorating = sum(1 for t in ticker_data if t.direction == "DETERIORATING")

    summary = (
        f"Analyst revision momentum across {len(ticker_data)} tickers: "
        f"{n_improving} improving, {n_stable} stable, {n_deteriorating} deteriorating. "
        f"Breadth score: {breadth_score:+.2f} ({signal})."
    )
    if top_improving:
        summary += f" Rising revisions: {', '.join(top_improving)}."
    if top_deteriorating:
        summary += f" Falling revisions: {', '.join(top_deteriorating)}."

    ctx = RevisionMomentumContext(
        tickers=ticker_data,
        breadth_score=breadth_score,
        signal=signal,
        direction=direction,
        top_improving=top_improving,
        top_deteriorating=top_deteriorating,
        report_date=date.today(),
        summary=summary,
    )
    logger.info(f"[revision] {len(ticker_data)} tickers | breadth={breadth_score:+.2f} ({signal})")
    _save_cache(ctx)
    return ctx
