"""Grade past recommendations against realized forward returns (Priority 0).

This is the feedback loop. For every logged recommendation that has now
"matured" (enough trading sessions have closed), we compute the realized
close-to-close return and store it. Aggregating those graded returns gives us
an *accurate*, reproducible measure of how well the calls actually did — the
numbers shown in the email and the basis for any future tuning.

Return convention
-----------------
* entry  = closing price of the first trading session on/after the call date.
* exit   = closing price ``horizon`` trading sessions later.
* raw_return     = exit / entry - 1.
* aligned_return = raw_return signed by the call: BUY -> +1, SELL -> -1.
  So a positive aligned_return always means "the call was directionally right",
  whether it was a BUY or a SELL. HOLD/WATCH carry sign 0 and are excluded from
  the directional hit-rate stats (but still recorded for reference).

Only *settled* sessions are used: today's potentially-partial bar is dropped so
we never grade against an unfinished close.
"""

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from loguru import logger

from src.data.market_data import get_history_range
from src.models import GradedRec, HorizonStats, Scorecard
from src.performance import store


# Forward-return horizons in trading sessions (matches the 1–5 day sentiment thesis).
HORIZONS = (1, 5)


def _signal_sign(action: str) -> int:
    """+1 for BUY, -1 for SELL, 0 for HOLD/WATCH (no directional bet)."""
    return {"BUY": 1, "SELL": -1}.get((action or "").upper(), 0)


def _to_date(value):
    """Coerce a stored ISO timestamp or a pandas/py datetime to a date."""
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    return value.date() if hasattr(value, "date") else value


def score_matured() -> int:
    """Compute and persist forward returns for any matured, ungraded recs.

    Safe to call on every pipeline run: it only does work for recommendations
    that have newly matured since the last run. Returns the number of newly
    graded (rec, horizon) pairs.
    """
    store.init_db()
    today = datetime.now(timezone.utc).date()
    total = 0

    for horizon in HORIZONS:
        ungraded = store.fetch_ungraded(horizon)
        if not ungraded:
            continue

        by_ticker: Dict[str, list] = defaultdict(list)
        for row in ungraded:
            by_ticker[row["ticker"]].append(row)

        for ticker, recs in by_ticker.items():
            results = _grade_ticker(ticker, recs, horizon, today)
            total += store.save_returns(results)

    if total:
        logger.info(f"Scored {total} matured recommendation-horizons")
    return total


def _grade_ticker(ticker: str, recs: list, horizon: int, today) -> List[dict]:
    """Grade all ungraded recs for one ticker at one horizon."""
    rec_dates = [_to_date(r["generated_at"]) for r in recs]
    # Fetch from a few days before the earliest call through today, with enough
    # buffer beyond the latest call to cover the horizon plus weekends/holidays.
    start = min(rec_dates) - timedelta(days=5)
    end = today + timedelta(days=1)

    df = get_history_range(ticker, start=start, end=end)
    if df.empty or "Close" not in df:
        return []

    # Keep only settled sessions (drop today's possibly-partial bar), sorted.
    settled = sorted(
        (d, float(p))
        for d, p in zip((_to_date(i) for i in df.index), df["Close"].tolist())
        if d < today and p is not None
    )
    if not settled:
        return []

    sess_dates = [d for d, _ in settled]
    sess_prices = [p for _, p in settled]
    now_iso = datetime.now(timezone.utc).isoformat()

    out: List[dict] = []
    for rec, rec_date in zip(recs, rec_dates):
        entry_i = bisect_left(sess_dates, rec_date)   # first session on/after call
        exit_i = entry_i + horizon
        if entry_i >= len(sess_dates) or exit_i >= len(sess_dates):
            continue  # not matured yet (no settled exit close available)

        entry_price = sess_prices[entry_i]
        exit_price = sess_prices[exit_i]
        if entry_price <= 0:
            continue

        raw = exit_price / entry_price - 1.0
        aligned = raw * _signal_sign(rec["action"])
        out.append({
            "rec_id": rec["id"],
            "horizon_days": horizon,
            "entry_date": sess_dates[entry_i].isoformat(),
            "entry_price": round(entry_price, 4),
            "exit_date": sess_dates[exit_i].isoformat(),
            "exit_price": round(exit_price, 4),
            "raw_return": round(raw, 6),
            "aligned_return": round(aligned, 6),
            "scored_at": now_iso,
        })
    return out


def build_scorecard(recent_limit: int = 10) -> Scorecard:
    """Assemble realized-performance stats from the graded recommendations."""
    store.init_db()

    horizons: List[HorizonStats] = []
    for horizon in HORIZONS:
        s = store.horizon_stats(horizon)
        graded = s.get("graded") or 0
        hits = s.get("hits") or 0
        horizons.append(HorizonStats(
            horizon_days=horizon,
            graded=graded,
            hits=hits,
            hit_rate=(hits / graded) if graded else 0.0,
            avg_aligned_return=s.get("avg_aligned") or 0.0,
            avg_raw_return=s.get("avg_raw") or 0.0,
            best=s.get("best") or 0.0,
            worst=s.get("worst") or 0.0,
        ))

    recent = [
        GradedRec(
            ticker=row["ticker"],
            action=row["action"],
            direction=row["direction"],
            confidence=row["confidence"],
            generated_at=datetime.fromisoformat(row["generated_at"]),
            horizon_days=row["horizon_days"],
            entry_date=row["entry_date"],
            entry_price=row["entry_price"],
            exit_date=row["exit_date"],
            exit_price=row["exit_price"],
            raw_return=row["raw_return"],
            aligned_return=row["aligned_return"],
            hit=row["aligned_return"] > 0,
        )
        for row in store.recent_graded(recent_limit)
    ]

    return Scorecard(
        horizons=horizons,
        recent=recent,
        total_logged=store.count_recommendations(),
        total_graded=store.count_graded(),
    )
