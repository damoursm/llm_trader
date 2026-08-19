"""catalyst_tilt — catalyst-class-conditioned news orientation (2026-08-15,
PANEL-FIRST at weight 0).

The news-event dataset (signals.news_catalyst live capture + the
news_event_backfill history, joined vs the pivot forward return) shows the news
read's VALUE depends on WHAT KIND of news it is: bullish analyst/company-PR/
management/index-add reads have been reliably wrong (fade), bullish contract/
insider-activity/product/guidance reads right (follow), bearish legal-
regulatory reads wrong, and so on. This method turns that accruing map into a
score:

    side           = bull | bear  (sign of the live news read)
    tilt(cat,side) = clip(mean_oriented_pivot_ret / 2.0, -1, +1) * n/(n+40)
    score          = news x tilt(catalyst, side)     (|tilt| < 0.05 -> abstain)

so tilt = +1 keeps the news read at full conviction, tilt = -1 FLIPS it, and a
thin or flat cell shrinks toward 0 = ABSTAIN — deliberately toward abstention,
not passthrough, so the method only ever emits class-informed views and never
duplicates `news` in the panel.

Self-calibrating (house idiom: evidence → Bayesian shrinkage → clamps →
fail-soft): the map refits from the event dataset every ~6h over the GATED
(price ≥ $5 / 20d dollar-volume ≥ $5M) pivot-labeled events. The fit consumes
only the news SIGN + catalyst + forward return — never the news magnitude — so
it is robust to the sentiment prompt's magnitude-rescaling epochs (direction
semantics have never changed); the live magnitude enters only at serving time.

Measured basis (2026-08-15, memory/news-interaction-methods-2026-08.md
follow-up): leave-one-month-out over 3,269 typed gated events, held-out daily
IC +0.039 (t +2.22) while raw news on the SAME rows measured -0.054 (t -2.23);
plateau-stable across prior 20-80 x scale 1.5-3 (all nine configs positive,
t +1.8..+2.6); the large cells' signs are fold-stable. Config pinned at the
pre-registered mid-plateau (prior 40, scale 2.0).

ABSTAINS (0.0) when: news is 0/None, the run captured no catalyst (provider
path, engine failure, pre-v4 cache), the cell is thin/flat, the calibration
found < 300 labeled events, or the refresh failed (fail-soft {} — a DB hiccup
must never fake a signal). Non-replayable (needs the stored catalyst capture),
like `news` itself.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple

from loguru import logger

from config.settings import settings

_SCALE_PCT = 2.0        # a +/-2% mean oriented pivot move maps to full +/-1 tilt
_PRIOR_N = 40           # shrinkage toward 0 (abstain), NOT toward passthrough
_MIN_ABS_TILT = 0.05    # smaller tilts abstain
_MIN_EVENTS = 300       # calibration evidence floor (total labeled events)
_TTL_SECONDS = 6 * 3600

_CACHE: dict = {"ts": 0.0, "tilts": None}
_LOCK = threading.Lock()   # single-flight: build_signals scores on many threads


def _fit(events) -> Dict[Tuple[str, str], float]:
    """{(catalyst, side): tilt} from the labeled gated event frame."""
    import numpy as np
    ev = events.copy()
    lo, hi = ev.fwd_ret_pivot.quantile(0.01), ev.fwd_ret_pivot.quantile(0.99)
    ev["oriented"] = np.sign(ev.news) * ev.fwd_ret_pivot.clip(lo, hi)
    ev["side"] = np.where(ev.news > 0, "bull", "bear")
    out: Dict[Tuple[str, str], float] = {}
    for (cat, side), g in ev.groupby(["catalyst", "side"]):
        n = len(g)
        tilt = float(np.clip(g.oriented.mean() / _SCALE_PCT, -1.0, 1.0)
                     * n / (n + _PRIOR_N))
        out[(str(cat), str(side))] = round(tilt, 4)
    return out


def calibrate_catalyst_tilt(force: bool = False) -> Dict[Tuple[str, str], float]:
    """The tilt map, refit at most every ``_TTL_SECONDS``. Fail-soft {}."""
    now = time.time()
    hit = _CACHE["tilts"]
    if not force and hit is not None and (now - _CACHE["ts"]) < _TTL_SECONDS:
        return hit
    with _LOCK:
        hit = _CACHE["tilts"]                       # re-check under the lock
        if not force and hit is not None and (time.time() - _CACHE["ts"]) < _TTL_SECONDS:
            return hit
        tilts: Dict[Tuple[str, str], float] = {}
        try:
            from src.analysis.news_events import load_news_events
            from src.data.cache import load_ohlcv
            ev = load_news_events()
            ev = ev[(ev.news.notna()) & (ev.news != 0.0)
                    & ev.catalyst.notna() & ev.fwd_ret_pivot.notna()]
            # Gate 4-equivalent floor per event date (memory: every pivot number
            # is meaningful only GATED — ungated events inflate the tilts with
            # microcap bounce).
            keep = []
            for tk, g in ev.groupby("ticker"):
                d = load_ohlcv(tk)
                if d is None or len(d) < 25:
                    continue
                close, volume = d["Close"].astype(float), d["Volume"].astype(float)
                addv = (close * volume).rolling(20).mean().shift(1)
                px = close.copy()
                px.index = addv.index = d.index.strftime("%Y-%m-%d")
                for _, r in g.iterrows():
                    day = str(r.signal_date)
                    if day in px.index and float(px.loc[day]) >= 5.0 \
                            and float(addv.loc[day]) >= 5e6:
                        keep.append(r)
            import pandas as pd
            gated = pd.DataFrame(keep)
            if len(gated) >= _MIN_EVENTS:
                tilts = _fit(gated)
                logger.info(f"[catalyst_tilt] refit on {len(gated)} gated events "
                            f"→ {sum(1 for v in tilts.values() if abs(v) >= _MIN_ABS_TILT)} "
                            f"active cells")
            else:
                logger.info(f"[catalyst_tilt] only {len(gated)} gated events "
                            f"(< {_MIN_EVENTS}) — method abstains")
        except Exception as e:
            logger.warning(f"[catalyst_tilt] calibration failed (method abstains): {e}")
            tilts = {}
        _CACHE.update(ts=time.time(), tilts=tilts)
        return tilts


def compute_catalyst_tilt_score(news_score: Optional[float],
                                catalyst: Optional[str],
                                tilts: Dict[Tuple[str, str], float]) -> float:
    """The tilted score ∈ [-1, +1]; 0.0 on every abstention path."""
    if not news_score or not catalyst or not tilts:
        return 0.0
    side = "bull" if news_score > 0 else "bear"
    tilt = tilts.get((str(catalyst), side), 0.0)
    if abs(tilt) < _MIN_ABS_TILT:
        return 0.0
    return round(max(-1.0, min(1.0, float(news_score) * tilt)), 4)


def reset_cache() -> None:
    """Test / asof hook."""
    _CACHE.update(ts=0.0, tilts=None)
