"""ml_buy — the FULL STACKER: a model whose features are ALL the per-method
scores from the signals panel, predicting the BUY outcome P(up).

Role B of the ML plan (the learned aggregator), buy-side. Where ``ml_ohlcv`` is a
NEW base signal from price/volume, this is a META-model over the EXISTING methods
— the learned counterpart to ``combined_buy_score`` — which can capture
interactions the hand-weighted linear combine cannot (momentum working only in a
clean trend, news mattering only when volume confirms, ...).

**Data: the signals PANEL only.** The method scores (news, options, sentiment,
insider, ...) are NOT replayable — they need point-in-time feeds nobody stored —
so unlike ml_ohlcv this cannot use the deep cache. It trains on the forward-
collected panel: tens of thousands of ticker-days but only WEEKS of distinct
days, so OVERFITTING to the single regime is the real risk and the panel
walk-forward is the honest judge. Nothing here is trusted until the forward IC
accrues. (Measured 2026-08-11: 20x row inflation from keeping every intraday
run adds nothing — same days, shared labels — and small-data GBM params beat
the lgb defaults at t +2.0; see STACKER_GBM_PARAMS.)

**Circularity — the load-bearing guard.** Features are the individual METHOD
scores, which do NOT depend on the weights, so training a model on them is sound.
``combined_score`` / ``combined_buy_score`` / ``confidence`` DO depend on the
weights and are EXCLUDED from the features (using them would fit the model on
values derived from the very weights it exists to inform). ``combined_score`` is
kept as a BASELINE — the bar the learned stacker must beat to be worth anything.

**Output: P(up) ∈ [0,1]** = the buy conviction (the two-sided [0,1] convention).
For panel IC-tracking it is emitted as the signed ``2*P(up)-1`` so it ranks
alongside the other methods' signed scores; when promoted it contributes P(up)
to the buy camp of the split combine.

CLI:  python -m src.analysis.ml_stacker [--horizons 1,5,10] [--model gbm]
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import date
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings
from src.analysis.ml_dataset import _benchmark_return, _benchmark_series
from src.analysis.ml_train import evaluate
from src.analysis.sentiment import NEWS_CATALYST_TYPES
from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS

# The stacker's features: every individual method score. All weight-INDEPENDENT
# (combined_score/confidence are NOT here — that is the circularity guard). The
# panel persists exactly these columns, so the dataset is essentially build_panel.
# Plus `tape_score` (2026-08-22): the score-independent price/volume tape
# composite — a replay CONTEXT column, not a method column. The panel gets it
# from the signals_replay merge; serving computes it live (the same
# `compute_tape_confirmation` the confidence factor uses). Measured as the one
# robust feature ADDITION of the 27-arm redesign: +0.0175 IC/day over the same
# model without it (t +2.72, wins 72% of days, same-sign halves) — until now the
# tape's direction was only a confidence qualifier, never scored.
STACKER_FEATURES: List[str] = (list(SIGNAL_BASE_METHOD_COLUMNS) + ["tape_score"]
                               + ["news_raw_score", "news_recency_mass",
                                  "news_article_count", "atr_pct", "bb_width_pct",
                                  "vol_ratio"])

# The LIVE feature set — 21 of the weighted methods in the aggregator's
# `method_score_map`. DELIBERATELY NOT the full post-2026-08-11 set of 27: the
# six promoted methods (mom_12_1/hi52/st_reversal/rsi2_rev/dloc_rev/ml_ohlcv)
# were measured as features on 2026-08-11 (scratchpad stacker_tune.py) and
# HURT — their panel history is mostly-NaN through the training windows (epochs
# + method age) while populated at serve time, a train/serve distribution shift
# (cur config 27f vs 21f: t −1.79). REVISIT once their panel history thickens
# (~2026-09); the extension is design-correct then. Until that re-test, the
# length-21 pin in tests/test_ml_stacker.py is the record of this decision.
STACKER_SIGNED_FEATURES: List[str] = [
    "news", "sent_velocity", "tech", "massive", "insider", "put_call", "max_pain",
    "oi_skew", "vwap", "pattern", "momentum", "sector_momentum", "market_momentum",
    "money_flow", "trend_strength", "pead", "iv_rank", "iv_expr", "coint",
    "ext_gap", "broker_advisor",
    # 22nd method (2026-08-24, user directive): the deep-cache price model's own
    # score. Panel coverage is ~7% (epoch-masked before 2026-08-13), so today it
    # is near-inert (the exit model measured byte-identical predictions at the
    # same coverage) — it self-activates as post-epoch history accrues. The
    # OTHER five 2026-08-11 promoted methods stay excluded (measured harmful as
    # a six-pack at t −1.79; re-test ~2026-09-10). Shared with the exit model
    # via EXIT_METHODS' derivation.
    "ml_ohlcv",
    # 23rd feature (2026-08-22, NOT a method column — see STACKER_FEATURES note):
    "tape_score",
    # 2026-09-07 (user directive): the rest of the NEWS family. The three
    # panel-first news methods are weight 0 in the combine but ordinary numeric
    # columns here, and `news_raw_score` is the PRE-SCALER verdict — the model
    # can learn its own evidence scaler instead of only seeing the scaled one.
    # All four are SIGNED, so the exit model orients them like any other score.
    "news_shock", "news_bear_fresh", "catalyst_tilt", "news_raw_score",
    # 2026-09-07 (user directive): the news read net of the move already made
    # since the story began — the freshest cluster, and every cluster.
    "news_unpriced", "news_unpriced_all", "news_bull_fresh",
    # 2026-09-11: `news_quiet` was the ONE news-family method missing from this
    # list, and it is the one that could least afford to be. With
    # `ml_combine_arm_share` at 1.0 the stackers ARE the combine — `combine_source`
    # reads "ml" on 100% of live rows, zero fail-soft — so a base weight only
    # reaches the per-side fail-soft that never fires. Its 0.10 was therefore
    # deciding nothing about DIRECTION; it still reached coherence,
    # `sources_agreeing` and the Sentiment family vote, which feed confidence and
    # so position SIZE, but not which names are picked.
    #
    # Signed like the rest (it carries the raw verdict on quiet names, 0.0 =
    # abstain), so `EXIT_METHODS` picks it up by derivation at the same retrain.
    "news_quiet",
]

# UNSIGNED context features (2026-09-07). Never oriented by direction — a count,
# a width and a one-hot have no bullish/bearish sense — so they are excluded
# from EXIT_METHODS rather than fed to it multiplied by a position's sign.
CATALYST_FEATURE_PREFIX = "cat_"
CATALYST_ONEHOT_FEATURES: List[str] = [f"{CATALYST_FEATURE_PREFIX}{c}"
                                       for c in NEWS_CATALYST_TYPES]
# The confidence COMPONENTS were requested here and are deliberately NOT added:
# `confidence` and `raw_confidence` are functions of the combine this model
# PRODUCES (raw_confidence = |combined| / divisor), and `coherence_factor`,
# `volume_factor`, `family_conf_factor`, `tape_conf_factor` and
# `sector_conf_factor` are all computed downstream of it — at serving time they
# do not exist yet when the stacker runs, so training on them would be a silent
# train/serve skew on top of the circularity. What they encode that IS available
# before the combine is the market state underneath them, so that goes in
# instead: `movement_factor`'s own inputs (atr_pct, bb_width_pct) and the volume
# factor's (vol_ratio). `tape_score` was already a feature.
STACKER_CONTEXT_FEATURES: List[str] = [
    "news_recency_mass", "news_article_count",
    "atr_pct", "bb_width_pct", "vol_ratio",
] + CATALYST_ONEHOT_FEATURES

STACKER_LIVE_FEATURES: List[str] = STACKER_SIGNED_FEATURES + STACKER_CONTEXT_FEATURES

# News-derived columns that `build_panel` does NOT mask (they are not method
# scores), but which the news-family scorer epoch applies to all the same: the
# raw verdict and the catalyst class are outputs of the sentiment PROMPT, and
# the article count and recency mass are outputs of the relevance FILTER — both
# of which changed at the same 2026-09-04 boundary. Masking them here is the
# standing calibration rule ("fit only what the current code produced") applied
# to the one place the panel's own mask cannot reach.
# Features consumed as a WITHIN-RUN CENTERED RANK rather than an absolute value
# (2026-09-07, user directive). Every one of them is a function of the news POOL
# and of the SCORING ENGINE, and both of those move underneath the model:
#
#   * the engines differ 2.7x in MAGNITUDE on identical digests (Qwen vs
#     DeepSeek, 571 shadow pairs) while agreeing 83% on sign and 0.68 on order;
#   * a backfilled digest is ~0.40x live magnitude for the same reason.
#
# A rank is invariant to both. It is the same normalisation the weighted combine
# already applies (`aggregator._rank_transform_run`), for the same reason: what
# these features carry is an ORDERING, and their level is an artifact of who
# scored them and from how many articles.
STACKER_RANKED_FEATURES: List[str] = [
    "news", "news_raw_score", "sent_velocity", "news_shock", "news_bear_fresh",
    "catalyst_tilt", "news_unpriced", "news_unpriced_all", "news_bull_fresh",
    "news_recency_mass", "news_article_count",
]

NEWS_EPOCH_MASKED_FEATURES: List[str] = (
    ["news_raw_score", "news_recency_mass", "news_article_count"]
    + CATALYST_ONEHOT_FEATURES)

# Kept as columns for the BASELINE comparison (NOT fed to the model): the current
# hand-weighted combine is the bar a learned stacker must clear to justify itself.
# Deliberately NOT a method score (those are features) — these are the aggregate
# combine columns, which the stacker never sees but must beat.
_BASELINE_COLUMNS = ("combined_score", "combined_buy_score")


def centered_rank(values) -> List[float]:
    """``2*(rank-1)/(n-1) - 1`` over the non-zero finite values, average ranks on
    ties — the run's strongest view +1, weakest -1, median ~0.

    ONE function, used by the dataset builder and by serving, because a rank the
    two sides compute differently is worse than no rank at all. Deliberately the
    PLAIN centered rank, NOT `aggregator._rank_transform_run`: that one also
    applies the fitted rank-SHAPING curves, which are fitted on the same forward
    returns the stacker trains against — payoff shaping belongs in the combine,
    not inside a model feature.

    Conventions match the house rank basis: a ZERO is an abstention and stays
    0.0 (never ranked), a NaN stays NaN, and a cross-section too thin to order
    (< 2 views) yields 0.0 — no cross-sectional information is a neutral view,
    not a fabricated extreme.
    """
    import numpy as _np
    out: List[float] = []
    idx = []
    vals = []
    for i, v in enumerate(values):
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(float("nan"))
            continue
        if f != f:
            out.append(float("nan"))
        elif f == 0.0:
            out.append(0.0)
        else:
            out.append(0.0)                 # placeholder, filled below
            idx.append(i)
            vals.append(f)
    n = len(vals)
    if n < 2:
        return out
    order = _np.argsort(_np.argsort(_np.asarray(vals, dtype=float), kind="stable"),
                        kind="stable").astype(float)
    # average ranks on ties, so identical evidence shares a rank
    arr = _np.asarray(vals, dtype=float)
    ranks = _np.empty(n, dtype=float)
    for v in set(vals):
        m = arr == v
        ranks[m] = order[m].mean()
    for k, i in enumerate(idx):
        out[i] = float(2.0 * ranks[k] / (n - 1) - 1.0)
    return out


def centered_rank_pooled(pairs, tradeable=None) -> dict:
    """`centered_rank` over the GATE-4 TRADEABLE pool, with every observe-only
    name INTERPOLATED against that same distribution.

    Why the pool (2026-09-09 user directive, "use the tradeable pool too for the
    news, like the other methods"): the observe-only segment is a different
    population — thinner, cheaper, different drift and reversal — so ranking one
    cross-section over both blends two regimes, and a news verdict's rank would
    mean something different from every other method's on the same row. This is
    the same argument `rank_tradeable_only` already settled for the combine.

    Mirrors `aggregator._rank_transform_run` exactly — tie-averaged 1-based
    ranks, an observe-only view placed at the equivalent average rank it WOULD
    take inside the tradeable distribution, `2*(rank-1)/(n-1) - 1` — MINUS the
    payoff shaping, which is fitted on the same forward returns the stacker
    trains against and has no business inside a model feature.
    `tests/test_stacker_news_features.py` pins the two against each other.

    ``tradeable=None`` ranks over everything, which is both the pre-directive
    behaviour and the fail-soft whenever the pool is unavailable or too thin.

    ``pairs`` is ``[(ticker, value)]``; conventions as `centered_rank` — a ZERO
    abstains and stays 0.0, a NaN stays NaN, a cross-section under 2 views is a
    neutral 0.0 rather than a fabricated extreme.
    """
    from bisect import bisect_left, bisect_right
    out: dict = {}
    live = []
    for tk, v in pairs:
        try:
            f = float(v)
        except (TypeError, ValueError):
            out[tk] = float("nan")
            continue
        if f != f:
            out[tk] = float("nan")
        elif f == 0.0:
            out[tk] = 0.0
        else:
            live.append((tk, f))
    if tradeable is not None:
        views = [(t, v) for t, v in live if t in tradeable]
        observe = [(t, v) for t, v in live if t not in tradeable]
    else:
        views, observe = list(live), []
    n = len(views)
    if n < 2:                                       # nothing to order
        for tk, _v in live:
            out[tk] = 0.0
        return out
    views.sort(key=lambda t: t[1])
    ranks: dict = {}
    i = 0
    while i < n:                                    # average ranks over tie groups
        j = i
        while j + 1 < n and views[j + 1][1] == views[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[views[k][0]] = avg
        i = j + 1
    tvals = [v for _t, v in views]
    for tk, v in observe:
        r_avg = (bisect_left(tvals, v) + bisect_right(tvals, v)) / 2.0 + 0.5
        ranks[tk] = min(max(r_avg, 1.0), float(n))
    for tk, _v in views + observe:
        out[tk] = float(2.0 * (ranks[tk] - 1.0) / (n - 1.0) - 1.0)
    return out


def rank_news_features(rows: dict, tradeable=None) -> dict:
    """``{ticker: {feature: centered rank}}`` for one run's cross-section.

    ``rows`` is ``{ticker: {feature: raw value}}``; ``tradeable`` is the run's
    Gate-4 pool (None = rank over everything). Only ``STACKER_RANKED_FEATURES``
    are transformed; anything else is ignored here and continues to reach the
    model raw.
    """
    tickers = list(rows)
    out = {t: {} for t in tickers}
    for f in STACKER_RANKED_FEATURES:
        ranked = centered_rank_pooled([(t, rows[t].get(f)) for t in tickers], tradeable)
        for t in tickers:
            out[t][f] = ranked.get(t, float("nan"))
    return out


def news_basis_of(art) -> str:
    """``"rank"`` only when the ARTIFACT says it trained that way.

    Serving dispatches on the artifact's own stamp, never on today's setting —
    the same rule `ml_exit` uses for `label_basis` / `combine_basis`. An artifact
    trained on absolute values must keep receiving absolute values, or the
    feature means one thing in training and another at serve.
    """
    if not art:
        return "absolute"
    return str((art.get("config") or {}).get("news_basis") or "absolute")


def catalyst_onehot(catalyst) -> dict:
    """``{cat_<class>: 1.0/0.0}`` over the FIXED taxonomy.

    One-hot rather than any target encoding: a fitted encoding of a categorical
    against the same forward returns the stacker trains on is circular, and the
    house already has the honest version of that encoding as a separate,
    walk-forward-fitted method (`catalyst_tilt`). An unknown or missing class
    yields all-zeros, which is what a row with no catalyst should look like —
    not a fabricated `none`, which is a real class meaning "read, and nothing
    was there".
    """
    row = {f"{CATALYST_FEATURE_PREFIX}{c}": 0.0 for c in NEWS_CATALYST_TYPES}
    key = str(catalyst or "").strip().lower()
    if key and f"{CATALYST_FEATURE_PREFIX}{key}" in row:
        row[f"{CATALYST_FEATURE_PREFIX}{key}"] = 1.0
    return row


def news_basis() -> str:
    """The configured basis for the news-family features (TRAINING side only —
    serving reads the artifact's stamp)."""
    v = str(getattr(settings, "stacker_news_basis", "rank") or "rank").strip().lower()
    return v if v in ("rank", "absolute") else "rank"


def apply_news_basis(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Replace the news-family columns with their WITHIN-DAY centered rank.

    Grouped by ``signal_date``, which is the cross-section the live transform
    ranks over (one run's universe) — ranking across days would mix regimes and
    would not be reproducible at serve time, where only the current run exists.
    Ranked over that day's GATE-4 TRADEABLE pool with the observe-only names
    interpolated, exactly as serving does. Uses `centered_rank_pooled`, the same
    function serving calls, so the two sides cannot drift.
    """
    if panel is None or panel.empty or news_basis() != "rank":
        return panel
    cols = [c for c in STACKER_RANKED_FEATURES if c in panel.columns]
    if not cols or "signal_date" not in panel.columns:
        return panel
    df = panel.copy()
    pools = _tradeable_pools(df)
    keys = list(zip(df["signal_date"].astype(str), df["ticker"].astype(str))) \
        if "ticker" in df.columns else None
    if keys is None:                            # no ticker column -> cannot pool
        pools = {}
    for c in cols:
        vals = []
        for day, g in df.groupby(df["signal_date"].astype(str)):
            ranked = centered_rank_pooled(
                list(zip(g["ticker"].astype(str), g[c])), pools.get(day))
            for i, tk in zip(g.index, g["ticker"].astype(str)):
                vals.append((i, ranked.get(tk, float("nan"))))
        df[c] = pd.Series(dict(vals)).reindex(df.index)
    n_pooled = sum(1 for v in pools.values() if v is not None)
    logger.info(f"[ml_stacker] news basis=rank: {len(cols)} column(s) replaced by their "
                f"within-day centered rank over the Gate-4 pool "
                f"({n_pooled}/{max(1, len(pools))} days pooled, rest full-universe)")
    return df


def _tradeable_pools(df: "pd.DataFrame") -> dict:
    """``{signal_date: set(ticker) | None}`` — that day's Gate-4 tradeable pool.

    POINT-IN-TIME by construction, unlike the tier-2 backtest's version of the
    same pool: the price floor reads the row's own stored `signals.price`, and
    the dollar-volume floor is `liquidity.dollar_volume`'s definition computed
    over the bars visible ON that signal date only (a `searchsorted` into the
    cached frame), never the cache tail as it stands today. Ranking is a
    consumption-time transform of a TRAINING feature, so a pool built from
    future liquidity would leak — the live path has no such option and neither
    should this.

    A pool under 30 names yields None — full-universe ranking for that day —
    mirroring the aggregator's own thin-pool fail-soft rather than emptying the
    cross-section.
    """
    days = [str(d) for d in df["signal_date"].astype(str).unique()]
    if "price" not in df.columns or "ticker" not in df.columns:
        return {d: None for d in days}
    try:
        import numpy as np
        from src.data.cache import load_ohlcv
        from src.data.liquidity import DOLLAR_VOLUME_WINDOW
    except Exception:                                       # noqa: BLE001
        return {d: None for d in days}
    min_px = float(getattr(settings, "trade_min_price", 5.0) or 0.0)
    min_dv = float(getattr(settings, "trade_min_dollar_volume", 5e6) or 0.0)
    series: dict = {}

    def _series(tk: str):
        """``(sorted date strings, per-bar Close*Volume)`` — parsed once."""
        if tk not in series:
            try:
                h = load_ohlcv(tk)
                idx = np.asarray([str(x)[:10] for x in h.index])
                dv = (pd.to_numeric(h["Close"], errors="coerce")
                      * pd.to_numeric(h["Volume"], errors="coerce")).to_numpy(dtype=float)
                series[tk] = (idx, dv)
            except Exception:                               # noqa: BLE001
                series[tk] = (np.asarray([]), np.asarray([]))
        return series[tk]

    out: dict = {}
    for day, g in df.groupby(df["signal_date"].astype(str)):
        pool = set()
        for tk, px in zip(g["ticker"].astype(str), g["price"]):
            if px is None or px != px or float(px) < min_px:
                continue
            idx, dv = _series(tk)
            if idx.size == 0:
                continue
            cut = int(np.searchsorted(idx, day, side="right"))   # bars visible that day
            tail = dv[max(0, cut - DOLLAR_VOLUME_WINDOW):cut]
            tail = tail[~np.isnan(tail)]
            if tail.size and float(tail.mean()) >= min_dv:
                pool.add(tk)
        out[str(day)] = pool if len(pool) >= 30 else None
    return out


def _news_epoch_day() -> Optional[str]:
    """The news family's scorer-epoch DAY (date-granular consumers use the day
    AFTER a mid-day change, `method_epochs.epoch_for`). None = no epoch, no
    mask."""
    try:
        from src.signals.method_epochs import epoch_for
        ep = epoch_for("news")
        return None if ep is None else ep.isoformat()
    except Exception:                                       # noqa: BLE001 - fail OPEN
        return None


def add_news_features(panel: "pd.DataFrame") -> "pd.DataFrame":
    """Derive the news CONTEXT features on a panel frame (2026-09-07).

    Adds the catalyst one-hot and applies the news-family epoch mask to every
    news-derived column the panel's own mask does not reach (`build_panel` masks
    METHOD scores and confidence; the raw verdict, the catalyst and the evidence
    counts are not method columns). The mask is what keeps the standing rule —
    a calibration fits only what the current code produced — true of a training
    set that now carries four more news-derived inputs.

    The catalyst is resolved repair -> live -> backfill, the same order
    `news_events` uses, so a repaired label reaches the model exactly when (and
    only when) `enable_catalyst_repair_resolution` is on.

    Fail-soft: any failure leaves the frame with all-zero one-hot columns, which
    the model imputes to its training median like any other missing feature.
    """
    import pandas as pd
    if panel is None or panel.empty:
        return panel
    df = panel
    cat = (df["news_catalyst"] if "news_catalyst" in df.columns
           else pd.Series([None] * len(df), index=df.index))
    cat = cat.astype("object").where(cat.notna(), None)
    try:
        cat = _resolve_catalyst_series(df, cat)
    except Exception as exc:                                # noqa: BLE001
        logger.debug(f"[ml_stacker] catalyst resolution skipped: {exc}")
    onehot = pd.DataFrame([catalyst_onehot(c) for c in cat], index=df.index)
    df = pd.concat([df, onehot], axis=1)
    if getattr(settings, "enable_stacker_news_epoch_mask", True):
        day = _news_epoch_day()
        if day and "signal_date" in df.columns:
            stale = df["signal_date"].astype(str) < day
            cols = [c for c in NEWS_EPOCH_MASKED_FEATURES if c in df.columns]
            if stale.any() and cols:
                df.loc[stale, cols] = float("nan")
                logger.debug(f"[ml_stacker] news-epoch mask: {int(stale.sum())} of "
                             f"{len(df)} rows masked on {len(cols)} news columns "
                             f"(< {day})")
    return df


def _resolve_catalyst_series(df, live_cat):
    """repair (resolved|override) -> live -> `news_event_backfill`."""
    import pandas as pd
    out = list(live_cat)
    # (1) the repair, when consumers are allowed to read it.
    if getattr(settings, "enable_catalyst_repair_resolution", False) \
            and "news_digest_id" in df.columns:
        from src.analysis.catalyst_repair import repair_lookup
        lookup = repair_lookup()
        if lookup:
            for i, (digest, cur) in enumerate(zip(df["news_digest_id"], out)):
                if digest is None or (isinstance(digest, float) and digest != digest):
                    continue
                hit = lookup.get((str(digest), "" if cur is None else str(cur)))
                if hit and hit["quality"] != "unresolved" and hit["catalyst"]:
                    out[i] = hit["catalyst"]
    # (2) the historical backfill, for rows the live capture never typed. Its
    # provenance is a later classifier over re-fetched headlines, so it fills
    # only NULLs — it never overwrites a label the scorer itself emitted.
    if any(c is None for c in out) and {"ticker", "signal_date"} <= set(df.columns):
        from src.db import repo
        bf = repo.fetch_df("SELECT ticker, signal_date, catalyst FROM news_event_backfill "
                           "WHERE catalyst IS NOT NULL")
        if bf is not None and not bf.empty:
            m = {(str(t), str(d)): str(c)
                 for t, d, c in zip(bf.ticker, bf.signal_date, bf.catalyst)}
            for i, (tk, day, cur) in enumerate(zip(df["ticker"], df["signal_date"], out)):
                if cur is None:
                    out[i] = m.get((str(tk), str(day)))
    return pd.Series(out, index=df.index)


def build_stacker_dataset(horizons: Sequence[int] = (1, 5, 10), days: Optional[int] = None,
                          benchmark: Optional[str] = None,
                          dedupe: str = "last") -> pd.DataFrame:
    """One row per (ticker, signal_date) from the panel: the method-score features
    + raw/market-relative forward labels + ``end_date_<h>d`` for the walk-forward
    point-in-time split. Same schema ``ml_train.evaluate`` consumes.

    The forward return is the panel's own (authoritative) ``fwd_ret_<h>d``; the
    market-relative leg and the horizon END DATE use the benchmark's session grid
    (SPY trades every session, so it is a clean universal calendar for the cutoff).
    """
    from src.analysis.signal_panel import build_panel
    benchmark = benchmark or settings.horizon_market_benchmark
    horizons = list(horizons)
    # ``dedupe="all"`` keeps every intraday run's row (~5-8x rows with shared
    # same-day labels — callers weighting by day should day-normalise); the
    # default "last" keeps the one-row-per-(day, ticker) contract unchanged.
    panel = build_panel(horizons=horizons, days=days, dedupe=dedupe)
    if panel is None or panel.empty:
        return pd.DataFrame()

    panel = add_news_features(panel)
    panel = apply_news_basis(panel)
    # The one-hot columns are DERIVED (they do not exist on the panel until
    # `add_news_features` builds them), so they join the selection list here
    # rather than in STACKER_FEATURES, which names panel columns.
    feats = [c for c in list(STACKER_FEATURES) + CATALYST_ONEHOT_FEATURES
             if c in panel.columns]
    if "tape_score" not in feats:
        # The 22nd feature comes from the signals_replay merge; a panel without
        # it means the replay materialisation is missing/stale. Training would
        # silently fit a 21-feature model — loud, because that is invisible.
        logger.warning("[ml_stacker] panel has no tape_score column — replay "
                       "materialisation missing? Training will drop the feature.")
    base = [c for c in _BASELINE_COLUMNS if c in panel.columns]
    # combined_sell_score is kept (not a feature, not a baseline) so the swap
    # validation can form combined_score = swapped_buy - combined_sell_score.
    extra = [c for c in ("combined_sell_score",) if c in panel.columns]
    fwd_cols = [f"fwd_ret_{h}d" for h in horizons if f"fwd_ret_{h}d" in panel.columns]
    # The pivot label + its per-row settle date ride along whenever the panel
    # carries them (2026-08-12) — the rank block below turns them into
    # `fwd_ret_rank_pv`/`end_date_pv`, the stackers' default training label.
    fwd_cols += [c for c in ("fwd_ret_pivot", "end_date_pivot") if c in panel.columns]
    # dict.fromkeys dedupes while preserving order — a column that is both a
    # feature and a baseline must not be selected twice (a duplicate column makes
    # df[col] 2-D and breaks the downstream metrics).
    cols = list(dict.fromkeys(["signal_date", "ticker"] + feats + base + extra + fwd_cols))
    df = panel[cols].copy()

    b_dates, b_closes = _benchmark_series(benchmark)

    def _enrich(row) -> pd.Series:
        out = {}
        D = date.fromisoformat(str(row["signal_date"])[:10])
        i = bisect_left(b_dates, D) if b_dates else 0
        for h in horizons:
            raw = row.get(f"fwd_ret_{h}d")
            raw = float(raw) if raw is not None and pd.notna(raw) else np.nan
            out[f"fwd_ret_raw_{h}d"] = raw
            end = b_dates[i + h] if (b_dates and i < len(b_dates) and i + h < len(b_dates)) else None
            out[f"end_date_{h}d"] = end.isoformat() if end is not None else None
            bench = (_benchmark_return(b_dates, b_closes, b_dates[i], end)
                     if (end is not None and i < len(b_dates)) else None)
            out[f"fwd_ret_rel_{h}d"] = (raw - bench) if (bench is not None and raw == raw) else np.nan
        return pd.Series(out)

    df = pd.concat([df.reset_index(drop=True), df.apply(_enrich, axis=1).reset_index(drop=True)], axis=1)

    # CROSS-SECTIONAL RANK label (2026-08-04) — the per-day percentile of the raw
    # forward return, centred at 0 (so ``> 0`` = beat that day's median name).
    #
    # This is the target the LIVE combine trains on, for two measured reasons:
    #
    # 1. ``rel`` (raw − benchmark) implicitly assumes EVERY stock has beta = 1.0,
    #    so it injects noise ∝ (β−1)×market_return. The rank removes the day's
    #    common factor non-parametrically, assuming no beta at all.
    # 2. **It makes probability calibration valid.** The combine ranks WITHIN a day
    #    and takes the top N, so the quantity that matters is the per-day IC. A
    #    label carrying day-level drift (``rel``) has pooled IC ≪ its daily IC
    #    (+0.0035 vs +0.0395 measured), so a calibrator fitted on POOLED
    #    predictions maps everything to the pooled base rate and destroys exactly
    #    the within-day ranking that carries the signal (conviction >0.15 went
    #    27.6% → 0.0% of the universe). The rank label is per-day demeaned BY
    #    CONSTRUCTION, so pooled ≈ daily (+0.0252 vs +0.0263) and the pooled
    #    calibrator becomes legitimate rather than destructive.
    #
    # Causality: the label ranks forward returns that all realise at the same
    # time, so this adds no look-ahead beyond the label horizon itself.
    for h in horizons:
        raw_c = f"fwd_ret_raw_{h}d"
        if raw_c in df.columns:
            df[f"fwd_ret_rank_{h}d"] = (df.groupby("signal_date")[raw_c]
                                          .rank(pct=True, method="average") - 0.5)

    # PIVOT label (2026-08-12, user directive): the within-day centred rank of
    # the SIGNED PIVOT TARGET — the ml_ohlcv-v2 objective, now the stackers'
    # default label (``stacker_label_basis``). Settled rows only (an unsettled
    # pivot is NaN and drops out of both the rank and the training set);
    # ``end_date_pv`` carries each row's OWN settle date so the walk-forward
    # trains strictly on printed labels.
    if "fwd_ret_pivot" in df.columns:
        df["fwd_ret_rank_pv"] = (df.groupby("signal_date")["fwd_ret_pivot"]
                                   .rank(pct=True, method="average") - 0.5)
        if "end_date_pivot" in df.columns:
            df["end_date_pv"] = df["end_date_pivot"]

    logger.info(f"[ml_stacker] {len(df):,} rows over {df['ticker'].nunique()} tickers "
                f"({df['signal_date'].min()} .. {df['signal_date'].max()}), "
                f"{len(feats)} method features")
    return df


def measure(horizons: Sequence[int] = (1, 5, 10), bases: Sequence[str] = ("raw", "rel"),
            deadband: float = 0.0, model_name: str = "gbm", days: Optional[int] = None,
            min_train_days: int = 8, step_days: int = 2) -> pd.DataFrame:
    """Walk-forward the stacker over the panel and return the go/no-go table —
    the learned model's OOS IC/ICIR/hit/simret vs the BASELINES (the hand-weighted
    combined_score it must beat). Thin by construction (~34 panel days)."""
    df = build_stacker_dataset(horizons=horizons, days=days)
    if df.empty:
        return pd.DataFrame()
    return evaluate(df, horizons=horizons, bases=bases, deadband=deadband,
                    model_name=model_name, baseline_features=_BASELINE_COLUMNS,
                    features=STACKER_FEATURES, min_train_days=min_train_days,
                    step_days=step_days, min_train_rows=1000)


def _feature_value(f: str, method_scores: dict, ranked: Optional[dict], art) -> float:
    """One feature for one ticker, on the basis the ARTIFACT was trained with."""
    import numpy as _np
    src = method_scores
    if (ranked is not None and f in STACKER_RANKED_FEATURES
            and news_basis_of(art) == "rank"):
        src = ranked
    v = src.get(f)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _np.nan
    return v if v == v else _np.nan


def buy_conviction_from_proba(p_up: float) -> float:
    """Map the stacker's P(up) to a combined_buy_score-compatible conviction in
    [0,1]: 0 at neutral (P=0.5), 1 at certain (P=1). Without this centering a raw
    P(up)~0.5 for a neutral name would swamp combined_sell_score~0 and make the
    whole universe look like a strong buy."""
    return max(0.0, min(1.0, 2.0 * float(p_up) - 1.0))


# ── live inference: the buy stacker AS combined_buy_score ─────────────────────
# The buy aggregator on the within-day PIVOT-rank label (rank_5d fallback —
# `_label_cfg`), trained on the 21 live method features
# and served at the aggregator's combine point. Native Booster (no sklearn), so
# the artifact loads in the production .venv. Fail-soft everywhere: a missing
# artifact / lightgbm returns None and the caller keeps the weighted combine.

import pickle as _pickle
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path as _Path

# Small-data GBM parameters (2026-08-11 retune): the classifier's lgb defaults
# (31 leaves / min_child 200) were sized for the deep cache, not a ~20k-row
# panel. The 13-arm walk-forward retune (scratchpad stacker_tune.py, 29 OOS
# days, paired per-day diffs) has this config at rank-5d IC +0.0611 vs the
# defaults' +0.0463 (t +2.00, the pre-registered bar) with the best day-
# stability of any arm (ICIR +0.75); two sibling small-capacity arms corroborate
# the direction at t +1.5-1.7. Also measured and REJECTED in the same run: the
# 27-feature extension (the six 2026-08-11 promotions are mostly-NaN through
# the training windows -> distribution shift; revisit when their panel history
# thickens) and all-runs row inflation (20x rows of the same days, t +0.6).
# Used by BOTH stackers AND the calibrator's OOF walk, so the calibration curve
# is fit on the same model class it corrects.
STACKER_GBM_PARAMS = dict(num_leaves=15, min_child_samples=20,
                          learning_rate=0.05, n_estimators=200)


def stacker_model_class() -> str:
    """The configured model class, normalised. Unknown values fall back to
    "logistic" (the measured default) rather than erroring — the revert knob is
    for operators, and a typo must not kill training."""
    v = str(getattr(settings, "stacker_model_class", "logistic")).strip().lower()
    return v if v in ("logistic", "gbm") else "logistic"


def _stacker_model_factory():
    """One factory for BOTH stackers AND the calibrator's OOF walk (the
    calibration curve must be fit on the same model class it corrects).
    "logistic" = SoftmaxLogistic (2026-08-22 default — see the setting's note);
    "gbm" reverts to the small-data LightGBM classifier."""
    if stacker_model_class() == "gbm":
        from src.analysis.ml_train import LightGBMModel
        return LightGBMModel(**STACKER_GBM_PARAMS)
    from src.analysis.ml_train import SoftmaxLogistic
    return SoftmaxLogistic()


_BUY_MODEL_PATH = _Path("cache/ml/ml_buy_model.pkl")
BUY_TRAIN_CONFIG = dict(horizon=5, basis="rank", deadband=0.0)
_BUY_ART: dict = {"mtime": None, "art": None}


def _fit_calibrator(df: pd.DataFrame, horizon: int, basis: str, feats: Sequence[str],
                    deadband: float, min_train_days: int = 8, step_days: int = 2):
    """Fit an ``IsotonicCalibrator`` on WALK-FORWARD OUT-OF-FOLD predictions.

    The OOF requirement is load-bearing, not a nicety: a calibrator fit on the
    model's own training rows learns the curve of a model that has already
    memorised them, which looks calibrated and is not. So this re-runs the same
    point-in-time walk-forward the evaluation uses and calibrates on predictions
    the model never trained on.

    Returns ``(calibrator, diagnostics)``; ``(None, {})`` when the panel is too
    thin to produce usable OOF predictions (caller then stores no calibrator and
    inference falls back to the raw probability)."""
    from src.analysis.ml_train import (IsotonicCalibrator, brier, brier_skill,
                                       make_model_factory, walk_forward_predict)
    try:
        oof = walk_forward_predict(df, horizon, basis, features=list(feats),
                                   deadband=deadband, min_train_days=min_train_days,
                                   step_days=step_days, min_train_rows=1000,
                                   model_factory=_stacker_model_factory)
    except Exception as e:
        logger.warning(f"[ml_calib] walk-forward for calibration failed: {e}")
        return None, {}
    if oof is None or oof.empty or "bull" not in oof.columns:
        return None, {}
    oof = oof[oof["fwd"].notna()]
    if len(oof) < 200:
        logger.warning(f"[ml_calib] only {len(oof)} OOF rows — not calibrating")
        return None, {}
    raw = pd.to_numeric(oof["bull"], errors="coerce").to_numpy(dtype=float)
    # The outcome the model's "bull" class asserts: the basis return cleared the
    # deadband upward. (For the sell model the basis is already negated, so this
    # reads "the short worked" — same code, mirrored label.)
    y = (pd.to_numeric(oof["fwd"], errors="coerce").to_numpy(dtype=float) > deadband).astype(int)
    cal = IsotonicCalibrator().fit(raw, y)
    if cal.n_fit_ == 0:
        return None, {}
    p_cal = cal.transform(raw)
    diag = {"n_oof": int(len(raw)), "base_rate": round(float(y.mean()), 4),
            "brier_raw": round(brier(raw, y), 4), "brier_cal": round(brier(p_cal, y), 4),
            "skill_raw": round(brier_skill(raw, y), 4),
            "skill_cal": round(brier_skill(p_cal, y), 4),
            "raw_range": [round(float(raw.min()), 4), round(float(raw.max()), 4)],
            "cal_range": [round(float(p_cal.min()), 4), round(float(p_cal.max()), 4)]}
    logger.info(f"[ml_calib] fitted on {diag['n_oof']:,} OOF rows — Brier {diag['brier_raw']} → "
                f"{diag['brier_cal']} (skill {diag['skill_raw']:+} → {diag['skill_cal']:+}); "
                f"P range {diag['raw_range']} → {diag['cal_range']}")
    return cal, diag


def _calibrate(art: dict, p: float) -> float:
    """Apply the artifact's calibrator to a raw probability. Fail-soft: no
    calibrator (or the setting off) returns the raw value unchanged."""
    if not settings.enable_ml_probability_calibration:
        return p
    cal = art.get("calibrator")
    if cal is None:
        return p
    try:
        return cal.transform_one(p)
    except Exception:
        return p


def _label_cfg(df: pd.DataFrame, cfg: dict, tag: str):
    """Resolve the training label per ``stacker_label_basis`` (2026-08-12):
    ``pivot_rank`` uses the within-day rank of the signed pivot target whenever
    the panel carries enough settled rows; anything else (or a thin pivot
    column) keeps the legacy fixed-horizon rank label. Returns ``(cfg, ycol)``
    with ``cfg["basis"]`` switched to ``rank_pv`` when the pivot label won."""
    want_pivot = str(getattr(settings, "stacker_label_basis", "pivot_rank")).lower() == "pivot_rank"
    if want_pivot and "fwd_ret_rank_pv" in df.columns:
        n = int(pd.to_numeric(df["fwd_ret_rank_pv"], errors="coerce").notna().sum())
        if n >= 500:
            out = dict(cfg)
            out["basis"] = "rank_pv"
            logger.info(f"[{tag}] label: PIVOT rank ({n:,} settled rows)")
            return out, "fwd_ret_rank_pv"
        logger.info(f"[{tag}] pivot label too thin ({n} rows) — falling back to "
                    f"{cfg['basis']}_{cfg['horizon']}d")
    return dict(cfg), f"fwd_ret_{cfg['basis']}_{cfg['horizon']}d"


def train_and_persist_buy(days: Optional[int] = None, path=_BUY_MODEL_PATH) -> Optional[dict]:
    """Train the buy stacker on the panel (5d market-relative) over the 21 live
    method features; pickle it. Returns the artifact or None."""
    import numpy as _np
    from src.analysis.ml_train import label_from_return
    h = BUY_TRAIN_CONFIG["horizon"]
    df = build_stacker_dataset(horizons=[h], days=days)
    if df.empty:
        logger.warning("[ml_buy] no panel data to train on")
        return None
    cfg, ycol = _label_cfg(df, BUY_TRAIN_CONFIG, "ml_buy")
    # The news basis rides the artifact so SERVING can dispatch on what this
    # model was actually trained with, never on today's setting.
    cfg["news_basis"] = news_basis()
    basis = cfg["basis"]
    if ycol not in df.columns:
        logger.warning("[ml_buy] no panel data to train on")
        return None
    feats = [f for f in STACKER_LIVE_FEATURES if f in df.columns]
    y_raw = df[ycol].map(lambda r: label_from_return(r, cfg["deadband"]))
    keep = y_raw.notna()
    X = df.loc[keep, feats].to_numpy(dtype=float)
    y = y_raw[keep].to_numpy(dtype=int)
    if len(X) < 500 or len(_np.unique(y)) < 2:
        logger.warning(f"[ml_buy] insufficient training rows ({len(X)})")
        return None
    model = _stacker_model_factory().fit(X, y)
    # Calibrate on OUT-OF-FOLD predictions BEFORE the final all-data fit is used
    # live, so the probability the combine consumes means what it says.
    cal, cal_diag = (_fit_calibrator(df, h, cfg["basis"], feats, cfg["deadband"])
                     if settings.enable_ml_probability_calibration else (None, {}))
    art = {"model": model, "features": feats, "config": dict(cfg),
           "model_class": stacker_model_class(),
           "calibrator": cal, "calibration": cal_diag,
           "trained_at": _dt.now(_tz.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        _pickle.dump(art, fh)
    _BUY_ART.update(mtime=None, art=None)
    _record_buy_registry(art)
    logger.info(f"[ml_buy] trained on {art['n_train']:,} rows (<= {art['train_max_date']}) -> {path}")
    return art


def _record_buy_registry(art: dict) -> None:
    try:
        import json
        from src.db.connection import connect
        with connect() as con:
            con.execute(
                "INSERT INTO ml_models (trained_at, method, model_type, horizon, basis, "
                "n_train, train_max_date, features, config) VALUES (?,?,?,?,?,?,?,?,?)",
                [art["trained_at"], "ml_buy", stacker_model_class(), int(art["config"]["horizon"]),
                 art["config"]["basis"], art["n_train"], art["train_max_date"],
                 json.dumps(art["features"]), json.dumps(art["config"])])
    except Exception as e:
        logger.debug(f"[ml_buy] registry write skipped: {e}")


def _load_buy_artifact() -> Optional[dict]:
    if not _BUY_MODEL_PATH.exists():
        return None
    try:
        mt = _BUY_MODEL_PATH.stat().st_mtime_ns
        if _BUY_ART["mtime"] == mt:
            return _BUY_ART["art"]
        with open(_BUY_MODEL_PATH, "rb") as fh:
            art = _pickle.load(fh)
        _BUY_ART.update(mtime=mt, art=art)
        return art
    except Exception as e:
        logger.debug(f"[ml_buy] artifact load failed: {e}")
        _BUY_ART.update(mtime=None, art=None)
        return None


def compute_buy_conviction(method_scores: dict, ranked: Optional[dict] = None) -> Optional[float]:
    """The stacker's buy conviction ∈ [0,1] for one ticker — the replacement for
    the weighted combined_buy_score. ``method_scores`` must be the DAILY method
    scores (the panel's basis the model trained on), keyed by method name. Returns
    None when the artifact/lightgbm is unavailable, so the caller keeps the
    weighted combine — an invisible degradation is impossible."""
    art = _load_buy_artifact()
    if art is None:
        return None
    try:
        import numpy as _np
        x = _np.array([[_feature_value(f, method_scores, ranked, art)
                        for f in art["features"]]], dtype=float)
        bull, _bear = art["model"].bull_bear(x)
        return buy_conviction_from_proba(_calibrate(art, float(bull[0])))
    except Exception as e:
        logger.debug(f"[ml_buy] conviction failed: {e}")
        return None


def eod_train_buy() -> Optional[dict]:
    """EOD entry point — retrain the buy stacker on the latest panel. Fail-soft."""
    return train_and_persist_buy()


def reset_buy_caches() -> None:
    """Test hook — drop the artifact memo."""
    _BUY_ART.update(mtime=None, art=None)


# ── the SELL stacker — symmetric to buy, AS combined_sell_score ───────────────
# Predicts P(the SHORT works) = P(the stock underperforms the benchmark at 5d),
# by training on the NEGATED market-relative return (so class "up" = short won).
# A separately-trained artifact so the two sides can diverge (per-side skill),
# even though with deadband 0 P(short works) is the complement of the buy model's
# P(up). Replaces the WEIGHTED combined_sell_score; the real question the eval
# answers is whether it beats THAT (unrelated to the buy model).

_SELL_MODEL_PATH = _Path("cache/ml/ml_sell_model.pkl")
SELL_TRAIN_CONFIG = dict(horizon=5, basis="rank", deadband=0.0)
_SELL_ART: dict = {"mtime": None, "art": None}


def train_and_persist_sell(days: Optional[int] = None, path=_SELL_MODEL_PATH) -> Optional[dict]:
    """Train the sell stacker on the panel (5d market-relative, label NEGATED so
    'up' = the short worked) over the 21 live method features; pickle it."""
    import numpy as _np
    from src.analysis.ml_train import label_from_return
    h = SELL_TRAIN_CONFIG["horizon"]
    df = build_stacker_dataset(horizons=[h], days=days)
    if df.empty:
        logger.warning("[ml_sell] no panel data to train on")
        return None
    cfg, ycol = _label_cfg(df, SELL_TRAIN_CONFIG, "ml_sell")
    cfg["news_basis"] = news_basis()
    basis = cfg["basis"]
    if ycol not in df.columns:
        logger.warning("[ml_sell] no panel data to train on")
        return None
    feats = [f for f in STACKER_LIVE_FEATURES if f in df.columns]
    # NEGATE the return: the short's outcome. class 2 ("up") now means the stock
    # FELL market-relative = the short worked. bull_bear's `bull` = P(short worked).
    y_raw = df[ycol].map(lambda r: label_from_return(-r, cfg["deadband"]))
    keep = y_raw.notna()
    X = df.loc[keep, feats].to_numpy(dtype=float)
    y = y_raw[keep].to_numpy(dtype=int)
    if len(X) < 500 or len(_np.unique(y)) < 2:
        logger.warning(f"[ml_sell] insufficient training rows ({len(X)})")
        return None
    model = _stacker_model_factory().fit(X, y)
    # Calibrate on OOF predictions of the SELL problem: walk_forward_predict needs
    # the negated-return basis column ("the short worked"), matching this model's label.
    cal, cal_diag = (None, {})
    if settings.enable_ml_probability_calibration:
        # The SHORT's outcome is the NEGATED basis (below the day's median on the
        # rank label = the short worked), matching this model's own label.
        if basis == "rank_pv":
            df["fwd_ret_sellinv_pv"] = -pd.to_numeric(df[ycol], errors="coerce")
            cal, cal_diag = _fit_calibrator(df, h, "sellinv_pv", feats, cfg["deadband"])
        else:
            scol = f"fwd_ret_sellinv_{h}d"
            df[scol] = -pd.to_numeric(df[ycol], errors="coerce")
            cal, cal_diag = _fit_calibrator(df, h, "sellinv", feats, cfg["deadband"])
    art = {"model": model, "features": feats, "config": dict(cfg),
           "model_class": stacker_model_class(),
           "calibrator": cal, "calibration": cal_diag,
           "trained_at": _dt.now(_tz.utc).isoformat(timespec="seconds"),
           "n_train": int(len(X)), "train_max_date": str(df["signal_date"].max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        _pickle.dump(art, fh)
    _SELL_ART.update(mtime=None, art=None)
    _record_sell_registry(art)
    logger.info(f"[ml_sell] trained on {art['n_train']:,} rows (<= {art['train_max_date']}) -> {path}")
    return art


def _record_sell_registry(art: dict) -> None:
    try:
        import json
        from src.db.connection import connect
        with connect() as con:
            con.execute(
                "INSERT INTO ml_models (trained_at, method, model_type, horizon, basis, "
                "n_train, train_max_date, features, config) VALUES (?,?,?,?,?,?,?,?,?)",
                [art["trained_at"], "ml_sell", stacker_model_class(), int(art["config"]["horizon"]),
                 art["config"]["basis"], art["n_train"], art["train_max_date"],
                 json.dumps(art["features"]), json.dumps(art["config"])])
    except Exception as e:
        logger.debug(f"[ml_sell] registry write skipped: {e}")


def _load_sell_artifact() -> Optional[dict]:
    if not _SELL_MODEL_PATH.exists():
        return None
    try:
        mt = _SELL_MODEL_PATH.stat().st_mtime_ns
        if _SELL_ART["mtime"] == mt:
            return _SELL_ART["art"]
        with open(_SELL_MODEL_PATH, "rb") as fh:
            art = _pickle.load(fh)
        _SELL_ART.update(mtime=mt, art=art)
        return art
    except Exception as e:
        logger.debug(f"[ml_sell] artifact load failed: {e}")
        _SELL_ART.update(mtime=None, art=None)
        return None


def compute_sell_conviction(method_scores: dict, ranked: Optional[dict] = None) -> Optional[float]:
    """The stacker's SELL conviction ∈ [0,1] for one ticker — the replacement for
    the weighted combined_sell_score. ``bull`` = P(short worked); centered the
    same way as the buy side. None when the artifact/lightgbm is unavailable, so
    the caller keeps the weighted combine."""
    art = _load_sell_artifact()
    if art is None:
        return None
    try:
        import numpy as _np
        x = _np.array([[_feature_value(f, method_scores, ranked, art)
                        for f in art["features"]]], dtype=float)
        bull, _bear = art["model"].bull_bear(x)           # bull = P(short worked)
        return buy_conviction_from_proba(_calibrate(art, float(bull[0])))
    except Exception as e:
        logger.debug(f"[ml_sell] conviction failed: {e}")
        return None


def eod_train_sell() -> Optional[dict]:
    """EOD entry point — retrain the sell stacker on the latest panel. Fail-soft."""
    return train_and_persist_sell()


def reset_sell_caches() -> None:
    """Test hook — drop the sell artifact memo."""
    _SELL_ART.update(mtime=None, art=None)


def validate_swap(horizon: int = 5, basis: str = "rel", days: Optional[int] = None,
                  deadband: float = 0.0, min_train_days: int = 8, step_days: int = 2,
                  features: Optional[Sequence[str]] = None) -> dict:
    """What does REPLACING combined_buy_score with the stacker do to the DECISIONS?

    Walk-forward the stacker, form ``swapped_combined = max(0, 2*P(up)-1) -
    combined_sell_score``, and compare the BUY decisions (combined > threshold) it
    would produce — count AND forward return — against the current weighted
    combine's, on the same OOS rows. The gate before letting it drive live orders:
    a sane buy count (not the whole universe, not zero) and buys that beat the
    current combine's.
    """
    from config.settings import settings as S
    from src.analysis.ml_train import make_model_factory, walk_forward_predict
    thr = float(S.buy_sell_diff_threshold)
    feats = list(features) if features is not None else STACKER_FEATURES
    df = build_stacker_dataset(horizons=[horizon], days=days)
    if df.empty:
        return {}
    preds = walk_forward_predict(df, horizon, basis, features=feats,
                                 deadband=deadband, min_train_days=min_train_days,
                                 step_days=step_days, min_train_rows=1000,
                                 model_factory=make_model_factory("gbm"))
    if preds.empty:
        return {}
    # preds already carries the rel forward return (its `fwd` column, since
    # basis="rel"); only pull fwd_raw + the combine columns from df, to avoid a
    # name collision on the rel column.
    keep = df[["signal_date", "ticker", "combined_score", "combined_sell_score",
               f"fwd_ret_raw_{horizon}d"]]
    m = preds.merge(keep, on=["signal_date", "ticker"], how="left")
    m["swapped_buy"] = (2.0 * m["bull"] - 1.0).clip(0.0, 1.0)
    sell = pd.to_numeric(m["combined_sell_score"], errors="coerce").fillna(0.0)
    m["swapped_combined"] = m["swapped_buy"] - sell
    old = pd.to_numeric(m["combined_score"], errors="coerce")
    fwd_raw = pd.to_numeric(m[f"fwd_ret_raw_{horizon}d"], errors="coerce")
    fwd_rel = pd.to_numeric(m["fwd"], errors="coerce")

    def stats(mask: pd.Series) -> dict:
        mask = mask.fillna(False)
        return {"n": int(mask.sum()), "pct": round(100.0 * float(mask.mean()), 1),
                "fwd_raw": round(float(fwd_raw[mask].mean()), 4) if mask.any() else None,
                "fwd_rel": round(float(fwd_rel[mask].mean()), 4) if mask.any() else None,
                "win_rel": round(100.0 * float((fwd_rel[mask] > 0).mean()), 1) if mask.any() else None}

    old_buy, new_buy = old > thr, m["swapped_combined"] > thr
    return {"horizon": horizon, "basis": basis, "threshold": thr, "n_rows": int(len(m)),
            "universe": stats(pd.Series(True, index=m.index)),
            "old_buys": stats(old_buy), "new_buys": stats(new_buy),
            "overlap": int((old_buy & new_buy).sum())}


def _print_swap(r: dict) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not r:
        return
    print(f"\nSWAP VALIDATION — replace combined_buy_score with the stacker (h={r['horizon']}, "
          f"basis={r['basis']}, buy threshold={r['threshold']})")
    print(f"OOS rows: {r['n_rows']:,}   |   buy = combined_score > threshold\n")
    head = f"{'decision set':<16}{'n':>8}{'% univ':>8}{'fwd_raw%':>10}{'fwd_rel%':>10}{'win_rel%':>10}"
    print(head); print("-" * len(head))
    for name, key in (("universe", "universe"), ("OLD combine buys", "old_buys"),
                      ("NEW stacker buys", "new_buys")):
        s = r[key]
        def f(v, w, fmt):
            return f"{format(v, fmt):>{w}}" if v is not None else f"{'—':>{w}}"
        print(f"{name:<16}{s['n']:>8}{s['pct']:>7}%{f(s['fwd_raw'],10,'+.3f')}"
              f"{f(s['fwd_rel'],10,'+.3f')}{f(s['win_rel'],10,'.1f')}")
    print("-" * len(head))
    print(f"overlap (both buy the same name): {r['overlap']}")
    print("\nGATE to flip live: NEW buys should be a sane count (not ~0, not the whole universe) "
          "AND beat the OLD combine's buys on fwd_rel. If it floods or its buys are worse, DON'T enable.")


def validate_sell_swap(horizon: int = 5, days: Optional[int] = None, deadband: float = 0.0,
                       min_train_days: int = 8, step_days: int = 2,
                       features: Optional[Sequence[str]] = None) -> dict:
    """What does REPLACING combined_sell_score with the sell stacker do to the SELL
    DECISIONS? Symmetric to ``validate_swap``: walk-forward the sell stacker, form
    ``swapped_combined = combined_buy - max(0, 2*P(short works)-1)``, and compare
    the SELL decisions (combined < −threshold) — count AND the SHORT's return
    (``−fwd_ret_rel``, so a sell wins when the stock underperforms) — against the
    weighted combine's. A sane sell count that beats the weighted sell combine on
    the short return is the gate to flip live."""
    from config.settings import settings as S
    from src.analysis.ml_train import make_model_factory, walk_forward_predict
    thr = float(S.buy_sell_diff_threshold)
    feats = list(features) if features is not None else STACKER_LIVE_FEATURES
    df = build_stacker_dataset(horizons=[horizon], days=days)
    if df.empty:
        return {}
    # The SHORT's outcome = the NEGATED market-relative return (a short profits
    # when the stock underperforms). Train + evaluate the stacker on it.
    scol = f"fwd_ret_sellrel_{horizon}d"
    df[scol] = -pd.to_numeric(df[f"fwd_ret_rel_{horizon}d"], errors="coerce")
    preds = walk_forward_predict(df, horizon, "sellrel", features=feats, deadband=deadband,
                                 min_train_days=min_train_days, step_days=step_days,
                                 min_train_rows=1000, model_factory=make_model_factory("gbm"))
    if preds.empty:
        return {}
    keep = df[["signal_date", "ticker", "combined_score", "combined_buy_score"]]
    m = preds.merge(keep, on=["signal_date", "ticker"], how="left")
    m["swapped_sell"] = (2.0 * m["bull"] - 1.0).clip(0.0, 1.0)     # bull = P(short works)
    buy = pd.to_numeric(m["combined_buy_score"], errors="coerce").fillna(0.0)
    m["swapped_combined"] = buy - m["swapped_sell"]
    old = pd.to_numeric(m["combined_score"], errors="coerce")
    short_ret = pd.to_numeric(m["fwd"], errors="coerce")          # the short's return (= −fwd_rel)

    def stats(mask: pd.Series) -> dict:
        mask = mask.fillna(False)
        return {"n": int(mask.sum()), "pct": round(100.0 * float(mask.mean()), 1),
                "short_ret": round(float(short_ret[mask].mean()), 4) if mask.any() else None,
                "win": round(100.0 * float((short_ret[mask] > 0).mean()), 1) if mask.any() else None}

    old_sell, new_sell = old < -thr, m["swapped_combined"] < -thr
    return {"horizon": horizon, "threshold": thr, "n_rows": int(len(m)),
            "universe": stats(pd.Series(True, index=m.index)),
            "old_sells": stats(old_sell), "new_sells": stats(new_sell),
            "overlap": int((old_sell & new_sell).sum())}


def _print_sell_swap(r: dict) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if not r:
        return
    print(f"\nSELL SWAP VALIDATION — replace combined_sell_score with the sell stacker "
          f"(h={r['horizon']}, threshold={r['threshold']})")
    print(f"OOS rows: {r['n_rows']:,}   |   sell = combined_score < −threshold;  "
          f"short_ret = −fwd_rel (a sell WINS when the stock underperforms)\n")
    head = f"{'decision set':<16}{'n':>8}{'% univ':>8}{'short_ret%':>12}{'win%':>8}"
    print(head); print("-" * len(head))
    for name, key in (("universe", "universe"), ("OLD combine sells", "old_sells"),
                      ("NEW stacker sells", "new_sells")):
        s = r[key]
        def f(v, w, fmt):
            return f"{format(v, fmt):>{w}}" if v is not None else f"{'—':>{w}}"
        print(f"{name:<16}{s['n']:>8}{s['pct']:>7}%{f(s['short_ret'],12,'+.3f')}{f(s['win'],8,'.1f')}")
    print("-" * len(head))
    print(f"overlap (both sell the same name): {r['overlap']}")
    print("\nGATE: NEW sells a sane count AND beat the OLD combine's sells on short_ret. NOTE a "
          "bearish window inflates BOTH (shorts win when everything falls) — the GAP is the signal.")


def _print(table: pd.DataFrame, model_name: str) -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if table is None or table.empty:
        print("No stacker results — the panel likely has too few forward-labelled days yet.")
        return
    print(f"\nFULL STACKER (ml_buy) — walk-forward OOS on the signals panel, model={model_name}")
    print("Features = all method scores (combined_score EXCLUDED for circularity; shown as baseline).")
    print("The learned stacker must BEAT combined_score to be worth more than the hand-weighted combine.\n")
    head = f"{'model':<18}{'basis':>6}{'h':>4}{'n':>8}{'IC':>9}{'ICIR':>8}{'hit%':>8}{'simret%':>9}"
    print(head); print("-" * len(head))

    def f(v, w, s):
        return f"{format(v, s):>{w}}" if v is not None and pd.notna(v) else f"{'—':>{w}}"

    for (basis, h), g in table.groupby(["basis", "horizon"], sort=True):
        for _, r in g.iterrows():
            line = f"{r['model']:<18}{basis:>6}{int(h):>4}{int(r['n']):>8}"
            line += f(r['ic'], 9, '+.4f') + f(r['icir'], 8, '+.3f')
            line += f(r['hit'], 8, '.2f') + f(r['simret'], 9, '+.4f')
            print(line)
        print("-" * len(head))
    print("\nPANEL-ONLY, ~34 days of ONE regime — high overfitting risk, so this is a directional "
          "read. GO only if the stacker's rel-basis IC/ICIR clears combined_score's; otherwise the "
          "hand-weighted combine is already capturing what the methods jointly say.")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Full stacker (ml_buy) — learned buy-side aggregator")
    p.add_argument("--horizons", default="1,5,10")
    p.add_argument("--bases", default="raw,rel")
    p.add_argument("--model", default="gbm", choices=("logistic", "gbm"))
    p.add_argument("--deadband", type=float, default=0.0)
    p.add_argument("--min-train-days", type=int, default=8)
    p.add_argument("--step-days", type=int, default=2)
    p.add_argument("--validate-swap", action="store_true",
                   help="measure the DECISION impact of replacing combined_buy_score with the stacker")
    p.add_argument("--swap-horizon", type=int, default=5, help="horizon the buy aggregator optimizes (default 5)")
    p.add_argument("--train", action="store_true",
                   help="retrain + persist BOTH live artifacts (ml_buy, ml_sell) instead of measuring")
    a = p.parse_args(argv)
    horizons = tuple(int(h) for h in str(a.horizons).split(",") if h.strip())
    bases = tuple(b for b in str(a.bases).split(",") if b.strip())
    from src.db import repo
    if a.train:
        # Deliberately BEFORE set_read_only: training appends to the `ml_models`
        # registry, so this branch needs the write path. Both sides are trained
        # together — they are one decision surface (buy/sell camps of the same
        # combine) and shipping a mismatched pair is never what you want.
        for _name, _fn in (("ml_buy", eod_train_buy), ("ml_sell", eod_train_sell)):
            art = _fn()
            print(f"{_name}: " + (f"{art['n_train']:,} rows (<= {art['train_max_date']}), "
                                  f"label basis={art['config']['basis']}"
                                  if art else "NO ARTIFACT (see log)"))
        return
    repo.set_read_only(True)
    if a.validate_swap:
        r = validate_swap(horizon=a.swap_horizon, basis="rel", deadband=a.deadband,
                          min_train_days=a.min_train_days, step_days=a.step_days)
        _print_swap(r)
        return
    table = measure(horizons=horizons, bases=bases, deadband=a.deadband, model_name=a.model,
                    min_train_days=a.min_train_days, step_days=a.step_days)
    _print(table, a.model)


if __name__ == "__main__":
    main()
