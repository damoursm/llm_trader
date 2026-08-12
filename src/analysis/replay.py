"""Replay historical ticker-days through the CURRENT scorers.

Purpose: calibrations must fit data the current code produced. Excluding
superseded rows (the epoch masks) achieves that by throwing evidence away;
REPLAYING achieves it by regenerating the evidence, which is strictly better
wherever it is faithful.

It is faithful for the OHLCV-derived methods. Measured 2026-07-27 over the FULL
materialised panel — 293,237 ticker-days, joined run-exact — replaying the
current scorers over the cached daily bars reproduces the stored value with a
median error of **0.0000**:

    method           exact   <=0.01   median
    iv_rank          86.3%    87.1%   0.0000
    trend_strength   82.5%    85.4%   0.0000
    momentum         82.1%    83.6%   0.0000
    tech             82.0%    84.7%   0.0000
    vwap             75.4%    84.6%   0.0000
    money_flow        6.4%     9.4%   0.2150   <- scorer CHANGED; see below

**Fidelity decays with age**, so quote the span: over the last two days `tech`
reproduces at **97.7%**, over the full five weeks at 82.0%. The residue is the
cache being retroactively split-adjusted plus per-ticker start dates that shift
the series-length-dependent normalisations — real, irreducible, and far cheaper
than the alternative of discarding the row.

**Truncation is the whole game, and getting it wrong looks like unreproducible
data.** Two earlier attempts failed for that reason alone:

  * `bars <= signal_date`  -> ~30% exact (median error 0.013-0.089). Wrong
    because the live pipeline DROPS the still-forming daily bar
    (`market_data._drop_forming_bar`), so an intraday run never saw that day's
    bar at all.
  * `bars < signal_date`   -> ~68% exact, median 0.0000. Better, but wrong for
    runs that happened AFTER the 16:00 ET close, which legitimately did see it.
  * time-aware (below)     -> 92-93%. The run's own ``generated_at`` decides.

So the replay must use each row's RUN TIMESTAMP, not just its date. An earlier
claim in this project that scores were only "~62% reproducible" was measured
with the first, wrong truncation.

The same "which run?" question governs the JOIN, and getting it wrong is worse
than getting the truncation wrong because it corrupts silently: `signals` holds
~43 runs per ticker-day, so joining a replayed value on (date, ticker) alone
grafts an arbitrary run's score onto the panel's row. Measured on live data,
that changed **56.8%** of `tech` values — a scorer that had not changed at all —
against **2.3%** with run-exact matching. `restore_replayed` therefore keys on
``generated_at``, and the control moving when it has no reason to is the tell.

Scope — what replay can and cannot regenerate:

  REPLAYABLE from cached OHLCV (21 methods): tech, vwap, momentum, money_flow,
  trend_strength, iv_rank, pattern, sector_momentum, market_momentum, hi52,
  mom_12_1, st_reversal, squeeze, avwap, resid_mom, vol_profile, coint, and the
  kaufman/adx pair.

  NOT REPLAYABLE — needs a point-in-time feed nobody stored: news and
  sent_velocity (historical headlines + LLM scoring), the options family
  (put_call, max_pain, oi_skew, iv_expr, iv_term), insider, pead, ext_gap,
  massive, broker_advisor.

Circularity — the one real constraint, and it applies only to DERIVED values:
method scores do not depend on the weights, so replaying them and then
calibrating weights from the result is sound (inputs regenerated, outputs
re-derived). `combined_score`, `confidence` and the gate outcomes DO depend on
the weights, so replaying those with today's weights produces a BACKTEST — fine
to inspect, invalid to re-calibrate those same weights from. This module
therefore replays SCORES; anything derived is opt-in and marked.

Writes to `signals_replay`, never to `signals`: the live panel records what
actually happened and stays the audit trail.

CLI:  python -m src.analysis.replay --validate           # fidelity report
      python -m src.analysis.replay --days 30 --write    # populate the table
"""

from __future__ import annotations

from datetime import timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
# loguru, not stdlib logging: the project configures loguru sinks only, so a
# stdlib logger here emits NOTHING. That silently muted this module's skip
# counts and — worse — the fail-soft WARNING that exists precisely so a broken
# restore is audible rather than looking like an unpopulated table.
from loguru import logger

_ET = timezone(timedelta(hours=-4))          # matches the pipeline's ET handling
_MARKET_CLOSE = pd.Timestamp("16:00").time()

# Methods that reproduce faithfully — the `signals_replay` columns. See the
# schema constant for why `pattern` and `sector_momentum` are excluded despite
# being OHLCV-driven (both carry state the replay cannot rewind).
from src.db.schema import REPLAYABLE_METHOD_COLUMNS as REPLAYABLE  # noqa: E402

# Minimum bars before a scorer is asked for a view (matches the scorers' own
# guards closely enough that a short series yields no view rather than noise).
_MIN_BARS = 60

# Split guard. The OHLCV cache is adjusted RETROACTIVELY, so for a ticker that
# split AFTER a signal date the cached bars carry information that date could
# not have had — the one genuinely future-facing input a replay cannot scrub,
# since the unadjusted history was never stored. It IS detectable: `signals.price`
# was recorded live on that day's own scale, so a large disagreement with the
# cached close for the same day means the cache has since been rescaled.
#
# Measured 2026-07-27 over 243,136 checkable rows: 496 (0.20%, 26 tickers) are
# rescaled, and their replay fidelity collapses — tech 36.3% exact vs 87.9% on
# unaffected rows, vwap 23.2% vs 79.7%. Those rows are therefore NOT replayed;
# the epoch mask handles them, which is the correct conservative fallback.
#
# Band is tighter than tracker's 0.5/2.0 because the jobs differ: that guard
# protects a RETURN from a fabricated split ratio and only needs to catch >=2x,
# whereas this one protects SCORE fidelity and should reject any rescaling.
# Observed snapshot-vs-close noise is 0.92-1.06 at the 1st/99th percentile, so
# 0.67/1.5 sits far outside quote noise and below the smallest real split.
_SPLIT_GUARD_LO, _SPLIT_GUARD_HI = 0.67, 1.5

# Series-integrity bound, the SECOND split detector — and the one that catches
# what the price check cannot. `_scale_is_consistent` needs a live-recorded
# price, and a ticker with NO priced row anywhere in the panel is undetectable
# by it (ADTX: all 7 rows NaN price, a serial reverse-splitter). But a split
# leaves a second fingerprint in the bars themselves: ATR is a 14-day average of
# true range over price, so for it to EXCEED the price the series must contain a
# discontinuity — sustained daily ranges equal to the entire price are not a
# thing a real tape does. Measured over 302k replayed rows: median atr_pct
# 0.0387, p99 0.157, p99.9 0.524, then a jump to 108.0. The bound is set at 1.0
# — roughly 2x beyond the 99.9th percentile — and rejects 139 rows (0.046%) over
# 8 tickers. A corrupt series invalidates the SCORES as much as the context, so
# the whole row is skipped rather than just its market conditions.
_MAX_PLAUSIBLE_ATR_PCT = 1.0


def _series_is_intact(ctx: Dict[str, float]) -> bool:
    """False when the recovered values themselves prove the bars are broken."""
    atr = ctx.get("atr_pct")
    if atr is None or atr != atr:
        return True                     # nothing to judge => fail open
    return abs(float(atr)) <= _MAX_PLAUSIBLE_ATR_PCT


def _scale_is_consistent(df: pd.DataFrame, signal_date: str,
                         recorded_price: Optional[float]) -> bool:
    """Is the cached price scale for this ticker-day still the one the run saw?

    Compares the LIVE-recorded snapshot price against the cached close for the
    same date. This is a data-integrity check on the SCALE only — the close is
    never fed to a scorer — so it introduces no look-ahead into the scores.
    Unknown/unavailable => True (fail-open): the fidelity table already reports
    the residue, and refusing to replay on a missing price would silently gut
    coverage.
    """
    # NaN must be checked EXPLICITLY: `not nan` is False and `nan <= 0` is False,
    # so a NaN price slips past both and then fails `lo < nan < hi`, rejecting the
    # row. That inverted this guard from fail-open to fail-CLOSED and silently
    # refused 41,807 rows (13.5% of the panel) whose price was simply NULL —
    # reported as "rescaled" when nothing had been rescaled at all.
    if recorded_price is None or recorded_price != recorded_price:
        return True
    if recorded_price <= 0 or df is None or df.empty:
        return True
    try:
        key = str(signal_date)[:10]
        idx = pd.Series(df.index).astype(str).str[:10]
        hit = df.loc[(idx == key).values, "Close"]
        if hit.empty:
            return True
        close = float(hit.iloc[-1])
        if close <= 0:
            return True
        ratio = float(recorded_price) / close
        return _SPLIT_GUARD_LO < ratio < _SPLIT_GUARD_HI
    except Exception:
        return True


def visible_history(df: pd.DataFrame, signal_date: str, generated_at) -> Optional[pd.DataFrame]:
    """The OHLCV the pipeline ACTUALLY had when it scored this row.

    Replicates `market_data._drop_forming_bar` for the daily interval: the
    signal-date bar exists only for a run that happened at or after the 16:00 ET
    close on that date. Getting this wrong is what makes replay look impossible
    — it costs ~60 percentage points of exact reproduction.
    """
    if df is None or df.empty:
        return None
    try:
        gen = pd.Timestamp(generated_at)
        if gen.tzinfo is None:
            gen = gen.tz_localize("UTC")
        et = gen.tz_convert(_ET)
        closed = (et.time() >= _MARKET_CLOSE
                  and et.date().isoformat() == str(signal_date)[:10])
    except Exception:
        closed = False                        # unknown time => assume forming
    idx = pd.to_datetime(pd.Series(df.index).astype(str).str[:10])
    cut = pd.Timestamp(str(signal_date)[:10])
    keep = (idx <= cut).values if closed else (idx < cut).values
    hist = df[keep]
    return hist if len(hist) >= _MIN_BARS else None


def replay_row(ticker: str, signal_date: str, generated_at,
               methods=REPLAYABLE, df: Optional[pd.DataFrame] = None,
               hist: Optional[pd.DataFrame] = None,
               tech_result=None) -> Dict[str, float]:
    """Current scorers' output for one historical ticker-day. ``{}`` if the
    cached history is too short or unavailable.

    ``hist`` lets a caller that has ALREADY truncated pass the visible window
    straight through. `visible_history` string-converts the whole index, which
    at ~300k rows is a material share of a full materialisation — computing it
    twice per row (once here, once for the context) doubled it.
    """
    from src.data.cache import load_ohlcv
    from src.signals.multi_timeframe import _score_one

    if hist is None:
        frame = df if df is not None else load_ohlcv(ticker)
        hist = visible_history(frame, signal_date, generated_at)
    if hist is None:
        return {}
    out: Dict[str, float] = {}
    for m in methods:
        try:
            # `_score_one("tech", …)` IS `compute_technical_score(...).score`, so
            # a shared result is the identical value — pinned by a drift test so
            # this shortcut cannot diverge from the live scoring path.
            if m == "tech" and tech_result is not None:
                v = float(tech_result.score)
            else:
                v = _score_one(m, ticker, hist, "1d")
        except Exception:
            continue
        if v is not None:
            out[m] = round(float(v), 6)
    return out


def replay_context(ticker: str, hist: pd.DataFrame,
                   tech_result=None) -> Dict[str, float]:
    """Market-condition values for one historical ticker-day.

    These come from the SAME cached OHLCV as the method scores and were
    previously computed and thrown away: `atr_pct`, `bb_width_pct` and
    `vol_ratio` are fields of the very `compute_technical_score` result that
    yields the `tech` score, and `tape_score` is the cache-only tape composite
    (passed ``df=`` so it is truncated like everything else — it would otherwise
    read TODAY's full history and be a silent look-ahead).

    `movement_factor` is the ONE confidence component that does not depend on
    the weights, so it is a genuine recovery rather than a backtest. It omits
    the dealer-gamma (GEX) modifier, which is options data and unreplayable —
    a bounded 0.85/1.15 term that is 1.0 whenever gamma is neither PINNED nor
    AMPLIFIED. Measured 91.8% exact against the captured values.

    Partial results are fine: each half is independent, so a tape failure still
    yields the technical values and vice versa.

    ``tech_result`` lets a caller share the technical result it already has —
    the `tech` SCORE comes from the same `compute_technical_score` call, and
    computing it twice per row was measured at 3.33 ms of a 22.1 ms row (~15%
    of a full materialisation) for no benefit. `tests/test_replay.py` pins that
    the shared object still yields the score `_score_one` would have produced,
    so the shortcut cannot drift from the live scoring path.
    """
    out: Dict[str, float] = {}
    if hist is None or hist.empty:
        return out
    try:
        from src.analysis.technical import compute_technical_score
        from src.signals.aggregator import _movement_factor

        t = (tech_result if tech_result is not None
             else compute_technical_score(ticker, df=hist))
        atr, bbw, vr = t.atr_pct, t.bb_width_pct, t.vol_ratio
        if atr is not None and atr == atr:
            out["atr_pct"] = round(float(atr), 6)
        if bbw is not None and bbw == bbw:
            out["bb_width_pct"] = round(float(bbw), 6)
        if vr is not None and vr == vr:
            out["vol_ratio"] = round(float(vr), 6)
        if "atr_pct" in out and "bb_width_pct" in out:
            mv = _movement_factor(atr, bbw)          # GEX modifier absent => 1.0
            out["movement_factor"] = round(max(0.70, min(1.30, float(mv))), 3)
    except Exception as e:
        logger.debug(f"[replay] {ticker} technical context failed: {e}")
    try:
        from src.signals.agreement import compute_tape_confirmation
        tape = compute_tape_confirmation(ticker, df=hist)
        if tape is not None and tape.score == tape.score:
            out["tape_score"] = round(float(tape.score), 6)
    except Exception as e:
        logger.debug(f"[replay] {ticker} tape context failed: {e}")
    return out


def validate(sample: int = 500, since: str = "2026-07-01",
             methods=REPLAYABLE) -> pd.DataFrame:
    """Fidelity report: replayed vs stored, per method.

    Run this after ANY scorer change. A method whose exact-match rate collapses
    is telling you either that the scorer changed (expected — that is what a
    replay is for) or that the replay's input reconstruction has drifted.
    """
    import numpy as np
    from src.data.cache import load_ohlcv
    from src.db import repo

    cols = ", ".join(methods)
    try:
        rows = repo.fetch_df(
            f"""SELECT signal_date, ticker, generated_at, {cols} FROM signals
                WHERE signal_date >= ? AND tech IS NOT NULL
                ORDER BY random() LIMIT ?""", [since, int(sample)])
    except Exception as e:
        logger.warning(f"[replay] cannot read signals: {e}")
        return pd.DataFrame()
    if rows is None or rows.empty:
        return pd.DataFrame()

    errs: Dict[str, list] = {m: [] for m in methods}
    for _, r in rows.iterrows():
        got = replay_row(r["ticker"], r["signal_date"], r["generated_at"], methods,
                         df=load_ohlcv(r["ticker"]))
        for m, v in got.items():
            stored = r.get(m)
            if stored is None or pd.isna(stored):
                continue
            errs[m].append(abs(v - float(stored)))

    out = []
    for m in methods:
        e = np.array(errs[m], dtype=float)
        if not len(e):
            continue
        out.append({"method": m, "n": len(e),
                    "exact_pct": round(100.0 * float((e < 1e-9).mean()), 1),
                    "within_001_pct": round(100.0 * float((e <= 0.01).mean()), 1),
                    "median_err": round(float(np.median(e)), 5),
                    "max_err": round(float(e.max()), 4)})
    return pd.DataFrame(out)


def replay_panel(days: Optional[int] = None, methods=REPLAYABLE,
                 limit: Optional[int] = None,
                 with_context: bool = True) -> pd.DataFrame:
    """Replay every stored ticker-day (optionally the last ``days``).

    One row per (signal_date, ticker) with the CURRENT scorers' values. Cached
    per ticker so each OHLCV frame is parsed once regardless of how many rows
    reference it.
    """
    from src.data.cache import load_ohlcv
    from src.db import repo

    where, params = "", []
    if days:
        where = "WHERE signal_date >= (CURRENT_DATE - INTERVAL (?) DAY)::VARCHAR"
        params = [int(days)]
    sql = (f"SELECT signal_date, ticker, run_id, generated_at, price FROM signals {where} "
           f"ORDER BY signal_date, ticker")
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = repo.fetch_df(sql, params)
    if rows is None or rows.empty:
        return pd.DataFrame()

    frames: Dict[str, Optional[pd.DataFrame]] = {}

    def _frame(tk: str) -> Optional[pd.DataFrame]:
        if tk not in frames:
            try:
                frames[tk] = load_ohlcv(tk)
            except Exception:
                frames[tk] = None
        return frames[tk]

    # Pass 1 — per-ticker rescale cutoff.
    #
    # The per-row check needs a live-recorded price, and 13.8% of rows have none,
    # so on its own it leaves a hole: a row with no price on a ticker that DID
    # split sails through unchecked (measured: 179 rows). A split contaminates
    # only the dates BEFORE it — after it, cache and record agree again — so the
    # fix is a per-ticker cutoff rather than banning the ticker outright, which
    # would drop 3,893 rows (1.29% of the panel) to fix 179 (0.06%) and throw
    # away clean post-split history for no correctness gain.
    cutoff: Dict[str, str] = {}
    for _, r in rows.iterrows():
        px = r.get("price")
        if px is None or px != px:
            continue
        if not _scale_is_consistent(_frame(r["ticker"]), r["signal_date"], px):
            tk, sd = r["ticker"], str(r["signal_date"])
            if sd > cutoff.get(tk, ""):
                cutoff[tk] = sd

    out: List[dict] = []
    skipped_rescaled = 0
    skipped_broken = 0
    for _, r in rows.iterrows():
        tk = r["ticker"]
        # Refuse to replay a ticker-day whose cache has been rescaled since:
        # those bars carry post-hoc split information the run never had.
        if not _scale_is_consistent(_frame(tk), r["signal_date"], r.get("price")):
            skipped_rescaled += 1
            continue
        cut = cutoff.get(tk)
        if cut is not None and str(r["signal_date"]) <= cut:
            skipped_rescaled += 1
            continue
        hist = visible_history(_frame(tk), r["signal_date"], r["generated_at"])
        if hist is None:
            continue
        # One technical result per row, shared by the `tech` score and the
        # context (see replay_context) — computing it twice was ~15% of the job.
        tech_res = None
        if with_context:
            try:
                from src.analysis.technical import compute_technical_score
                tech_res = compute_technical_score(tk, df=hist)
            except Exception:
                tech_res = None
        scores = replay_row(tk, r["signal_date"], r["generated_at"], methods,
                            hist=hist, tech_result=tech_res)
        ctx = replay_context(tk, hist, tech_result=tech_res) if with_context else {}
        # Second split detector: an impossible ATR means the bars contain a
        # discontinuity, which invalidates the scores as much as the context.
        if ctx and not _series_is_intact(ctx):
            skipped_broken += 1
            continue
        if not scores and not ctx:
            continue
        rec = {"signal_date": r["signal_date"], "ticker": tk,
               "run_id": r.get("run_id"), "generated_at": r.get("generated_at")}
        rec.update(scores)
        rec.update(ctx)
        out.append(rec)
    if skipped_rescaled:
        logger.info(f"[replay] skipped {skipped_rescaled:,} ticker-days whose OHLCV "
                    f"cache was rescaled since (post-hoc split info)")
    if skipped_broken:
        logger.info(f"[replay] skipped {skipped_broken:,} ticker-days whose bars carry a "
                    f"price discontinuity (impossible ATR — undetectable split)")
    return pd.DataFrame(out)


def materialize(days: Optional[int] = None, methods=REPLAYABLE) -> int:
    """Replay history into `signals_replay`, replacing any prior replay of the
    same span. Returns the row count written.

    Batch job, not a hot path: replaying is a per-ticker-day scorer run, far too
    slow to do inline. `restore_replayed` then serves the materialised values to
    the panel for free.
    """
    from datetime import datetime
    from src.db.connection import connect

    df = replay_panel(days=days, methods=methods)
    if df.empty:
        logger.info("[replay] nothing to materialize")
        return 0
    from src.db.schema import REPLAY_TABLE_COLUMNS
    df["replayed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = (["signal_date", "ticker", "run_id", "generated_at", "replayed_at"]
            + [c for c in REPLAY_TABLE_COLUMNS if c in df.columns])
    df = df[cols]

    # The scoring above costs tens of minutes; `connect`'s ~16s lock budget is
    # sized for a momentary collision with the pipeline writer, not for handing
    # that work back. The result is already in memory, so retry the WRITE on a
    # much longer budget rather than recomputing.
    import time
    deadline = time.monotonic() + 300.0
    delay = 2.0
    # The DELETE boundary comes from the STAGED FRAME, not a fresh CURRENT_DATE:
    # the SELECT window was computed hours earlier (scoring is slow) and a date
    # rollover between the two clocks shifts the DELETE one day past the INSERT,
    # leaving a stale day duplicated underneath the fresh copy (the 2026-06-22
    # twins that broke restore_replayed from 08-05 to 08-11). One boundary, one
    # source of truth, and DELETE+INSERT in one explicit transaction so a retry
    # or a concurrent writer can never observe (or leave) a half-applied window.
    lo = str(df["signal_date"].astype(str).min())
    while True:
        try:
            with connect() as con:
                con.execute("BEGIN TRANSACTION")
                try:
                    if days:
                        con.execute("DELETE FROM signals_replay "
                                    "WHERE signal_date >= ?", [lo])
                    else:
                        con.execute("DELETE FROM signals_replay")
                    con.register("_replay_df", df)
                    con.execute(f"INSERT INTO signals_replay ({', '.join(cols)}) "
                                f"SELECT {', '.join(cols)} FROM _replay_df")
                    con.unregister("_replay_df")
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
            break
        except Exception as e:
            if time.monotonic() >= deadline:
                raise
            logger.info(f"[replay] write blocked ({str(e)[:60]}…), retrying in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    logger.info(f"[replay] materialized {len(df):,} ticker-days "
                f"over {df['ticker'].nunique()} tickers")
    return len(df)


def restore_replayed(df: pd.DataFrame, methods=None) -> tuple:
    """Overwrite superseded method scores in a panel frame with replayed ones.

    Called by `signal_panel.build_panel` BEFORE the epoch mask, so a value the
    mask would blank is instead REGENERATED wherever the replay table has it.
    That is the whole point: the mask preserves correctness by discarding
    evidence, and this recovers the evidence at the same correctness.

    Returns ``(df, {method: boolean Series of restored cells})``. The caller MUST
    exclude those cells from the epoch mask — a restored value carries the
    current scorer's semantics, so blanking it by its original date would undo
    the restore and leave the panel exactly where it started.

    Fail-soft — any problem returns the frame untouched with no restored cells,
    so the mask still runs and correctness never depends on the replay table
    being present or fresh.
    """
    from src.db import repo
    from src.db.schema import REPLAY_TABLE_COLUMNS

    if df is None or df.empty:
        return df, {}
    if methods is None:
        methods = REPLAY_TABLE_COLUMNS
    try:
        # Context columns (atr_pct, vol_ratio, tape_score...) were NEVER
        # persisted to `signals`, so they are ADDED to the panel rather than
        # restored. That is the point: they are recovered market conditions the
        # panel has never carried, and every consumer treats an absent column as
        # absent, so adding one cannot change an existing analysis.
        cols = [m for m in methods if m in df.columns or m in REPLAY_TABLE_COLUMNS]
        if not cols:
            return df, {}
        rep = repo.fetch_df(
            "SELECT signal_date, ticker, generated_at, replayed_at, "
            f"{', '.join(cols)} FROM signals_replay")
        if rep is None or rep.empty:
            return df, {}
        rep["signal_date"] = rep["signal_date"].astype(str)

        # Match on the RUN, not just the day. `signals` holds ~43 runs per
        # ticker-day and `build_panel` keeps the last one; joining on
        # (date, ticker) alone would graft a DIFFERENT run's score onto the
        # panel's row — measured to triple the apparent change rate, including
        # for a scorer that never changed. generated_at makes the row identity
        # exact; without it, fall back to the panel's own last-run rule rather
        # than to arbitrary row order.
        key = ["signal_date", "ticker"]
        has_run = ("generated_at" in rep.columns
                   and rep["generated_at"].notna().any())
        if has_run and "generated_at" in df.columns:
            key = ["signal_date", "ticker", "generated_at"]
            rep["generated_at"] = rep["generated_at"].astype(str)
        # The exact key SHOULD be unique, but a historical double-materialise
        # (two window clocks — see materialize) proved it isn't guaranteed, and
        # one duplicate fans the left-merge out and kills the whole restore.
        # Dedupe unconditionally: newest replayed_at wins within a key (latest-
        # code values, matching materialize's replace semantics); in the 2-key
        # fallback the panel's own last-run rule stays primary (generated_at
        # sorted LAST so it decides keep="last", replayed_at the tiebreak).
        if "replayed_at" in rep.columns:
            rep = rep.sort_values("replayed_at", kind="stable")
        if has_run and "generated_at" not in key:
            rep = rep.sort_values("generated_at", kind="stable")
        rep = rep.drop_duplicates(subset=key, keep="last")
        rep = rep.drop(columns=[c for c in ("generated_at", "replayed_at")
                                if c in rep.columns and c not in key])

        orig_index = df.index
        left = df.copy()
        left["signal_date"] = left["signal_date"].astype(str)
        if "generated_at" in key:
            left["generated_at"] = left["generated_at"].astype(str)
        merged = left.merge(rep, on=key, how="left", suffixes=("", "_rp"))
        if len(merged) != len(left):
            # A left-merge only grows on duplicate right-side keys; the dedupe
            # above makes that impossible, so this is a real invariant, not a
            # tolerance. Raising lands in the fallback below with a message
            # naming the cause instead of pandas' opaque index error.
            raise AssertionError(
                f"replay join fanned out ({len(left)} -> {len(merged)} rows): "
                f"signals_replay holds duplicate {key} keys")
        merged.index = orig_index

        restored = {}
        for m in cols:
            # A column absent from the panel was merged in under its own name
            # (no collision, so no `_rp` suffix); one already present collides
            # and pandas suffixes it.
            rp = merged.get(f"{m}_rp")
            if rp is None and m not in df.columns:
                rp = merged.get(m)
            if rp is None:
                continue
            take = rp.notna()
            if take.any():
                if m not in df.columns:
                    df[m] = float("nan")
                df.loc[take.values, m] = rp[take].values
                restored[m] = take
        return df, restored
    except Exception as e:
        # WARNING, not DEBUG. Falling back to the mask is CORRECT but degraded —
        # the panel silently loses the regenerated evidence — and a bug in this
        # function looks exactly like "the table isn't populated yet". A broad
        # except here already hid one real defect (a missing column raising
        # KeyError), so the failure has to be audible.
        logger.warning(f"[replay] restore failed, falling back to epoch mask: {e}")
        return df, {}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Replay history through current scorers")
    ap.add_argument("--validate", action="store_true", help="fidelity report only")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--since", default="2026-07-01")
    ap.add_argument("--days", type=int, default=None,
                    help="limit the span; omit for all history")
    ap.add_argument("--write", action="store_true",
                    help="materialize into the signals_replay table")
    a = ap.parse_args()

    if a.write:
        n = materialize(days=a.days)
        print(f"materialized {n:,} rows into signals_replay")
    else:
        rep = validate(sample=a.sample, since=a.since)
        if rep.empty:
            print("No comparable rows.")
        else:
            print(f"\nReplay fidelity — current scorers vs stored "
                  f"({a.sample} sampled rows since {a.since})\n")
            print(rep.to_string(index=False))
            print("\nLOW exact_pct means either the scorer CHANGED since those rows")
            print("were written (expected — regenerating them is the point) or the")
            print("input reconstruction drifted. Split by epoch date to tell which.")
