"""LIVE capture of the models' technical-indicator feature vectors (2026-09-25,
user directive: the news AND technical-indicator features must accrue from
2026-09-28 — test set 2 — as the live pipeline's own values).

What a tick already persists: every method score (tech, vwap, momentum, ...),
their 30-minute / weekly variants and the market state (``atr_pct``,
``bb_width_pct``, ``vol_ratio``, ``tape_score``) in `signals`. What it did not:
the INDICATOR vectors the next models train on (`ml30` / `sel_models`) — the
85 base features at the last completed 30-minute bar, the 73 deep features,
and the daily model's previous-session row. Those are recomputable from bars
later, but only as a reconstruction (revised bars, a tick cache the deep store
later overwrites, a snapshot rebuilt from another day's data). So each tick now
writes what it computed:

  ``data/live_features/30m/<ET date>/<run_id>.parquet``
      one row per scored name per tick: the 30-minute base features exactly as
      `ml_model.features_30m` served them to ``ml_ohlcv`` (the SAME function —
      a memo hit for every name the tick scored), the deep features from the
      bar's session snapshot (`deep_features.serving_vector`, NaN with
      ``deep_status='NO_SNAPSHOT'`` when the pre-open run did not build it), and
      ``status`` / ``bar_ts`` / ``session_day`` / ``bar_idx`` / ``close``.
  ``data/live_features/daily/<session date>.parquet``
      one row per name per SESSION: the daily model's inputs — the daily
      feature frame and leg state of the last daily bar BEFORE the session
      (``ml30.ticker_rows(rows="daily")``'s X), captured by the first tick
      that scores the name for that session.

Column names are `ml30.base_features()` + `deep_features.DEEP_FEATURES`, so a
model fitted on `ml30` arrays reads these files without a mapping. It runs in a
background thread after the tick has persisted — never on the critical path —
and single-flight: a capture still running when the next tick lands is left
alone and that tick is skipped (logged). ``data/`` rather than ``cache/``: these
are the ONLY copy of what the live tick saw, not a cache to rebuild.
"""
from __future__ import annotations

import bisect
import threading
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

LIVE_DIR = Path("data/live_features")
_ET = ZoneInfo("America/New_York")
_OVERNIGHT_FROM = dtime(20, 0)       # a tick from 20:00 ET serves the NEXT session
_DAILY_MIN_BARS = 60                 # `ml30.ticker_rows(rows="daily")`'s floor

_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def capture_session(t: datetime) -> date:
    """The NYSE session a tick at ``t`` serves: its own ET date on a market day
    before 20:00, else the next market day (an overnight tick after 20:00 leads
    into tomorrow; a weekend tick into Monday)."""
    from src.performance.market_calendar import is_market_day
    et = t.astimezone(_ET) if t.tzinfo else t.replace(tzinfo=timezone.utc).astimezone(_ET)
    d = et.date()
    if is_market_day(d) and et.time() < _OVERNIGHT_FROM:
        return d
    d += timedelta(days=1)
    while not is_market_day(d):
        d += timedelta(days=1)
    return d


def _base_columns() -> List[str]:
    from src.analysis import ml30
    return ml30.base_features()


def _deep_columns() -> List[str]:
    from src.analysis import deep_features as dfe
    return list(dfe.DEEP_FEATURES)


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def rows_30m(tickers: Sequence[str], now_utc: datetime) -> pd.DataFrame:
    """The 30-minute capture for ``tickers`` at ``now_utc`` (no I/O beyond what
    serving does)."""
    from src.analysis import deep_features as dfe
    from src.signals import ml_model
    base_cols, deep_cols = _base_columns(), _deep_columns()
    now = pd.Timestamp(now_utc)
    now = now.tz_convert("UTC").tz_localize(None) if now.tzinfo is not None else now
    out = []
    snaps: Dict[int, Optional[dict]] = {}
    for tk in dict.fromkeys(str(t).upper() for t in tickers):
        frame, label = ml_model.features_30m(tk, now)
        rec = {"ticker": tk, "status": label}
        if frame is None:
            rec.update({"bar_ts": pd.NaT, "session_day": None, "bar_idx": np.nan, "close": np.nan,
                        "n_bars": np.nan, "deep_status": None})
            rec.update({c: np.nan for c in base_cols + deep_cols})
            out.append(rec)
            continue
        feats = frame["features"]
        sday = int(frame["sday"])
        rec.update({"bar_ts": frame["bar_ts"],
                    "session_day": str(np.datetime64(sday, "D")),
                    "bar_idx": frame["bar_idx"], "close": frame["close"], "n_bars": frame["n_bars"]})
        rec.update({c: _f(feats.get(c)) for c in base_cols})
        if sday not in snaps:
            snaps[sday] = dfe.load_session_snapshot(sday)
        snap = snaps[sday]
        if snap is None:
            rec["deep_status"] = "NO_SNAPSHOT"
            rec.update({c: np.nan for c in deep_cols})
        else:
            deep = dfe.serving_vector(tk, sday, frame["close"], frame["bar_idx"], snap)
            rec["deep_status"] = "OK" if tk in snap else "NOT_IN_SNAPSHOT"
            rec.update({c: _f(deep.get(c)) for c in deep_cols})
        out.append(rec)
    return pd.DataFrame(out)


def daily_row(ticker: str, session: date) -> Dict[str, float]:
    """The daily model's input row for ``session``: the daily feature frame +
    daily leg state of the last daily bar STRICTLY before it — what
    ``ml30.ticker_rows(rows="daily")`` hands the model as X. ``{"status": ...}``
    plus the base columns (NaN when there is no row)."""
    from src.analysis import ml_dataset as md
    from src.analysis.pivot_target import LEG_FEATURES, leg_feature_rows
    from src.analysis.predictability import _hlc_by_session
    cols = _base_columns()
    empty = {"status": "NO_DATA", "feature_date": None, **{c: np.nan for c in cols}}
    try:
        dh = _hlc_by_session(ticker)
        if dh is None:
            return empty
        days = list(dh[0])
        n = bisect.bisect_left(days, session)            # bars strictly before the session
        if n < _DAILY_MIN_BARS:
            return empty
        cut = (days[:n], dh[1].iloc[:n], dh[2].iloc[:n], dh[3].iloc[:n], dh[4].iloc[:n])
        fs = md.ticker_feature_frame(ticker, hlc=cut)
        if fs is None or fs.empty:
            return empty
        row = fs.reindex(columns=md.ALL_FEATURE_COLUMNS).iloc[-1]
        legs = leg_feature_rows(dh[3].iloc[:n].to_numpy(float), dh[1].iloc[:n].to_numpy(float),
                                dh[2].iloc[:n].to_numpy(float), list(days[:n]), only_last=True)
        leg = legs[0] if legs else {}
        out = {"status": "OK", "feature_date": str(days[n - 1])}
        out.update({c: _f(row.get(c)) for c in md.ALL_FEATURE_COLUMNS})
        out.update({f: _f(leg.get(f)) for f in LEG_FEATURES})
        return out
    except Exception as e:                            # noqa: BLE001 — one name never sinks the file
        logger.debug(f"[live-features] daily row failed for {ticker}: {e}")
        return {**empty, "status": "ERROR"}


def _write(df: pd.DataFrame, path: Path) -> int:
    from src.data.deep import write_parquet
    return write_parquet(df, path)


def capture(run_id: str, start: datetime, tickers: Sequence[str],
            base_dir: Optional[Path] = None) -> dict:
    """Write this tick's 30-minute rows and the session's daily rows for any
    name not yet captured. Returns a summary dict."""
    base_dir = Path(base_dir) if base_dir else LIVE_DIR
    t0 = time.time()
    start = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    names = list(dict.fromkeys(str(t).upper() for t in tickers or []))
    gen = start.astimezone(timezone.utc).isoformat()
    summary: dict = {"run_id": run_id, "names": len(names)}

    df = rows_30m(names, start)
    if not df.empty:
        df.insert(0, "generated_at", gen)
        df.insert(0, "run_id", str(run_id))
        day = start.astimezone(_ET).date().isoformat()
        p30 = base_dir / "30m" / day / f"{run_id}.parquet"
        summary["rows_30m"] = _write(df, p30)
        summary["ok_30m"] = int((df["status"] == "OK").sum())
        summary["deep_ok"] = int((df["deep_status"] == "OK").sum())

    session = capture_session(start)
    pd_path = base_dir / "daily" / f"{session.isoformat()}.parquet"
    from src.data.deep import read_parquet
    have = read_parquet(pd_path) if pd_path.exists() else pd.DataFrame()
    done = set(have["ticker"].astype(str)) if not have.empty else set()
    todo = [t for t in names if t not in done]
    if todo:
        recs = []
        for tk in todo:
            r = daily_row(tk, session)
            recs.append({"run_id": str(run_id), "generated_at": gen, "ticker": tk,
                         "session": session.isoformat(), **r})
        new = pd.DataFrame(recs)
        merged = pd.concat([have, new], ignore_index=True) if not have.empty else new
        _write(merged, pd_path)
        summary["daily_new"] = len(new)
        summary["daily_ok"] = int((new["status"] == "OK").sum())
    summary["seconds"] = round(time.time() - t0, 1)
    logger.info(f"[live-features] {summary}")
    return summary


def capture_async(run_id: str, start: datetime, tickers: Sequence[str]) -> bool:
    """`capture` in a background daemon thread; single-flight (a capture still
    running when the next tick lands makes THAT tick skip, logged)."""
    global _THREAD
    with _LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            logger.warning(f"[live-features] previous capture still running — run {run_id} skipped")
            return False

        def _work():
            try:
                capture(run_id, start, tickers)
            except Exception as e:                    # noqa: BLE001 — never the tick's problem
                logger.warning(f"[live-features] capture failed for {run_id}: {e}")

        _THREAD = threading.Thread(target=_work, name="live-features", daemon=True)
        _THREAD.start()
    return True


def load(kind: str = "30m", start: Optional[str] = None, end: Optional[str] = None,
         base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Every captured row of ``kind`` (``30m`` | ``daily``) whose ET date /
    session falls in ``[start, end]`` (ISO dates, inclusive; open-ended when
    omitted) — the set-2 feature panel."""
    import duckdb
    base_dir = Path(base_dir) if base_dir else LIVE_DIR
    if kind == "30m":
        files = sorted(p for d in (base_dir / "30m").glob("*") if d.is_dir()
                       and (start is None or d.name >= start) and (end is None or d.name <= end)
                       for p in d.glob("*.parquet"))
    elif kind == "daily":
        files = sorted(p for p in (base_dir / "daily").glob("*.parquet")
                       if (start is None or p.stem >= start) and (end is None or p.stem <= end))
    else:
        raise ValueError(f"kind must be 30m or daily, got {kind!r}")
    if not files:
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        return con.execute("SELECT * FROM read_parquet(?, union_by_name = true)",
                           [[p.as_posix() for p in files]]).df()
    finally:
        con.close()
