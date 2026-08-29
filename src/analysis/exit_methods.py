"""Exit-method scoring — signed hold-conviction per held position.

The exit side of the aggregator's sign convention (see ``aggregator.py``). Every
exit method emits, for one OPEN position at one tick, a signed **hold-conviction**
score in ``[-1, +1]``:

* ``+`` = the position should KEEP running in its direction (hold),
* ``−`` = the position should REVERSE (exit),
* ``|score|`` = confidence.

Crucially every score is **position-oriented** — already multiplied by the
position's direction (``+1`` for a BUY/long, ``−1`` for a SELL/short) — so the
exit panel (``exit_panel.compute_exit_method_perf``) can correlate it directly
against the position's *direction-oriented* forward return. A persistently
POSITIVE IC then means the method correctly holds winners / exits losers, exactly
as a positive entry IC means the method correctly picks direction.

Two families of exit method:

* **Exit-decision overlays** — the exit-specific inputs to
  ``tracker._evaluate_decay``: the synthesized ``llm_review`` (the method that
  actually decides), the ``aggregator`` combined score, the ``macro_regime``
  risk overlay, the ``horizon`` time-stop, and the position-path excursion
  signals ``mfe`` (peak give-back / trailing) and ``mae`` (drawdown / stop).
* **Signal methods** — the same per-ticker entry methods (news, tech, momentum,
  …), re-read on the held ticker and oriented to the position, so we learn which
  entry signals are also good EXIT predictors.

Only NON-zero scores are returned — a ``0`` means "no exit view" and is excluded
from the panel, exactly like a ``0`` entry score is excluded from entry IC.
"""

from __future__ import annotations

from typing import Dict, Optional

from config.settings import settings

# The exit-specific overlay methods (distinct from the entry signal methods,
# which are ALSO re-scored as exit signals on a held position). ``mfe`` / ``mae``
# / ``ml_exit`` are held-only signals — they need a position's ratcheted
# excursions / state, so (like horizon / llm_review) they never appear in the
# universe shadow book.
EXIT_DECISION_METHODS = ("llm_review", "aggregator", "macro_regime", "horizon",
                         "edge_decay", "mfe", "mae", "ml_exit", "held_rank")

# Dashboard Exit-IC table grouping (mirrors signal_panel.IC_CATEGORY_ORDER).
EXIT_CATEGORY_DECISION = "Exit decision (synthesized review + overlays)"
EXIT_CATEGORY_SIGNAL = "Signal methods (re-scored as exit signals)"
EXIT_CATEGORY_ORDER = (EXIT_CATEGORY_DECISION, EXIT_CATEGORY_SIGNAL)

# Human labels for the exit-specific methods (signal-method labels come from
# tracker.METHOD_LABELS).
EXIT_METHOD_LABELS: Dict[str, str] = {
    "llm_review":   "LLM hold-review (synthesized decider)",
    "aggregator":   "Aggregator combined score",
    "macro_regime": "Macro regime overlay",
    "horizon":      "Horizon time-stop",
    "edge_decay":   "Edge-decay time-stop (realized edge window)",
    "mfe":          "Favorable excursion / give-back (trailing)",
    "mae":          "Adverse excursion / drawdown (stop)",
    "ml_exit":      "ML exit model (learned exit-timer)",
    "held_rank":    "Held-window score rank (signal decay vs own history)",
}

# Regime → hold-pressure for a LONG position (× the position's dir_sign). Only
# the regimes the exit rule actually reacts to are non-zero; NEUTRAL/CAUTION → 0
# (no exit view). Mirrors tracker._evaluate_decay's PANIC/RISK_OFF long-exit.
_REGIME_PRESSURE = {"RISK_ON": 0.5, "RISK_OFF": -0.7, "PANIC": -1.0}

# MFE / MAE excursion-signal scaling (all in position-P&L %). A peak worth
# protecting must exceed _MFE_MIN_PCT; a drawdown must be deeper than _MAE_MIN_PCT
# to signal; _MAE_SCALE_PCT is the depth at which the drawdown exit saturates to −1.
_MFE_MIN_PCT = 1.0
_MAE_MIN_PCT = 1.0
_MAE_SCALE_PCT = 8.0


def exit_category_for(method: str) -> str:
    """Map an exit method to its Exit-IC table category."""
    return EXIT_CATEGORY_DECISION if method in EXIT_DECISION_METHODS else EXIT_CATEGORY_SIGNAL


def _dir_sign(trade: dict) -> int:
    """+1 for a long (BUY) position, −1 for a short (SELL)."""
    return 1 if (trade.get("action") or "").upper() == "BUY" else -1


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _excursion_scores(trade: dict) -> Dict[str, float]:
    """Hold-conviction from the position's OWN path — the max favorable / adverse
    excursion vs the current mark (a trailing-profit + drawdown pair):

    * ``mfe`` — peak RETENTION: ``+1`` while the mark sits at its high-water peak
      (still running), ramping to ``−1`` as that peak is given back (momentum
      exhaustion → exit). Meaningful only once a real peak formed (MFE >
      ``_MFE_MIN_PCT``).
    * ``mae`` — drawdown SEVERITY relieved by recovery: negative, deeper the further
      the position bled (vs ``_MAE_SCALE_PCT``), fading to ``0`` as it recovers off
      its lows. Exit-biased (≤ 0); fires only past a real drawdown (> ``_MAE_MIN_PCT``).

    Both are ALREADY position-oriented — ``return_pct`` / MFE / MAE are P&L-signed
    (a short's favorable excursion is positive when the stock falls) — so they are
    NOT multiplied by ``dir_sign``. Whether give-back actually predicts reversal (or
    a deep MAE predicts further decline) is exactly what the exit-panel IC measures."""
    out: Dict[str, float] = {}
    cur = float(trade.get("return_pct") or 0.0)
    mfe = float(trade.get("max_favorable_excursion") or 0.0)
    mae = float(trade.get("max_adverse_excursion") or 0.0)
    if mfe > _MFE_MIN_PCT:
        give_back = (mfe - cur) / mfe              # 0 at the peak, 1 back at entry, >1 below
        out["mfe"] = _clip(1.0 - 2.0 * give_back, -1.0, 1.0)
    depth = -mae                                   # ≥ 0 once the position has drawn down
    if depth > _MAE_MIN_PCT:
        severity = _clip(depth / _MAE_SCALE_PCT, 0.0, 1.0)
        recovery = _clip((cur - mae) / depth, 0.0, 1.0)   # 0 at the lows → 1 recovered to entry
        out["mae"] = -severity * (1.0 - recovery)
    return out


def _horizon_pressure(trade: dict) -> float:
    """One-sided hold-conviction from the horizon time-stop: ``0`` while the
    position is within its target-horizon window, then increasingly negative once
    it has outlived that window (the ``horizon_expired`` exit rule, as a signal).
    ``0`` when horizon synthesis produced no target or its duration is unknown."""
    target_h = trade.get("target_horizon")
    if not target_h:
        return 0.0
    from src.signals.edge_curve import horizon_hours
    from src.performance.tracker import _held_hours
    window = horizon_hours(target_h)
    held = _held_hours(trade)
    if not window or held is None or held < window:
        return 0.0
    return -min(1.0, held / window - 1.0)


def method_horizon_days(trade: dict) -> Optional[float]:
    """The position's target holding period in trading days, derived from the
    MEASURED best horizon of the methods that actually drove the entry.

    Each method's edge lives at a particular holding period (see
    `analysis.method_horizons`): `sent_velocity` peaks at 1 day and decays,
    `pattern` builds to 5-10 days, `oi_skew` is slowest. So a position opened
    mostly on `sent_velocity` should be judged on a much shorter clock than one
    opened on `pattern`, and a single global time-stop cannot express that.

    Weighted by each method's CONVICTION AND IMPORTANCE at entry —
    ``|method score| × |base weight|`` — so the horizon follows whichever methods
    actually carried the decision. Only PROVEN methods have a measured horizon
    and therefore contribute; a trade driven entirely by unproven methods returns
    ``None`` (no view) rather than a fabricated deadline.
    """
    scores = trade.get("method_scores") or {}
    if not scores:
        return None
    try:
        from src.analysis.method_horizons import method_best_days
        from src.signals.aggregator import _BASE_WEIGHTS
    except Exception:
        return None
    num = den = 0.0
    for m, sc in scores.items():
        days = method_best_days(m)
        if not days:
            continue                       # unproven / disproven → no clock
        try:
            w = abs(float(sc or 0.0)) * abs(float(_BASE_WEIGHTS.get(m, 0.0)))
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        num += w * float(days)
        den += w
    return (num / den) if den > 0 else None


def _method_horizon_pressure(trade: dict) -> float:
    """One-sided hold-conviction from the METHOD-DERIVED horizon: ``0`` while the
    position is inside the window its own methods say their edge lives in, then
    increasingly negative once it has outlived it.

    Distinct from the two time-stops already present, and deliberately kept
    alongside them rather than replacing either: ``horizon`` uses the LLM's
    stated target, ``edge_decay`` a single system-wide window measured on
    `combined_score`. This one is per-position and derived from the METHODS that
    opened it, so it is the only one that can say a `sent_velocity` trade is
    stale after a day while a `pattern` trade still has a week to run.
    """
    days = method_horizon_days(trade)
    if not days:
        return 0.0
    held = _held_trading_days(trade)
    if held is None or held < days:
        return 0.0
    return -min(1.0, held / days - 1.0)


def _held_trading_days(trade: dict) -> Optional[int]:
    """NYSE trading days a position has been held (entry → today), for the
    edge-decay window (measured in trading sessions, matching the panel's forward
    horizons). None when the entry date is unparseable."""
    import numpy as np
    from datetime import date as _date, datetime
    raw = trade.get("entry_date") or trade.get("entry_datetime")
    if not raw:
        return None
    try:
        ed = datetime.fromisoformat(str(raw)[:10]).date()
    except (ValueError, TypeError):
        return None
    today = _date.today()
    return int(np.busday_count(ed, today)) if today > ed else 0


def _edge_decay_pressure(trade: dict) -> float:
    """One-sided hold-conviction from the EDGE-DECAY time-stop: ``0`` while the
    position is held within the measured edge-positive window, then increasingly
    negative once past it. Distinct from ``horizon`` (which uses the entry
    target-horizon): this uses the REALIZED edge-decay window measured over the
    signals panel (``horizon_edge.calibrate_edge_horizon``). Raw / unthrottled —
    the evidence strength is applied at the floor nudge, so the panel measures the
    signal's own predictiveness cleanly. ``0`` when disabled / no measured window."""
    from src.analysis.horizon_edge import calibrate_edge_horizon
    edge_days = calibrate_edge_horizon().get("edge_days")
    if not edge_days:
        return 0.0
    held = _held_trading_days(trade)
    if held is None or held <= edge_days:
        return 0.0
    return -min(1.0, held / edge_days - 1.0)


# -- held-window score rank (2026-08-22, PANEL-FIRST) -------------------------
#
# The user-directed exit method: for a HELD position, where does the ticker's
# CURRENT aggregate score rank within the pool of that same ticker's scores over
# every tick since the position was entered? The entry side ranks a name against
# the day's UNIVERSE ("which stock"); this ranks it against its own held-window
# history ("is this position's signal now at its weakest since we got in").
# Self-normalizing per ticker, so cross-ticker scale never enters.
#
# Panel-first: scored + persisted to `exit_signals` (IC accrues live), in
# exit_conviction._CONSENSUS_SKIP (never nudges the confidence floor), no
# closing rule -- the same probation every new entry method serves.
#
# The pool reads the ABSOLUTE-basis shadow combine (`combined_score_abs`,
# falling back to `combined_score` on pre-shadow rows, which ARE absolute) for
# the same reason ml_exit's `ex_combine` does: a held window can span an
# ml-arm/weighted flip or a shaped-curve refresh, and a rank over two different
# scales in one pool is not a rank. Today's in-memory value uses the same
# preference, so pool and current observation are one quantity.
#
# Ordering invariant (pinned by a test): `monitor_open_positions` runs BEFORE
# `_persist_run` writes this tick's signals rows, so the DB pool never contains
# today's score -- it is appended from the in-memory TickerSignal, exactly once.

_HELD_RANK_TTL_SECONDS = 20 * 60          # one pool query per tick (runner >=30m)
_HELD_RANK_CACHE: Dict[str, object] = {"ts": 0.0, "key": None, "pools": None}


def _abs_combine(sig) -> Optional[float]:
    """The basis-invariant combine of a TickerSignal (abs shadow preferred)."""
    if sig is None:
        return None
    v = getattr(sig, "combined_score_abs", None)
    if v is None or v != v:
        v = getattr(sig, "combined_score", None)
    try:
        return float(v) if v is not None and v == v else None
    except (TypeError, ValueError):
        return None


def _held_rank_pools(trades: list, force: bool = False) -> Dict[str, list]:
    """``{ticker: [(generated_at, score), ...]}`` since the EARLIEST open entry.

    One query per tick for every held ticker (the per-trade entry filter happens
    in ``_held_rank_from_live`` -- positions on the same ticker can differ).
    Fail-soft to ``{}``: the method then abstains rather than scoring on a
    partial pool.
    """
    import time as _time
    # Callers pass whatever ledger slice they hold (the monitor passes ALL
    # trades); the pool is only about OPEN positions, so filter here -- one
    # wrong caller must not widen the query to the whole closed history.
    trades = [t for t in (trades or [])
              if str(t.get("status") or "OPEN").upper() == "OPEN"]
    open_keys = sorted({str(t.get("ticker")) for t in trades if t.get("ticker")})
    min_entry = min((str(t.get("entry_datetime") or t.get("entry_date") or "")
                     for t in trades if (t.get("entry_datetime") or t.get("entry_date"))),
                    default="")
    key = (tuple(open_keys), min_entry[:10])
    now = _time.time()
    if (not force and _HELD_RANK_CACHE["pools"] is not None
            and _HELD_RANK_CACHE["key"] == key
            and (now - float(_HELD_RANK_CACHE["ts"])) < _HELD_RANK_TTL_SECONDS):
        return _HELD_RANK_CACHE["pools"]        # type: ignore[return-value]
    pools: Dict[str, list] = {}
    if open_keys and min_entry:
        try:
            from src.db import repo
            ph = ", ".join(["?"] * len(open_keys))
            df = repo.fetch_df(
                f"SELECT ticker, generated_at, combined_score_abs, combined_score "
                f"FROM signals WHERE ticker IN ({ph}) AND generated_at >= ? "
                f"ORDER BY generated_at",
                [*open_keys, min_entry])
            if df is not None and not df.empty:
                for r in df.itertuples(index=False):
                    v = r.combined_score_abs
                    if v is None or v != v:
                        v = r.combined_score
                    if v is None or v != v:
                        continue
                    pools.setdefault(str(r.ticker), []).append(
                        (str(r.generated_at), float(v)))
        except Exception as e:
            logger.warning(f"[held_rank] pool query failed (method abstains): {e}")
            pools = {}
    _HELD_RANK_CACHE.update(ts=now, key=key, pools=pools)
    return pools


def reset_cache() -> None:
    """Test / asof hook (mirrors news_shock.reset_cache)."""
    _HELD_RANK_CACHE.update(ts=0.0, key=None, pools=None)


def held_rank_score(oriented_pool: list, oriented_now: float,
                    min_ticks: Optional[int] = None) -> float:
    """Signed hold-conviction from the held-window rank: ``2*pct - 1``.

    ``oriented_pool`` = the position-ORIENTED scores of every tick since entry
    (excluding now); ``oriented_now`` = today's oriented score. pct is the
    average-tie rank of now within pool + {now}, over its size -- so today at
    its held-window best -> +1 (keep), at its worst -> -1 (exit pressure), and
    a pool of near-ties -> ~0 (no view). Abstains (0.0) below ``min_ticks``
    total observations: a rank in a pool of two is a coin, not a signal.
    """
    if oriented_now is None or oriented_now != oriented_now:
        return 0.0
    floor = int(min_ticks if min_ticks is not None
                else getattr(settings, "held_rank_min_ticks", 5))
    vals = [float(v) for v in oriented_pool if v is not None and v == v]
    vals.append(float(oriented_now))
    n = len(vals)
    if n < max(2, floor):
        return 0.0
    below = sum(1 for v in vals if v < oriented_now)
    ties = sum(1 for v in vals if v == oriented_now)
    # MID-rank percentile ((rank - 0.5)/n with average ties): symmetric by
    # construction -- the pool minimum scores -(n-1)/n and the maximum +(n-1)/n,
    # an all-tie pool scores exactly 0. The naive (rank/n) form is asymmetric
    # (min of 5 -> -0.6 while max -> +1.0), which would bias the persisted
    # panel bullish for no reason.
    pct = (below + ties / 2.0) / n
    return round(2.0 * pct - 1.0, 6)


def _held_rank_from_live(trade: dict, today_signal, all_open_trades: list) -> float:
    """The live wrapper: pool since THIS position's entry, oriented, scored."""
    now_raw = _abs_combine(today_signal)
    if now_raw is None:
        return 0.0
    ds = _dir_sign(trade)
    if ds == 0.0:
        return 0.0
    entry = str(trade.get("entry_datetime") or trade.get("entry_date") or "")
    if not entry:
        return 0.0
    pools = _held_rank_pools(all_open_trades)
    series = pools.get(str(trade.get("ticker"))) or []
    oriented = [v * ds for ts, v in series if ts >= entry]
    return held_rank_score(oriented, now_raw * ds)


def build_exit_scores(trade: dict, hold_review, signals_by_ticker, macro_regime_context,
                      _hr_all_trades: Optional[list] = None) -> Dict[str, float]:
    """Signed hold-conviction score per exit method for one held position.

    ``trade`` is the open-trade dict; ``hold_review`` its opener-pinned
    ``Recommendation`` this tick (or ``None``); ``signals_by_ticker`` the run's
    aggregator cross-section; ``macro_regime_context`` the run regime. Returns
    ``{method: score}`` for the NON-zero scores only (0 = no exit view). See the
    module docstring for the sign convention.
    """
    scores: Dict[str, float] = {}
    dir_sign = _dir_sign(trade)
    entry_action = (trade.get("action") or "").upper()
    ticker = trade.get("ticker")
    today_signal = (signals_by_ticker or {}).get(ticker)

    # 1. Synthesized LLM hold-review — the method that actually decides. +conf when
    #    it reaffirms the position's direction, −conf when it flips; HOLD/WATCH is
    #    no directional view (0 → omitted).
    if hold_review is not None:
        rev_action = (getattr(hold_review, "action", "") or "").upper()
        conf = float(getattr(hold_review, "confidence", 0.0) or 0.0)
        if rev_action == entry_action:
            scores["llm_review"] = conf
        elif rev_action in ("BUY", "SELL"):
            scores["llm_review"] = -conf

    # 2. Aggregator combined score, oriented to the position.
    if today_signal is not None:
        scores["aggregator"] = float(getattr(today_signal, "combined_score", 0.0) or 0.0) * dir_sign

    # 3. Macro regime risk overlay, oriented to the position.
    regime = (getattr(macro_regime_context, "regime", "") or "").upper()
    scores["macro_regime"] = _REGIME_PRESSURE.get(regime, 0.0) * dir_sign

    # 4. Horizon time-stop pressure (one-sided: 0 within window, negative past it).
    scores["horizon"] = _horizon_pressure(trade)

    # 4-b. Method-derived horizon (2026-07-26): the holding window the position's
    #      OWN methods say their edge lives in, weighted by their conviction at
    #      entry. Complements rather than replaces `horizon` (LLM-stated) and
    #      `edge_decay` (one global window) — only this one is per-position.
    scores["method_horizon"] = _method_horizon_pressure(trade)

    # 4a. Edge-decay time-stop — exit pressure once held past the REALIZED
    #     edge-positive window (measured over the signals panel), distinct from the
    #     entry target-horizon above. Its own exit-IC is measured in the panel.
    scores["edge_decay"] = _edge_decay_pressure(trade)

    # 4b. MFE / MAE excursion signals from the position's own path (already
    #     position-oriented — P&L terms — so no dir_sign). Held-only.
    scores.update(_excursion_scores(trade))

    # 4c. ML exit model (2026-08-02) — the learned exit-timer (position state +
    #     the oriented method scores), a signed hold-conviction (+ = keep, − =
    #     exit). Persisted for EVERY held position so its exit IC accrues live;
    #     it only CLOSES trades stamped `ml_arm` (the entry-arm cohort —
    #     tracker.monitor_open_positions gates the decision on that). Fail-soft:
    #     None (no artifact / lightgbm) ⇒ omitted, no exit view.
    if settings.enable_ml_exit_model:
        try:
            from src.analysis.ml_exit_dataset import compute_exit_model_score
            mx = compute_exit_model_score(trade, signals_by_ticker, today_signal)
            if mx is not None:
                scores["ml_exit"] = mx
        except Exception:
            pass

    # 4d. Held-window score rank (2026-08-22, panel-first): today's aggregate
    #     score ranked within THIS position's own tick history since entry --
    #     signal decay against the position's own past, not the universe.
    #     `_hr_all_trades` is threaded by the monitor/persist callers so ONE
    #     pool query serves every held position; absent (tests, ad-hoc), the
    #     pool covers just this trade.
    if getattr(settings, "enable_held_rank_exit_signal", True):
        try:
            _hr = _held_rank_from_live(trade, today_signal,
                                       _hr_all_trades if _hr_all_trades is not None
                                       else [trade])
            if _hr:
                scores["held_rank"] = _hr
        except Exception as e:
            logger.debug(f"[held_rank] score failed for {ticker}: {e}")

    # 5. The entry signal methods, re-scored on the held ticker and oriented.
    if today_signal is not None:
        from src.performance.tracker import _method_scores_from_signal
        raw = _method_scores_from_signal(ticker, trade.get("direction"), signals_by_ticker)
        for m, v in raw.items():
            scores[m] = float(v or 0.0) * dir_sign

    # Persist only non-zero scores (0 = no view, mirrors the entry panel).
    return {m: round(s, 6) for m, s in scores.items() if s}
