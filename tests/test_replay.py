"""Replay of historical ticker-days through the current scorers.

The property under test is NOT "replay reproduces stored values" — where a
scorer changed it deliberately does not, and regenerating those rows is the
entire point. It is:

  1. the pipeline's forming-bar rule is honoured, so a run is replayed against
     the bars it could actually see (getting this wrong silently costs ~60pp of
     fidelity and looks like "history is unreproducible");
  2. a restored cell SURVIVES the epoch mask, otherwise the restore is undone
     and the panel is exactly where it started;
  3. everything fails soft — an absent or stale replay table degrades to plain
     masking, never to a wrong number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import replay


def _frame(days: int = 120, start: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=days)
    base = np.linspace(100.0, 130.0, days)
    return pd.DataFrame(
        {"Open": base, "High": base * 1.01, "Low": base * 0.99,
         "Close": base, "Volume": np.full(days, 1_000_000.0)},
        index=idx)


# ── the forming-bar rule ──────────────────────────────────────────────────────

def test_intraday_run_cannot_see_its_own_days_bar():
    """The live pipeline drops the still-forming daily bar, so a run at 10:00 ET
    never saw it. Replaying inclusively is the mistake that makes history look
    unreproducible."""
    df = _frame()
    sig = df.index[-1].strftime("%Y-%m-%d")
    hist = replay.visible_history(df, sig, f"{sig}T14:00:00+00:00")  # 10:00 ET
    assert hist.index[-1].strftime("%Y-%m-%d") < sig


def test_post_close_run_does_see_its_own_bar():
    df = _frame()
    sig = df.index[-1].strftime("%Y-%m-%d")
    hist = replay.visible_history(df, sig, f"{sig}T21:30:00+00:00")  # 17:30 ET
    assert hist.index[-1].strftime("%Y-%m-%d") == sig


def test_unparseable_timestamp_assumes_forming_bar():
    """Fail toward the CONSERVATIVE side: an unknown run time must not grant
    the replay a bar the run may never have had (that would be look-ahead)."""
    df = _frame()
    sig = df.index[-1].strftime("%Y-%m-%d")
    hist = replay.visible_history(df, sig, "not-a-timestamp")
    assert hist.index[-1].strftime("%Y-%m-%d") < sig


def test_short_history_yields_no_view():
    assert replay.visible_history(_frame(days=10), "2026-06-01", None) is None
    assert replay.visible_history(pd.DataFrame(), "2026-06-01", None) is None


def test_replay_row_scores_are_in_contract_range():
    df = _frame()
    sig = df.index[-1].strftime("%Y-%m-%d")
    out = replay.replay_row("TEST", sig, f"{sig}T14:00:00+00:00", df=df)
    assert out, "a 120-bar frame should produce at least one score"
    for method, score in out.items():
        assert -1.0 <= score <= 1.0, f"{method} out of [-1,1]: {score}"


# ── no look-ahead ─────────────────────────────────────────────────────────────
# The replay's whole legitimacy rests on scoring a past date with only what that
# date could see. These are adversarial probes, not smoke tests.

def test_future_bars_cannot_change_a_replayed_score():
    """Append 60 EXTREME future bars (100x price, huge volume). Every score must
    be bit-identical: if any future information reaches a scorer — through the
    frame, a re-fetch, or module state — this moves."""
    base = _frame(200)
    sig = base.index[-1].strftime("%Y-%m-%d")
    gen = f"{sig}T14:00:00+00:00"

    fut_idx = pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=60)
    fut = pd.DataFrame({"Open": [1e4] * 60, "High": [2e4] * 60, "Low": [5e3] * 60,
                        "Close": [1e4] * 60, "Volume": [9e9] * 60}, index=fut_idx)

    clean = replay.replay_row("TEST", sig, gen, df=base)
    poisoned = replay.replay_row("TEST", sig, gen, df=pd.concat([base, fut]))

    assert clean, "expected scores from a 200-bar frame"
    assert clean == poisoned, f"future data leaked into: " \
        f"{ {k for k in clean if clean.get(k) != poisoned.get(k)} }"


def test_replay_never_touches_a_live_data_source(monkeypatch):
    """Every scorer takes ``df=`` and must honour it. If one falls back to a
    fetch it would be reading TODAY's data while pricing a past date — the
    quietest possible look-ahead, invisible in the output."""
    import src.data.cache as cache_mod
    import src.data.market_data as md

    def boom(*a, **k):
        raise AssertionError("replay reached for live data")

    monkeypatch.setattr(md, "get_history", boom, raising=False)
    monkeypatch.setattr(cache_mod, "load_ohlcv", boom, raising=False)
    for mod in ("src.analysis.technical", "src.signals.vwap",
                "src.signals.price_momentum", "src.signals.money_flow",
                "src.signals.trend_strength", "src.signals.iv_rank"):
        m = __import__(mod, fromlist=["x"])
        for attr in ("get_history", "load_ohlcv"):
            if hasattr(m, attr):
                monkeypatch.setattr(m, attr, boom, raising=False)

    base = _frame(200)
    sig = base.index[-1].strftime("%Y-%m-%d")
    out = replay.replay_row("TEST", sig, f"{sig}T14:00:00+00:00", df=base)
    assert len(out) == len(replay.REPLAYABLE), \
        f"only {len(out)}/{len(replay.REPLAYABLE)} scored without live data"


def test_no_bar_after_the_signal_date_is_ever_visible():
    base = _frame(200)
    sig = base.index[-1].strftime("%Y-%m-%d")
    for gen in (f"{sig}T14:00:00+00:00", f"{sig}T21:30:00+00:00"):
        hist = replay.visible_history(base, sig, gen)
        assert hist.index.max().strftime("%Y-%m-%d") <= sig


def test_rescaled_cache_is_refused_not_replayed():
    """A ticker that split AFTER the signal date has a retroactively adjusted
    cache — post-hoc information the run never had. Detected by the live-recorded
    price disagreeing with the cached close, and such rows must be SKIPPED so the
    epoch mask handles them instead."""
    df = _frame(120)
    sig = df.index[-1].strftime("%Y-%m-%d")
    close = float(df["Close"].iloc[-1])

    assert replay._scale_is_consistent(df, sig, close)          # same scale
    assert not replay._scale_is_consistent(df, sig, close * 4)  # 4:1 split
    assert not replay._scale_is_consistent(df, sig, close / 10)  # reverse split
    # Ordinary intraday drift between snapshot and close must NOT be rejected.
    assert replay._scale_is_consistent(df, sig, close * 1.05)


@pytest.mark.parametrize("price", [None, 0, float("nan"), np.nan])
def test_scale_guard_fails_open_on_missing_price(price):
    """No recorded price => replay proceeds.

    NaN is the case that actually bit: `not nan` is False and `nan <= 0` is
    False, so a NaN slips past the obvious guards and then fails
    `lo < nan < hi`, flipping this from fail-open to fail-CLOSED. It silently
    refused 41,807 rows (13.5% of the panel) whose price was merely NULL, and
    reported them as "rescaled". A `None`/`0`-only test does not catch it.
    """
    df = _frame(120)
    sig = df.index[-1].strftime("%Y-%m-%d")
    assert replay._scale_is_consistent(df, sig, price) is True


def test_rescale_cutoff_covers_priceless_rows_before_a_split(monkeypatch):
    """The per-row check needs a live price and 13.8% of rows have none, so a
    priceless row on a ticker that DID split would sail through unchecked.

    A split contaminates only the dates BEFORE it, so the cutoff must suppress
    the earlier priceless row and KEEP the later clean one — banning the whole
    ticker would throw away good post-split history for no correctness gain.
    """
    df = _frame(200)
    dates = [d.strftime("%Y-%m-%d") for d in df.index]
    early, split_day, late = dates[-40], dates[-20], dates[-1]
    close_at = lambda d: float(df.loc[df.index.strftime("%Y-%m-%d") == d, "Close"].iloc[0])

    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": [early, split_day, late],
        "ticker": ["AAA", "AAA", "AAA"],
        "run_id": ["r1", "r2", "r3"],
        "generated_at": [f"{early}T14:00:00", f"{split_day}T14:00:00", f"{late}T14:00:00"],
        # No price on the early row; the split is visible on the middle row.
        "price": [float("nan"), close_at(split_day) * 4.0, close_at(late)],
    }))
    monkeypatch.setattr("src.data.cache.load_ohlcv", lambda *a, **k: df)

    got = replay.replay_panel(methods=("tech",))
    kept = set(got["signal_date"]) if not got.empty else set()

    assert early not in kept, "priceless row BEFORE the split must be suppressed"
    assert split_day not in kept, "the detected split row itself must be suppressed"
    assert late in kept, "clean post-split history must still be replayed"


def test_scale_guard_fails_open_on_missing_bars():
    df = _frame(120)
    sig = df.index[-1].strftime("%Y-%m-%d")
    assert replay._scale_is_consistent(df, "1999-01-01", 100.0)
    assert replay._scale_is_consistent(pd.DataFrame(), sig, 100.0)


# ── restore survives the mask ─────────────────────────────────────────────────

def _panel_with_superseded_rows():
    """Two rows before money_flow's epoch — exactly what the mask blanks."""
    return pd.DataFrame({
        "signal_date": ["2026-07-01", "2026-07-02"],
        "ticker": ["AAA", "BBB"],
        "money_flow": [0.11, 0.22],
        "tech": [0.30, 0.40],
    })


def test_restored_cells_survive_the_epoch_mask(monkeypatch):
    """The regression that matters, driven through the REAL `build_panel`.

    Restoring a value and then letting the mask blank it by its ORIGINAL date
    leaves the panel exactly where it started. Asserting this against a
    reimplementation of the mask would pass whether or not production honours
    the restore, so it goes through `build_panel(signals_df=...)` — which also
    bypasses the panel cache.
    """
    from src.analysis import signal_panel
    from src.signals.method_epochs import epoch_for

    ep = epoch_for("money_flow")
    assert ep is not None, "test presumes money_flow has an epoch"

    df = _panel_with_superseded_rows()
    assert (df["signal_date"] < ep.isoformat()).all(), "rows must be pre-epoch"

    # Only AAA has a replayed value; BBB stays superseded.
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame(
        {"signal_date": ["2026-07-01"], "ticker": ["AAA"], "money_flow": [0.99]}))
    # Forward returns need no OHLCV for this assertion.
    monkeypatch.setattr(signal_panel, "_forward_returns",
                        lambda *a, **k: pd.DataFrame(), raising=False)

    panel = signal_panel.build_panel(signals_df=df, horizons=(1,))
    got = panel.set_index("ticker")["money_flow"]

    assert got["AAA"] == pytest.approx(0.99), \
        "a RESTORED cell was re-blanked by the epoch mask — restore is a no-op"
    assert pd.isna(got["BBB"]), \
        "an unrestored pre-epoch cell must still be masked"


def test_restore_is_fail_soft_without_the_table(monkeypatch):
    """No replay table => frame untouched, no restored cells, mask still runs."""
    monkeypatch.setattr(replay, "REPLAYABLE", ("money_flow",), raising=False)

    def _boom(*a, **k):
        raise RuntimeError("signals_replay does not exist")

    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", _boom)

    df = _panel_with_superseded_rows()
    out, restored = replay.restore_replayed(df, methods=("money_flow",))
    assert restored == {}
    assert out["money_flow"].tolist() == [0.11, 0.22]


def test_restore_overwrites_only_matched_rows(monkeypatch):
    """A replayed value replaces the stored one; a row with no replay is left
    alone (and stays the mask's business)."""
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame(
        {"signal_date": ["2026-07-01"], "ticker": ["AAA"], "money_flow": [0.99]}))

    df = _panel_with_superseded_rows()
    out, restored = replay.restore_replayed(df, methods=("money_flow",))

    assert out.loc[0, "money_flow"] == pytest.approx(0.99)   # replayed
    assert out.loc[1, "money_flow"] == pytest.approx(0.22)   # untouched
    assert restored["money_flow"].tolist() == [True, False]


def test_restore_matches_the_RUN_not_just_the_day(monkeypatch):
    """`signals` holds ~43 runs per ticker-day and the panel keeps ONE of them.

    Joining the replay on (date, ticker) alone grafts an arbitrary run's score
    onto the panel's row — measured on live data to change 56.8% of values for
    a scorer that had not changed at all, versus 2.3% with run-exact matching.
    """
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-07-01", "2026-07-01"],
        "ticker": ["AAA", "AAA"],
        "generated_at": ["2026-07-01T14:00:00", "2026-07-01T20:00:00"],
        "money_flow": [0.10, 0.90],
    }))

    df = pd.DataFrame({
        "signal_date": ["2026-07-01", "2026-07-01"],
        "ticker": ["AAA", "AAA"],
        "generated_at": ["2026-07-01T14:00:00", "2026-07-01T20:00:00"],
        "money_flow": [0.0, 0.0],
    })
    out, restored = replay.restore_replayed(df, methods=("money_flow",))

    assert out["money_flow"].tolist() == [pytest.approx(0.10), pytest.approx(0.90)], \
        "each run must receive ITS OWN replayed score"
    assert restored["money_flow"].all()


def test_restore_without_run_column_takes_the_panels_last_run(monkeypatch):
    """Fallback when the frame carries no `generated_at`: pick the LAST run, the
    same rule `build_panel` dedupes by — never arbitrary row order."""
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-07-01", "2026-07-01"],
        "ticker": ["AAA", "AAA"],
        # Deliberately out of order so row order and time order disagree.
        "generated_at": ["2026-07-01T20:00:00", "2026-07-01T14:00:00"],
        "money_flow": [0.90, 0.10],
    }))
    df = pd.DataFrame({"signal_date": ["2026-07-01"], "ticker": ["AAA"],
                       "money_flow": [0.0]})
    out, _ = replay.restore_replayed(df, methods=("money_flow",))
    assert out.loc[0, "money_flow"] == pytest.approx(0.90), "expected the LAST run"


def test_restore_preserves_frame_index(monkeypatch):
    """A merge that silently reindexes would misalign every other column in the
    panel against its own forward returns."""
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame(
        {"signal_date": ["2026-07-02"], "ticker": ["BBB"], "money_flow": [0.77]}))

    df = _panel_with_superseded_rows()
    df.index = [17, 42]
    out, restored = replay.restore_replayed(df, methods=("money_flow",))

    assert out.index.tolist() == [17, 42]
    assert out.loc[42, "money_flow"] == pytest.approx(0.77)
    assert out.loc[17, "money_flow"] == pytest.approx(0.11)


# ── scope ─────────────────────────────────────────────────────────────────────

def test_stateful_methods_are_not_claimed_replayable():
    """`pattern` blends a live accuracy registry and `sector_momentum` fetches a
    sector ETF at current time — neither can be rewound to a past date, so
    neither may be advertised as faithfully replayable (measured: pattern 44%
    exact vs ~92% for the real ones)."""
    from src.db.schema import REPLAYABLE_METHOD_COLUMNS
    assert "pattern" not in REPLAYABLE_METHOD_COLUMNS
    assert "sector_momentum" not in REPLAYABLE_METHOD_COLUMNS


def test_replayable_columns_exist_on_the_signals_table():
    from src.db.schema import REPLAYABLE_METHOD_COLUMNS, SIGNAL_METHOD_COLUMNS
    missing = set(REPLAYABLE_METHOD_COLUMNS) - set(SIGNAL_METHOD_COLUMNS)
    assert not missing, f"replay columns absent from signals: {missing}"


# ── tier 1: recovered market conditions ───────────────────────────────────────

def test_context_values_are_recovered_from_the_same_frame():
    """atr/bb/vol come off the SAME technical result that yields the `tech`
    score, and the tape composite from the same bars — all previously computed
    and discarded."""
    df = _frame(200)
    sig = df.index[-1].strftime("%Y-%m-%d")
    hist = replay.visible_history(df, sig, f"{sig}T14:00:00+00:00")
    ctx = replay.replay_context("TEST", hist)
    assert {"atr_pct", "bb_width_pct", "vol_ratio", "movement_factor"} <= set(ctx)
    assert 0.70 <= ctx["movement_factor"] <= 1.30
    if "tape_score" in ctx:
        assert -1.0 <= ctx["tape_score"] <= 1.0


def test_context_cannot_see_future_bars():
    """The tape composite loads the cache itself unless handed a frame, so a
    missing ``df=`` here would silently read TODAY's history while pricing a
    past date. Same probe as the score path."""
    base = _frame(200)
    sig = base.index[-1].strftime("%Y-%m-%d")
    gen = f"{sig}T14:00:00+00:00"
    fut_idx = pd.bdate_range(base.index[-1] + pd.Timedelta(days=1), periods=60)
    fut = pd.DataFrame({"Open": [1e4] * 60, "High": [2e4] * 60, "Low": [5e3] * 60,
                        "Close": [1e4] * 60, "Volume": [9e9] * 60}, index=fut_idx)

    clean = replay.replay_context("TEST", replay.visible_history(base, sig, gen))
    poisoned = replay.replay_context(
        "TEST", replay.visible_history(pd.concat([base, fut]), sig, gen))
    assert clean == poisoned, "future data leaked into the recovered context"


def test_recovered_movement_factor_survives_the_CONFIDENCE_mask(monkeypatch):
    """movement_factor is a CONFIDENCE_EPOCH column, so the confidence mask —
    not the scorer mask — is what would blank it. It is the one component that
    does not depend on the weights, so a recovered value must survive."""
    from src.analysis import signal_panel
    from src.signals.method_epochs import confidence_epoch

    cep = confidence_epoch()
    assert cep is not None, "test presumes the confidence epoch is enabled"

    df = pd.DataFrame({
        "signal_date": ["2026-07-01", "2026-07-01"],
        "ticker": ["AAA", "BBB"],
        "generated_at": ["2026-07-01T14:00:00", "2026-07-01T14:00:00"],
        "movement_factor": [0.0, 0.0],
        "confidence": [0.5, 0.5],
    })
    assert (df["signal_date"] < cep.isoformat()).all(), "rows must be pre-epoch"

    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-07-01"], "ticker": ["AAA"],
        "generated_at": ["2026-07-01T14:00:00"], "movement_factor": [1.21]}))
    monkeypatch.setattr(signal_panel, "_forward_returns",
                        lambda *a, **k: pd.DataFrame(), raising=False)

    panel = signal_panel.build_panel(signals_df=df, horizons=(1,))
    got = panel.set_index("ticker")

    assert got.loc["AAA", "movement_factor"] == pytest.approx(1.21), \
        "recovered movement_factor was re-blanked by the confidence mask"
    assert pd.isna(got.loc["BBB", "movement_factor"]), \
        "an unrecovered pre-epoch component must still be masked"
    # The weight-DEPENDENT components stay masked regardless — they are a
    # backtest, not a recovery.
    assert pd.isna(got.loc["AAA", "confidence"])


def test_context_columns_are_added_when_absent_from_the_panel(monkeypatch):
    """atr_pct & co. were never persisted to `signals`, so restoring them means
    ADDING a column. An absent column is absent to every consumer, so this
    cannot change an existing analysis."""
    import src.db.repo as repo
    monkeypatch.setattr(repo, "fetch_df", lambda *a, **k: pd.DataFrame({
        "signal_date": ["2026-07-01"], "ticker": ["AAA"],
        "generated_at": ["2026-07-01T14:00:00"], "atr_pct": [0.031]}))

    df = pd.DataFrame({"signal_date": ["2026-07-01", "2026-07-01"],
                       "ticker": ["AAA", "BBB"],
                       "generated_at": ["2026-07-01T14:00:00", "2026-07-01T14:00:00"]})
    out, restored = replay.restore_replayed(df, methods=("atr_pct",))
    assert "atr_pct" in out.columns
    assert out.set_index("ticker").loc["AAA", "atr_pct"] == pytest.approx(0.031)
    assert pd.isna(out.set_index("ticker").loc["BBB", "atr_pct"])
    assert restored["atr_pct"].tolist() == [True, False]


def test_shared_tech_result_matches_the_live_scoring_path():
    """The replay shares ONE `compute_technical_score` result between the `tech`
    score and the recovered context (measured ~15% of a full materialisation).

    That shortcut is only safe while it yields exactly what `_score_one` would,
    so pin it: if the live dispatch ever wraps, scales or post-processes `tech`,
    this fails instead of the replay silently diverging from production.
    """
    from src.analysis.technical import compute_technical_score
    from src.signals.multi_timeframe import _score_one

    df = _frame(200)
    sig = df.index[-1].strftime("%Y-%m-%d")
    hist = replay.visible_history(df, sig, f"{sig}T14:00:00+00:00")

    t = compute_technical_score("TEST", df=hist)
    shared = replay.replay_row("TEST", sig, f"{sig}T14:00:00+00:00",
                               methods=("tech",), hist=hist, tech_result=t)
    plain = replay.replay_row("TEST", sig, f"{sig}T14:00:00+00:00",
                              methods=("tech",), hist=hist)

    assert shared["tech"] == plain["tech"] == round(float(_score_one("tech", "TEST", hist, "1d")), 6)


def test_shared_tech_result_does_not_change_the_context():
    df = _frame(200)
    sig = df.index[-1].strftime("%Y-%m-%d")
    hist = replay.visible_history(df, sig, f"{sig}T14:00:00+00:00")
    from src.analysis.technical import compute_technical_score
    assert (replay.replay_context("TEST", hist)
            == replay.replay_context("TEST", hist,
                                     tech_result=compute_technical_score("TEST", df=hist)))


def test_impossible_atr_marks_a_broken_series():
    """The SECOND split detector, for what the price check cannot see.

    `_scale_is_consistent` needs a live-recorded price; a ticker with none
    anywhere in the panel is invisible to it (ADTX: all rows NaN price, a serial
    reverse-splitter, replayed atr_pct 108.0). ATR averages true range over
    price, so exceeding the price at all requires a discontinuity in the bars.
    """
    assert replay._series_is_intact({"atr_pct": 0.0387})     # panel median
    assert replay._series_is_intact({"atr_pct": 0.524})      # p99.9, still real
    assert not replay._series_is_intact({"atr_pct": 44.24})  # observed ADTX
    assert not replay._series_is_intact({"atr_pct": 108.04})
    # Fail OPEN when there is nothing to judge.
    assert replay._series_is_intact({})
    assert replay._series_is_intact({"atr_pct": float("nan")})
