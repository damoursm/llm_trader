"""Uniform contract every OHLCV scorer must satisfy (2026-07-25).

Written after an audit found `pattern_recognition` (349 LOC, weight 0.18, rank 5
of 21) had ZERO tests — it appeared in the suite only as a flag being switched
off. Rather than write bespoke tests for one module, this asserts the contract
across ALL of them through `multi_timeframe._score_one`, the single uniform
entry point, so a new scorer is covered the moment it joins TECHNICAL_METHODS.

The three properties, each earned from a real bug:

1. **Bounded** — `[-1, +1]` is the documented contract the combine, the IC
   tables and the inversion logic all depend on.
2. **Responds to direction** — a scorer that returns the same number for a tape
   and its MIRROR cannot express direction at all. That is exactly the
   `money_flow` bug found 2026-07-24 (two defects, both silent: the MFI
   contrarian term cancelled CMF, and the OBV z-score measured acceleration
   rather than level).
3. **Fail-soft on degenerate input, and NOT hallucinating a view** — an empty /
   one-row / all-NaN frame must produce ~no view, never a confident direction.
   `_score_one` swallows exceptions into `None`, which silently DROPS the
   method from the frame, so a crash here is invisible in production.

Property 3 caught a live bug: a single TRAILING NaN bar (yfinance's forming-bar
row, present in ~1% of cached frames) swung `tech` by +0.47 on identical price
history. Fixed in `market_data._trim_trailing_nan_bars`.

All synthetic, no network.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals.multi_timeframe import TECHNICAL_METHODS, _score_one

# `sector_momentum` deliberately ignores the passed frame — it re-fetches BOTH
# legs at the interval because it needs the benchmark ETF at the same candle
# (documented in multi_timeframe._score_frame). It therefore cannot be driven by
# a synthetic frame and is excluded from the frame-driven properties below.
_DF_DRIVEN = tuple(m for m in TECHNICAL_METHODS if m != "sector_momentum")

# `pattern` scores against a LEARNED per-ticker library (cache/patterns/*.json).
# With no library for a synthetic ticker it correctly returns 0.0 — that is
# fail-soft, not an inability to express direction, so it is exempt from the
# mirror-tape property but still bound by the others.
_MIRROR_EXEMPT = {"pattern"}


def _frame(closes, volume=1e6):
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if n == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return pd.DataFrame(
        {"Open": c * 0.995, "High": c * 1.01, "Low": c * 0.99, "Close": c,
         "Volume": np.full(n, float(volume))},
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )


_RISING = _frame(np.linspace(100, 200, 260))
_FALLING = _frame(np.linspace(200, 100, 260))


# ── 1. bounded ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", _DF_DRIVEN)
@pytest.mark.parametrize("label", ["rising", "falling", "flat", "volatile"])
def test_score_is_within_the_documented_range(method, label):
    frames = {
        "rising": _RISING, "falling": _FALLING,
        "flat": _frame([100.0] * 260),
        "volatile": _frame(100 + 30 * np.sin(np.arange(260) / 3.0)),
    }
    s = _score_one(method, "TEST", frames[label], "1d")
    if s is None:
        pytest.skip(f"{method} produced no score on {label}")
    assert -1.0 - 1e-9 <= s <= 1.0 + 1e-9, f"{method} out of [-1,+1] on {label}: {s}"


# ── 2. direction-aware ─────────────────────────────────────────────────────

@pytest.mark.parametrize("method", [m for m in _DF_DRIVEN if m not in _MIRROR_EXEMPT])
def test_scorer_responds_to_a_mirrored_tape(method):
    """A scorer returning the same value for a tape and its mirror cannot
    express direction — the `money_flow` failure mode, which was invisible for
    months because the score still looked plausible."""
    up = _score_one(method, "TEST", _RISING, "1d")
    down = _score_one(method, "TEST", _FALLING, "1d")
    assert up is not None and down is not None, f"{method} scored nothing"
    assert abs(up - down) > 1e-6, (
        f"{method} returns {up} for BOTH a rising and a falling tape — it "
        "cannot express direction")


# ── 3. fail-soft, without inventing a view ─────────────────────────────────

@pytest.mark.parametrize("method", _DF_DRIVEN)
@pytest.mark.parametrize("label", ["empty", "single_row", "five_rows"])
def test_degenerate_frames_yield_no_view(method, label):
    """Too little data must mean NO VIEW (~0), never a confident direction.
    `_score_one` turns an exception into None, which silently drops the method
    from the frame — so a crash here would be invisible in production."""
    frames = {"empty": _frame([]), "single_row": _frame([100.0]),
              "five_rows": _frame([100, 101, 102, 103, 104])}
    s = _score_one(method, "TEST", frames[label], "1d")
    assert s is not None, f"{method} raised on a {label} frame (silently dropped in prod)"
    assert abs(s) < 0.05, f"{method} invented a view ({s}) from a {label} frame"


@pytest.mark.parametrize("method", _DF_DRIVEN)
def test_all_nan_close_yields_no_view_through_the_production_path(method):
    """A frame with no usable Close carries zero information, and emitting a
    direction from it would be a silent fabrication reaching the combine.

    The guard lives at the `market_data` boundary, not inside each scorer:
    every scoring frame arrives via `get_history` → `_drop_forming_bar` →
    `_trim_trailing_nan_bars`, which reduces an all-NaN frame to empty, and
    every scorer already returns no view on empty. Asserted through that path
    because that is where the invariant is enforced — three scorers (`tech`,
    `money_flow`, `trend_strength`) DO fabricate a view (+0.40 / −0.58 / +0.20)
    when handed the raw frame directly, so bypassing the sanitiser is unsafe.
    """
    from src.data.market_data import _trim_trailing_nan_bars
    df = _frame([100.0] * 260)
    df["Close"] = np.nan
    sanitised = _trim_trailing_nan_bars(df)
    assert sanitised.empty, "an all-NaN Close must sanitise to an empty frame"
    s = _score_one(method, "TEST", sanitised, "1d")
    assert s is not None, f"{method} raised on an empty frame"
    assert abs(s) < 0.05, f"{method} invented a view ({s}) from no price data"


# ── the live bug property 3 found ──────────────────────────────────────────

@pytest.mark.parametrize("method", _DF_DRIVEN)
def test_a_trailing_nan_bar_does_not_change_the_score(method):
    """yfinance appends a NaN row for the still-forming bar, and ~1% of cached
    daily frames carry one. It holds no information, so it must not move any
    score — before `market_data._trim_trailing_nan_bars` it swung `tech`
    (weight 0.30) by +0.47, flipping it from bearish to bullish.
    """
    from src.data.market_data import _trim_trailing_nan_bars
    # APPEND a NaN bar rather than blanking the last real one — blanking would
    # remove a genuine bar, so any delta would be correct rather than a bug.
    nan_row = pd.DataFrame(
        {"Open": [np.nan], "High": [np.nan], "Low": [np.nan], "Close": [np.nan],
         "Volume": [0.0]},
        index=[_RISING.index[-1] + pd.Timedelta(days=1)])
    dirty = pd.concat([_RISING, nan_row])
    clean_s = _score_one(method, "TEST", _RISING, "1d")
    dirty_s = _score_one(method, "TEST", _trim_trailing_nan_bars(dirty), "1d")
    if clean_s is None or dirty_s is None:
        pytest.skip(f"{method} produced no score")
    assert abs(clean_s - dirty_s) < 1e-9, (
        f"{method}: a trailing NaN bar moved the score {clean_s} → {dirty_s}")


def test_trim_helper_only_touches_the_tail():
    from src.data.market_data import _trim_trailing_nan_bars
    assert len(_trim_trailing_nan_bars(_RISING)) == 260        # clean untouched
    one = _RISING.copy(); one.iloc[-1, :4] = np.nan
    assert len(_trim_trailing_nan_bars(one)) == 259
    three = _RISING.copy(); three.iloc[-3:, :4] = np.nan
    assert len(_trim_trailing_nan_bars(three)) == 257
    interior = _RISING.copy(); interior.iloc[100:110, :4] = np.nan
    assert len(_trim_trailing_nan_bars(interior)) == 260, (
        "interior gaps are REAL data — trimming them would change bar spacing")


# ── the guard belongs at the CACHE, not only at market_data ────────────────

def test_cache_trims_trailing_nan_at_parse_time(tmp_path, monkeypatch):
    """Several scorers read `cache.load_ohlcv` DIRECTLY, bypassing
    `market_data.get_history` — agreement's tape confirmation, classic_anomalies,
    anchored_vwap, cointegration, extended_session. Guarding only the
    market_data path left those exposed, so the trim runs at the single
    cache-read entry point instead.
    """
    import json
    import src.data.cache as cache

    df = _RISING.copy()
    nan_row = pd.DataFrame(
        {"Open": [np.nan], "High": [np.nan], "Low": [np.nan], "Close": [np.nan],
         "Volume": [0.0]},
        index=[df.index[-1] + pd.Timedelta(days=1)])
    dirty = pd.concat([df, nan_row])

    d = tmp_path / "ohlcv"
    d.mkdir()
    (d / "ZZTEST.json").write_text(dirty.to_json(orient="split"), encoding="utf-8")
    monkeypatch.setattr(cache, "OHLCV_CACHE_DIR", d, raising=False)
    monkeypatch.setattr(cache, "_OHLCV_PARSE_CACHE", type(cache._OHLCV_PARSE_CACHE)())
    monkeypatch.setattr(cache, "_OHLCV_PARSE_BYTES", 0, raising=False)

    loaded = cache.load_ohlcv("ZZTEST")
    if loaded is None:
        pytest.skip("cache dir not monkeypatchable by that name")
    assert len(loaded) == len(df), "the trailing NaN bar survived the cache read"
    assert not pd.isna(pd.to_numeric(loaded["Close"], errors="coerce").iloc[-1])


def test_tape_confirmation_survives_a_trailing_nan_bar():
    """The tape feeds the confidence multiplier on EVERY ticker. Before the
    cache-level trim a trailing NaN bar silently blanked it from a real reading
    to NO_DATA — a fail-SAFE degradation, but ~17 tickers per run losing the
    signal for no reason."""
    from src.data.cache import trim_trailing_nan_bars
    from src.signals.agreement import compute_tape_confirmation

    nan_row = pd.DataFrame(
        {"Open": [np.nan], "High": [np.nan], "Low": [np.nan], "Close": [np.nan],
         "Volume": [0.0]},
        index=[_RISING.index[-1] + pd.Timedelta(days=1)])
    dirty = pd.concat([_RISING, nan_row])

    clean_tape = compute_tape_confirmation("T", df=_RISING)
    trimmed_tape = compute_tape_confirmation("T", df=trim_trailing_nan_bars(dirty))
    assert clean_tape.label != "NO_DATA"
    assert trimmed_tape.label == clean_tape.label
    assert trimmed_tape.score == pytest.approx(clean_tape.score)


def test_market_data_and_cache_share_one_implementation():
    """Two copies would drift. market_data keeps a thin alias because
    _drop_forming_bar also sees frames straight from a provider fetch."""
    from src.data.cache import trim_trailing_nan_bars as cache_impl
    from src.data.market_data import _trim_trailing_nan_bars as md_alias
    nan_row = pd.DataFrame(
        {"Open": [np.nan], "High": [np.nan], "Low": [np.nan], "Close": [np.nan],
         "Volume": [0.0]},
        index=[_RISING.index[-1] + pd.Timedelta(days=1)])
    dirty = pd.concat([_RISING, nan_row])
    assert len(md_alias(dirty)) == len(cache_impl(dirty)) == len(_RISING)
