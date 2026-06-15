"""Extended-hours Phase 0: session classification, session-aware spread,
scheduler observation slots, and extended-session pseudo-trade costs.

The suite-wide conftest fixtures isolate the DB (settings.db_path → tmp file)
and pin commission_model='none', so all hand-math below is spread-only.
"""

import inspect
from datetime import datetime, time, timezone

import pandas as pd
import pytest

from config.settings import settings
from src.performance.market_calendar import current_session
from src.performance.spread import _dynamic_half_spread, _one_side_cost, _pct_return


@pytest.fixture(autouse=True)
def _pinned_session_multipliers(monkeypatch):
    """Pin the extended/overnight spread multipliers so a .env override can't
    skew the hand-math (same rationale as the suite-wide commission pin)."""
    monkeypatch.setattr(settings, "spread_extended_multiplier", 4.0)
    monkeypatch.setattr(settings, "spread_overnight_multiplier", 10.0)


# ── market_calendar.current_session ───────────────────────────────────────

@pytest.mark.parametrize("hh,mm,expected", [
    (9, 30, "rth"),        # open boundary
    (15, 59, "rth"),
    (16, 0, "extended"),   # close boundary → after-hours
    (19, 59, "extended"),
    (20, 0, "overnight"),
    (3, 59, "overnight"),
    (4, 0, "extended"),    # pre-market opens
    (9, 29, "extended"),
])
def test_current_session_boundaries(hh, mm, expected):
    # Naive datetimes are interpreted as Eastern.
    assert current_session(datetime(2026, 6, 10, hh, mm)) == expected


def test_current_session_converts_tz_aware():
    # 20:35 UTC == 16:35 ET → after-hours.
    assert current_session(datetime(2026, 6, 10, 20, 35, tzinfo=timezone.utc)) == "extended"


# ── session-aware half-spread ─────────────────────────────────────────────

@pytest.mark.parametrize("session,mult", [
    (None, 1.0),
    ("rth", 1.0),
    ("extended", 4.0),
    ("overnight", 10.0),
])
def test_half_spread_session_multiplier(session, mult):
    base = _dynamic_half_spread(100.0, "STOCK")          # mega-cap tier, 3 bp
    assert _dynamic_half_spread(100.0, "STOCK", session) == pytest.approx(base * mult)


def test_session_multiplier_applies_to_spread_not_commission(monkeypatch):
    """Commission is session-independent — only the spread term widens."""
    monkeypatch.setattr(settings, "commission_model", "ibkr_fixed")
    monkeypatch.setattr(settings, "commission_buffer", 1.0)
    spread_rth = _dynamic_half_spread(100.0, "STOCK")
    commission = _one_side_cost(100.0, "STOCK") - spread_rth
    assert commission > 0
    assert _one_side_cost(100.0, "STOCK", "extended") == pytest.approx(
        spread_rth * 4.0 + commission
    )


def test_pct_return_extended_entry_leg():
    """Long $100 → $110, entered after-hours, exited at an RTH mark.

    Entry: mega-cap 3 bp × 4 (extended) = 12 bp → eff_entry = 100.12
    Exit:  3 bp RTH                              → eff_exit  = 109.967
    return = (109.967 − 100.12) / 100.12 × 100 ≈ +9.835 %  (vs +9.934 % RTH)
    """
    r_ext = _pct_return("BUY", 100.0, 110.0, "STOCK", entry_session="extended")
    r_rth = _pct_return("BUY", 100.0, 110.0, "STOCK")
    assert r_ext == pytest.approx(9.835, abs=1e-3)
    assert r_ext < r_rth


# ── scheduler slots ───────────────────────────────────────────────────────

def _slot_times(slots, kind):
    return [t.strftime("%H:%M") for t, k in slots if k == kind]


def test_session_slots_off_mode_has_no_extended(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "off")
    from src.scheduler.runner import _session_slots
    slots, end_t = _session_slots()
    assert _slot_times(slots, "extended") == []
    assert _slot_times(slots, "rth")[0] == "09:30"
    assert _slot_times(slots, "rth")[-1] == "16:00"
    assert end_t == time(16, 0)


def test_session_slots_observe_mode_excludes_rth_collisions(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    monkeypatch.setattr(settings, "extended_windows", "07:00-09:30,16:00-17:30")
    monkeypatch.setattr(settings, "extended_tick_minutes", 30)
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    # 09:30 and 16:00 belong to the RTH slot set — extended must cede them.
    assert _slot_times(slots, "extended") == [
        "07:00", "07:30", "08:00", "08:30", "09:00", "16:30", "17:00", "17:30",
    ]
    # Merged list is time-ordered.
    times = [t for t, _ in slots]
    assert times == sorted(times)


def test_session_slots_bad_windows_ignored(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    monkeypatch.setattr(settings, "extended_windows", "garbage,25:00-26:00,10:00-09:00,16:00-17:00")
    monkeypatch.setattr(settings, "extended_tick_minutes", 15)
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    # Only the one valid window survives; 15-min cadence; 16:00 cedes to RTH.
    assert _slot_times(slots, "extended") == ["16:15", "16:30", "16:45", "17:00"]


def test_current_slot_returns_kind(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    monkeypatch.setattr(settings, "extended_windows", "16:00-17:30")
    monkeypatch.setattr(settings, "extended_tick_minutes", 30)
    from src.scheduler.runner import _current_slot, _session_slots
    slots, _ = _session_slots()
    # Wednesday 2026-06-10.
    slot = _current_slot(datetime(2026, 6, 10, 16, 40), slots)
    assert slot is not None and slot[1] == "extended"
    assert slot[0].time() == time(16, 30)
    slot = _current_slot(datetime(2026, 6, 10, 12, 5), slots)
    assert slot is not None and slot[1] == "rth"
    # Saturday → no slot at all.
    assert _current_slot(datetime(2026, 6, 13, 12, 5), slots) is None


def test_run_pipeline_accepts_observe_only():
    """The runner passes observe_only= on every extended tick — losing the
    kwarg would make the scheduler raise (and log) on each one."""
    from src.pipeline import run_pipeline
    assert "observe_only" in inspect.signature(run_pipeline).parameters


# ── extended recommendation → pseudo-trade with extended entry cost ───────

def test_llm_perf_extended_rec_bears_extended_entry_cost(monkeypatch):
    from src.db import repo
    import src.data.cache as cache_mod
    from src.performance import tracker

    run_id = "2026-06-09_203500"
    repo.insert_run({
        "run_id": run_id,
        "started_at": "2026-06-09T20:35:00+00:00",
        "llm_synthesis_provider": "deepseek",
        "llm_sentiment_provider": "deepseek×5",
    })
    # 20:35 UTC = 16:35 ET → after-hours recommendation.
    repo.insert_recommendations([{
        "run_id": run_id,
        "generated_at": "2026-06-09T20:35:00+00:00",
        "ticker": "TEST",
        "type": "STOCK",
        "direction": "BULLISH",
        "action": "BUY",
        "confidence": 0.9,
        "actionable": False,
        "llm_provider": "deepseek-v4-flash",
    }])
    repo.insert_signals(
        run_id, generated_at="2026-06-09T20:35:00+00:00", signal_date="2026-06-09",
        rows=[{"ticker": "TEST", "type": "STOCK", "direction": "BULLISH",
               "combined_score": 0.5, "confidence": 0.9, "price": 100.0,
               "scores": {}}],
    )

    bars = pd.DataFrame(
        {"Close": [100.0, 110.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-06-09"), pd.Timestamp("2026-06-10")]),
    )
    monkeypatch.setattr(cache_mod, "load_ohlcv", lambda ticker: bars)

    perf = tracker._compute_llm_perf()
    stats = (perf.get("synthesis") or {}).get("deepseek-v4-flash")
    assert stats is not None and stats["trades"] == 1
    # Extended entry leg: 3 bp × 4 = 12 bp → +9.835 % (RTH entry would be +9.934 %).
    # Segment stats round avg_return to 2 dp, so allow that quantisation.
    assert stats["avg_return"] == pytest.approx(9.835, abs=0.01)

    # Session bucketing: the rec lands in 'extended', not 'rth'.
    ext = tracker._compute_llm_perf(session="extended")
    assert (ext.get("synthesis") or {}).get("deepseek-v4-flash", {}).get("trades") == 1
    rth = tracker._compute_llm_perf(session="rth")
    assert (rth.get("synthesis") or {}).get("deepseek-v4-flash") is None


# ── per-window cadence + holiday awareness (4AM–8PM coverage) ─────────────

def test_session_slots_per_window_cadence(monkeypatch):
    """Dead zones tick hourly via the @60 suffix; shoulders keep the 30-min
    default — full 04:00–20:00 coverage without 32 LLM runs a day."""
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    monkeypatch.setattr(
        settings, "extended_windows",
        "04:00-07:00@60,07:00-09:30,16:00-17:30,18:00-20:00@60",
    )
    monkeypatch.setattr(settings, "extended_tick_minutes", 30)
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    assert _slot_times(slots, "extended") == [
        "04:00", "05:00", "06:00",
        "07:00", "07:30", "08:00", "08:30", "09:00",   # 09:30 ceded to RTH
        "16:30", "17:00", "17:30",                      # 16:00 ceded to RTH
        "18:00", "19:00", "20:00",
    ]


def test_session_slots_single_point_window_for_pre_close_tick(monkeypatch):
    """A window with equal endpoints (19:50-19:50) is ONE slot at exactly that
    time. The default schedule uses it for the last after-hours tick: the
    pipeline needs ~4 min from tick to order submission AND the every-tick
    hold-review adds more, so a 20:00 (or 19:55) slot's orders reached IBKR after
    the session close and could never fill same-day; 19:50 leaves them a live
    book with a ~10-min buffer."""
    monkeypatch.setattr(settings, "extended_hours_mode", "trade")
    monkeypatch.setattr(settings, "extended_windows", "18:00-19:00@60,19:50-19:50")
    monkeypatch.setattr(settings, "extended_tick_minutes", 30)
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    assert _slot_times(slots, "extended") == ["18:00", "19:00", "19:50"]


def test_parse_windows_bad_cadence_token_ignored(monkeypatch):
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    monkeypatch.setattr(settings, "extended_windows", "04:00-06:00@xx,18:00-19:00@60")
    monkeypatch.setattr(settings, "extended_tick_minutes", 30)
    from src.scheduler.runner import _session_slots
    slots, _ = _session_slots()
    assert _slot_times(slots, "extended") == ["18:00", "19:00"]


def test_current_slot_skips_nyse_holidays(monkeypatch):
    """No session — regular or extended — on a closed market: Juneteenth 2026
    (Fri 06-19) and observed Independence Day (Fri 07-03) yield no slot."""
    monkeypatch.setattr(settings, "extended_hours_mode", "observe")
    monkeypatch.setattr(settings, "extended_windows", "04:00-07:00@60")
    from src.scheduler.runner import _current_slot, _session_slots
    slots, _ = _session_slots()
    assert _current_slot(datetime(2026, 6, 19, 12, 5), slots) is None
    assert _current_slot(datetime(2026, 7, 3, 5, 30), slots) is None
    # The Thursday before Juneteenth is a normal session.
    assert _current_slot(datetime(2026, 6, 18, 12, 5), slots) is not None


# ── extended-session gap momentum scorer (ext_gap) ────────────────────────

def _gap_bars(last_day: str, n: int = 30, close: float = 100.0):
    """Completed daily bars ending at *last_day* with constant close and a
    constant 2-point daily range → ATR = 2.0 exactly (TR is constant)."""
    idx = pd.bdate_range(end=pd.Timestamp(last_day), periods=n)
    return pd.DataFrame(
        {"High": close + 1.0, "Low": close - 1.0, "Close": close},
        index=idx,
    )


@pytest.fixture()
def _pinned_gap_settings(monkeypatch):
    monkeypatch.setattr(settings, "enable_extended_gap", True)
    monkeypatch.setattr(settings, "extended_gap_deadband_atr", 0.25)
    monkeypatch.setattr(settings, "extended_gap_scale_atr", 1.5)


def test_ext_gap_premarket_references_previous_close(monkeypatch, _pinned_gap_settings):
    """Pre-market Wed 2026-06-10 08:00 ET: ref close = Tue 06-09. ATR = 2 →
    a +1% gap is 0.5 ATR → tanh(0.5/1.5) = 0.3215 → +0.322 at 3 dp."""
    import src.signals.extended_session as ext
    monkeypatch.setattr(ext, "load_ohlcv", lambda t: _gap_bars("2026-06-09"))
    score, gap_pct = ext.compute_extended_gap_score(
        "TEST", 101.0, session="extended", now=datetime(2026, 6, 10, 8, 0),
    )
    assert gap_pct == pytest.approx(1.0, abs=1e-6)
    assert score == pytest.approx(0.322, abs=1e-3)


def test_ext_gap_afterhours_references_todays_close(monkeypatch, _pinned_gap_settings):
    """After-hours 17:00: today's bar is complete → ref close = today."""
    import src.signals.extended_session as ext
    monkeypatch.setattr(ext, "load_ohlcv", lambda t: _gap_bars("2026-06-10"))
    score, gap_pct = ext.compute_extended_gap_score(
        "TEST", 98.0, session="extended", now=datetime(2026, 6, 10, 17, 0),
    )
    assert gap_pct == pytest.approx(-2.0, abs=1e-6)
    # -2% = 1 ATR → tanh(1/1.5) ≈ -0.583
    assert score == pytest.approx(-0.583, abs=1e-3)


def test_ext_gap_deadband_and_rth_zero(monkeypatch, _pinned_gap_settings):
    import src.signals.extended_session as ext
    monkeypatch.setattr(ext, "load_ohlcv", lambda t: _gap_bars("2026-06-09"))
    # |gap| = 0.3% = 0.15 ATR < 0.25 deadband → no view, raw gap preserved.
    score, gap_pct = ext.compute_extended_gap_score(
        "TEST", 100.3, session="extended", now=datetime(2026, 6, 10, 8, 0),
    )
    assert score == 0.0 and gap_pct == pytest.approx(0.3, abs=1e-6)
    # RTH runs never produce a view (open gap already in the technical stack).
    assert ext.compute_extended_gap_score(
        "TEST", 105.0, session="rth", now=datetime(2026, 6, 10, 11, 0),
    ) == (0.0, 0.0)


def test_ext_gap_fails_closed_on_stale_cache(monkeypatch, _pinned_gap_settings):
    """Cache ends 3 sessions before the expected reference close → 0.0 (a gap
    vs an old close is a phantom), not a huge score."""
    import src.signals.extended_session as ext
    monkeypatch.setattr(ext, "load_ohlcv", lambda t: _gap_bars("2026-06-04"))
    score, gap_pct = ext.compute_extended_gap_score(
        "TEST", 110.0, session="extended", now=datetime(2026, 6, 10, 8, 0),
    )
    assert (score, gap_pct) == (0.0, 0.0)


# ── extended-session weight overlay ───────────────────────────────────────

def test_extended_weight_overlay_scales_stale_and_fresh(monkeypatch):
    monkeypatch.setattr(settings, "extended_stale_options_weight_mult", 0.5)
    monkeypatch.setattr(settings, "extended_news_weight_mult", 1.25)
    from src.signals.aggregator import _BASE_WEIGHTS, _extended_session_weight_overlay
    out = _extended_session_weight_overlay(dict(_BASE_WEIGHTS))
    # Options-derived (frozen at the RTH close) halved …
    for m in ("put_call", "max_pain", "oi_skew", "iv_expr"):
        assert out[m] == pytest.approx(_BASE_WEIGHTS[m] * 0.5)
    # … live off-hours information up-weighted …
    for m in ("news", "sent_velocity", "ext_gap"):
        assert out[m] == pytest.approx(_BASE_WEIGHTS[m] * 1.25)
    # … and everything else untouched.
    assert out["tech"] == pytest.approx(_BASE_WEIGHTS["tech"])
    assert out["momentum"] == pytest.approx(_BASE_WEIGHTS["momentum"])


# ── earnings blackout: post-release exemption ─────────────────────────────

def test_blackout_exempts_already_reported(caplog):
    """A ticker whose actual EPS is already in the PEAD context (report out)
    must NOT be blacked out — the binary event has resolved and the
    PEAD / gap-reaction entry is exactly the trade the system wants."""
    from types import SimpleNamespace
    from src.data.catalyst_timing import compute_catalyst_context

    earnings_context = SimpleNamespace(upcoming=[
        SimpleNamespace(ticker="AAPL", days_until=0),   # reported (see PEAD below)
        SimpleNamespace(ticker="MSFT", days_until=1),   # genuinely upcoming
    ])
    pead_context = SimpleNamespace(signals=[
        SimpleNamespace(ticker="AAPL", days_since_report=0),
        SimpleNamespace(ticker="NVDA", days_since_report=30),  # stale: no effect
    ])
    ctx = compute_catalyst_context(
        earnings_context=earnings_context, pead_context=pead_context,
    )
    assert "MSFT" in ctx.earnings_blackout_tickers
    assert "AAPL" not in ctx.earnings_blackout_tickers

    # Without the PEAD evidence both stay blocked (legacy behavior).
    ctx2 = compute_catalyst_context(earnings_context=earnings_context)
    assert set(ctx2.earnings_blackout_tickers) == {"AAPL", "MSFT"}
