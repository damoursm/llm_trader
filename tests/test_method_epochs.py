"""Scorer-version epochs (2026-07-24).

When a scorer is FIXED, every score already stored was produced by a function
that no longer exists — yet those numbers feed live decisions (the win-rate
filters, the adaptive tilt, the IC-weight layer). The epoch registry withholds
pre-change evidence from the changed method ONLY, so a freshly-fixed scorer is
judged as "unproven" rather than convicted on its predecessor's record.

Retrofitting the stored rows instead was measured and rejected: replaying the
current scorer over the cached OHLCV reproduces the stored value exactly for
only ~62% of rows, because the cache is retroactively split-adjusted and its
per-ticker length differs from what each run held. See the module docstring.
"""

from datetime import date, datetime, timezone

import pytest

from src.signals import method_epochs as me


@pytest.fixture(autouse=True)
def _epoch(monkeypatch):
    monkeypatch.setattr(me, "METHOD_SCORER_EPOCH", {
        "money_flow": datetime(2026, 7, 24, 20, 1, tzinfo=timezone.utc),
        "legacy_day": datetime(2026, 5, 1, tzinfo=timezone.utc),   # midnight epoch
    })


# ── the comparability gate ──────────────────────────────────────────────────

def test_unchanged_method_always_comparable():
    assert me.score_is_comparable("tech", "2020-01-01T00:00:00+00:00")
    assert me.score_is_comparable("tech", None)


def test_score_before_the_change_is_withheld():
    assert not me.score_is_comparable("money_flow", "2026-07-24T13:09:54+00:00")
    assert not me.score_is_comparable("money_flow", "2026-06-01T12:00:00+00:00")


def test_score_after_the_change_counts():
    assert me.score_is_comparable("money_flow", "2026-07-24T20:01:00+00:00")
    assert me.score_is_comparable("money_flow", "2026-07-25T09:30:00+00:00")


def test_mid_day_precision_matters():
    """The regression this exists for: a deploy lands mid-session, so trades
    entered EARLIER the same day carry the old scorer's numbers."""
    assert not me.score_is_comparable("money_flow", "2026-07-24T17:09:34+00:00")
    assert me.score_is_comparable("money_flow", "2026-07-24T20:30:00+00:00")


def test_naive_and_date_only_timestamps_are_handled():
    # Naive → treated as UTC.
    assert not me.score_is_comparable("money_flow", "2026-07-24T10:00:00")
    # Date-only → midnight, so the whole changeover day is conservatively excluded.
    assert not me.score_is_comparable("money_flow", "2026-07-24")
    assert me.score_is_comparable("money_flow", "2026-07-25")
    assert not me.score_is_comparable("money_flow", date(2026, 7, 24))


def test_fails_open_on_an_unparseable_timestamp():
    """A malformed date must never silently erase a method's whole history."""
    assert me.score_is_comparable("money_flow", "not-a-date")
    assert me.score_is_comparable("money_flow", "")


# ── the date-granular view used by the signals panel ────────────────────────

def test_date_epoch_excludes_the_ambiguous_partial_day():
    # Mid-day epoch → panel consumers (which only have signal_date) start the
    # day AFTER, rather than half-admitting the changeover day.
    assert me.epoch_for("money_flow") == date(2026, 7, 25)
    # A midnight epoch is already unambiguous.
    assert me.epoch_for("legacy_day") == date(2026, 5, 1)
    assert me.epoch_for("tech") is None


# ── wiring: the win-rate record actually respects it ────────────────────────

def test_gross_winrate_excludes_pre_epoch_trades(monkeypatch):
    import src.performance.tracker as tracker
    trades = [
        # PRE-epoch: old scorer, bullish view, stock rose — must NOT count.
        {"status": "CLOSED", "entry_date": "2026-07-01",
         "entry_datetime": "2026-07-01T14:00:00+00:00",
         "entry_price": 100.0, "exit_price": 110.0,
         "method_scores": {"money_flow": 0.8, "tech": 0.8}},
        # POST-epoch: current scorer, bullish view, stock fell.
        {"status": "CLOSED", "entry_date": "2026-07-25",
         "entry_datetime": "2026-07-25T14:00:00+00:00",
         "entry_price": 100.0, "exit_price": 90.0,
         "method_scores": {"money_flow": 0.8, "tech": 0.8}},
    ]
    monkeypatch.setattr(tracker, "_load_trades", lambda: trades)
    out = tracker.compute_solo_method_gross_winrate()
    # money_flow is judged ONLY on the post-epoch trade (a loss) …
    assert out["money_flow"] == {"trades": 1, "win_rate": 0.0}
    # … while an unchanged method keeps both.
    assert out["tech"] == {"trades": 2, "win_rate": 50.0}


# ── central masking: build_panel protects every consumer at once ────────────

def test_build_panel_masks_superseded_scores_but_keeps_the_rows(monkeypatch):
    """The panel is the single entry point for every analysis, so the epoch is
    applied THERE — a new analysis is protected by default instead of having to
    remember the registry. The ROW survives: its forward returns and every
    unchanged method column are still valid evidence."""
    import pandas as pd
    from src.analysis import signal_panel as sp

    monkeypatch.setattr(sp, "_epoch_for",
                        lambda m: date(2026, 7, 25) if m == "money_flow" else None)
    raw = pd.DataFrame({
        "signal_date": ["2026-07-20", "2026-07-26"],
        "ticker": ["AAA", "AAA"],
        "generated_at": ["2026-07-20T12:00:00", "2026-07-26T12:00:00"],
        "money_flow": [0.5, 0.6],
        "tech": [0.4, 0.7],
        "price": [10.0, 11.0],
    })
    out = sp.build_panel(horizons=(1,), signals_df=raw)
    pre = out[out["signal_date"] == "2026-07-20"].iloc[0]
    post = out[out["signal_date"] == "2026-07-26"].iloc[0]
    assert pd.isna(pre["money_flow"]), "pre-epoch score must be blanked"
    assert post["money_flow"] == 0.6, "post-epoch score must survive"
    # Unchanged method + the row itself are untouched on BOTH sides.
    assert pre["tech"] == 0.4 and post["tech"] == 0.7
    assert len(out) == 2, "rows are kept — only the superseded column is masked"


def test_build_panel_is_a_noop_without_epochs(monkeypatch):
    import pandas as pd
    from src.analysis import signal_panel as sp
    monkeypatch.setattr(sp, "_epoch_for", lambda m: None)
    raw = pd.DataFrame({
        "signal_date": ["2026-01-01"], "ticker": ["AAA"],
        "generated_at": ["2026-01-01T12:00:00"], "money_flow": [0.5], "price": [10.0],
    })
    out = sp.build_panel(horizons=(1,), signals_df=raw)
    assert out.iloc[0]["money_flow"] == 0.5


# ── wiring: the category-bundle path (compute_macro_eval) ───────────────────
#
# A THIRD, more diluted exposure: the Macro Evaluation table's "Bundle ·
# <category>" rows sum several raw method scores straight from the signals
# table (e.g. money_flow is 1 of 21 members of the "Technical" bundle). A
# superseded member's pre-epoch score must drop out of that sum, not merely
# be excluded from its OWN row (there is no per-method row here — it's rolled
# into the category total).

def test_macro_eval_bundle_excludes_a_superseded_members_pre_epoch_score(monkeypatch):
    import pandas as pd
    from src.db import repo
    import src.performance.tracker as tracker

    monkeypatch.setattr(tracker, "METHOD_CATEGORIES", {"Technical": ["money_flow", "tech"]})
    monkeypatch.setattr(tracker, "_BUNDLE_VIEW_FLOOR", 0.01)

    # PRE-epoch row: money_flow=+0.9 (stale, must be excluded), tech=-0.9 →
    # bundle direction should be SELL (tech alone), not BUY (if money_flow leaked in).
    # POST-epoch row: money_flow=+0.9 (valid) dominates tech=+0.1 → BUY.
    sig = pd.DataFrame({
        "generated_at": ["2026-07-24T10:00:00+00:00", "2026-07-25T10:00:00+00:00"],
        "ticker": ["AAA", "BBB"], "type": ["STOCK", "STOCK"],
        "direction": ["NEUTRAL", "NEUTRAL"], "price": [10.0, 20.0],
        "money_flow": [0.9, 0.9], "tech": [-0.9, 0.1],
    })

    def _fake_fetch_df(sql, *a, **kw):
        if "recommendations" in sql:
            return pd.DataFrame()
        if "FROM signals" in sql:
            return sig
        return pd.DataFrame()

    monkeypatch.setattr(repo, "fetch_df", _fake_fetch_df)

    captured = {}
    orig_build = tracker._build_pseudo_trades

    def _spy_build(calls, *a, **kw):
        # `calls` is the FULL list for one bundle (every ticker's call that
        # tick), not a single ticker — capture each entry, not just the first.
        for c in calls:
            captured[c["ticker"]] = c["action"]
        return orig_build(calls, *a, **kw)

    monkeypatch.setattr(tracker, "_build_pseudo_trades", _spy_build)
    tracker.compute_macro_eval()

    assert captured.get("AAA") == "SELL", "stale money_flow must not flip the pre-epoch bundle direction"
    assert captured.get("BBB") == "BUY", "post-epoch money_flow correctly dominates"
