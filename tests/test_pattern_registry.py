"""Live pattern outcome registry (`src/signals/pattern_registry.py`).

The synthetic per-ticker pattern library says "when this shape formed, did price
move the expected way?". This registry says something different and harder: when
the SYSTEM actually traded on the pattern — after 9 other signals, the regime
gate, the earnings blackout and the confidence threshold had their say — what
happened? Its output feeds back into `pattern_recognition`'s score, so a
mis-aggregated bucket becomes a live scoring error.

Two properties carry most of the risk and both are easy to break silently:

* **`won` is GROSS, `avg_return` is COST-ADJUSTED** — the same split the rest of
  the system uses (CLAUDE.md's win-rate convention). They are computed from
  DIFFERENT fields (`is_gross_win` on raw prices vs `return_pct`), so a "tidy-up"
  that derives one from the other silently rewrites the pattern's win rate.
* **`pattern_correct` is NOT `won`** — the system trades against a pattern's
  inherent direction whenever the other methods outvote it, and the registry
  tracks both so a pattern that predicts well but gets traded badly is
  distinguishable from one that is simply wrong.

Every test points `REGISTRY_PATH` at tmp_path: the module-level default is the
RELATIVE `cache/pattern_registry.json`, so an unpatched test would read and
overwrite the developer's real registry.
"""

from __future__ import annotations

import json

import pytest

from src.signals import pattern_registry as pr


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "REGISTRY_PATH", tmp_path / "pattern_registry.json")


def _trade(pattern="double_bottom", action="BUY", entry=100.0, exit_=110.0,
           ret=9.0, ticker="MSFT", entry_date="2026-04-15", exit_date="2026-04-22",
           status="CLOSED", **kw) -> dict:
    t = {
        "ticker": ticker, "action": action, "status": status,
        "pattern_at_entry": pattern, "entry_price": entry, "exit_price": exit_,
        "return_pct": ret, "entry_date": entry_date, "exit_date": exit_date,
        "exit_reason": "signal_decay",
    }
    t.update(kw)
    return t


# ── the pattern-direction test ──────────────────────────────────────────────

def test_pattern_correct_is_action_times_outcome_against_inherent_direction():
    """`double_bottom` is bullish (+1): the pattern was RIGHT whenever price rose,
    whichever way the system traded it. A BUY that won and a SELL that lost are
    the same statement about the pattern."""
    c = pr._pattern_was_correct
    assert c("double_bottom", "BUY", +5.0) is True      # bought, price rose
    assert c("double_bottom", "SELL", -5.0) is True     # shorted, lost -> price rose
    assert c("double_bottom", "BUY", -5.0) is False
    assert c("double_bottom", "SELL", +5.0) is False    # shorted, won -> price fell

    # ...and the mirror for a bearish pattern.
    assert c("double_top", "SELL", +5.0) is True        # shorted, won -> price fell
    assert c("double_top", "BUY", -5.0) is True
    assert c("double_top", "BUY", +5.0) is False


def test_unknown_pattern_is_never_credited_as_correct():
    """A pattern with no entry in `_PATTERN_DIR` has no inherent direction, so
    there is nothing to be right about — it must not default to True and inflate
    every accuracy figure."""
    assert pr._pattern_was_correct("not_a_pattern", "BUY", +5.0) is False
    assert pr._pattern_was_correct("", "BUY", +5.0) is False


def test_every_known_pattern_has_a_direction():
    """The registry's accuracy column is meaningless for a pattern the direction
    map doesn't know, and the failure is silent (accuracy pinned at 0)."""
    from src.signals.pattern_recognition import _PATTERN_DIR
    assert _PATTERN_DIR, "pattern direction map is empty"
    assert all(d in (+1, -1) for d in _PATTERN_DIR.values())


# ── the gross/cost split ────────────────────────────────────────────────────

def test_won_is_GROSS_while_avg_return_stays_cost_adjusted():
    """The decisive case, mirroring tests/test_gross_winrate_convention.py: a
    trade whose direction was RIGHT but whose move was smaller than the round
    trip. It is a WIN (the pattern and the system both pointed the right way)
    carried alongside a NEGATIVE average return (trading it cost more than it
    made). Deriving one from the other collapses two different questions."""
    t = _trade(entry=100.0, exit_=100.20, ret=-0.35)     # +0.2% gross, -0.35% net
    assert pr.record_outcome(t) is True
    stats = pr.pattern_stats("double_bottom")
    assert stats["n_wins"] == 1, "gross direction was right — must count as a win"
    assert stats["win_rate"] == 1.0
    assert stats["avg_return"] < 0, "cost-adjusted return must stay negative"


def test_flat_round_trip_is_not_a_win():
    """`is_gross_win` treats exit == entry as a loss (the direction did not pay)
    — the registry inherits that rather than re-deciding it."""
    assert pr.record_outcome(_trade(entry=100.0, exit_=100.0, ret=0.0)) is True
    assert pr.pattern_stats("double_bottom")["n_wins"] == 0


def test_unusable_prices_are_skipped_entirely(monkeypatch):
    """`is_gross_win` returns None when the price pair can't yield an outcome.
    Guessing (or defaulting to False) would let a data gap masquerade as a
    losing pattern — the trade must leave BOTH numerator and denominator."""
    t = _trade(entry=None, exit_=None)
    assert pr.record_outcome(t) is False
    assert pr.pattern_stats("double_bottom") is None


# ── aggregation arithmetic ──────────────────────────────────────────────────

def test_running_average_matches_the_plain_mean():
    """`avg_return` is accumulated incrementally (avg += (x-avg)/n) rather than
    stored as a list, so an off-by-one in n silently skews every pattern."""
    rets = [10.0, -4.0, 6.0, 1.0]
    reg = pr.load_registry()
    for i, r in enumerate(rets):
        pr.record_outcome(_trade(ret=r, exit_=100.0 + r, entry_date=f"2026-04-{i + 1:02d}"),
                          reg=reg)
    pr.save_registry(reg)
    stats = pr.pattern_stats("double_bottom")
    assert stats["n_trades"] == 4
    assert stats["avg_return"] == pytest.approx(sum(rets) / len(rets), abs=1e-4)


def test_buys_and_sells_are_bucketed_separately():
    reg = pr.load_registry()
    pr.record_outcome(_trade(action="BUY", ret=8.0, exit_=108.0), reg=reg)
    pr.record_outcome(_trade(action="SELL", ret=-3.0, exit_=103.0,
                             entry_date="2026-05-01"), reg=reg)
    pr.save_registry(reg)
    s = pr.pattern_stats("double_bottom")
    assert s["buys"]["n"] == 1 and s["sells"]["n"] == 1
    assert s["buys"]["wins"] == 1                       # BUY, price up
    assert s["sells"]["wins"] == 0                      # SELL, price up -> lost
    # The pattern was bullish and price rose BOTH times, so it was correct twice
    # even though the system only made money once. This is the whole reason
    # pattern_accuracy exists next to win_rate.
    assert s["pattern_accuracy"] == 1.0
    assert s["win_rate"] == 0.5


def test_ticker_pattern_bucket_is_scoped_to_the_ticker():
    reg = pr.load_registry()
    pr.record_outcome(_trade(ticker="MSFT", ret=5.0, exit_=105.0), reg=reg)
    pr.record_outcome(_trade(ticker="AAPL", ret=-5.0, exit_=95.0), reg=reg)
    pr.save_registry(reg)
    assert pr.ticker_pattern_stats("MSFT", "double_bottom")["n"] == 1
    assert pr.ticker_pattern_stats("AAPL", "double_bottom")["wins"] == 0
    assert pr.ticker_pattern_stats("msft", "double_bottom")["n"] == 1   # case-folded
    assert pr.ticker_pattern_stats("TSLA", "double_bottom") is None
    assert pr.ticker_pattern_stats("", "double_bottom") is None
    assert pr.ticker_pattern_stats("MSFT", "") is None


# ── idempotency ─────────────────────────────────────────────────────────────

def test_recording_the_same_trade_twice_is_a_no_op():
    """`update_open_trades` re-walks the whole ledger every tick, so a
    non-idempotent registry would inflate n_trades without bound — and the
    pattern score reads n_trades as evidence weight."""
    t = _trade()
    assert pr.record_outcome(t) is True
    assert pr.record_outcome(t) is False
    assert pr.pattern_stats("double_bottom")["n_trades"] == 1


def test_record_batch_is_idempotent_across_runs():
    trades = [_trade(ticker="MSFT"), _trade(ticker="AAPL", ret=-2.0, exit_=98.0)]
    assert pr.record_batch(trades) == 2
    assert pr.record_batch(trades) == 0                 # re-run on same ledger
    assert pr.pattern_stats("double_bottom")["n_trades"] == 2


def test_identity_is_ticker_plus_both_dates():
    """Same ticker and pattern, different round trip -> a genuinely new outcome."""
    reg = pr.load_registry()
    pr.record_outcome(_trade(entry_date="2026-04-15", exit_date="2026-04-22"), reg=reg)
    assert pr.record_outcome(_trade(entry_date="2026-05-15", exit_date="2026-05-22"),
                             reg=reg) is True
    pr.save_registry(reg)
    assert pr.pattern_stats("double_bottom")["n_trades"] == 2


# ── skip conditions ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mutation", [
    {"status": "OPEN"},                 # still running — no outcome yet
    {"pattern": None},                  # no pattern was active at entry
    {"pattern": ""},
    {"ret": None},                      # return not yet derived
])
def test_incomplete_trades_are_skipped(mutation):
    assert pr.record_outcome(_trade(**mutation)) is False
    assert pr.all_pattern_stats() == {}


# ── persistence ─────────────────────────────────────────────────────────────

def test_missing_file_yields_an_empty_skeleton():
    reg = pr.load_registry()
    assert reg["patterns"] == {} and reg["by_ticker_pattern"] == {}
    assert reg["schema_version"] == pr.SCHEMA_VERSION
    assert pr.pattern_stats("double_bottom") is None
    assert pr.all_pattern_stats() == {}


def test_schema_mismatch_reinitialises_rather_than_misreading():
    """An old-schema file carries buckets with different keys; reading it as
    current would produce plausible-looking nonsense (missing pattern_correct
    silently reads as 0 accuracy)."""
    pr.REGISTRY_PATH.write_text(json.dumps({
        "schema_version": pr.SCHEMA_VERSION - 1,
        "patterns": {"double_bottom": {"n_trades": 99}},
    }), encoding="utf-8")
    assert pr.load_registry()["patterns"] == {}


def test_corrupt_file_starts_fresh_instead_of_raising():
    """The registry is written on the live tick path — a truncated file must not
    take the run down."""
    pr.REGISTRY_PATH.write_text("{not json", encoding="utf-8")
    assert pr.load_registry()["patterns"] == {}


def test_save_creates_the_directory_and_stamps_last_updated(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "REGISTRY_PATH", tmp_path / "nested" / "deeper" / "reg.json")
    reg = pr._empty_registry()
    reg["last_updated"] = "1999-01-01T00:00:00+00:00"
    pr.save_registry(reg)
    assert pr.REGISTRY_PATH.exists()
    assert json.loads(pr.REGISTRY_PATH.read_text(encoding="utf-8"))["last_updated"] > "2020"


def test_round_trip_through_disk_preserves_the_aggregates():
    pr.record_outcome(_trade(ret=7.5, exit_=107.5))
    before = pr.pattern_stats("double_bottom")
    assert pr.load_registry()["patterns"]["double_bottom"]["n_trades"] == 1
    assert pr.pattern_stats("double_bottom") == before      # read back from disk


# ── growth bounds ───────────────────────────────────────────────────────────

def test_per_trade_detail_is_capped_but_aggregates_are_not():
    """The trades list is capped at 200 so the file can't grow without bound;
    the AGGREGATES must keep counting past the cap or a long-lived pattern's
    evidence would silently stop accumulating."""
    reg = pr.load_registry()
    for i in range(205):
        pr.record_outcome(_trade(ret=1.0, exit_=101.0,
                                 entry_date=f"2020-01-01+{i}",
                                 exit_date=f"2020-02-01+{i}"), reg=reg)
    pr.save_registry(reg)
    assert len(reg["patterns"]["double_bottom"]["trades"]) == 200
    assert pr.pattern_stats("double_bottom")["n_trades"] == 205


def test_idempotency_survives_the_display_cap():
    """The duplicate check may NOT read the capped `trades` list: past 200 round
    trips a pattern would forget its oldest ones and re-add them every tick —
    the 2026-08-14 inflation bug again, just slower. Re-running the batch after
    the cap has trimmed must still be a no-op."""
    trades = [_trade(ret=1.0, exit_=101.0, entry_date=f"2020-01-01+{i}",
                     exit_date=f"2020-02-01+{i}") for i in range(205)]
    assert pr.record_batch(trades) == 205
    assert pr.record_batch(trades) == 0
    assert pr.pattern_stats("double_bottom")["n_trades"] == 205


def test_a_corrupted_v2_registry_is_discarded_not_migrated():
    """The live file written by the broken guard holds ~46k phantom trades. It
    must be REINITIALISED on the version bump rather than read forward — the
    counts are unrecoverable, and `pattern_recognition` weights the blend by n,
    so carrying them over would keep overriding the synthetic prior."""
    pr.REGISTRY_PATH.write_text(json.dumps({
        "schema_version": 2,
        "patterns": {"double_top": {"n_trades": 46250, "n_wins": 30000,
                                    "pattern_accuracy": 0.79, "trades": []}},
        "by_ticker_pattern": {"MSFT|double_top": {"n": 156}},
    }), encoding="utf-8")
    reg = pr.load_registry()
    assert reg["patterns"] == {} and reg["by_ticker_pattern"] == {}
    assert pr.pattern_stats("double_top") is None
    # ...and the ledger rebuilds it correctly on the next tick.
    assert pr.record_batch([_trade(pattern="double_top", action="SELL",
                                   exit_=95.0, ret=4.5)]) == 1
    assert pr.pattern_stats("double_top")["n_trades"] == 1


def test_all_pattern_stats_hides_patterns_with_no_trades():
    reg = pr.load_registry()
    reg["patterns"]["ghost"] = {"n_trades": 0}
    pr.save_registry(reg)
    assert "ghost" not in pr.all_pattern_stats()
    pr.record_outcome(_trade())
    assert set(pr.all_pattern_stats()) == {"double_bottom"}
