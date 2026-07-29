"""The actionable filter — Gates 1, 1b, 2, 3, 4, 5 (2026-07-25).

This is the code that decides what actually TRADES, and coverage showed it had
**zero** test coverage: 0 of 31 statements. The individual gate helpers
(`_is_tradeable`, `_is_overextended`, `_passes_agreement_gate`) were well tested,
but nothing exercised the loop that SEQUENCES them — so a wrong gate order, a
missing `continue`, or a mis-attributed counter would have been invisible. It
was extracted from `run_pipeline` into `_apply_actionable_gates` purely to make
this possible.

Two things are pinned, and the second is the subtle one:
  * each gate rejects what it should and passes what it should;
  * a drop is attributed to the FIRST gate that rejects it. `compute_stage_eval`
    builds the decision funnel from those `gate_outcomes` stamps, so a
    mis-attribution silently corrupts the "which gate filters losers" analysis
    even when the trading decision itself is right.

All synthetic, no network, no DB.
"""

import pytest

import src.pipeline as pl


class _Rec:
    def __init__(self, ticker, action="BUY", confidence=0.99, direction=None):
        self.ticker = ticker
        self.action = action
        self.confidence = confidence
        self.direction = direction or ("BULLISH" if action == "BUY" else "BEARISH")


def _diag():
    return {k: 0 for k in ("buy_sell_candidates", "dropped_below_threshold",
                           "dropped_low_agreement", "dropped_buy_blocked",
                           "dropped_earnings_blackout", "dropped_untradeable",
                           "dropped_overextended", "actionable_survivors")}


@pytest.fixture(autouse=True)
def _all_gates_open(monkeypatch):
    """Every gate passes by default; each test closes exactly one."""
    monkeypatch.setattr(pl, "_passes_agreement_gate", lambda direction, sig: True)
    monkeypatch.setattr(pl, "_is_tradeable", lambda t, b: True)
    monkeypatch.setattr(pl, "_is_overextended", lambda t: False)


def _run(recs, **kw):
    diag, outcomes = _diag(), {}
    params = dict(confidence_threshold=0.85,
                  side_threshold_adj={"BUY": 0.0, "SELL": 0.0},
                  signals_by_ticker={}, allow_buys=True,
                  earnings_blackout=set(), trade_gate_budget={"n": 10},
                  gate_diag=diag, gate_outcomes=outcomes)
    params.update(kw)
    out = pl._apply_actionable_gates(recs, **params)
    return out, diag, outcomes


# ── baseline ───────────────────────────────────────────────────────────────

def test_a_clean_candidate_passes_every_gate():
    out, diag, outcomes = _run([_Rec("AAA")])
    assert [r.ticker for r in out] == ["AAA"]
    assert diag["actionable_survivors"] == 1
    assert diag["buy_sell_candidates"] == 1
    assert outcomes["AAA"] == "pass"


def test_hold_and_watch_are_not_candidates():
    """They never reach a gate, so they must not inflate buy_sell_candidates —
    the funnel's denominator."""
    out, diag, outcomes = _run([_Rec("A", "HOLD"), _Rec("B", "WATCH")])
    assert out == []
    assert diag["buy_sell_candidates"] == 0
    assert outcomes == {}


# ── each gate in isolation ─────────────────────────────────────────────────

def test_gate1_confidence_threshold():
    out, diag, outcomes = _run([_Rec("LOW", confidence=0.84),
                                _Rec("OK", confidence=0.85)])
    assert [r.ticker for r in out] == ["OK"]          # boundary is inclusive
    assert diag["dropped_below_threshold"] == 1
    assert outcomes["LOW"] == "below_threshold"


def test_gate1_per_side_adjustment_tightens_only_the_named_side():
    """The SELL bar is raised; an identical-confidence BUY must still pass."""
    out, diag, _ = _run([_Rec("B", "BUY", 0.86), _Rec("S", "SELL", 0.86)],
                        side_threshold_adj={"BUY": 0.0, "SELL": 0.05})
    assert [r.ticker for r in out] == ["B"]
    assert diag["dropped_below_threshold"] == 1


def test_gate1b_agreement_floor(monkeypatch):
    monkeypatch.setattr(pl, "_passes_agreement_gate", lambda direction, sig: False)
    out, diag, outcomes = _run([_Rec("AAA")])
    assert out == []
    assert diag["dropped_low_agreement"] == 1
    assert outcomes["AAA"] == "low_agreement"


def test_gate2_buy_block_spares_sells():
    """PANIC/RISK_OFF blocks BUYs only — shorting into a falling market is the
    whole point of keeping SELLs open."""
    out, diag, outcomes = _run([_Rec("B", "BUY"), _Rec("S", "SELL")],
                               allow_buys=False)
    assert [r.ticker for r in out] == ["S"]
    assert diag["dropped_buy_blocked"] == 1
    assert outcomes["B"] == "buy_blocked"


def test_gate3_earnings_blackout():
    out, diag, outcomes = _run([_Rec("ERN"), _Rec("OK")],
                               earnings_blackout={"ERN"})
    assert [r.ticker for r in out] == ["OK"]
    assert diag["dropped_earnings_blackout"] == 1
    assert outcomes["ERN"] == "earnings_blackout"


def test_gate4_liquidity_floor(monkeypatch):
    monkeypatch.setattr(pl, "_is_tradeable", lambda t, b: t != "PENNY")
    out, diag, outcomes = _run([_Rec("PENNY"), _Rec("OK")])
    assert [r.ticker for r in out] == ["OK"]
    assert diag["dropped_untradeable"] == 1
    assert outcomes["PENNY"] == "untradeable"


def test_gate5_overextension_is_BUY_only(monkeypatch):
    """A SELL on a spiked name is measured EDGE (fading spikes works), so the
    anti-chase gate must never touch it."""
    monkeypatch.setattr(pl, "_is_overextended", lambda t: True)
    out, diag, outcomes = _run([_Rec("HOT", "BUY"), _Rec("SPIKE", "SELL")])
    assert [r.ticker for r in out] == ["SPIKE"]
    assert diag["dropped_overextended"] == 1
    assert outcomes["HOT"] == "overextended"


# ── ordering / attribution — the subtle half ───────────────────────────────

def test_a_drop_is_attributed_to_the_FIRST_failing_gate(monkeypatch):
    """A candidate failing several gates must be stamped with the earliest one.
    compute_stage_eval's funnel reads these stamps, so mis-attribution corrupts
    the "which gate filters losers" analysis even when the trade decision is
    right."""
    monkeypatch.setattr(pl, "_is_tradeable", lambda t, b: False)
    monkeypatch.setattr(pl, "_is_overextended", lambda t: True)
    out, diag, outcomes = _run([_Rec("BAD", "BUY", confidence=0.10)],
                               earnings_blackout={"BAD"}, allow_buys=False)
    assert out == []
    assert outcomes["BAD"] == "below_threshold", "Gate 1 must claim it"
    # And ONLY that counter moves.
    assert diag["dropped_below_threshold"] == 1
    for k in ("dropped_buy_blocked", "dropped_earnings_blackout",
              "dropped_untradeable", "dropped_overextended"):
        assert diag[k] == 0, f"{k} also fired — gates are not short-circuiting"


def test_expensive_gates_are_not_consulted_after_an_early_drop(monkeypatch):
    """Gate order is load-bearing for COST too: the liquidity gate can trigger a
    network fetch, so it must never run for an already-rejected candidate."""
    calls = []
    monkeypatch.setattr(pl, "_is_tradeable", lambda t, b: calls.append(t) or True)
    _run([_Rec("LOW", confidence=0.10)])
    assert calls == [], "_is_tradeable ran on a candidate Gate 1 had rejected"


def test_counters_and_outcomes_accumulate_across_a_mixed_batch(monkeypatch):
    monkeypatch.setattr(pl, "_is_tradeable", lambda t, b: t != "THIN")
    monkeypatch.setattr(pl, "_is_overextended", lambda t: t == "HOT")
    recs = [_Rec("PASS1"), _Rec("LOW", confidence=0.5), _Rec("THIN"),
            _Rec("HOT"), _Rec("ERN"), _Rec("PASS2", "SELL"), _Rec("IGN", "HOLD")]
    out, diag, outcomes = _run(recs, earnings_blackout={"ERN"})
    assert [r.ticker for r in out] == ["PASS1", "PASS2"]
    assert diag["buy_sell_candidates"] == 6          # HOLD excluded
    assert diag["actionable_survivors"] == 2
    assert diag["dropped_below_threshold"] == 1
    assert diag["dropped_untradeable"] == 1
    assert diag["dropped_overextended"] == 1
    assert diag["dropped_earnings_blackout"] == 1
    assert outcomes == {"PASS1": "pass", "LOW": "below_threshold",
                        "THIN": "untradeable", "HOT": "overextended",
                        "ERN": "earnings_blackout", "PASS2": "pass"}
    # Every candidate is accounted for exactly once.
    assert sum(v for k, v in diag.items()
               if k.startswith("dropped_")) + diag["actionable_survivors"] \
        == diag["buy_sell_candidates"]


def test_gate_outcomes_is_written_back_to_gate_diag():
    _, diag, outcomes = _run([_Rec("AAA")])
    assert diag["gate_outcomes"] is outcomes


def test_empty_recommendations_is_a_clean_noop():
    out, diag, outcomes = _run([])
    assert out == [] and outcomes == {}
    assert diag["buy_sell_candidates"] == 0
