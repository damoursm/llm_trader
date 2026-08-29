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
                           "dropped_rank_cap", "dropped_low_agreement",
                           "dropped_buy_blocked", "dropped_earnings_blackout",
                           "dropped_untradeable", "dropped_overextended",
                           "actionable_survivors")}


@pytest.fixture(autouse=True)
def _all_gates_open(monkeypatch):
    """Every gate passes by default; each test closes exactly one."""
    from config.settings import settings
    monkeypatch.setattr(pl, "_passes_agreement_gate", lambda direction, sig: True)
    monkeypatch.setattr(pl, "_is_tradeable", lambda t, b: True)
    monkeypatch.setattr(pl, "_is_overextended", lambda t: False)
    # Gate 1c off by default here so each single-gate test stays single-gate;
    # the cap has its own tests below.
    monkeypatch.setattr(settings, "gate1_rank_cap", 0)


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


# ── Gate 1c — the per-run rank cap (2026-08-21) ────────────────────────────
# Hybrid floor+cap: the absolute floor preserves ABSTENTION, the cap kills the
# trade-rate's dependence on the LLM's confidence SCALE (the 08-17 incident
# doubled the rate with no signal change). Measured selection-neutral because
# within-run confidence rank carries no information.

def _cap_on(monkeypatch, k):
    from config.settings import settings
    monkeypatch.setattr(settings, "gate1_rank_cap", k)


def test_gate1c_caps_the_floor_passers_at_k(monkeypatch):
    _cap_on(monkeypatch, 2)
    recs = [_Rec("AAA", confidence=0.95), _Rec("BBB", confidence=0.90),
            _Rec("CCC", confidence=0.86), _Rec("DDD", confidence=0.99)]
    out, diag, outcomes = _run(recs)
    assert sorted(r.ticker for r in out) == ["AAA", "DDD"]   # the top-2
    assert diag["dropped_rank_cap"] == 2
    assert outcomes["BBB"] == "rank_capped" and outcomes["CCC"] == "rank_capped"
    assert diag["actionable_survivors"] == 2


def test_gate1c_never_fires_below_the_cap(monkeypatch):
    _cap_on(monkeypatch, 3)
    out, diag, _ = _run([_Rec("AAA", confidence=0.9), _Rec("BBB", confidence=0.86)])
    assert len(out) == 2 and diag["dropped_rank_cap"] == 0


def test_gate1c_preserves_abstention(monkeypatch):
    """The cap must NOT turn into a rank gate: a run where nothing clears the
    absolute floor still trades NOTHING (the empty run is information — pure
    top-K was measured worse for exactly this reason)."""
    _cap_on(monkeypatch, 3)
    out, diag, outcomes = _run([_Rec("AAA", confidence=0.80),
                                _Rec("BBB", confidence=0.70)])
    assert out == []
    assert diag["dropped_below_threshold"] == 2
    assert diag["dropped_rank_cap"] == 0


def test_gate1c_counts_floor_failures_before_capping(monkeypatch):
    """The cap ranks only FLOOR PASSERS — a sub-floor call must not occupy a
    cap slot, and its drop stays attributed to Gate 1."""
    _cap_on(monkeypatch, 2)
    recs = [_Rec("HI1", confidence=0.99), _Rec("LOW", confidence=0.5),
            _Rec("HI2", confidence=0.90), _Rec("HI3", confidence=0.88)]
    out, diag, outcomes = _run(recs)
    assert sorted(r.ticker for r in out) == ["HI1", "HI2"]
    assert outcomes["LOW"] == "below_threshold"
    assert outcomes["HI3"] == "rank_capped"


def test_gate1c_tie_break_is_deterministic(monkeypatch):
    """Equal confidences at the boundary resolve by ticker, so a rerun cannot
    reshuffle which name traded."""
    _cap_on(monkeypatch, 1)
    recs = [_Rec("ZZZ", confidence=0.90), _Rec("AAA", confidence=0.90)]
    out, _, outcomes = _run(recs)
    assert [r.ticker for r in out] == ["AAA"]
    assert outcomes["ZZZ"] == "rank_capped"


def test_gate1c_respects_the_per_side_floor(monkeypatch):
    """The cap's floor is the SAME side-adjusted bar Gate 1 uses — a SELL that
    fails its tightened bar is not a floor passer."""
    _cap_on(monkeypatch, 2)
    recs = [_Rec("B1", confidence=0.90), _Rec("S1", "SELL", confidence=0.88),
            _Rec("B2", confidence=0.86)]
    out, diag, outcomes = _run(recs, side_threshold_adj={"BUY": 0.0, "SELL": 0.04})
    # S1 (0.88) beats B2 (0.86) on raw confidence, but fails its side-adjusted
    # 0.89 bar — so it must NOT occupy a cap slot, and B2 trades.
    assert sorted(r.ticker for r in out) == ["B1", "B2"]
    assert outcomes["S1"] == "below_threshold"


def test_gate1c_side_floor_rescues_the_best_sell(monkeypatch):
    """THE CROWDING FIX (2026-08-22): SELLs are only ~28% of floor passers by
    composition, so a pooled top-K came out all-BUY in 28% of SELL-passing
    runs — starving the side the funnel measures as profitable. Each side with
    a floor passer gets one guaranteed slot, evicting the lowest kept call."""
    _cap_on(monkeypatch, 3)
    recs = [_Rec("B1", confidence=0.99), _Rec("B2", confidence=0.95),
            _Rec("B3", confidence=0.93), _Rec("S1", "SELL", confidence=0.90),
            _Rec("B4", confidence=0.88)]
    out, diag, outcomes = _run(recs)
    assert sorted(r.ticker for r in out) == ["B1", "B2", "S1"]
    assert outcomes["B3"] == "rank_capped"       # evicted: lowest of the kept
    assert outcomes["B4"] == "rank_capped"
    assert diag["actionable_survivors"] == 3     # K unchanged — a slot moved


def test_gate1c_side_floor_is_symmetric(monkeypatch):
    _cap_on(monkeypatch, 2)
    recs = [_Rec("S1", "SELL", confidence=0.99), _Rec("S2", "SELL", confidence=0.95),
            _Rec("B1", confidence=0.90)]
    out, _, outcomes = _run(recs)
    assert sorted(r.ticker for r in out) == ["B1", "S1"]
    assert outcomes["S2"] == "rank_capped"


def test_gate1c_side_floor_never_invents_a_passer(monkeypatch):
    """A side with NO floor passer gets nothing — the floor's abstention is
    untouched (a sub-bar SELL must not ride in on the guarantee)."""
    _cap_on(monkeypatch, 2)
    recs = [_Rec("B1", confidence=0.99), _Rec("B2", confidence=0.95),
            _Rec("S1", "SELL", confidence=0.70)]
    out, _, outcomes = _run(recs)
    assert sorted(r.ticker for r in out) == ["B1", "B2"]
    assert outcomes["S1"] == "below_threshold"


def test_gate1c_side_floor_noop_when_both_sides_kept(monkeypatch):
    _cap_on(monkeypatch, 3)
    recs = [_Rec("B1", confidence=0.99), _Rec("S1", "SELL", confidence=0.97),
            _Rec("B2", confidence=0.93), _Rec("B3", confidence=0.90)]
    out, _, outcomes = _run(recs)
    assert sorted(r.ticker for r in out) == ["B1", "B2", "S1"]
    assert outcomes["B3"] == "rank_capped"


def test_gate1c_side_floor_skipped_at_k1(monkeypatch):
    """K=1 cannot host a guarantee for both sides — the pooled winner stands."""
    _cap_on(monkeypatch, 1)
    recs = [_Rec("B1", confidence=0.99), _Rec("S1", "SELL", confidence=0.95)]
    out, _, outcomes = _run(recs)
    assert [r.ticker for r in out] == ["B1"]
    assert outcomes["S1"] == "rank_capped"


def test_gate1c_zero_disables(monkeypatch):
    _cap_on(monkeypatch, 0)
    recs = [_Rec(f"T{i}", confidence=0.86 + i / 100) for i in range(6)]
    out, diag, _ = _run(recs)
    assert len(out) == 6 and diag["dropped_rank_cap"] == 0


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
