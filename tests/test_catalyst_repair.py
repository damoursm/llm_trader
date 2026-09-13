"""The catalyst-REPAIR pass (``src/analysis/catalyst_repair.py``).

The label is repaired OUTSIDE the scoring prompt: a deterministic fund
override, a keyword vote that only ever DETECTS a disagreement (never
labels), per-engine unreliable-class triggers, and a score-free local
specialist whose answer is a 3-vote majority in shuffled taxonomy order.
Rows land in ``catalyst_repairs``; ``signals.news_catalyst`` is never touched.

``tests/conftest.py`` switches the pass OFF for every other test; this file
re-enables it explicitly with a stubbed specialist and drains the pool.
"""
from __future__ import annotations

import types

import pytest

from config.settings import settings
from src.analysis import catalyst_repair as cr
from src.analysis import sentiment as sent
from src.data import company_names


@pytest.fixture
def repair_on(monkeypatch):
    monkeypatch.setattr(settings, "enable_catalyst_repair", True, raising=False)
    monkeypatch.setattr(settings, "enable_local_llm", True, raising=False)
    monkeypatch.setattr(settings, "catalyst_repair_monitor_share", 0.0, raising=False)
    monkeypatch.setattr(settings, "catalyst_repair_max_pending", 200, raising=False)
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes",
                        "local=product,capital_structure,other,management,company_pr",
                        raising=False)
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "run-test", raising=False)
    cr._reset_for_tests()
    yield
    cr._drain_for_tests()
    cr._reset_for_tests()


def _seed_names():
    company_names._seed_for_tests(
        {"AR": "Antero Resources Corp", "GLD": "SPDR Gold Shares",
         "XLV": "Health Care Select Sector SPDR Fund", "LLY": "Eli Lilly & Co"},
        industries={"AR": "Crude Petroleum & Natural Gas", "GLD": "exchange-traded fund"})


# ── Keyword vote: a DISAGREEMENT DETECTOR, never a label ───────────────────
@pytest.mark.parametrize("headline, classes", [
    ("Vanguard Group discloses 13F stake in Antero Resources", {"insider_activity"}),
    ("Schedule 13G filed by BlackRock on AR", {"insider_activity"}),
    ("CEO sells shares under 10b5-1 plan, Form 4 shows", {"insider_activity"}),
    ("Insider buying at Antero: director bought 10,000 shares", {"insider_activity"}),
    ("Phase 3 trial met primary endpoint", {"fda_clinical"}),
    ("PDUFA date set for March", {"fda_clinical"}),
    ("FDA approves new indication", {"fda_clinical"}),
    ("Antero reports Q3 EPS of $0.45, beats estimates", {"earnings"}),
    ("Third-quarter results top expectations", {"earnings"}),
    ("Company raises full-year guidance", {"guidance"}),
    ("Outlook raised on strong demand", {"guidance"}),
    ("Morgan Stanley upgrades AR to Overweight, lifts price target", {"analyst"}),
    ("Downgraded to Neutral at JPMorgan", {"analyst"}),
    ("Initiates coverage with Buy rating", {"analyst"}),
    ("EQT to acquire Antero in $12B deal", {"ma_deal"}),
    ("Merger talks confirmed; tender offer launched", {"ma_deal"}),
    ("Antero Resources joins S&P 500", {"index_membership"}),
    ("Set to be added to the Russell 2000 index", {"index_membership"}),
    ("Index rebalance: AR added", {"index_membership"}),
    # Commentary, tape and routine items carry no anchor vocabulary: no vote,
    # so the detector can never contradict a model that read them as soft.
    ("Shares of Antero Resources rose 3% on Tuesday as natural gas futures climbed", set()),
    ("Analysts expect strong demand for natural gas this winter", set()),
    ("Antero announces new drilling program in Appalachia", set()),
    ("Stock hits 52-week high on heavy volume", set()),
    ("Dividend declared at $0.10 per share", set()),
])
def test_keyword_classes_fire_only_on_the_anchor_vocabulary(headline, classes):
    assert cr.keyword_classes(headline) == classes


def test_keyword_disagreement_respects_the_compatible_pairs():
    # earnings ⇄ guidance and analyst ⇄ earnings are neighbours the vocabulary
    # cannot separate ("beats estimates" is both an earnings print and an
    # analyst frame), so those pairs are NOT disagreements.
    assert not cr.keyword_disagrees("earnings", {"guidance"})
    assert not cr.keyword_disagrees("guidance", {"earnings"})
    assert not cr.keyword_disagrees("analyst", {"earnings"})
    assert not cr.keyword_disagrees("earnings", {"analyst"})
    assert cr.keyword_disagrees("guidance", {"analyst"})
    assert cr.keyword_disagrees("none", {"ma_deal"})
    assert cr.keyword_disagrees("product", {"fda_clinical"})
    assert not cr.keyword_disagrees("ma_deal", {"ma_deal"})
    # No vote, or no label to compare: nothing to disagree with.
    assert not cr.keyword_disagrees("ma_deal", set())
    assert not cr.keyword_disagrees(None, {"ma_deal"})


def test_keyword_text_joins_headlines_and_the_first_pass_rationale():
    """The user's rule: run on headlines AND on the first-pass rationale — a
    rationale that names a price target flags ``analyst`` even when every
    headline is plain."""
    assert cr.keyword_text(["a", "b"], "r") == "a\nb\nr"
    assert cr.keyword_text(["a"], None) == "a"
    plain = ["Antero announces new drilling program in Appalachia"]
    assert cr.keyword_classes(cr.keyword_text(plain, None)) == set()
    assert cr.keyword_classes(cr.keyword_text(
        plain, "The price target raise by Barclays supports the read")) == {"analyst"}


def test_unreliable_classes_parse_per_engine_case_insensitively(monkeypatch):
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes",
                        "local=product,capital_structure,other,management,company_pr;"
                        "qwen=Other", raising=False)
    exp = frozenset({"product", "capital_structure", "other", "management", "company_pr"})
    assert cr.unreliable_classes("local") == exp
    assert cr.unreliable_classes("LOCAL") == exp
    assert cr.unreliable_classes("qwen") == frozenset({"other"})
    assert cr.unreliable_classes("deepseek") == frozenset()
    assert cr.unreliable_classes(None) == frozenset()
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes", "", raising=False)
    assert cr.unreliable_classes("local") == frozenset()


def test_unreliable_classes_are_keyed_per_model_with_the_engine_as_fallback(monkeypatch):
    """A measured error rate belongs to a CHECKPOINT: the model id wins, the
    bare engine name is the fallback for a table written before ids were
    known, and a model swapped in under the same engine name inherits NOTHING."""
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes",
                        "local/qwen3:8b=product,other;local=management", raising=False)
    assert cr.unreliable_classes("local", "local/qwen3:8b") == frozenset({"product", "other"})
    assert cr.unreliable_classes("local", "LOCAL/QWEN3:8B") == frozenset({"product", "other"})
    assert cr.unreliable_classes("local", "local/qwen3:14b") == frozenset({"management"})
    assert cr.unreliable_classes("local") == frozenset({"management"})
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes",
                        "local/qwen3:8b=product,other", raising=False)
    assert cr.unreliable_classes("local", "local/qwen3:14b") == frozenset()   # the swap
    assert cr.unreliable_classes("local") == frozenset()
    assert cr.unreliable_classes("deepseek", "deepseek-v4-flash") == frozenset()


def test_trigger_resolves_the_engines_current_model_and_a_swap_empties_the_set(repair_on, monkeypatch):
    """`trigger_for` keys on the engine's CURRENT model id (the one
    `sentiment_model_for` stamps on the row), so the live default
    `local/qwen3:8b=...` fires for local today and stops firing - leaving the
    fund / keyword / monitor triggers - the day `local_sentiment_model` changes."""
    _seed_names()
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes",
                        "local/qwen3:8b=product", raising=False)
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:8b", raising=False)
    assert cr._model_id("local") == "local/qwen3:8b"
    assert cr.trigger_for(ticker="AR", engine="local", first_pass="product") == "unreliable_class"
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:14b", raising=False)
    assert cr._model_id("local") == "local/qwen3:14b"
    assert cr.trigger_for(ticker="AR", engine="local", first_pass="product") is None
    # ... the other triggers still fire for the unmeasured model
    # `fund` is off by default since 2026-09-11; enabled for this one
    # check because what is under test is that a model SWAP empties
    # only the unreliable set.
    monkeypatch.setattr(settings, "catalyst_repair_fund_trigger", True, raising=False)
    assert cr.trigger_for(ticker="GLD", engine="local", first_pass="product") == "fund"
    assert cr.trigger_for(ticker="AR", engine="local", first_pass="product",
                          headlines=["Phase 3 trial met primary endpoint"]) == "keyword"
    # An explicit model id overrides the engine's current one.
    assert cr.trigger_for(ticker="AR", engine="local", first_pass="product",
                          model="local/qwen3:8b") == "unreliable_class"


def test_the_resolved_unreliable_set_is_logged_once_per_model(repair_on, monkeypatch):
    """An EMPTY set after a model swap is invisible everywhere else: the pass
    keeps running, rows keep landing (fund / keyword / monitor), and nothing
    says the unreliable-class trigger went dark. One INFO line per model id."""
    _seed_names()
    monkeypatch.setattr(settings, "catalyst_repair_unreliable_classes",
                        "local/qwen3:8b=product,other", raising=False)
    monkeypatch.setattr(settings, "local_sentiment_model", "qwen3:8b", raising=False)
    lines = []
    sink = cr.logger.add(lambda m: lines.append(str(m)), level="INFO",
                         filter=lambda r: "unreliable set" in r["message"])
    try:
        for _ in range(3):
            cr.trigger_for(ticker="AR", engine="local", first_pass="product")
        cr.trigger_for(ticker="AR", engine="deepseek", first_pass="product")
        cr.trigger_for(ticker="AR", engine="deepseek", first_pass="guidance")
    finally:
        cr.logger.remove(sink)
    assert len(lines) == 2
    assert "unreliable set for local/qwen3:8b: other, product" in lines[0]
    assert "unreliable set for " + sent.sentiment_model_for("deepseek") + ": NONE" in lines[1]
    assert "only the fund / keyword / monitor triggers fire" in lines[1]


def test_monitor_sample_is_deterministic_per_run_and_ticker(monkeypatch):
    assert not cr._monitor_sampled("AR", 0.0)
    assert not cr._monitor_sampled("AR", -1.0)
    assert cr._monitor_sampled("AR", 1.0)
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "run-A", raising=False)
    a = [cr._monitor_sampled(t, 0.5) for t in ["AR", "EQT", "GLD", "XLV", "LLY", "BAC", "DIS", "MSFT"]]
    assert a == [cr._monitor_sampled(t, 0.5) for t in ["AR", "EQT", "GLD", "XLV", "LLY", "BAC", "DIS", "MSFT"]]
    assert True in a and False in a                                  # a sample, not a constant
    monkeypatch.setattr(sent, "_CURRENT_RUN_ID", "run-B", raising=False)
    b = [cr._monitor_sampled(t, 0.5) for t in ["AR", "EQT", "GLD", "XLV", "LLY", "BAC", "DIS", "MSFT"]]
    assert a != b                                                    # re-drawn per run


# ── Trigger priority and off-switches ───────────────────────────────────────
def test_trigger_priority_fund_then_unreliable_then_keyword_then_monitor(repair_on, monkeypatch):
    _seed_names()
    # `fund` is OFF by default since 2026-09-11 (see the dedicated tests below),
    # so a fund falls through to whatever else applies.
    monkeypatch.setattr(settings, "catalyst_repair_fund_trigger", True, raising=False)
    # fund wins even over an unreliable class; a Polygon-typed fund counts.
    assert cr.trigger_for(ticker="XLV", engine="local", first_pass="product") == "fund"
    assert cr.trigger_for(ticker="GLD", engine="deepseek", first_pass="earnings") == "fund"
    monkeypatch.setattr(settings, "catalyst_repair_fund_trigger", False, raising=False)
    # per-engine unreliable class
    assert cr.trigger_for(ticker="AR", engine="local", first_pass="product") == "unreliable_class"
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="product") is None
    # keyword disagreement (headline or rationale), never when they agree
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="product",
                          headlines=["Phase 3 trial met primary endpoint"]) == "keyword"
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="fda_clinical",
                          headlines=["Phase 3 trial met primary endpoint"]) is None
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="earnings",
                          headlines=["Company raises full-year guidance"]) is None
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="none",
                          rationale="the 13F filing shows a new stake") == "keyword"
    # monitor sample last, and only when allowed
    monkeypatch.setattr(settings, "catalyst_repair_monitor_share", 1.0, raising=False)
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="product") == "monitor"
    assert cr.trigger_for(ticker="AR", engine="deepseek", first_pass="product",
                          allow_monitor=False) is None
    assert cr.trigger_for(ticker="AR", engine="local", first_pass="product") == "unreliable_class"
    # nothing to repair
    assert cr.trigger_for(ticker="XLV", engine="local", first_pass=None) is None


def test_trigger_is_none_when_the_pass_or_the_local_engine_is_off(repair_on, monkeypatch):
    _seed_names()
    monkeypatch.setattr(settings, "enable_local_llm", False, raising=False)
    assert cr.trigger_for(ticker="XLV", engine="local", first_pass="product") is None
    monkeypatch.setattr(settings, "enable_local_llm", True, raising=False)
    monkeypatch.setattr(settings, "enable_catalyst_repair", False, raising=False)
    assert cr.trigger_for(ticker="XLV", engine="local", first_pass="product") is None


# ── Specialist prompt scaffolding ───────────────────────────────────────────
def test_enriched_header_reads_the_same_identity_as_the_scoring_header():
    _seed_names()
    assert cr._enriched_header("ar") == "TARGET: AR — Antero Resources Corp (Crude Petroleum & Natural Gas)"
    gld = cr._enriched_header("GLD")                                 # Polygon-typed fund, no fund word
    assert gld.startswith("TARGET: GLD — SPDR Gold Shares (exchange-traded fund)\n")
    assert "FUND / ETF" in gld and "never the fund's class" in gld
    xlv = cr._enriched_header("XLV")                                 # name-word fund, no industry
    assert xlv.startswith("TARGET: XLV — Health Care Select Sector SPDR Fund\n")
    assert "FUND / ETF" in xlv and "(" not in xlv.splitlines()[0]
    assert cr._enriched_header("ZQZX") == "TARGET: ZQZX"
    # The trigger and the header agree about what a fund is.
    for t in ("GLD", "XLV", "AR", "ZQZX"):
        assert ("FUND / ETF" in cr._enriched_header(t)) == cr._is_fund(t)


def test_class_order_is_canonical_first_then_a_deterministic_shuffle():
    canon = cr.class_order("d1", 0)
    assert canon[:3] == ["earnings", "guidance", "analyst"]
    assert canon == list(sent.NEWS_CATALYST_TYPES)
    s1 = cr.class_order("d1", 1)
    s2 = cr.class_order("d1", 2)
    assert s1 == cr.class_order("d1", 1) and s2 == cr.class_order("d1", 2)
    assert sorted(s1) == sorted(canon) and sorted(s2) == sorted(canon)
    assert s1 != canon and s2 != canon and s1 != s2
    assert cr.class_order("d2", 1) != s1                             # seeded per digest


def test_specialist_prompt_carries_the_header_the_order_and_no_score_ask():
    _seed_names()
    order = cr.class_order("d1", 1)
    p = cr.build_specialist_prompt("AR", "DIGEST BODY", "first-pass rationale",
                                   order=order, header=cr._enriched_header("AR"))
    assert "TARGET: AR — Antero Resources Corp (Crude Petroleum & Natural Gas)" in p
    assert "DIGEST BODY" in p and "first-pass rationale" in p
    assert p.index(f"- {order[0]}:") < p.index(f"- {order[-1]}:")
    assert "sentiment" in p.lower() and "score" in p.lower()         # told what it is NOT doing
    assert '"score"' not in p


def test_vote_from_applies_the_consistency_rules():
    v = cr.vote_from({"about_target": "holdings", "event_family": "results_outlook",
                      "catalyst": "earnings", "evidence": "x" * 500})
    assert v["catalyst"] == "macro_sector" and v["about_target"] == "holdings"
    assert v["event_family"] == "results_outlook" and len(v["evidence"]) == 200
    assert cr.vote_from({"about_target": "related", "catalyst": "ma_deal"})["catalyst"] == "macro_sector"
    assert cr.vote_from({"about_target": "unrelated", "catalyst": "earnings"})["catalyst"] == "none"
    assert cr.vote_from({"about_target": "no_event", "catalyst": "product"})["catalyst"] == "none"
    assert cr.vote_from({"about_target": "target", "catalyst": "Fda Clinical"})["catalyst"] == "fda_clinical"
    assert cr.vote_from({"about_target": "target", "catalyst": "fda-clinical"})["catalyst"] == "fda_clinical"
    bad = cr.vote_from({"about_target": "elsewhere", "event_family": "weird", "catalyst": "guidance"})
    assert bad["catalyst"] == "guidance" and bad["about_target"] is None and bad["event_family"] is None
    assert cr.vote_from({"about_target": "target", "catalyst": "bogus"}) is None
    assert cr.vote_from({"about_target": "target"}) is None
    assert cr.vote_from("guidance") is None
    assert cr.vote_from(None) is None


DIGEST = """[Reuters | 2h] Acme Corp names Jane Doe chief executive
The board said the appointment is effective “Monday, March 3”.

[Yahoo | 1d] 3 stocks to watch this week
Analysts weigh in."""


def test_other_needs_a_quoted_dated_event_or_the_vote_is_discarded():
    """The spec's one HARD rule on the dumping-ground class: ``other`` must
    quote a named, dated event FROM the digest, or it is not a vote at all."""
    base = {"about_target": "target", "event_family": "operations", "catalyst": "other"}
    # No evidence at all → discarded, with or without a digest to check.
    assert cr.vote_from(dict(base, evidence="")) is None
    assert cr.vote_from(dict(base, evidence=None), digest_text=DIGEST) is None
    assert cr.vote_from(dict(base, evidence="  \"  "), digest_text=DIGEST) is None
    # Evidence that is NOT in the digest → discarded.
    assert cr.vote_from(dict(base, evidence="announced a $2B buyback"), digest_text=DIGEST) is None
    # Evidence lifted from the digest → kept, and grounded — case, surrounding
    # quote marks, smart quotes and line wraps are all forgiven.
    v = cr.vote_from(dict(base, evidence="\"names Jane Doe chief executive\""), digest_text=DIGEST)
    assert v is not None and v["catalyst"] == "other" and v["grounded"] is True
    v = cr.vote_from(dict(base, evidence="effective 'monday,   march 3'"), digest_text=DIGEST)
    assert v is not None and v["grounded"] is True
    # Without a digest there is nothing to check against: kept, grounded unknown.
    v = cr.vote_from(dict(base, evidence="names Jane Doe chief executive"))
    assert v is not None and v["grounded"] is None


def test_grounded_is_measured_on_every_vote_but_only_other_is_discarded():
    v = cr.vote_from({"about_target": "target", "event_family": "operations",
                      "catalyst": "management", "evidence": "fired the whole board"},
                     digest_text=DIGEST)
    assert v is not None and v["catalyst"] == "management" and v["grounded"] is False
    v = cr.vote_from({"about_target": "target", "event_family": "operations",
                      "catalyst": "management", "evidence": "names Jane Doe chief executive"},
                     digest_text=DIGEST)
    assert v["grounded"] is True
    v = cr.vote_from({"about_target": "target", "catalyst": "management"}, digest_text=DIGEST)
    assert v["grounded"] is None and v["evidence"] is None


def test_vote_memo_is_salted_with_the_specialist_version_and_model(repair_on, monkeypatch):
    """A prompt edit or a model swap must never serve the previous specialist's
    answers — the same discipline the sentiment cache key enforces."""
    cr._store_vote("d1", 0, {"catalyst": "guidance"})
    assert cr._cached_vote("d1", 0) == (True, {"catalyst": "guidance"})
    monkeypatch.setattr(cr, "SPECIALIST_VERSION", "cr9-test")
    assert cr._cached_vote("d1", 0) == (False, None)
    monkeypatch.undo()
    assert cr._cached_vote("d1", 0)[0] is True
    monkeypatch.setattr(cr, "specialist_model", lambda: "other-model:1b")
    assert cr._cached_vote("d1", 0) == (False, None)


# ── Voting procedure ────────────────────────────────────────────────────────
class _Specialist:
    """Scripted stand-in for ``_call_specialist``: one answer per call, in order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, prompt, order, label="", think=False):
        self.calls.append((prompt, list(order), label, think))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return {"about_target": "target", "event_family": "operations", "catalyst": a}


def _vote(cls):
    return {"about_target": "target", "event_family": "operations", "catalyst": cls}


def test_an_ungrounded_other_vote_is_discarded_end_to_end(repair_on, monkeypatch):
    """Through ``classify_digest``: the specialist's ``other`` carries no quoted
    event, so it is not a vote — the disagreement path runs, and two votes for
    a different class decide; a first pass of ``other`` cannot be confirmed by
    an evidence-free echo of itself."""
    spec = _Specialist(["other", "management", "management"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    out = cr.classify_digest(digest_id="e1", ticker="ACME", digest_text=DIGEST,
                             rationale="CEO change", first_pass="other")
    assert out["n_calls"] == 3
    assert [v["catalyst"] for v in out["votes"]] == ["management", "management"]
    assert out["final"] == "management" and out["quality"] == "resolved"
    assert out["error"] is None


def test_first_vote_agreeing_resolves_in_one_call(repair_on, monkeypatch):
    spec = _Specialist(["guidance"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    out = cr.classify_digest(digest_id="d-agree", ticker="AR", digest_text="t",
                             rationale="r", first_pass="guidance", trigger="monitor")
    assert out["n_calls"] == 1 and out["quality"] == "resolved" and out["final"] == "guidance"
    assert [v["catalyst"] for v in out["votes"]] == ["guidance"]
    assert out["votes"][0]["shuffled"] is False
    assert out["about_target"] == "target" and out["error"] is None
    assert out["trigger"] == "monitor" and out["specialist_version"] == cr.SPECIALIST_VERSION
    assert out["specialist_model"] == f"local/{cr.specialist_model()}"
    assert spec.calls[0][1] == cr.class_order("d-agree", 0)


def test_disagreement_takes_two_more_shuffled_votes_and_the_majority(repair_on, monkeypatch):
    spec = _Specialist(["analyst", "guidance", "analyst"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    out = cr.classify_digest(digest_id="d-maj", ticker="AR", digest_text="t",
                             rationale="r", first_pass="guidance")
    assert out["n_calls"] == 3 and out["quality"] == "resolved" and out["final"] == "analyst"
    assert [v["shuffled"] for v in out["votes"]] == [False, True, True]
    assert spec.calls[1][1] == cr.class_order("d-maj", 1)
    assert spec.calls[2][1] == cr.class_order("d-maj", 2)


def test_no_majority_stays_unresolved_on_the_first_pass(repair_on, monkeypatch):
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["analyst", "guidance", "ma_deal"]))
    out = cr.classify_digest(digest_id="d-split", ticker="AR", digest_text="t",
                             rationale="r", first_pass="product")
    assert out["n_calls"] == 3 and out["quality"] == "unresolved" and out["final"] == "product"


def test_majority_may_confirm_the_first_pass_after_a_dissenting_first_vote(repair_on, monkeypatch):
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["analyst", "guidance", "guidance"]))
    out = cr.classify_digest(digest_id="d-back", ticker="AR", digest_text="t",
                             rationale="r", first_pass="guidance")
    assert out["quality"] == "resolved" and out["final"] == "guidance" and out["n_calls"] == 3


def test_mechanical_override_wins_whatever_the_vote(repair_on, monkeypatch):
    _seed_names()
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["earnings"]))
    out = cr.classify_digest(digest_id="d-ovr", ticker="XLV", digest_text="t", rationale="r",
                             first_pass="earnings", final_label="macro_sector", trigger="fund")
    assert out["quality"] == "override" and out["final"] == "macro_sector"
    assert out["n_calls"] == 1                                       # monitoring only


def test_a_failing_first_call_is_recorded_and_stops_the_vote(repair_on, monkeypatch):
    monkeypatch.setattr(cr, "_call_specialist", _Specialist([RuntimeError("box down")]))
    out = cr.classify_digest(digest_id="d-err", ticker="AR", digest_text="t",
                             rationale="r", first_pass="guidance")
    assert out["votes"] == [] and out["n_calls"] == 1
    assert out["quality"] == "unresolved" and out["final"] == "guidance"
    assert "RuntimeError: box down" in out["error"]


def test_votes_are_memoised_per_digest_so_a_shadow_row_reuses_them(repair_on, monkeypatch):
    spec = _Specialist(["analyst", "analyst", "analyst"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    a = cr.classify_digest(digest_id="d-memo", ticker="AR", digest_text="t",
                           rationale="r", first_pass="guidance")
    assert a["final"] == "analyst" and len(spec.calls) == 3
    b = cr.classify_digest(digest_id="d-memo", ticker="AR", digest_text="t",
                           rationale="r", first_pass="product")
    assert b["final"] == "analyst" and b["quality"] == "resolved"
    assert len(spec.calls) == 3                                      # no new call
    # A failed vote is NOT memoised: the next row re-asks.
    spec2 = _Specialist([RuntimeError("x"), "guidance"])
    monkeypatch.setattr(cr, "_call_specialist", spec2)
    cr.classify_digest(digest_id="d-retry", ticker="AR", digest_text="t", rationale="r", first_pass="guidance")
    out = cr.classify_digest(digest_id="d-retry", ticker="AR", digest_text="t", rationale="r", first_pass="guidance")
    assert out["quality"] == "resolved" and len(spec2.calls) == 2


# ── Background pool + ledger rows ───────────────────────────────────────────
def _arts(*titles):
    return [types.SimpleNamespace(title=t) for t in titles]


def test_maybe_submit_queues_a_triggered_repair_and_yields_one_row(repair_on, monkeypatch):
    _seed_names()
    # First vote dissents from the first pass, so the majority round runs: 3 calls.
    monkeypatch.setattr(cr, "_call_specialist",
                        _Specialist(["fda_clinical", "fda_clinical", "product"]))
    trig = cr.maybe_submit(ticker="AR", engine="deepseek", first_pass="product",
                           final_label="product", rationale="r", digest_id="dg-1",
                           digest_text="body", articles=_arts("Phase 3 trial met primary endpoint"))
    assert trig == "keyword"
    cr._drain_for_tests()
    rows = cr.pop_catalyst_repair_rows()
    assert len(rows) == 1
    r = rows[0]
    assert r["digest_id"] == "dg-1" and r["run_id"] == "run-test" and r["ticker"] == "AR"
    assert r["engine"] == "deepseek" and r["model"] == sent.sentiment_model_for("deepseek")
    assert r["first_pass"] == "product" and r["trigger"] == "keyword"
    assert r["final"] == "fda_clinical" and r["quality"] == "resolved" and r["n_calls"] == 3
    assert r["generated_at"]
    assert cr.pop_catalyst_repair_rows() == []                       # drained
    assert cr.catalyst_repair_pending() == 0


def test_maybe_submit_dedupes_per_digest_and_engine_and_stamps_the_role(repair_on, monkeypatch):
    _seed_names()
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["product", "product", "product", "product"]))
    kw = dict(ticker="AR", first_pass="product", final_label="product", rationale="r",
              digest_id="dg-2", digest_text="body")
    assert cr.maybe_submit(engine="local", **kw) == "unreliable_class"
    assert cr.maybe_submit(engine="local", **kw) is None             # same (digest, engine)
    assert cr.maybe_submit(engine="local", role="shadow", **dict(kw, digest_id="dg-3")) == "unreliable_class"
    cr._drain_for_tests()
    rows = {r["digest_id"]: r for r in cr.pop_catalyst_repair_rows()}
    assert set(rows) == {"dg-2", "dg-3"}
    assert rows["dg-2"]["trigger"] == "unreliable_class"
    assert rows["dg-3"]["trigger"] == "unreliable_class@shadow"


def test_maybe_submit_refuses_incomplete_calls_and_untriggered_labels(repair_on, monkeypatch):
    _seed_names()
    spec = _Specialist(["guidance"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    base = dict(ticker="AR", engine="deepseek", first_pass="guidance", final_label="guidance",
                rationale="r", digest_id="dg-4", digest_text="body")
    assert cr.maybe_submit(**dict(base, digest_id=None)) is None
    assert cr.maybe_submit(**dict(base, digest_text="")) is None
    assert cr.maybe_submit(**dict(base, engine=None)) is None
    assert cr.maybe_submit(**base) is None                           # DeepSeek + guidance: no trigger
    cr._drain_for_tests()
    assert cr.pop_catalyst_repair_rows() == [] and spec.calls == []


def test_maybe_submit_backs_off_at_the_pending_cap(repair_on, monkeypatch):
    _seed_names()
    monkeypatch.setattr(settings, "catalyst_repair_max_pending", 1, raising=False)
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["product"] * 4))
    monkeypatch.setattr(cr, "_PENDING", 1)                           # one already in flight
    kw = dict(ticker="AR", engine="local", first_pass="product", final_label="product",
              rationale="r", digest_text="body")
    assert cr.maybe_submit(digest_id="dg-5", **kw) is None
    assert cr._WARNED["cap"] is True
    monkeypatch.setattr(cr, "_PENDING", 0)
    assert cr.maybe_submit(digest_id="dg-5", **kw) == "unreliable_class"   # re-offered once it drains
    cr._drain_for_tests()
    assert [r["digest_id"] for r in cr.pop_catalyst_repair_rows()] == ["dg-5"]


def test_trigger_counts_keep_the_identity_and_reset_on_pop(repair_on, monkeypatch):
    """The per-tick tally is how the "~15-20% of calls" estimate becomes a
    MEASUREMENT: offered = no_digest + untriggered + triggered, and triggered =
    dedup + cap + queued, per role. Off: nothing is counted at all."""
    _seed_names()
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["fda_clinical"] * 12))
    kw = dict(ticker="AR", rationale="r", digest_text="body",
              articles=_arts("Phase 3 trial met primary endpoint"))
    assert cr.maybe_submit(engine="deepseek", first_pass="guidance", final_label="guidance",
                           digest_id=None, **kw) is None                       # no digest
    assert cr.maybe_submit(engine="deepseek", first_pass="fda_clinical",
                           final_label="fda_clinical", digest_id="dg-c1", **kw) is None   # untriggered
    assert cr.maybe_submit(engine="deepseek", first_pass="guidance", final_label="guidance",
                           digest_id="dg-c2", **kw) == "keyword"               # queued
    assert cr.maybe_submit(engine="deepseek", first_pass="guidance", final_label="guidance",
                           digest_id="dg-c2", **kw) is None                    # dedup
    assert cr.maybe_submit(engine="local", first_pass="guidance", final_label="guidance",
                           digest_id="dg-c2", role="shadow", run_id="run-shadow",
                           **kw) == "keyword"                                  # shadow queued
    counts = cr.pop_trigger_counts()
    assert counts == {
        "offered:primary": 4, "offered:shadow": 1, "no_digest": 1, "untriggered": 1,
        "triggered:keyword": 3, "dedup": 1,
        "queued:keyword": 1, "queued:keyword@shadow": 1,
    }
    offered = sum(v for k, v in counts.items() if k.startswith("offered:"))
    triggered = sum(v for k, v in counts.items() if k.startswith("triggered:"))
    queued = sum(v for k, v in counts.items() if k.startswith("queued:"))
    assert offered == counts["no_digest"] + counts["untriggered"] + triggered
    assert triggered == counts["dedup"] + counts.get("cap", 0) + queued
    assert cr.pop_trigger_counts() == {}                                       # copy-and-clear
    cr._drain_for_tests()
    rows = cr.pop_catalyst_repair_rows()
    assert sorted(r["trigger"] for r in rows) == ["keyword", "keyword@shadow"]
    assert {r["run_id"] for r in rows} == {"run-test", "run-shadow"}
    # The cap is a triggered-but-not-queued outcome too.
    monkeypatch.setattr(settings, "catalyst_repair_max_pending", 1, raising=False)
    monkeypatch.setattr(cr, "_PENDING", 1)
    assert cr.maybe_submit(engine="deepseek", first_pass="guidance", final_label="guidance",
                           digest_id="dg-c3", **kw) is None
    monkeypatch.setattr(cr, "_PENDING", 0)
    assert cr.pop_trigger_counts() == {"offered:primary": 1, "triggered:keyword": 1, "cap": 1}
    # Off: no offer is counted (the line would otherwise report a pass that is not running).
    monkeypatch.setattr(settings, "enable_catalyst_repair", False, raising=False)
    assert cr.maybe_submit(engine="deepseek", first_pass="guidance", final_label="guidance",
                           digest_id="dg-c4", **kw) is None
    assert cr.pop_trigger_counts() == {}


def test_tick_summary_line_is_exact():
    counts = {"offered:primary": 161, "offered:shadow": 14,
              "triggered:fund": 3, "triggered:unreliable_class": 9,
              "triggered:keyword": 22, "triggered:monitor": 7,
              "queued:fund": 2, "queued:keyword": 17, "queued:keyword@shadow": 12,
              "dedup": 10}
    assert cr.format_tick_summary(counts, 27, 4) == (
        "offered 175 (primary 161 / shadow 14); "
        "triggered 41 (23.4%: fund 3 / unreliable_class 9 / keyword 22 / monitor 7); "
        "queued 31 (shadow 12); dedup 10; cap 0; no digest 0; pending 4; persisted 27 row(s)")
    # The rate excludes offers that carried no digest; an idle tick reads n/a, never 0%.
    assert cr.format_tick_summary({"offered:primary": 5, "no_digest": 5}, 0, 0).startswith(
        "offered 5 (primary 5 / shadow 0); triggered 0 (n/a: none); queued 0 (shadow 0); "
        "dedup 0; cap 0; no digest 5;")
    assert cr.format_tick_summary({}, 0, 0) == (
        "offered 0 (primary 0 / shadow 0); triggered 0 (n/a: none); queued 0 (shadow 0); "
        "dedup 0; cap 0; no digest 0; pending 0; persisted 0 row(s)")


def test_pipeline_logs_the_tick_summary_from_the_drained_counts():
    from src import pipeline
    src = open(pipeline.__file__, encoding="utf-8").read()
    assert "catalyst_repair.pop_trigger_counts()" in src
    assert "catalyst_repair.format_tick_summary(counts, n_repair, pending)" in src


def test_dedupe_is_primed_from_the_db_across_a_restart(repair_on, monkeypatch):
    """The sentiment cache outlives the process, the in-memory dedupe did not:
    a restart used to re-run the specialist over every verdict the cache still
    served. The last 7 days of `catalyst_repairs` seed the dedupe once per
    process; an older row does not."""
    from datetime import datetime, timedelta, timezone
    from src.db import repo
    _seed_names()
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["fda_clinical"] * 6))
    kw = dict(ticker="AR", first_pass="guidance", final_label="guidance",
              rationale="r", digest_text="body",
              articles=_arts("Phase 3 trial met primary endpoint"))
    # Produce one real row, persist it, then forget everything in memory.
    assert cr.maybe_submit(digest_id="dg-prime-fresh", engine="deepseek", **kw) == "keyword"
    cr._drain_for_tests()
    rows = cr.pop_catalyst_repair_rows()
    assert len(rows) == 1
    old_ts = (datetime.now(timezone.utc) - timedelta(days=cr._PRIME_DAYS + 3)).isoformat()
    stale = dict(rows[0], digest_id="dg-prime-stale", generated_at=old_ts)
    repo.insert_catalyst_repairs(rows + [stale])
    try:
        cr._reset_for_tests()
        cr._PRIMED["done"] = False                                 # a fresh process
        assert cr.maybe_submit(digest_id="dg-prime-fresh", engine="deepseek", **kw) is None      # primed
        assert cr.maybe_submit(digest_id="dg-prime-stale", engine="deepseek", **kw) == "keyword" # too old
        assert cr.maybe_submit(digest_id="dg-prime-fresh", engine="local", **kw) == "keyword"    # other engine
        counts = cr.pop_trigger_counts()
        assert counts["dedup"] == 1 and counts["queued:keyword"] == 2
        assert cr._PRIMED["done"] is True
        cr._drain_for_tests()
        cr.pop_catalyst_repair_rows()
    finally:
        repo.fetch_df("DELETE FROM catalyst_repairs WHERE digest_id LIKE 'dg-prime-%'")


def test_shadow_first_pass_is_submitted_under_its_own_role():
    """The shadow engine's label is the other half of the pair the eval is
    measured on; it is submitted from `_run_shadow` (own (digest, engine) row,
    trigger stamped `<trigger>@shadow`), attributed to the engine that ANSWERED
    and carrying the shadow thread's own run id."""
    src = open(sent.__file__, encoding="utf-8").read()
    body = src.split("def _run_shadow(")[1].split("def _submit_shadow(")[0]
    assert "catalyst_repair.maybe_submit(" in body
    assert 'role="shadow"' in body and "run_id=run_id" in body
    assert "engine=answered" in body and 'digest_text=meta.get("digest")' in body
    assert 'answered = str(meta.get("engine") or engine)' in body


def test_scorer_wires_the_repair_after_the_fund_override():
    """The scorer hands the specialist the RAW first pass and the overridden
    final label, on the primary's own digest — the meta it returns keeps both."""
    src = open(sent.__file__, encoding="utf-8").read()
    assert "catalyst_repair.maybe_submit(" in src
    assert "first_pass=catalyst_raw" in src and "final_label=catalyst" in src
    assert "fund_catalyst_override(ticker, catalyst_raw)" in src


def test_repair_rows_round_trip_through_the_repo(repair_on, monkeypatch):
    """A drained row lands in `catalyst_repairs` with its votes serialised, and a
    later outcome for the same (digest, engine) REPLACES the earlier one — the
    idempotency the non-blocking drain relies on (a repair can outlive its
    tick, so there is no run-wide delete)."""
    from src.db import repo
    _seed_names()
    monkeypatch.setattr(cr, "_call_specialist",
                        _Specialist(["fda_clinical", "fda_clinical", "product"]))
    cr.maybe_submit(ticker="AR", engine="deepseek", first_pass="product",
                    final_label="product", rationale="r", digest_id="dg-rt",
                    digest_text="body", articles=_arts("Phase 3 trial met primary endpoint"))
    cr._drain_for_tests()
    rows = cr.pop_catalyst_repair_rows()
    assert len(rows) == 1
    assert rows[0]["arm"] == "live" and rows[0]["rationale"] == "r"
    repo.insert_catalyst_repairs(rows)
    got = repo.fetch_df("SELECT digest_id, engine, first_pass, trigger, final, quality, "
                        "n_calls, votes, specialist_model FROM catalyst_repairs")
    assert len(got) == 1
    r = got.iloc[0]
    assert (r["digest_id"], r["engine"], r["first_pass"], r["trigger"]) == \
        ("dg-rt", "deepseek", "product", "keyword")
    assert (r["final"], r["quality"], int(r["n_calls"])) == ("fda_clinical", "resolved", 3)
    assert r["specialist_model"] == f"local/{cr.specialist_model()}"
    import json
    votes = json.loads(r["votes"])
    assert [v["catalyst"] for v in votes] == ["fda_clinical", "fda_clinical", "product"]
    # Same (digest, engine) again: replaced, not duplicated.
    repo.insert_catalyst_repairs([dict(rows[0], quality="unresolved", final="product")])
    got = repo.fetch_df("SELECT quality, final FROM catalyst_repairs WHERE digest_id = 'dg-rt'")
    assert len(got) == 1 and got.iloc[0]["quality"] == "unresolved"
    # A different engine's first pass on the same digest keeps its own row.
    repo.insert_catalyst_repairs([dict(rows[0], engine="local")])
    got = repo.fetch_df("SELECT engine FROM catalyst_repairs WHERE digest_id = 'dg-rt' ORDER BY engine")
    assert list(got["engine"]) == ["deepseek", "local"]


def test_pipeline_drains_the_repair_rows_into_the_db():
    """The pass computes in a background pool and buffers in memory; a drain
    nobody calls is a row buffer that grows for the life of the scheduler and
    a table that never fills — indistinguishable from a working pass. Pin the
    wiring in `_persist_run` mechanically."""
    from src import pipeline
    src = open(pipeline.__file__, encoding="utf-8").read()
    assert "catalyst_repair.pop_catalyst_repair_rows()" in src
    assert "repo.insert_catalyst_repairs(repair_rows)" in src
    assert "catalyst_repair.catalyst_repair_pending()" in src


# ── Offline THINK arm (thinking ON, measurement only, never the tick path) ──
def _fake_local(sink, *, content, finish="stop", reasoning=None, prompt_tokens=900):
    """An OpenAI-compatible client whose one completion is scripted; every
    ``create`` kwargs dict lands in ``sink``."""
    class _Completions:
        def create(self, **kw):
            sink.append(kw)
            return types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=prompt_tokens),
                choices=[types.SimpleNamespace(
                    finish_reason=finish,
                    message=types.SimpleNamespace(content=content, reasoning=reasoning))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))


_ANSWER = ('{"about_target": "target", "evidence": "trial met primary endpoint", '
           '"event_family": "operations", "catalyst": "fda_clinical"}')


def test_think_extra_body_switches_thinking_on_in_the_servers_own_dialect(monkeypatch):
    """The arm speaks the dialect the server is KNOWN to honour (the sentiment
    route's ``reasoning_effort``); a dialect without that key is refused rather
    than guessed at — a key the server ignores would run the arm with thinking
    OFF and label the result as the think arm."""
    monkeypatch.setattr(settings, "local_sentiment_extra_body",
                        '{"reasoning_effort": "none"}', raising=False)
    body = cr.think_extra_body()
    assert body["reasoning_effort"] == cr._THINK_EFFORT != "none"
    monkeypatch.setattr(settings, "local_sentiment_extra_body", '{"foo": 1}', raising=False)
    with pytest.raises(RuntimeError, match="reasoning_effort"):
        cr.think_extra_body()


def test_think_call_budgets_the_chain_and_skips_the_ratio_guard(repair_on, monkeypatch):
    """Probed on Ollama 0.33.2 / qwen3:8b: ``max_tokens`` caps chain + answer
    together, and the chain is counted INSIDE ``prompt_tokens`` — so the arm
    reserves its 1,500-token budget out of the context BEFORE judging the
    prompt, and the post-call truncation ratio (meaningless once the chain is
    in the count) is skipped. The live arm keeps both guards as they were."""
    from src.analysis import local_llm
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 8192, raising=False)
    monkeypatch.setattr(settings, "local_sentiment_extra_body",
                        '{"reasoning_effort": "none"}', raising=False)
    fits, reported, sent_kw = [], [], []
    monkeypatch.setattr(local_llm, "check_fits",
                        lambda prompt, *, context_tokens, label: fits.append(context_tokens) or 100)
    monkeypatch.setattr(local_llm, "check_reported", lambda *a, **k: reported.append(k))
    monkeypatch.setattr(sent, "_get_local",
                        lambda: _fake_local(sent_kw, content=_ANSWER, reasoning="chain " * 50))
    order = cr.class_order("d-think", 0)
    data = cr._call_specialist("prompt", order, label="t", think=True)
    kw = sent_kw[-1]
    assert kw["max_tokens"] == cr._THINK_MAX_TOKENS == 1500
    assert kw["extra_body"]["reasoning_effort"] == cr._THINK_EFFORT
    assert fits == [8192 - 1500] and reported == []
    assert data["catalyst"] == "fda_clinical"
    assert data["_reasoning_chars"] == len("chain " * 50)
    # Live arm on the same client: the 512-token answer budget, effort off,
    # the ratio guard consulted, no chain field on the answer.
    data = cr._call_specialist("prompt", order, label="t")
    kw = sent_kw[-1]
    assert kw["max_tokens"] == cr._SPECIALIST_MAX_TOKENS == 512
    assert kw["extra_body"] == {"reasoning_effort": "none"}
    assert fits[-1] == 8192 - 512 and len(reported) == 1
    assert "_reasoning_chars" not in data


def test_an_exhausted_think_budget_is_refused_and_becomes_a_discarded_vote(repair_on, monkeypatch):
    """``finish_reason == "length"`` on the think arm means the chain ate the
    answer (probed: EMPTY content) — refused as an error, never parsed; through
    ``classify_digest`` the digest stays ``unresolved`` on its first pass."""
    from src.analysis import local_llm
    monkeypatch.setattr(settings, "local_sentiment_context_tokens", 8192, raising=False)
    monkeypatch.setattr(settings, "local_sentiment_extra_body",
                        '{"reasoning_effort": "none"}', raising=False)
    monkeypatch.setattr(local_llm, "check_fits", lambda prompt, *, context_tokens, label: 100)
    monkeypatch.setattr(sent, "_get_local",
                        lambda: _fake_local([], content="", finish="length", reasoning="x" * 6000))
    with pytest.raises(RuntimeError, match="think budget exhausted"):
        cr._call_specialist("prompt", cr.class_order("d-len", 0), label="t", think=True)
    out = cr.classify_digest(digest_id="d-len", ticker="AR", digest_text="t", rationale="r",
                             first_pass="guidance", arm=cr.ARM_THINK)
    assert out["votes"] == [] and out["n_calls"] == 1
    assert out["quality"] == "unresolved" and out["final"] == "guidance"
    assert "think budget exhausted" in out["error"] and out["arm"] == "think"


def test_vote_memo_is_salted_by_arm(repair_on):
    v = {"catalyst": "earnings", "about_target": "target"}
    cr._store_vote("d-memo", 0, v)
    assert cr._cached_vote("d-memo", 0, arm=cr.ARM_THINK) == (False, None)
    cr._store_vote("d-memo", 0, dict(v, catalyst="guidance"), cr.ARM_THINK)
    assert cr._cached_vote("d-memo", 0) == (True, v)
    assert cr._cached_vote("d-memo", 0, arm=cr.ARM_THINK)[1]["catalyst"] == "guidance"


def test_classify_digest_on_the_think_arm_thinks_on_every_call_and_stamps_the_arm(repair_on, monkeypatch):
    spec = _Specialist(["fda_clinical", "fda_clinical", "product"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    out = cr.classify_digest(digest_id="d-arm", ticker="AR", digest_text="t", rationale="r",
                             first_pass="product", arm=cr.ARM_THINK)
    assert [c[3] for c in spec.calls] == [True, True, True]
    assert out["arm"] == "think" and out["final"] == "fda_clinical" and out["quality"] == "resolved"
    assert out["specialist_version"] == cr.SPECIALIST_VERSION + "+think"
    # The live arm on the SAME digest: its own memo (the think votes above must
    # not be served — one agreeing call resolves it), no think kw, its own stamp.
    spec2 = _Specialist(["product"])
    monkeypatch.setattr(cr, "_call_specialist", spec2)
    out = cr.classify_digest(digest_id="d-arm", ticker="AR", digest_text="t", rationale="r",
                             first_pass="product")
    assert spec2.calls[0][3] is False and out["n_calls"] == 1
    assert out["arm"] == "live" and out["specialist_version"] == cr.SPECIALIST_VERSION


def _live_row(**over):
    from datetime import datetime, timezone
    row = {"digest_id": "dg-live", "run_id": "run-live",
           "generated_at": datetime.now(timezone.utc).isoformat(), "ticker": "AR",
           "engine": "deepseek", "model": "deepseek-v4-flash", "first_pass": "product",
           "trigger": "keyword", "specialist_model": "local/qwen3:8b",
           "specialist_version": cr.SPECIALIST_VERSION, "votes": "[]", "n_calls": 1,
           "final": "fda_clinical", "quality": "resolved", "about_target": "target",
           "latency_s": 1.0, "error": None, "arm": "live", "rationale": "phase 3 readout"}
    row.update(over)
    return row


def test_run_think_arm_replays_the_live_rows_on_their_stored_digest(repair_on, monkeypatch):
    """The offline arm re-judges exactly what the live pass judged — the stored
    digest text and the stored first-pass rationale, the live row's first pass
    and trigger — and lands beside it as ``arm='think'``: the live row is never
    overwritten, a row whose digest was not stored is skipped (nothing to
    replay), and a second run finds nothing left (resumable)."""
    from src.db import repo
    _seed_names()
    repo.insert_catalyst_repairs([_live_row(), _live_row(digest_id="dg-orphan")])
    repo.insert_sentiment_digests([{"digest_id": "dg-live", "ticker": "AR", "run_id": "run-live",
                                    "generated_at": _live_row()["generated_at"], "n_articles": 1,
                                    "digest_text": "Phase 3 trial met primary endpoint"}])
    spec = _Specialist(["product"])                  # agrees with the first pass: one call
    monkeypatch.setattr(cr, "_call_specialist", spec)
    summary = cr.run_think_arm(days=7)
    assert (summary["selected"], summary["done"], summary["inserted"], summary["errors"]) == (1, 1, 1, 0)
    assert len(spec.calls) == 1 and spec.calls[0][3] is True
    prompt = spec.calls[0][0]
    assert "phase 3 readout" in prompt and "Phase 3 trial met primary endpoint" in prompt
    got = repo.fetch_df("SELECT arm, run_id, first_pass, trigger, final, quality, rationale, "
                        "specialist_version FROM catalyst_repairs WHERE digest_id = 'dg-live' "
                        "ORDER BY arm")
    assert list(got["arm"]) == ["live", "think"]
    live, think = got.iloc[0], got.iloc[1]
    assert (live["final"], live["quality"]) == ("fda_clinical", "resolved")      # untouched
    assert (think["run_id"], think["first_pass"], think["trigger"], think["rationale"]) == \
        ("run-live", "product", "keyword", "phase 3 readout")
    assert (think["final"], think["quality"]) == ("product", "resolved")
    assert think["specialist_version"] == cr.SPECIALIST_VERSION + "+think"
    orphan = repo.fetch_df("SELECT arm FROM catalyst_repairs WHERE digest_id = 'dg-orphan'")
    assert list(orphan["arm"]) == ["live"]
    # Resumable: the think row now exists, so nothing is selected again.
    again = cr.run_think_arm(days=7)
    assert again["selected"] == 0 and len(spec.calls) == 1


def test_run_think_arm_honours_the_limit_and_the_override(repair_on, monkeypatch):
    """``limit`` bounds one CLI session; an overridden live row (fund target)
    replays with the override as its final label so the think row is stamped
    ``override`` exactly as the live one was."""
    from src.db import repo
    _seed_names()
    rows = [_live_row(digest_id=f"dg-{i}", ticker="XLV", engine="local", model="local/qwen3:8b",
                      first_pass="earnings", final="macro_sector", quality="override",
                      trigger="fund") for i in range(3)]
    repo.insert_catalyst_repairs(rows)
    repo.insert_sentiment_digests([{"digest_id": f"dg-{i}", "ticker": "XLV", "run_id": "run-live",
                                    "generated_at": rows[i]["generated_at"], "n_articles": 1,
                                    "digest_text": "Sector fund quarterly results"} for i in range(3)])
    spec = _Specialist(["earnings", "earnings", "earnings"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    summary = cr.run_think_arm(days=7, limit=2)
    assert summary["selected"] == 2 and summary["done"] == 2 and len(spec.calls) == 2
    got = repo.fetch_df("SELECT digest_id, final, quality FROM catalyst_repairs "
                        "WHERE arm = 'think' ORDER BY digest_id")
    assert len(got) == 2
    assert set(got["final"]) == {"macro_sector"} and set(got["quality"]) == {"override"}
    assert cr.run_think_arm(days=7)["selected"] == 1        # the third one is left


def test_live_and_think_rows_coexist_under_one_digest_and_engine(repair_on):
    """The idempotency key is ``(digest_id, engine, arm)``: a think row can
    never replace the live row it is compared against, each arm replaces only
    itself, and a row written without an arm (pre-arm history) IS the live row."""
    from src.db import repo
    base = _live_row(digest_id="dg-pair")
    repo.insert_catalyst_repairs([base])
    repo.insert_catalyst_repairs([dict(base, arm="think", final="product", quality="unresolved")])
    got = repo.fetch_df("SELECT arm, final FROM catalyst_repairs WHERE digest_id = 'dg-pair' ORDER BY arm")
    assert list(zip(got["arm"], got["final"])) == [("live", "fda_clinical"), ("think", "product")]
    repo.insert_catalyst_repairs([dict(base, arm="think", final="earnings", quality="resolved")])
    got = repo.fetch_df("SELECT arm, final FROM catalyst_repairs WHERE digest_id = 'dg-pair' ORDER BY arm")
    assert list(zip(got["arm"], got["final"])) == [("live", "fda_clinical"), ("think", "earnings")]
    repo.insert_catalyst_repairs([dict(base, arm=None, final="guidance")])
    got = repo.fetch_df("SELECT arm, final FROM catalyst_repairs WHERE digest_id = 'dg-pair' ORDER BY arm")
    assert list(zip(got["arm"], got["final"])) == [("live", "guidance"), ("think", "earnings")]


def test_think_arm_block_pairs_the_arms_and_grades_both_against_one_gold():
    """Pairs on ``(digest_id, engine)`` only; a judgment is digest-level truth
    for either engine's row, DeepSeek's label grades a LOCAL row only (a
    DeepSeek row is never graded against itself), and agreement between the
    arms is reported separately from error — agreement is not skill."""
    import pandas as pd
    live = pd.DataFrame([
        {"digest_id": "d1", "engine": "deepseek", "final": "fda_clinical", "quality": "resolved",
         "n_calls": 1, "latency_s": 1.0, "error": None, "generated_at": "2026-09-07T10:00:00+00:00"},
        {"digest_id": "d2", "engine": "local", "final": "product", "quality": "unresolved",
         "n_calls": 3, "latency_s": 3.0, "error": None, "generated_at": "2026-09-07T11:00:00+00:00"},
        {"digest_id": "d3", "engine": "local", "final": "earnings", "quality": "resolved",
         "n_calls": 1, "latency_s": 1.0, "error": None, "generated_at": "2026-09-07T12:00:00+00:00"},
    ])
    think = pd.DataFrame([
        {"digest_id": "d1", "engine": "deepseek", "final": "fda_clinical", "quality": "resolved",
         "n_calls": 1, "latency_s": 13.0, "error": None, "first_pass": "product",
         "generated_at": "2026-09-07T20:00:00+00:00"},
        {"digest_id": "d2", "engine": "local", "final": "fda_clinical", "quality": "resolved",
         "n_calls": 1, "latency_s": 15.0, "error": "x", "first_pass": "product",
         "generated_at": "2026-09-07T20:01:00+00:00"},
        {"digest_id": "d4", "engine": "deepseek", "final": "earnings", "quality": "resolved",
         "n_calls": 1, "latency_s": 13.0, "error": None, "first_pass": "earnings",
         "generated_at": "2026-09-07T20:02:00+00:00"},
    ])
    pairs = pd.DataFrame([{"digest_id": "d2", "gold": "fda_clinical"},
                          {"digest_id": "d1", "gold": "product"}])       # DeepSeek's own label
    judgments = pd.DataFrame([{"digest_id": "d1", "verdict": "err",
                               "correct_catalyst": "guidance", "model_catalyst": "fda_clinical"}])
    t = cr._think_arm_block(live, think, pairs, judgments)
    assert t["n_pairs"] == 2 and t["agreement"] == 0.5
    assert t["resolved_share"] == {"live": 0.5, "think": 1.0}
    assert t["mean_calls"] == {"live": 2.0, "think": 1.0}
    assert t["mean_latency_s"] == {"live": 2.0, "think": 14.0}
    assert t["error_share"] == {"live": 0.0, "think": 0.5}
    # d1: judged gold "guidance" (overrides the pairs label, grades the DeepSeek row);
    # d2: DeepSeek's label grades the local row.
    assert t["n_judged"] == 2
    assert t["first_pass_err"] == 1.0 and t["live_err"] == 1.0 and t["think_err"] == 0.5
    assert t["by_engine"] == {"deepseek": {"n": 1, "live_err": 1.0, "think_err": 1.0},
                              "local": {"n": 1, "live_err": 1.0, "think_err": 0.0}}
    assert t["halves_live_err"] == [1.0, 1.0] and t["halves_think_err"] == [1.0, 0.0]
    # Without a judgment, a DeepSeek row has no gold at all.
    t2 = cr._think_arm_block(live, think, pairs, None)
    assert t2["n_pairs"] == 2 and t2["n_judged"] == 1
    assert t2["by_engine"] == {"local": {"n": 1, "live_err": 1.0, "think_err": 0.0}}
    assert cr._think_arm_block(live, think.iloc[0:0], pairs, judgments) == {"n_pairs": 0}


def test_evaluate_runs_end_to_end_over_the_string_timestamps(repair_on, monkeypatch, capsys):
    """Every ``generated_at`` here is a VARCHAR ISO string, and DuckDB refuses
    ``varchar >= now() - INTERVAL n DAY`` with a binder error — the three
    day-bounded reads (live rows, think rows, shadow pairs) must bound on an
    ISO cutoff string instead. Seeded with one live row, its think twin, the
    DeepSeek/local shadow pair it came from and one judgment, the report must
    see all of them."""
    from src.db import repo
    live = _live_row(digest_id="dg-eval", engine="local", model="local/qwen3:8b",
                     first_pass="product", trigger="unreliable_class@shadow",
                     final="fda_clinical", quality="resolved")
    repo.insert_catalyst_repairs([live, dict(live, arm="think", final="fda_clinical",
                                             latency_s=13.0)])
    repo.insert_sentiment_shadow([{
        "run_id": "run-live", "generated_at": live["generated_at"], "ticker": "AR",
        "digest_hash": "h", "n_articles": 1, "digest_id": "dg-eval",
        "primary_engine": "deepseek", "primary_model": "deepseek-v4-flash",
        "primary_raw": 0.3, "primary_score": 0.2, "primary_catalyst": "fda_clinical",
        "shadow_engine": "local", "shadow_model": "local/qwen3:8b",
        "shadow_raw": 0.25, "shadow_score": 0.15, "shadow_catalyst": "product",
        "shadow_latency_s": 2.0}])
    repo.insert_catalyst_judgments([{
        "judgment_id": "j1", "judged_at": live["generated_at"], "source": "test",
        "engine": "local", "model": "local/qwen3:8b", "ticker": "AR", "run_id": "run-live",
        "digest_id": "dg-eval", "model_catalyst": "product", "verdict": "err",
        "correct_catalyst": "fda_clinical", "entity_error": False}])
    out = cr.evaluate(days=30)
    capsys.readouterr()
    assert out["n_repairs"] == 1 and out["n_pairs"] == 1
    assert out["repair_volume"]["rows"] == 1 and out["repair_volume"]["by_trigger"] == {"unreliable_class@shadow": 1}
    assert out["local_specialist"]["flagged"]["n"] == 1
    assert out["local_specialist"]["flagged"]["specialist_err"] == 0.0
    assert out["local_specialist"]["flagged"]["first_pass_err"] == 1.0
    assert out["think_arm"]["n_pairs"] == 1 and out["think_arm"]["agreement"] == 1.0
    assert out["think_arm"]["think_err"] == 0.0 and out["think_arm"]["live_err"] == 0.0
    # Outside the window: nothing is read, and the report still renders.
    monkeypatch.setattr(cr, "_days_cutoff_iso", lambda days: "2999-01-01T00:00:00+00:00")
    out = cr.evaluate(days=30)
    capsys.readouterr()
    assert out["n_repairs"] == 0 and out["n_pairs"] == 0 and out.get("think_arm", {"n_pairs": 0})["n_pairs"] == 0


def test_run_think_arm_keeps_a_batch_whose_write_lost_the_lock(repair_on, monkeypatch):
    """The arm runs beside a live scheduler (the sole writer). A flush that
    fails keeps its rows for the next flush point; rows still unwritten at
    the end are reported and re-selected by the next run — never dropped
    silently, never raised out of a measurement."""
    from src.db import repo
    _seed_names()
    repo.insert_catalyst_repairs([_live_row()])
    repo.insert_sentiment_digests([{"digest_id": "dg-live", "ticker": "AR", "run_id": "run-live",
                                    "generated_at": _live_row()["generated_at"], "n_articles": 1,
                                    "digest_text": "Phase 3 trial met primary endpoint"}])
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["product"]))
    real = repo.insert_catalyst_repairs

    def _boom(rows):
        raise RuntimeError("Could not set lock on file")
    monkeypatch.setattr(repo, "insert_catalyst_repairs", _boom)
    summary = cr.run_think_arm(days=7)
    assert summary["done"] == 1 and summary["inserted"] == 0
    assert summary["unwritten"] == 1 and summary["stopped"] == "write_failed"
    monkeypatch.setattr(repo, "insert_catalyst_repairs", real)
    monkeypatch.setattr(cr, "_call_specialist", _Specialist(["product"]))
    assert cr.run_think_arm(days=7)["inserted"] == 1        # re-selected, written


# ── consumer resolution: repair → live → backfill ───────────────────────────

def test_repair_lookup_keys_on_the_label_the_repair_started_from():
    """One digest can hold two repairs — the primary engine's and the shadow
    engine's, which start from DIFFERENT labels. The lookup is therefore keyed
    by (digest, starting label) so a consumer gets the row that repaired the
    label it holds: `first_pass` normally, `final` on an override row (the fund
    override had already rewritten the label the panel stores). Unresolved rows
    are returned (the caller must be able to EXCLUDE them); the offline think
    arm is not (a later replay must never resolve a walk-forward view)."""
    from src.db import repo
    repo.insert_catalyst_repairs([
        _live_row(digest_id="dg1", engine="deepseek", first_pass="product",
                  final="fda_clinical", quality="resolved", trigger="keyword"),
        _live_row(digest_id="dg1", engine="local", first_pass="other",
                  final="management", quality="resolved", trigger="keyword@shadow"),
        _live_row(digest_id="dg2", first_pass="earnings", final="macro_sector",
                  quality="override", trigger="fund"),
        _live_row(digest_id="dg3", first_pass="company_pr", final="company_pr",
                  quality="unresolved"),
        _live_row(digest_id="dg4", first_pass="other", final="guidance",
                  quality="resolved", arm="think"),
    ])
    lk = cr.repair_lookup()
    assert lk[("dg1", "product")]["catalyst"] == "fda_clinical"     # primary row
    assert lk[("dg1", "other")]["catalyst"] == "management"         # shadow row, own key
    assert lk[("dg2", "macro_sector")]["quality"] == "override"     # keyed on the FINAL label
    assert ("dg2", "earnings") not in lk
    assert lk[("dg3", "company_pr")]["quality"] == "unresolved"     # returned, not dropped
    assert ("dg4", "other") not in lk                               # think arm excluded


def test_repair_lookup_prefers_the_primary_row_over_a_later_shadow():
    """Both engines can start from the SAME label and repair it differently.
    The panel's label is the primary's, so the primary row wins even when the
    shadow row was written later."""
    from datetime import datetime, timedelta, timezone
    from src.db import repo
    now = datetime.now(timezone.utc)
    repo.insert_catalyst_repairs([
        _live_row(digest_id="dg5", engine="deepseek", first_pass="other",
                  final="management", trigger="keyword",
                  generated_at=(now - timedelta(minutes=5)).isoformat()),
        _live_row(digest_id="dg5", engine="local", first_pass="other",
                  final="product", trigger="keyword@shadow",
                  generated_at=now.isoformat()),
    ])
    assert cr.repair_lookup()[("dg5", "other")]["catalyst"] == "management"


def test_repair_lookup_is_bounded_by_the_asof_cutoff():
    """A repair runs AFTER the verdict it repairs, so under a walk-forward
    cutoff it is knowable only once its own `generated_at` has passed."""
    from src.analysis.asof import analysis_asof
    from src.db import repo
    repo.insert_catalyst_repairs([_live_row(digest_id="dg6", first_pass="product")])
    assert ("dg6", "product") in cr.repair_lookup()
    with analysis_asof("2026-01-01"):
        assert cr.repair_lookup() == {}


def _signal_row(**over):
    from src.db.schema import SIGNAL_METHOD_COLUMNS
    row = {"ticker": "AR", "type": "STOCK", "direction": "BULLISH",
           "combined_score": 0.2, "confidence": 0.8, "n_methods_agreeing": 1,
           "dominant_method": "news", "price": 30.0, "news": 0.4,
           "news_catalyst": "product", "news_raw_score": 0.5,
           "news_digest_id": "dg1",
           "scores": {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}}
    row.update(over)
    row["scores"]["news"] = row["news"]
    return row


def _load_events(monkeypatch):
    import pandas as pd
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "build_panel", lambda **kw: pd.DataFrame())
    from src.analysis.news_events import load_news_events
    return load_news_events()


def test_news_events_resolves_repair_then_live(monkeypatch):
    """The resolution order is repair → live → backfill: a resolved repair
    replaces the label and stamps its provenance, an UNRESOLVED one leaves the
    live label in place and only records that it was checked, and a digest with
    no repair is untouched."""
    from src.db import repo
    monkeypatch.setattr(settings, "enable_catalyst_repair_resolution", True, raising=False)
    repo.insert_signals("run-1", "2026-08-20T14:00:00+00:00", "2026-08-20", [
        _signal_row(ticker="AR", news_digest_id="dg1", news_catalyst="product"),
        _signal_row(ticker="LLY", news_digest_id="dg3", news_catalyst="company_pr"),
        _signal_row(ticker="BAC", news_digest_id="dg9", news_catalyst="earnings"),
    ])
    repo.insert_catalyst_repairs([
        _live_row(digest_id="dg1", first_pass="product", final="fda_clinical",
                  quality="resolved"),
        _live_row(digest_id="dg3", first_pass="company_pr", final="company_pr",
                  quality="unresolved"),
    ])
    ev = _load_events(monkeypatch).set_index("ticker")
    assert (ev.loc["AR", "catalyst"], ev.loc["AR", "catalyst_source"],
            ev.loc["AR", "catalyst_quality"]) == ("fda_clinical", "repair", "resolved")
    assert (ev.loc["LLY", "catalyst"], ev.loc["LLY", "catalyst_source"],
            ev.loc["LLY", "catalyst_quality"]) == ("company_pr", "live", "unresolved")
    assert (ev.loc["BAC", "catalyst"], ev.loc["BAC", "catalyst_source"]) == ("earnings", "live")
    assert ev.loc["BAC", "catalyst_quality"] is None


def test_news_events_is_byte_identical_with_the_resolution_off(monkeypatch):
    """The pass stays pure ACCRUAL until the pre-registered bar clears: with the
    flag off the repair is not consulted at all, so a repaired label cannot
    reach a calibration by accident."""
    from src.db import repo
    monkeypatch.setattr(settings, "enable_catalyst_repair_resolution", False, raising=False)
    repo.insert_signals("run-1", "2026-08-20T14:00:00+00:00", "2026-08-20",
                        [_signal_row(news_digest_id="dg1", news_catalyst="product")])
    repo.insert_catalyst_repairs([_live_row(digest_id="dg1", first_pass="product",
                                            final="fda_clinical", quality="resolved")])
    ev = _load_events(monkeypatch)
    assert ev.iloc[0]["catalyst"] == "product"
    assert ev.iloc[0]["catalyst_source"] == "live"
    assert ev.iloc[0]["catalyst_quality"] is None


# ── the judged gold sample (~100 rows/week) ─────────────────────────────────

def test_judge_sheet_samples_both_strata_and_skips_judged_rows(tmp_path):
    """The bar is measured on two populations, so the sheet draws from both —
    flagged rows (where the specialist must earn its place) and monitor rows
    (where it must not make things worse) — at random WITHIN each stratum, not
    by picking the interesting disagreements, which would inflate both error
    rates. A digest already judged for that model is not offered again."""
    import json
    from src.db import repo
    rows = []
    for i in range(6):
        rows.append(_live_row(digest_id=f"f{i}", trigger="unreliable_class",
                              first_pass="product", final="fda_clinical"))
    for i in range(6):
        rows.append(_live_row(digest_id=f"m{i}", trigger="monitor",
                              first_pass="earnings", final="earnings"))
    repo.insert_catalyst_repairs(rows)
    repo.insert_sentiment_digests([{"digest_id": "f0", "ticker": "AR", "run_id": "run-live",
                                    "generated_at": _live_row()["generated_at"],
                                    "n_articles": 1, "digest_text": "Phase 3 readout"}])
    repo.insert_catalyst_judgments([{"judgment_id": "j1", "digest_id": "f1",
                                     "model": "deepseek-v4-flash", "engine": "deepseek",
                                     "verdict": "ok", "judged_at": "2026-09-06T00:00:00+00:00"}])
    out_path = str(tmp_path / "sheet.json")
    summary = cr.judge_sheet(n=6, days=7, out_path=out_path, seed=7)
    assert summary["by_stratum"] == {"flagged": 3, "monitor": 3}
    sheet = json.load(open(out_path, encoding="utf-8"))
    ids = [r["digest_id"] for r in sheet["rows"]]
    assert "f1" not in ids                                  # already judged
    assert len(set(ids)) == 6
    row = next(r for r in sheet["rows"] if r["digest_id"].startswith("f"))
    # the loader's schema, blank for the human, plus the context to decide on
    assert row["verdict"] == "" and row["correct_catalyst"] == ""
    assert row["model_catalyst"] == "product" and row["_specialist"] == "fda_clinical"
    assert row["rationale"] == "phase 3 readout"
    assert set(sheet["_classes"]) >= {"earnings", "fda_clinical", "none"}
    # the digest the scorer saw rides the row when it was stored
    assert next(r for r in sheet["rows"] if r["digest_id"] == "f0")["_digest"] == "Phase 3 readout"


def test_judge_sheet_spends_its_whole_budget_when_one_stratum_is_thin(tmp_path):
    from src.db import repo
    repo.insert_catalyst_repairs([_live_row(digest_id=f"f{i}", trigger="keyword")
                                  for i in range(8)]
                                 + [_live_row(digest_id="m0", trigger="monitor")])
    summary = cr.judge_sheet(n=6, days=7, out_path=str(tmp_path / "s.json"), seed=1)
    assert summary["by_stratum"] == {"flagged": 5, "monitor": 1}
    assert summary["written"] == 6


def test_seed_judgments_skips_blanks_and_refuses_an_unusable_correction(tmp_path):
    """A blank verdict is a sheet line nobody reached, not a judgment — so a
    partly-filled sheet is safe to load and re-load. An `err` whose corrected
    class is missing or invented is DROPPED and named: a wrong gold moves every
    error rate measured against it, which is worse than a missing one."""
    import json
    from src.db import repo
    path = tmp_path / "filled.json"
    path.write_text(json.dumps({"rows": [
        {"judgment_id": "a", "digest_id": "d1", "verdict": "", "correct_catalyst": ""},
        {"judgment_id": "b", "digest_id": "d2", "verdict": "ok", "model_catalyst": "earnings",
         "_stratum": "monitor"},
        {"judgment_id": "c", "digest_id": "d3", "verdict": "err",
         "correct_catalyst": "fda_clinical"},
        {"judgment_id": "d", "digest_id": "d4", "verdict": "err", "correct_catalyst": "banana"},
        {"judgment_id": "e", "digest_id": "d5", "verdict": "err", "correct_catalyst": ""},
    ]}), encoding="utf-8")
    assert cr.seed_judgments(str(path)) == 2
    df = repo.fetch_df("SELECT * FROM catalyst_judgments ORDER BY judgment_id")
    assert list(df.judgment_id) == ["b", "c"]
    assert df.judged_at.notna().all()                       # stamped on load
    assert "_stratum" not in df.columns                     # sheet context is not persisted


def test_evaluate_reports_the_error_rate_as_a_weekly_series(repair_on, monkeypatch):
    """One pooled number cannot separate "the specialist improved" from "that
    week's digests were easier" — the first pass and the specialist are scored
    on the SAME rows each week, so both curves are reported."""
    from datetime import datetime, timedelta, timezone
    from src.db import repo
    now = datetime.now(timezone.utc)
    last_week = now - timedelta(days=8)
    rows, pairs, judgments = [], [], []
    for tag, when in (("a", last_week), ("b", now)):
        for i in range(2):
            d = f"{tag}{i}"
            rows.append(_live_row(digest_id=d, engine="local", model="local/qwen3:8b",
                                  trigger="unreliable_class", first_pass="product",
                                  final="fda_clinical", generated_at=when.isoformat()))
            pairs.append({"run_id": "r1", "generated_at": when.isoformat(), "ticker": "AR",
                          "digest_id": d, "primary_engine": "deepseek",
                          "primary_catalyst": "fda_clinical", "shadow_engine": "local",
                          "shadow_catalyst": "product", "primary_score": 0.2,
                          "shadow_score": 0.1})
    repo.insert_catalyst_repairs(rows)
    repo.insert_sentiment_shadow(pairs)
    out = cr.evaluate(days=30)
    series = out["series"]
    assert len(series) == 2 and [r["n"] for r in series] == [2, 2]
    # DeepSeek is the gold here: the local first pass is wrong on every row and
    # the specialist right on every row, in BOTH weeks.
    assert [r["first_pass_err"] for r in series] == [1.0, 1.0]
    assert [r["specialist_err"] for r in series] == [0.0, 0.0]
    assert [r["n_flagged"] for r in series] == [2, 2]
    assert series[0]["week"] < series[1]["week"]


def test_the_eval_pairs_engines_regardless_of_WHICH_ROLE_they_played():
    """The gold is DeepSeek's LABEL, and a label does not care which engine
    happened to drive the combine that run.

    This query used to require `primary_engine='deepseek' AND
    shadow_engine='local'` — true when written, false from 2026-09-09 when local
    took 100% of the primary route and DeepSeek became the shadow. The eval's
    population silently went to ZERO while 1,639 paired digests sat unused, and
    it reported that as "no paired rows with a digest id yet" — indistinguishable
    from "not accrued". It gates a production flag
    (`enable_catalyst_repair_resolution`), so a silently empty population is a
    flag that can never be evaluated.
    """
    import inspect
    import re

    import src.analysis.catalyst_repair as cr
    src = inspect.getsource(cr.evaluate)
    i = src.index("FROM sentiment_shadow")
    where = src[i:i + 700]
    assert "primary_engine = 'local'" in where and "shadow_engine = 'deepseek'" in where,         "the pair query is still role-directional"
    # and it must not hardcode a single direction anywhere in that clause
    assert where.count("OR") >= 1


def test_eval_printing_is_ascii_safe():
    """`print` goes to a Windows console on cp1252; loguru sinks do not. A
    non-ASCII character in a print statement raises UnicodeEncodeError and takes
    the whole eval down — which is what the BAR line did the first time the pair
    query was fixed and that code path became reachable at all."""
    import inspect

    import src.analysis.catalyst_repair as cr
    src = inspect.getsource(cr)
    bad = []
    depth, in_print = 0, False
    for ln in src.splitlines():
        if ln.lstrip().startswith("print("):
            in_print, depth = True, 0
        if in_print:
            for ch in ln:
                if ord(ch) > 127:
                    bad.append((ch, ln.strip()[:70]))
            depth += ln.count("(") - ln.count(")")
            if depth <= 0:
                in_print = False
    assert not bad, f"non-ASCII inside print(): {bad[:4]}"


def test_an_overridden_row_does_not_buy_the_tie_break_votes(repair_on, monkeypatch):
    """The mechanical fund override has already decided the row, so the MAJORITY
    cannot change the outcome. The first vote is still bought — that is the
    "monitoring only" the docstring means — but the two tie-break votes exist
    solely to resolve a dissent into a majority, and on an overridden row they
    are discarded by construction.

    Measured 2026-09-11: override rows averaged **2.99** calls, the MAXIMUM,
    because the specialist reliably dissents from a fund's company-event first
    pass and a dissent buys two more. 664 of 8,548 specialist calls (7.8%,
    ~2,280 s on the GPU that also serves live sentiment) were spent this way."""
    spec = _Specialist(["macro_sector", "macro_sector", "macro_sector"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    out = cr.classify_digest(digest_id="d-ovr", ticker="SPY", digest_text="t",
                             rationale="r", first_pass="earnings",
                             final_label="macro_sector")
    assert out["quality"] == "override" and out["final"] == "macro_sector"
    assert out["n_calls"] == 1, "the tie-break votes cannot change an overridden row"


def test_a_NON_overridden_row_still_takes_the_full_majority(repair_on, monkeypatch):
    """The saving must be narrow: when the override did not fire (a fund whose
    first pass was already `macro_sector`/`none`, or any non-fund), the vote
    decides and all three calls are still bought."""
    spec = _Specialist(["analyst", "guidance", "analyst"])
    monkeypatch.setattr(cr, "_call_specialist", spec)
    out = cr.classify_digest(digest_id="d-same", ticker="SPY", digest_text="t",
                             rationale="r", first_pass="guidance",
                             final_label="guidance")
    assert out["n_calls"] == 3 and out["quality"] == "resolved"


# ── the fund trigger is OFF (2026-09-11, measured) ──────────────────────────

def test_the_fund_trigger_is_off_by_default(repair_on):
    """It was the single largest consumer of the pass — 2,667 of 5,716 rows and
    **4.00 of 9.23 GPU-hours (43%)** on the box that also serves live sentiment —
    and it bought nothing: over 3,213 pairs the specialist scored 23.0% error
    against the first pass's 22.8%.

    It was the weakest trigger by construction: a fund's label is already fixed
    MECHANICALLY by `sentiment.fund_catalyst_override`, so the specialist was
    being asked to re-derive an answer computed for free."""
    from config.settings import Settings
    _seed_names()
    assert Settings.model_fields["catalyst_repair_fund_trigger"].default is False
    assert cr.trigger_for(ticker="GLD", engine="deepseek", first_pass="earnings") is None


def test_dropping_the_trigger_does_NOT_drop_the_LABEL_FIX(repair_on):
    """The override is a different mechanism and stays. This removes a specialist
    CALL, never the correction — a fund labelled with a company-event class is
    still retyped to `macro_sector` at zero cost."""
    import src.analysis.sentiment as sent
    _seed_names()                      # GLD only resolves as a fund once seeded
    assert sent.fund_catalyst_override("GLD", "earnings") == "macro_sector"
    assert sent.fund_catalyst_override("AR", "earnings") == "earnings"


def test_funds_are_still_MEASURED_through_the_monitor_sample(repair_on, monkeypatch):
    """Dropping the trigger must not blind the pass to funds, or the decision
    could never be revisited. The 5% monitor hash sample still draws them, so
    funds keep an UNBIASED read without paying for every one."""
    _seed_names()
    monkeypatch.setattr(settings, "catalyst_repair_monitor_share", 1.0, raising=False)
    assert cr.trigger_for(ticker="GLD", engine="deepseek", first_pass="earnings") == "monitor"
