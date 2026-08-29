"""A trade may legitimately carry confidence=None (2026-08-26 production crash).

Follow-through entries are MECHANICAL — there is no LLM verdict — so the field
is present-but-NULL rather than absent. That distinction broke production:

    t.get("confidence", 0.0)

returns the default ONLY when the key is MISSING. With the key present and
None, it hands None to sorted(), and every scheduler tick died with
"'<' not supported between instances of 'float' and 'NoneType'" from
_compute_confidence_ranked -- for 2.5 hours, from the moment one such trade
closed, taking the whole pipeline down (recommendations, trading, email).

The failure needed a CLOSED None-confidence trade, so it survived every test
and only fired once one hit the ledger. These pin the contract.
"""

import pytest

import src.performance.tracker as tr


def _trade(conf, ticker="AAA", ret=1.0, **over):
    t = {"ticker": ticker, "action": "BUY", "direction": "BULLISH",
         "status": "CLOSED", "confidence": conf, "return_pct": ret,
         "entry_price": 10.0,
         "exit_price": (10.0 * (1 + ret / 100.0)) if ret is not None else None,
         "entry_date": "2026-08-20", "exit_date": "2026-08-21"}
    t.update(over)
    return t


def test_confidence_ranked_survives_a_none_confidence_trade():
    """THE regression. Mixed None and float confidences must sort, not raise."""
    rows = tr._compute_confidence_ranked([
        _trade(0.9, "HIGH"), _trade(None, "MECH"), _trade(0.8, "MID")])
    assert [r["ticker"] for r in rows] == ["HIGH", "MID", "MECH"]


def test_none_confidence_sorts_last_and_stays_none():
    """It ranks last (a trade with no confidence cannot be ranked BY confidence)
    and the value is preserved -- coercing to 0.0 would assert it had the LOWEST
    confidence, which is a different and false claim in a confidence table."""
    rows = tr._compute_confidence_ranked([_trade(None, "MECH"), _trade(0.1, "LOW")])
    assert rows[-1]["ticker"] == "MECH"
    assert rows[-1]["confidence"] is None
    assert rows[0]["confidence"] == 0.1


def test_all_none_confidences_do_not_raise():
    rows = tr._compute_confidence_ranked([_trade(None, "A"), _trade(None, "B")])
    assert len(rows) == 2 and all(r["confidence"] is None for r in rows)


def test_none_return_pct_does_not_break_the_running_total():
    """An unpriced leg leaves return_pct null; the cumulative walk must treat it
    as 0 contribution rather than raising mid-sum."""
    rows = tr._compute_confidence_ranked([
        _trade(0.9, "A", ret=2.0), _trade(0.8, "B", ret=None)])
    assert rows[-1]["cumulative_avg"] == pytest.approx(1.0)


def test_position_multiplier_accepts_none_as_neutral():
    """A mechanical entry has no confidence-based opinion about size. 1.0x is
    also what every confidence <= 0.78 maps to, so this is consistent, not a
    special case."""
    assert tr._position_multiplier(None) == 1.0
    assert tr._position_multiplier(0.70) == 1.0


def test_the_get_default_trap_is_gone():
    """`dict.get(k, default)` does NOT substitute for a present None. Asserted at
    the source because the correct-looking idiom is exactly what shipped."""
    import inspect
    # Strip comment lines: the docstring/comments deliberately QUOTE the broken
    # idiom to explain the incident, and an assertion that cannot tell code from
    # prose would fail on its own explanation.
    src = chr(10).join(
        ln for ln in inspect.getsource(tr._compute_confidence_ranked).splitlines()
        if not ln.lstrip().startswith("#"))
    body = src.split('"""')[-1]                 # past the docstring
    assert 'get("confidence", 0.0)' not in body, (
        "the .get default cannot cover a present-but-None confidence")


def test_email_bundle_builds_with_a_none_confidence_closed_trade(monkeypatch):
    """End-to-end guard: the crash was inside get_performance_for_email, which
    every tick calls -- so a break there takes down trading, not just reporting."""
    trades = [_trade(0.9, "AAA"), _trade(None, "MECH", ret=-8.5)]
    monkeypatch.setattr(tr, "_load_trades", lambda *a, **k: trades)
    perf = tr.get_performance_for_email()
    assert perf is not None
    assert len(perf.get("closed_trades") or []) == 2


# -- the EMAIL RENDER path (second crash, 2026-08-26 10:58) -------------------
#
# Fixing _compute_confidence_ranked to PRESERVE None (correct for the data) moved
# the crash downstream into the Jinja template: `{{ row.confidence * 100 }}` on
# line 426 raised "unsupported operand type(s) for *: 'NoneType' and 'int'",
# killing every tick again -- from inside the template, so no Python call site
# named it. 18 OPEN positions currently carry confidence=None, so this was not
# an edge case.
#
# The lesson these pin: unit tests on the DATA structure passed while production
# burned, because nothing rendered the actual template. The guard has to run the
# real path.

def test_email_template_renders_with_a_none_confidence_trade():
    """THE second regression, at the layer that broke: render the real template
    through the real entry point with a None-confidence trade in the ledger."""
    from unittest.mock import MagicMock, patch

    from src.notifications import email_sender as es

    perf = {
        "stats": {}, "portfolio_metrics": {},
        "closed_trades": [_trade(0.9, "AAA"), _trade(None, "MECH", ret=-8.5)],
        "open_trades": [_trade(None, "OPENMECH", ret=1.2, status="OPEN")],
        "confidence_ranked": tr._compute_confidence_ranked(
            [_trade(0.9, "AAA"), _trade(None, "MECH", ret=-8.5)]),
    }
    with patch("smtplib.SMTP", MagicMock()):
        es.send_recommendations([], total_analysed=0, performance=perf)


def test_template_guards_the_confidence_multiplication():
    """Asserted at the source too: the template must not multiply confidence
    without a None guard. A renderer error is invisible to every Python-level
    test that does not actually render."""
    import re

    from src.notifications import email_sender as es
    tpl = es.HTML_TEMPLATE
    for m in re.finditer(r"\{\{[^}]*row\.confidence[^}]*\}\}", tpl):
        expr = m.group(0)
        assert "is not none" in expr or "default" in expr, (
            f"unguarded confidence expression in the email template: {expr}")
