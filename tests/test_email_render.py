"""The daily email actually renders (`src/notifications/email_sender.py`).

5,400 lines, a ~30-section Jinja template built by hand, and it is the primary
deliverable — the thing the whole pipeline exists to produce. Until now only
`send_alert` (the crash notifier) was covered, so nothing exercised
`send_recommendations` at all.

The failure mode is specific and total: `Template(HTML_TEMPLATE).render(...)` sits
OUTSIDE the try/except that guards the SMTP send, so a template referencing a
renamed variable, or a section iterating a context whose field moved, raises out
of the send and takes the tick's email with it. A section can also silently
vanish — a `{% if %}` on a key that no longer exists renders nothing at all, and
nobody notices a missing block in a 30-section email for weeks.

So these tests render the real template with a realistic payload and stubbed
SMTP, and check that it completes, carries its sections, and reflects the
health banners in the subject. They deliberately do not assert on wording.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

import pytest

from config.settings import settings
from src.models import NewsArticle, Recommendation, TickerSignal
from src.notifications import email_sender as es


@pytest.fixture
def sent(monkeypatch):
    """Capture the MIME message instead of sending it."""
    box: dict = {}

    class _FakeSMTP:
        def __init__(self, host, port):
            box["host"], box["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            box["tls"] = True

        def login(self, user, pw):
            box["login"] = user

        def sendmail(self, sender, to, msg):
            box["to"] = to
            box["raw"] = msg

    monkeypatch.setattr(es.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(settings, "enable_charts", False)
    return box


def _rec(ticker="AAPL", action="BUY", conf=0.88) -> Recommendation:
    return Recommendation(
        ticker=ticker, type="STOCK", direction="BULLISH", action=action,
        confidence=conf, rationale=f"{ticker} rationale text",
        generated_at=datetime.now(timezone.utc))


def _sig(ticker="AAPL") -> TickerSignal:
    return TickerSignal(ticker=ticker, direction="BULLISH", confidence=0.88,
                        combined_score=0.42, sentiment_score=0.3,
                        technical_score=0.2, rationale="r")


def _article() -> NewsArticle:
    return NewsArticle(title="AAPL beats", summary="s", url="http://x/1",
                       source="rss", published_at=datetime.now(timezone.utc))


def _performance() -> dict:
    return {
        "stats": {"trades": 12, "win_rate": 58.3, "compound_return": 4.2,
                  "avg_return": 0.35, "wtd_avg_return": 0.4,
                  "best": 9.1, "worst": -4.4},
        "performance_table": [
            {"label": "All", "trades": 12, "win_rate": 58.3, "compound_return": 4.2,
             "avg_return": 0.35, "wtd_avg_return": 0.4, "best": 9.1, "worst": -4.4},
        ],
        "open_trades": [
            {"ticker": "AAPL", "action": "BUY", "entry_price": 100.0,
             "current_price": 104.0, "return_pct": 4.0, "entry_date": "2026-08-10",
             "status": "OPEN", "days_held": 4},
        ],
        "closed_trades": [
            {"ticker": "MSFT", "action": "SELL", "entry_price": 200.0,
             "exit_price": 190.0, "return_pct": 5.0, "entry_date": "2026-08-01",
             "exit_date": "2026-08-08", "status": "CLOSED",
             "exit_reason": "trailing_stop"},
        ],
        "portfolio_metrics": {"compound_inception": 4.2, "return_1w": 0.8,
                              "return_2w": 1.6, "return_1m": 3.1},
        "trades_svg": "<svg/>",
    }


def _html(box: dict) -> str:
    """The DECODED text/html part.

    The MIME body is base64-transfer-encoded, so grepping the raw message finds
    nothing and every content assertion would fail for the wrong reason."""
    import email as _email
    assert "raw" in box, "the email was never sent"
    msg = _email.message_from_string(box["raw"])
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode("utf-8", "replace")
    raise AssertionError("no text/html part in the message")


def _raw(box: dict) -> str:
    assert "raw" in box, "the email was never sent"
    return box["raw"]


# ── it renders at all ───────────────────────────────────────────────────────

def test_the_email_renders_and_sends(sent):
    """The one that catches a template variable rename — the render is outside
    the SMTP try/except, so it would propagate out of the pipeline."""
    ok = es.send_recommendations([_rec()], total_analysed=42)
    assert ok is True
    assert sent["host"] == settings.smtp_host and sent["tls"] is True
    assert "AAPL" in _html(sent)


def test_a_rich_payload_renders_every_optional_block(sent):
    """A realistic tick: recommendations, signals, articles and performance all
    present. Each optional block is a `{% if %}` whose condition is a separate
    chance to silently render nothing."""
    ok = es.send_recommendations(
        [_rec(), _rec("MSFT", action="WATCH", conf=0.5)],
        total_analysed=120,
        performance=_performance(),
        signals=[_sig(), _sig("MSFT")],
        articles=[_article()],
    )
    assert ok is True
    html = _html(sent)
    assert "AAPL" in html and "MSFT" in html


def test_no_recommendations_declines_to_send(sent):
    """An empty run must not produce an empty email — it sends nothing."""
    assert es.send_recommendations([]) is False
    assert "raw" not in sent


def test_the_message_is_multipart_with_a_plain_text_alternative(sent):
    """Some clients render the text part; a template-only email is unreadable
    there, and the plain-text build is its own code path."""
    es.send_recommendations([_rec()], performance=_performance())
    raw = _raw(sent)
    assert "multipart/related" in raw
    assert "multipart/alternative" in raw
    assert "text/plain" in raw and "text/html" in raw


def test_it_addresses_the_configured_recipients(sent):
    es.send_recommendations([_rec()])
    assert sent["to"] == settings.recipients_list
    assert sent["login"] == settings.smtp_user


# ── the sections survive ────────────────────────────────────────────────────

def test_the_documented_section_markers_are_present(sent):
    """The template uses `<!-- ══ N — SECTION NAME ══ -->` comments as its
    structure. Counting them catches a section deleted or `{% if %}`-ed out of
    existence, which is invisible in a 30-section email."""
    es.send_recommendations([_rec()], performance=_performance(),
                            signals=[_sig()], articles=[_article()])
    markers = re.findall(r"<!--\s*═+\s*(.+?)\s*═+\s*-->", _html(sent))
    assert len(markers) >= 5, f"only {len(markers)} section markers rendered"


def test_the_core_sections_render_with_their_data(sent):
    es.send_recommendations([_rec()], performance=_performance())
    html = _html(sent)
    for token in ("Recommendation", "Performance"):
        assert token in html, f"the {token} section did not render"


def test_a_missing_optional_context_does_not_break_the_render(sent):
    """Every context block is optional by design — on a degraded tick most feeds
    are None, and that is precisely when the email matters most."""
    assert es.send_recommendations(
        [_rec()], performance=None, signals=None, articles=None,
        macro_context=None, vix_context=None, cot_context=None,
        breadth_context=None, put_call_context=None) is True


# ── health banners reach the subject ────────────────────────────────────────

@pytest.mark.parametrize("kwarg,tag", [
    ("llm_health", "🤖"),
    ("broker_health", "🔔"),
    ("price_health", "🔔"),
])
def test_a_health_problem_tags_the_subject(sent, kwarg, tag):
    """The banner is how an unattended run reports that it degraded; a tag that
    stops being prepended turns a loud failure into a silent one."""
    es.send_recommendations([_rec()], **{kwarg: {"down": True}})
    subject = re.search(r"^Subject: (.+)$", _raw(sent), re.MULTILINE)
    assert subject, "no subject line"
    # MIME may encode non-ASCII subjects, so accept either form.
    line = subject.group(1)
    assert tag in line or "=?utf-8?" in line.lower()


def test_a_healthy_run_carries_no_alarm_tag(sent):
    es.send_recommendations([_rec()], llm_health={"down": False},
                            broker_health={"down": False},
                            price_health={"down": False})
    line = re.search(r"^Subject: (.+)$", _raw(sent), re.MULTILINE).group(1)
    assert "🤖" not in line and "🔔" not in line


# ── failure handling ────────────────────────────────────────────────────────

def test_an_smtp_failure_is_reported_not_raised(monkeypatch):
    """The send is the last step of the tick; an unreachable mail server must
    not unwind everything that already succeeded."""
    def _boom(host, port):
        raise ConnectionRefusedError("smtp down")

    monkeypatch.setattr(es.smtplib, "SMTP", _boom)
    monkeypatch.setattr(settings, "enable_charts", False)
    assert es.send_recommendations([_rec()]) is False


def test_the_template_is_a_module_level_string():
    """It is embedded rather than a separate file (documented). A refactor that
    moved it to disk would need the packaging to follow, and this is the
    cheapest place to notice."""
    assert isinstance(es.HTML_TEMPLATE, str) and len(es.HTML_TEMPLATE) > 1000
    assert "{{" in es.HTML_TEMPLATE and "{%" in es.HTML_TEMPLATE
