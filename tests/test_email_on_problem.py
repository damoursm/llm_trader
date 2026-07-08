"""email_on_problem — a detected problem forces the report email out of schedule.

Unit-tests the pure email gate ``pipeline._email_decision``. The run always sends
on a scheduled slot / manual run; the new guarantee is that a broker / LLM / price
health problem forces a send even on a non-email tick, so an issue between the
04:00/09:30/16:00/19:50 slots reaches the user at the next ~30-min tick.
"""

from src.pipeline import _email_decision


def _d(**kw):
    # default: scheduler NON-email slot with email configured (the user's real setup)
    base = dict(observe_only=False, send_email=False, email_if_configured=False,
                email_configured=True, health_problem=False)
    base.update(kw)
    return _email_decision(**base)


def test_non_email_tick_suppresses_without_problem():
    assert _d() == "suppress"


def test_problem_forces_send_on_non_email_tick():
    # THE new behaviour: no scheduled email this slot, but a problem forces it out.
    assert _d(health_problem=True) == "send"


def test_scheduled_email_slot_sends():
    assert _d(send_email=True) == "send"


def test_manual_run_sends():
    assert _d(email_if_configured=True) == "send"


def test_observe_only_never_emails_even_on_problem():
    # observe ticks are analysis-only by design — a problem does not force a send.
    assert _d(observe_only=True, health_problem=True) == "observe"


def test_manual_run_but_smtp_unconfigured_skips():
    # wants email (manual) but SMTP not set up -> the 'skip' / not-configured branch.
    assert _d(email_if_configured=True, email_configured=False) == "skip"
