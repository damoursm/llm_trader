"""A crashing scheduled tick must ALERT the operator, not just log (2026-07-14).

Silent-failure class: `run_pipeline` raising was only `logger.exception`'d
(console + file, no email sink), so a persistently crashing pipeline — including
a report-email TEMPLATE bug, since the email send is the last pipeline step —
could run a whole session dark with no notification. `_alert_crash` closes it
(throttled per exception type). Plus the blank-error guard for bare TimeoutError.
"""

import src.scheduler.runner as runner


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr(runner.settings, "scheduler_alert_email", True)
    monkeypatch.setattr(runner, "_alert",
                        lambda subject, body: sent.append((subject, body)))
    runner._last_crash_alert.clear()
    return sent


def test_crash_alert_fires_with_type_and_traceback(monkeypatch):
    sent = _capture(monkeypatch)
    try:
        raise ValueError("boom in step 4")
    except ValueError as e:
        runner._alert_crash("pipeline tick", e)
    assert len(sent) == 1
    subject, body = sent[0]
    assert "pipeline tick crashed" in subject
    assert "ValueError" in subject and "boom in step 4" in subject
    assert "Traceback" in body                     # full traceback included
    assert "positions may be unmarked" in body     # operator-actionable context


def test_blank_message_exception_shows_type(monkeypatch):
    """A bare TimeoutError() has str()=='' — the alert must not be blank."""
    sent = _capture(monkeypatch)
    try:
        raise TimeoutError()
    except TimeoutError as e:
        runner._alert_crash("pipeline tick", e)
    subject, _ = sent[0]
    assert "TimeoutError" in subject
    assert subject.rstrip().endswith("TimeoutError")   # no trailing blank ": "


def test_same_type_is_throttled_but_new_type_passes(monkeypatch):
    sent = _capture(monkeypatch)
    for _ in range(3):
        try:
            raise ValueError("x")
        except ValueError as e:
            runner._alert_crash("pipeline tick", e)
    assert len(sent) == 1                          # 2nd/3rd ValueError throttled
    try:
        raise KeyError("y")
    except KeyError as e:
        runner._alert_crash("pipeline tick", e)
    assert len(sent) == 2                          # a DIFFERENT crash type still alerts


def test_alert_gated_off_when_disabled(monkeypatch):
    sent = []
    monkeypatch.setattr(runner.settings, "scheduler_alert_email", False)
    monkeypatch.setattr(runner, "_alert", runner._alert)   # real _alert (returns early)
    runner._last_crash_alert.clear()
    # _alert early-returns when scheduler_alert_email is off — no send attempted.
    called = []
    monkeypatch.setattr("src.notifications.email_sender.send_alert",
                        lambda s, b: called.append(1))
    try:
        raise ValueError("z")
    except ValueError as e:
        runner._alert_crash("pipeline tick", e)
    assert called == []
