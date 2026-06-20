"""Tests for the scheduler auto-restart supervisor (src.scheduler.supervisor)."""

import src.scheduler.supervisor as sup


def test_supervisor_respawns_on_exit_then_stops_clean(monkeypatch):
    """A child that exits is relaunched; KeyboardInterrupt stops the child and returns."""
    launches = {"n": 0}

    class _FakeChild:
        def __init__(self):
            self.terminated = False

        def wait(self, timeout=None):
            if timeout is not None:        # the cleanup _terminate_child() path
                return 0
            launches["n"] += 1
            if launches["n"] >= 3:         # stop after 2 respawns
                raise KeyboardInterrupt
            return 1                        # quick non-zero exit (crash)

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    monkeypatch.setattr(sup.subprocess, "Popen", lambda *a, **k: _FakeChild())
    monkeypatch.setattr(sup.time, "sleep", lambda *_: None)   # skip real backoff waits

    sup.run_supervised()                   # returns cleanly on KeyboardInterrupt
    assert launches["n"] == 3              # launched 3× (2 respawns + the interrupted one)


def test_supervisor_survives_spawn_failure(monkeypatch):
    """A subprocess spawn/wait error must NOT kill the supervisor (the silent-death
    bug) — it logs, backs off, and retries. A KeyboardInterrupt then stops it clean."""
    attempts = {"n": 0}

    def _popen(cmd, **kwargs):
        attempts["n"] += 1
        if attempts["n"] >= 3:
            raise KeyboardInterrupt        # stop the test after 2 failed spawns
        raise OSError("transient spawn failure")

    monkeypatch.setattr(sup.subprocess, "Popen", _popen)
    monkeypatch.setattr(sup.time, "sleep", lambda *_: None)

    sup.run_supervised()                   # must return cleanly, NOT propagate OSError
    assert attempts["n"] == 3              # retried after each OSError until interrupted


def test_supervisor_launches_the_schedule_child(monkeypatch):
    """The supervised child is exactly `python main.py --schedule`, run from root."""
    seen = {}

    class _FakeChild:
        def wait(self, timeout=None):
            if timeout is not None:        # cleanup path — don't interrupt
                return 0
            raise KeyboardInterrupt        # stop after the first launch
        def poll(self):
            return 0
        def terminate(self):
            pass
        def kill(self):
            pass

    def _popen(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return _FakeChild()

    monkeypatch.setattr(sup.subprocess, "Popen", _popen)
    monkeypatch.setattr(sup.time, "sleep", lambda *_: None)

    sup.run_supervised()
    assert seen["cmd"][1:] == ["main.py", "--schedule"]
    assert seen["cwd"] == str(sup._ROOT)
