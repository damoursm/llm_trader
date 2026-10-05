"""The scheduler's suspend/resume detector measures the SLEEP between polls,
never a tick's own runtime (2026-09-29: "wall clock jumped N min
(suspend/resume)" fired after every tick longer than 5 minutes — 27 false
alarms in a day — because ``prev_poll`` was stamped BEFORE the tick ran)."""
from __future__ import annotations

import ast
import inspect
import textwrap


def _loop_body():
    from src.scheduler import runner
    tree = ast.parse(textwrap.dedent(inspect.getsource(runner.start_scheduler)))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)
             and isinstance(n.test, ast.Constant) and n.test.value is True]
    assert loops, "the poll loop (while True) was not found"
    return loops[0].body


def _assigns_prev_poll(stmt) -> bool:
    return any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prev_poll"
                                                  for t in n.targets)
               for n in ast.walk(stmt))


def test_prev_poll_is_stamped_after_the_tick_right_before_the_sleep():
    body = _loop_body()
    stamps = [i for i, s in enumerate(body) if _assigns_prev_poll(s)]
    assert len(stamps) == 1, f"prev_poll must be stamped exactly once per pass, found at {stamps}"
    i = stamps[0]
    last = body[-1]
    assert i == len(body) - 2, "the stamp must sit right before the loop's sleep"
    assert "sleep" in ast.unparse(last), "the loop must end with its sleep"
    # the stamp is a FRESH clock read, never the pass's start time
    assert "now_et" in ast.unparse(body[i])
