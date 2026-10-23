"""Research runs at IDLE priority from the interpreter's start (`scripts/research_idle.py`, 2026-10-21).

What must hold: production's command lines are NEVER classified as research (they would run at IDLE beside the
market — the one failure that matters); research's are (scripts in the research tree or a scratchpad, inline code and
stdin programs started in either project, the test suite, the research CLIs); production never launches inline Python
(the classification's premise); the hook lowers only through the typed call on a private kernel32 handle; and the
hook is installed for the interpreter running these tests (a rebuilt venv loses it silently otherwise).
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("research_idle", ROOT / "scripts" / "research_idle.py")
RI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RI)

PY = r"C:\Users\x\PycharmProjects\llm_trader\.venv\Scripts\python.exe"
PROJECT = r"C:\Users\x\PycharmProjects\llm_trader"
RESEARCH = r"C:\Users\x\PycharmProjects\llm_trader_research"
SCRATCH = r"C:\Users\x\AppData\Local\Temp\claude"


def _cls(argv, cwd=PROJECT):
    return RI.classify([PY] + list(argv), cwd, research=RESEARCH, project=PROJECT, scratch=SCRATCH)


@pytest.mark.parametrize("argv", [
    ["main.py", "--supervise"], ["main.py", "--schedule"], ["main.py", "--dashboard"], ["main.py", "--email"],
    ["-m", "src.signals.sel_short", "--run", "--day", "2026-10-21", "--bars", "0,1"],
    ["-m", "src.signals.sel_short", "--prepare", "--day", "2026-10-21"],
    ["-m", "src.scheduler.eod", "--day", "2026-10-21"],
    ["-m", "src.accounts.run", "--account", "second"],          # the second account's tick (2026-10-23)
    ["-m", "src.data.deep.refresh", "--profile", "nightly", "--budget-seconds", "14400"],
    ["-m", "src.performance.spread_sweep"],
    ["-m", "src.analysis.deep_features", "--snapshot", "2026-10-21"],
    ["-m", "dashboard.warm_worker", r"C:\tmp\x.json"],
    ["-c", "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=1, pipe_handle=2)",
     "--multiprocessing-fork"],
    ["-B", "-c", "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=1, pipe_handle=2)",
     "--multiprocessing-fork"],
])
def test_production_command_lines_are_never_research(argv):
    assert _cls(argv) is None


@pytest.mark.parametrize("argv,cwd,why", [
    ([r"..\llm_trader_research\studies\prereg169\account169_l2.py", "work"], PROJECT, "script"),
    (["account169_l2.py", "verdict"], RESEARCH + r"\studies\prereg169", "script"),
    ([RESEARCH + r"\studies\prereg171\lockup171.py"], r"C:\Windows", "script"),
    (["-u", r"..\llm_trader_research\studies\x.py"], PROJECT, "script"),
    ([SCRATCH + r"\abc\scratchpad\idle_pytest.py"], PROJECT, "script"),
    (["-c", "import pandas; print(1)"], PROJECT, "inline"),
    (["-X", "utf8", "-c", "print(1)"], RESEARCH, "inline"),
    (["-"], PROJECT, "inline"),
    (["-m", "pytest", "tests/test_bank.py", "--tb=line"], PROJECT, "module"),
    (["-m", "src.backtest.reliability", "--build"], PROJECT, "module"),
    (["-m", "src.backtest.bank"], PROJECT, "module"),
    (["-m", "src.analysis.ml30", "--build"], PROJECT, "module"),
    ([PROJECT + r"\.venv\Scripts\pytest.exe", "tests"], PROJECT, "module"),
])
def test_research_command_lines_are_research(argv, cwd, why):
    assert _cls(argv, cwd) == why


def test_outside_both_projects_inline_code_is_left_alone():
    assert _cls(["-c", "print(1)"], r"C:\Users\x\other_project") is None
    assert _cls(["-m", "pytest"], r"C:\Users\x\other_project") is None
    assert _cls(["other.py"], r"C:\Users\x\other_project") is None
    assert _cls([], PROJECT) is None                                     # an interactive interpreter


def test_production_never_launches_inline_python():
    """The premise of the inline rule: no production subprocess runs ``python -c`` / ``python -``."""
    bad = []
    for p in list((ROOT / "src").rglob("*.py")) + [ROOT / "main.py"] + list((ROOT / "dashboard").rglob("*.py")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"sys\.executable\s*,\s*(['\"])(-c|-)\1", txt):
            bad.append(f"{p.relative_to(ROOT)}: {m.group(0)}")
    assert bad == []


def test_apply_lowers_only_a_research_process_through_the_typed_call():
    calls = []

    class K:
        def GetCurrentProcess(self):
            return -1

        def SetPriorityClass(self, h, cls):
            calls.append((h, cls))
            return 1
    assert RI.apply([PY, "main.py", "--schedule"], str(ROOT), kernel=K()) is None and calls == []
    assert RI.apply([PY, "-c", "print(1)"], str(ROOT), kernel=K()) == "inline"
    assert calls == [(-1, 0x40)]

    class Broken(K):
        def SetPriorityClass(self, h, cls):
            raise OSError("no")
    assert RI.apply([PY, "-c", "print(1)"], str(ROOT), kernel=Broken()) is None    # never raises


def test_the_pth_line_runs_this_file_as_the_hook():
    line = RI.pth_line(r"C:\x\research_idle.py")
    assert line.startswith("import os; ") and line.endswith("\n") and line.count("\n") == 1
    assert "'research_idle_hook'" in line and r"C:\x\research_idle.py" in line


@pytest.mark.skipif(os.name != "nt", reason="the hook acts on Windows priority classes")
def test_the_hook_is_installed_for_this_interpreter():
    pth = Path(sysconfig.get_paths()["purelib"]) / RI.PTH_NAME
    assert pth.exists(), f"run: {sys.executable} scripts/research_idle.py --install"
    assert pth.read_text(encoding="utf-8") == RI.pth_line(), "stale: rerun scripts/research_idle.py --install"
