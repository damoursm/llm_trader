"""RESEARCH RUNS AT IDLE PRIORITY — at the interpreter's start, not by remembering (2026-10-21; the user's "Fix all"
after the audit of 2026-10-20: the low-priority call had to be written into every research script and one-liner and
was forgotten — the untyped call of 2026-10-19 that left research at NORMAL beside the 09:30 tick, three study scripts
of that evening without it).

A one-line ``.pth`` file in an interpreter's site-packages (``--install``) runs THIS file at the start of every Python
process of that interpreter — the project's venv and miniconda (the bare ``python`` of the shells). `classify` decides
from the process's command line and working directory alone whether it is RESEARCH, and only then `apply` lowers it to
IDLE (0x40, the typed call on a private kernel32 handle: the untyped call silently fails, and typed prototypes on the
shared ``ctypes.windll.kernel32`` would change other code's calls). Research is identified POSITIVELY, never by
elimination, so production is never lowered:

* a script under the research tree (``llm_trader_research/``) or a Claude scratchpad (``%TEMP%/claude/``);
* ``python -c CODE`` or ``python -`` (a program on stdin) with the working directory in either project or a
  scratchpad — production never runs code that way (``tests/test_research_idle.py`` pins it: its subprocesses are
  ``main.py --schedule``, ``-m src.signals.sel_short``, ``-m src.scheduler.eod``, ``-m src.data.deep.refresh``,
  ``-m src.performance.spread_sweep``, ``-m src.analysis.deep_features``, ``-m dashboard.warm_worker``) — except
  multiprocessing's workers (``--multiprocessing-fork``), left alone: Windows gives the child of an IDLE or
  BELOW_NORMAL process its parent's class, so a research worker is IDLE already and production's keep theirs;
* ``-m pytest`` (the test suite) and the research command lines of the project (``-m src.backtest.*`` — the card
  builds, the bank, the registration checker, the calibration rerun — and ``-m src.analysis.ml30``) with the working
  directory in either project or a scratchpad.

Fail-safe: any error leaves the priority as it was; silent unless ``RESEARCH_IDLE_VERBOSE`` is set.

    python scripts/research_idle.py --install [--python EXE ...]   the .pth into each interpreter's site-packages
    python scripts/research_idle.py --check   [--python EXE ...]   each .pth, and a research probe's priority class
    python scripts/research_idle.py --uninstall [--python EXE ...]
"""
import os
import sys

IDLE = 0x40
PTH_NAME = "llm_trader_research_idle.pth"
RESEARCH_MODULES = ("pytest", "src.analysis.ml30")         # plus every module under src.backtest.
RESEARCH_MODULE_PREFIXES = ("src.backtest.",)
_HERE = os.path.abspath(globals().get("__file__") or sys.argv[0])
PROJECT = os.path.dirname(os.path.dirname(_HERE))                     # llm_trader/
RESEARCH = os.path.join(os.path.dirname(PROJECT), "llm_trader_research")


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def _scratch_root():
    return os.path.join(os.environ.get("TEMP") or os.environ.get("TMP") or "", "claude")


def _under(path, roots):
    p = _norm(path)
    for r in roots:
        if not r:
            continue
        r = _norm(r)
        if p == r or p.startswith(r.rstrip(os.sep) + os.sep):
            return True
    return False


def _invocation(argv):
    """(kind, value, rest) of a Python command line (its argv after the interpreter): kind "script" (value = the
    path), "inline" (``-c``: the code), "stdin" (``-``), "module" (``-m``: the module) or None (an interactive
    interpreter). The interpreter's own options are skipped: ``-X`` / ``-W`` with their argument, combined flags."""
    args = list(argv[1:])
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-X", "-W", "--check-hash-based-pycs"):
            i += 2
            continue
        if a == "-":
            return "stdin", "", args[i + 1:]
        if a.startswith("-c"):
            code = a[2:] if len(a) > 2 else (args[i + 1] if i + 1 < len(args) else "")
            return "inline", code, args[i + (1 if len(a) > 2 else 2):]
        if a.startswith("-m"):
            mod = a[2:] if len(a) > 2 else (args[i + 1] if i + 1 < len(args) else "")
            return "module", mod, args[i + (1 if len(a) > 2 else 2):]
        if a.startswith("-") and not a.startswith("--"):
            i += 1                                             # -u, -B, -I, -E, -O, -s, -S, -q, combined "-uB"
            continue
        if a.startswith("--"):
            i += 1
            continue
        return "script", a, args[i + 1:]
    return None, "", []


def classify(argv, cwd, research=None, project=None, scratch=None):
    """Why the process with command line ``argv`` (``sys.orig_argv``: the interpreter first) started in ``cwd`` is
    RESEARCH — "script", "inline" or "module" — or None (production, multiprocessing's workers, anything else)."""
    research = RESEARCH if research is None else research
    project = PROJECT if project is None else project
    scratch = _scratch_root() if scratch is None else scratch
    if "--multiprocessing-fork" in argv:
        return None
    kind, value, _rest = _invocation(argv)
    home = _under(cwd, (project, research, scratch))
    if kind == "script":
        path = value if os.path.isabs(value) else os.path.join(cwd, value)
        if _under(path, (research, scratch)):
            return "script"
        base = os.path.basename(value).lower()
        if home and base in ("pytest", "pytest.exe", "py.test", "py.test.exe"):
            return "module"
        return None
    if kind in ("inline", "stdin"):
        if "multiprocessing" in value:
            return None
        return "inline" if home else None
    if kind == "module":
        mod = value.strip()
        if home and (mod in RESEARCH_MODULES or mod.startswith(RESEARCH_MODULE_PREFIXES)):
            return "module"
    return None


def _kernel():
    """A PRIVATE kernel32 handle with typed prototypes (never ``ctypes.windll.kernel32``'s shared function objects)."""
    import ctypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.GetCurrentProcess.argtypes = []
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.SetPriorityClass.restype = ctypes.c_int
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.GetPriorityClass.restype = ctypes.c_uint32
    return k


def apply(argv=None, cwd=None, kernel=None):
    """Lower this process to IDLE when `classify` calls it research. Returns the reason, or None (nothing done).
    Never raises."""
    try:
        argv = list(argv if argv is not None else (getattr(sys, "orig_argv", None) or sys.argv))
        cwd = cwd if cwd is not None else os.getcwd()
        why = classify(argv, cwd)
        if why is None or (kernel is None and os.name != "nt"):
            return None
        k = kernel if kernel is not None else _kernel()
        ok = k.SetPriorityClass(k.GetCurrentProcess(), IDLE)
        if os.environ.get("RESEARCH_IDLE_VERBOSE"):
            sys.stderr.write(f"[research_idle] {why}: IDLE {'set' if ok else 'FAILED'}\n")
        return why if ok else None
    except BaseException:                                      # noqa: BLE001 — a startup hook never breaks a process
        return None


# ── install / check ─────────────────────────────────────────────────────────

def pth_line(hook=None):
    """The .pth's one line: run this file at the interpreter's start (absent file: nothing)."""
    hook = hook or _HERE
    return ("import os; _p = r'%s'; os.path.exists(_p) and exec(compile(open(_p, encoding='utf-8').read(), _p, "
            "'exec'), {'__name__': 'research_idle_hook', '__file__': _p}); del _p\n" % hook)


def _default_pythons():
    out = [sys.executable]
    conda = os.path.join(os.path.expanduser("~"), "miniconda3", "python.exe")
    if os.path.exists(conda) and _norm(conda) != _norm(sys.executable):
        out.append(conda)
    return out


def _site_packages(python):
    import subprocess
    r = subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                       capture_output=True, text=True, timeout=60, cwd=os.path.expanduser("~"))
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"{python}: no site-packages ({r.stderr.strip()[:200]})")
    return r.stdout.strip().splitlines()[-1]


def _probe(python):
    """The priority class a research one-liner of ``python`` runs at (started in the research tree)."""
    import subprocess
    code = ("import ctypes; k = ctypes.WinDLL('kernel32'); k.GetCurrentProcess.restype = ctypes.c_void_p; "
            "k.GetPriorityClass.argtypes = [ctypes.c_void_p]; print(hex(k.GetPriorityClass(k.GetCurrentProcess())))")
    cwd = RESEARCH if os.path.isdir(RESEARCH) else PROJECT
    r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60, cwd=cwd)
    return r.stdout.strip() or r.stderr.strip()[:200]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="research processes at IDLE priority, from the interpreter's start")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--python", action="append", help="an interpreter (default: this one and miniconda's)")
    a = ap.parse_args(argv)
    pythons = a.python or _default_pythons()
    bad = 0
    for py in pythons:
        sp = _site_packages(py)
        pth = os.path.join(sp, PTH_NAME)
        if a.install:
            with open(pth, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(pth_line())
            print(f"installed {pth}")
        elif a.uninstall:
            if os.path.exists(pth):
                os.remove(pth)
            print(f"removed {pth}")
        line = open(pth, encoding="utf-8").read() if os.path.exists(pth) else None
        if a.check or a.install:
            ok_line = line == pth_line()
            cls = _probe(py) if os.name == "nt" else "n/a"
            ok = ok_line and cls == hex(IDLE)
            bad += 0 if ok else 1
            print(f"{'OK ' if ok else 'BAD'} {py}: .pth {'current' if ok_line else ('stale' if line else 'missing')}, "
                  f"a research one-liner runs at {cls} (IDLE = {hex(IDLE)})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
elif __name__ == "research_idle_hook":
    apply()
