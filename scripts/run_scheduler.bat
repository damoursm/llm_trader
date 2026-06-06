@echo off
REM ============================================================================
REM  Production launcher for the LLM Trader scheduler.
REM
REM  Point your Windows service (NSSM) or Task Scheduler action at THIS file.
REM  It guarantees the two things that broke us in testing:
REM    1. correct working directory  -> .env / data / cache / logs all resolve
REM    2. correct interpreter (venv) -> ib_async (the IBKR library) is present
REM
REM  Do NOT launch `python main.py --schedule` directly in production — the bare
REM  `python` is miniconda (no ib_async) and the CWD is wherever the shell sits.
REM ============================================================================

REM cd to the repo root (this script lives in <root>\scripts\)
cd /d "%~dp0.."

REM Run the scheduler with the project's venv python (absolute via CWD).
".venv\Scripts\python.exe" main.py --schedule

REM If the process exits, %ERRORLEVEL% propagates so the service manager can
REM detect the failure and restart it.
exit /b %ERRORLEVEL%
