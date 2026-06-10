@echo off
REM ============================================================================
REM  Production launcher for the LLM Trader monitoring dashboard.
REM
REM  Point Task Scheduler (or NSSM) at THIS file — same contract as
REM  run_scheduler.bat:
REM    1. correct working directory  -> .env / data / cache / logs all resolve
REM    2. correct interpreter (venv) -> dash / waitress are present
REM
REM  Do NOT launch `python main.py --dashboard` directly in production — the
REM  bare `python` is miniconda and the CWD is wherever the shell sits.
REM ============================================================================

REM cd to the repo root (this script lives in <root>\scripts\)
cd /d "%~dp0.."

REM Run the dashboard with the project's venv python (absolute via CWD).
".venv\Scripts\python.exe" main.py --dashboard

REM If the process exits, %ERRORLEVEL% propagates so the service manager can
REM detect the failure and restart it.
exit /b %ERRORLEVEL%
