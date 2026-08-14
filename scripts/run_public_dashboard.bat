@echo off
REM ============================================================================
REM  Launcher for the PUBLIC ngrok tunnel in front of the monitoring dashboard.
REM
REM  Same contract as run_dashboard.bat: cd to the repo root first so .env
REM  (which holds NGROK_DOMAIN / DASHBOARD_PORT) resolves.
REM
REM  This publishes the dashboard at https://<NGROK_DOMAIN> with NO login.
REM  -Force replaces a stale agent: the ngrok free plan allows only one at a
REM  time, and an orphan from a previous session would otherwise block startup.
REM ============================================================================

cd /d "%~dp0.."

powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0run_public_dashboard.ps1" -Force

REM Propagate the exit code so Task Scheduler can restart a dead tunnel.
exit /b %ERRORLEVEL%
