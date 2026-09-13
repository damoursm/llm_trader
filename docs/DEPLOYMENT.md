# Production deployment (Windows) — reliable always-on scheduling

Three independent pieces must stay up for unattended trading (what is LIVE on this box as of 2026-09-05 is in the last column):

| Piece | Job | Kept alive by |
|---|---|---|
| **Scheduler** (`main.py --supervise`, via `scripts\run_scheduler.bat`) | runs the pipeline every 30 min, places paper/live orders | Task Scheduler job **`LlmTraderScheduler`** (auto-restart on crash); inside it `--supervise` relaunches the `--schedule` child whenever a watchdog force-exits it |
| **IB Gateway** (headless, paper port 4002) | the broker connection the scheduler talks to | **IBC** under Task Scheduler job **`IBC Gateway`** (`C:\IBC\StartGateway.bat /INLINE`, at log on + daily `AutoRestartTime` 23:50) |
| **Ollama** (`scripts\run_ollama.bat`, loopback `127.0.0.1:11434`) | the LOCAL sentiment engine — primary on half the runs (`SENTIMENT_LOCAL_SHARE=0.5`) and the shadow scorer on the other half | Task Scheduler job **`LlmTraderOllama`** (`scripts\register_ollama_task.ps1`) — **not yet registered here**: the running server was started by hand from `run_ollama.bat`, so a reboot leaves the sentiment A/B silently 100% DeepSeek until it is relaunched |

The scheduler tolerates a down broker (it skips the broker sync and alerts, internal
sim unaffected) and a down model server (a local-primary run falls through to the hosted
engines; only if those are down too does the run record sentiment provider `none`, which
reaches the email banner), so strict start order isn't required — but in steady state all
three run 24/7. **The one supported way to bounce the trader stack is
`scripts\restart_all.ps1`** (stops both `LlmTrader*` tasks, force-kills any straggler
including a surviving `--supervise` parent, restarts the tasks); the model server is
deliberately NOT part of that bounce (it holds a ~5 GB model resident in VRAM).

---

## 1 — Run the scheduler as a supervised service

Always launch via **`scripts\run_scheduler.bat`** (never bare `python main.py --schedule`).
The launcher forces the repo root as the working directory and the **venv** python — the
two things that broke us (wrong CWD → empty `.env`/DB; miniconda → no `ib_async`).

The LIVE setup is Option B via `scripts\register_scheduler_task.ps1` (task `LlmTraderScheduler`).

### Option A — NSSM (a true service, restarts on crash, starts at boot — not what runs here)

1. Download NSSM (`nssm.exe`) from https://nssm.cc/ and put it somewhere on PATH.
2. Install the service (run as admin):
   ```
   nssm install LlmTraderScheduler "C:\Users\mathi\PycharmProjects\llm_trader\scripts\run_scheduler.bat"
   nssm set LlmTraderScheduler AppDirectory "C:\Users\mathi\PycharmProjects\llm_trader"
   nssm set LlmTraderScheduler AppStdout "C:\Users\mathi\PycharmProjects\llm_trader\logs\service_out.log"
   nssm set LlmTraderScheduler AppStderr "C:\Users\mathi\PycharmProjects\llm_trader\logs\service_err.log"
   nssm set LlmTraderScheduler AppExit Default Restart
   nssm set LlmTraderScheduler AppRestartDelay 5000
   nssm set LlmTraderScheduler Start SERVICE_AUTO_START
   nssm start LlmTraderScheduler
   ```
3. NSSM restarts the process automatically if it ever exits, and starts it at boot.
   Manage with `nssm restart|stop|status LlmTraderScheduler`.

### Option B — Task Scheduler (no install)

Create a task that runs `scripts\run_scheduler.bat`:
- **General:** "Run whether user is logged on or not" *(or "only when logged on"
  to avoid storing a password)*; "Run with highest privileges".
- **Triggers:** *At log on* (and/or *At startup*).
- **Actions:** Start a program → `C:\Users\mathi\PycharmProjects\llm_trader\scripts\run_scheduler.bat`
  → **Start in** = `C:\Users\mathi\PycharmProjects\llm_trader`.
- **Settings:** "If the task fails, restart every **1 minute**, up to **3** times";
  "Run task as soon as possible after a scheduled start is missed"; **uncheck**
  "Stop the task if it runs longer than…" (it's a daemon).

### Verify
New lines appear in `logs\llm_trader_<date>.log` at the next :00/:30 tick, and a tick
logs `[broker:ibkr] connected … (clientId=11)` (once the gateway/IBC is up — see §2).
A restart is only done when `Get-CimInstance Win32_Process` shows exactly ONE
`--supervise` and ONE `--schedule` python — two schedulers ticking concurrently race the
DuckDB writer and can double broker submissions.

**Watchdogs, not just restart-on-failure.** A HUNG process never exits, so neither the
supervisor nor Task Scheduler would ever see it; two watchdogs convert hangs into exits
(`broker_sync_watchdog_seconds` 600 around the broker sync, `tick_watchdog_seconds` 2700
around the whole tick) via `os._exit(1)`, which `--supervise` then relaunches.

---

## 2 — Keep the gateway logged in with IBC (auto-login + daily restart)

**Live here:** headless **IB Gateway** on paper port **4002** (`.env`: `BROKER_MODE=ibkr_paper`,
`IBKR_PORT=4002`), IBC config at **`C:\IBC\config.ini`** (`TradingMode=paper`,
`OverrideTwsApiPort=4002`, `ReadOnlyApi=no`, `AcceptIncomingConnectionAction=accept`,
`AutoRestartTime=11:50 PM`, plus `BypassOrderPrecautions=yes` +
`BypassRedirectOrderWarning=yes` — the two API-precaution bypasses the overnight-venue
orders need), launched by the `IBC Gateway` task. The scheduler also has a gateway
auto-recovery (`broker_gateway_auto_restart`, paper-only): on a wedged-but-alive or dead
gateway it kills the java process on the port and fires that task. The TWS walkthrough
below is the original desktop setup and still works — only the port (7497) differs.

IBC ([IbcAlpha/IBC](https://github.com/IbcAlpha/IBC)) logs into TWS for you and, via
**`AutoRestartTime`**, restarts it daily **without re-authenticating** — so it runs the
whole week on one Monday login. This replaces the daily logoff that kept breaking us.

1. **Download IBC** (latest release: https://github.com/IbcAlpha/IBC/releases) and install
   (default `C:\IBC`).
2. **Config:** copy IBC's sample `config.ini` to `%USERPROFILE%\Documents\IBC\config.ini`
   and apply the settings from **`scripts\ibc-config.ini.template`** in this repo — the
   important ones: your **paper** `IbLoginId`/`IbPassword`, `TradingMode=paper`,
   **`AutoRestartTime`**, `OverrideTwsApiPort=7497`, `ReadOnlyApi=no`,
   `AcceptIncomingConnectionAction=accept`.
   Keep this file out of git (it holds your password).
3. **Start script:** edit IBC's `StartTWS.bat` — set `TWS_PATH` (your TWS install, e.g.
   `C:\Jts`), `IBC_PATH` (`C:\IBC`), `CONFIG` (the config.ini path above),
   `TRADING_MODE=paper`, `TWOFA_TIMEOUT_ACTION=restart`.
4. **Launch TWS via `StartTWS.bat`** from now on (not the TWS shortcut). IBC logs in,
   auto-accepts the pipeline's API connection, and holds the session with daily restarts.

### Caveats
- **TWS is a GUI app** — it needs a logged-in Windows desktop session. For a dedicated
  box, enable **Windows auto-logon** so IBC can start TWS at boot. *(If you drop the
  monitoring UI later, switch to headless **IB Gateway** — lighter, same IBC flow —
  and set `IBKR_PORT=4002` in `.env`.)*
- **2FA:** the once-a-day `AutoRestartTime` restart does **not** re-prompt 2FA. Only the
  initial weekly login does — acknowledge it on your phone.
- The IBC API port (`7497`) must match `.env`'s `IBKR_PORT`, and the scheduler's
  `IBKR_CLIENT_ID` (11) must be free (don't run a second client on it).

---

## 3 — Monitoring dashboard (optional, not trading-critical)

The read-only dashboard (`main.py --dashboard`, bound to loopback `127.0.0.1:8050` and
published ONLY through Tailscale Funnel at `https://victushp.tail8e1bf1.ts.net`, HTTP Basic
auth on one shared credential) is kept always-on the same way as the scheduler — a Task
Scheduler job pointed at a venv launcher:

```
powershell -ExecutionPolicy Bypass -File "C:\Users\mathi\PycharmProjects\llm_trader\scripts\register_dashboard_task.ps1"
```

(run once, as administrator). Creates task **`LlmTraderDashboard`** → runs
**`scripts\run_dashboard.bat`** at log on, auto-restarts on failure (1 min × 3), kills any
manually-started dashboard first so port 8050 is free. The dashboard opens the DuckDB
**read-only** with retry/backoff, so it never blocks the scheduler's writes. If it's down,
trading is unaffected — only monitoring is.

Manage: `Start-ScheduledTask` / `Stop-ScheduledTask` / `Unregister-ScheduledTask -TaskName LlmTraderDashboard -Confirm:$false`.

**Restarting the dashboard does not reliably replace it.** Two `--dashboard` processes can
BOTH hold `LISTENING` on 8050 (waitress sets `SO_REUSEADDR` and Windows permits the double
bind) and connections keep going to the OLD one while the new one logs a clean startup.
`Get-NetTCPConnection -LocalPort 8050` hides this; `netstat -ano | Select-String ":8050"`
shows both rows — confirm exactly one PID after every restart, and `Stop-Process` the
survivor before `Start-ScheduledTask`.

---

## Sanity checklist before flipping to `ibkr_live`
- [ ] Scheduler runs as a service and **survives a reboot + a kill** (test both).
- [ ] TWS/IBC survives its `AutoRestartTime` without manual login (watch it cross that time once).
- [ ] A market-hours tick logs `connected … 7497` and `sync — … drift=0`.
- [ ] Paper validation window complete (slippage / tracking-error / reject-rate acceptable).
- [ ] Circuit breakers in place (daily-loss kill switch, etc. — see Tier-1 #4).
