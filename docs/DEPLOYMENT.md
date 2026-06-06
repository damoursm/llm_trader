# Production deployment (Windows) — reliable always-on scheduling

Two independent pieces must stay up for unattended trading:

| Piece | Job | Kept alive by |
|---|---|---|
| **Scheduler** (`main.py --schedule`) | runs the pipeline every 30 min, places paper/live orders | a Windows **service** (NSSM) or **Task Scheduler**, auto-restart on crash |
| **TWS / IB Gateway** | the broker connection the scheduler talks to | **IBC** (auto-login + daily `AutoRestartTime`) |

The scheduler tolerates a down broker (it skips the broker sync and alerts, internal
sim unaffected), so strict start order isn't required — but in steady state both run 24/7.

---

## 1 — Run the scheduler as a supervised service

Always launch via **`scripts\run_scheduler.bat`** (never bare `python main.py --schedule`).
The launcher forces the repo root as the working directory and the **venv** python — the
two things that broke us (wrong CWD → empty `.env`/DB; miniconda → no `ib_async`).

### Option A — NSSM (recommended: a true service, restarts on crash, starts at boot)

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
logs `[broker:ibkr] connected … (clientId=11)` (once TWS/IBC is up — see §2).

---

## 2 — Keep TWS logged in with IBC (auto-login + daily restart)

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

## Sanity checklist before flipping to `ibkr_live`
- [ ] Scheduler runs as a service and **survives a reboot + a kill** (test both).
- [ ] TWS/IBC survives its `AutoRestartTime` without manual login (watch it cross that time once).
- [ ] A market-hours tick logs `connected … 7497` and `sync — … drift=0`.
- [ ] Paper validation window complete (slippage / tracking-error / reject-rate acceptable).
- [ ] Circuit breakers in place (daily-loss kill switch, etc. — see Tier-1 #4).
