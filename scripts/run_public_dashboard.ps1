<#
  Publish the monitoring dashboard on a PUBLIC HTTPS URL via an ngrok tunnel.

    powershell -ExecutionPolicy Bypass -File "<repo>\scripts\run_public_dashboard.ps1"

  How this works: ngrok dials OUT from this PC to ngrok's edge and forwards the
  public hostname to 127.0.0.1:<dashboard_port>. Nothing is port-forwarded, no
  inbound firewall rule is needed, and your home IP is never exposed. The tunnel
  only lives as long as this process.

  The public URL is gated by HTTP Basic-Auth with ONE SHARED PASSWORD, handed out
  together with the link. Anyone holding both sees live positions, P&L, IBKR
  account NAV and fill prices; anyone holding neither gets a 401. The dashboard
  opens the database READ-ONLY, so even an authenticated visitor can only read.

  This script REFUSES to publish unless it has verified the gate by probing the
  running dashboard (anonymous request must return 401). Config alone cannot
  distinguish "gated" from "wide open" - see the preflight below.

  One-time setup (see CLAUDE.md for the walkthrough):
    1. Sign up free at https://dashboard.ngrok.com/signup
    2. Put these in .env (gitignored - they are credentials):
         NGROK_DOMAIN=your-name.ngrok-free.dev
         NGROK_TOKEN=<authtoken from dashboard.ngrok.com/get-started/your-authtoken>
         DASHBOARD_AUTH_PASSWORD=<the shared password you hand out>
         DASHBOARD_AUTH_USERNAME=viewer          # optional, this is the default
       Changing the password requires RESTARTING the dashboard - settings are
       read at import, so a running process keeps serving the old one.
       The token is exported as NGROK_AUTHTOKEN for the child process only, so
       no global `ngrok config add-authtoken` and no ngrok.yml is required.
       A pre-existing ngrok.yml authtoken is honoured too, if you prefer that.
    3. The free static domain is claimed at https://dashboard.ngrok.com/domains

  Params override .env:  -Domain <host>  -Port <n>  -Force (kill a running agent)

  NOTE: keep this file pure ASCII. PowerShell 5.1 reads a BOM-less file as
  cp1252, where a UTF-8 dash decodes into a smart quote - which the parser
  treats as a string delimiter and the whole script fails to parse.

  NOTE: ngrok enforces a MINIMUM AGENT VERSION per account (>=3.20.0 here, error
  ERR_NGROK_121 at connect). The winget package installs 3.3.1, which is too old;
  `ngrok update` self-updated the binary in place to 3.39.11. winget still has it
  recorded as 3.3.1, so a future `winget upgrade` can DOWNGRADE the exe back below
  the minimum and the tunnel starts failing to authenticate. Fix is `ngrok update`.
#>
param(
    [string]$Domain,
    [int]$Port,
    [switch]$Force
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root

# --- .env (the single source of config, same as the Python side) -------------
$envFile = Join-Path $root ".env"
$cfg = @{}
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#") -or ($t -notmatch "=")) { continue }
        $kv = $t -split "=", 2
        $cfg[$kv[0].Trim()] = $kv[1].Trim().Trim('"').Trim("'")
    }
}

if (-not $Domain) { $Domain = $env:NGROK_DOMAIN }
if (-not $Domain) { $Domain = $cfg["NGROK_DOMAIN"] }
if (-not $Port)   { if ($cfg["DASHBOARD_PORT"]) { $Port = [int]$cfg["DASHBOARD_PORT"] } else { $Port = 8050 } }

if (-not $Domain) {
    Write-Host "No NGROK_DOMAIN set." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Claim your free static domain at https://dashboard.ngrok.com/domains"
    Write-Host "  then add this line to .env :"
    Write-Host ""
    Write-Host "      NGROK_DOMAIN=your-name.ngrok-free.app" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  (or pass it once:  -Domain your-name.ngrok-free.app)"
    exit 1
}

# --- locate ngrok (winget adds it to PATH, but not to an already-open shell) --
$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) {
    # Explicit loop, NOT `@(a,b) | Where-Object`: that assigns the pipeline's
    # result, so a single surviving path comes back as a String whose .Count is 1
    # and whose [0] is the character 'C' rather than the path.
    foreach ($c in @(
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\ngrok.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe"
    )) {
        if (-not $ngrok -and (Test-Path $c)) { $ngrok = $c }
    }
}
if (-not $ngrok) {
    Write-Host "ngrok not found. Install it with:" -ForegroundColor Red
    Write-Host "    winget install --id ngrok.ngrok" -ForegroundColor Cyan
    exit 1
}

# --- preflight: authtoken available? -----------------------------------------
# Two accepted sources, in order: NGROK_TOKEN in .env (exported to the child
# process only, so the credential never lands in a global config file), or a
# pre-existing ngrok.yml written by `ngrok config add-authtoken`.
$token = $env:NGROK_AUTHTOKEN
if (-not $token) { $token = $cfg["NGROK_TOKEN"] }
$ngrokYml = Join-Path $env:LOCALAPPDATA "ngrok\ngrok.yml"
$ymlHasToken = (Test-Path $ngrokYml) -and ((Get-Content $ngrokYml -Raw) -match "authtoken")
if ($token) {
    $env:NGROK_AUTHTOKEN = $token
} elseif (-not $ymlHasToken) {
    Write-Host "ngrok has no authtoken (required for a reserved domain)." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Copy yours from https://dashboard.ngrok.com/get-started/your-authtoken"
    Write-Host "  then add it to .env :"
    Write-Host ""
    Write-Host "      NGROK_TOKEN=<YOUR_TOKEN>" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# --- preflight: one ngrok agent at a time on the free plan (ERR_NGROK_108) ----
$running = Get-Process -Name ngrok -ErrorAction SilentlyContinue
if ($running) {
    if ($Force) {
        $running | Stop-Process -Force
        Write-Host "stopped existing ngrok agent (PID $($running.Id -join ', '))" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    } else {
        Write-Host "An ngrok agent is already running (PID $($running.Id -join ', '))." -ForegroundColor Yellow
        Write-Host "The free plan allows ONE at a time. Re-run with -Force to replace it."
        exit 1
    }
}

# --- preflight: is the dashboard actually up? --------------------------------
$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Host "Nothing is listening on 127.0.0.1:$Port - start the dashboard first:" -ForegroundColor Red
    Write-Host "    scripts\run_dashboard.bat" -ForegroundColor Cyan
    Write-Host "Refusing to open the tunnel: the auth gate cannot be verified while the"
    Write-Host "dashboard is down, and an unverified gate is the whole risk here."
    exit 1
}

# --- preflight: THE GATE. Verified by probe, never by config. -----------------
# The failure this prevents: DASHBOARD_AUTH_PASSWORD is set in .env but the gate
# is not actually in force - the dashboard process started before the password
# was added, the middleware silently did not install, whatever. From .env alone
# "gated" and "wide open" look identical, and being wrong publishes live P&L.
# So: ask the running dashboard, and refuse the tunnel unless it says 401.
$authUser = $cfg["DASHBOARD_AUTH_USERNAME"]; if (-not $authUser) { $authUser = "viewer" }
$authPass = $cfg["DASHBOARD_AUTH_PASSWORD"]
$local = "http://127.0.0.1:$Port/"

if (-not $authPass) {
    Write-Host "DASHBOARD_AUTH_PASSWORD is not set in .env - refusing to publish." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Add a shared password, then RESTART the dashboard so it loads it:"
    Write-Host "      DASHBOARD_AUTH_PASSWORD=<something long>" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# 1. Unauthenticated request MUST be rejected.
$anon = $null
try   { $anon = [int](Invoke-WebRequest -Uri $local -UseBasicParsing -TimeoutSec 300).StatusCode }
catch { if ($_.Exception.Response) { $anon = [int]$_.Exception.Response.StatusCode } }
if ($anon -ne 401) {
    Write-Host "GATE CHECK FAILED - refusing to publish." -ForegroundColor Red
    Write-Host ""
    if ($null -eq $anon) {
        Write-Host "  The dashboard did not answer, so the gate could not be verified."
    } else {
        Write-Host "  An unauthenticated request to $local returned HTTP $anon, expected 401."
        Write-Host "  The dashboard is NOT gated. Most likely it is still running from"
        Write-Host "  before DASHBOARD_AUTH_PASSWORD was set - restart it:"
        Write-Host "      Stop-ScheduledTask -TaskName LlmTraderDashboard; Start-ScheduledTask -TaskName LlmTraderDashboard" -ForegroundColor Cyan
    }
    Write-Host ""
    exit 1
}

# 2. The shared credentials MUST actually open it, or you hand out a dead link.
#    A 401 here means .env and the running process disagree on the password.
$pair = "$($authUser):$($authPass)"
$hdr  = @{ Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pair)) }
$authed = $null
try   { $authed = [int](Invoke-WebRequest -Uri $local -Headers $hdr -UseBasicParsing -TimeoutSec 300).StatusCode }
catch { if ($_.Exception.Response) { $authed = [int]$_.Exception.Response.StatusCode } }
if ($authed -eq 401) {
    Write-Host "The password in .env does not open the running dashboard - refusing to publish." -ForegroundColor Red
    Write-Host "Restart the dashboard so it picks up the current DASHBOARD_AUTH_PASSWORD."
    exit 1
} elseif ($authed -ne 200) {
    # Timeout or a slow cold cache rebuild: not a security failure, so warn only.
    Write-Host "Note: authenticated probe returned '$authed' (cold cache rebuilds can be slow)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  PUBLIC (password-gated):  https://$Domain" -ForegroundColor Green
Write-Host "  forwarding to             http://127.0.0.1:$Port"
Write-Host "  Gate verified: anonymous request got 401, shared credentials got $authed." -ForegroundColor Green
Write-Host "  Share the link WITH the username/password - anyone holding both sees" -ForegroundColor Yellow
Write-Host "  live positions, P&L and account NAV." -ForegroundColor Yellow
Write-Host "  Ctrl+C stops the tunnel (the dashboard itself keeps running)."
Write-Host ""

# The ngrok inspect UI stays local-only on http://127.0.0.1:4040 (never tunnelled).
# --url supersedes the deprecated --domain (agent 3.39+); it wants a full URL, so
# normalise whatever form NGROK_DOMAIN takes.
$publicUrl = "https://" + ($Domain -replace '^https?://', '')
& $ngrok http "--url=$publicUrl" "127.0.0.1:$Port" --log=stdout --log-level=info
exit $LASTEXITCODE
