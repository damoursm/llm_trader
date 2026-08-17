<#
  DECOMMISSIONED 2026-08-16 -- superseded by Tailscale Funnel. DO NOT RUN.

  The dashboard now has exactly ONE public access point:

      https://victushp.tail8e1bf1.ts.net

  set up with `tailscale funnel --bg --https=443 http://127.0.0.1:8050`. Funnel
  gives a real Let's Encrypt certificate, needs no second agent process, and has
  none of the free-tier ngrok friction (the one-time browser interstitial and the
  enforced minimum agent version whose ERR_NGROK_121 reads like an auth error).

  This script is kept as the record of HOW the gate was verified rather than
  trusted -- the two-probe pattern below is still the right shape for any future
  exposure -- but running it would re-open a SECOND way in, which is exactly what
  "one point of access" was meant to end. The guard immediately below stops that.

  Note the environment it was written for no longer exists: DASHBOARD_HOST is now
  127.0.0.1 (so nothing but a local proxy can reach the app at all) and the
  localhost/tailnet bypasses are EMPTY (so the password gates every request,
  including this PC's). The "LOCALHOST IS EXEMPT" paragraph below is therefore
  historical.

  To genuinely retire ngrok, delete NGROK_TOKEN and NGROK_DOMAIN from .env and
  revoke the token in the ngrok dashboard.

  ---- original documentation follows ----

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

  LOCALHOST IS EXEMPT: browsing http://127.0.0.1:<port> on this PC needs no
  password. That exemption is the reason the checks below are shaped the way they
  are - ngrok forwards the public hostname to 127.0.0.1, so a public visitor's
  peer address is loopback too, and "localhost bypass" done naively would hand the
  open internet a free pass. The dashboard distinguishes them by the proxy headers
  and the Host header the tunnel relays (dashboard/app.py::_is_local_request).

  This script REFUSES to publish unless it has verified the gate, TWICE and by
  probe rather than by config, because "gated" and "wide open" look identical
  from .env alone:
    - before the tunnel opens, a local request wearing the tunnel's headers must
      come back 401 (catches a dashboard whose bypass is too generous);
    - after the tunnel opens, an anonymous request to the REAL PUBLIC URL must
      come back 401, or the tunnel is torn down again within seconds.
  The second check is the one that cannot be fooled: it is the actual request an
  actual stranger would make.

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

# ---- DECOMMISSIONED GUARD (2026-08-16) ----------------------------------
# A second public entrance is worse than no second entrance: it would be the
# one nobody remembers to check when the password is rotated or the gate is
# changed. Refuse loudly rather than quietly publishing a rival URL.
#
# Note this ALSO cannot work as written any more: the dashboard binds
# 127.0.0.1 and every bypass is empty, so the script's own pre-flight probes
# (which expect a localhost request to succeed WITHOUT the password) would
# fail. Better to say why than to let it die confusingly.
Write-Host ""
Write-Host "  This script is DECOMMISSIONED." -ForegroundColor Yellow
Write-Host "  The dashboard's single public access point is Tailscale Funnel:"
Write-Host "      https://victushp.tail8e1bf1.ts.net" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Check it with:   tailscale funnel status"
Write-Host "  Re-arm it with:  tailscale funnel --bg --https=443 http://127.0.0.1:8050"
Write-Host "  Turn it off:     tailscale funnel --https=443 off"
Write-Host ""
Write-Host "  Opening an ngrok tunnel now would create a SECOND way in. If you"
Write-Host "  truly intend that, read the header of this file first and remove"
Write-Host "  this guard deliberately."
Write-Host ""
exit 1
# -------------------------------------------------------------------------

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

# Headers a request picks up on its way through the tunnel. The dashboard treats
# their PRESENCE (never their value) as proof the request is not local, so sending
# them at the loopback port reproduces what a stranger's request will look like.
$tunnelLike = @{ "X-Forwarded-For" = "203.0.113.7"; "X-Forwarded-Proto" = "https" }
$pair = "$($authUser):$($authPass)"
$basic = "Basic " + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pair))

function Get-Status {
    param([string]$Uri, [hashtable]$Headers = @{}, [int]$TimeoutSec = 300)
    try {
        return [int](Invoke-WebRequest -Uri $Uri -Headers $Headers -UseBasicParsing `
                                       -TimeoutSec $TimeoutSec -MaximumRedirection 0).StatusCode
    } catch {
        if ($_.Exception.Response) { return [int]$_.Exception.Response.StatusCode }
        return $null
    }
}

if (-not $authPass) {
    Write-Host "DASHBOARD_AUTH_PASSWORD is not set in .env - refusing to publish." -ForegroundColor Red
    Write-Host ""
    Write-Host "  Add a shared password, then RESTART the dashboard so it loads it:"
    Write-Host "      DASHBOARD_AUTH_PASSWORD=<something long>" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# 1. A request wearing the tunnel's headers MUST be rejected, even though it
#    arrives on loopback. This is the localhost bypass being tested from the
#    dangerous side: if it answers 200 here it will answer 200 to the internet.
$anon = Get-Status -Uri $local -Headers $tunnelLike
if ($anon -ne 401) {
    Write-Host "GATE CHECK FAILED - refusing to publish." -ForegroundColor Red
    Write-Host ""
    if ($null -eq $anon) {
        Write-Host "  The dashboard did not answer, so the gate could not be verified."
    } else {
        Write-Host "  An unauthenticated tunnel-shaped request to $local returned HTTP $anon,"
        Write-Host "  expected 401. The dashboard is NOT gated against public traffic."
        Write-Host "  Most likely it is still running from before DASHBOARD_AUTH_PASSWORD"
        Write-Host "  was set (settings are read at import) - restart it:"
        Write-Host "      Stop-ScheduledTask -TaskName LlmTraderDashboard; Start-ScheduledTask -TaskName LlmTraderDashboard" -ForegroundColor Cyan
        Write-Host "  If it was restarted, check DASHBOARD_AUTH_BYPASS_NETWORKS in .env:"
        Write-Host "  a bypass wider than loopback can let tunnelled traffic through."
    }
    Write-Host ""
    exit 1
}

# 2. The shared credentials MUST actually open it, or you hand out a dead link.
#    A 401 here means .env and the running process disagree on the password.
$authed = Get-Status -Uri $local -Headers ($tunnelLike + @{ Authorization = $basic })
if ($authed -eq 401) {
    Write-Host "The password in .env does not open the running dashboard - refusing to publish." -ForegroundColor Red
    Write-Host "Restart the dashboard so it picks up the current DASHBOARD_AUTH_PASSWORD."
    exit 1
} elseif ($authed -ne 200) {
    # Timeout or a slow cold cache rebuild: not a security failure, so warn only.
    Write-Host "Note: authenticated probe returned '$authed' (cold cache rebuilds can be slow)." -ForegroundColor Yellow
}

# 3. Informational: confirm this PC still browses without a password. Not a
#    security condition - an owner who gated themselves too is merely annoyed.
$localAnon = Get-Status -Uri $local -TimeoutSec 60
if ($localAnon -eq 200) {
    Write-Host "  Local check: http://127.0.0.1:$Port opens with no password." -ForegroundColor Green
} else {
    Write-Host "  Note: local browsing returned '$localAnon' - this PC will be asked for the" -ForegroundColor Yellow
    Write-Host "  password too (DASHBOARD_AUTH_BYPASS_NETWORKS empty, or an older dashboard)." -ForegroundColor Yellow
}

# --- open the tunnel, then VERIFY IT FROM THE OUTSIDE -------------------------
# The ngrok inspect UI stays local-only on http://127.0.0.1:4040 (never tunnelled).
# --url supersedes the deprecated --domain (agent 3.39+); it wants a full URL, so
# normalise whatever form NGROK_DOMAIN takes.
$publicUrl = "https://" + ($Domain -replace '^https?://', '')
$log = Join-Path $env:TEMP "ngrok_dashboard.log"

Write-Host ""
Write-Host "  Opening tunnel $publicUrl -> http://127.0.0.1:$Port ..."
$proc = Start-Process -FilePath $ngrok -PassThru -NoNewWindow `
    -ArgumentList @("http", "--url=$publicUrl", "127.0.0.1:$Port", "--log=stdout", "--log-level=info") `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"

# The check that cannot be fooled: the exact request a stranger makes. Everything
# before this was a local simulation of the tunnel; this is the tunnel.
# ngrok-skip-browser-warning gets past the free tier's click-through interstitial
# (ERR_NGROK_6024), which is an edge page - without it the probe would grade
# ngrok's own HTML instead of the dashboard's answer.
$skip = @{ "ngrok-skip-browser-warning" = "1" }
$public = $null
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    $public = Get-Status -Uri "$publicUrl/" -Headers $skip -TimeoutSec 30
    # 404/502 = edge is up but the tunnel has not registered yet; keep waiting.
    if ($public -eq 401 -or $public -eq 200) { break }
}

if ($proc.HasExited) {
    Write-Host ""
    Write-Host "ngrok exited immediately (code $($proc.ExitCode)). Last log lines:" -ForegroundColor Red
    if (Test-Path $log) { Get-Content $log -Tail 20 }
    if (Test-Path "$log.err") { Get-Content "$log.err" -Tail 20 }
    exit 1
}

if ($public -ne 401) {
    Write-Host ""
    Write-Host "PUBLIC GATE CHECK FAILED - tearing the tunnel down." -ForegroundColor Red
    Write-Host ""
    if ($null -eq $public) {
        Write-Host "  $publicUrl never answered, so the gate could not be verified from"
        Write-Host "  outside. Refusing to leave an unverified tunnel open."
    } else {
        Write-Host "  An anonymous request to $publicUrl returned HTTP $public, expected 401."
        Write-Host "  Live positions and P&L would be readable by anyone with the link."
    }
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Write-Host "  Tunnel stopped. The local dashboard is untouched and still running."
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "  PUBLIC (password-gated):  $publicUrl" -ForegroundColor Green
Write-Host "  forwarding to             http://127.0.0.1:$Port"
Write-Host "  Gate verified FROM THE PUBLIC URL: anonymous got 401, credentials got $authed." -ForegroundColor Green
Write-Host "  Share the link WITH the username/password - anyone holding both sees" -ForegroundColor Yellow
Write-Host "  live positions, P&L and account NAV." -ForegroundColor Yellow
Write-Host "  This PC still browses http://127.0.0.1:$Port with no password."
Write-Host "  Live traffic inspector: http://127.0.0.1:4040   agent log: $log"
Write-Host "  Ctrl+C stops the tunnel (the dashboard itself keeps running)."
Write-Host ""

try {
    Wait-Process -Id $proc.Id
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Tunnel closed. The dashboard is still running locally." -ForegroundColor Yellow
}
exit 0
