<#
  Register (and start) the PUBLIC ngrok tunnel as a Windows Task Scheduler job,
  so the public dashboard URL comes back by itself after a reboot.

  RUN THIS ONCE, AS ADMINISTRATOR:
    Right-click Start  ->  "Terminal (Admin)",  then:

      powershell -ExecutionPolicy Bypass -File "C:\Users\mathi\PycharmProjects\llm_trader\scripts\register_public_dashboard_task.ps1"

  Creates task 'LlmTraderPublicTunnel':
    - runs scripts\run_public_dashboard.bat (repo-root CWD -> reads .env) at log on
    - auto-restarts on failure (every 1 min, up to 3x)
    - runs only while YOU are logged on (no stored password)

  !! This makes the dashboard PUBLIC on every boot, with no login. To take it
    down temporarily:  Stop-ScheduledTask -TaskName LlmTraderPublicTunnel
    To remove entirely: Unregister-ScheduledTask -TaskName LlmTraderPublicTunnel -Confirm:$false
    Either way the LOCAL dashboard (LlmTraderDashboard) is unaffected.

  NOTE: keep this file pure ASCII - PowerShell 5.1 reads a BOM-less file as
  cp1252, where a UTF-8 dash decodes into a smart quote that breaks the parse.
#>
$ErrorActionPreference = "Stop"
$TaskName = "LlmTraderPublicTunnel"

# 0. Must be elevated - creating a task writes to the protected task store.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this in an ELEVATED PowerShell (Run as administrator)." -ForegroundColor Red
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root
$bat  = Join-Path $root "scripts\run_public_dashboard.bat"
if (-not (Test-Path $bat)) { throw "launcher not found: $bat" }

# 1. Refuse to register a task that would fail on every boot.
$envFile = Join-Path $root ".env"
$domain = $null
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line.Trim() -match "^NGROK_DOMAIN\s*=\s*(.+)$") { $domain = $Matches[1].Trim().Trim('"') }
    }
}
if (-not $domain) { throw "NGROK_DOMAIN is not set in .env - see scripts\run_public_dashboard.ps1 header." }

# 2. Stop any manual tunnel so only the supervised one runs (free plan: one agent).
Get-Process -Name ngrok -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.Id -Force; Write-Host "stopped existing ngrok PID $($_.Id)" }

# 3. Register the task.
$action    = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger   = New-ScheduledTaskTrigger -AtLogOn
$settings  = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
               -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
               -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Public ngrok tunnel in front of the read-only monitoring dashboard (NO login)." -Force | Out-Null
Write-Host "Registered '$TaskName' to run as $env:USERDOMAIN\$env:USERNAME (only when logged on)." -ForegroundColor Green

# 4. Start it and confirm the tunnel actually came up.
Start-ScheduledTask -TaskName $TaskName
$deadline = (Get-Date).AddSeconds(30); $up = $false
while ((Get-Date) -lt $deadline) {
    if (Get-Process -Name ngrok -ErrorAction SilentlyContinue) { $up = $true; break }
    Start-Sleep -Seconds 3
}
Write-Host "Task state: $((Get-ScheduledTask -TaskName $TaskName).State)"
if ($up) { Write-Host "Public (no login): https://$domain" -ForegroundColor Green }
else     { Write-Host "ngrok did not start - run scripts\run_public_dashboard.ps1 by hand to see why." -ForegroundColor Yellow }
Write-Host "Take it down:  Stop-ScheduledTask -TaskName $TaskName"
