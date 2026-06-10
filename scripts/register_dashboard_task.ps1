<#
  Register (and start) the LLM Trader monitoring dashboard as a Windows Task Scheduler job.

  RUN THIS ONCE, AS ADMINISTRATOR:
    Right-click Start  ->  "Terminal (Admin)"  (or "Windows PowerShell (Admin)"),  then:

      powershell -ExecutionPolicy Bypass -File "C:\Users\mathi\PycharmProjects\llm_trader\scripts\register_dashboard_task.ps1"

  Creates task 'LlmTraderDashboard':
    - runs scripts\run_dashboard.bat (repo-root CWD + venv python) at log on
    - auto-restarts on failure (every 1 min, up to 3x)
    - runs only while YOU are logged on (no stored password; safe on a laptop)

  Remove later:  Unregister-ScheduledTask -TaskName LlmTraderDashboard -Confirm:$false
#>
$ErrorActionPreference = "Stop"
$TaskName = "LlmTraderDashboard"

# 0. Must be elevated — creating a task writes to the protected task store.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this in an ELEVATED PowerShell (Run as administrator)." -ForegroundColor Red
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root
$bat  = Join-Path $root "scripts\run_dashboard.bat"
if (-not (Test-Path $bat)) { throw "launcher not found: $bat" }

# 1. Stop any manual dashboard so only the supervised one runs (frees port 8050).
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*main.py*--dashboard*' }
foreach ($p in $running) { Stop-Process -Id $p.ProcessId -Force; Write-Host "stopped existing dashboard PID $($p.ProcessId)" }

# 2. Register the task.
$action    = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger   = New-ScheduledTaskTrigger -AtLogOn
$settings  = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
               -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
               -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Run main.py --dashboard at logon via the venv launcher; auto-restart on failure." -Force | Out-Null
Write-Host "Registered '$TaskName' to run as $env:USERDOMAIN\$env:USERNAME (only when logged on)." -ForegroundColor Green

# 3. Start it now and wait for the port to come up (dash/plotly imports take a few seconds).
Start-ScheduledTask -TaskName $TaskName
$deadline = (Get-Date).AddSeconds(30); $listening = $false
while ((Get-Date) -lt $deadline) {
    if (Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue) { $listening = $true; break }
    Start-Sleep -Seconds 3
}
$state = (Get-ScheduledTask -TaskName $TaskName).State
Write-Host "Task state: $state"
if ($listening) { Write-Host "Dashboard serving on http://127.0.0.1:8050" -ForegroundColor Green }
else            { Write-Host "Port 8050 not listening yet - check Task Scheduler -> History and logs\." -ForegroundColor Yellow }
Write-Host "Manage:  Start-ScheduledTask / Stop-ScheduledTask / Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
