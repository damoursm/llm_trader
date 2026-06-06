<#
  Register (and start) the LLM Trader scheduler as a Windows Task Scheduler job.

  RUN THIS ONCE, AS ADMINISTRATOR:
    Right-click Start  ->  "Terminal (Admin)"  (or "Windows PowerShell (Admin)"),  then:

      powershell -ExecutionPolicy Bypass -File "C:\Users\mathi\PycharmProjects\llm_trader\scripts\register_scheduler_task.ps1"

  Creates task 'LlmTraderScheduler':
    - runs scripts\run_scheduler.bat (repo-root CWD + venv python) at log on
    - auto-restarts on failure (every 1 min, up to 3x)
    - runs only while YOU are logged on (no stored password; safe on a laptop)

  Remove later:  Unregister-ScheduledTask -TaskName LlmTraderScheduler -Confirm:$false
#>
$ErrorActionPreference = "Stop"
$TaskName = "LlmTraderScheduler"

# 0. Must be elevated — creating a task writes to the protected task store.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this in an ELEVATED PowerShell (Run as administrator)." -ForegroundColor Red
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root
$bat  = Join-Path $root "scripts\run_scheduler.bat"
if (-not (Test-Path $bat)) { throw "launcher not found: $bat" }

# 1. Stop any manual scheduler so only the supervised one runs (frees IBKR clientId 11).
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*main.py*--schedule*' }
foreach ($p in $running) { Stop-Process -Id $p.ProcessId -Force; Write-Host "stopped existing scheduler PID $($p.ProcessId)" }

# 2. Register the task.
$action    = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger   = New-ScheduledTaskTrigger -AtLogOn
$settings  = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
               -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
               -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Run main.py --schedule at logon via the venv launcher; auto-restart on failure." -Force | Out-Null
Write-Host "Registered '$TaskName' to run as $env:USERDOMAIN\$env:USERNAME (only when logged on)." -ForegroundColor Green

# 3. Start it now (idles outside market hours; live for the next RTH session).
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 4
$state = (Get-ScheduledTask -TaskName $TaskName).State
$proc  = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like '*main.py*--schedule*' }
Write-Host "Task state: $state"
if ($proc) { Write-Host ("Scheduler process running: PID {0}" -f $proc.ProcessId) -ForegroundColor Green }
else       { Write-Host "Scheduler not detected yet - check Task Scheduler -> History." -ForegroundColor Yellow }
Write-Host "Manage:  Start-ScheduledTask / Stop-ScheduledTask / Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
