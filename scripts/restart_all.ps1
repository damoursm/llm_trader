<#
  Restart both the LLM Trader dashboard and scheduler via their registered
  Task Scheduler jobs (LlmTraderDashboard / LlmTraderScheduler) - the ONE
  supported way to bounce them. Never launches python.exe ad-hoc: mixing
  ad-hoc Start-Process with the AtLogOn-triggered tasks is how duplicate
  scheduler/dashboard processes crept in (observed 2026-07-02 - two
  schedulers ticking concurrently is a real risk: racing DuckDB writes,
  doubled LLM spend, possible duplicate broker submissions).

  Does NOT require elevation (Stop/Start-ScheduledTask don't need it once the
  tasks already exist - only registration does).

    powershell -ExecutionPolicy Bypass -File "C:\Users\mathi\PycharmProjects\llm_trader\scripts\restart_all.ps1"

  Register the tasks first (one-time, admin) if they don't exist yet:
    scripts\register_dashboard_task.ps1
    scripts\register_scheduler_task.ps1
#>
$ErrorActionPreference = "Stop"
$Tasks = "LlmTraderDashboard", "LlmTraderScheduler"

foreach ($t in $Tasks) {
    if (-not (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue)) {
        Write-Host "Task '$t' is not registered. Run scripts\register_dashboard_task.ps1 /" -ForegroundColor Red
        Write-Host "register_scheduler_task.ps1 (elevated, one-time) first." -ForegroundColor Red
        exit 1
    }
}

# 1. Stop both tasks - Task Scheduler kills the tracked python.exe process tree.
Write-Host "Stopping..." -ForegroundColor Cyan
foreach ($t in $Tasks) { Stop-ScheduledTask -TaskName $t }

# 2. Wait for the underlying processes to actually exit; force-kill any straggler
#    (covers ad-hoc-launched duplicates Task Scheduler isn't tracking).
$deadline = (Get-Date).AddSeconds(20)
$left = $null
do {
    $left = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'main\.py --(dashboard|schedule)' }
    if (-not $left) { break }
    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $deadline)
if ($left) {
    Write-Host "Force-killing straggler(s): $($left.ProcessId -join ', ')" -ForegroundColor Yellow
    $left | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# 3. Start both tasks.
Write-Host "Starting..." -ForegroundColor Cyan
foreach ($t in $Tasks) { Start-ScheduledTask -TaskName $t }

# 4. Verify: wait for the dashboard port to come up (imports take a few seconds)
#    and report the final process list + task states.
$deadline = (Get-Date).AddSeconds(30)
$listening = $false
while ((Get-Date) -lt $deadline) {
    if (Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue) { $listening = $true; break }
    Start-Sleep -Seconds 2
}
Write-Host ""
if ($listening) {
    Write-Host "Dashboard serving on http://127.0.0.1:8050" -ForegroundColor Green
} else {
    Write-Host "Port 8050 not listening yet - check Task Scheduler > History or logs." -ForegroundColor Yellow
}

Get-ScheduledTask -TaskName $Tasks | Select-Object TaskName, State | Format-Table -AutoSize
Write-Host "Live processes:"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'main\.py --(dashboard|schedule)' } |
    Select-Object ProcessId, CommandLine | Format-Table -AutoSize
