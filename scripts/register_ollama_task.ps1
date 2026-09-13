<#
  Register (and start) the LOCAL LLM server (Ollama) as a Windows Task Scheduler job.

  RUN THIS ONCE, AS ADMINISTRATOR:
    Right-click Start  ->  "Terminal (Admin)",  then:

      powershell -ExecutionPolicy Bypass -File "C:\Users\mathi\PycharmProjects\llm_trader\scripts\register_ollama_task.ps1"

  Creates task 'LlmTraderOllama':
    - runs scripts\run_ollama.bat (loopback bind, models dir, keep-alive) at log on
    - auto-restarts on failure (every 1 min, up to 3x)
    - runs only while YOU are logged on (no stored password; safe on a laptop)

  WHY IT IS ITS OWN TASK, not part of restart_all.ps1:
  restart_all bounces the TRADER stack (dashboard + scheduler). The model server
  is a DEPENDENCY of that stack, not a member of it - bouncing it on every code
  deploy would evict a 5 GB model from VRAM and make the next tick pay a cold
  load for no reason. It is restarted only when the model or its settings change.

  A DEAD SERVER IS NOT SILENT: `sentiment._sentiment_engine_order` falls through
  to the hosted engines, and if those are down too the run records provider
  'none', which `pipeline._assess_llm_health` reports as sentiment DOWN (CRITICAL
  log + email banner). That is the alarm; this task is what keeps it quiet.

  Remove later:  Unregister-ScheduledTask -TaskName LlmTraderOllama -Confirm:$false
#>
$ErrorActionPreference = "Stop"
$TaskName = "LlmTraderOllama"

# 0. Must be elevated - creating a task writes to the protected task store.
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this in an ELEVATED PowerShell (Run as administrator)." -ForegroundColor Red
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot          # scripts\ -> repo root
$bat  = Join-Path $root "scripts\run_ollama.bat"
if (-not (Test-Path $bat)) { throw "launcher not found: $bat" }

# 1. Stop any hand-started server so only the supervised one runs (frees 11434).
$running = Get-Process ollama -ErrorAction SilentlyContinue
foreach ($p in $running) {
    Stop-Process -Id $p.Id -Force
    Write-Host "stopped existing ollama PID $($p.Id)"
}

# 2. Register the task.
$action   = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
              -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable `
              -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Local LLM server (Ollama) for the llm_trader local sentiment engine" -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "registered and started '$TaskName'" -ForegroundColor Green

# 3. Verify the endpoint answers before claiming success (a registered task that
#    fails to bind looks identical to a working one from the task list alone).
$deadline = (Get-Date).AddSeconds(60)
$up = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { $up = $true; break }
    } catch { Start-Sleep -Seconds 3 }
}
if ($up) {
    Write-Host "server answering on http://127.0.0.1:11434" -ForegroundColor Green
} else {
    Write-Host "port 11434 not answering yet - check Task Scheduler > History." -ForegroundColor Yellow
}
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State | Format-Table -AutoSize
