# [2.0 자동 실행 등록 - 최초 1회] 1_0_run_daily.py(수집 → 파싱·통합 → LLM 입력)를 매일 지정 시각에 실행하도록 Windows 작업 스케줄러에 등록한다.
# PC 가 꺼져 있어 시간을 놓치면 다음 부팅/로그인 후 바로 1회 실행된다 (StartWhenAvailable).
#
# 사용법:  powershell -ExecutionPolicy Bypass -File .\2_0_register_task.ps1 [-At 16:00]
# 해제:    Unregister-ScheduledTask -TaskName SnortRuleFetch -Confirm:$false

param(
    [string]$At = "16:00",
    [string]$TaskName = "SnortRuleFetch"
)

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $dir "1_0_run_daily.py"

# pyenv shim(.bat) 대신 실제 python.exe 경로를 사용
$python = (& python -c "import sys; print(sys.executable)").Trim()
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
if (Test-Path $pythonw) { $python = $pythonw }   # 실행 시 콘솔 창이 뜨지 않도록

$action   = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $dir
$trigger  = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "등록 완료: '$TaskName' 매일 $At 실행 ($python)"
