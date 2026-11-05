@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  올리기 전 검사 — 처음 한 번 설정 (PC 마다)
rem  - git push 때 검사가 돌게 훅 폴더(.githooks)를 켠다 (이 저장소에만)
rem  - gitleaks 8.30.1 공식 zip 을 SHA256 확인 후 %%LOCALAPPDATA%%\agent-office-tools 에 받는다
rem  - pytest·Agent Office 패키지가 있는지 확인한다
rem  다시 실행해도 괜찮다 (이미 된 것은 건너뜀)
rem ============================================================
set "REPO=%~dp0.."
cd /d "%REPO%"
rem 파이썬 찾기 (pyenv 의 python.bat 같은 껍데기여도 실제 python.exe 경로를 얻는다)
set "PY="
for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%p"
if not defined PY for /f "delims=" %%p in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%p"
if not defined PY (
  echo 파이썬을 찾지 못했습니다. Python 3.11 을 설치하고 PATH 에 넣은 뒤 다시 실행하세요.
  pause
  exit /b 1
)
"%PY%" tools\prepush\first_setup.py
if errorlevel 1 (
  echo.
  echo 설정을 마치지 못했습니다. 위 안내를 확인하세요.
) else (
  echo.
  echo 완료.
)
if /i not "%~1"=="nopause" pause
