@echo off
rem ============================================================
rem  Agent Office: check, then upload (git push origin main:check/main)
rem  1. pre-push check on this PC (secret scan + tests for changed files)
rem  2. if it passes, push main to check/main on GitHub
rem  3. GitHub runs the full checks and moves main forward only if all pass
rem  First run also does the one-time setup.
rem  Copy this file next to the Project folder (Desktop\...\Project).
rem  Source: Project\tools\ in the repository.
rem  (This file is ASCII only on purpose: cmd misreads UTF-8 text in .bat files.)
rem ============================================================
chcp 65001 >nul
setlocal
set "REPO="
if exist "%~dp0Project\tools\prepush\push_check_main.py" set "REPO=%~dp0Project"
if not defined REPO if exist "%~dp0..\tools\prepush\push_check_main.py" set "REPO=%~dp0.."
if not defined REPO (
  echo [ERROR] Repository not found. Put this file next to the Project folder, or inside Project\tools.
  pause
  exit /b 1
)
cd /d "%REPO%"
rem Find python (resolve pyenv's python.bat shim to the real python.exe)
set "PY="
for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%p"
if not defined PY for /f "delims=" %%p in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%p"
if not defined PY (
  echo [ERROR] Python not found. Install Python 3.11 and add it to PATH, then run this again.
  pause
  exit /b 1
)
"%PY%" tools\prepush\push_check_main.py
pause
