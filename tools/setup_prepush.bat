@echo off
rem ============================================================
rem  Pre-push check: one-time setup (run once on each PC).
rem  - git config core.hooksPath .githooks  (this repository only)
rem  - download gitleaks 8.30.1 (official zip, SHA256 verified)
rem    into %LOCALAPPDATA%\agent-office-tools
rem  - check that pytest and Agent Office packages are installed
rem  Safe to run again. Guide: docs folder in the repository.
rem  (This file is ASCII only on purpose: cmd misreads UTF-8 text in .bat files.)
rem ============================================================
chcp 65001 >nul
setlocal
set "REPO=%~dp0.."
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
"%PY%" tools\prepush\first_setup.py
if /i not "%~1"=="nopause" pause
