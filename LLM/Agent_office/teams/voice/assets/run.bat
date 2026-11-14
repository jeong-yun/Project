@echo off
REM ============================================================
REM  SVC preprocess launcher (double-click to run)
REM  Checks Python, then runs preprocess.py in interactive mode.
REM  Dependencies are auto-installed by the script itself.
REM ============================================================
setlocal

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo IMPORTANT: check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

python "%~dp0preprocess.py" %*
