@echo off
chcp 65001 >nul
setlocal
rem ============================================================
rem  Agent Office 검사 후 올리기
rem  - 내 PC 에서 '올리기 전 검사'(비밀 검사·바뀐 부분 테스트, 1~2분)를 돌리고
rem  - 통과하면 main 을 GitHub 의 check/main 으로 올린다 (git push origin main:check/main)
rem  - GitHub 이 전체 테스트·비밀 검사를 돌려 모두 통과하면 main 에 자동 반영한다
rem  - 처음이면 설정(훅 폴더·gitleaks 내려받기)부터 한다
rem  원본: 저장소 tools\Agent_Office_검사후_올리기.bat — Project 폴더 옆(바탕화면\프로젝트)에 복사해 쓴다
rem ============================================================
set "REPO="
if exist "%~dp0Project\tools\prepush\push_check_main.py" set "REPO=%~dp0Project"
if not defined REPO if exist "%~dp0..\tools\prepush\push_check_main.py" set "REPO=%~dp0.."
if not defined REPO (
  echo 저장소를 찾지 못했습니다. 이 파일을 Project 폴더 옆 또는 Project\tools 안에 두세요.
  pause
  exit /b 1
)
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
"%PY%" tools\prepush\push_check_main.py
pause
