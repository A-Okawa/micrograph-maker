@echo off
rem Micrograph Maker (Windows): double-click to start. First run installs the packages.
cd /d "%~dp0"
set "PY=py -3"
where py >nul 2>nul || set "PY=python"
if exist ".venv\Scripts\python.exe" goto run
echo First-time setup, this takes a few minutes...
%PY% -m venv .venv || goto fail
if exist wheels (
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements.txt || goto fail
) else (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
)
:run
echo The app opens at http://localhost:8501 . Close this window to quit.
".venv\Scripts\python.exe" -m streamlit run app.py --browser.gatherUsageStats false
goto :eof
:fail
echo Setup failed. Install Python 3.10 or later from https://www.python.org (tick "Add python.exe to PATH").
echo Offline? Put a "wheels" folder next to this file (see README).
if exist .venv rmdir /s /q .venv
pause
