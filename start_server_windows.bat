@echo off
rem Micrograph Maker as the lab's shared app (Windows): other PCs on the campus network use it in a browser.
cd /d "%~dp0"
set "PY=py -3"
where py >nul 2>nul || set "PY=python"
if exist ".venv\Scripts\python.exe" goto venv
%PY% serve.py --check >nul 2>nul && goto system
echo First-time setup, this takes a few minutes...
%PY% -m venv .venv || goto fail
if exist wheels goto offline
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
goto venv
:offline
".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements.txt || goto fail
:venv
".venv\Scripts\python.exe" serve.py
pause
goto :eof
:system
%PY% serve.py
pause
goto :eof
:fail
echo Setup failed. Install Python 3.10 or later from https://www.python.org (tick "Add python.exe to PATH").
echo Offline? Put a "wheels" folder next to this file (see README).
if exist .venv rmdir /s /q .venv
pause
