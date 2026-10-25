@echo off
rem Pronunciation Coach -- share the app publicly from this computer via a free
rem Cloudflare quick tunnel (no account needed). The link works only while this
rem window and the app window stay open; it changes every time you run this.
rem Needs cloudflared:  winget install Cloudflare.cloudflared
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set "CF=cloudflared"
if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" set "CF=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
set PYTHONIOENCODING=utf-8
set HOSTED=1
start "Pronunciation Coach server" "%PY%" server.py --no-browser
echo Waiting for the app to load the model...
:wait
timeout /t 3 /nobreak >nul
curl -s -o nul http://127.0.0.1:7860/api/info || goto wait
echo.
echo Starting the tunnel. Your public link is the https://....trycloudflare.com line below.
echo Close this window to stop sharing.
echo.
"%CF%" tunnel --no-autoupdate --url http://127.0.0.1:7860
