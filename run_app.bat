@echo off
rem Pronunciation Coach -- double-click to start the local app.
rem Uses .venv\ in this folder if present, otherwise the "python" on PATH.
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
"%PY%" server.py %*
if errorlevel 1 pause
