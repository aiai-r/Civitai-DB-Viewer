@echo off
cd /d "%~dp0"
call "venv\Scripts\python.exe" app.py
if errorlevel 1 pause
