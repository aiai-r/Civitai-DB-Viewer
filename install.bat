@echo off
echo Creating a Python virtual environment and installing required libraries.

rem Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not on PATH.
    pause
    exit /b
)

rem Create virtual environment
echo Creating virtual environment (venv)...
python -m venv venv

rem Activate virtual environment
call venv\Scripts\activate

rem Install required libraries
echo Installing required libraries...
pip install -r requirements.txt

echo.
echo Setup completed.
echo To use the virtual environment, run:
echo venv\Scripts\activate
echo.
pause
