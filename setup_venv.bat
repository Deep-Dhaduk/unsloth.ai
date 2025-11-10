@echo off
REM Batch script to setup virtual environment and install dependencies
REM For Windows Command Prompt users

echo Setting up Python virtual environment...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ===================================
echo Setup complete!
echo ===================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, simply run:
echo   deactivate
echo.
pause
