@echo off
REM ML Model Comparison Platform - Setup Script
REM For Windows systems

echo =========================================
echo ML Model Comparison Platform - Setup
echo =========================================
echo.

REM Check Python
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo Dependencies installed
echo.

REM Create required directories
echo Creating required directories...
if not exist "uploads" mkdir uploads
if not exist "static\plots" mkdir static\plots
if not exist "static\css" mkdir static\css
if not exist "static\js" mkdir static\js
if not exist "templates" mkdir templates
echo Directories created
echo.

echo =========================================
echo Setup Complete!
echo =========================================
echo.
echo To start the application:
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Run the app: python app.py
echo 3. Open browser: http://localhost:5000
echo.
echo To deactivate virtual environment: deactivate
echo.

pause