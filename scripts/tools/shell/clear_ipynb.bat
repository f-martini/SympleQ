@echo off

REM Change the working directory to the script's
REM directory and load environment variables
cd /d %~dp0
call env.bat
cd %PROJECT_ROOT%

REM Check if the virtual environment exists
if not exist "%SRC_VENV%" (
    echo Virtual environment not found.
    exit /b 1
)

echo Activating virtual environment...
call %SRC_VENV%/Scripts/activate

if not exist "%DEV_REQUIREMENTS%" (
    echo %DEV_REQUIREMENTS% not found.
    exit /b 1
)

echo Installing requirements from %DEV_REQUIREMENTS%...
call pip install -r %DEV_REQUIREMENTS% >nul 2>&1

echo Clearing Jupyter notebooks...
call python %CLEAR_NOTEBOOKS_SCRIPT% %NOTEBOOKS_ROOT_DIR%

echo Done!
call deactivate
