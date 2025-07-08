@echo off

REM Change the working directory to the script's
REM directory and load environment variables
cd /d %~dp0
call env.bat
cd %PROJECT_ROOT%

if not exist "%SRC_VENV%" (
    echo Virtual environment not found.
    exit /b 1
)

if not exist "%SRC_REQUIREMENTS%" (
    echo requirements.txt not found.
    exit /b 1
)

call %SRC_VENV%/Scripts/activate
call pip freeze > %CONFIGS_ROOT%/current_requirements.txt
call python %SYNC_REQUIREMENTS_SCRIPT% --r %SRC_REQUIREMENTS%

cd %CONFIGS_ROOT%
move updated_requirements.txt requirements.txt
del current_requirements.txt
echo Requirements have been synced and updated.

cd %PROJECT_ROOT%
call deactivate