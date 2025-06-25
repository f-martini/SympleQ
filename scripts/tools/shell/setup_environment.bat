@echo off

REM Change the working directory to the script's
REM directory and load environment variables
cd /d %~dp0
call env.bat
cd %PROJECT_ROOT%

REM Initializing virtual environment...
if not exist %SRC_VENV% (
    echo Creating virtual environment %SRC_VENV%...
    python -m venv %SRC_VENV%
)

if not exist "%SRC_REQUIREMENTS%" (
    echo requirements.txt not found.
) else (
    call %SRC_VENV%/Scripts/activate
    call python -m pip install -e %PYTHON_PY_SETUP%
    call deactivate
)

REM Generating unversioned folders...
set "folders=%PERSONAL_FOLDER%"
for %%F in (%folders%) do (
    if not exist "%%F" (
        mkdir "%%F"
        echo Created folder: %%F
    )
)

REM Writing unversioned files...
REM if not exist "./configs/.env" (
REM    echo GITHUB_TOKEN="undefined" > ./configs/.env
REM )
REM echo "Please fill the .env file with the required secrets if not already done."