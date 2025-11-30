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
    call %SRC_VENV%/Scripts/activate
    call python -m pip install -r %DEV_REQUIREMENTS%
    call deactivate
)

call %SRC_VENV%/Scripts/activate
call python -m pip install --upgrade pip setuptools setuptools-scm
call python -m pip install -e %PYTHON_PY_SETUP%
call deactivate

REM Generating unversioned folders...
set "folders=%PERSONAL_FOLDER%"
for %%F in (%folders%) do (
    if not exist "%%F" (
        mkdir "%%F"
        echo Created folder: %%F
    )
)
