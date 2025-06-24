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

if not exist "configs/requirements.txt" (
    echo requirements.txt not found.
) else (
    call venv/Scripts/activate
    call pip install -r ./configs/requirements.txt
    call deactivate
)

REM Generating unversioned folders...
set "folders=scripts/personal"
for %%F in (%folders%) do (
    if not exist "%%F" (
        mkdir "%%F"
        echo Created folder: %%F
    )
)

REM Writing unversioned .env file...
if not exist "./configs/.env" (
    echo GITHUB_TOKEN="undefined" > ./configs/.env
)
echo "Please fill the .env file with the required secrets if not already done."