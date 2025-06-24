@echo off

REM Change the working directory to the script's
REM directory and load environment variables
cd /d %~dp0
call env.bat
cd %PROJECT_ROOT%

if not exist "venv" (
    echo Virtual environment not found.
    exit /b 1
)

if not exist "./configs/requirements.txt" (
    echo requirements.txt not found.
    exit /b 1
)

call venv/Scripts/activate
call pip freeze > ./configs/current_requirements.txt
call python scripts/python/tools/sync_requirements.py  --r ./configs/requirements.txt

cd configs
move /y updated_requirements.txt requirements.txt
del current_requirements.txt
echo Requirements have been synced and updated.