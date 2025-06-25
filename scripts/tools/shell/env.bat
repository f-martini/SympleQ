REM All the variable below assume PROJECT_ROOT as working directory
set PROJECT_ROOT=%~dp0../../..

REM Project env variables
set CONFIGS_ROOT=scripts/configs
set SRC_ROOT=src
set DOC_ROOT=docs
set SRC_VENV=venv
set SRC_REQUIREMENTS=scripts/configs/requirements.txt

REM Docs env variables
set DOC_VENV=%DOC_ROOT%/doc_venv
set DOC_REQUIREMENTS=%DOC_ROOT%/doc_requirements.txt
set DOC_AUTOSUMMARY=%DOC_ROOT%/index/_autosummary
set DOC_BUILD_DIR=%DOC_ROOT%/_build/html
set DOC_INDEX=%DOC_ROOT%/_build/html/index.html

REM Setup dev environment env variables
set PYTHON_PY_SETUP=%CONFIGS_ROOT%
set PERSONAL_FOLDER=scripts/personal

REM Python requirements synching env variables
set SYNC_REQUIREMENTS_SCRIPT=scripts/tools/python/sync_requirements.py

REM Clear notebooks env variables
set NOTEBOOKS_ROOT_DIR=%PROJECT_ROOT%
set DEV_REQUIREMENTS=scripts/configs/dev_requirements.txt
set CLEAR_NOTEBOOKS_SCRIPT=scripts/tools/python/clear_notebooks.py