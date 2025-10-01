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
set DOC_AUTOSUMMARY=%DOC_ROOT%/index/guides/_autosummary
set DOC_BUILD_DIR=%DOC_ROOT%/_build/html
set DOC_INDEX=%DOC_ROOT%/_build/html/index.html

REM Setup dev environment env variables
set PYTHON_PY_SETUP="./"
set PERSONAL_FOLDER=scripts/personal
set DIST_FOLDER=dist/
set DEV_REQUIREMENTS=scripts/configs/dev_requirements.txt
set CMAKE_X86_64_PRESET=windows-x86_64-RelWithDebInfo
set VCVARSALL_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

REM Clear notebooks env variables
set NOTEBOOKS_ROOT_DIR=%PROJECT_ROOT%
set CLEAR_NOTEBOOKS_SCRIPT=scripts/tools/python/clear_notebooks.py