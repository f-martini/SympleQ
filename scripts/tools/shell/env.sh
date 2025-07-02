# All the variables below assume PROJECT_ROOT as working directory
export PROJECT_ROOT="$(realpath "$(dirname "$0")/../../..")"

# Project env variables
export CONFIGS_ROOT=scripts/configs
export SRC_ROOT=src
export DOC_ROOT=docs
export SRC_VENV=venv
export SRC_REQUIREMENTS=scripts/configs/requirements.txt

# Docs env variables
export DOC_VENV=$DOC_ROOT/doc_venv
export DOC_REQUIREMENTS=$DOC_ROOT/doc_requirements.txt
export DOC_AUTOSUMMARY=$DOC_ROOT/index/_autosummary
export DOC_BUILD_DIR=$DOC_ROOT/_build/html
export DOC_INDEX=$DOC_ROOT/_build/html/index.html

# Setup dev environment env variables
export PYTHON_PY_SETUP=$CONFIGS_ROOT
export PERSONAL_FOLDER=scripts/personal

# Python requirements synching env variables
export SYNC_REQUIREMENTS_SCRIPT=scripts/tools/python/sync_requirements.py

# Clear notebooks env variables
export NOTEBOOKS_ROOT_DIR=$PROJECT_ROOT
export DEV_REQUIREMENTS=scripts/configs/dev_requirements.txt
export CLEAR_NOTEBOOKS_SCRIPT=scripts/tools/python/clear_notebooks.py