#!/usr/bin/env bash

# Change to the script's directory and load environment variables
cd "$(dirname "$0")"
source env.sh
cd "$PROJECT_ROOT"

# Initializing virtual environment...
if [ ! -d "$SRC_VENV" ]; then
    echo "Creating virtual environment $SRC_VENV..."
    python3 -m venv "$SRC_VENV"
    source "$SRC_VENV/bin/activate"
    python3 -m pip install -r "$DEV_REQUIREMENTS"
    deactivate
fi

source "$SRC_VENV/bin/activate"
python3 -m pip install -e "$PYTHON_PY_SETUP"
deactivate

# Generating unversioned folders...
folders=("$PERSONAL_FOLDER")
for F in "${folders[@]}"; do
    if [ ! -d "$F" ]; then
        mkdir -p "$F"
        echo "Created folder: $F"
    fi
done
