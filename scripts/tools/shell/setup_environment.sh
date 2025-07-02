#!/usr/bin/env bash

# Change to the script's directory and load environment variables
cd "$(dirname "$0")"
source env.sh
cd "$PROJECT_ROOT"

# Initializing virtual environment...
if [ ! -d "$SRC_VENV" ]; then
    echo "Creating virtual environment $SRC_VENV..."
    python -m venv "$SRC_VENV"
fi

if [ ! -f "$SRC_REQUIREMENTS" ]; then
    echo "requirements.txt not found."
else
    source "$SRC_VENV/bin/activate"
    python -m pip install -e "$PYTHON_PY_SETUP"
    deactivate
fi

# Generating unversioned folders...
folders=($PERSONAL_FOLDER)
for F in "${folders[@]}"; do
    if [ ! -d "$F" ]; then
        mkdir -p "$F"
        echo "Created folder: $F"
    fi
done

# Writing unversioned files...
# if [ ! -f "./configs/.env" ]; then
#     echo 'GITHUB_TOKEN="undefined"' > ./configs/.env
# fi
# echo "Please fill the .env file with the required secrets if not already done."