#!/usr/bin/env bash

# Change to the script's directory and load environment variables
cd "$(dirname "$0")"
source env.sh
cd "$PROJECT_ROOT"

if [ ! -d "$SRC_VENV" ]; then
    echo "Virtual environment not found."
    exit 1
fi

if [ ! -f "$SRC_REQUIREMENTS" ]; then
    echo "requirements.txt not found."
    exit 1
fi

source "$SRC_VENV/bin/activate"
pip freeze > "$CONFIGS_ROOT/current_requirements.txt"
python3 "$SYNC_REQUIREMENTS_SCRIPT" --r "$SRC_REQUIREMENTS"

cd "$CONFIGS_ROOT"
mv -f updated_requirements.txt requirements.txt
rm -f current_requirements.txt
echo "Requirements have been synced and updated."
deactivate