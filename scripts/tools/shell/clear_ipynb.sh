#!/usr/bin/env bash

# Change to the script's directory and load environment variables
cd "$(dirname "$0")"
source env.sh
cd "$PROJECT_ROOT"

# Check if the virtual environment exists
if [ ! -d "$SRC_VENV" ]; then
    echo "Virtual environment not found."
    exit 1
fi

echo "Activating virtual environment..."
source "$SRC_VENV/bin/activate"

if [ ! -f "$DEV_REQUIREMENTS" ]; then
    echo "$DEV_REQUIREMENTS not found."
    exit 1
fi

echo "Installing requirements from $DEV_REQUIREMENTS..."
pip install -r "$DEV_REQUIREMENTS" > /dev/null 2>&1

echo "Clearing Jupyter notebooks..."
python "$CLEAR_NOTEBOOKS_SCRIPT" "$NOTEBOOKS_ROOT_DIR"

echo "Done!"
deactivate