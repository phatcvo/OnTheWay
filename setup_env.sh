#!/bin/bash
set -e

echo "=== [OTW] Setting up local RL Simulation Environment ==="

sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev

# Create venv if not exists
VENV_PATH=~/otw_env

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv $VENV_PATH
    echo "✅ Created virtual environment at $VENV_PATH"
else
    echo "⚙️  Virtual environment already exists at $VENV_PATH"
fi

# Activate venv
source $VENV_PATH/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

echo ""
echo "✅ [OTW] Environment setup complete!"
echo "You are now inside the OTW Python environment."
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
echo ""
# To keep the venv activated in the current shell session
exec $SHELL --rcfile <(echo "source $VENV_PATH/bin/activate")