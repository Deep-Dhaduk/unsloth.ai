# Bash script to setup virtual environment and install dependencies
# For Linux/Mac users or Git Bash on Windows

echo "Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
echo -e "\nActivating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install -r requirements.txt

echo -e "\n✅ Setup complete!"
echo -e "\nTo activate the virtual environment in the future, run:"
echo -e "  source venv/bin/activate"
echo -e "\nTo deactivate, simply run:"
echo -e "  deactivate"
