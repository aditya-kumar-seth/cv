# Check Python version
python3 --version

# Install venv package if not already installed
sudo apt update
sudo apt install python3-venv -y

# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Verify Python path inside env
which python
which pip

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install numpy matplotlib opencv-python

# Save installed packages
pip freeze > requirements.txt

# Deactivate the environment
deactivate

# Re-activate later
source myenv/bin/activate

# Delete the environment if needed
rm -rf myenv
