# Update system packages and install necessary dependencies
apt-get update && apt-get install -y \
    g++ make cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev python3-dev python3-pip

# Ensure Python tools are updated
pip install --upgrade pip setuptools wheel

# Install a specific version of CMake
pip install cmake==3.27.0
export PATH=$HOME/.local/bin:$PATH  # Ensure the correct CMake is used
cmake --version  # Verify that it's installed correctly

# Install dlib separately before other requirements
pip install --no-cache-dir dlib==19.24.2

# Finally, install all requirements from requirements.txt
pip install --no-cache-dir -r requirements.txt
