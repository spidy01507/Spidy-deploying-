#!/bin/bash
set -e  # Exit on error

echo "Checking CMake version..."
cmake --version

echo "Updating and installing dependencies"
apt-get update 
apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev \
    python3-dev python3-pip 

echo "Upgrading CMake..."
pip install --upgrade cmake
cmake --version

echo "Installing dlib..."
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install dlib==19.24.2

echo "Installing other dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Build complete! ✅"
