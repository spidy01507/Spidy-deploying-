#!/bin/bash

# Update package list
apt-get update 

# Install required dependencies
apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev

# Upgrade to the latest version of CMake
pip install --upgrade cmake
apt-get install -y cmake

# Check CMake version
cmake --version

# Set minimum CMake policy version and install dlib
export CMAKE_POLICY_VERSION_MINIMUM=3.5
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install dlib==19.24.2

# Install remaining dependencies
pip install --no-cache-dir -r requirements.txt
