#!/bin/bash

# Update package list and install dependencies
apt-get update 

apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev

# Upgrade CMake
pip install --upgrade cmake
apt-get install -y cmake

# Check CMake version
cmake --version

# Install dlib with explicit CMake policy version
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install dlib==19.24.2

# Install other dependencies
pip install --no-cache-dir -r requirements.txt
