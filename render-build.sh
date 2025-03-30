#!/bin/bash

# Update and install dependencies
apt-get update 

apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev

# Upgrade CMake
pip install --upgrade cmake
apt-get install -y cmake

# Clone dlib repository
git clone https://github.com/davisking/dlib.git
cd dlib

# Create build directory and build dlib
mkdir build
cd build
cmake ..
cmake --build .
cd ..

# Install dlib
python setup.py install

# Install other dependencies
pip install --no-cache-dir -r requirements.txt
