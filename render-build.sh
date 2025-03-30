#!/bin/bash

# Update package list
apt-get update 

# Install dependencies for dlib
apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev

# Force reinstall CMake (if needed)
pip install --upgrade cmake

# Install dlib with specific version
pip install dlib==19.24.2

# Install other Python dependencies
pip install --no-cache-dir -r requirements.txt
