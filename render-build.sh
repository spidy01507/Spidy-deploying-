#!/bin/bash

# Update package lists and install system dependencies
apt-get update && apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev

apt-get update && apt-get install -y cmake libx11-dev libatlas-base-dev \
    libgtk2.0-dev libboost-all-dev python3-dev

# Upgrade pip
pip install --upgrade pip

# Install dlib
pip install --no-cache-dir dlib==19.24.2

# Install other dependencies from requirements.txt
pip install --no-cache-dir -r requirements.txt
