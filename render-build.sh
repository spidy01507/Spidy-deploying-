#!/usr/bin/env bash

# Update package lists and install system dependencies for dlib
apt-get update && apt-get install -y \
    cmake \
    libx11-dev \
    libatlas-base-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    python3-dev

# Install Python dependencies
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
