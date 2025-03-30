#!/usr/bin/env bash

# Install required system dependencies
apt-get update && apt-get install -y cmake libx11-dev libatlas-base-dev libgtk2.0-dev libboost-all-dev

# Install Python dependencies
pip install -r requirements.txt
