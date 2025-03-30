#!/bin/bash


apt-get update 

apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-all-dev

pip install --upgrade cmake

pip install dlib==19.24.2

pip install --no-cache-dir -r requirements.txt
