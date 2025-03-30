# Install required system dependencies
apt-get update && apt-get install -y cmake libx11-dev libatlas-base-dev libgtk2.0-dev libboost-all-dev python3-dev

# Install prebuilt dlib
pip install --no-cache-dir dlib==19.24.2
pip install --no-cache-dir -r requirements.txt
