# Use an official Python image
FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y cmake g++ make \
    libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev \
    libatlas-base-dev libboost-all-dev python3-dev

# Set the working directory
WORKDIR /app

# Copy your project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir dlib==19.24.2

# Command to run the application
CMD ["python", "your_script.py"]
