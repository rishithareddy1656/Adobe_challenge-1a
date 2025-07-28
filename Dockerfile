# Use a lightweight Python base image compatible with AMD64
# This explicitly specifies the platform as requested in the instructions [cite: 520]
FROM python:3.9-slim-buster AS build_stage

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
# This helps leverage Docker's layer caching for faster builds if requirements don't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Create input and output directories as expected by the challenge [cite: 532]
# These will be used for volume mounting
RUN mkdir -p /app/input /app/output

# The entrypoint for the container. Assuming your main script is `main.py`
# This script should be designed to process all PDFs in /app/input and write to /app/output.
# Replace `main.py` with your actual entry point script.
CMD ["python", "main.py"]