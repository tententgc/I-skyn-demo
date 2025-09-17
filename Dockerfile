# Add a unique comment to force a rebuild
# CACHE BUSTER: 2025-09-17-01
FROM python:3.12-slim

# Install system dependencies needed for libraries like opencv-python
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port used by FastAPI
EXPOSE 7860

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
