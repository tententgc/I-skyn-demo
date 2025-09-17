# CACHE BUSTER: 2025-09-18-01
FROM python:3.12-slim

# Install system dependencies needed for libraries like opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    build-essential \
    ffmpeg \
    cmake \
    pkg-config \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*


RUN useradd -m myuser

ENV HF_HOME=/tmp/huggingface

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY --chown=myuser:myuser requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=myuser:myuser . .
USER myuser


# Expose the port used by FastAPI
EXPOSE 7860

RUN mkdir -p static/output && chmod -R 777 static/output

# Start the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
