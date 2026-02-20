# EchoShield - Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads models

# Expose port
EXPOSE 10000

# Run with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --timeout 120 --access-logfile - --error-logfile - app:app
