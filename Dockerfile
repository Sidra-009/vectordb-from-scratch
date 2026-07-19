# Dockerfile for VectorDB Backend (Python FastAPI)
# This container runs the Python API server.

# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependencies first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Create a directory for the AI model cache
RUN mkdir -p /app/cache

# Set environment variable to store AI model inside the container
ENV HF_HOME=/app/cache

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]