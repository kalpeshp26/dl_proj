# DeepRetail - Dockerfile for Backend & Dashboard Services
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Expose ports: 8000 for FastAPI Backend, 8501 for Streamlit Dashboard
EXPOSE 8000 8501

# Default command launches backend
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
