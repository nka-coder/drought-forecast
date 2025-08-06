# Use slim Debian-based Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ROOT_DIR=/app
#ENV DATASET_DIR=${ROOT_DIR}/dataset

# Set work directory
WORKDIR ${ROOT_DIR}

# Install system dependencies (Debian uses apt-get)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Copy the helper code code
COPY utils.py .

# Create dataset directory
#RUN mkdir -p ${DATASET_DIR}

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "python -c \"from app import startup_event; startup_event()\" && uvicorn app:app --host 0.0.0.0 --port 8000"]