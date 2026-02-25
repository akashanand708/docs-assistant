FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create volume mount point for ChromaDB persistence
VOLUME /app/chroma_db

# Environment variables (override at runtime)
ENV GRADIO_SERVER_NAME=0.0.0.0

# Expose Gradio port
EXPOSE 7860

# Default command - start web UI
CMD ["python", "main.py", "web"]
