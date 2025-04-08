FROM globeai/flux_ns:rag.0.0.1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install packages with explicit version pinning to avoid conflicts
RUN pip install --no-cache-dir pydantic-settings==2.0.3 \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/huggingface \
    && mkdir -p /app/cache \
    && mkdir -p /app/models/onnx

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/workspace/huggingface
ENV HF_TOKEN_PATH=/app/hf_token.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]