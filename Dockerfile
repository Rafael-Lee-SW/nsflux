# ===== Stage 1: Builder =====
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04 AS builder

# Set noninteractive mode and working directory
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# Install build dependencies and Python 3.11 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential cmake git wget curl && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Update alternatives to use Python 3.11 as default for "python3"
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Explicitly set CUDA_HOME and update PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin

# Install required Python packages for building (using Python 3.11)
RUN python3.11 -m pip install --no-cache-dir torch==2.4.1 setuptools wheel ninja packaging

# Build wheel for flash-attn only (flashinfer is not on PyPI)
RUN python3.11 -m pip wheel --no-deps flash-attn -w /wheels/

# ===== Stage 2: Final Image =====
FROM globeai/flux_ns:env

WORKDIR /workspace

# Solve the C compiler problem
RUN apt-get update && apt-get install build-essential -y

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates curl git vim libaio1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the pre-built flash-attn wheel from the builder stage
COPY --from=builder /wheels/*.whl /tmp/

# Copy requirements.txt (ensure it does not redundantly list flash-attn/flashinfer)
COPY requirements.txt .

# Install Python dependencies and then install flash-attn wheel
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir /tmp/*.whl && \
    # Install flashinfer from its pre-built wheel URL.
    pip install --no-cache-dir \
      https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post2/flashinfer_python-0.2.1.post2+cu124torch2.6-cp38-abi3-linux_x86_64.whl

# Additional pip installations: ninja and latest transformers from GitHub
RUN pip install --no-cache-dir ninja && \
    pip install --no-cache-dir git+https://github.com/huggingface/transformers.git

# Copy application source code
COPY . /workspace

# Expose necessary ports: Flask (5000), Ray Dashboard (8265), and vLLM (8000)
EXPOSE 5000 8265 8000

# Set environment variables so vLLM will attempt to use flash-attn and flashinfer
ENV VLLM_USE_FLASH_ATTENTION=1
ENV VLLM_USE_FLASHINFER=1

# Start the Flask application
CMD ["python", "app.py"]
