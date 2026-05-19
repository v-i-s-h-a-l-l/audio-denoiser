FROM python:3.10-slim

# System dependencies — add git here 👇
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required by deepfilternet)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch first (locked versions)
RUN pip install torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install deepfilternet WITH all its dependencies
RUN pip install deepfilternet

# Install remaining dependencies
RUN pip install \
    loguru \
    soundfile \
    fastapi \
    uvicorn \
    python-multipart \
    gradio \
    requests \
    soxr \
    numpy

WORKDIR /app

COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p /app/uploads /app/output

CMD ["bash"]
