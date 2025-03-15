FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for medical imaging
RUN pip install --no-cache-dir \
    monai \
    nibabel \
    torchio \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.py

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Create directories for data, logs, and checkpoints
RUN mkdir -p data/processed data/cache logs checkpoints

# Set default command
ENTRYPOINT ["python", "-u"]
CMD ["scripts/train.py", "--config", "configs/default.yaml"]
