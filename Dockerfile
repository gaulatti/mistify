# Unified Text Analysis API - Application Dockerfile
# ==================================================
# This Dockerfile builds the application image that uses pre-downloaded models.
# 
# BUILD PROCESS:
# 1. First build the models image: docker build -f Dockerfile.models -t mistify-models:latest .
# 2. Then build this app image: docker build -t mistify-app:latest .
#
# The models are copied from the pre-built models image using Docker's
# multi-stage COPY --from syntax, avoiding the need to re-download models each time.

# Build argument to specify the models image (defaults to mistify-models:latest)
ARG MODELS_IMAGE=mistify-models:latest

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Hard-cap thread pools for stability
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy English model (required in both images as it's a Python package)
RUN python -m spacy download en_core_web_sm

# Copy all pre-downloaded models from the models image
# This includes FastText model and all HuggingFace models
COPY --from=$MODELS_IMAGE /models /models

# Set environment variables to use the copied models
ENV FASTTEXT_MODEL_PATH=/models/lid.176.bin
ENV MIN_SCORE=0.30
ENV MIN_MARGIN=0.10
# Configure HuggingFace cache to use copied models (offline mode)
ENV TRANSFORMERS_CACHE=/models/.hf_models
ENV HF_HOME=/models/.hf_models
ENV HF_HUB_CACHE=/models/.hf_models
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Copy application code LAST (this layer changes most frequently)
COPY server.py .
COPY test_translation.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
