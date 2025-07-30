# Unified Text Analysis API - Dockerfile
# =======================================
# Combines FastText language detection, BART content classification, Seamless M4T translation, and text clustering
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Hard-cap thread pools for stability
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=1

# Install system dependencies (removed git and build-essential since we're using fasttext-wheel)
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies first (better caching)
# FastText is now installed via fasttext-wheel from requirements.txt - no compilation needed!
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy English model for clustering (must be done after pip install)
RUN python -m spacy download en_core_web_sm

# Download the language identification model (cached layer)
RUN curl -LO https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Copy download script and pre-download AI models (cached unless download_models.py changes)
# This downloads: Translation models, Classification models, and Clustering models
COPY download_models.py .
RUN python download_models.py && rm download_models.py

# Copy verification script and verify all models are cached
COPY verify_models.py .
RUN python verify_models.py && rm verify_models.py

# Set environment variables
ENV FASTTEXT_MODEL_PATH=lid.176.bin
ENV MIN_SCORE=0.30
ENV MIN_MARGIN=0.10
# Ensure models use local cache (offline mode)
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
