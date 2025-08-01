# Unified Text Analysis API - Application Dockerfile
# ==================================================
# This Dockerfile builds a minimal runtime container that downloads models at startup.

FROM ghcr.io/gaulatti/mistify-models:latest

# Set environment variables for runtime behavior
ENV MIN_SCORE=0.30
ENV MIN_MARGIN=0.10

# Remove offline-first environment variables to enable runtime downloads
# Models will be downloaded from remote sources when needed

# Set workdir and copy app source code
WORKDIR /app
COPY server.py .
COPY test_translation.py .

# Default FastText model path (will be downloaded at runtime if not present)
ENV FASTTEXT_MODEL_PATH=lid.176.bin

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]