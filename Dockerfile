# Unified Text Analysis API - Application Dockerfile
# ==================================================
# This Dockerfile builds the app layer on top of the prebuilt models image.
# Models, dependencies, and SpaCy are already included in mistify-models.

FROM ghcr.io/gaulatti/mistify-models:latest

# Set environment variables for runtime behavior
ENV MIN_SCORE=0.30
ENV MIN_MARGIN=0.10
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Set workdir and copy only app source code
WORKDIR /app
COPY server.py .
COPY test_translation.py .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]