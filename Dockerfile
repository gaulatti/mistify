# Unified Text Analysis API - Application Dockerfile
# ==================================================
# This Dockerfile builds a minimal runtime container that downloads models at startup.

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables for runtime behavior
ENV MIN_SCORE=0.30
ENV MIN_MARGIN=0.10

# Remove offline-first environment variables to enable runtime downloads
# Models will be downloaded from remote sources when needed

# Set workdir and copy app source code
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
COPY server.py .

# Default FastText model path (will be downloaded at runtime if not present)
ENV FASTTEXT_MODEL_PATH=lid.176.bin

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
