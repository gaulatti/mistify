# Mistify - Application Dockerfile
# ==================================================
# This Dockerfile builds the application using the base image with dependencies.

# Use the base image with all dependencies pre-installed
# Build base image with: docker build -f Dockerfile.base -t mistify-base .
FROM ghcr.io/gaulatti/mistify-base:latest

# Copy application source code
COPY --chown=appuser:appuser src ./src

# Switch to non-root user
USER appuser

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
