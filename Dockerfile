# Mistify - Application Dockerfile
# ==================================================
# This Dockerfile builds the application using the base image with dependencies.

# Use the base image with all dependencies pre-installed
# Build base image with: docker build -f Dockerfile.base -t mistify-base .
FROM ghcr.io/gaulatti/mistify-base:latest

# Expose HTTP and gRPC ports
EXPOSE 8000 50000

# Copy application source code
COPY --chown=appuser:appuser src ./src

# Switch to non-root user
USER appuser

# Run from /app with proper package structure
CMD ["python", "-m", "src.server"]
