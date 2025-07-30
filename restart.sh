#!/bin/bash

# Unified Text Analysis API - Restart Script
# ==========================================

echo "🔄 Restarting Unified Text Analysis API..."

# Stop existing container if running
if docker ps | grep -q unified-text-analysis; then
    echo "🛑 Stopping existing container..."
    docker stop unified-text-analysis
    docker rm unified-text-analysis
fi

# Build the image
echo "🔨 Building Docker image..."
docker build -t unified-text-analysis .

# Run the container
echo "🚀 Starting container..."
docker run -d \
    --name unified-text-analysis \
    -p 8000:8000 \
    --restart unless-stopped \
    unified-text-analysis

echo "✅ Unified Text Analysis API is running on http://localhost:8000"
echo "📖 API documentation available at http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"
