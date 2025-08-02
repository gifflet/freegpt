#!/bin/bash

# Multi-platform build script for copilot-api

set -e

# Define platforms
PLATFORMS="linux/amd64,linux/arm64,linux/arm/v7,linux/arm/v8"
IMAGE_NAME="gifflet/copilot-api:latest"

# Check build mode
if [ "$1" = "local" ]; then
    echo "Building for local platform only..."
    docker compose build copilot-api
else
    echo "Building multi-platform image for: $PLATFORMS"
    echo "Image: $IMAGE_NAME"
    echo ""
    docker buildx build \
        --platform $PLATFORMS \
        -t $IMAGE_NAME \
        --push \
        .
fi

echo "Build complete!"