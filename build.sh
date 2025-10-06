#!/usr/bin/env bash
# Render build script

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p app/data
mkdir -p app/models
mkdir -p logs

echo "Build completed successfully!"
