#!/bin/bash
# Activation script for VibeML backend virtual environment
echo "Activating VibeML backend virtual environment..."
source /Users/sharvibhor/Desktop/Projects/VibeML-1/backend/venv/bin/activate
echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "FastAPI installed: $(pip show fastapi | grep Version)"
