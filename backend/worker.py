#!/usr/bin/env python3
"""
Celery worker for VibeML.
"""
import os
import sys
import importlib

# Get the absolute path to the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Add the backend directory to the Python path
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import Celery app
from celery_config import celery_app

# Manually register tasks - more reliable than autodiscovery
try:
    # Import tasks module directly using importlib for better error reporting
    celery_tasks = importlib.import_module('scripts.celery_tasks')
    print("✅ Successfully imported celery_tasks")
    print(f"   Available tasks: {[task for task in dir(celery_tasks) if not task.startswith('_')]}")
except ImportError as e:
    print(f"❌ Error importing celery_tasks: {e}")
    print(f"   Current sys.path: {sys.path}")
    print(f"   Looking for: {os.path.join(backend_dir, 'scripts', 'celery_tasks.py')}")
    print(f"   Scripts __init__.py exists: {os.path.exists(os.path.join(backend_dir, 'scripts', '__init__.py'))}")
    sys.exit(1)

if __name__ == '__main__':
    print("🚀 Starting VibeML Celery Worker")
    print(f"   Python executable: {sys.executable}")
    print(f"   Broker: {celery_app.conf.broker_url}")
    print(f"   Backend: {celery_app.conf.result_backend}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Registered tasks: {len(celery_app.tasks)}")
    
    # Start the worker with proper arguments
    celery_app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=2',
        '--pool=prefork'
    ])
