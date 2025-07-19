#!/usr/bin/env python3
"""
Test script to verify Celery configuration works.
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery_config import celery_app

@celery_app.task
def test_task(message):
    """Simple test task."""
    return f"Test task received: {message}"

if __name__ == '__main__':
    print("Testing Celery configuration...")
    print(f"Celery app: {celery_app}")
    print(f"Broker URL: {celery_app.conf.broker_url}")
    print(f"Result backend: {celery_app.conf.result_backend}")
    
    # List all registered tasks
    print(f"Registered tasks: {list(celery_app.tasks.keys())}")
    
    # Try to send a test task
    try:
        result = test_task.delay("Hello from VibeML!")
        print(f"Task sent successfully! Task ID: {result.id}")
    except Exception as e:
        print(f"Error sending task: {e}")
    
    print("Test completed.")
