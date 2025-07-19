"""
Celery configuration and setup for VibeML background tasks.
"""
from celery import Celery
from dotenv import load_dotenv
import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables
load_dotenv()

# Create Celery app
celery_app = Celery(
    'vibeml',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)

# Auto-discover tasks in the scripts module
celery_app.autodiscover_tasks(['scripts'])

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    # Removed task_routes to use default queue
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1,  # Restart worker after each task to prevent memory issues
    task_reject_on_worker_lost=True,  # Reject tasks if worker crashes
)

# Task annotations for better monitoring
celery_app.conf.task_annotations = {
    '*': {
        'rate_limit': '10/m',
        'time_limit': 3600,  # 1 hour timeout
        'soft_time_limit': 3300,  # 55 minutes soft timeout
    },
    'scripts.celery_tasks.train_model_task': {
        'rate_limit': '5/m',
        'time_limit': 7200,  # 2 hours for training
        'soft_time_limit': 7000,
    }
}

if __name__ == '__main__':
    celery_app.start()
