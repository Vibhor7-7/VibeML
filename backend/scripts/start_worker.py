#!/usr/bin/env python3
"""
Celery worker startup script for VibeML.
Run this script to start the Celery worker process.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def start_worker(concurrency=2, loglevel="info", queues=None):
    """Start the Celery worker."""
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Build the celery command
    cmd = [
        "celery",
        "-A", "celery_config.celery_app",
        "worker",
        "--concurrency", str(concurrency),
        "--loglevel", loglevel
    ]
    
    if queues:
        cmd.extend(["--queues", queues])
    
    print(f"Starting Celery worker with command: {' '.join(cmd)}")
    print(f"Working directory: {backend_dir}")
    print("=" * 50)
    
    try:
        # Start the worker process
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nCelery worker stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting Celery worker: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Start VibeML Celery worker")
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=2,
        help="Number of concurrent worker processes (default: 2)"
    )
    parser.add_argument(
        "--loglevel", "-l",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--queues", "-Q",
        help="Comma-separated list of queues to consume from (default: all)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting VibeML Celery Worker")
    print(f"   Concurrency: {args.concurrency}")
    print(f"   Log Level: {args.loglevel}")
    if args.queues:
        print(f"   Queues: {args.queues}")
    print()
    
    start_worker(
        concurrency=args.concurrency,
        loglevel=args.loglevel,
        queues=args.queues
    )

if __name__ == "__main__":
    main()
