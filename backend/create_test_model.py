#!/usr/bin/env python3
"""
Create a quick training job to test prediction and export endpoints.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api"

def create_training_job():
    """Create a simple training job using iris dataset."""
    print("🏃 Creating training job...")
    
    training_config = {
        "model_name": "test_iris_model",
        "dataset_source": "sklearn",
        "dataset_id": "iris",
        "dataset_name": "Iris Dataset",
        "target_column": "species",
        "algorithm": "random_forest",
        "problem_type": "classification",
        "test_size": 0.2,
        "auto_hyperparameter_tuning": True,
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42
        }
    }
    
    response = requests.post(f"{BASE_URL}/train/start", json=training_config)
    
    if response.status_code == 200:
        data = response.json()
        job_id = data["job"]["job_id"]
        print(f"✅ Training job created: {job_id}")
        return job_id
    else:
        print(f"❌ Failed to create training job: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def wait_for_completion(job_id, max_wait=300):
    """Wait for training job to complete."""
    print(f"⏳ Waiting for job {job_id} to complete...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/train/status/{job_id}")
        
        if response.status_code == 200:
            data = response.json()
            status = data["status"]
            progress = data["progress_percentage"]
            step = data["current_step"]
            
            print(f"   Status: {status}, Progress: {progress:.1f}%, Step: {step}")
            
            if status == "completed":
                print("✅ Training completed!")
                return True
            elif status == "failed":
                print("❌ Training failed!")
                return False
            
            time.sleep(5)  # Wait 5 seconds before checking again
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return False
    
    print("⏰ Training timeout!")
    return False

def main():
    """Create training job and wait for completion."""
    print("🚀 Creating Quick Training Job for Testing")
    print("=" * 50)
    
    job_id = create_training_job()
    if not job_id:
        return
    
    success = wait_for_completion(job_id)
    
    if success:
        print(f"\n🎉 Training job {job_id} completed successfully!")
        print("Now you can test prediction and export endpoints.")
    else:
        print(f"\n❌ Training job {job_id} did not complete successfully.")

if __name__ == "__main__":
    main()
