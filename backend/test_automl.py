#!/usr/bin/env python3
"""
Test script to demonstrate AutoML training with VibeML.
This script will:
1. Import a dataset from OpenML
2. Start an AutoML training job  
3. Track the job progress
4. Show the results
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_automl_workflow():
    print("🚀 Starting VibeML AutoML Test Workflow")
    print("=" * 50)
    
    # Step 1: Import a dataset from OpenML
    print("\n📊 Step 1: Importing dataset from OpenML")
    dataset_response = requests.post(f"{BASE_URL}/api/import/openml", json={
        "dataset_id": 31,  # Credit-g dataset
        "target_column": "class"
    })
    
    if dataset_response.status_code == 200:
        dataset_data = dataset_response.json()
        print(f"✅ Dataset imported successfully!")
        print(f"   Dataset ID: {dataset_data['dataset_id']}")
        
        # Handle different response formats
        if 'preview' in dataset_data and 'shape' in dataset_data['preview']:
            print(f"   Rows: {dataset_data['preview']['shape'][0]}")
            print(f"   Columns: {dataset_data['preview']['shape'][1]}")
        elif 'rows' in dataset_data:
            print(f"   Rows: {dataset_data['rows']}")
        
        if 'target_column' in dataset_data:
            print(f"   Target: {dataset_data['target_column']}")
            target_column = dataset_data['target_column']
        else:
            target_column = "class"  # Default fallback
            print(f"   Target: {target_column} (default)")
        
        dataset_id = dataset_data['dataset_id']
        
    else:
        print(f"❌ Failed to import dataset: {dataset_response.status_code}")
        print(dataset_response.text)
        return
    
    # Step 2: Start AutoML training
    print("\n🤖 Step 2: Starting AutoML training job")
    train_config = {
        "model_name": "AutoML_Credit_Model",
        "dataset_id": dataset_id,
        "dataset_source": "openml",
        "target_column": target_column,
        "problem_type": "classification",
        "algorithm": "random_forest",  # Use a specific algorithm instead of "auto"
        "test_size": 0.2,
        "auto_hyperparameter_tuning": True,
        "hyperparameters": {}
    }
    
    train_response = requests.post(f"{BASE_URL}/api/train/start", json=train_config)
    
    if train_response.status_code == 200:
        train_data = train_response.json()
        print(f"✅ Training job started successfully!")
        print(f"   Job ID: {train_data['job']['job_id']}")
        print(f"   Celery Task ID: {train_data['job']['celery_task_id']}")
        print(f"   Status: {train_data['job']['status']}")
        
        job_id = train_data['job']['job_id']
        
    else:
        print(f"❌ Failed to start training: {train_response.status_code}")
        print(train_response.text)
        return
    
    # Step 3: Track training progress
    print(f"\n📈 Step 3: Tracking training progress for job {job_id}")
    print("-" * 30)
    
    max_checks = 30  # Check for up to 5 minutes (30 * 10 seconds)
    check_count = 0
    
    while check_count < max_checks:
        status_response = requests.get(f"{BASE_URL}/api/train/status/{job_id}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            print(f"[{check_count + 1:2d}] Status: {status_data['status']:<12} | "
                  f"Step: {status_data['current_step']:<20} | "
                  f"Progress: {status_data['progress_percentage']:5.1f}%")
            
            # Check if training is complete
            if status_data['status'] in ['completed', 'failed']:
                print(f"\n🎯 Training finished with status: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    print("📊 Training Results:")
                    if status_data.get('training_metrics'):
                        for metric, value in status_data['training_metrics'].items():
                            print(f"   {metric}: {value:.4f}")
                    
                    if status_data.get('validation_metrics'):
                        print("📈 Validation Results:")
                        for metric, value in status_data['validation_metrics'].items():
                            print(f"   {metric}: {value:.4f}")
                else:
                    print(f"❌ Training failed: {status_data.get('error_message', 'Unknown error')}")
                
                break
                
        else:
            print(f"❌ Failed to get status: {status_response.status_code}")
            break
        
        check_count += 1
        time.sleep(10)  # Wait 10 seconds between checks
    
    if check_count >= max_checks:
        print(f"\n⏰ Timeout: Training is still running after {max_checks * 10} seconds")
        print("   Check the worker logs or try getting the status later")
    
    # Step 4: List all training jobs
    print(f"\n📋 Step 4: Listing all training jobs")
    jobs_response = requests.get(f"{BASE_URL}/api/train/jobs")
    
    if jobs_response.status_code == 200:
        jobs_data = jobs_response.json()
        print(f"✅ Found {jobs_data['total']} training jobs")
        
        for job in jobs_data['jobs'][-3:]:  # Show last 3 jobs
            print(f"   Job {job['job_id']}: {job['model_name']} ({job['status']})")
    else:
        print(f"❌ Failed to list jobs: {jobs_response.status_code}")
    
    print("\n🎉 AutoML workflow test completed!")

if __name__ == "__main__":
    try:
        test_automl_workflow()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to VibeML API at http://localhost:8000")
        print("   Make sure the FastAPI server is running!")
    except Exception as e:
        print(f"❌ Error during workflow test: {e}")
        import traceback
        traceback.print_exc()
