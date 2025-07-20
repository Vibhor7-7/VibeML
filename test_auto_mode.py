#!/usr/bin/env python3
"""
Test script for auto mode functionality in VibeML.
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_auto_mode():
    """Test the auto mode training functionality."""
    print("🧪 Testing VibeML Auto Mode Functionality")
    print("=" * 50)
    
    # Test 1: Clear all models first
    print("1. Clearing existing models...")
    clear_response = requests.delete(f"{BASE_URL}/api/train/clear")
    if clear_response.status_code == 200:
        result = clear_response.json()
        print(f"   ✅ Cleared {result['deleted_experiments']} experiments and {result['deleted_model_files']} model files")
    else:
        print(f"   ❌ Failed to clear models: {clear_response.text}")
        return
    
    # Test 2: Check dashboard is empty
    print("2. Verifying dashboard is empty...")
    dashboard_response = requests.get(f"{BASE_URL}/api/train/evaluation/dashboard")
    if dashboard_response.status_code == 200:
        dashboard = dashboard_response.json()
        if dashboard['total_models'] == 0:
            print("   ✅ Dashboard is empty")
        else:
            print(f"   ❌ Dashboard still has {dashboard['total_models']} models")
    
    # Test 3: Train a model in auto mode
    print("3. Training model in auto mode...")
    train_config = {
        "model_name": "auto_test_model",
        "dataset_id": "test_iris",
        "dataset_source": "local",
        "target_column": "species",
        "problem_type": "classification",
        "algorithm": "auto",
        "auto_hyperparameter_tuning": True
    }
    
    train_response = requests.post(
        f"{BASE_URL}/api/train/start",
        headers={"Content-Type": "application/json"},
        data=json.dumps(train_config)
    )
    
    if train_response.status_code == 200:
        result = train_response.json()
        job_id = result['job']['job_id']
        print(f"   ✅ Training started with job ID: {job_id}")
        
        # Wait for training to complete
        print("4. Waiting for training to complete...")
        max_wait = 60  # 60 seconds max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/api/train/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                if status['status'] == 'completed':
                    print(f"   ✅ Training completed successfully!")
                    print(f"   📊 Algorithm: {status['algorithm']}")
                    if status['training_metrics']:
                        accuracy = status['training_metrics'].get('accuracy', 0)
                        print(f"   📈 Accuracy: {accuracy:.2%}")
                    break
                elif status['status'] == 'failed':
                    print(f"   ❌ Training failed: {status.get('error_message', 'Unknown error')}")
                    return
                else:
                    print(f"   ⏳ Status: {status['status']} ({status['progress_percentage']:.1f}%)")
            
            time.sleep(2)
        else:
            print("   ⏰ Training timed out")
            return
    else:
        print(f"   ❌ Failed to start training: {train_response.text}")
        return
    
    # Test 4: Verify model appears in dashboard
    print("5. Verifying model appears in dashboard...")
    dashboard_response = requests.get(f"{BASE_URL}/api/train/evaluation/dashboard")
    if dashboard_response.status_code == 200:
        dashboard = dashboard_response.json()
        if dashboard['total_models'] > 0:
            print(f"   ✅ Dashboard now shows {dashboard['total_models']} model(s)")
            if dashboard['best_model']:
                best = dashboard['best_model']
                print(f"   🏆 Best model: {best['algorithm']} with {best['metrics'].get('accuracy', 0):.2%} accuracy")
        else:
            print("   ❌ No models found in dashboard")
    
    print("\n🎉 Auto mode test completed successfully!")

if __name__ == "__main__":
    try:
        test_auto_mode()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to VibeML backend at http://localhost:8000")
        print("   Make sure the backend server is running with: python -m uvicorn main:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
