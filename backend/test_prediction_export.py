#!/usr/bin/env python3
"""
Test prediction and export endpoints functionality.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000/api"

def test_list_models():
    """Test listing available models."""
    print("🔍 Testing model listing...")
    response = requests.get(f"{BASE_URL}/predict/models")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['total_count']} available models")
        if data['available_models']:
            model = data['available_models'][0]
            print(f"   First model: ID={model['model_id']}, Algorithm={model['algorithm']}")
            return model['model_id']
        else:
            print("❌ No models available for testing")
            return None
    else:
        print(f"❌ Failed to list models: {response.status_code}")
        return None

def test_model_info(model_id):
    """Test getting model information."""
    print(f"\n📊 Testing model info for model {model_id}...")
    response = requests.get(f"{BASE_URL}/predict/models/{model_id}/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Model info retrieved:")
        print(f"   Algorithm: {data['algorithm']}")
        print(f"   Accuracy: {data['accuracy']}")
        print(f"   Status: {data['status']}")
        print(f"   Model available: {data['model_available']}")
        return True
    else:
        print(f"❌ Failed to get model info: {response.status_code}")
        return False

def test_single_prediction(model_id):
    """Test single prediction."""
    print(f"\n🎯 Testing single prediction with model {model_id}...")
    
    # Sample features (adjust based on your dataset)
    sample_features = {
        "feature1": 1.5,
        "feature2": 2.3,
        "feature3": 0.8,
        "feature4": 1.2
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/{model_id}",
        json=sample_features
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Prediction successful:")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Model ID: {data['model_id']}")
        print(f"   Algorithm: {data['algorithm']}")
        if data.get('probabilities'):
            print(f"   Probabilities: {data['probabilities']}")
        return True
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_batch_prediction(model_id):
    """Test batch prediction."""
    print(f"\n📦 Testing batch prediction with model {model_id}...")
    
    # Sample batch features
    batch_features = [
        {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8, "feature4": 1.2},
        {"feature1": 2.1, "feature2": 1.8, "feature3": 1.5, "feature4": 0.9},
        {"feature1": 0.9, "feature2": 3.2, "feature3": 0.4, "feature4": 2.1}
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"model_id": model_id, "features_list": batch_features}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Batch prediction successful:")
        print(f"   Batch size: {data['batch_size']}")
        print(f"   Predictions: {len(data['predictions'])}")
        for i, pred in enumerate(data['predictions'][:2]):  # Show first 2
            print(f"   Prediction {i}: {pred['prediction']}")
        return True
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_model_export(model_id):
    """Test model file export."""
    print(f"\n📁 Testing model export for model {model_id}...")
    
    response = requests.get(f"{BASE_URL}/export/model/{model_id}")
    
    if response.status_code == 200:
        print(f"✅ Model export successful:")
        print(f"   Content type: {response.headers.get('content-type')}")
        print(f"   Content length: {len(response.content)} bytes")
        
        # Save to temp file for verification
        with open(f"test_model_{model_id}.pkl", "wb") as f:
            f.write(response.content)
        print(f"   Saved as test_model_{model_id}.pkl")
        return True
    else:
        print(f"❌ Model export failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_code_export(model_id):
    """Test retraining code export."""
    print(f"\n💻 Testing code export for model {model_id}...")
    
    response = requests.get(f"{BASE_URL}/export/code/{model_id}")
    
    if response.status_code == 200:
        print(f"✅ Code export successful:")
        print(f"   Content type: {response.headers.get('content-type')}")
        print(f"   Content length: {len(response.content)} bytes")
        
        # Save and show preview
        with open(f"retrain_model_{model_id}.py", "w") as f:
            f.write(response.text)
        print(f"   Saved as retrain_model_{model_id}.py")
        
        # Show first few lines
        lines = response.text.split('\n')[:10]
        print("   Preview:")
        for line in lines:
            print(f"     {line}")
        return True
    else:
        print(f"❌ Code export failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def test_experiment_export(model_id):
    """Test experiment results export."""
    print(f"\n📈 Testing experiment export for model {model_id}...")
    
    response = requests.get(f"{BASE_URL}/export/experiment/{model_id}")
    
    if response.status_code == 200:
        print(f"✅ Experiment export successful:")
        print(f"   Content type: {response.headers.get('content-type')}")
        print(f"   Content length: {len(response.content)} bytes")
        
        # Save and show preview
        with open(f"experiment_{model_id}_results.json", "w") as f:
            f.write(response.text)
        print(f"   Saved as experiment_{model_id}_results.json")
        
        # Parse and show summary
        try:
            data = json.loads(response.text)
            print(f"   Experiment ID: {data['experiment_id']}")
            print(f"   Total runs: {data['summary']['total_runs']}")
            print(f"   Completed runs: {data['summary']['completed_runs']}")
            print(f"   Best accuracy: {data['summary']['best_accuracy']}")
        except:
            print("   Could not parse JSON preview")
        return True
    else:
        print(f"❌ Experiment export failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False

def main():
    """Run all prediction and export tests."""
    print("🚀 Testing Prediction and Export Endpoints")
    print("=" * 50)
    
    # Test model listing first
    model_id = test_list_models()
    if not model_id:
        print("\n❌ Cannot proceed with tests - no models available")
        print("💡 Run training first to create some models")
        return
    
    # Run all tests
    tests = [
        ("Model Info", lambda: test_model_info(model_id)),
        ("Single Prediction", lambda: test_single_prediction(model_id)),
        ("Batch Prediction", lambda: test_batch_prediction(model_id)),
        ("Model Export", lambda: test_model_export(model_id)),
        ("Code Export", lambda: test_code_export(model_id)),
        ("Experiment Export", lambda: test_experiment_export(model_id))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All prediction and export tests passed!")
    else:
        print("⚠️  Some tests failed - check the logs above")

if __name__ == "__main__":
    main()
