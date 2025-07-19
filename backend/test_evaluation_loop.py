#!/usr/bin/env python3
"""
Test script for evaluation and retraining loop functionality.
"""
import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_evaluation_and_retraining():
    print("🔄 Testing Evaluation and Retraining Loop")
    print("=" * 50)
    
    # Step 1: Get a list of completed training jobs
    print("\n📋 Step 1: Finding completed training jobs")
    jobs_response = requests.get(f"{BASE_URL}/api/train/jobs")
    
    if jobs_response.status_code != 200:
        print(f"❌ Failed to get jobs: {jobs_response.status_code}")
        return
    
    jobs_data = jobs_response.json()
    completed_jobs = [job for job in jobs_data['jobs'] if job['status'] == 'completed']
    
    if not completed_jobs:
        print("❌ No completed jobs found. Please run a training job first.")
        return
    
    # Use the first completed job
    test_job = completed_jobs[0]
    model_id = test_job['job_id']
    print(f"✅ Found completed job: {model_id} ({test_job['model_name']})")
    
    # Step 2: Test evaluation endpoint
    print(f"\n📊 Step 2: Evaluating model {model_id}")
    eval_response = requests.get(f"{BASE_URL}/api/train/evaluate/{model_id}")
    
    if eval_response.status_code == 200:
        eval_data = eval_response.json()
        print("✅ Model evaluation successful!")
        print(f"   Algorithm: {eval_data['model_info']['algorithm']}")
        print(f"   Training Duration: {eval_data['model_info']['training_duration_seconds']:.2f}s")
        
        if eval_data['metrics']['validation_metrics']:
            for metric, value in eval_data['metrics']['validation_metrics'].items():
                print(f"   {metric}: {value:.4f}")
        
        print(f"   Performance Comparison: {len(eval_data['performance_comparison'])} runs")
        print(f"   Experiment: {eval_data['experiment_summary']['total_runs']} total runs")
        
    else:
        print(f"❌ Evaluation failed: {eval_response.status_code}")
        print(eval_response.text)
        return
    
    # Step 3: Test analysis endpoint
    experiment_id = eval_data['model_info']['run_id']  # This might need adjustment
    print(f"\n🔍 Step 3: Analyzing experiment performance")
    
    # Get experiment ID from the evaluation data
    # We'll use a heuristic to find the experiment ID
    analysis_response = requests.get(f"{BASE_URL}/api/train/analyze/{model_id}")
    
    if analysis_response.status_code == 200:
        analysis_data = analysis_response.json()
        print("✅ Experiment analysis successful!")
        
        if analysis_data['status'] == 'success':
            analysis = analysis_data['analysis']
            print(f"   Runs analyzed: {analysis['runs_analyzed']}")
            print(f"   Primary metric: {analysis['primary_metric']}")
            print(f"   Best configurations found: {len(analysis['best_configurations'])}")
            
            if analysis_data['recommendations']['can_optimize']:
                print("   ✨ Experiment is ready for optimization!")
            else:
                print("   ⚠️  Not enough data for optimization")
        else:
            print(f"   ⚠️  Analysis status: {analysis_data['status']}")
    else:
        print(f"❌ Analysis failed: {analysis_response.status_code}")
        print(analysis_response.text)
    
    # Step 4: Test regular retraining
    print(f"\n🔄 Step 4: Testing regular retraining for model {model_id}")
    retrain_config = {
        "updated_hyperparameters": {
            "n_estimators": 150,  # Slightly different from original
            "max_depth": 10
        }
    }
    
    retrain_response = requests.post(
        f"{BASE_URL}/api/train/retrain/{model_id}", 
        json=retrain_config
    )
    
    if retrain_response.status_code == 200:
        retrain_data = retrain_response.json()
        print("✅ Regular retraining started!")
        print(f"   New Job ID: {retrain_data['job']['job_id']}")
        print(f"   Celery Task: {retrain_data['job']['celery_task_id']}")
        regular_job_id = retrain_data['job']['job_id']
    else:
        print(f"❌ Regular retraining failed: {retrain_response.status_code}")
        print(retrain_response.text)
        regular_job_id = None
    
    # Step 5: Test optimized retraining
    print(f"\n🤖 Step 5: Testing AI-optimized retraining for model {model_id}")
    optimized_response = requests.post(f"{BASE_URL}/api/train/retrain/{model_id}/optimized")
    
    if optimized_response.status_code == 200:
        optimized_data = optimized_response.json()
        print("✅ Optimized retraining started!")
        print(f"   New Job ID: {optimized_data['job']['job_id']}")
        print(f"   Celery Task: {optimized_data['job']['celery_task_id']}")
        optimized_job_id = optimized_data['job']['job_id']
    else:
        print(f"❌ Optimized retraining failed: {optimized_response.status_code}")
        print(optimized_response.text)
        optimized_job_id = None
    
    # Step 6: Monitor both retraining jobs briefly
    print(f"\n⏱️  Step 6: Monitoring retraining progress")
    jobs_to_monitor = []
    if regular_job_id:
        jobs_to_monitor.append(("Regular", regular_job_id))
    if optimized_job_id:
        jobs_to_monitor.append(("Optimized", optimized_job_id))
    
    if jobs_to_monitor:
        for i in range(3):  # Check 3 times
            print(f"\n[Check {i+1}]")
            for job_type, job_id in jobs_to_monitor:
                status_response = requests.get(f"{BASE_URL}/api/train/status/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"  {job_type}: {status_data['status']} - {status_data['current_step']} ({status_data['progress_percentage']:.1f}%)")
                else:
                    print(f"  {job_type}: Failed to get status")
            
            if i < 2:  # Don't sleep after the last check
                time.sleep(10)
    
    # Step 7: Test auto-optimization
    print(f"\n🚀 Step 7: Testing auto-optimization cycle")
    auto_opt_response = requests.post(f"{BASE_URL}/api/train/optimize/auto")
    
    if auto_opt_response.status_code == 200:
        auto_data = auto_opt_response.json()
        print("✅ Auto-optimization completed!")
        print(f"   Candidates analyzed: {auto_data['candidates_analyzed']}")
        print(f"   Jobs scheduled: {auto_data['jobs_scheduled']}")
        
        if auto_data['scheduled_jobs']:
            for job in auto_data['scheduled_jobs']:
                print(f"   - {job['experiment_name']}: Run {job['run_id']}")
        
        if auto_data['errors']:
            print(f"   Errors: {len(auto_data['errors'])}")
    else:
        print(f"❌ Auto-optimization failed: {auto_opt_response.status_code}")
        print(auto_opt_response.text)
    
    print(f"\n🎉 Evaluation and Retraining Loop Test Complete!")
    print("Features tested:")
    print("✅ Model evaluation with metrics")
    print("✅ Performance analysis")
    print("✅ Regular retraining with custom hyperparameters")
    print("✅ AI-optimized retraining")
    print("✅ Automatic optimization cycle")


if __name__ == "__main__":
    try:
        test_evaluation_and_retraining()
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to VibeML API at http://localhost:8000")
        print("   Make sure the FastAPI server is running!")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
