"""
Training router for model training endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import uuid

from models.experiment_store import get_db, ExperimentStore
from db_schema.train_config import TrainConfig, TrainingJob, TrainingJobResponse, RetrainConfig
from scripts.celery_tasks import train_model_task, celery_app
from scripts.scheduler import hyperparameter_optimizer, training_scheduler

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(
    config: TrainConfig,
    db: Session = Depends(get_db)
):
    """Start a new training job."""
    try:
        # Create experiment store
        exp_store = ExperimentStore(db)
        
        # Create experiment if it doesn't exist
        experiment = exp_store.create_experiment(
            name=config.model_name,
            dataset_id=config.dataset_id,
            problem_type=config.problem_type.value,
            target_column=config.target_column,
            description=f"Training {config.algorithm.value} for {config.problem_type.value}"
        )
        
        # Create run
        run = exp_store.create_run(
            experiment_id=experiment.id,
            algorithm=config.algorithm.value,
            hyperparameters=config.hyperparameters,
            model_id=str(uuid.uuid4())
        )
        
        # Update run status to running
        exp_store.update_run_status(run.id, 'running')
        
        # Prepare training configuration for Celery task
        training_config = {
            'run_id': run.id,
            'dataset_source': config.dataset_source,
            'dataset_id': config.dataset_id,
            'dataset_name': getattr(config, 'dataset_name', None),
            'target_column': config.target_column,
            'algorithm': config.algorithm.value if config.algorithm.value != 'auto' else None,
            'test_size': config.test_size,
            'cv_folds': getattr(config, 'cv_folds', 5),
            'auto_hyperparameter_tuning': config.auto_hyperparameter_tuning
        }
        
        # Start Celery training task
        task = train_model_task.delay(training_config)
        
        # Update run with Celery task ID
        run.celery_task_id = task.id
        db.commit()
        
        # Create training job response
        job = TrainingJob(
            job_id=str(run.id),
            model_name=config.model_name,
            status="running",
            algorithm=config.algorithm,
            problem_type=config.problem_type,
            progress_percentage=0.0,
            current_step="Initializing",
            created_at=run.created_at,
            celery_task_id=task.id
        )
        
        return TrainingJobResponse(
            job=job,
            message=f"Training job {job.job_id} started successfully with task ID {task.id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.get("/status/{job_id}")
async def get_training_status(job_id: str, db: Session = Depends(get_db)):
    """Get training job status."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(job_id))
        
        if not run:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        # Check Celery task status if available
        celery_task_id = getattr(run, 'celery_task_id', None)
        progress_percentage = 0.0
        current_step = run.status.title()
        
        if celery_task_id:
            try:
                # Get task result
                task_result = celery_app.AsyncResult(celery_task_id)
                task_state = task_result.state
                task_info = task_result.info or {}
                
                if task_state == 'PENDING':
                    progress_percentage = 0.0
                    current_step = "Queued"
                elif task_state == 'PROGRESS':
                    progress_percentage = task_info.get('progress', 0.0)
                    current_step = task_info.get('status', 'Processing')
                elif task_state == 'SUCCESS':
                    progress_percentage = 100.0
                    current_step = "Completed"
                    # Don't continue polling once completed
                elif task_state == 'FAILURE':
                    progress_percentage = 0.0
                    current_step = "Failed"
                    
            except Exception as e:
                # If can't get Celery status, fall back to DB status
                current_step = run.status.title()
                progress_percentage = 100.0 if run.status == "completed" else 0.0
        else:
            # No Celery task, use DB status
            progress_percentage = 100.0 if run.status == "completed" else 0.0
        
        job = TrainingJob(
            job_id=str(run.id),
            model_name=run.experiment.name,
            status=run.status,
            algorithm=run.algorithm,
            problem_type=run.experiment.problem_type,
            progress_percentage=progress_percentage,
            current_step=current_step,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            training_metrics=run.training_metrics,
            validation_metrics=run.validation_metrics,
            error_message=run.error_message,
            celery_task_id=celery_task_id
        )
        
        return job
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@router.get("/evaluate/{model_id}")
async def evaluate_model(model_id: str, db: Session = Depends(get_db)):
    """Get detailed metrics and evaluation for a trained model."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if run.status != "completed":
            raise HTTPException(status_code=400, detail="Model training not completed")
        
        # Get experiment info
        experiment = exp_store.get_experiment(run.experiment_id)
        
        # Get all runs for comparison
        all_runs = exp_store.get_runs_by_experiment(run.experiment_id)
        completed_runs = [r for r in all_runs if r.status == "completed"]
        
        # Calculate performance comparison
        performance_comparison = []
        if run.validation_metrics:
            primary_metric = "accuracy" if run.validation_metrics.get("accuracy") else list(run.validation_metrics.keys())[0]
            
            for other_run in completed_runs:
                if other_run.validation_metrics and primary_metric in other_run.validation_metrics:
                    performance_comparison.append({
                        "run_id": other_run.id,
                        "algorithm": other_run.algorithm,
                        "metric_value": other_run.validation_metrics[primary_metric],
                        "created_at": other_run.created_at.isoformat(),
                        "hyperparameters": other_run.hyperparameters
                    })
        
        # Sort by performance
        performance_comparison.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return {
            "model_info": {
                "model_id": model_id,
                "run_id": run.id,
                "experiment_name": experiment.name,
                "algorithm": run.algorithm,
                "status": run.status,
                "created_at": run.created_at.isoformat(),
                "training_duration_seconds": run.training_duration_seconds
            },
            "metrics": {
                "training_metrics": run.training_metrics,
                "validation_metrics": run.validation_metrics,
                "test_metrics": run.test_metrics
            },
            "hyperparameters": run.hyperparameters,
            "performance_comparison": performance_comparison,
            "experiment_summary": {
                "total_runs": len(all_runs),
                "completed_runs": len(completed_runs),
                "best_run_id": performance_comparison[0]["run_id"] if performance_comparison else None,
                "problem_type": experiment.problem_type,
                "target_column": experiment.target_column
            }
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate model: {str(e)}")


@router.post("/retrain/{model_id}", response_model=TrainingJobResponse)
async def retrain_model(
    model_id: str,
    config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Retrain an existing model with optional hyperparameter updates."""
    try:
        exp_store = ExperimentStore(db)
        
        # Get original model run
        original_run = exp_store.get_run(int(model_id))
        if not original_run:
            raise HTTPException(status_code=404, detail="Original model not found")
        
        # Create new run for retraining
        new_hyperparams = original_run.hyperparameters.copy() if original_run.hyperparameters else {}
        if config and config.get('updated_hyperparameters'):
            new_hyperparams.update(config['updated_hyperparameters'])
        
        run = exp_store.create_run(
            experiment_id=original_run.experiment_id,
            algorithm=original_run.algorithm,
            hyperparameters=new_hyperparams,
            model_id=str(uuid.uuid4())
        )
        
        # Update run status to running
        exp_store.update_run_status(run.id, 'running')
        
        # Prepare retraining configuration for Celery task
        training_config = {
            'run_id': run.id,
            'dataset_source': original_run.experiment.dataset_id.split('_')[0] if '_' in original_run.experiment.dataset_id else 'unknown',
            'dataset_id': original_run.experiment.dataset_id,
            'target_column': original_run.experiment.target_column,
            'algorithm': original_run.algorithm,
            'test_size': 0.2,  # Default for retraining
            'cv_folds': 5,
            'auto_hyperparameter_tuning': True
        }
        
        # Start Celery retraining task
        task = train_model_task.delay(training_config)
        
        # Update run with Celery task ID
        run.celery_task_id = task.id
        db.commit()
        
        job = TrainingJob(
            job_id=str(run.id),
            model_name=f"{original_run.experiment.name}_retrain",
            status="running",
            algorithm=original_run.algorithm,
            problem_type=original_run.experiment.problem_type,
            progress_percentage=0.0,
            current_step="Initializing Retraining",
            created_at=run.created_at,
            celery_task_id=task.id
        )
        
        return TrainingJobResponse(
            job=job,
            message=f"Retraining job {job.job_id} started successfully with task ID {task.id}"
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")


@router.post("/retrain/{model_id}/optimized", response_model=TrainingJobResponse)
async def retrain_model_optimized(
    model_id: str,
    db: Session = Depends(get_db)
):
    """Retrain model with AI-optimized hyperparameters based on experiment history."""
    try:
        exp_store = ExperimentStore(db)
        
        # Get original model run
        original_run = exp_store.get_run(int(model_id))
        if not original_run:
            raise HTTPException(status_code=404, detail="Original model not found")
        
        # Use the scheduler to create an optimized retraining job
        result = hyperparameter_optimizer.schedule_retraining(
            experiment_id=original_run.experiment_id,
            algorithm=original_run.algorithm
        )
        
        if result["status"] != "scheduled":
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to schedule optimized retraining"))
        
        # Get the created run
        new_run = exp_store.get_run(result["run_id"])
        
        job = TrainingJob(
            job_id=str(new_run.id),
            model_name=f"{original_run.experiment.name}_optimized",
            status="running",
            algorithm=new_run.algorithm,
            problem_type=original_run.experiment.problem_type,
            progress_percentage=0.0,
            current_step="Initializing Optimized Training",
            created_at=new_run.created_at,
            celery_task_id=result["task_id"]
        )
        
        return TrainingJobResponse(
            job=job,
            message=f"Optimized retraining job {job.job_id} started with AI-generated hyperparameters"
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start optimized retraining: {str(e)}")


@router.post("/optimize/experiment/{experiment_id}")
async def run_experiment_optimization(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """Run optimization cycle for a specific experiment."""
    try:
        result = hyperparameter_optimizer.schedule_retraining(
            experiment_id=int(experiment_id)
        )
        
        if result["status"] == "scheduled":
            return {
                "status": "success",
                "message": "Optimization job scheduled successfully",
                "run_id": result["run_id"],
                "task_id": result["task_id"],
                "optimized_hyperparameters": result["optimized_hyperparameters"]
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Optimization failed"))
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run optimization: {str(e)}")


@router.post("/optimize/auto")
async def run_auto_optimization():
    """Run automatic optimization cycle across all eligible experiments."""
    try:
        result = training_scheduler.run_optimization_cycle()
        
        return {
            "status": "completed",
            "candidates_analyzed": result["candidates_found"],
            "jobs_scheduled": len(result["scheduled_jobs"]),
            "scheduled_jobs": result["scheduled_jobs"],
            "errors": result["errors"],
            "message": f"Optimization cycle completed. {len(result['scheduled_jobs'])} jobs scheduled."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto optimization failed: {str(e)}")


@router.get("/analyze/{experiment_id}")
async def analyze_experiment_performance(
    experiment_id: str,
    db: Session = Depends(get_db)
):
    """Analyze experiment performance and get optimization insights."""
    try:
        analysis = hyperparameter_optimizer.analyze_experiment_performance(int(experiment_id))
        
        if analysis.get("status") in ["insufficient_data", "no_metric_data"]:
            return {
                "status": analysis["status"],
                "message": "Not enough data for analysis" if analysis["status"] == "insufficient_data" else "No metric data available",
                "runs_analyzed": analysis.get("runs_analyzed", 0)
            }
        
        return {
            "status": "success",
            "analysis": analysis,
            "recommendations": {
                "can_optimize": len(analysis.get("performance_trends", [])) >= 3,
                "best_algorithm": analysis["best_configurations"][0]["algorithm"] if analysis.get("best_configurations") else None,
                "improvement_potential": "high" if len(analysis.get("performance_trends", [])) >= 5 else "medium"
            }
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze experiment: {str(e)}")


@router.get("/evaluation/model/{model_id}/analytics")
async def get_model_analytics(model_id: str, db: Session = Depends(get_db)):
    """Get detailed analytics for a specific model."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if run.status != "completed":
            raise HTTPException(status_code=400, detail="Model training not completed")
        
        experiment = exp_store.get_experiment(run.experiment_id)
        
        # Get metrics
        training_metrics = run.training_metrics or {}
        validation_metrics = run.validation_metrics or {}
        test_metrics = run.test_metrics or {}
        
        # Combine all metrics for display
        all_metrics = {**training_metrics, **validation_metrics, **test_metrics}
        
        # Calculate performance insights
        insights = []
        
        if all_metrics.get("accuracy"):
            acc = all_metrics["accuracy"]
            if acc >= 0.95:
                insights.append("Excellent accuracy - model performs very well")
            elif acc >= 0.85:
                insights.append("Good accuracy - model shows strong performance")
            elif acc >= 0.75:
                insights.append("Moderate accuracy - consider feature engineering")
            else:
                insights.append("Low accuracy - model needs improvement")
        
        if all_metrics.get("precision") and all_metrics.get("recall"):
            prec = all_metrics["precision"]
            rec = all_metrics["recall"]
            if abs(prec - rec) > 0.1:
                insights.append("Imbalanced precision/recall - check class distribution")
        
        # Get dataset info
        dataset_info = {
            "name": experiment.dataset_id,
            "size": run.dataset_size or 0,
            "features": run.feature_count or 0,
            "target": experiment.target_column,
            "problem_type": experiment.problem_type
        }
        
        # Feature importance (mock for now, would need actual model introspection)
        feature_importance = []
        if run.feature_count:
            # Generate mock feature importance based on dataset
            feature_names = [f"feature_{i+1}" for i in range(min(run.feature_count, 10))]
            if experiment.dataset_id == "test_iris":
                feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            elif experiment.dataset_id == "61":  # OpenML Iris
                feature_names = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
            elif experiment.dataset_id == "test_housing":
                feature_names = ["median_income", "total_rooms", "housing_median_age", "latitude", "longitude"]
            
            import random
            random.seed(42)  # Consistent mock data
            for i, name in enumerate(feature_names[:min(10, len(feature_names))]):
                importance = random.uniform(0.05, 0.95)
                feature_importance.append({"feature": name, "importance": round(importance, 3)})
            
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "model_info": {
                "model_id": model_id,
                "name": experiment.name,
                "algorithm": run.algorithm,
                "status": run.status,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "training_duration": run.training_duration_seconds or 0
            },
            "metrics": {
                "accuracy": all_metrics.get("accuracy"),
                "precision": all_metrics.get("precision"),
                "recall": all_metrics.get("recall"),
                "f1_score": all_metrics.get("f1_score"),
                "roc_auc": all_metrics.get("roc_auc"),
                "r2_score": all_metrics.get("r2_score"),
                "mse": all_metrics.get("mse"),
                "mae": all_metrics.get("mae")
            },
            "dataset_info": dataset_info,
            "feature_importance": feature_importance,
            "hyperparameters": run.hyperparameters or {},
            "insights": insights,
            "training_details": {
                "cv_score": validation_metrics.get("cv_score"),
                "training_score": training_metrics.get("score"),
                "test_score": test_metrics.get("accuracy") or test_metrics.get("score"),
                "overfitting_risk": "Low" if validation_metrics.get("cv_score", 0) > 0.8 else "Medium"
            }
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model analytics: {str(e)}")


@router.get("/evaluation/dashboard")
async def get_evaluation_dashboard(db: Session = Depends(get_db)):
    """Get dashboard data for model evaluation page."""
    try:
        exp_store = ExperimentStore(db)
        
        # Get all completed models
        experiments = exp_store.get_experiments()
        completed_models = []
        
        for experiment in experiments:
            runs = exp_store.get_runs_by_experiment(experiment.id)
            completed_runs = [r for r in runs if r.status == "completed"]
            
            for run in completed_runs:
                if run.validation_metrics or run.test_metrics:
                    # Get the primary metrics
                    metrics = run.test_metrics or run.validation_metrics or {}
                    model_info = {
                        "model_id": str(run.id),
                        "model_name": experiment.name,
                        "algorithm": run.algorithm,
                        "problem_type": experiment.problem_type,
                        "target_column": experiment.target_column,
                        "dataset_name": experiment.dataset_id,
                        "training_duration": run.training_duration_seconds,
                        "created_at": run.created_at.isoformat() if run.created_at else None,
                        "dataset_size": run.dataset_size,
                        "feature_count": run.feature_count,
                        "metrics": metrics,
                        "hyperparameters": run.hyperparameters or {}
                    }
                    completed_models.append(model_info)
        
        # Sort by creation date (most recent first)
        completed_models.sort(key=lambda x: x["created_at"] or "0", reverse=True)
        
        # Get the best model (highest accuracy/score)
        best_model = None
        if completed_models:
            # Find model with highest accuracy, f1_score, or first metric available
            for model in completed_models:
                metrics = model["metrics"]
                score = metrics.get("accuracy") or metrics.get("f1_score") or metrics.get("r2_score") or 0
                if best_model is None or score > (best_model["metrics"].get("accuracy", 0) or best_model["metrics"].get("f1_score", 0) or best_model["metrics"].get("r2_score", 0)):
                    best_model = model
        
        return {
            "models": completed_models,
            "total_models": len(completed_models),
            "best_model": best_model,
            "summary": {
                "total_experiments": len(experiments),
                "completed_models": len(completed_models),
                "problem_types": list(set([m["problem_type"] for m in completed_models])),
                "algorithms_used": list(set([m["algorithm"] for m in completed_models]))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation dashboard: {str(e)}")


@router.get("/jobs")
async def list_training_jobs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all training jobs."""
    try:
        exp_store = ExperimentStore(db)
        experiments = exp_store.get_experiments(skip=skip, limit=limit)
        
        jobs = []
        for experiment in experiments:
            runs = exp_store.get_runs_by_experiment(experiment.id)
            for run in runs:
                job = TrainingJob(
                    job_id=str(run.id),
                    model_name=experiment.name,
                    status=run.status,
                    algorithm=run.algorithm,
                    problem_type=experiment.problem_type,
                    progress_percentage=100.0 if run.status == "completed" else 0.0,
                    current_step=run.status.title(),
                    created_at=run.created_at,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    training_metrics=run.training_metrics,
                    validation_metrics=run.validation_metrics
                )
                jobs.append(job)
        
        return {"jobs": jobs, "total": len(jobs)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list training jobs: {str(e)}")


@router.delete("/clear")
async def clear_all_models(db: Session = Depends(get_db)):
    """Clear all trained models, experiments, and cached data."""
    try:
        import os
        import glob
        from models.experiment_store import Experiment, Run, Hyperparameter
        
        # Count models before deletion
        experiments = db.query(Experiment).all()
        total_experiments = len(experiments)
        total_runs = sum(len(exp.runs) for exp in experiments)
        
        # Delete all hyperparameters first (foreign key constraint)
        db.query(Hyperparameter).delete()
        
        # Delete all runs
        db.query(Run).delete()
        
        # Delete all experiments
        db.query(Experiment).delete()
        
        db.commit()
        
        # Clear model storage directory
        model_storage_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_storage')
        deleted_files = 0
        if os.path.exists(model_storage_path):
            model_files = glob.glob(os.path.join(model_storage_path, '*.pkl'))
            for model_file in model_files:
                try:
                    os.remove(model_file)
                    deleted_files += 1
                except Exception as e:
                    print(f"Warning: Could not delete {model_file}: {e}")
        
        return {
            "message": "All models and experiments cleared successfully",
            "deleted_experiments": total_experiments,
            "deleted_runs": total_runs,
            "deleted_model_files": deleted_files
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear models: {str(e)}")
