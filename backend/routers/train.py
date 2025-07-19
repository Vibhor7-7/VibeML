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
