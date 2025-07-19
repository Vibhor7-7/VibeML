"""
Training router for model training endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any
import uuid

from models.experiment_store import get_db, ExperimentStore
from db_schema.train_config import TrainConfig, TrainingJob, TrainingJobResponse, RetrainConfig

router = APIRouter(prefix="/train", tags=["training"])


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(
    config: TrainConfig,
    background_tasks: BackgroundTasks,
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
        
        # Create training job
        job = TrainingJob(
            job_id=str(run.id),
            model_name=config.model_name,
            status="pending",
            algorithm=config.algorithm,
            problem_type=config.problem_type,
            progress_percentage=0.0,
            current_step="Initializing",
            created_at=run.created_at
        )
        
        # Add background training task (placeholder)
        # background_tasks.add_task(train_model_task, run.id, config)
        
        return TrainingJobResponse(
            job=job,
            message=f"Training job {job.job_id} started successfully"
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
        
        job = TrainingJob(
            job_id=str(run.id),
            model_name=run.experiment.name,
            status=run.status,
            algorithm=run.algorithm,
            problem_type=run.experiment.problem_type,
            progress_percentage=100.0 if run.status == "completed" else 0.0,
            current_step=run.status.title(),
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
            training_metrics=run.training_metrics,
            validation_metrics=run.validation_metrics,
            error_message=run.error_message
        )
        
        return job
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@router.post("/retrain", response_model=TrainingJobResponse)
async def retrain_model(
    config: RetrainConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retrain an existing model."""
    try:
        exp_store = ExperimentStore(db)
        
        # Get original model run
        original_run = exp_store.get_run(int(config.model_id))
        if not original_run:
            raise HTTPException(status_code=404, detail="Original model not found")
        
        # Create new run for retraining
        new_hyperparams = original_run.hyperparameters.copy()
        if config.updated_hyperparameters:
            new_hyperparams.update(config.updated_hyperparameters)
        
        run = exp_store.create_run(
            experiment_id=original_run.experiment_id,
            algorithm=original_run.algorithm,
            hyperparameters=new_hyperparams,
            model_id=str(uuid.uuid4())
        )
        
        job = TrainingJob(
            job_id=str(run.id),
            model_name=f"{original_run.experiment.name}_retrain",
            status="pending",
            algorithm=original_run.algorithm,
            problem_type=original_run.experiment.problem_type,
            progress_percentage=0.0,
            current_step="Initializing Retraining",
            created_at=run.created_at
        )
        
        return TrainingJobResponse(
            job=job,
            message=f"Retraining job {job.job_id} started successfully"
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")


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
