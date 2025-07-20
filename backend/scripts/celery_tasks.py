"""
Celery tasks for background processing in VibeML.
"""
from celery import current_task
from celery_config import celery_app
import pandas as pd
import logging
from typing import Dict, Any
import traceback
from datetime import datetime
import numpy as np

from models.experiment_store import SessionLocal, ExperimentStore
from services.training_engine import training_engine
from services.open_ml import openml_service
from services.kaggle import kaggle_service

logger = logging.getLogger(__name__)


def safe_update_status(run_id, status, error_message=None):
    """Safely update run status with error handling and retry logic to prevent database locking issues."""
    if not run_id:
        return
    
    max_retries = 5  # Increased retries
    for attempt in range(max_retries):
        session = None
        try:
            # Create a fresh database session for each update
            session = SessionLocal()
            exp_store = ExperimentStore(session)
            exp_store.update_run_status(run_id, status, error_message)
            session.commit()
            session.close()
            logger.info(f"Successfully updated run {run_id} status to {status}")
            return
        except Exception as e:
            if session:
                try:
                    session.rollback()
                    session.close()
                except:
                    pass  # Ignore errors when closing
            
            logger.warning(f"Attempt {attempt + 1} failed to update status for run {run_id}: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"Failed to update status for run {run_id} after {max_retries} attempts: {e}")
                return  # Don't raise - continue execution even if DB update fails
            
            # Exponential backoff with jitter
            import time
            import random
            base_delay = 0.5 * (2 ** attempt)
            jitter = random.uniform(0.1, 0.3)  # Add randomness to prevent thundering herd
            time.sleep(base_delay + jitter)
            time.sleep(0.1 * (2 ** attempt))


def safe_update_metrics(run_id, training_results):
    """Safely update run metrics with error handling and retry logic."""
    if not run_id:
        return
    
    max_retries = 5  # Increased retries
    for attempt in range(max_retries):
        session = None
        try:
            session = SessionLocal()
            exp_store = ExperimentStore(session)
            exp_store.update_run_metrics(
                run_id,
                training_metrics=training_results.get('best_test_metrics', {}),
                validation_metrics={'cv_score': training_results.get('best_cv_score', 0)},
                test_metrics=training_results.get('best_test_metrics', {})
            )
            exp_store.update_run_model_info(
                run_id,
                model_path=training_results.get('model_path'),
                training_duration_seconds=training_results.get('training_duration_seconds'),
                model_size_bytes=None
            )
            session.commit()
            session.close()
            logger.info(f"Successfully updated metrics for run {run_id}")
            return
        except Exception as e:
            if session:
                try:
                    session.rollback()
                    session.close()
                except:
                    pass  # Ignore errors when closing
            
            logger.warning(f"Attempt {attempt + 1} failed to update metrics for run {run_id}: {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"Failed to update metrics for run {run_id} after {max_retries} attempts: {e}")
                return  # Don't raise - continue execution
            
            # Exponential backoff with jitter
            import time
            import random
            base_delay = 0.5 * (2 ** attempt)
            jitter = random.uniform(0.1, 0.3)
            time.sleep(base_delay + jitter)


@celery_app.task(bind=True)
def train_model_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task for model training.
    
    Args:
        config: Training configuration containing dataset info, algorithm, etc.
    
    Returns:
        Dictionary with training results
    """
    task_id = self.request.id
    logger.info(f"Starting training task {task_id} with config: {config}")
    
    # Get database session
    db = SessionLocal()
    exp_store = ExperimentStore(db)
    
    try:
        run_id = config.get('run_id')
        # Safely update status to running
        safe_update_status(run_id, 'running')
        logger.info(f"Starting training for run_id: {run_id}")
        
        # Update task status
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Loading dataset', 'progress': 10}
        )
        
        # Load dataset based on source
        dataset_source = config.get('dataset_source', 'unknown')
        dataset_id = config.get('dataset_id')
        
        if dataset_source == 'openml':
            # Extract OpenML dataset ID from dataset_id
            try:
                if str(dataset_id).startswith('openml_'):
                    openml_id = int(dataset_id.replace('openml_', ''))
                else:
                    openml_id = int(dataset_id)
                df = openml_service.get_dataset(openml_id)
                logger.info(f"Loaded OpenML dataset {openml_id} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load OpenML dataset {dataset_id}: {e}")
                raise ValueError(f"Could not load OpenML dataset {dataset_id}: {e}")
        
        elif dataset_source == 'kaggle':
            # For Kaggle datasets, use the dataset_id directly as the dataset name
            try:
                df = kaggle_service.download_dataset(dataset_id)
                logger.info(f"Loaded Kaggle dataset {dataset_id} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load Kaggle dataset {dataset_id}: {e}")
                raise ValueError(f"Could not load Kaggle dataset {dataset_id}: {e}")
        
        elif dataset_source == 'upload':
            # For uploaded files, try to load from temporary storage first
            import os
            import tempfile
            
            # Check if the dataset was temporarily stored
            temp_dir = tempfile.gettempdir()
            possible_paths = [
                os.path.join(temp_dir, f"dataset_upload_{dataset_id}.csv"),
                os.path.join(temp_dir, f"dataset_{dataset_source}_{dataset_id.replace('/', '_')}.csv"),
                os.path.join(temp_dir, f"{dataset_id}.csv"),
                dataset_id  # In case dataset_id is actually a file path
            ]
            
            df = None
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        logger.info(f"Loaded uploaded dataset from {path} with shape {df.shape}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
                    continue
            
            if df is None:
                raise ValueError(f"Could not find uploaded dataset file for ID: {dataset_id}")
        
        elif dataset_source == 'local':
            # For local test datasets
            import os
            
            # Path to local test datasets
            script_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from scripts/
            dataset_path = os.path.join(script_dir, 'test_datasets', f"{dataset_id}.csv")
            
            try:
                if os.path.exists(dataset_path):
                    df = pd.read_csv(dataset_path)
                    logger.info(f"Loaded local dataset from {dataset_path} with shape {df.shape}")
                else:
                    raise FileNotFoundError(f"Local dataset file not found: {dataset_path}")
            except Exception as e:
                logger.error(f"Failed to load local dataset {dataset_id}: {e}")
                raise ValueError(f"Could not load local dataset {dataset_id}: {e}")
        
        else:
            raise ValueError(f"Unknown dataset source: {dataset_source}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Preprocessing data', 'progress': 25}
        )
        
        # Get training parameters
        target_column = config.get('target_column')
        algorithm = config.get('algorithm')
        test_size = config.get('test_size', 0.2)
        cv_folds = config.get('cv_folds', 5)
        enable_tuning = config.get('auto_hyperparameter_tuning', True)
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Training models', 'progress': 50}
        )
        
        # Train the model
        if algorithm and algorithm != 'auto':
            # Train specific algorithm
            training_results = training_engine.auto_train(
                df=df,
                target_column=target_column,
                test_size=test_size,
                cv_folds=cv_folds,
                enable_hyperparameter_tuning=enable_tuning,
                algorithms=[algorithm]
            )
        else:
            # AutoML - try all algorithms
            training_results = training_engine.auto_train(
                df=df,
                target_column=target_column,
                test_size=test_size,
                cv_folds=cv_folds,
                enable_hyperparameter_tuning=enable_tuning
            )
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Saving results', 'progress': 90}
        )
        
        # Get the run from database
        run_id = config.get('run_id')
        # Safely update database with results
        safe_update_status(run_id, 'completed')
        safe_update_metrics(run_id, training_results)
        
        logger.info(f"Training completed successfully for run_id: {run_id}")
        
        # Final status update
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Training completed successfully',
                'progress': 100,
                'results': training_results
            }
        )
        
        logger.info(f"Training task {task_id} completed successfully")
        return {
            'status': 'completed',
            'progress': 100,
            'results': training_results,
            'message': 'Training completed successfully'
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(f"Training task {task_id} failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Update run status if run_id is available
        run_id = config.get('run_id')
        # Safely update status to failed
        safe_update_status(run_id, 'failed', error_msg)
        
        logger.error(f"Training failed for run_id: {run_id} - {error_msg}")
        
        # Update task status
        self.update_state(
            state='FAILURE',
            meta={'status': error_msg, 'progress': 0}
        )
        
        raise Exception(error_msg)
    
    finally:
        db.close()


@celery_app.task(bind=True)
def predict_task(self, model_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background task for making predictions.
    
    Args:
        model_path: Path to the saved model
        data: Input data for prediction
    
    Returns:
        Dictionary with prediction results
    """
    task_id = self.request.id
    logger.info(f"Starting prediction task {task_id}")
    
    try:
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        predictions = training_engine.predict(model_path, df)
        
        # Try to get probabilities if available
        try:
            probabilities = training_engine.predict_proba(model_path, df)
            prob_dict = {f'prob_class_{i}': float(prob) for i, prob in enumerate(probabilities[0])}
        except:
            prob_dict = {}
        
        result = {
            'prediction': predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0],
            'probabilities': prob_dict,
            'status': 'success'
        }
        
        logger.info(f"Prediction task {task_id} completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(f"Prediction task {task_id} failed: {error_msg}")
        
        self.update_state(
            state='FAILURE',
            meta={'status': error_msg}
        )
        
        raise Exception(error_msg)


@celery_app.task
def cleanup_old_models():
    """Background task to cleanup old model files."""
    # TODO: Implement model cleanup logic
    logger.info("Model cleanup task executed")
    return {"status": "cleanup_completed"}


@celery_app.task
def health_check():
    """Health check task for Celery worker."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
