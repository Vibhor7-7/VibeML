"""
Prediction router for model prediction endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any
import pandas as pd

from models.experiment_store import get_db, ExperimentStore

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/single")
async def predict_single(
    model_id: str,
    features: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Make a single prediction using a trained model."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run or run.status != "completed":
            raise HTTPException(status_code=404, detail="Trained model not found")
        
        # TODO: Load model and make prediction
        # model = load_model(run.model_path)
        # prediction = model.predict([list(features.values())])
        
        # Placeholder response
        prediction = {
            "prediction": "placeholder_prediction",
            "confidence": 0.85,
            "model_id": model_id,
            "features_used": features
        }
        
        return prediction
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch")
async def predict_batch(
    model_id: str,
    features_list: List[Dict[str, Any]],
    db: Session = Depends(get_db)
):
    """Make batch predictions using a trained model."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run or run.status != "completed":
            raise HTTPException(status_code=404, detail="Trained model not found")
        
        # TODO: Load model and make batch predictions
        # model = load_model(run.model_path)
        # df = pd.DataFrame(features_list)
        # predictions = model.predict(df)
        
        # Placeholder response
        predictions = [
            {
                "prediction": f"placeholder_prediction_{i}",
                "confidence": 0.85,
                "index": i
            }
            for i, _ in enumerate(features_list)
        ]
        
        return {
            "predictions": predictions,
            "model_id": model_id,
            "total_predictions": len(predictions)
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/models")
async def list_available_models(db: Session = Depends(get_db)):
    """List all available trained models for prediction."""
    try:
        exp_store = ExperimentStore(db)
        experiments = exp_store.get_experiments()
        
        models = []
        for experiment in experiments:
            runs = exp_store.get_runs_by_experiment(experiment.id)
            completed_runs = [run for run in runs if run.status == "completed"]
            
            for run in completed_runs:
                model_info = {
                    "model_id": str(run.id),
                    "model_name": experiment.name,
                    "algorithm": run.algorithm,
                    "problem_type": experiment.problem_type,
                    "target_column": experiment.target_column,
                    "created_at": run.created_at.isoformat(),
                    "validation_metrics": run.validation_metrics,
                    "training_duration": run.training_duration_seconds
                }
                models.append(model_info)
        
        return {
            "models": models,
            "total_count": len(models)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_id}/info")
async def get_model_info(model_id: str, db: Session = Depends(get_db)):
    """Get detailed information about a specific model."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = {
            "model_id": str(run.id),
            "model_name": run.experiment.name,
            "algorithm": run.algorithm,
            "problem_type": run.experiment.problem_type,
            "target_column": run.experiment.target_column,
            "dataset_id": run.experiment.dataset_id,
            "hyperparameters": run.hyperparameters,
            "training_metrics": run.training_metrics,
            "validation_metrics": run.validation_metrics,
            "test_metrics": run.test_metrics,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "training_duration": run.training_duration_seconds,
            "model_size_bytes": run.model_size_bytes,
            "feature_count": run.feature_count
        }
        
        return model_info
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
