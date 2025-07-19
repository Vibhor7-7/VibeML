"""
Prediction router for model prediction endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Any
import pandas as pd
import os

from models.experiment_store import get_db, ExperimentStore
from services.training_engine import training_engine

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/{model_id}")
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
        
        if not run.model_path or not os.path.exists(run.model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Load model and make prediction
        predictions = training_engine.predict(run.model_path, df)
        
        # Try to get prediction probabilities if available
        try:
            probabilities = training_engine.predict_proba(run.model_path, df)
            prob_dict = {f'prob_class_{i}': float(prob) for i, prob in enumerate(probabilities[0])}
        except:
            prob_dict = {}
        
        prediction_result = {
            "prediction": predictions[0].tolist() if hasattr(predictions[0], 'tolist') else predictions[0],
            "probabilities": prob_dict,
            "model_id": model_id,
            "algorithm": run.algorithm,
            "features_used": features,
            "prediction_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return prediction_result
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Add API endpoint alias for model inference
@router.post("/api/{model_id}/predict")
async def predict_api(
    model_id: str,
    features: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Inference API endpoint for model predictions."""
    return await predict_single(model_id, features, db)


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
        
        if not run.model_path or not os.path.exists(run.model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # Convert features to DataFrame
        df = pd.DataFrame(features_list)
        
        # Load model and make predictions
        predictions = training_engine.predict(run.model_path, df)
        
        # Try to get prediction probabilities if available
        try:
            probabilities = training_engine.predict_proba(run.model_path, df)
            prob_data = [
                {f'prob_class_{i}': float(prob) for i, prob in enumerate(prob_row)}
                for prob_row in probabilities
            ]
        except:
            prob_data = [{}] * len(predictions)
        
        batch_results = []
        for i, (prediction, prob_dict, original_features) in enumerate(zip(predictions, prob_data, features_list)):
            result = {
                "id": i,
                "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                "probabilities": prob_dict,
                "features_used": original_features
            }
            batch_results.append(result)
        
        response = {
            "predictions": batch_results,
            "model_id": model_id,
            "algorithm": run.algorithm,
            "batch_size": len(features_list),
            "prediction_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return response
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.get("/models")
async def list_available_models(db: Session = Depends(get_db)):
    """List all trained models available for prediction."""
    try:
        exp_store = ExperimentStore(db)
        experiments = exp_store.get_experiments()
        
        available_models = []
        for experiment in experiments:
            runs = exp_store.get_runs_by_experiment(experiment.id)
            for run in runs:
                if run.status == "completed" and run.model_path and os.path.exists(run.model_path):
                    model_info = {
                        "model_id": run.id,
                        "algorithm": run.algorithm,
                        "dataset": run.dataset_name if hasattr(run, 'dataset_name') else experiment.dataset_id,
                        "accuracy": run.accuracy if hasattr(run, 'accuracy') else None,
                        "target_column": experiment.target_column,
                        "created_at": run.created_at.isoformat() if run.created_at else None,
                        "model_size_mb": round(os.path.getsize(run.model_path) / (1024 * 1024), 2) if run.model_path else None
                    }
                    available_models.append(model_info)
        
        return {
            "available_models": available_models,
            "total_count": len(available_models)
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
            "model_id": run.id,
            "algorithm": run.algorithm,
            "dataset": run.dataset_name,
            "target_column": run.target_column,
            "status": run.status,
            "accuracy": run.accuracy,
            "hyperparameters": run.hyperparameters,
            "training_duration": run.training_duration,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            "model_available": run.model_path and os.path.exists(run.model_path) if run.model_path else False
        }
        
        if run.model_path and os.path.exists(run.model_path):
            model_info["model_size_mb"] = round(os.path.getsize(run.model_path) / (1024 * 1024), 2)
        
        return model_info
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
