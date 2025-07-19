"""
Export router for model and result export endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import json
import pickle
from typing import Dict, Any
import pandas as pd

from models.experiment_store import get_db, ExperimentStore

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/model/{model_id}")
async def download_model(model_id: str, db: Session = Depends(get_db)):
    """Download a trained model file."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run:
            raise HTTPException(status_code=404, detail="Model not found")
        
        if not run.model_path or not os.path.exists(run.model_path):
            raise HTTPException(status_code=404, detail="Model file not found")
        
        filename = f"model_{model_id}_{run.algorithm}.pkl"
        
        return FileResponse(
            path=run.model_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.get("/code/{model_id}")
async def download_retrain_code(model_id: str, db: Session = Depends(get_db)):
    """Generate and download retraining code for a model."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Generate Python retraining code
        retrain_code = f'''"""
Auto-generated retraining script for Model ID: {model_id}
Algorithm: {run.algorithm}
Dataset: {run.dataset_name}
Target: {run.target_column}
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

# Model configuration
MODEL_CONFIG = {{
    "algorithm": "{run.algorithm}",
    "target_column": "{run.target_column}",
    "dataset_name": "{run.dataset_name}",
    "hyperparameters": {run.hyperparameters if run.hyperparameters else '{}'}
}}

def load_data(data_path):
    """Load your dataset here."""
    # Replace this with your actual data loading logic
    # df = pd.read_csv(data_path)
    raise NotImplementedError("Please implement data loading logic")

def preprocess_data(df):
    """Preprocess the data."""
    # Handle missing values
    df = df.fillna(df.mean() if df.select_dtypes(include=['number']).shape[1] > 0 else df.mode().iloc[0])
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != MODEL_CONFIG["target_column"]:
            df[col] = pd.Categorical(df[col]).codes
    
    return df

def get_model():
    """Get the model based on algorithm."""
    algorithm = MODEL_CONFIG["algorithm"]
    hyperparameters = MODEL_CONFIG["hyperparameters"]
    
    if algorithm == "RandomForestClassifier":
        return RandomForestClassifier(**hyperparameters)
    elif algorithm == "RandomForestRegressor":
        return RandomForestRegressor(**hyperparameters)
    elif algorithm == "LogisticRegression":
        return LogisticRegression(**hyperparameters)
    elif algorithm == "LinearRegression":
        return LinearRegression(**hyperparameters)
    elif algorithm == "SVC":
        return SVC(**hyperparameters)
    elif algorithm == "SVR":
        return SVR(**hyperparameters)
    else:
        raise ValueError(f"Unsupported algorithm: {{algorithm}}")

def train_model(data_path, output_path="retrained_model.pkl"):
    """Retrain the model with new data."""
    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)
    
    # Split features and target
    target_col = MODEL_CONFIG["target_column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = get_model()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    if MODEL_CONFIG["algorithm"] in ["RandomForestClassifier", "LogisticRegression", "SVC"]:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {{accuracy:.4f}}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model MSE: {{mse:.4f}}")
    
    # Save the retrained model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {{output_path}}")
    return model

if __name__ == "__main__":
    # Example usage
    # train_model("path/to/your/dataset.csv", "retrained_model_{model_id}.pkl")
    print("Please implement the load_data function and call train_model with your dataset path.")
    print(f"Original model accuracy: {run.accuracy if run.accuracy else 'N/A'}")
'''        
        # Create temporary file for download
        temp_file = f"/tmp/retrain_model_{model_id}.py"
        with open(temp_file, 'w') as f:
            f.write(retrain_code)
        
        return FileResponse(
            path=temp_file,
            filename=f"retrain_model_{model_id}.py",
            media_type='text/plain'
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate retrain code: {str(e)}")


@router.get("/experiment/{experiment_id}")
async def export_experiment_results(experiment_id: str, db: Session = Depends(get_db)):
    """Export complete experiment results as JSON."""
    try:
        exp_store = ExperimentStore(db)
        
        # Get all runs for this experiment
        runs = exp_store.get_all_runs()
        experiment_runs = [run for run in runs if str(run.id) == experiment_id or 
                          (hasattr(run, 'experiment_id') and str(run.experiment_id) == experiment_id)]
        
        if not experiment_runs:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        # Compile experiment data
        experiment_data = {
            "experiment_id": experiment_id,
            "runs": [],
            "summary": {
                "total_runs": len(experiment_runs),
                "completed_runs": len([r for r in experiment_runs if r.status == "completed"]),
                "best_accuracy": max([r.accuracy for r in experiment_runs if r.accuracy], default=0),
                "algorithms_tested": list(set([r.algorithm for r in experiment_runs]))
            }
        }
        
        for run in experiment_runs:
            run_data = {
                "run_id": run.id,
                "algorithm": run.algorithm,
                "status": run.status,
                "accuracy": run.accuracy,
                "hyperparameters": run.hyperparameters,
                "training_duration": run.training_duration,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "dataset_name": run.dataset_name,
                "target_column": run.target_column
            }
            experiment_data["runs"].append(run_data)
        
        # Create temporary JSON file
        temp_file = f"/tmp/experiment_{experiment_id}_results.json"
        with open(temp_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        return FileResponse(
            path=temp_file,
            filename=f"experiment_{experiment_id}_results.json",
            media_type='application/json'
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export experiment: {str(e)}")


@router.get("/experiments/{experiment_id}/results")
async def export_experiment_results(experiment_id: str, format: str = "json", db: Session = Depends(get_db)):
    """Export experiment results in various formats."""
    try:
        exp_store = ExperimentStore(db)
        summary = exp_store.get_experiment_summary(int(experiment_id))
        
        if not summary:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if format.lower() == "json":
            return Response(
                content=json.dumps(summary, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}_results.json"}
            )
        elif format.lower() == "csv":
            # TODO: Implement CSV export
            return {"message": "CSV export not implemented yet"}
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format. Use 'json' or 'csv'")
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/experiments/{experiment_id}/metrics")
async def export_experiment_metrics(experiment_id: str, db: Session = Depends(get_db)):
    """Export detailed metrics for all runs in an experiment."""
    try:
        exp_store = ExperimentStore(db)
        experiment = exp_store.get_experiment(int(experiment_id))
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        runs = exp_store.get_runs_by_experiment(int(experiment_id))
        
        metrics_data = {
            "experiment_info": {
                "id": experiment.id,
                "name": experiment.name,
                "problem_type": experiment.problem_type,
                "target_column": experiment.target_column,
                "dataset_id": experiment.dataset_id
            },
            "runs": []
        }
        
        for run in runs:
            run_data = {
                "run_id": run.id,
                "run_number": run.run_number,
                "algorithm": run.algorithm,
                "status": run.status,
                "hyperparameters": run.hyperparameters,
                "training_metrics": run.training_metrics,
                "validation_metrics": run.validation_metrics,
                "test_metrics": run.test_metrics,
                "training_duration_seconds": run.training_duration_seconds,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None
            }
            metrics_data["runs"].append(run_data)
        
        return metrics_data
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics export failed: {str(e)}")


@router.get("/models/compare")
async def compare_models(model_ids: str, db: Session = Depends(get_db)):
    """Compare multiple models side by side."""
    try:
        model_id_list = [int(id.strip()) for id in model_ids.split(",")]
        exp_store = ExperimentStore(db)
        
        comparison = {
            "models": [],
            "comparison_timestamp": json.dumps({"timestamp": "2024-01-01T00:00:00"})  # Placeholder
        }
        
        for model_id in model_id_list:
            run = exp_store.get_run(model_id)
            if run:
                model_data = {
                    "model_id": run.id,
                    "model_name": run.experiment.name,
                    "algorithm": run.algorithm,
                    "problem_type": run.experiment.problem_type,
                    "hyperparameters": run.hyperparameters,
                    "validation_metrics": run.validation_metrics,
                    "training_duration": run.training_duration_seconds,
                    "status": run.status
                }
                comparison["models"].append(model_data)
        
        if not comparison["models"]:
            raise HTTPException(status_code=404, detail="No valid models found")
        
        return comparison
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")
