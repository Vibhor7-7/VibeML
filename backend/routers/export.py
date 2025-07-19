"""
Export router for model and result export endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import Dict, Any
import json

from models.experiment_store import get_db, ExperimentStore

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/models/{model_id}/download")
async def download_model(model_id: str, db: Session = Depends(get_db)):
    """Download a trained model file."""
    try:
        exp_store = ExperimentStore(db)
        run = exp_store.get_run(int(model_id))
        
        if not run or run.status != "completed":
            raise HTTPException(status_code=404, detail="Trained model not found")
        
        if not run.model_path:
            raise HTTPException(status_code=404, detail="Model file not found")
        
        # TODO: Implement actual file download
        # with open(run.model_path, 'rb') as f:
        #     model_data = f.read()
        
        # return Response(
        #     content=model_data,
        #     media_type='application/octet-stream',
        #     headers={"Content-Disposition": f"attachment; filename=model_{model_id}.pkl"}
        # )
        
        # Placeholder response
        return {"message": f"Model {model_id} download endpoint - not implemented yet"}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model download failed: {str(e)}")


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
