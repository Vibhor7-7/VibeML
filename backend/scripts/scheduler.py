"""
Scheduler for automated hyperparameter optimization and retraining.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from models.experiment_store import SessionLocal, ExperimentStore
from scripts.celery_tasks import train_model_task
import numpy as np
import json

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Intelligent hyperparameter optimization based on previous runs."""
    
    def __init__(self):
        self.db = SessionLocal()
        self.exp_store = ExperimentStore(self.db)
    
    def analyze_experiment_performance(self, experiment_id: int) -> Dict[str, Any]:
        """Analyze performance trends across runs in an experiment."""
        runs = self.exp_store.get_runs_by_experiment(experiment_id)
        completed_runs = [r for r in runs if r.status == "completed" and r.validation_metrics]
        
        if len(completed_runs) < 2:
            return {"status": "insufficient_data", "runs_analyzed": len(completed_runs)}
        
        # Determine primary metric based on problem type
        experiment = self.exp_store.get_experiment(experiment_id)
        primary_metric = "accuracy" if experiment.problem_type == "classification" else "r2_score"
        
        analysis = {
            "experiment_id": experiment_id,
            "runs_analyzed": len(completed_runs),
            "primary_metric": primary_metric,
            "performance_trends": [],
            "best_configurations": [],
            "parameter_insights": {}
        }
        
        # Extract performance data
        performances = []
        for run in completed_runs:
            if primary_metric in run.validation_metrics:
                performances.append({
                    "run_id": run.id,
                    "score": run.validation_metrics[primary_metric],
                    "hyperparameters": run.hyperparameters or {},
                    "algorithm": run.algorithm,
                    "created_at": run.created_at
                })
        
        if not performances:
            return {"status": "no_metric_data", "message": f"No {primary_metric} found"}
        
        # Sort by performance
        performances.sort(key=lambda x: x["score"], reverse=True)
        analysis["performance_trends"] = performances
        analysis["best_configurations"] = performances[:3]  # Top 3
        
        # Analyze hyperparameter patterns
        analysis["parameter_insights"] = self._analyze_hyperparameter_patterns(performances)
        
        return analysis
    
    def _analyze_hyperparameter_patterns(self, performances: List[Dict]) -> Dict[str, Any]:
        """Analyze which hyperparameters correlate with better performance."""
        if len(performances) < 3:
            return {"status": "insufficient_data"}
        
        # Group performances by algorithm
        algo_performances = {}
        for perf in performances:
            algo = perf["algorithm"]
            if algo not in algo_performances:
                algo_performances[algo] = []
            algo_performances[algo].append(perf)
        
        insights = {}
        
        for algo, perfs in algo_performances.items():
            if len(perfs) < 2:
                continue
                
            # Analyze top vs bottom performers
            top_performers = perfs[:len(perfs)//2] if len(perfs) > 2 else perfs[:1]
            bottom_performers = perfs[len(perfs)//2:] if len(perfs) > 2 else perfs[1:]
            
            # Extract common hyperparameters
            top_params = self._extract_common_params(top_performers)
            bottom_params = self._extract_common_params(bottom_performers)
            
            insights[algo] = {
                "top_performer_patterns": top_params,
                "bottom_performer_patterns": bottom_params,
                "recommended_ranges": self._get_recommended_ranges(top_performers)
            }
        
        return insights
    
    def _extract_common_params(self, performances: List[Dict]) -> Dict[str, Any]:
        """Extract common hyperparameter patterns from a group of runs."""
        if not performances:
            return {}
        
        # Get all hyperparameter keys
        all_keys = set()
        for perf in performances:
            all_keys.update(perf["hyperparameters"].keys())
        
        common_patterns = {}
        for key in all_keys:
            values = [perf["hyperparameters"].get(key) for perf in performances if key in perf["hyperparameters"]]
            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric parameter
                    common_patterns[key] = {
                        "type": "numeric",
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values)
                    }
                else:
                    # Categorical parameter
                    from collections import Counter
                    value_counts = Counter(values)
                    common_patterns[key] = {
                        "type": "categorical",
                        "most_common": value_counts.most_common(3)
                    }
        
        return common_patterns
    
    def _get_recommended_ranges(self, top_performers: List[Dict]) -> Dict[str, Any]:
        """Generate recommended hyperparameter ranges based on top performers."""
        if not top_performers:
            return {}
        
        recommendations = {}
        all_keys = set()
        for perf in top_performers:
            all_keys.update(perf["hyperparameters"].keys())
        
        for key in all_keys:
            values = [perf["hyperparameters"].get(key) for perf in top_performers if key in perf["hyperparameters"]]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            if numeric_values:
                mean_val = np.mean(numeric_values)
                std_val = np.std(numeric_values)
                recommendations[key] = {
                    "suggested_min": max(0, mean_val - std_val),
                    "suggested_max": mean_val + std_val,
                    "suggested_value": mean_val
                }
        
        return recommendations
    
    def generate_optimized_hyperparameters(self, experiment_id: int, algorithm: str) -> Dict[str, Any]:
        """Generate optimized hyperparameters based on experiment history."""
        analysis = self.analyze_experiment_performance(experiment_id)
        
        if analysis.get("status") in ["insufficient_data", "no_metric_data"]:
            return self._get_default_hyperparameters(algorithm)
        
        insights = analysis.get("parameter_insights", {})
        algo_insights = insights.get(algorithm, {})
        
        if not algo_insights:
            return self._get_default_hyperparameters(algorithm)
        
        # Use recommended ranges from top performers
        recommended = algo_insights.get("recommended_ranges", {})
        optimized_params = {}
        
        for param, ranges in recommended.items():
            if "suggested_value" in ranges:
                # Add some variation to explore nearby space
                base_value = ranges["suggested_value"]
                if param in ["n_estimators", "max_depth"]:
                    # Integer parameters
                    optimized_params[param] = int(base_value)
                else:
                    # Float parameters
                    optimized_params[param] = float(base_value)
        
        # Fill in missing parameters with defaults
        default_params = self._get_default_hyperparameters(algorithm)
        for key, value in default_params.items():
            if key not in optimized_params:
                optimized_params[key] = value
        
        logger.info(f"Generated optimized hyperparameters for {algorithm}: {optimized_params}")
        return optimized_params
    
    def _get_default_hyperparameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default hyperparameters for an algorithm."""
        defaults = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "logistic_regression": {
                "C": 1.0,
                "max_iter": 1000
            },
            "svm": {
                "C": 1.0,
                "kernel": "rbf"
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3
            }
        }
        return defaults.get(algorithm, {})
    
    def schedule_retraining(self, experiment_id: int, algorithm: str = None) -> Dict[str, Any]:
        """Schedule an optimized retraining job."""
        try:
            experiment = self.exp_store.get_experiment(experiment_id)
            if not experiment:
                return {"status": "error", "message": "Experiment not found"}
            
            # Get the best run to retrain
            best_run = self.exp_store.get_best_run(experiment_id)
            if not best_run:
                return {"status": "error", "message": "No completed runs found"}
            
            # Use specified algorithm or the best performing one
            target_algorithm = algorithm or best_run.algorithm
            
            # Generate optimized hyperparameters
            optimized_params = self.generate_optimized_hyperparameters(experiment_id, target_algorithm)
            
            # Create new run
            run = self.exp_store.create_run(
                experiment_id=experiment_id,
                algorithm=target_algorithm,
                hyperparameters=optimized_params,
                model_id=f"scheduled_{int(datetime.now().timestamp())}"
            )
            
            # Update run status
            self.exp_store.update_run_status(run.id, 'running')
            
            # Prepare training configuration
            training_config = {
                'run_id': run.id,
                'dataset_source': experiment.dataset_id.split('_')[0] if '_' in experiment.dataset_id else 'unknown',
                'dataset_id': experiment.dataset_id,
                'target_column': experiment.target_column,
                'algorithm': target_algorithm,
                'test_size': 0.2,
                'cv_folds': 5,
                'auto_hyperparameter_tuning': False  # Use our optimized params
            }
            
            # Schedule the training task
            task = train_model_task.delay(training_config)
            
            # Update run with task ID
            run.celery_task_id = task.id
            self.db.commit()
            
            return {
                "status": "scheduled",
                "run_id": run.id,
                "task_id": task.id,
                "algorithm": target_algorithm,
                "optimized_hyperparameters": optimized_params,
                "message": f"Retraining scheduled for experiment {experiment_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to schedule retraining: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close database session."""
        self.db.close()


class TrainingScheduler:
    """Automated training scheduler for continuous improvement."""
    
    def __init__(self):
        self.optimizer = HyperparameterOptimizer()
    
    def find_experiments_for_optimization(self, min_runs: int = 3, 
                                         max_age_days: int = 7) -> List[Dict[str, Any]]:
        """Find experiments that are good candidates for optimization."""
        db = SessionLocal()
        exp_store = ExperimentStore(db)
        
        try:
            experiments = exp_store.get_experiments(limit=100)
            candidates = []
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            for experiment in experiments:
                runs = exp_store.get_runs_by_experiment(experiment.id)
                completed_runs = [r for r in runs if r.status == "completed"]
                recent_runs = [r for r in completed_runs if r.created_at >= cutoff_date]
                
                if len(completed_runs) >= min_runs and len(recent_runs) > 0:
                    best_run = exp_store.get_best_run(experiment.id)
                    candidates.append({
                        "experiment_id": experiment.id,
                        "name": experiment.name,
                        "total_runs": len(completed_runs),
                        "recent_runs": len(recent_runs),
                        "best_score": best_run.validation_metrics.get("accuracy", 0) if best_run and best_run.validation_metrics else 0,
                        "last_run_date": max(r.created_at for r in recent_runs)
                    })
            
            # Sort by potential for improvement (fewer recent runs = more potential)
            candidates.sort(key=lambda x: (x["best_score"], -x["recent_runs"]))
            return candidates
            
        finally:
            db.close()
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete optimization cycle."""
        candidates = self.find_experiments_for_optimization()
        results = {
            "candidates_found": len(candidates),
            "scheduled_jobs": [],
            "errors": []
        }
        
        for candidate in candidates[:3]:  # Limit to top 3 candidates
            try:
                result = self.optimizer.schedule_retraining(candidate["experiment_id"])
                if result["status"] == "scheduled":
                    results["scheduled_jobs"].append({
                        "experiment_id": candidate["experiment_id"],
                        "experiment_name": candidate["name"],
                        "run_id": result["run_id"],
                        "task_id": result["task_id"]
                    })
                else:
                    results["errors"].append({
                        "experiment_id": candidate["experiment_id"],
                        "error": result.get("message", "Unknown error")
                    })
            except Exception as e:
                results["errors"].append({
                    "experiment_id": candidate["experiment_id"],
                    "error": str(e)
                })
        
        return results
    
    def close(self):
        """Close optimizer."""
        self.optimizer.close()


# Global instances
hyperparameter_optimizer = HyperparameterOptimizer()
training_scheduler = TrainingScheduler()
