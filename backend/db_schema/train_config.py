"""
Pydantic schemas for training configuration and retraining.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MLAlgorithm(str, Enum):
    """Supported ML algorithms."""
    AUTO = "auto"  # AutoML - automatically select best algorithm
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    KNN = "knn"
    DECISION_TREE = "decision_tree"
    NAIVE_BAYES = "naive_bayes"
    NEURAL_NETWORK = "neural_network"
    XG_BOOST = "xgboost"
    LIGHT_GBM = "lightgbm"


class ProblemType(str, Enum):
    """Types of ML problems."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HyperparameterConfig(BaseModel):
    """Schema for hyperparameter configuration."""
    parameter_name: str = Field(..., description="Name of the hyperparameter")
    value: Any = Field(..., description="Value of the hyperparameter")
    parameter_type: str = Field(..., description="Type of the parameter (int, float, str, bool)")
    description: Optional[str] = Field(None, description="Description of the parameter")


class TrainConfig(BaseModel):
    """Schema for training configuration."""
    model_name: str = Field(..., description="Name for the trained model")
    dataset_id: str = Field(..., description="ID of the dataset to use")
    dataset_source: str = Field(..., description="Source of the dataset (openml, kaggle, upload)")
    target_column: str = Field(..., description="Target column for prediction")
    problem_type: ProblemType = Field(..., description="Type of ML problem")
    algorithm: MLAlgorithm = Field(..., description="ML algorithm to use")
    
    # Feature configuration
    feature_columns: Optional[List[str]] = Field(None, description="Specific columns to use as features")
    exclude_columns: Optional[List[str]] = Field(None, description="Columns to exclude from training")
    
    # Training parameters
    test_size: float = Field(0.2, description="Proportion of data for testing", ge=0.1, le=0.5)
    random_state: int = Field(42, description="Random seed for reproducibility")
    cross_validation_folds: int = Field(5, description="Number of CV folds", ge=2, le=10)
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific hyperparameters")
    
    # Advanced options
    auto_hyperparameter_tuning: bool = Field(False, description="Whether to perform automatic hyperparameter tuning")
    hyperparameter_search_space: Optional[Dict[str, List[Any]]] = Field(None, description="Search space for hyperparameter tuning")
    scoring_metric: Optional[str] = Field(None, description="Metric to optimize during training")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RetrainConfig(BaseModel):
    """Schema for model retraining configuration."""
    model_id: str = Field(..., description="ID of the model to retrain")
    new_dataset_id: Optional[str] = Field(None, description="New dataset ID (if different from original)")
    
    # Retraining options
    incremental_learning: bool = Field(False, description="Whether to use incremental learning")
    retrain_from_scratch: bool = Field(True, description="Whether to retrain from scratch")
    merge_with_previous_data: bool = Field(False, description="Whether to merge with previous training data")
    
    # Updated parameters
    updated_hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Updated hyperparameters")
    updated_feature_columns: Optional[List[str]] = Field(None, description="Updated feature columns")
    updated_test_size: Optional[float] = Field(None, description="Updated test size")
    
    # Validation
    validate_against_previous: bool = Field(True, description="Whether to validate against previous model performance")
    minimum_improvement_threshold: float = Field(0.0, description="Minimum improvement required to replace model")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingJob(BaseModel):
    """Schema for training job information."""
    job_id: str = Field(..., description="Unique job ID")
    model_name: str = Field(..., description="Name of the model being trained")
    status: TrainingStatus = Field(..., description="Current status of the training job")
    algorithm: MLAlgorithm = Field(..., description="Algorithm being used")
    problem_type: ProblemType = Field(..., description="Type of ML problem")
    
    # Progress tracking
    progress_percentage: float = Field(0.0, description="Training progress percentage", ge=0.0, le=100.0)
    current_step: Optional[str] = Field(None, description="Current training step")
    
    # Celery task tracking
    celery_task_id: Optional[str] = Field(None, description="Celery background task ID")
    
    # Timestamps
    created_at: datetime = Field(..., description="When the job was created")
    started_at: Optional[datetime] = Field(None, description="When training started")
    completed_at: Optional[datetime] = Field(None, description="When training completed")
    
    # Results
    training_metrics: Optional[Dict[str, float]] = Field(None, description="Training metrics")
    validation_metrics: Optional[Dict[str, float]] = Field(None, description="Validation metrics")
    error_message: Optional[str] = Field(None, description="Error message if training failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrainingJobResponse(BaseModel):
    """Schema for training job response."""
    job: TrainingJob = Field(..., description="Training job information")
    message: str = Field(..., description="Response message")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelPerformanceMetrics(BaseModel):
    """Schema for model performance metrics."""
    accuracy: Optional[float] = Field(None, description="Accuracy score")
    precision: Optional[float] = Field(None, description="Precision score")
    recall: Optional[float] = Field(None, description="Recall score")
    f1_score: Optional[float] = Field(None, description="F1 score")
    roc_auc: Optional[float] = Field(None, description="ROC AUC score")
    
    # Regression metrics
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    r2_score: Optional[float] = Field(None, description="R-squared score")
    
    # Additional metrics
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
