"""
Training engine with AutoML capabilities using scikit-learn and basic auto-ML.
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AutoMLEngine:
    """AutoML training engine with automatic algorithm selection and hyperparameter tuning."""
    
    def __init__(self, model_storage_path: str = "./model_storage"):
        self.model_storage_path = model_storage_path
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Define algorithm pools
        self.classification_algorithms = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'naive_bayes': GaussianNB(),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        self.regression_algorithms = {
            'random_forest': RandomForestRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'svm': SVR(),
            'knn': KNeighborsRegressor(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Hyperparameter grids for tuning
        self.hyperparameter_grids = {
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [3, 5, 10, None],
                'model__min_samples_split': [2, 5, 10]
            },
            'logistic_regression': {
                'model__C': [0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'lbfgs']
            },
            'svm': {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['rbf', 'linear']
            },
            'knn': {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.15, validation_size: float = 0.15, is_retrain: bool = False) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Prepare data for training with automatic preprocessing.
        Now implements 85%/15% split strategy with reserved validation data.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of training data for testing (default 15% of training split)
            validation_size: Proportion of total data reserved for retraining (default 15%)
            is_retrain: Whether this is a retraining operation
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, preprocessor)
        """
        logger.info("Starting data preparation...")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values in target
        if y.isnull().sum() > 0:
            logger.warning(f"Removing {y.isnull().sum()} rows with missing target values")
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
        
        # Identify column types
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Found {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_columns),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
            ],
            remainder='passthrough'
        )
        
        if is_retrain:
            # For retraining, use the reserved validation data (last 15% of dataset)
            validation_start_idx = int(len(X) * (1 - validation_size))
            X_validation = X.iloc[validation_start_idx:]
            y_validation = y.iloc[validation_start_idx:]
            
            # Split validation data into train/test (80/20)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_validation, y_validation, test_size=0.2, random_state=42, 
                    stratify=y_validation if self._is_classification(y_validation) else None
                )
            except ValueError as e:
                if "least populated class" in str(e):
                    # Fall back to non-stratified split if stratification fails
                    logger.warning(f"Stratified split failed ({str(e)}), falling back to random split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_validation, y_validation, test_size=0.2, random_state=42, stratify=None
                    )
                else:
                    raise e
            logger.info(f"Retraining with validation data - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        else:
            # For initial training, use first 85% of data and split it 80/20
            training_end_idx = int(len(X) * (1 - validation_size))
            X_training_pool = X.iloc[:training_end_idx]
            y_training_pool = y.iloc[:training_end_idx]
            
            # Split the 85% into train/test
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_training_pool, y_training_pool, test_size=test_size/(1-validation_size), random_state=42,
                    stratify=y_training_pool if self._is_classification(y_training_pool) else None
                )
            except ValueError as e:
                if "least populated class" in str(e):
                    # Fall back to non-stratified split if stratification fails
                    logger.warning(f"Stratified split failed ({str(e)}), falling back to random split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_training_pool, y_training_pool, test_size=test_size/(1-validation_size), random_state=42, stratify=None
                    )
                else:
                    raise e
            logger.info(f"Initial training with 85% data - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Reserved: {len(X) - training_end_idx}")
        
        return X_train, X_test, y_train, y_test, preprocessor
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Determine if the problem is classification or regression."""
        # Check if target is numeric and has many unique values
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = y.nunique() / len(y)
            # If less than 10 unique values or less than 5% unique ratio, treat as classification
            return y.nunique() <= 10 or unique_ratio < 0.05
        else:
            # Non-numeric is always classification
            return True
    
    def auto_train(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        cv_folds: int = 5,
        enable_hyperparameter_tuning: bool = True,
        algorithms: Optional[list] = None,
        is_retrain: bool = False,
        external_validation_set: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Automatically train and select the best model.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            test_size: Test set proportion from training data
            validation_size: Proportion of total data reserved for retraining
            cv_folds: Cross-validation folds
            enable_hyperparameter_tuning: Whether to tune hyperparameters
            algorithms: Specific algorithms to try (None for all)
            is_retrain: Whether this is a retraining operation
        
        Returns:
            Dictionary with training results and best model info
        """
        logger.info("Starting AutoML training...")
        start_time = datetime.utcnow()
        
        # Validate target column before proceeding
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        y_full = df[target_column]
        
        # Check for too many unique values (likely an ID column)
        unique_ratio = y_full.nunique() / len(y_full)
        if unique_ratio > 0.95:
            raise ValueError(f"Target column '{target_column}' appears to be a unique identifier (95%+ unique values). Please select a different target column for prediction.")
        
        # For classification problems, check minimum class sizes
        if self._is_classification(y_full):
            class_counts = y_full.value_counts()
            min_class_size = class_counts.min()
            
            # If external validation is provided, we need at least 2 samples per class for stratified splitting
            # If no external validation, we need enough for both CV folds and train/test split
            min_required = 2 if external_validation_set is not None else max(cv_folds, 2)
            
            if min_class_size < min_required:
                problematic_classes = class_counts[class_counts < min_required].index.tolist()
                if external_validation_set is not None:
                    error_msg = f"Classification target '{target_column}' has classes with too few samples: {dict(class_counts[class_counts < min_required])}. Each class needs at least 2 samples when using an external validation set."
                else:
                    error_msg = f"Classification target '{target_column}' has classes with too few samples: {dict(class_counts[class_counts < min_required])}. Each class needs at least {min_required} samples for {cv_folds}-fold cross-validation."
                
                raise ValueError(error_msg)
        
        # Prepare data differently based on whether external validation set is provided
        if external_validation_set is not None:
            logger.info("Using external validation set for evaluation")
            
            # Use entire training dataset for training (no internal validation split)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle missing values in target
            if y.isnull().sum() > 0:
                logger.warning(f"Removing {y.isnull().sum()} rows with missing target values from training set")
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]
            
            # Prepare external validation set
            X_val_external = external_validation_set.drop(columns=[target_column])
            y_val_external = external_validation_set[target_column]
            
            # Handle missing values in validation target
            if y_val_external.isnull().sum() > 0:
                logger.warning(f"Removing {y_val_external.isnull().sum()} rows with missing target values from validation set")
                val_mask = ~y_val_external.isnull()
                X_val_external = X_val_external[val_mask]
                y_val_external = y_val_external[val_mask]
            
            # Create train/test split from training data only
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42,
                    stratify=y if self._is_classification(y) else None
                )
            except ValueError as e:
                if "least populated class" in str(e):
                    # Fall back to non-stratified split if stratification fails
                    logger.warning(f"Stratified split failed ({str(e)}), falling back to random split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=None
                    )
                else:
                    raise e
            
            # Identify column types for preprocessing
            numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logger.info(f"Found {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_columns),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
                ],
                remainder='passthrough'
            )
            
            logger.info(f"External validation mode - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, External Val: {X_val_external.shape[0]}")
        else:
            # Use existing prepare_data method for standard train/validation/test split
            X_train, X_test, y_train, y_test, preprocessor = self.prepare_data(
                df, target_column, test_size, validation_size, is_retrain
            )
            X_val_external = None
            y_val_external = None
        
        # Determine problem type
        is_classification = self._is_classification(y_train)
        problem_type = "classification" if is_classification else "regression"
        logger.info(f"Problem type detected: {problem_type}")
        
        # Select algorithms
        if algorithms:
            if is_classification:
                available_algorithms = {k: v for k, v in self.classification_algorithms.items() if k in algorithms}
            else:
                available_algorithms = {k: v for k, v in self.regression_algorithms.items() if k in algorithms}
        else:
            available_algorithms = self.classification_algorithms if is_classification else self.regression_algorithms
        
        best_model = None
        best_score = float('-inf')
        best_algorithm = None
        results = {}
        
        # Try each algorithm
        for algo_name, algorithm in available_algorithms.items():
            logger.info(f"Training {algo_name}...")
            
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', algorithm)
                ])
                
                # Hyperparameter tuning if enabled
                if enable_hyperparameter_tuning and algo_name in self.hyperparameter_grids:
                    param_grid = self.hyperparameter_grids[algo_name]
                    grid_search = GridSearchCV(
                        pipeline, 
                        param_grid, 
                        cv=cv_folds, 
                        scoring='accuracy' if is_classification else 'neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    cv_score = grid_search.best_score_
                else:
                    model = pipeline
                    model.fit(X_train, y_train)
                    best_params = {}
                    
                    # Manual cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train, 
                        cv=cv_folds, 
                        scoring='accuracy' if is_classification else 'neg_mean_squared_error'
                    )
                    cv_score = cv_scores.mean()
                
                # Evaluate on test set
                test_metrics = self._evaluate_model(model, X_test, y_test, is_classification)
                
                # Store results
                results[algo_name] = {
                    'algorithm': algo_name,
                    'cv_score': float(cv_score),
                    'best_params': best_params,
                    'test_metrics': test_metrics,
                    'model': model
                }
                
                # Update best model
                score = cv_score if cv_score > 0 else -cv_score  # Handle negative MSE
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_algorithm = algo_name
                
                logger.info(f"{algo_name} - CV Score: {cv_score:.4f}, Test Score: {test_metrics.get('accuracy' if is_classification else 'r2_score', 'N/A')}")
                
            except Exception as e:
                logger.error(f"Error training {algo_name}: {str(e)}")
                results[algo_name] = {
                    'algorithm': algo_name,
                    'error': str(e)
                }
        
        if best_model is None:
            raise ValueError("No models were successfully trained")
        
        # Generate model ID and save
        model_id = f"model_{int(datetime.utcnow().timestamp())}"
        model_path = self.save_model(best_model, model_id)
        
        # Evaluate on external validation set if provided
        external_validation_metrics = None
        if X_val_external is not None and y_val_external is not None:
            logger.info("Evaluating best model on external validation set")
            external_validation_metrics = self._evaluate_model(best_model, X_val_external, y_val_external, is_classification)
            logger.info(f"External validation metrics: {external_validation_metrics}")
        
        # Calculate training duration
        training_duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Prepare final results
        final_results = {
            'model_id': model_id,
            'model_path': model_path,
            'best_algorithm': best_algorithm,
            'problem_type': problem_type,
            'training_duration_seconds': training_duration,
            'best_cv_score': float(best_score),
            'best_test_metrics': results[best_algorithm]['test_metrics'],
            'external_validation_metrics': external_validation_metrics,
            'best_hyperparameters': results[best_algorithm]['best_params'],
            'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()},
            'dataset_info': {
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'validation_samples': len(X_val_external) if X_val_external is not None else 0,
                'features': X_train.shape[1],
                'target_column': target_column,
                'has_external_validation': X_val_external is not None
            }
        }
        
        logger.info(f"AutoML training completed. Best algorithm: {best_algorithm} with score: {best_score:.4f}")
        return final_results
    
    def _evaluate_model(self, model, X_test, y_test, is_classification: bool) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if is_classification:
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            
            # Handle multi-class vs binary classification
            average = 'weighted' if len(np.unique(y_test)) > 2 else 'binary'
            
            try:
                metrics['precision'] = float(precision_score(y_test, y_pred, average=average, zero_division=0))
                metrics['recall'] = float(recall_score(y_test, y_pred, average=average, zero_division=0))
                metrics['f1_score'] = float(f1_score(y_test, y_pred, average=average, zero_division=0))
            except Exception as e:
                logger.warning(f"Could not calculate precision/recall/f1: {e}")
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
        
        else:
            metrics['mae'] = float(mean_absolute_error(y_test, y_pred))
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['r2_score'] = float(r2_score(y_test, y_pred))
        
        return metrics
    
    def save_model(self, model, model_id: str) -> str:
        """Save model to disk."""
        model_filename = f"{model_id}.pkl"
        model_path = os.path.join(self.model_storage_path, model_filename)
        
        # Save using joblib for better sklearn compatibility
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to: {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """Load model from disk."""
        return joblib.load(model_path)
    
    def predict(self, model_path: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using saved model."""
        model = self.load_model(model_path)
        return model.predict(X)
    
    def predict_proba(self, model_path: str, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using saved model."""
        model = self.load_model(model_path)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")


# Global instance
training_engine = AutoMLEngine()
