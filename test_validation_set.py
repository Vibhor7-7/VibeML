#!/usr/bin/env python3
"""
Test validation set functionality directly.
"""
import sys
import os
import pandas as pd

# Add backend to path
sys.path.insert(0, '/Users/sharvibhor/Desktop/Projects/VibeML-3/backend')

from services.training_engine import AutoMLEngine

def test_validation_set():
    print("Testing validation set functionality...")
    
    # Load training data
    training_data = pd.read_csv('/Users/sharvibhor/Desktop/Projects/VibeML-3/test_iris_mini.csv')
    print(f"Training data shape: {training_data.shape}")
    print(f"Training data target distribution:\n{training_data['species'].value_counts()}")
    
    # Load validation data
    validation_data = pd.read_csv('/Users/sharvibhor/Desktop/Projects/VibeML-3/test_iris_validation.csv')
    print(f"Validation data shape: {validation_data.shape}")
    print(f"Validation data target distribution:\n{validation_data['species'].value_counts()}")
    
    # Test training with external validation set
    engine = AutoMLEngine()
    
    config = {
        'target_column': 'species',
        'algorithm': 'random_forest',
        'test_size': 0.15,
        'validation_size': 0.0,  # Using external validation
        'random_state': 42
    }
    
    print("\nStarting training with external validation set...")
    
    try:
        results = engine.auto_train(
            df=training_data,
            target_column=config['target_column'],
            test_size=config['test_size'],
            validation_size=config['validation_size'],
            cv_folds=2,  # Use 2 folds for small dataset
            algorithms=[config['algorithm']],
            external_validation_set=validation_data
        )
        
        print("Training completed successfully!")
        print(f"Training metrics: {results.get('training_metrics', {})}")
        print(f"Validation metrics: {results.get('validation_metrics', {})}")
        print(f"External validation metrics: {results.get('external_validation_metrics', {})}")
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation_set()
    sys.exit(0 if success else 1)
