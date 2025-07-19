"""
Create test datasets for development and testing.
"""
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification, make_regression
import tempfile

def create_iris_dataset():
    """Create a simple Iris-like dataset."""
    np.random.seed(42)
    
    # Create a classification dataset
    X, y = make_classification(
        n_samples=150,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=3,
        random_state=42
    )
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = pd.Categorical.from_codes(y, categories=['setosa', 'versicolor', 'virginica'])
    
    return df

def create_housing_dataset():
    """Create a housing price regression dataset."""
    np.random.seed(42)
    
    # Create a regression dataset
    X, y = make_regression(
        n_samples=200,
        n_features=8,
        noise=0.1,
        random_state=42
    )
    
    # Create DataFrame with meaningful column names
    columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
               'total_bedrooms', 'population', 'households', 'median_income']
    df = pd.DataFrame(X, columns=columns)
    df['median_house_value'] = y
    
    # Scale values to realistic ranges
    df['longitude'] = -122 + (df['longitude'] - df['longitude'].min()) * 0.5
    df['latitude'] = 37 + (df['latitude'] - df['latitude'].min()) * 0.5
    df['housing_median_age'] = np.abs(df['housing_median_age']) * 5 + 1
    df['total_rooms'] = np.abs(df['total_rooms']) * 100 + 500
    df['total_bedrooms'] = df['total_rooms'] * 0.2
    df['population'] = np.abs(df['population']) * 500 + 1000
    df['households'] = df['population'] * 0.3
    df['median_income'] = np.abs(df['median_income']) * 2 + 3
    df['median_house_value'] = np.abs(df['median_house_value']) * 100000 + 200000
    
    return df

def create_titanic_dataset():
    """Create a Titanic-like dataset."""
    np.random.seed(42)
    
    n_samples = 891
    
    # Create basic features
    passenger_id = range(1, n_samples + 1)
    pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
    age = np.random.normal(29, 14, n_samples)
    age = np.clip(age, 0.42, 80)  # Realistic age range
    sibsp = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.07, 0.01, 0.005, 0.005])
    parch = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.01])
    
    # Calculate fare based on class
    fare = np.where(pclass == 1, np.random.normal(84, 78),
                   np.where(pclass == 2, np.random.normal(20, 13),
                           np.random.normal(13, 11)))
    fare = np.clip(fare, 0, 512)
    
    # Calculate survival probability based on features
    # Higher survival for females, higher class, younger age
    survival_prob = (0.3 + 
                    0.4 * (sex == 'female') + 
                    0.2 * (pclass == 1) + 
                    0.1 * (pclass == 2) - 
                    0.01 * np.maximum(age - 15, 0))
    survival_prob = np.clip(survival_prob, 0.05, 0.95)
    survived = np.random.binomial(1, survival_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'PassengerId': passenger_id,
        'Survived': survived,
        'Pclass': pclass,
        'Sex': sex,
        'Age': age.round(1),
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare.round(2),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    })
    
    return df

def save_test_datasets():
    """Save test datasets to temporary directory."""
    # Save datasets in backend/test_datasets for backend registration
    dataset_dir = os.path.join(os.path.dirname(__file__), 'test_datasets')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create datasets
    iris_df = create_iris_dataset()
    housing_df = create_housing_dataset()
    titanic_df = create_titanic_dataset()
    
    # Save to CSV files
    datasets = {
        'test_iris': iris_df,
        'test_housing': housing_df,
        'test_titanic': titanic_df
    }
    
    saved_files = {}
    for name, df in datasets.items():
        filepath = os.path.join(dataset_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        saved_files[name] = filepath
        print(f"Created {name} dataset: {df.shape} -> {filepath}")
    
    return saved_files

if __name__ == "__main__":
    saved_files = save_test_datasets()
    print("\nTest datasets created successfully!")
    print("You can now use these for testing the training pipeline.")
