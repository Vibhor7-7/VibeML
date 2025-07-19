"""
Data preprocessing utilities for dataset analysis and cleaning.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from db_schema.dataset import ColumnInfo, DataType
import logging

logger = logging.getLogger(__name__)


def generate_column_schema(df: pd.DataFrame) -> List[ColumnInfo]:
    """
    Generate detailed column schema from DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
    
    Returns:
        List of ColumnInfo objects with detailed statistics
    """
    columns_info = []
    
    for col in df.columns:
        col_data = df[col]
        
        # Determine data type
        data_type = _infer_data_type(col_data)
        
        # Basic statistics
        null_count = int(col_data.isnull().sum())
        unique_count = int(col_data.nunique())
        
        column_info = ColumnInfo(
            name=col,
            data_type=data_type,
            null_count=null_count,
            unique_count=unique_count
        )
        
        # Add numeric statistics
        if data_type in [DataType.INTEGER, DataType.FLOAT]:
            try:
                column_info.min_value = float(col_data.min()) if not pd.isna(col_data.min()) else None
                column_info.max_value = float(col_data.max()) if not pd.isna(col_data.max()) else None
                column_info.mean_value = float(col_data.mean()) if not pd.isna(col_data.mean()) else None
                column_info.std_value = float(col_data.std()) if not pd.isna(col_data.std()) else None
            except (TypeError, ValueError):
                logger.warning(f"Could not calculate numeric statistics for column: {col}")
        
        # Add categorical statistics
        if data_type in [DataType.STRING, DataType.CATEGORICAL]:
            try:
                top_values = col_data.value_counts().head(5).index.tolist()
                column_info.top_values = [str(val) for val in top_values]
            except Exception:
                logger.warning(f"Could not calculate top values for column: {col}")
        
        columns_info.append(column_info)
    
    return columns_info


def _infer_data_type(series: pd.Series) -> DataType:
    """
    Infer the data type of a pandas Series.
    
    Args:
        series: pandas Series to analyze
    
    Returns:
        DataType enum value
    """
    # Check for boolean
    if pd.api.types.is_bool_dtype(series):
        return DataType.BOOLEAN
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return DataType.DATETIME
    
    # Check for numeric types
    if pd.api.types.is_integer_dtype(series):
        return DataType.INTEGER
    
    if pd.api.types.is_float_dtype(series):
        return DataType.FLOAT
    
    # For object/string types, determine if categorical
    if pd.api.types.is_object_dtype(series):
        # Heuristic: if unique values are less than 50% of total, consider categorical
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.5 and series.nunique() < 50:
            return DataType.CATEGORICAL
        else:
            return DataType.STRING
    
    # Default to string
    return DataType.STRING


def impute_missing_values(df: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """
    Impute missing values in DataFrame.
    
    Args:
        df: DataFrame with missing values
        strategy: Imputation strategy ('auto', 'mean', 'median', 'mode', 'drop', 'zero')
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() == 0:
            continue
        
        col_dtype = _infer_data_type(df_imputed[col])
        
        try:
            if strategy == "auto":
                # Choose strategy based on data type
                if col_dtype in [DataType.INTEGER, DataType.FLOAT]:
                    # Use median for numeric data
                    fill_value = df_imputed[col].median()
                elif col_dtype in [DataType.STRING, DataType.CATEGORICAL]:
                    # Use mode for categorical data
                    fill_value = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else "Unknown"
                elif col_dtype == DataType.BOOLEAN:
                    # Use mode for boolean
                    fill_value = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else False
                else:
                    fill_value = "Unknown"
            
            elif strategy == "mean":
                if col_dtype in [DataType.INTEGER, DataType.FLOAT]:
                    fill_value = df_imputed[col].mean()
                else:
                    continue  # Skip non-numeric columns
            
            elif strategy == "median":
                if col_dtype in [DataType.INTEGER, DataType.FLOAT]:
                    fill_value = df_imputed[col].median()
                else:
                    continue
            
            elif strategy == "mode":
                mode_values = df_imputed[col].mode()
                fill_value = mode_values.iloc[0] if not mode_values.empty else ("Unknown" if col_dtype == DataType.STRING else 0)
            
            elif strategy == "zero":
                fill_value = 0 if col_dtype in [DataType.INTEGER, DataType.FLOAT] else "Unknown"
            
            elif strategy == "drop":
                # This will be handled at DataFrame level
                continue
            
            else:
                logger.warning(f"Unknown imputation strategy: {strategy}")
                continue
            
            df_imputed[col].fillna(fill_value, inplace=True)
            logger.info(f"Imputed {col} with {strategy} strategy (value: {fill_value})")
            
        except Exception as e:
            logger.warning(f"Could not impute column {col}: {str(e)}")
    
    # Handle 'drop' strategy at DataFrame level
    if strategy == "drop":
        initial_rows = len(df_imputed)
        df_imputed = df_imputed.dropna()
        logger.info(f"Dropped {initial_rows - len(df_imputed)} rows with missing values")
    
    return df_imputed


def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, List[int]]:
    """
    Detect outliers in numeric columns.
    
    Args:
        df: DataFrame to analyze
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Dictionary mapping column names to lists of outlier indices
    """
    outliers = {}
    
    for col in df.columns:
        if _infer_data_type(df[col]) not in [DataType.INTEGER, DataType.FLOAT]:
            continue
        
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        try:
            if method == "iqr":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_indices = col_data[outlier_mask].index.tolist()
            
            elif method == "zscore":
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outlier_indices = col_data[z_scores > threshold].index.tolist()
            
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            if outlier_indices:
                outliers[col] = outlier_indices
                logger.info(f"Found {len(outlier_indices)} outliers in column {col}")
        
        except Exception as e:
            logger.warning(f"Could not detect outliers in column {col}: {str(e)}")
    
    return outliers


def encode_categorical_variables(df: pd.DataFrame, encoding_method: str = "onehot") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame with categorical variables
        encoding_method: Encoding method ('onehot', 'label', 'target')
    
    Returns:
        Tuple of (encoded DataFrame, encoding information)
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    for col in df.columns:
        if _infer_data_type(df[col]) not in [DataType.STRING, DataType.CATEGORICAL]:
            continue
        
        try:
            if encoding_method == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=True)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                encoding_info[col] = {
                    "method": "onehot",
                    "categories": dummies.columns.tolist()
                }
            
            elif encoding_method == "label":
                # Label encoding
                unique_values = df_encoded[col].dropna().unique()
                label_map = {val: idx for idx, val in enumerate(unique_values)}
                label_map[np.nan] = -1  # Handle NaN
                
                df_encoded[col] = df_encoded[col].map(label_map).fillna(-1)
                encoding_info[col] = {
                    "method": "label",
                    "label_map": label_map
                }
            
            else:
                logger.warning(f"Unknown encoding method: {encoding_method}")
                continue
            
            logger.info(f"Encoded categorical column {col} using {encoding_method}")
        
        except Exception as e:
            logger.warning(f"Could not encode column {col}: {str(e)}")
    
    return df_encoded, encoding_info


def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Dictionary with dataset summary
    """
    try:
        summary = {
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "memory_usage": {
                "total_bytes": int(df.memory_usage(deep=True).sum()),
                "per_column": df.memory_usage(deep=True).to_dict()
            },
            "missing_values": {
                "total": int(df.isnull().sum().sum()),
                "per_column": df.isnull().sum().to_dict(),
                "percentage": (df.isnull().sum() / len(df) * 100).to_dict()
            },
            "data_types": {
                "counts": df.dtypes.value_counts().to_dict(),
                "per_column": df.dtypes.astype(str).to_dict()
            },
            "duplicates": {
                "count": int(df.duplicated().sum()),
                "percentage": float(df.duplicated().sum() / len(df) * 100)
            }
        }
        
        # Add numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Add categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                cat_summary[col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(5).to_dict()
                }
            summary["categorical_summary"] = cat_summary
        
        return summary
    
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        return {"error": f"Failed to generate summary: {str(e)}"}


def create_sample_preview(df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
    """
    Create a preview of the dataset with sample rows.
    
    Args:
        df: DataFrame to preview
        n_rows: Number of rows to include in preview
    
    Returns:
        Dictionary with preview data
    """
    try:
        sample_df = df.head(n_rows)
        
        # Convert to JSON-serializable format
        preview_data = []
        for _, row in sample_df.iterrows():
            row_dict = {}
            for col, value in row.items():
                # Handle different data types for JSON serialization
                if pd.isna(value):
                    row_dict[col] = None
                elif isinstance(value, np.integer):
                    row_dict[col] = int(value)
                elif isinstance(value, np.floating):
                    row_dict[col] = float(value)
                elif isinstance(value, (np.bool_, np.bool)):
                    row_dict[col] = bool(value)
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    row_dict[col] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                else:
                    row_dict[col] = str(value)
            
            preview_data.append(row_dict)
        
        return {
            "sample_data": preview_data,
            "total_rows": len(df),
            "sample_size": len(preview_data),
            "columns": list(df.columns)
        }
    
    except Exception as e:
        logger.error(f"Error creating sample preview: {str(e)}")
        return {
            "error": f"Failed to create preview: {str(e)}",
            "sample_data": [],
            "total_rows": 0,
            "sample_size": 0,
            "columns": []
        }
