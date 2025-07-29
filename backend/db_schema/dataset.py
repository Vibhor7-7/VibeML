"""
Pydantic schemas for dataset-related operations.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DataType(str, Enum):
    """Supported data types for columns."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


class ColumnInfo(BaseModel):
    """Schema for column information."""
    name: str = Field(..., description="Column name")
    data_type: DataType = Field(..., description="Data type of the column")
    null_count: int = Field(0, description="Number of null values")
    unique_count: int = Field(0, description="Number of unique values")
    min_value: Optional[Union[int, float]] = Field(None, description="Minimum value for numeric columns")
    max_value: Optional[Union[int, float]] = Field(None, description="Maximum value for numeric columns")
    mean_value: Optional[float] = Field(None, description="Mean value for numeric columns")
    std_value: Optional[float] = Field(None, description="Standard deviation for numeric columns")
    top_values: Optional[List[str]] = Field(None, description="Most frequent values")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetInfo(BaseModel):
    """Schema for dataset information."""
    name: str = Field(..., description="Dataset name")
    file_path: Optional[str] = Field(None, description="Path to the dataset file")
    rows: int = Field(0, description="Number of rows")
    columns: int = Field(0, description="Number of columns")
    size_bytes: int = Field(0, description="File size in bytes")
    upload_timestamp: Optional[datetime] = Field(None, description="When dataset was uploaded")
    column_info: List[ColumnInfo] = Field(default_factory=list, description="Information about each column")
    target_column: Optional[str] = Field(None, description="Target column for ML")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetPreview(BaseModel):
    """Schema for dataset preview with sample data."""
    info: DatasetInfo
    sample_data: List[Dict[str, Any]] = Field(..., description="Sample rows from the dataset")
    head_rows: int = Field(5, description="Number of head rows included")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetUploadRequest(BaseModel):
    """Schema for dataset upload request."""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    target_column: Optional[str] = Field(None, description="Target column for ML")
    

class DatasetUploadResponse(BaseModel):
    """Schema for dataset upload response."""
    dataset_id: str = Field(..., description="Unique dataset ID")
    message: str = Field(..., description="Upload status message")
    preview: DatasetPreview = Field(..., description="Dataset preview")
    target_suggestions: Optional[dict] = Field(None, description="Suggestions for target column selection")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetListResponse(BaseModel):
    """Schema for listing datasets."""
    datasets: List[DatasetInfo] = Field(..., description="List of available datasets")
    total_count: int = Field(..., description="Total number of datasets")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataPreprocessingOptions(BaseModel):
    """Schema for data preprocessing options."""
    handle_missing_values: bool = Field(True, description="Whether to handle missing values")
    missing_value_strategy: str = Field("mean", description="Strategy for handling missing values")
    encode_categorical: bool = Field(True, description="Whether to encode categorical variables")
    normalize_features: bool = Field(False, description="Whether to normalize/scale features")
    remove_outliers: bool = Field(False, description="Whether to remove outliers")
    feature_selection: bool = Field(False, description="Whether to perform feature selection")
    train_test_split_ratio: float = Field(0.2, description="Test set ratio for train/test split")


class KaggleImportRequest(BaseModel):
    """Schema for Kaggle dataset import request."""
    dataset_name: str = Field(..., description="Kaggle dataset name (e.g., 'titanic/titanic')")
    file_name: Optional[str] = Field(None, description="Specific file to extract from dataset")
    name: Optional[str] = Field(None, description="Custom name for the dataset")
    description: Optional[str] = Field(None, description="Dataset description")


class OpenMLImportRequest(BaseModel):
    """Schema for OpenML dataset import request."""
    dataset_id: int = Field(..., description="OpenML dataset ID")
    name: Optional[str] = Field(None, description="Custom name for the dataset")
    description: Optional[str] = Field(None, description="Dataset description")


class DatasetImportResponse(BaseModel):
    """Schema for dataset import response."""
    success: bool = Field(..., description="Whether import was successful")
    message: str = Field(..., description="Import status message")
    dataset_id: Optional[str] = Field(None, description="Generated dataset ID")
    dataset_info: Optional[DatasetInfo] = Field(None, description="Dataset information")
    preview: Optional[Dict[str, Any]] = Field(None, description="Dataset preview")
    column_schema: Optional[List[ColumnInfo]] = Field(None, description="Column schema information")
    source: str = Field(..., description="Data source (kaggle, openml, upload)")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DatasetSearchRequest(BaseModel):
    """Schema for dataset search request."""
    query: Optional[str] = Field(None, description="Search query")
    source: str = Field("all", description="Data source to search (kaggle, openml, all)")
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional search filters")


class DatasetSearchResponse(BaseModel):
    """Schema for dataset search response."""
    datasets: List[Dict[str, Any]] = Field(..., description="List of found datasets")
    total_count: int = Field(..., description="Total number of results")
    query: Optional[str] = Field(None, description="Search query used")
    source: str = Field(..., description="Data source searched")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
