"""
Import router for dataset upload and management endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
import io

from ..models.experiment_store import get_db
from ..db_schema.dataset import (
    DatasetUploadRequest, DatasetUploadResponse, DatasetPreview, 
    DatasetListResponse, DatasetInfo, ColumnInfo, DataType
)

router = APIRouter(prefix="/import", tags=["data-import"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(None),
    description: str = Form(None),
    target_column: str = Form(None),
    validation_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Upload a dataset file with optional validation set."""
    try:
        # Read main file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or Excel files.")
        
        # Read validation file if provided
        validation_df = None
        validation_message = ""
        if validation_file:
            validation_content = await validation_file.read()
            
            # Determine validation file type and read accordingly
            if validation_file.filename.endswith('.csv'):
                validation_df = pd.read_csv(io.BytesIO(validation_content))
            elif validation_file.filename.endswith(('.xlsx', '.xls')):
                validation_df = pd.read_excel(io.BytesIO(validation_content))
            else:
                raise HTTPException(status_code=400, detail="Unsupported validation file format. Please use CSV or Excel files.")
            
            # Validate that validation set has same columns as training set
            missing_cols = set(df.columns) - set(validation_df.columns)
            extra_cols = set(validation_df.columns) - set(df.columns)
            
            if missing_cols or extra_cols:
                error_msg = "Validation set columns do not match training set columns."
                if missing_cols:
                    error_msg += f" Missing columns: {list(missing_cols)}."
                if extra_cols:
                    error_msg += f" Extra columns: {list(extra_cols)}."
                raise HTTPException(status_code=400, detail=error_msg)
            
            validation_message = f" Validation set ({len(validation_df)} rows) also uploaded."
        
        # Generate dataset info
        dataset_name = dataset_name or file.filename.split('.')[0]
        dataset_info = _generate_dataset_info(df, dataset_name, target_column)
        
        # Create preview with sample data
        sample_data = df.head(5).to_dict('records')
        preview = DatasetPreview(
            info=dataset_info,
            sample_data=sample_data,
            head_rows=len(sample_data)
        )
        
        # Save dataset to temporary storage for training
        import tempfile
        import os
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Analyze columns for target selection suggestions
        def suggest_target_columns(df: pd.DataFrame) -> dict:
            """Suggest good target columns and warn about problematic ones."""
            suggestions = {
                'good_targets': [],
                'avoid_targets': [],
                'warnings': []
            }
            
            for col in df.columns:
                col_data = df[col]
                unique_ratio = col_data.nunique() / len(col_data)
                
                # Skip likely ID columns
                if unique_ratio > 0.95:
                    suggestions['avoid_targets'].append({
                        'column': col,
                        'reason': f'Appears to be unique identifier ({col_data.nunique()}/{len(col_data)} unique values)'
                    })
                    continue
                
                # Check if it's a good classification target
                if pd.api.types.is_numeric_dtype(col_data):
                    if col_data.nunique() <= 10 and unique_ratio < 0.5:
                        # Good numeric classification target
                        class_counts = col_data.value_counts()
                        min_class_size = class_counts.min()
                        if min_class_size >= 2:
                            suggestions['good_targets'].append({
                                'column': col,
                                'type': 'classification',
                                'classes': col_data.nunique(),
                                'reason': f'Numeric with {col_data.nunique()} balanced classes'
                            })
                        else:
                            suggestions['warnings'].append({
                                'column': col,
                                'reason': f'Has classes with only {min_class_size} sample(s)'
                            })
                    elif col_data.nunique() > 10 and unique_ratio > 0.1:
                        # Good regression target
                        suggestions['good_targets'].append({
                            'column': col,
                            'type': 'regression',
                            'reason': f'Continuous numeric values ({col_data.nunique()} unique)'
                        })
                else:
                    # Categorical target
                    class_counts = col_data.value_counts()
                    min_class_size = class_counts.min()
                    if min_class_size >= 2 and col_data.nunique() <= 20:
                        suggestions['good_targets'].append({
                            'column': col,
                            'type': 'classification',
                            'classes': col_data.nunique(),
                            'reason': f'Categorical with {col_data.nunique()} balanced classes'
                        })
                    elif min_class_size < 2:
                        suggestions['warnings'].append({
                            'column': col,
                            'reason': f'Has classes with only {min_class_size} sample(s)'
                        })
            
            return suggestions
        
        target_suggestions = suggest_target_columns(df)
        
        # Generate a unique dataset ID
        dataset_id = f"dataset_{int(time.time() * 1000000)}"
        
        # Save the dataset to temp directory
        temp_dir = tempfile.gettempdir()
        dataset_path = os.path.join(temp_dir, f"dataset_upload_{dataset_id}.csv")
        backup_path = os.path.join(temp_dir, f"{dataset_id}.csv")
        
        # Save the dataset files
        try:
            df.to_csv(dataset_path, index=False)
            df.to_csv(backup_path, index=False)
            logger.info(f"Successfully saved training dataset to {dataset_path}")
            
            # Save validation file if provided
            if validation_df is not None:
                validation_path = os.path.join(temp_dir, f"validation_upload_{dataset_id}.csv")
                validation_backup_path = os.path.join(temp_dir, f"validation_{dataset_id}.csv")
                
                validation_df.to_csv(validation_path, index=False)
                validation_df.to_csv(validation_backup_path, index=False)
                logger.info(f"Successfully saved validation dataset to {validation_path}")
                
        except Exception as e:
            logger.error(f"Error saving dataset files: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save dataset files: {str(e)}")
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            message=f"Dataset '{dataset_name}' uploaded successfully to {dataset_path}.{validation_message}",
            preview=preview,
            target_suggestions=target_suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(db: Session = Depends(get_db)):
    """List all available datasets."""
    try:
        # TODO: Implement actual dataset storage and retrieval
        datasets = []
        
        return DatasetListResponse(
            datasets=datasets,
            total_count=len(datasets)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, db: Session = Depends(get_db)):
    """Get preview of a specific dataset."""
    try:
        # TODO: Implement dataset retrieval and preview
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset preview: {str(e)}")


def _generate_dataset_info(df: pd.DataFrame, name: str, target_column: str = None) -> DatasetInfo:
    """Generate dataset information from DataFrame."""
    columns_info = []
    
    for col in df.columns:
        col_data = df[col]
        
        # Determine data type
        if pd.api.types.is_integer_dtype(col_data):
            data_type = DataType.INTEGER
        elif pd.api.types.is_float_dtype(col_data):
            data_type = DataType.FLOAT
        elif pd.api.types.is_bool_dtype(col_data):
            data_type = DataType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            data_type = DataType.DATETIME
        else:
            data_type = DataType.STRING
            if col_data.nunique() / len(col_data) < 0.5:  # Heuristic for categorical
                data_type = DataType.CATEGORICAL
        
        # Calculate statistics
        null_count = col_data.isnull().sum()
        unique_count = col_data.nunique()
        
        column_info = ColumnInfo(
            name=col,
            data_type=data_type,
            null_count=int(null_count),
            unique_count=int(unique_count)
        )
        
        # Add numeric statistics if applicable
        if data_type in [DataType.INTEGER, DataType.FLOAT]:
            column_info.min_value = float(col_data.min()) if not pd.isna(col_data.min()) else None
            column_info.max_value = float(col_data.max()) if not pd.isna(col_data.max()) else None
            column_info.mean_value = float(col_data.mean()) if not pd.isna(col_data.mean()) else None
            column_info.std_value = float(col_data.std()) if not pd.isna(col_data.std()) else None
        
        # Add top values for categorical/string columns
        if data_type in [DataType.STRING, DataType.CATEGORICAL]:
            top_values = col_data.value_counts().head(5).index.tolist()
            column_info.top_values = [str(val) for val in top_values]
        
        columns_info.append(column_info)
    
    return DatasetInfo(
        name=name,
        rows=len(df),
        columns=len(df.columns),
        size_bytes=df.memory_usage(deep=True).sum(),
        column_info=columns_info,
        target_column=target_column
    )
