"""
Import router for dataset upload and management endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import io

from models.experiment_store import get_db
from db_schema.dataset import (
    DatasetUploadRequest, DatasetUploadResponse, DatasetPreview, 
    DatasetListResponse, DatasetInfo, ColumnInfo, DataType,
    KaggleImportRequest, OpenMLImportRequest, DatasetImportResponse,
    DatasetSearchRequest, DatasetSearchResponse
)
from services.kaggle import kaggle_service
from services.open_ml import openml_service
from utils.preprocessing import generate_column_schema, create_sample_preview, generate_data_summary

router = APIRouter(prefix="/import", tags=["data-import"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = None,
    description: str = None,
    target_column: str = None,
    db: Session = Depends(get_db)
):
    """Upload a dataset file."""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or Excel files.")
        
        # Generate dataset info
        dataset_name = name or file.filename.split('.')[0]
        dataset_info = _generate_dataset_info(df, dataset_name, target_column)
        
        # Create preview with sample data
        sample_data = df.head(5).to_dict('records')
        preview = DatasetPreview(
            info=dataset_info,
            sample_data=sample_data,
            head_rows=len(sample_data)
        )
        
        # TODO: Save dataset to storage
        dataset_id = f"dataset_{hash(dataset_name)}"
        
        return DatasetUploadResponse(
            dataset_id=dataset_id,
            message=f"Dataset '{dataset_name}' uploaded successfully",
            preview=preview
        )
        
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


@router.post("/kaggle", response_model=DatasetImportResponse)
async def import_kaggle_dataset(
    request: KaggleImportRequest,
    db: Session = Depends(get_db)
):
    """Import dataset from Kaggle."""
    try:
        # Download dataset from Kaggle
        df = kaggle_service.download_dataset(
            dataset_name=request.dataset_name,
            file_name=request.file_name
        )
        
        # Generate dataset info
        dataset_name = request.name or request.dataset_name.split('/')[-1]
        
        # Generate column schema
        column_schema = generate_column_schema(df)
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=dataset_name,
            rows=len(df),
            columns=len(df.columns),
            size_bytes=int(df.memory_usage(deep=True).sum()),
            column_info=column_schema,
            target_column=None  # Will be set later during training setup
        )
        
        # Create preview
        preview = create_sample_preview(df, n_rows=10)
        
        # TODO: Save dataset to persistent storage
        dataset_id = f"kaggle_{hash(request.dataset_name)}"
        
        return DatasetImportResponse(
            success=True,
            message=f"Successfully imported Kaggle dataset: {request.dataset_name}",
            dataset_id=dataset_id,
            dataset_info=dataset_info,
            preview=preview,
            column_schema=column_schema,
            source="kaggle"
        )
        
    except Exception as e:
        return DatasetImportResponse(
            success=False,
            message=f"Failed to import Kaggle dataset: {str(e)}",
            source="kaggle"
        )


@router.post("/openml", response_model=DatasetImportResponse)
async def import_openml_dataset(
    request: OpenMLImportRequest,
    db: Session = Depends(get_db)
):
    """Import dataset from OpenML."""
    try:
        # Download dataset from OpenML
        df = openml_service.get_dataset(request.dataset_id)
        
        # Get dataset info from OpenML
        openml_info = openml_service.get_dataset_info(request.dataset_id)
        
        # Generate dataset name
        dataset_name = request.name or openml_info.get('name', f'openml_dataset_{request.dataset_id}')
        
        # Generate column schema
        column_schema = generate_column_schema(df)
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=dataset_name,
            rows=len(df),
            columns=len(df.columns),
            size_bytes=int(df.memory_usage(deep=True).sum()),
            column_info=column_schema,
            target_column=openml_info.get('default_target_attribute')
        )
        
        # Create preview
        preview = create_sample_preview(df, n_rows=10)
        
        # TODO: Save dataset to persistent storage
        dataset_id = f"openml_{request.dataset_id}"
        
        return DatasetImportResponse(
            success=True,
            message=f"Successfully imported OpenML dataset: {dataset_name} (ID: {request.dataset_id})",
            dataset_id=dataset_id,
            dataset_info=dataset_info,
            preview=preview,
            column_schema=column_schema,
            source="openml"
        )
        
    except Exception as e:
        return DatasetImportResponse(
            success=False,
            message=f"Failed to import OpenML dataset: {str(e)}",
            source="openml"
        )


@router.post("/search", response_model=DatasetSearchResponse)
async def search_datasets(
    request: DatasetSearchRequest,
    db: Session = Depends(get_db)
):
    """Search for datasets across different sources."""
    try:
        all_datasets = []
        
        if request.source in ["all", "kaggle"]:
            try:
                kaggle_results = kaggle_service.search_datasets(
                    query=request.query or "",
                    max_results=request.max_results
                )
                all_datasets.extend(kaggle_results.get("datasets", []))
            except Exception as e:
                # Log error but continue with other sources
                pass
        
        if request.source in ["all", "openml"]:
            try:
                openml_results = openml_service.search_datasets(
                    query=request.query,
                    max_results=request.max_results,
                    **(request.filters or {})
                )
                all_datasets.extend(openml_results.get("datasets", []))
            except Exception as e:
                # Log error but continue with other sources
                pass
        
        # Limit results
        all_datasets = all_datasets[:request.max_results]
        
        return DatasetSearchResponse(
            datasets=all_datasets,
            total_count=len(all_datasets),
            query=request.query,
            source=request.source
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search datasets: {str(e)}")


@router.get("/popular")
async def get_popular_datasets(source: str = "all", limit: int = 20):
    """Get popular/recommended datasets."""
    try:
        popular_datasets = []
        
        if source in ["all", "openml"]:
            try:
                openml_popular = openml_service.get_popular_datasets(limit=limit)
                popular_datasets.extend(openml_popular.get("datasets", []))
            except Exception as e:
                # Log error but continue
                pass
        
        # Limit results
        popular_datasets = popular_datasets[:limit]
        
        return {
            "datasets": popular_datasets,
            "total_count": len(popular_datasets),
            "source": source
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular datasets: {str(e)}")


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
