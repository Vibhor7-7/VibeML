"""
Dataset search and import endpoints.
"""
import glob
import pandas as pd
import os
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import logging
from services.kaggle import kaggle_service
from services.open_ml import openml_service

# Path to local test datasets
LOCAL_DATASET_DIR = os.path.join(os.path.dirname(__file__), '../test_datasets')

def list_local_datasets():
    """List all CSV files in the test_datasets directory as available datasets."""
    files = glob.glob(os.path.join(LOCAL_DATASET_DIR, '*.csv'))
    datasets = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        datasets.append({
            'title': name.replace('_', ' ').title(),
            'name': name,
            'source': 'local',
            'path': f,
            'description': f"Local test dataset for development - {name}"
        })
    return datasets
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("/search")
async def search_datasets(
    query: str = Query(..., description="Search query for datasets"),
    source: Optional[str] = Query("all", description="Dataset source: 'kaggle', 'openml', or 'all'"),
    max_results: int = Query(10, description="Maximum number of results per source", ge=1, le=50)
) -> Dict[str, Any]:
    """
    Search for datasets across multiple sources.
    
    Args:
        query: Search term
        source: Which source to search ('kaggle', 'openml', 'all')
        max_results: Maximum results per source
    
    Returns:
        Combined search results from specified sources
    """
    try:
        results = {
            "query": query,
            "datasets": [],
            "sources": []
        }
        
        # Local test datasets - always include these in dev mode
        if source in ["all", "local"]:
            logger.info(f"Searching local datasets for '{query}'")
            local_datasets = list_local_datasets()
            # Filter by query or include all if query is 'test'
            if query.lower() == 'test' or not query.strip():
                filtered = local_datasets  # Show all test datasets
            else:
                filtered = [d for d in local_datasets if query.lower() in d['title'].lower()]
            
            if filtered:
                logger.info(f"Found {len(filtered)} local datasets")
                results["datasets"].extend(filtered)
                results["sources"].append("local")
        
        # Search Kaggle
        if source in ["all", "kaggle"]:
            try:
                kaggle_results = kaggle_service.search_datasets(query, max_results)
                if kaggle_results.get("datasets"):
                    results["datasets"].extend(kaggle_results["datasets"])
                    results["sources"].append("kaggle")
                logger.info(f"Found {len(kaggle_results.get('datasets', []))} Kaggle datasets")
            except Exception as e:
                logger.warning(f"Kaggle search failed: {str(e)}")
                # Don't fail the entire request if one source fails
        
        # Search OpenML
        if source in ["all", "openml"]:
            try:
                openml_results = openml_service.search_datasets(query, max_results)
                if openml_results.get("datasets"):
                    results["datasets"].extend(openml_results["datasets"])
                    results["sources"].append("openml")
                logger.info(f"Found {len(openml_results.get('datasets', []))} OpenML datasets")
            except Exception as e:
                logger.warning(f"OpenML search failed: {str(e)}")
                # Don't fail the entire request if one source fails
        
        results["total"] = len(results["datasets"])
        
        # Sort by relevance (for now, just by title similarity)
        results["datasets"].sort(key=lambda x: query.lower() in x.get("title", "").lower(), reverse=True)
        
        return results
        
    except Exception as e:
        logger.error(f"Dataset search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/info/{source}/{dataset_name:path}")
async def get_dataset_info(
    source: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Get detailed information about a specific dataset.
    
    Args:
        source: Dataset source ('kaggle' or 'openml')
        dataset_name: Dataset identifier
    
    Returns:
        Detailed dataset information
    """
    try:
        if source == "kaggle":
            return kaggle_service.get_dataset_info(dataset_name)
        elif source == "openml":
            return openml_service.get_dataset_info(dataset_name)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
            
    except Exception as e:
        logger.error(f"Failed to get dataset info: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Dataset not found: {str(e)}")


@router.get("/preview/{source}/{dataset_name:path}")
async def get_dataset_preview(
    source: str,
    dataset_name: str,
    max_rows: int = Query(10, description="Maximum number of preview rows", ge=1, le=50)
) -> Dict[str, Any]:
    """
    Get dataset preview with actual data and column information.
    
    Args:
        source: Dataset source ('kaggle' or 'openml')
        dataset_name: Dataset identifier
        max_rows: Maximum number of rows to return in preview
    
    Returns:
        Dataset preview with columns and sample data
    """
    try:
        if source == "local":
            csv_path = os.path.join(LOCAL_DATASET_DIR, f"{dataset_name}.csv")
            if not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail=f"Local dataset not found: {dataset_name}")
            df = pd.read_csv(csv_path)
        elif source == "kaggle":
            # For Kaggle, we need to actually download and inspect the dataset
            df = kaggle_service.download_dataset(dataset_name)
        elif source == "openml":
            # For OpenML, extract dataset ID from name and download
            dataset_id = int(dataset_name.split('_')[-1]) if dataset_name.startswith('openml_') else int(dataset_name)
            df = openml_service.get_dataset(dataset_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
        
        # Get basic dataset info
        columns = df.columns.tolist()
        data_types = df.dtypes.astype(str).to_dict()
        
        # Get sample data (limit rows and convert to records)
        sample_df = df.head(max_rows)
        sample_data = sample_df.to_dict('records')
        
        # Get basic statistics
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Detect potential target columns (last column, or columns with specific names)
        potential_targets = []
        target_indicators = ['target', 'label', 'class', 'y', 'outcome', 'response', 'survived', 'species', 'quality']
        
        for col in columns:
            if any(indicator in col.lower() for indicator in target_indicators):
                potential_targets.append(col)
        
        # If no obvious target found, suggest the last column
        if not potential_targets and columns:
            potential_targets.append(columns[-1])
        
        return {
            "dataset_name": dataset_name,
            "source": source,
            "columns": columns,
            "data_types": data_types,
            "sample_data": sample_data,
            "statistics": stats,
            "potential_target_columns": potential_targets,
            "preview_rows": len(sample_data)
        }
        
    except Exception as e:
        logger.error(f"Failed to get dataset preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.get("/download/{source}/{dataset_name:path}")
async def download_dataset(
    source: str,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Download and prepare dataset for training.
    
    Args:
        source: Dataset source ('kaggle' or 'openml')
        dataset_name: Dataset identifier
    
    Returns:
        Dataset information and temporary storage location
    """
    try:
        if source == "local":
            csv_path = os.path.join(LOCAL_DATASET_DIR, f"{dataset_name}.csv")
            if not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail=f"Local dataset not found: {dataset_name}")
            df = pd.read_csv(csv_path)
        elif source == "kaggle":
            df = kaggle_service.download_dataset(dataset_name)
        elif source == "openml":
            dataset_id = int(dataset_name.split('_')[-1]) if dataset_name.startswith('openml_') else int(dataset_name)
            df = openml_service.get_dataset(dataset_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
        
        # Store dataset temporarily for training
        import tempfile
        
        # Create a temporary CSV file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"dataset_{source}_{dataset_name.replace('/', '_')}.csv"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        df.to_csv(temp_path, index=False)
        
        return {
            "dataset_name": dataset_name,
            "source": source,
            "temp_path": temp_path,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "size_mb": os.path.getsize(temp_path) / 1024 / 1024
        }
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/download")
async def download_dataset(
    source: str,
    dataset_name: str,
    file_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download a dataset and return basic info.
    
    Args:
        source: Dataset source ('kaggle' or 'openml')
        dataset_name: Dataset identifier
        file_name: Specific file to download (optional)
    
    Returns:
        Dataset download information
    """
    try:
        if source == "kaggle":
            df = kaggle_service.download_dataset(dataset_name, file_name)
        elif source == "openml":
            df = openml_service.download_dataset(dataset_name)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
        
        # Return basic info about the downloaded dataset
        return {
            "success": True,
            "dataset_name": dataset_name,
            "source": source,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "head": df.head().to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
