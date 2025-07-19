"""
Dataset search and import endpoints.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import logging
from services.kaggle import kaggle_service
from services.open_ml import openml_service

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
