"""
OpenML dataset import service.
"""
import openml
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class OpenMLService:
    """Service for importing datasets from OpenML."""
    
    def __init__(self):
        # Configure OpenML (optional: set cache directory)
        openml.config.cache_directory = './openml_cache'
    
    def get_dataset(self, dataset_id: int) -> pd.DataFrame:
        """
        Download dataset from OpenML by ID and return as DataFrame.
        
        Args:
            dataset_id: OpenML dataset ID
        
        Returns:
            pandas DataFrame with the dataset
        """
        try:
            logger.info(f"Downloading OpenML dataset ID: {dataset_id}")
            
            # Download dataset
            dataset = openml.datasets.get_dataset(dataset_id)
            
            # Get the data
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                dataset_format="dataframe",
                target=dataset.default_target_attribute
            )
            
            # Combine features and target if target exists
            if y is not None:
                if isinstance(y, pd.Series):
                    df = X.copy()
                    df[dataset.default_target_attribute] = y
                else:
                    df = pd.concat([X, y], axis=1)
            else:
                df = X
            
            logger.info(f"Successfully loaded OpenML dataset with shape: {df.shape}")
            logger.info(f"Dataset name: {dataset.name}")
            logger.info(f"Dataset description: {dataset.description[:100]}...")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading OpenML dataset {dataset_id}: {str(e)}")
            raise ValueError(f"Failed to download OpenML dataset {dataset_id}: {str(e)}")
    
    def search_datasets(self, query: str = None, max_results: int = 10, **filters) -> Dict[str, Any]:
        """
        Search for datasets on OpenML.
        
        Args:
            query: Search query (searches in name and description)
            max_results: Maximum number of results to return
            **filters: Additional filters (e.g., number_instances, number_features)
        
        Returns:
            Dictionary with search results
        """
        try:
            # Build search criteria
            search_kwargs = {
                'size': max_results,
                'output_format': 'dataframe'
            }
            
            # Add filters
            if filters:
                search_kwargs.update(filters)
            
            logger.info(f"Searching OpenML datasets with query: '{query}', filters: {filters}")
            
            # Search datasets
            datasets_df = openml.datasets.list_datasets(**search_kwargs)
            
            if datasets_df is None or datasets_df.empty:
                return {"datasets": [], "total": 0, "query": query}
            
            # Filter by query if provided
            if query:
                mask = (
                    datasets_df['name'].str.contains(query, case=False, na=False) |
                    datasets_df['description'].str.contains(query, case=False, na=False)
                )
                datasets_df = datasets_df[mask]
            
            # Convert to list of dictionaries
            datasets = []
            for _, row in datasets_df.head(max_results).iterrows():
                dataset_info = {
                    "id": int(row.name),  # Index is the dataset ID
                    "name": row.get('name', 'Unknown'),
                    "description": row.get('description', '')[:200] + '...' if len(str(row.get('description', ''))) > 200 else row.get('description', ''),
                    "instances": int(row.get('NumberOfInstances', 0)),
                    "features": int(row.get('NumberOfFeatures', 0)),
                    "classes": int(row.get('NumberOfClasses', 0)) if pd.notna(row.get('NumberOfClasses')) else None,
                    "missing_values": bool(row.get('NumberOfMissingValues', 0) > 0),
                    "url": f"https://www.openml.org/d/{int(row.name)}",
                    "source": "openml"
                }
                datasets.append(dataset_info)
            
            return {
                "datasets": datasets,
                "total": len(datasets),
                "query": query,
                "filters": filters
            }
            
        except Exception as e:
            logger.error(f"Error searching OpenML datasets: {str(e)}")
            raise ValueError(f"Failed to search OpenML datasets: {str(e)}")
    
    def get_dataset_info(self, dataset_id: int) -> Dict[str, Any]:
        """
        Get detailed information about an OpenML dataset.
        
        Args:
            dataset_id: OpenML dataset ID
        
        Returns:
            Dictionary with dataset information
        """
        try:
            logger.info(f"Getting info for OpenML dataset ID: {dataset_id}")
            
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False)
            
            info = {
                "id": dataset_id,
                "name": dataset.name,
                "description": dataset.description,
                "version": dataset.version,
                "format": dataset.format,
                "upload_date": str(dataset.upload_date) if hasattr(dataset, 'upload_date') and dataset.upload_date else None,
                "licence": getattr(dataset, 'licence', None),
                "url": getattr(dataset, 'url', f"https://www.openml.org/d/{dataset_id}"),
                "default_target_attribute": dataset.default_target_attribute,
                "row_id_attribute": getattr(dataset, 'row_id_attribute', None),
                "ignore_attributes": getattr(dataset, 'ignore_attribute', None),
                "version_label": getattr(dataset, 'version_label', None),
                "citation": getattr(dataset, 'citation', None),
                "creator": getattr(dataset, 'creator', None),
                "contributor": getattr(dataset, 'contributor', None),
                "collection_date": getattr(dataset, 'collection_date', None),
                "language": getattr(dataset, 'language', None),
                "source": "openml"
            }
            
            # Add qualities if available
            if hasattr(dataset, 'qualities') and dataset.qualities:
                qualities = {}
                for key, value in dataset.qualities.items():
                    if value is not None:
                        qualities[key] = value
                info["qualities"] = qualities
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting OpenML dataset info {dataset_id}: {str(e)}")
            raise ValueError(f"Failed to get OpenML dataset info {dataset_id}: {str(e)}")
    
    def get_popular_datasets(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get popular/recommended datasets from OpenML.
        
        Args:
            limit: Number of datasets to return
        
        Returns:
            Dictionary with popular datasets
        """
        try:
            # Get datasets sorted by number of downloads or likes
            datasets_df = openml.datasets.list_datasets(
                size=limit * 2,  # Get more to filter out problematic ones
                output_format='dataframe'
            )
            
            if datasets_df is None or datasets_df.empty:
                return {"datasets": [], "total": 0}
            
            # Sort by number of downloads if available
            if 'NumberOfDownloads' in datasets_df.columns:
                datasets_df = datasets_df.sort_values('NumberOfDownloads', ascending=False)
            
            # Filter out very large datasets (> 100MB or > 100k instances)
            if 'NumberOfInstances' in datasets_df.columns:
                datasets_df = datasets_df[datasets_df['NumberOfInstances'] <= 100000]
            
            datasets = []
            for _, row in datasets_df.head(limit).iterrows():
                dataset_info = {
                    "id": int(row.name),
                    "name": row.get('name', 'Unknown'),
                    "description": row.get('description', '')[:200] + '...' if len(str(row.get('description', ''))) > 200 else row.get('description', ''),
                    "instances": int(row.get('NumberOfInstances', 0)),
                    "features": int(row.get('NumberOfFeatures', 0)),
                    "downloads": int(row.get('NumberOfDownloads', 0)),
                    "url": f"https://www.openml.org/d/{int(row.name)}",
                    "source": "openml"
                }
                datasets.append(dataset_info)
            
            return {
                "datasets": datasets,
                "total": len(datasets)
            }
            
        except Exception as e:
            logger.error(f"Error getting popular OpenML datasets: {str(e)}")
            raise ValueError(f"Failed to get popular OpenML datasets: {str(e)}")


# Global instance
openml_service = OpenMLService()
