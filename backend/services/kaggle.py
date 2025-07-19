"""
Kaggle dataset import service.
"""
import os
import zipfile
import tempfile
import shutil
import pandas as pd
from typing import Optional, Dict, Any
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class KaggleService:
    """Service for importing datasets from Kaggle."""
    
    def __init__(self):
        self.temp_dir = None
        self._check_kaggle_credentials()
    
    def _check_kaggle_credentials(self):
        """Check if Kaggle credentials are configured."""
        kaggle_config_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_config_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            logger.warning("Kaggle credentials not found. Please configure ~/.kaggle/kaggle.json")
            return False
        
        # Check environment variables as alternative
        if not (os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')):
            logger.info("Using Kaggle credentials from ~/.kaggle/kaggle.json")
        
        return True
    
    def download_dataset(self, dataset_name: str, file_name: Optional[str] = None) -> pd.DataFrame:
        """
        Download dataset from Kaggle and return as DataFrame.
        
        Args:
            dataset_name: Kaggle dataset name (e.g., 'titanic/titanic')
            file_name: Specific file to extract (if multiple files in dataset)
        
        Returns:
            pandas DataFrame with the dataset
        """
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Download dataset using Kaggle CLI
            cmd = ['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', self.temp_dir]
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Kaggle download completed: {result.stdout}")
            
            # Find downloaded file
            downloaded_files = list(Path(self.temp_dir).glob('*'))
            if not downloaded_files:
                raise ValueError("No files downloaded from Kaggle")
            
            # Extract if zip file
            zip_file = None
            for file in downloaded_files:
                if file.suffix == '.zip':
                    zip_file = file
                    break
            
            if zip_file:
                logger.info(f"Extracting zip file: {zip_file}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_dir)
                
                # Remove the zip file
                zip_file.unlink()
                
                # Update file list after extraction
                downloaded_files = [f for f in Path(self.temp_dir).glob('*') if f.is_file()]
            
            # Find CSV file to load
            csv_file = None
            
            if file_name:
                # Look for specific file
                csv_file = Path(self.temp_dir) / file_name
                if not csv_file.exists():
                    raise ValueError(f"Specified file '{file_name}' not found in dataset")
            else:
                # Find first CSV file
                for file in downloaded_files:
                    if file.suffix.lower() == '.csv':
                        csv_file = file
                        break
                
                if not csv_file:
                    raise ValueError("No CSV file found in downloaded dataset")
            
            logger.info(f"Loading CSV file: {csv_file}")
            
            # Load CSV into DataFrame
            df = pd.read_csv(csv_file)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            
            return df
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle CLI error: {e.stderr}")
            raise ValueError(f"Failed to download dataset from Kaggle: {e.stderr}")
        
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {str(e)}")
            raise
        
        finally:
            # Clean up temporary directory
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def search_datasets(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for datasets on Kaggle.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            Dictionary with search results
        """
        try:
            cmd = ['kaggle', 'datasets', 'list', '-s', query, '--max-size', str(max_results)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the output (simplified - Kaggle CLI returns tabular data)
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return {"datasets": [], "total": 0}
            
            datasets = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        datasets.append({
                            "name": parts[0],
                            "title": " ".join(parts[1:]),
                            "url": f"https://www.kaggle.com/datasets/{parts[0]}"
                        })
            
            return {
                "datasets": datasets,
                "total": len(datasets),
                "query": query
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle search error: {e.stderr}")
            raise ValueError(f"Failed to search Kaggle datasets: {e.stderr}")
        
        except Exception as e:
            logger.error(f"Error searching Kaggle datasets: {str(e)}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a Kaggle dataset.
        
        Args:
            dataset_name: Kaggle dataset name
        
        Returns:
            Dictionary with dataset information
        """
        try:
            cmd = ['kaggle', 'datasets', 'list', '-s', dataset_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse basic info from the output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            # This is a simplified parser - in production you might want to use Kaggle API
            return {
                "name": dataset_name,
                "url": f"https://www.kaggle.com/datasets/{dataset_name}",
                "source": "kaggle"
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Kaggle info error: {e.stderr}")
            raise ValueError(f"Failed to get dataset info: {e.stderr}")
        
        except Exception as e:
            logger.error(f"Error getting dataset info: {str(e)}")
            raise


# Global instance
kaggle_service = KaggleService()
