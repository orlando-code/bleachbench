"""
Feature caching system for SST time series data.
Pre-computes and caches features to avoid repeated SST loading.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm.auto import tqdm
import hashlib
import json

from bleachbench.processing.loader import SSTTimeSeriesLoader
from bleachbench.models.classic import BaselineClassifier


class FeatureCache:
    """
    Cache system for pre-computed SST features.
    """
    
    def __init__(self, cache_dir: Path, dataloader: SSTTimeSeriesLoader):
        """
        Initialize feature cache.
        
        Args:
            cache_dir: Directory to store cached features
            dataloader: SSTTimeSeriesLoader instance
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataloader = dataloader
        
        # Create cache key based on dataloader configuration
        self.cache_key = self._generate_cache_key()
        self.cache_file = self.cache_dir / f"features_{self.cache_key}.pkl"
        self.metadata_file = self.cache_dir / f"metadata_{self.cache_key}.json"
        
        self.features_cache = {}
        self.metadata = {}
        
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on dataloader configuration."""
        config_dict = {
            "base_dir": str(self.dataloader.base_dir),
            "var": self.dataloader.var,
            "window_days": getattr(self.dataloader, 'window_days', getattr(self.dataloader, 'window_days_before', 365)),
            "interpolate_method": self.dataloader.interpolate_method,
            "df_hash": self._hash_dataframe(self.dataloader.df)
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash of dataframe for cache invalidation."""
        # Use a subset of columns that matter for SST loading
        key_columns = ["latitude", "longitude", "date", "bleach_presence"]
        df_subset = df[key_columns].copy()
        df_str = df_subset.to_string()
        return hashlib.md5(df_str.encode()).hexdigest()[:16]
    
    def load_cache(self) -> bool:
        """
        Load cached features if available.
        
        Returns:
            True if cache was loaded successfully, False otherwise
        """
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                self.features_cache = pickle.load(f)
            
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            print(f"Loaded {len(self.features_cache)} cached features")
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def save_cache(self) -> None:
        """Save features to cache."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.features_cache, f)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
            
            print(f"Saved {len(self.features_cache)} features to cache")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def compute_features_batch(
        self, 
        indices: List[int], 
        force_recompute: bool = False,
        save_interval: int = 100
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute features for a batch of indices.
        
        Args:
            indices: List of dataframe indices to process
            force_recompute: Whether to recompute even if cached
            save_interval: Save cache every N items
            
        Returns:
            Dictionary mapping index to features
        """
        results = {}
        new_features = 0
        
        for i, idx in enumerate(tqdm(indices, desc="Computing features")):
            # Check if already cached
            if not force_recompute and idx in self.features_cache:
                results[idx] = self.features_cache[idx]
                continue
            
            try:
                # Load SST data
                item = self.dataloader[idx]
                if item is None:
                    continue
                
                # Extract features
                sst_series = item["sst"]
                if hasattr(sst_series, 'cpu'):
                    sst_series = sst_series.cpu().numpy()
                
                if np.isnan(sst_series).all():
                    continue
                
                # Compute features
                features = BaselineClassifier().extract_features(sst_series)
                
                # Store result
                result = {
                    "features": features,
                    "bleach_presence": item.get("bleach_presence"),
                    "mean_percent_bleached": item.get("mean_percent_bleached"),
                    "lat": item["lat"],
                    "lon": item["lon"],
                    "date": item["date"].isoformat() if hasattr(item["date"], 'isoformat') else str(item["date"])
                }
                
                results[idx] = result
                self.features_cache[idx] = result
                new_features += 1
                
                # Save periodically
                if new_features % save_interval == 0:
                    self.save_cache()
                    
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                continue
        
        # Final save
        if new_features > 0:
            self.save_cache()
        
        print(f"Computed {new_features} new features, {len(results)} total")
        return results
    
    def get_features_matrix(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get features matrix and labels for given indices.
        
        Args:
            indices: List of dataframe indices
            
        Returns:
            Tuple of (features_matrix, labels)
        """
        features_list = []
        labels_list = []
        
        for idx in indices:
            if idx in self.features_cache:
                result = self.features_cache[idx]
                features_list.append(result["features"])
                labels_list.append(result["bleach_presence"])
        
        if not features_list:
            return np.array([]), np.array([])
        
        return np.array(features_list), np.array(labels_list)
    
    def get_cached_indices(self) -> List[int]:
        """Get list of cached indices."""
        return list(self.features_cache.keys())
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.features_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("Cache cleared")


def precompute_features(
    dataloader: SSTTimeSeriesLoader,
    cache_dir: Path,
    max_samples: int = None,
    force_recompute: bool = False
) -> FeatureCache:
    """
    Pre-compute and cache features for all valid indices.
    
    Args:
        dataloader: SSTTimeSeriesLoader instance
        cache_dir: Directory to store cache
        max_samples: Maximum number of samples to process
        force_recompute: Whether to recompute existing features
        
    Returns:
        FeatureCache instance
    """
    cache = FeatureCache(cache_dir, dataloader)
    
    # Try to load existing cache
    if not force_recompute:
        cache.load_cache()
    
    # Get indices to process
    all_indices = dataloader.get_valid_indices(max_samples=max_samples)
    cached_indices = set(cache.get_cached_indices())
    indices_to_process = [idx for idx in all_indices if idx not in cached_indices]
    
    if not indices_to_process:
        print("All features already cached!")
        return cache
    
    print(f"Processing {len(indices_to_process)} new indices...")
    
    # Compute features in batches
    batch_size = 50
    for i in range(0, len(indices_to_process), batch_size):
        batch_indices = indices_to_process[i:i + batch_size]
        cache.compute_features_batch(batch_indices, force_recompute=force_recompute)
    
    return cache
