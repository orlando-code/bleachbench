"""
Cached dataset for LSTM training to avoid repeated SST loading.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
from pathlib import Path
import pickle
import json
import hashlib
from tqdm.auto import tqdm

from bleachbench.processing.loader import SSTTimeSeriesLoader


class CachedSSTDataset(Dataset):
    """
    Cached version of SSTDataset that pre-loads and stores SST time series.
    """
    
    def __init__(
        self, 
        dataloader: SSTTimeSeriesLoader,
        cache_dir: Path,
        debug_samples: int = 200,
        force_recompute: bool = False
    ):
        """
        Initialize cached dataset.
        
        Args:
            dataloader: SSTTimeSeriesLoader instance
            cache_dir: Directory to store cached data
            debug_samples: Maximum number of samples to cache
            force_recompute: Whether to recompute existing cache
        """
        self.dataloader = dataloader
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.debug_samples = debug_samples
        
        # Create cache key
        self.cache_key = self._generate_cache_key()
        self.cache_file = self.cache_dir / f"sst_data_{self.cache_key}.pkl"
        self.metadata_file = self.cache_dir / f"sst_metadata_{self.cache_key}.json"
        
        self.cached_data = {}
        self.valid_indices = []
        
        # Load or compute cache
        if not force_recompute and self._load_cache():
            print(f"Loaded {len(self.cached_data)} cached SST time series")
        else:
            print("Computing and caching SST time series...")
            self._compute_cache()
    
    def _generate_cache_key(self) -> str:
        """Generate cache key based on dataloader configuration."""
        config_dict = {
            "base_dir": str(self.dataloader.base_dir),
            "var": self.dataloader.var,
            "window_days": getattr(self.dataloader, 'window_days', getattr(self.dataloader, 'window_days_before', 365)),
            "interpolate_method": self.dataloader.interpolate_method,
            "to_tensor": self.dataloader.to_tensor,
            "debug_samples": self.debug_samples,
            "df_hash": self._hash_dataframe(self.dataloader.df)
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Generate hash of dataframe."""
        key_columns = ["latitude", "longitude", "date", "bleach_presence"]
        df_subset = df[key_columns].copy()
        df_str = df_subset.to_string()
        return hashlib.md5(df_str.encode()).hexdigest()[:16]
    
    def _load_cache(self) -> bool:
        """Load cached data if available."""
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.valid_indices = metadata.get("valid_indices", [])
            
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self) -> None:
        """Save cached data."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cached_data, f)
            
            metadata = {"valid_indices": self.valid_indices}
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            print(f"Saved {len(self.cached_data)} SST time series to cache")
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _compute_cache(self) -> None:
        """Compute and cache SST time series data."""
        # Get valid indices
        all_indices = self.dataloader.get_valid_indices(max_samples=self.debug_samples)
        
        print(f"Computing cache for {len(all_indices)} indices...")
        
        successful_loads = 0
        for idx in tqdm(all_indices, desc="Loading SST data"):
            try:
                item = self.dataloader[idx]
                if item is None:
                    continue
                
                # Store the data
                self.cached_data[idx] = {
                    "sst": item["sst"],
                    "bleach_presence": item.get("bleach_presence"),
                    "mean_percent_bleached": item.get("mean_percent_bleached"),
                    "lat": item["lat"],
                    "lon": item["lon"],
                    "date": item["date"].isoformat() if hasattr(item["date"], 'isoformat') else str(item["date"])
                }
                
                self.valid_indices.append(idx)
                successful_loads += 1
                
            except Exception as e:
                print(f"Error loading index {idx}: {e}")
                continue
        
        print(f"Successfully cached {successful_loads} SST time series")
        self._save_cache()
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get cached item."""
        actual_idx = self.valid_indices[idx]
        item = self.cached_data[actual_idx]
        
        # Extract SST time series
        sst = item["sst"]
        if hasattr(sst, 'cpu'):  # Already a tensor
            sst_tensor = sst
        else:
            sst_tensor = torch.tensor(sst, dtype=torch.float32)
        
        # Ensure 2D shape (seq_len, features=1) - DataLoader will add batch dimension
        if sst_tensor.dim() == 1:
            sst_tensor = sst_tensor.unsqueeze(-1)  # (seq_len, 1)
        elif sst_tensor.dim() == 2 and sst_tensor.shape[0] == 1:
            # If it's (1, seq_len), transpose to (seq_len, 1)
            sst_tensor = sst_tensor.squeeze(0).unsqueeze(-1)  # (seq_len, 1)
        elif sst_tensor.dim() == 3:
            # If it's (1, seq_len, 1), squeeze out the batch dimension
            sst_tensor = sst_tensor.squeeze(0)  # (seq_len, 1)
        
        # Extract label
        label = torch.tensor(item["bleach_presence"], dtype=torch.long)
        
        return sst_tensor, label
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cached_data = {}
        self.valid_indices = []
        if self.cache_file.exists():
            self.cache_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        print("Cache cleared")


def create_cached_dataset(
    dataloader: SSTTimeSeriesLoader,
    cache_dir: Path = None,
    debug_samples: int = 200,
    force_recompute: bool = False
) -> CachedSSTDataset:
    """
    Create a cached SST dataset.
    
    Args:
        dataloader: SSTTimeSeriesLoader instance
        cache_dir: Directory for cache
        debug_samples: Maximum samples to cache
        force_recompute: Whether to recompute cache
        
    Returns:
        CachedSSTDataset instance
    """
    if cache_dir is None:
        cache_dir = Path("sst_cache")
    
    return CachedSSTDataset(
        dataloader=dataloader,
        cache_dir=cache_dir,
        debug_samples=debug_samples,
        force_recompute=force_recompute
    )
