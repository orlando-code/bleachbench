"""
Classic machine learning models for coral bleaching prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from pathlib import Path
from tqdm.auto import tqdm


class BaselineClassifier:
    """
    Baseline classifier using classic ML methods on SST time series features.
    """
    
    def __init__(self, model_type: str = "random_forest", n_jobs: int = 64, random_state: int = 42, **kwargs):
        """
        Initialize baseline classifier.
        
        Args:
            model_type: Type of model to use ("random_forest", "logistic", "svm")
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.feature_names = None
        self.is_fitted = False
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = self._create_model(**kwargs)
        
    def _create_model(self, **kwargs):
        """Create the specified model type."""
        if self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "n_jobs": self.n_jobs,
                "random_state": self.random_state
            }
            default_params.update(kwargs)
            return RandomForestClassifier(**default_params)
            
        elif self.model_type == "logistic":
            default_params = {
                "max_iter": 1000,
                "random_state": self.random_state
            }
            default_params.update(kwargs)
            return LogisticRegression(**default_params)
            
        elif self.model_type == "svm":
            default_params = {
                "probability": True,
                "random_state": self.random_state
            }
            default_params.update(kwargs)
            return SVC(**default_params)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def extract_features(self, sst_series: np.ndarray) -> np.ndarray:
        """
        Extract features from SST time series.
        
        Args:
            sst_series: SST time series of shape (time_steps,)
            
        Returns:
            Feature vector (np.ndarray) of shape (n_features,). Mean, std, min, max, median, 90th, 95th, 99th percentiles, linear trend, degree heating weeks, maximum consecutive days above threshold, recent vs early period comparison.
        """
        features = []
        
        # Basic statistics
        features.extend([
            np.nanmean(sst_series),
            np.nanstd(sst_series),
            np.nanmin(sst_series),
            np.nanmax(sst_series),
            np.nanmedian(sst_series)
        ])
        
        # Temperature extremes
        features.extend([
            np.nanpercentile(sst_series, 90),  # 90th percentile
            np.nanpercentile(sst_series, 95),  # 95th percentile
            np.nanpercentile(sst_series, 99),  # 99th percentile
        ])
        
        # # Trend features
        # if len(sst_series) > 1:
        #     # Linear trend
        #     x = np.arange(len(sst_series))
        #     valid_mask = ~np.isnan(sst_series)
        #     if np.sum(valid_mask) > 1:
        #         slope, _ = np.polyfit(x[valid_mask], sst_series[valid_mask], 1)
        #         features.append(slope)
        #     else:
        #         features.append(0.0)
        # else:
        #     features.append(0.0)
        
        # # Degree heating weeks (DHW) - cumulative heat stress
        # # DHW = sum of weekly SST anomalies above MMM
        # mmm = np.nanmean(sst_series)  # Use series mean as MMM proxy
        # weekly_anomalies = []
        # for i in range(0, len(sst_series), 7):
        #     week_data = sst_series[i:i+7]
        #     if len(week_data) > 0:
        #         week_mean = np.nanmean(week_data)
        #         if week_mean > mmm:
        #             weekly_anomalies.append(week_mean - mmm)
        
        # dhw = np.sum(weekly_anomalies) if weekly_anomalies else 0.0
        # features.append(dhw)
        
        # # Maximum consecutive days above threshold
        # threshold = mmm + 1.0  # 1°C above mean
        # above_threshold = sst_series > threshold
        # if np.any(above_threshold):
        #     # Find consecutive runs
        #     diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        #     run_starts = np.where(diff == 1)[0]
        #     run_ends = np.where(diff == -1)[0]
        #     max_consecutive = np.max(run_ends - run_starts) if len(run_starts) > 0 else 0
        # else:
        #     max_consecutive = 0
        # features.append(max_consecutive)
        
        # # Recent vs early period comparison
        # if len(sst_series) >= 30:
        #     early_period = sst_series[:len(sst_series)//3]
        #     recent_period = sst_series[-len(sst_series)//3:]
        #     features.append(np.nanmean(recent_period) - np.nanmean(early_period))
        # else:
        #     features.append(0.0)
        
        return np.array(features)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineClassifier':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if len(self.model.classes_) == 2 else None
        
        metrics = {
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        }
        
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_proba)
        
        return metrics
    
    def save(self, filepath: Path) -> None:
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Path) -> 'BaselineClassifier':
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.model_type = model_data["model_type"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = model_data["is_fitted"]
        return self


def train_baseline_models(
    dataloader,
    debug_samples: int = 200,
    test_size: float = 0.2,
    random_state: int = 42,
    model_types: List[str] = None,
    use_cache: bool = True,
    cache_dir: Path = None
) -> Dict[str, BaselineClassifier]:
    """
    Train multiple baseline models and return the best performing one.
    
    Args:
        dataloader: SSTTimeSeriesLoader instance
        debug_samples: Maximum number of samples to use
        test_size: Fraction of data to use for testing
        random_state: Random seed
        model_types: List of model types to try
        use_cache: Whether to use feature caching
        cache_dir: Directory for feature cache
        
    Returns:
        Dictionary of trained models
    """
    if model_types is None:
        model_types = ["random_forest", "logistic", "svm"]
    
    # Extract features and labels
    if use_cache:
        print("Using feature cache for fast training...")
        from bleachbench.models.feature_cache import precompute_features
        
        if cache_dir is None:
            cache_dir = Path("feature_cache")
        
        # Pre-compute and cache features
        cache = precompute_features(
            dataloader=dataloader,
            cache_dir=cache_dir,
            max_samples=debug_samples,
            force_recompute=False
        )
        
        # Get cached features
        samples = dataloader.get_valid_indices(max_samples=debug_samples)
        X, y = cache.get_features_matrix(samples)
        
        if len(X) == 0:
            raise ValueError("No cached features found")
            
    else:
        print("Extracting features from SST time series (no cache)...")
        features_list = []
        labels_list = []
        
        # Pre-filter valid indices from dataframe (fast check)
        samples = dataloader.get_valid_indices(max_samples=debug_samples)
        
        print(f"Processing {len(samples)} samples...")
        
        for i in tqdm(samples, desc="Extracting features"):
            try:
                item = dataloader[i]
                if item is None:  # Skip failed loads
                    continue
                    
                sst_series = item["sst"]
                if hasattr(sst_series, 'cpu'):  # PyTorch tensor
                    sst_series = sst_series.cpu().numpy()
                
                if "bleach_presence" in item and item["bleach_presence"] is not None and not np.isnan(sst_series).all():
                    features = BaselineClassifier(random_state=random_state).extract_features(sst_series)
                    features_list.append(features)
                    labels_list.append(item["bleach_presence"])
                    
            except Exception as e:
                print(f"Skipping item {i} due to error: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid data found in dataloader")
        
        X = np.array(features_list)
        y = np.array(labels_list)
    
    print(f"Extracted features for {len(X)} samples")
    print(f"Feature shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train models
    models = {}
    results = {}
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        model = BaselineClassifier(model_type=model_type)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        results[model_type] = metrics
        
        print(f"{model_type} - ROC AUC: {metrics.get('roc_auc', 'N/A')}")
        print(f"{model_type} - Accuracy: {metrics['classification_report']['accuracy']:.3f}")
        
        models[model_type] = model
    
    return models, results
