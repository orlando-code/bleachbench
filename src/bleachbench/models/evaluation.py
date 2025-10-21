"""
Model evaluation and comparison utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
import torch
from torch.utils.data import DataLoader


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.
    """
    
    def __init__(self, models: Dict[str, Any], model_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            models: Dictionary of trained models
            model_names: Names for the models (optional)
        """
        self.models = models
        self.model_names = model_names or list(models.keys())
        self.results = {}
    
    def evaluate_baseline_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate a baseline (sklearn) model."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "accuracy": classification_report(y_test, y_pred, output_dict=True)["accuracy"]
        }
        
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            metrics["average_precision"] = average_precision_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            metrics["pr_curve"] = {"precision": precision, "recall": recall}
        
        return metrics
    
    def evaluate_lstm_model(
        self, 
        trainer, 
        test_loader: DataLoader,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate an LSTM model."""
        trainer.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(trainer.device), target.to(trainer.device)
                output = trainer.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
                all_targets.extend(target.cpu().numpy())
        
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        y_test = np.array(all_targets)
        
        metrics = {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "accuracy": classification_report(y_test, y_pred, output_dict=True)["accuracy"],
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics["roc_curve"] = {"fpr": fpr, "tpr": tpr}
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        metrics["pr_curve"] = {"precision": precision, "recall": recall}
        
        return metrics
    
    def plot_roc_curves(self, save_path: Path = None):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            if "roc_curve" in results:
                fpr = results["roc_curve"]["fpr"]
                tpr = results["roc_curve"]["tpr"]
                auc = results.get("roc_auc", 0)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, save_path: Path = None):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            if "pr_curve" in results:
                precision = results["pr_curve"]["precision"]
                recall = results["pr_curve"]["recall"]
                ap = results.get("average_precision", 0)
                plt.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, save_path: Path = None):
        """Plot confusion matrices for all models."""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = results["confusion_matrix"]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of all model results."""
        summary_data = []
        
        for model_name, results in self.results.items():
            row = {
                "Model": model_name,
                "Accuracy": results["accuracy"],
                "ROC AUC": results.get("roc_auc", "N/A"),
                "Average Precision": results.get("average_precision", "N/A")
            }
            
            # Add precision, recall, F1 for each class
            if "classification_report" in results:
                report = results["classification_report"]
                for class_label in ["0", "1"]:
                    if class_label in report:
                        row[f"Precision_{class_label}"] = report[class_label]["precision"]
                        row[f"Recall_{class_label}"] = report[class_label]["recall"]
                        row[f"F1_{class_label}"] = report[class_label]["f1-score"]
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, save_path: Path):
        """Save all results to a JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, dict) and "fpr" in value:  # ROC curve
                    serializable_results[model_name][key] = {
                        "fpr": value["fpr"].tolist(),
                        "tpr": value["tpr"].tolist()
                    }
                elif isinstance(value, dict) and "precision" in value:  # PR curve
                    serializable_results[model_name][key] = {
                        "precision": value["precision"].tolist(),
                        "recall": value["recall"].tolist()
                    }
                else:
                    serializable_results[model_name][key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)


def run_comprehensive_evaluation(
    dataloader,
    test_size: float = 0.2,
    debug_samples: int = 200,
    random_state: int = 42,
    save_dir: Path = None,
    use_cache: bool = True
) -> ModelEvaluator:
    """
    Run comprehensive evaluation of baseline and LSTM models.
    
    Args:
        dataloader: SSTTimeSeriesLoader instance
        test_size: Fraction of data to use for testing
        debug_samples: Maximum number of samples to use
        random_state: Random seed
        save_dir: Directory to save results and plots
        use_cache: Whether to use feature caching for faster training
        
    Returns:
        ModelEvaluator with all results
    """
    from sklearn.model_selection import train_test_split
    from bleachbench.models.classic import train_baseline_models, BaselineClassifier
    from bleachbench.models.pytorch import train_lstm_model, SSTDataset
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    if debug_samples:
        print(f"Using {debug_samples} samples for evaluation")
    
    # Train baseline models
    print("\n1. Training Baseline Models...")
    baseline_models, baseline_results = train_baseline_models(
        dataloader=dataloader,
        debug_samples=debug_samples,
        test_size=test_size,
        random_state=random_state,
        use_cache=use_cache,
        cache_dir=save_dir / "feature_cache" if save_dir else None
    )
    
    # Train LSTM model
    print("\n2. Training LSTM Model...")
    lstm_model, lstm_trainer, lstm_results = train_lstm_model(
        dataloader=dataloader,
        debug_samples=debug_samples,
        test_size=test_size,
        random_state=random_state,
        use_cache=use_cache,
        cache_dir=save_dir / "sst_cache" if save_dir else None
    )
    
    # Create evaluator
    all_models = {**baseline_models, "LSTM": lstm_trainer}
    evaluator = ModelEvaluator(all_models)
    
    # Evaluate baseline models
    print("\n3. Evaluating Models...")
    for model_name, model in baseline_models.items():
        # Extract test data for baseline models
        features_list = []
        labels_list = []
        
        # Create test dataset
        dataset = SSTDataset(dataloader)
        test_size_actual = int(len(dataset) * test_size)
        test_indices = list(range(len(dataset) - test_size_actual, len(dataset)))
        
        for idx in test_indices:
            try:
                item = dataloader[dataset.valid_indices[idx]]
                if item is not None and "bleach_presence" in item:
                    sst_series = item["sst"]
                    if hasattr(sst_series, 'cpu'):
                        sst_series = sst_series.cpu().numpy()
                    
                    features = BaselineClassifier().extract_features(sst_series)
                    features_list.append(features)
                    labels_list.append(item["bleach_presence"])
            except Exception:
                continue
        
        if features_list:
            X_test = np.array(features_list)
            y_test = np.array(labels_list)
            
            results = evaluator.evaluate_baseline_model(model, X_test, y_test, model_name)
            evaluator.results[model_name] = results
    
    # Evaluate LSTM model
    dataset = SSTDataset(dataloader)
    test_size_actual = int(len(dataset) * test_size)
    test_indices = list(range(len(dataset) - test_size_actual, len(dataset)))
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    lstm_results = evaluator.evaluate_lstm_model(lstm_trainer, test_loader, "LSTM")
    evaluator.results["LSTM"] = lstm_results
    
    # Print summary
    print("\n4. Results Summary:")
    summary_df = evaluator.create_summary_table()
    print(summary_df.to_string(index=False))
    
    # Save results
    if save_dir:
        print(f"\n5. Saving results to {save_dir}")
        evaluator.save_results(save_dir / "evaluation_results.json")
        summary_df.to_csv(save_dir / "summary_table.csv", index=False)
        
        # Save plots
        evaluator.plot_roc_curves(save_dir / "roc_curves.png")
        evaluator.plot_precision_recall_curves(save_dir / "pr_curves.png")
        evaluator.plot_confusion_matrices(save_dir / "confusion_matrices.png")
    
    return evaluator
