"""
PyTorch models for coral bleaching prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from tqdm.auto import tqdm


class SSTLSTMClassifier(nn.Module):
    """
    LSTM-based classifier for SST time series.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_classes: int = 2
    ):
        super(SSTLSTMClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Define LSTM layers with the above parameters
        self.lstm = nn.LSTM(
            input_size=input_size,          # number of input features per time step (1, daily SST value)
            hidden_size=hidden_size,        # hidden state dimensionality
            num_layers=num_layers,          # number of stacked layers
            dropout=dropout if num_layers > 1 else 0,  # apply dropout if stacking
            bidirectional=bidirectional,    # use bidirectional LSTM if True (can process both forward and backward information)
            batch_first=True                # input/output tensors have shape (batch, seq, feature)
        )

        # The output of the LSTM at each time step:
        # shape = (batch, seq_len, hidden_size * num_directions) (number of directions is 2 if bidirectional, 1 otherwise)
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from the sequence
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * num_directions)
        
        # Classification
        output = self.classifier(last_output)
        
        return output


class SSTDataset(Dataset):
    """
    PyTorch Dataset wrapper for SST time series data.
    """
    
    def __init__(self, dataloader, debug_samples: int = 200, transform=None):
        """
        Initialize dataset.
        
        Args:
            dataloader: SSTTimeSeriesLoader instance
            transform: Optional transform to apply to data
        """
        self.dataloader = dataloader
        self.transform = transform
        self.valid_indices = []
        self.debug_samples = debug_samples
        
        # Pre-compute valid indices without loading SST data
        self._find_valid_indices_fast()
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.dataloader[actual_idx]
        
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
        
        if self.transform:
            sst_tensor = self.transform(sst_tensor)
        
        return sst_tensor, label
    
    def _find_valid_indices_fast(self):
        """
        Fast method to find valid indices without loading SST data.
        Only checks dataframe validity, not SST data availability.
        """
        self.valid_indices = self.dataloader.get_valid_indices(max_samples=self.debug_samples)
        print(f"Found {len(self.valid_indices)} valid indices")


class LSTMTrainer:
    """
    Trainer class for LSTM models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        save_path: Path = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs to train
            patience (int): Early stopping patience
            save_path (Path): Path to save best model
            
        Returns:
            Training history (Dict[str, List[float]])
                - train_losses (List[float]): Training losses
                - val_losses (List[float]): Validation losses
                - train_accuracies (List[float]): Training accuracies
                - val_accuracies (List[float]): Validation accuracies
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs)):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }
    
    def save_model(self, path: Path) -> None:
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'bidirectional': self.model.bidirectional
            }
        }, path)
    
    def load_model(self, path: Path) -> None:
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_lstm_model(
    dataloader,
    debug_samples: int = 200,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 32,
    epochs: int = 50,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    save_path: Path = None,
    use_cache: bool = True,
    cache_dir: Path = None
) -> Tuple[SSTLSTMClassifier, LSTMTrainer, Dict[str, Any]]:
    """
    Train an LSTM model for bleaching classification.
    
    Args:
        dataloader: SSTTimeSeriesLoader instance
        debug_samples: Maximum number of samples to use
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed
        batch_size: Batch size for training
        epochs: Number of epochs to train
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        learning_rate: Learning rate
        save_path: Path to save the trained model
        use_cache: Whether to use cached dataset
        cache_dir: Directory for cache
        
    Returns:
        Tuple of (model, trainer, results)
    """
    from sklearn.model_selection import train_test_split
    
    # Create dataset (cached or regular)
    if use_cache:
        from bleachbench.models.cached_dataset import create_cached_dataset
        
        if cache_dir is None:
            cache_dir = Path("sst_cache")
        
        print("Using cached SST dataset for fast training...")
        dataset = create_cached_dataset(
            dataloader=dataloader,
            cache_dir=cache_dir,
            debug_samples=debug_samples,
            force_recompute=False
        )
    else:
        print("Using regular SST dataset...")
        dataset = SSTDataset(dataloader, debug_samples=debug_samples)
    
    if len(dataset) == 0:
        raise ValueError("No valid data found in dataloader")
    
    # remove any entries from dataset with nans in the sst tensor
    og_length = len(dataset)
    dataset = [item for item in dataset if not np.isnan(item[0]).all()] # TODO: this shouldn't be necessary but some are all nans
    print(f"Removed {og_length - len(dataset)} samples with nans in the SST tensor")
    
    print(f"Created dataset with {len(dataset)} valid samples")
    
    # Split indices
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(debug_samples, len(dataset)), replace=False)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=[dataset[i][1].item() for i in indices]
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size, random_state=random_state, stratify=[dataset[i][1].item() for i in train_indices]
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model and trainer
    model = SSTLSTMClassifier(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    trainer = LSTMTrainer(model, learning_rate=learning_rate)
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=save_path
    )
    
    # Evaluate on test set
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    results = {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "history": history,
        "model_config": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
    }
    
    return model, trainer, results
