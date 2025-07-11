import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set manual seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoralHeatRNN(nn.Module):
    """
    Recurrent neural network (RNN) for binary classification tasks using PyTorch.

    Args:
        input_size (int): Dimensionality of input features.
        hidden_size (int): Number of hidden units in the RNN.
        layer_size (int): Number of RNN layers.
        output_size (int): Size of the output (usually 1 for binary classification).
        dropout (float): Dropout probability applied between RNN layers.
        init_func (callable, optional): Initialization function for the final linear layer weights.
    """

    def __init__(self, input_size, hidden_size, layer_size, output_size, dropout, init_func=None):
        super(CoralHeatRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_size,
            batch_first=True,
            dropout=dropout,
            nonlinearity="relu",
        )
        self.fc = nn.Linear(hidden_size, output_size)
        if init_func is not None:
            init_func(self.fc.weight)

        self.hidden_size = hidden_size
        self.layer_size = layer_size

    def forward(self, x):
        """
        Forward pass through the RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(device)
        rnn_out, _ = self.rnn(x, h0)
        out = self.fc(rnn_out[:, -1, :]).sigmoid()
        return out


def add_rolling_averages(ts, windows=[30, 90, 180]):
    """
    Compute rolling averages over a 1D time series for given window sizes.

    Args:
        ts (array-like): Input time series (1D array).
        windows (list of int): List of window sizes to compute averages over.

    Returns:
        np.ndarray: 2D array of shape (len(ts), 1 + len(windows)),
                    where the first column is the original series and
                    the others are the rolling averages.
    """
    ts = np.asarray(ts)
    n = len(ts)
    result = np.full((n, 1 + len(windows)), np.nan)
    result[:, 0] = ts

    for i, window in enumerate(windows):
        for j in range(window, n):
            result[j, i + 1] = np.mean(ts[j - window:j])

    return result


def train_model(model, train_x, train_y, lr, weight_decay, epochs, test_x=None, test_y=None):
    """
    Trains a PyTorch model using binary cross-entropy loss.

    Args:
        model (nn.Module): A PyTorch model.
        train_x (torch.Tensor): Input features for training, shape (batch_size, seq_len, input_dim).
        train_y (torch.Tensor): Target labels for training, shape (batch_size, 1).
        lr (float): Learning rate.
        weight_decay (float): L2 regularization factor.
        epochs (int): Number of training epochs.
        test_x (torch.Tensor, optional): Features for evaluation.
        test_y (torch.Tensor, optional): Labels for evaluation.

    Returns:
        tuple:
            model (nn.Module): Trained model in eval mode.
            training_loss (np.ndarray): Array of training losses for each epoch.
            test_loss (np.ndarray): Array of test losses for each epoch (zeros if no test data).
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        training_loss[epoch] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if test_x is not None:
            model.eval()
            with torch.no_grad():
                pred_test = model(test_x)
                loss_test = criterion(pred_test, test_y)
                test_loss[epoch] = loss_test.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Test Loss: {loss_test.item():.4f}")
        elif epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {loss.item():.4f}")

    return model.eval(), training_loss, test_loss


# -----------------------------
# Test functions
# -----------------------------

def test_model_forward():
    """
    Test the forward pass of the CoralHeatRNN model using random input.
    Verifies output shape and prints predictions.
    """
    model = CoralHeatRNN(
        input_size=5,
        hidden_size=64,
        layer_size=2,
        output_size=1,
        dropout=0.2,
        init_func=nn.init.xavier_uniform_,
    )
    x = torch.randn(8, 10, 5)
    out = model(x)
    print("Forward pass output shape:", out.shape)
    print("Model output:\n", out)


def test_training_loop():
    """
    Test training loop with dummy data.
    Verifies that training runs without error and prints final training loss.
    """
    model = CoralHeatRNN(
        input_size=5,
        hidden_size=64,
        layer_size=2,
        output_size=1,
        dropout=0.2,
        init_func=nn.init.xavier_uniform_,
    )

    x = torch.randn(8, 10, 5)
    y = torch.ones(8, 1)

    trained_model, train_loss, _ = train_model(
        model=model,
        train_x=x,
        train_y=y,
        lr=1e-4,
        weight_decay=1e-3,
        epochs=50,
    )
    print("Final training loss:", train_loss[-1])


def test_rolling_avg():
    """
    Test the rolling average function with random 1D input.
    Prints result shape and example output.
    """
    ts = np.random.rand(200)
    result = add_rolling_averages(ts)
    print("Rolling average result shape:", result.shape)
    print("Sample output:\n", result[-5:])  # print last 5 rows


# test_model_forward()
# test_training_loop()
test_rolling_avg()
