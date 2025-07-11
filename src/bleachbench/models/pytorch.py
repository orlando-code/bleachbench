import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as to

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoralHeatRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, layer_size, output_size, dropout, init_func=None
    ):
        super(CoralHeatRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.dropout = dropout
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            layer_size,
            batch_first=True,
            dropout=dropout,
            nonlinearity="relu",
        )
        self.fc = nn.Linear(hidden_size, output_size)
        if init_func is not None:
            init_func(self.fc.weight)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        h0 = torch.zeros(self.layer_size, batch_size, self.hidden_size).to(device)
        r, hn = self.rnn(inputs, h0)
        r = self.fc(r[:, -1, :]).sigmoid()
        return r


def test_CoralHeatRNN():
    """

    
    """

    # create input data for CoralHeatRNN
    model = CoralHeatRNN(
        input_size=5,
        hidden_size=64,
        layer_size=2,
        output_size=1,
        dropout=0.2,
        init_func=nn.init.xavier_uniform_,
    )
    # create example
    # create random input tensor: batch_size=8, seq_len=10, input_size=5
    x = torch.randn(8, 10, 5)
    y = torch.ones(8)
    # run forward pass
    out = model(x)
    print("Output shape:", out.shape)

    # create some random data
    print(out)


def training(
    model, training_x, training_y, l_rate, w_decay, epochs, test_x=None, test_y=None
):
    """
    Trains a PyTorch model using the provided training data.

    Args:
        model: The PyTorch model to be trained.
        training_x: Input features for training (tensor).
        training_y: Target labels for training (tensor).
        l_rate: Learning rate for the optimizer.
        w_decay: Weight decay (L2 regularization) for the optimizer.
        epochs: Number of training epochs.
        test_x: (Optional) Input features for validation/testing (tensor).
        test_y: (Optional) Target labels for validation/testing (tensor).

    Returns:
        model: The trained model in evaluation mode.
        training_loss: Numpy array of training loss values for each epoch.
        test_loss: Numpy array of test loss values for each epoch (zeros if no test data).
    """
    loss_function = nn.BCELoss()
    optimiser = to.Adam(model.parameters(), lr=l_rate, weight_decay=w_decay)
    training_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    for i in range(epochs):
        y_pred = model(training_x)
        loss = loss_function(y_pred.float(), training_y)
        if test_x is not None:
            with torch.no_grad():
                y_test_pred = model(test_x)
                loss_test = loss_function(y_test_pred.float(), test_y)
            test_loss[i] = loss_test.item()
            if i % 500 == 0:
                print(
                    f"Epoch {i} train loss: {loss.item()} test loss: {loss_test.item()}"
                )
        elif i % 10 == 0:
            print(f"Epoch {i} train loss: {loss.item()}")
        training_loss[i] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval(), training_loss, test_loss


# test_CoralHeatRNN()
def test_train():

    model = CoralHeatRNN(
        input_size=5,
        hidden_size=64,
        layer_size=2,
        output_size=1,
        dropout=0.2,
        init_func=nn.init.xavier_uniform_,
    )
    x = torch.randn(8, 10, 5)
    y = torch.ones(8)
    y = y.reshape(-1, 1)  # Shape: torch.Size([8,

    print(y)

    epochs = 50
    training(
        model=model, training_x=x, training_y=y, l_rate=0.0001, w_decay=0.001, epochs=epochs
    )


def add_rolling_averages(ts, windows=[30, 90, 180]):
    ts = np.asarray(ts)
    n = len(ts)
    
    # Prepare array to store results
    result = np.full((n, 1 + len(windows)), np.nan)
    result[:, 0] = ts  # first column is original data

    for i, window in enumerate(windows):
        for j in range(window, n):
            result[j, i + 1] = np.mean(ts[j - window:j])

    return result

# x_torch = torch.from_numpy(x).to(device).float()
def test_add_rolling_averages():
    from numpy import random
    input = random.rand((200))
    result = add_rolling_averages(input)
    print(result)


test_add_rolling_averages()
