
from bleachbench.models.torch_models import CoralHeatRNN, train_model, add_rolling_averages
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
# test_rolling_avg()