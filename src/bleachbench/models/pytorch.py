import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.optim as to
from torch.nn import SiLU, ReLU

'''
'''
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
'''

class CoralHeatRNN(nn.module):
    def __init__(self, input_size, hidden_size, layer_size, output_size,
                 dropout, init_func=None):
        super(CoralHeatRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.dropout = dropout
        self.rnn = nn.RNN(input_size, hidden_size, layer_size, batch_first=True,
                          dropout=dropout, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)
        self.a = nn.SiLU()
        if init_func is not None:
            init_func(self.fc.weight)
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        r, hn = self.rnn(inputs, h0)
        r = self.fc(r[:, -1, :])
        return r
### Train Network
def training(model, training_x, training_y, l_rate, w_decay,
             epochs, test_x=None, test_y=None):
    loss_function = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=w_decay)
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
            if i%500 == 0:
                print(f'Epoch {i} train loss: {loss.item()} test loss: {loss_test.item()}')
        elif i%500 == 0:
            print(f'Epoch {i} train loss: {loss.item()}')
        training_loss[i] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval(), training_loss, test_loss
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)