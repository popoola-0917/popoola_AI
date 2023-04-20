# With activation function our model can learn more complex task and perform better

# Most popular activation function; step function, sigmoid, TanH, ReLU, Leaky ReLU, Softmax.

import torch
import torch.nn as nn 
import torch.nn.functional as F 

# option1 (create nn modules)
class NeuralNet (nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init()

        # The first linear layer
        self.linear1 = nn.Linear(input_size, hidden_size)

        # ReLU activation function
        self.relu = nn.ReLU()

        # nn.Sigmoid
        # nn.Softmax
        # nn.TanH

        # The second linear layer
        self.linear2 = nn.Linear(hidden_size, 1)

        # The next sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    # forward pass
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out 