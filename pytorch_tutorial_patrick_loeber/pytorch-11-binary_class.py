import torch
import torch.nn as nn 

# Binary classification
class NeuralNet2(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init()

        # the first linear layer get an input size and hidden size
        self.linear1 = nn.Linear(input_size, hidden_size)
        
        # activation function in-between
        self.relu = nn.Relu()

        # the last linear layer get the hidden size and number of classes
        self.linear2 = nn.Linear(hidden_size, 1)

    
    # forward pass
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet2(input_size=28*28, hidden_size=5)

# using Binary Cross Entropy Loss as the criterion
criterion = nn.BCELoss() 