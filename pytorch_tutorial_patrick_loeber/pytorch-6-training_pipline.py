# Design model(input, output size, forward pass)
# Consruct loss and optimizer
# Training loop
#     forward pass: compute prediction
#     backward pass: gradients
#     update weights


import torch
import torch.nn as nn

# function = weight * input
# f = w * x   

# f = 2 * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32) # a test tensor
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# print(f'Prediction before training: f(5) = {model(5):.3f}')
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')



# Training
learning_rate = 0.01
n_iters = 100

# loss
loss = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



# training loop
for epoch in range(in_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() #dl/dw
    

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    #  print our updates every "10" steps
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        # print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
        print(f'epoch {epoch + 1}: w = {w[0][0].item().3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}') # as our weight increases, our loss decreases which implies that our model is performing well.



